"""Experimental MaxViT with cross-scale block/grid decoding for field fault masks.

No pretrained weights; no claim of demonstrated superiority. The original encoder
is retained. The heavy concatenation decoder is replaced by context-conditioned
cross-attention at 16^3/32^3; detail queries read deep context keys/values.
The block/grid/both ablations share all attention parameters (but not FLOPs).
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import timm_3d
from .maxvit3d import DoubleConv3D


def partition(x, size=4, mode='block'):
    """BDHWC -> (B * groups, size^3, C); grid tokens are spatially dispersed."""
    b,d,h,w,c=x.shape;g=size
    if any(s % g for s in (d,h,w)):
        raise ValueError('Feature spatial dimensions must be divisible by partition size')
    if mode=='block':
        x=x.reshape(b,d//g,g,h//g,g,w//g,g,c).permute(0,1,3,5,2,4,6,7)
    elif mode=='grid':
        x=x.reshape(b,g,d//g,g,h//g,g,w//g,c).permute(0,2,4,6,1,3,5,7)
    else:raise ValueError(mode)
    return x.reshape(-1,g**3,c)


def unpartition(x, shape, size=4, mode='block'):
    b,d,h,w,c=shape;g=size
    x=x.reshape(b,d//g,h//g,w//g,g,g,g,c)
    if mode=='block':x=x.permute(0,1,4,2,5,3,6,7)
    elif mode=='grid':x=x.permute(0,4,1,5,2,6,3,7)
    else:raise ValueError(mode)
    return x.reshape(shape)


class ContextCrossAttention3D(nn.Module):
    def __init__(self, dim, mode='both', size=4, head_dim=16):
        super().__init__()
        if dim % head_dim or mode not in ('block','grid','both'):raise ValueError((dim,mode))
        self.mode,self.size,self.heads,self.head_dim=mode,size,dim//head_dim,head_dim
        self.norm_q=nn.LayerNorm(dim);self.norm_kv=nn.LayerNorm(dim)
        self.q=nn.Linear(dim,dim);self.kv=nn.Linear(dim,2*dim);self.proj=nn.Linear(dim,dim)
        xyz=torch.stack(torch.meshgrid(*[torch.arange(size)]*3,indexing='ij')).flatten(1)
        delta=xyz[:,:,None]-xyz[:,None,:]+size-1
        index=delta[0]*(2*size-1)**2+delta[1]*(2*size-1)+delta[2]
        self.register_buffer('relative_index',index.long(),persistent=False)
        self.relative_bias=nn.Parameter(torch.zeros((2*size-1)**3,self.heads))
        nn.init.trunc_normal_(self.relative_bias,std=.02)

    def forward(self,local,context):
        x=local.permute(0,2,3,4,1);z=context.permute(0,2,3,4,1)
        shape=x.shape;qn=self.norm_q(x);kn=self.norm_kv(z)
        modes=('block','grid') if self.mode=='both' else (self.mode,)
        bias=self.relative_bias[self.relative_index].permute(2,0,1)[None]
        outputs=[]
        for mode in modes:
            q=self.q(partition(qn,self.size,mode));kv=self.kv(partition(kn,self.size,mode))
            b,n,c=q.shape
            q=q.reshape(b,n,self.heads,self.head_dim).transpose(1,2)
            k,v=kv.reshape(b,n,2,self.heads,self.head_dim).permute(2,0,3,1,4)
            a=F.scaled_dot_product_attention(q,k,v,attn_mask=bias.to(q.dtype))
            a=self.proj(a.transpose(1,2).reshape(b,n,c))
            outputs.append(unpartition(a,shape,self.size,mode))
        return torch.stack(outputs).mean(0).permute(0,4,1,2,3).contiguous()


class ContextFusion3D(nn.Module):
    def __init__(self,local_ch,context_ch,dim,mode='both'):
        super().__init__()
        if mode not in ('conv','block','grid','both'):raise ValueError(mode)
        self.mode=mode
        self.local=nn.Conv3d(local_ch,dim,1,bias=False)
        self.context=nn.Conv3d(context_ch,dim,1,bias=False)
        self.attention=ContextCrossAttention3D(dim,mode) if mode!='conv' else None
        channels=3*dim if self.attention is not None else 2*dim
        self.mix=nn.Sequential(nn.Conv3d(channels,dim,1,bias=False),nn.InstanceNorm3d(dim,affine=True),nn.GELU(),
            nn.Conv3d(dim,dim,3,padding=1,groups=dim,bias=False),nn.InstanceNorm3d(dim,affine=True),nn.GELU(),
            nn.Conv3d(dim,dim,1,bias=False))

    def forward(self,local,context):
        x=self.local(local);z=F.interpolate(self.context(context),size=x.shape[2:],mode='trilinear',align_corners=False)
        parts=[x,z]
        if self.attention is not None:parts.append(self.attention(x,z))
        # This is the actual decoder stage, not an additive zero-gated side branch.
        return self.mix(torch.cat(parts,dim=1))


class ContextGridMaxViT3D(nn.Module):
    def __init__(self,mode='both',drop_path_rate=.2,img_size=128,checkpoint_highres=True):
        super().__init__()
        self.checkpoint_highres=bool(checkpoint_highres)
        self.backbone=timm_3d.create_model('maxvit_tiny_tf_224.in1k',pretrained=False,in_chans=1,
            features_only=True,drop_path_rate=drop_path_rate,img_size=img_size)
        c0,c1,c2,c3,c4=self.backbone.feature_info.channels()
        self.deep4=nn.Conv3d(c4,128,1,bias=False);self.skip8=nn.Conv3d(c3,128,1,bias=False)
        self.context8=DoubleConv3D(256,128)
        self.fuse16=ContextFusion3D(c2,128,96,mode)
        self.fuse32=ContextFusion3D(c1,96,64,mode)
        self.dec64=DoubleConv3D(64+c0,64)
        self.detail=DoubleConv3D(1,16)
        self.dec128=DoubleConv3D(64+16,32)
        self.out=nn.Conv3d(32,1,1)

    def _full(self,module,x):
        if self.training and self.checkpoint_highres:
            return checkpoint(module,x,use_reentrant=False)
        return module(x)

    def forward(self,x):
        if x.ndim!=5 or x.shape[1]!=1 or any(s%32 for s in x.shape[2:]):
            raise ValueError('Expected B,1,D,H,W with sizes divisible by 32')
        f0,f1,f2,f3,f4=self.backbone(x)
        def up(a,b):return F.interpolate(a,size=b.shape[2:],mode='trilinear',align_corners=False)
        d=self.context8(torch.cat((up(self.deep4(f4),f3),self.skip8(f3)),1))
        d=self.fuse16(f2,d);d=self.fuse32(f1,d)
        d=self.dec64(torch.cat((up(d,f0),f0),1))
        detail=self._full(self.detail,x)
        return self.out(self._full(self.dec128,torch.cat((up(d,x),detail),1)))
