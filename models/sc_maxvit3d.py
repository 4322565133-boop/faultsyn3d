"""Surface/side contrast context retrieval on the unchanged 3-D MaxViT UNet.

Coordinates use centered, augmented fine-image sample units. Context extents
carry the anisotropic FOV and its axis permutation; no labels enter the model.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .maxvit3d import MaxViT3DUNet


def orthogonal_frame(v):
    a,b=v[..., :3],v[..., 3:]
    default=torch.zeros_like(a);default[...,0]=1
    a=torch.where(a.norm(dim=-1,keepdim=True)>1e-5,a,default)
    t1=F.normalize(a,dim=-1,eps=1e-6)
    b=b-(b*t1).sum(-1,keepdim=True)*t1
    ref=F.one_hot(t1.abs().argmin(-1),3).to(t1.dtype)
    fallback=ref-(ref*t1).sum(-1,keepdim=True)*t1
    b=torch.where(b.norm(dim=-1,keepdim=True)>1e-5,b,fallback)
    t2=F.normalize(b,dim=-1,eps=1e-6)
    return t1,t2,torch.cross(t1,t2,dim=-1)


def sample_volume(features,valid,points,extent):
    """B,Q,K,(D,H,W) points -> B,Q,K,C features and B,Q,K validity."""
    grid=(2*points/extent[:,None,None,:])[..., [2,1,0]].unsqueeze(3)
    # FP32 coordinates/interpolation preserve small offsets under AMP.
    with torch.autocast(device_type=features.device.type,enabled=False):
        h=F.grid_sample(features.float(),grid.float(),align_corners=False,padding_mode='zeros')
        m=F.grid_sample(valid.float(),grid.float(),align_corners=False,padding_mode='zeros')
    return h.squeeze(-1).permute(0,2,3,1),m[:,0,:,:,0]>.999


class ContextBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.project=nn.Sequential(nn.Conv3d(cin,cout,1,bias=False),nn.GroupNorm(8,cout),nn.GELU())
        self.residual=nn.Sequential(nn.Conv3d(cout,cout,3,padding=1,groups=cout,bias=False),
            nn.GroupNorm(8,cout),nn.GELU(),nn.Conv3d(cout,cout,1,bias=False))
    def forward(self,x):
        x=self.project(F.avg_pool3d(x,2));return x+self.residual(x)


class ContextEncoder(nn.Module):
    def __init__(self,dim=96):
        super().__init__();self.blocks=nn.ModuleList([ContextBlock(1,24),ContextBlock(24,48),ContextBlock(48,96)])
        self.proj32=nn.Conv3d(48,dim,1,bias=False);self.proj16=nn.Conv3d(96,dim,1,bias=False)
    def forward(self,x,mask):
        values=[]
        for i,block in enumerate(self.blocks):
            x=block(x);mask=F.avg_pool3d(mask,2);x=x*(mask>0)
            if i==1:values.extend((self.proj32(x),mask))
            if i==2:values.extend((self.proj16(x),mask))
        return tuple(values)


class SurfaceContrastAttention(nn.Module):
    def __init__(self,channels=128,dim=96,heads=4,chunk=512):
        super().__init__();self.dim=dim;self.heads=heads;self.chunk=chunk
        self.query=nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,dim))
        self.geometry=nn.Linear(dim,50)
        nn.init.normal_(self.geometry.weight,std=.001);nn.init.zeros_(self.geometry.bias)
        with torch.no_grad():
            self.geometry.bias[:6]=torch.tensor([1.,0,0,0,1.,0])
            theta=torch.arange(8)*math.pi/4
            radius=torch.tensor([16.,32.,64.,88.,16.,32.,64.,88.])
            ab=torch.stack((radius*theta.cos(),radius*theta.sin(),torch.zeros(8)),-1)
            self.geometry.bias[6:30]=torch.atanh(ab/torch.tensor([96.,96.,8.])).flatten()
            free=torch.tensor([[80.,0,0],[-80.,0,0],[0,80.,0],[0,-80.,0]])
            self.geometry.bias[38:50]=torch.atanh(free/96).flatten()
        self.surface=nn.Linear(dim*3+2,dim)
        self.free=nn.Linear(dim,dim)
        self.key=nn.Linear(dim,dim,bias=False);self.value=nn.Linear(dim,dim,bias=False)
        self.out=nn.Linear(dim,channels,bias=False);nn.init.normal_(self.out.weight,std=.01)
        self.last_diagnostics={}

    def forward(self,local,c32,m32,c16,m16,extent):
        b,c,d,h,w=local.shape
        local_tokens=local.permute(0,2,3,4,1).reshape(b,-1,c)
        q=self.query(local_tokens)
        axes=[(torch.arange(n,device=q.device,dtype=torch.float32)+.5)*128/n-64 for n in (d,h,w)]
        base=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(1,-1,3)
        outputs=[];valid_stats=[];external_stats=[];offset_stats=[]
        for start in range(0,q.shape[1],self.chunk):
            qq=q[:,start:start+self.chunk];z=self.geometry(qq).float()
            t1,t2,n=orthogonal_frame(z[...,:6]);anchor=base[:,start:start+self.chunk]
            ab=z[...,6:30].reshape(b,-1,8,3).tanh()*z.new_tensor([96.,96.,8.])
            p=anchor[:,:,None]+ab[...,0,None]*t1[:,:,None]+ab[...,1,None]*t2[:,:,None]+ab[...,2,None]*n[:,:,None]
            delta=8+8*z[...,30:38].sigmoid()
            plus=p+delta[...,None]*n[:,:,None];minus=p-delta[...,None]*n[:,:,None]
            free=anchor[:,:,None]+96*z[...,38:50].reshape(b,-1,4,3).tanh()
            points=torch.cat((torch.stack((p,plus,minus),3).flatten(2,3),free),2)
            tokens=[];masks=[]
            for feat,mask in ((c32,m32),(c16,m16)):
                hh,v=sample_volume(feat,mask,points,extent)
                hh0=hh[:,:,:24].reshape(b,-1,8,3,self.dim);vv=v[:,:,:24].reshape(b,-1,8,3)
                h0,hp,hm=hh0.unbind(3);vc=vv[...,0];vs=vv.all(-1)
                desc=torch.cat((h0,(h0-(hp+hm)/2)*vs[...,None],(hp-hm).abs()*vs[...,None],
                    vc[...,None].float(),vs[...,None].float()),-1)
                tokens.extend((self.surface(desc),self.free(hh[:,:,24:])));masks.extend((vc,v[:,:,24:]))
            tok=torch.cat(tokens,2);mask=torch.cat(masks,2);nt=tok.shape[2];hd=self.dim//self.heads
            keys=self.key(tok).reshape(b,-1,nt,self.heads,hd);values=self.value(tok).reshape_as(keys)
            score=torch.einsum('bqhd,bqnhd->bqhn',qq.reshape(b,-1,self.heads,hd),keys)/math.sqrt(hd)
            score=score.float().masked_fill(~mask[:,:,None],-1e4)
            weight=score.softmax(-1)*mask[:,:,None];weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
            out=torch.einsum('bqhn,bqnhd->bqhd',weight.to(values.dtype),values).flatten(-2)
            outputs.append(self.out(out))
            valid_stats.append(mask.float().mean().detach())
            external_stats.append((points.abs()>64).any(-1).float().mean().detach())
            offset_stats.append(ab[...,2].abs().mean().detach())
        residual=torch.cat(outputs,1).transpose(1,2).reshape_as(local)
        self.last_diagnostics=dict(valid_token_fraction=torch.stack(valid_stats).mean(),
            points_outside_fine_fraction=torch.stack(external_stats).mean(),
            normal_residual_mean=torch.stack(offset_stats).mean(),
            residual_to_local_norm=(residual.detach().float().norm()/local.detach().float().norm().clamp_min(1e-8)))
        return local+residual


class SCMaxViT3D(MaxViT3DUNet):
    def __init__(self,checkpoint_highres=True):
        super().__init__(full_res_skip=True,drop_path_rate=.2,img_size=128,pretrained=False)
        self.context_encoder=ContextEncoder();self.context_fusion=SurfaceContrastAttention()
        self.checkpoint_highres=checkpoint_highres

    def _safe(self,fn,*args):
        if self.training and self.checkpoint_highres:return checkpoint(fn,*args,use_reentrant=False)
        return fn(*args)

    def forward(self,x,context,context_valid,extent):
        f0,f1,f2,f3,f4=self.backbone(x)
        cx=self._safe(self.context_encoder,context,context_valid)
        f2=self._safe(self.context_fusion,f2,*cx,extent)
        def up(a,b):return F.interpolate(a,size=b.shape[2:],mode='trilinear',align_corners=False)
        d=self.dec4(torch.cat((up(f4,f3),f3),1));d=self.dec3(torch.cat((up(d,f2),f2),1))
        d=self.dec2(torch.cat((up(d,f1),f1),1));d=self._safe(self.dec1,torch.cat((up(d,f0),f0),1))
        detail=self._safe(self.enc_full,x)
        d=self._safe(self.dec0,torch.cat((up(d,x),detail),1))
        return self.outc(d)
