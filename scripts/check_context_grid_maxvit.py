import json,time
from pathlib import Path
import torch
from models.context_grid_maxvit3d import partition,unpartition,ContextCrossAttention3D,ContextFusion3D
from models import build

torch.set_num_threads(2);torch.manual_seed(2026)
result={}
x=torch.arange(1*8*12*16*2).reshape(1,8,12,16,2).float()
for mode in ['block','grid']:
 assert torch.equal(x,unpartition(partition(x,4,mode),x.shape,4,mode))
a=partition(torch.arange(8**3).reshape(1,8,8,8,1),4,'block')[0,:,0]
b=partition(torch.arange(8**3).reshape(1,8,8,8,1),4,'grid')[0,:,0]
assert a[1]-a[0]==1 and b[1]-b[0]==2
result['partition']='PASS: exact inversion on non-cubic shapes; block adjacent/grid dispersed'
for mode in ['block','grid','both','conv']:
 f=ContextFusion3D(16,32,32,mode);local=torch.randn(1,16,8,8,8,requires_grad=True);context=torch.randn(1,32,4,4,4,requires_grad=True)
 out=f(local,context);out.square().mean().backward()
 assert torch.isfinite(out).all() and local.grad.abs().sum()>0 and context.grad.abs().sum()>0
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters())
 result['fusion_'+mode]=dict(shape=list(out.shape),params=sum(p.numel() for p in f.parameters()),gradient='all finite; local and context inputs used')
# Actual SDPA versus explicit softmax on the same projection/relative bias.
f=ContextCrossAttention3D(32,mode='grid');l=torch.randn(1,32,8,8,8);z=torch.randn_like(l)
with torch.no_grad():
 x=l.permute(0,2,3,4,1);y=z.permute(0,2,3,4,1);shape=x.shape
 q=f.q(partition(f.norm_q(x),4,'grid'));kv=f.kv(partition(f.norm_kv(y),4,'grid'));bs,n,c=q.shape
 q=q.reshape(bs,n,f.heads,f.head_dim).transpose(1,2);k,v=kv.reshape(bs,n,2,f.heads,f.head_dim).permute(2,0,3,1,4)
 bias=f.relative_bias[f.relative_index].permute(2,0,1)[None]
 ref=((q@k.transpose(-2,-1)/f.head_dim**.5+bias).softmax(-1)@v).transpose(1,2).reshape(bs,n,c)
 ref=unpartition(f.proj(ref),shape,4,'grid').permute(0,4,1,2,3)
 assert torch.allclose(ref,f(l,z),atol=1e-6,rtol=1e-5)
result['attention_reference']='PASS'
t=time.time();model=build('context_grid_maxvit3d',mode='both');model.eval()
result['params']=sum(p.numel() for p in model.parameters());result['encoder_params']=sum(p.numel() for p in model.backbone.parameters())
with torch.inference_mode():y=model(torch.randn(1,1,128,128,128))
assert y.shape==(1,1,128,128,128) and torch.isfinite(y).all()
result['full_cpu_forward']=dict(shape=list(y.shape),seconds=time.time()-t,status='PASS')
import numpy as np
model.train();model.zero_grad(set_to_none=True)
s=np.load('data/thebe_cubes/seismic/thebe_train_0000.npy').astype(np.float32)
target=np.load('data/thebe_cubes/labels/thebe_train_0000.npy').astype(np.float32)
s=(s-s.mean())/(s.std()+1e-6);t=time.time()
logits=model(torch.from_numpy(s)[None,None]);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(target)[None,None]);loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
result['full_cpu_backward']=dict(loss=float(loss.detach()),seconds=time.time()-t,status='PASS; real Thebe train cube; all parameter gradients present and finite; no optimizer update')
result['limitations']='No CUDA AMP or DDP benchmark yet: 4 GPUs occupied by existing training. No training or efficacy claim.'
Path('reports/thebe_model_design_20260917/model_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
