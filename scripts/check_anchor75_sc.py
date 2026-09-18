"""Meaningful coordinate, mask, gradient, replay and GPU feasibility checks."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.sc_maxvit3d import SurfaceContrastAttention,sample_volume,orthogonal_frame
from train.dataset_thebe_anchor import AnchorThebe,down_context,transform
from train.dataset_thebe_spatial import SpatialThebe
from train.train_thebe_spatial_ddp import loss_fn
from models import build


def check_cpu():
    torch.set_num_threads(4)
    extent=torch.tensor([[256.,128.,128.]])
    zz,yy,xx=torch.meshgrid(torch.arange(4),torch.arange(6),torch.arange(8),indexing='ij')
    f=(zz+10*yy+100*xx).float()[None,None];m=torch.ones_like(f)
    ix=torch.tensor([[1.,2.,3.],[2.,4.,5.]])
    points=((ix+.5)/torch.tensor([4.,6.,8.])-.5)*extent
    actual,valid=sample_volume(f,m,points[None,:,None],extent)
    assert torch.allclose(actual.flatten(),torch.tensor([321.,542.]),atol=1e-4) and valid.all()
    # Real anisotropic context decimation + paired augmentation remains aligned.
    coords=np.indices((256,256,128),dtype=np.float32)
    raw=.125*coords[0]+.25*coords[1]+.5*coords[2]
    fine=raw[64:192,64:192,:].copy();ctx,cm=down_context(raw,np.ones_like(raw))
    ix=torch.tensor([[40.,48.,56.],[80.,72.,64.]])
    for k in range(4):
        for flips in ((False,False),(True,False),(False,True),(True,True)):
            c=transform(ctx,k,flips);mf=transform(cm,k,flips);ref=transform(fine,k,flips)
            ex=torch.tensor([[256.,128.,256.] if k%2 else [256.,256.,128.]])
            actual,v=sample_volume(torch.from_numpy(c[None,None]),torch.from_numpy(mf[None,None]),(ix-63.5)[None,:,None],ex)
            expected=torch.tensor([ref[tuple(map(int,t))] for t in ix])
            assert torch.allclose(actual.flatten(),expected,atol=2e-4),(k,flips,actual,expected)
            assert v.all()
    # Underlying file includes a neighboring split: access must still clip to record shape.
    ds=object.__new__(AnchorThebe);ds.shape=(16,16,8)
    ds._x=np.ones((16,16,16),np.float32);ds._x[:,:,8:]=9999
    ds.first=np.zeros((16,8),np.int16);ds.last=np.full((16,8),16,np.int16)
    v,mask=ds.read_context((-32,-32,-32));assert v.max()==1 and int(mask.sum())==16*16*8
    # Stable frame for zero and parallel predicted axes.
    frame=orthogonal_frame(torch.tensor([[0.,0,0,0,0,0],[1.,0,0,2.,0,0]]))
    assert all(torch.isfinite(t).all() for t in frame)
    assert torch.allclose((frame[0]*frame[1]).sum(-1),torch.zeros(2))
    module=SurfaceContrastAttention(channels=8,dim=8,heads=2,chunk=20)
    local=torch.randn(1,8,4,4,4,requires_grad=True)
    ctx=torch.randn(1,8,16,16,16,requires_grad=True);mask=torch.ones(1,1,16,16,16)
    out=module(local,ctx,mask,ctx,mask,torch.tensor([[256.,256.,128.]]));out.square().mean().backward()
    grad=module.geometry.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    with torch.no_grad():
        out=module(local,ctx,mask*0,ctx,mask*0,torch.tensor([[256.,256.,128.]]))
        assert torch.equal(local,out),'All-invalid context must be identity'
    return dict(coordinate_ramp=True,anisotropic_augmentation_16_cases=True,split_sentinel=True,
        degenerate_frame=True,geometry_gradient=True,all_invalid_identity=True)


def check_data(plans):
    man=json.loads((Path(plans)/'manifest.json').read_text())
    for ep in range(1,61):
        d=np.load(Path(plans)/f'epoch_{ep:03d}.npz');k=d['kinds']
        assert len(k)==2400
        for batch in k.reshape(-1,8):assert np.bincount(batch,minlength=3).tolist()==[4,2,2]
        # The actual rank-strided sampler reconstructs exactly these groups.
        for per_rank,accum in ((2,1),(1,2)):
            shards=[np.arange(2400)[r::4] for r in range(4)]
            for step in (0,7,299):
                inds=np.concatenate([s[step*per_rank*accum:(step+1)*per_rank*accum] for s in shards])
                assert np.bincount(k[inds],minlength=3).tolist()==[4,2,2]
    old=SpatialThebe('val');new=AnchorThebe('val')
    assert old.origins==new.origins
    for i in (0,600,1500,2432):
        a=old[i];b=new[i]
        assert torch.allclose(a[0],b[0],atol=1e-6,rtol=1e-6)
        assert all(torch.equal(x,y) for x,y in zip(a[1:],b[1:]))
    plain=AnchorThebe('train',plans=plans);dual=AnchorThebe('train',plans=plans,context=True)
    plain.epoch=dual.epoch=1
    for i in range(8):
        a=plain[i];b=dual[i]
        assert torch.equal(a[0],b[0][0]) and all(torch.equal(x,y) for x,y in zip(a[1:],b[1:]))
    return dict(all_60_epoch_quotas=True,ddp_accumulation_quotas=True,validation_unchanged=True,
        same_fine_data_across_models=True,plan_sha256=man['plan_sha256'])


def check_gpu(model_name,plans,batch,device):
    torch.cuda.set_device(device);torch.set_num_threads(2);torch.manual_seed(2026)
    torch.backends.cudnn.benchmark=True
    context=model_name=='sc_maxvit';ds=AnchorThebe('train',plans=plans,context=context);ds.epoch=1
    from torch.utils.data._utils.collate import default_collate
    from train.train_thebe_anchor_ddp import move,forward
    inp,y,_,ml=default_collate([ds[i] for i in range(batch)])
    inp,y,ml=move(inp,device),move(y,device),move(ml,device)
    model=build('sc_maxvit3d',checkpoint_highres=True) if context else build('unet3d',base=42,checkpoint_highres=True)
    model=model.to(device).train();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);scaler=torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats(device);times=[];losses=[]
    for _ in range(2):
        t=time.time();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):logits=forward(model,inp)
        loss=loss_fn(logits,y,ml);scaler.scale(loss).backward();scaler.unscale_(opt)
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            # Initial AMP scaling may overflow; record rather than conceal it.
            scaler.step(opt);scaler.update();continue
        if context:
            grad=model.context_fusion.geometry.weight.grad
            assert grad is not None and grad.abs().sum()>0
        scaler.step(opt);scaler.update();torch.cuda.synchronize(device);times.append(time.time()-t);losses.append(float(loss.detach()))
    if not times:raise RuntimeError('No finite optimizer step in GPU preflight')
    return dict(model=model_name,batch=batch,params=sum(p.numel() for p in model.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        finite_steps=len(times),step_seconds=times,losses=losses,
        diagnostics={k:float(v) for k,v in model.context_fusion.last_diagnostics.items()} if context else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plans',default=str(ROOT/'data/thebe_anchor75q_60'))
    p.add_argument('--mode',choices=['cpu','data','gpu'],default='cpu');p.add_argument('--model',default='sc_maxvit')
    p.add_argument('--batch',type=int,default=2);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    r=check_cpu() if a.mode=='cpu' else check_data(a.plans) if a.mode=='data' else check_gpu(a.model,a.plans,a.batch,torch.device(f'cuda:{a.gpu}'))
    out=ROOT/'reports/anchor75_sc_60';out.mkdir(parents=True,exist_ok=True)
    suffix=a.mode if a.mode!='gpu' else f'gpu_{a.model}_b{a.batch}'
    (out/f'{suffix}_checks.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))


if __name__=='__main__':main()
