"""60-epoch replayed anchor75 training; no test access or pretrained weights."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,Subset

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models import build
from train.dataset_thebe_anchor import AnchorThebe
from train.train_thebe_spatial_ddp import loss_fn,atomic_save


def move(x,device):
    if isinstance(x,(list,tuple)):return tuple(move(v,device) for v in x)
    return x.to(device,non_blocking=True)


def forward(model,x):
    return model(*x) if isinstance(x,(tuple,list)) else model(x)


@torch.no_grad()
def evaluate(model,loader,device):
    for buf in model.buffers():dist.broadcast(buf,0)
    model.eval();total=torch.zeros(7,dtype=torch.float64,device=device)
    hist=torch.zeros(2,2001,dtype=torch.int64,device=device)
    for inp,y,m,_ in loader:
        inp,y,m=move(inp,device),move(y,device),move(m,device)
        with torch.autocast('cuda',dtype=torch.float16):logits=forward(model,inp)
        valid=m>.5;pred=logits.float()>0;gt=y>.5
        total+=torch.stack(((pred&gt&valid).sum(),(pred&~gt&valid).sum(),(~pred&gt&valid).sum(),
            valid.sum(),(gt&valid).sum(),loss_fn(logits,y,m).double(),torch.ones((),device=device))).double()
        bins=(logits.float().sigmoid()[valid]*2000).floor().long().clamp(0,2000);g=gt[valid]
        hist[0]+=torch.bincount(bins[g],minlength=2001);hist[1]+=torch.bincount(bins[~g],minlength=2001)
    dist.all_reduce(total);dist.all_reduce(hist)
    tp,fp,fn,n,pos,loss,count=total.tolist();hp,hn=hist.double().flip(1).cumsum(1)
    recall=hp/hp[-1].clamp_min(1);ap=((recall-torch.cat((recall.new_zeros(1),recall[:-1])))*hp/(hp+hn).clamp_min(1)).sum().item()
    return dict(iou=tp/max(tp+fp+fn,1),dice=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        ap=ap,valid_voxels=int(n),positive_voxels=int(pos),loss=loss/max(count,1),cores=int(count))


def atomic_json(data,path):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--model',choices=['unet_l','sc_maxvit'],required=True)
    p.add_argument('--plans',default=str(ROOT/'data/thebe_anchor75q_60'))
    p.add_argument('--root',default=str(ROOT/'data/thebe_spatial_v3'))
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--stop-after',type=int,default=60)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--per-rank',type=int,default=2);p.add_argument('--accum',type=int,default=1)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    if world*a.per_rank*a.accum!=8:raise ValueError('Anchor quotas require global batch eight')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=True
    out=Path(a.run);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume:raise RuntimeError('Refusing to overwrite existing run')
    context=a.model=='sc_maxvit'
    tr=AnchorThebe('train',a.root,a.plans,context);va=AnchorThebe('val',a.root,context=context)
    if a.epochs>tr.plan_manifest['spec']['epochs']:raise ValueError('Insufficient epoch plans')
    plan_hash=tr.plan_manifest['plan_sha256'];protocol=tr.manifest['protocol_sha256']
    if a.smoke:
        tr.samples=16;va.origins=va.origins[:8]
    sampler=DistributedSampler(tr,world,rank,shuffle=False,drop_last=True)
    tl=DataLoader(tr,batch_size=a.per_rank,sampler=sampler,num_workers=a.workers,pin_memory=True)
    vl=DataLoader(Subset(va,list(range(rank,len(va),world))),batch_size=1,num_workers=a.workers,pin_memory=True)
    name,kwargs=('sc_maxvit3d',{'checkpoint_highres':True}) if context else ('unet3d',{'base':42,'checkpoint_highres':True})
    model=build(name,**kwargs).to(device);params=sum(t.numel() for t in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');model=DDP(model,device_ids=[local],broadcast_buffers=False)
    start=1;best=-1.;history=[]
    if a.resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert ck['plan_sha256']==plan_hash and ck['protocol_sha256']==protocol
        assert ck['world_size']==world
        for k in ('model','epochs','seed','per_rank','accum','smoke'):assert ck['args'][k]==vars(a)[k],k
        model.module.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        best=ck['best'];history=ck['history'];start=ck['epoch']+1
        state=ck['rng'][rank];torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'],device)
        random.setstate(state['python']);np.random.set_state(state['numpy'])
    source_paths=['train/train_thebe_anchor_ddp.py','train/dataset_thebe_anchor.py','train/dataset_thebe_spatial.py',
        'train/train_thebe_spatial_ddp.py','models/sc_maxvit3d.py','models/maxvit3d.py','models/unet3d.py','models/__init__.py',
        'scripts/prepare_thebe_anchor75.py']
    if rank==0:
        sources={}
        for s in source_paths:
            file=ROOT/s;sources[s]=hashlib.sha256(file.read_bytes()).hexdigest()
            dst=out/'source_snapshot'/s;dst.parent.mkdir(parents=True,exist_ok=True)
            if a.resume and dst.exists() and dst.read_bytes()!=file.read_bytes():raise RuntimeError(f'Source changed on resume: {s}')
            dst.write_bytes(file.read_bytes())
        atomic_json(dict(args=vars(a),params=params,model=name,model_kwargs=kwargs,world_size=world,effective_batch=8,
            sources=sources,protocol_sha256=protocol,plan_sha256=plan_hash,initialization='random; no pretrained weights',
            sampling='anchor75 train lower-quartile density stratification',full_val_cores=len(va),
            note='UNet single FOV; SC uses additional context. Historical uniform/100-epoch curves are references only.'),out/'config.json')
        atomic_json(tr.plan_manifest,out/'sampling_manifest.json')
        print('START',a.model,'params',params,'epochs',a.epochs,'plan',plan_hash,flush=True)
    dist.barrier()
    for epoch in range(start,min(a.epochs,a.stop_after)+1):
        t0=time.time();tr.epoch=epoch;sampler.set_epoch(epoch);model.train()
        lr=1e-6+(1e-4-1e-6)*(epoch-1)/9 if epoch<=10 else 1e-7+.5*(1e-4-1e-7)*(1+math.cos(math.pi*(epoch-10)/(a.epochs-10)))
        for g in optimizer.param_groups:g['lr']=lr
        optimizer.zero_grad(set_to_none=True);tot=torch.zeros(3,dtype=torch.float64,device=device)
        context_sums=torch.zeros(6,dtype=torch.float64,device=device)
        context_keys=('valid_token_fraction','points_outside_fine_fraction','normal_residual_mean','residual_to_local_norm')
        for step,(inp,y,_,ml) in enumerate(tl):
            inp,y,ml=move(inp,device),move(y,device),move(ml,device);sync=(step+1)%a.accum==0
            with contextlib.nullcontext() if sync else model.no_sync():
                with torch.autocast('cuda',dtype=torch.float16):logits=forward(model,inp)
                loss=loss_fn(logits,y,ml)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {epoch}:{step}')
                scaler.scale(loss/a.accum).backward()
            if sync:
                scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                tot[2]+=int(scaler.get_scale()<scale)
                if context and (step==0 or (step+1)%100==0) and scaler.get_scale()>=scale:
                    d=model.module.context_fusion.last_diagnostics
                    grad=model.module.context_fusion.geometry.weight.grad
                    context_sums[:4]+=torch.stack([d[k] for k in context_keys]).double()
                    context_sums[4]+=grad.detach().float().norm().double()
                    context_sums[5]+=1
                optimizer.zero_grad(set_to_none=True)
            tot[0]+=loss.detach();tot[1]+=1
            if rank==0 and (step==0 or (step+1)%100==0):
                print(f'epoch {epoch} microstep {step+1}/{len(tl)} loss={loss.item():.5f} lr={lr:.3g} seconds={time.time()-t0:.0f}',flush=True)
        dist.all_reduce(tot)
        if context:dist.all_reduce(context_sums)
        if rank==0:print('FULL VALIDATION',epoch,len(va),flush=True)
        metrics=evaluate(model.module,vl,device)
        if not a.smoke:assert metrics['valid_voxels']==va.record['support_voxels']
        improved=metrics['iou']>best;best=max(best,metrics['iou'])
        diagnostics={k:float(v) for k,v in model.module.context_fusion.last_diagnostics.items()} if context else None
        row=dict(epoch=epoch,lr=lr,train_loss=float(tot[0]/tot[1]),fast_val=None,full_val=metrics,
            seconds=time.time()-t0,skipped_optimizer_steps_all_ranks=int(tot[2]),
            validation_context_diagnostics_last_core=diagnostics,
            training_context_diagnostics={k:float(context_sums[j]/context_sums[5].clamp_min(1))
                for j,k in enumerate((*context_keys,'geometry_gradient_norm'))} if context else None)
        history.append(row)
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),python=random.getstate(),numpy=np.random.get_state())
        states=[None]*world;dist.all_gather_object(states,rng)
        if rank==0:
            ck=dict(epoch=epoch,model=model.module.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                best=best,history=history,rng=states,args=vars(a),world_size=world,protocol_sha256=protocol,plan_sha256=plan_hash)
            atomic_save(ck,out/'last.pt')
            if improved:atomic_save(ck,out/'best_fullval.pt')
            if epoch%10==0:atomic_save(ck,out/f'epoch_{epoch:03d}.pt')
            atomic_json(history,out/'history.json')
            atomic_json(dict(status='running',last_completed_epoch=epoch,best_fullval_iou=best),out/'status.json')
            print(json.dumps(row),flush=True)
        dist.barrier()
    if rank==0:atomic_json(dict(status='completed' if min(a.epochs,a.stop_after)==a.epochs else 'completed_trial',
        last_completed_epoch=min(a.epochs,a.stop_after),best_fullval_iou=best),out/'status.json')
    dist.destroy_process_group()


if __name__=='__main__':main()
