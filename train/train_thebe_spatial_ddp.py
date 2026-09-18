"""Masked Thebe baseline; sealed test; full spatial validation selects checkpoints."""
import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import build
from train.dataset_thebe_spatial import SpatialThebe, DEFAULT, label_blocks
from train.train_old_recipe_ddp import VARIANTS          # model name + kwargs per variant (unet_l, baseline, context_both, ...)

def loss_fn(logits,y,m):
    p=logits.float().sigmoid(); dims=(1,2,3,4)
    dice=1-(2*(p*y*m).sum(dims)+1e-6)/((p*m).sum(dims)+(y*m).sum(dims)+1e-6)
    pt=torch.where(y>.5,p,1-p); at=torch.where(y>.5,.75,.25)
    focal=at*(1-pt).square()*F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')
    focal=(focal*m).sum(dims)/m.sum(dims).clamp_min(1)
    return (.6*dice+.4*focal).mean()

@torch.no_grad()
def evaluate(model,loader,device):
    for b in model.buffers(): dist.broadcast(b,0)     # BatchNorm running stats (MaxViT MBConv) identical on every rank
    model.eval(); totals=torch.zeros(7,dtype=torch.float64,device=device)
    for x,y,m,_ in loader:
        x,y,m=[a.to(device,non_blocking=True) for a in (x,y,m)]
        with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
        pr=logits.float()>0; gt=y>.5; v=m>.5
        totals+=torch.stack(((pr&gt&v).sum(),(pr&~gt&v).sum(),(~pr&gt&v).sum(),
            v.sum(),(gt&v).sum(),loss_fn(logits,y,m).double(),torch.ones((),device=device))).double()
    dist.all_reduce(totals)
    tp,fp,fn,n,pos,l,count=totals.tolist()
    return dict(iou=tp/max(tp+fp+fn,1),dice=2*tp/max(2*tp+fp+fn,1),
        precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),valid_voxels=int(n),
        positive_voxels=int(pos),loss=l/max(count,1),cores=int(count))

def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

def plot_history(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot([r['epoch'] for r in rows],[r['fast_val']['iou'] for r in rows],label='Fixed 96-core validation (diagnostic)')
    full=[r for r in rows if r.get('full_val')]
    if full: axs[0].plot([r['epoch'] for r in full],[r['full_val']['iou'] for r in full],'o-',label='Full spatial validation (selection)')
    axs[0].set(ylabel='IoU @ 0.5',xlabel='Epoch',ylim=(0,1)); axs[0].legend(fontsize=7)
    axs[1].plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label='Training loss')
    axs[1].plot([r['epoch'] for r in rows],[r['fast_val']['loss'] for r in rows],label='Fast validation loss')
    axs[1].set(xlabel='Epoch',ylabel='Masked Dice + Focal'); axs[1].legend(fontsize=8)
    fig.suptitle('Thebe spatial v3 | from scratch | TEST sealed')
    fig.tight_layout(); fig.savefig(out/'live_curves.png',dpi=140); plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=str(DEFAULT)); p.add_argument('--run',required=True)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--stop-after',type=int,default=30)
    p.add_argument('--samples',type=int,default=2400); p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--full-every',type=int,default=10)
    p.add_argument('--resume',action='store_true'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--variant',default='unet_l',choices=list(VARIANTS),help='model variant (train_old_recipe_ddp.VARIANTS)')
    p.add_argument('--label-frac',type=float,default=1.0,help='fraction of the train region (contiguous blocks along axis 1) whose labels are used')
    p.add_argument('--label-blocks',type=int,default=2)
    p.add_argument('--fg-frac',type=float,default=0.0,help='fraction of training draws forced to contain faults (nnU-Net style foreground oversampling)')
    p.add_argument('--fg-min',type=float,default=0.005,help='minimum fault fraction (of labelled support voxels) for a forced-foreground draw')
    p.add_argument('--semi',action='store_true',help='mean-teacher semi-supervision on the label-hidden rest of the train region')
    p.add_argument('--ema',type=float,default=0.99); p.add_argument('--lambda-u',type=float,default=1.0); p.add_argument('--ramp',type=int,default=20)
    p.add_argument('--no-smc',action='store_true',help='disable spatial-masking consistency (keep copy-paste only)')
    p.add_argument('--box',type=int,default=64); p.add_argument('--mask-boxes',type=int,default=4); p.add_argument('--mask-size',type=int,default=32)
    p.add_argument('--per-rank',type=int,default=1); p.add_argument('--accum',type=int,default=2,
        help='effective batch = world * per_rank * accum (kept at 8 for every model)')
    a=p.parse_args(); rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
    micro=a.per_rank*a.accum
    local=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local); device=torch.device('cuda',local)
    torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=True   # cuDNN algorithm autotuning (False was for bitwise reproducibility; ~3x slower 3D convs)
    out=Path(a.run); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume: raise RuntimeError('Existing run: use --resume or new directory')
    blocks=label_blocks(a.label_frac,a.label_blocks)
    tr=SpatialThebe('train',a.root,samples=a.samples,seed=a.seed,labeled_ranges=blocks,fg_frac=a.fg_frac,fg_min=a.fg_min)
    un=SpatialThebe('train',a.root,samples=a.samples,seed=a.seed+777,labeled_ranges=blocks,unlabeled=True) if a.semi else None
    va=SpatialThebe('val',a.root,fast=True); full=SpatialThebe('val',a.root)
    if a.smoke: va.origins=va.origins[:8]
    assert a.samples%(world*micro)==0, 'Epoch must contain whole effective batches'
    sampler=DistributedSampler(tr,num_replicas=world,rank=rank,seed=a.seed,shuffle=True)
    tl=DataLoader(tr,batch_size=a.per_rank,sampler=sampler,num_workers=a.workers,pin_memory=True,persistent_workers=False)
    if a.semi:
        usampler=DistributedSampler(un,num_replicas=world,rank=rank,seed=a.seed+777,shuffle=True)
        ul=DataLoader(un,batch_size=a.per_rank,sampler=usampler,num_workers=a.workers,pin_memory=True,persistent_workers=False)
    def vl(ds): return DataLoader(Subset(ds,list(range(rank,len(ds),world))),batch_size=1,num_workers=a.workers,pin_memory=True)
    fast_loader,full_loader=vl(va),vl(full)
    model_name,mkw,_,_=VARIANTS[a.variant]
    model=build(model_name,**mkw).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda'); model=DDP(model,device_ids=[local],broadcast_buffers=False)
    teacher=None
    if a.semi:                                  # EMA teacher, never trained directly, identical on every rank
        teacher=copy.deepcopy(model.module).eval()
        for q in teacher.parameters(): q.requires_grad_(False)
    def ema_update():
        with torch.no_grad():
            for pt,ps in zip(teacher.parameters(),model.module.parameters()): pt.mul_(a.ema).add_(ps.detach(),alpha=1-a.ema)
            for bt,bs in zip(teacher.buffers(),model.module.buffers()): bt.copy_(bs)
    eval_model=lambda: teacher if a.semi else model.module
    params=sum(p.numel() for p in model.parameters()); start=1; best=-1.; history=[]
    protocol=tr.manifest['protocol_sha256']
    if a.resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==protocol
        for key in ('epochs','samples','seed','variant','per_rank','accum','fg_frac','fg_min'):
            assert ck['args'].get(key,vars(a)[key])==vars(a)[key], f'Cannot change {key} on resume'
        assert ck['world_size']==world
        model.module.load_state_dict(ck['student'] if ck.get('student') is not None else ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        if a.semi: teacher.load_state_dict(ck['model'])
        start=ck['epoch']+1; best=ck['best']; history=ck['history']
        states=ck['rng'][rank]; torch.set_rng_state(states['cpu']); torch.cuda.set_rng_state(states['cuda'],device)
        random.setstate(states['python']); np.random.set_state(states['numpy'])
    if rank==0:
        model_file=Path(sys.modules[type(model.module).__module__].__file__)
        sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),ROOT/'train/dataset_thebe_spatial.py',model_file,ROOT/'models/__init__.py',ROOT/'scripts/prepare_thebe_spatial.py']}
        for rel in sources:
            dest=out/'source_snapshot'/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/rel).read_bytes())
        (out/'config.json').write_text(json.dumps(dict(args=vars(a),params=params,world_size=world,effective_batch=world*micro,
            model=model_name,model_kwargs={k:(list(v) if isinstance(v,tuple) else v) for k,v in mkw.items()},
            labeled_ranges=blocks,semi=a.semi,
            protocol_sha256=protocol,sources=sources,initialization=(('pretrained 2-D %s from timm (%s), rest random'%(mkw.get('sam') or mkw.get('backbone'),'frozen' if mkw.get('freeze_sam') else 'fine-tuned')) if mkw.get('pretrained') else 'random; no pretrained weights'),
            full_val_cores=len(full),fast_val_cores=len(va),torch_version=torch.__version__,
            device_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]),indent=2))
        (out/'data_manifest.json').write_text(json.dumps(tr.manifest,indent=2))
        (out/'validation_origins.json').write_text(json.dumps({'fast':va.origins,'full':full.origins}))
        print(f'{a.variant} ({model_name}) params={params:,} world={world} batch={world*micro} train={len(tr)} fast={len(va)} full={len(full)}',flush=True)
    dist.barrier()
    for epoch in range(start,min(a.epochs,a.stop_after)+1):
        t0=time.time(); model.train(); tr.epoch=epoch; sampler.set_epoch(epoch)
        lr=(1e-6+(1e-4-1e-6)*(epoch-1)/9) if epoch<=10 else 1e-7+.5*(1e-4-1e-7)*(1+math.cos(math.pi*(epoch-10)/(a.epochs-10)))
        for group in opt.param_groups: group['lr']=lr
        opt.zero_grad(set_to_none=True); losses=torch.zeros(2,dtype=torch.float64,device=device); skipped=0
        lam=a.lambda_u*(math.exp(-5*(1-min(epoch,a.ramp)/a.ramp)**2) if a.ramp>0 else 1.0)
        if a.semi: usampler.set_epoch(epoch); un.epoch=epoch
        it=zip(tl,ul) if a.semi else ((b,None) for b in tl)
        unsup_acc=torch.zeros(2,dtype=torch.float64,device=device)
        for i,(bl,bu) in enumerate(it):
            x,y,m,ml=[t.to(device,non_blocking=True) for t in bl]
            sync=(i+1)%a.accum==0
            with contextlib.nullcontext() if sync else model.no_sync():
                with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
                loss=loss_fn(logits,y,ml)                                   # supervised: labelled voxels only
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss: epoch {epoch} step {i}')
                scaler.scale(loss/a.accum).backward()
                if a.semi:
                    xu,_,mu,_=[t.to(device,non_blocking=True) for t in bu]
                    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16): pt=teacher(xu).float().sigmoid()
                    pseudo=(pt>.5).float(); conf=((pt>.7)|(pt<.3)).float()*mu
                    # copy-paste: the 64^3 box of the labelled cube with the most labelled faults, pasted into the unlabelled cube
                    B=a.box; dens=F.avg_pool3d((y*ml).float(),B,stride=16)
                    Xm=xu.clone(); Tm=pseudo.clone(); Mm=conf.clone()
                    for b in range(x.shape[0]):
                        idx=int(dens[b,0].flatten().argmax()); dz,dy,dx=np.unravel_index(idx,dens.shape[2:]); z0,y0,x0=dz*16,dy*16,dx*16
                        sl=(b,slice(None),slice(z0,z0+B),slice(y0,y0+B),slice(x0,x0+B))
                        Xm[sl]=x[sl]; Tm[sl]=y[sl]; Mm[sl]=ml[sl]
                    with torch.autocast('cuda',dtype=torch.float16): lm=model(Xm)
                    lcp=loss_fn(lm,Tm,Mm); scaler.scale(lam*lcp/a.accum).backward(); ltot=lcp.detach()
                    if not a.no_smc:                                       # spatial-masking consistency
                        Xs=xu.clone(); S=a.mask_size
                        for b in range(x.shape[0]):
                            for _ in range(a.mask_boxes):
                                z0,y0,x0=[int(v) for v in np.random.randint(0,128-S,3)]; Xs[b,:,z0:z0+S,y0:y0+S,x0:x0+S]=0
                        with torch.autocast('cuda',dtype=torch.float16): ls_=model(Xs)
                        lsmc=loss_fn(ls_,pseudo,conf); scaler.scale(lam*lsmc/a.accum).backward(); ltot=ltot+lsmc.detach()
                    unsup_acc[0]+=ltot; unsup_acc[1]+=1
            if sync:
                oldscale=scaler.get_scale(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                skipped+=int(scaler.get_scale()<oldscale)
                if a.semi and scaler.get_scale()>=oldscale: ema_update()
            losses[0]+=loss.detach(); losses[1]+=1
            if rank==0 and (i==0 or (i+1)%100==0):
                print(f'epoch {epoch} microstep {i+1}/{len(tl)} loss={loss.item():.5f} lr={lr:.3g} elapsed={time.time()-t0:.0f}s',flush=True)
        dist.all_reduce(losses)
        if a.semi: dist.all_reduce(unsup_acc)
        fast=evaluate(eval_model(),fast_loader,device)
        exhaustive=None
        if not a.smoke and (epoch%a.full_every==0 or epoch==min(a.epochs,a.stop_after)):
            if rank==0: print(f'epoch {epoch}: full validation starts ({len(full)} cores)',flush=True)
            exhaustive=evaluate(eval_model(),full_loader,device)
            expected=full.record['support_voxels']
            assert exhaustive['valid_voxels']==expected,(exhaustive['valid_voxels'],expected)
        improved=exhaustive is not None and exhaustive['iou']>best
        if improved: best=exhaustive['iou']
        row=dict(epoch=epoch,lr=lr,train_loss=(losses[0]/losses[1]).item(),fast_val=fast,full_val=exhaustive,
            seconds=time.time()-t0,skipped_optimizer_steps=skipped,
            unsup_loss=(unsup_acc[0]/unsup_acc[1]).item() if a.semi and unsup_acc[1]>0 else None,lambda_u=lam if a.semi else None)
        history.append(row)
        state=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),python=random.getstate(),numpy=np.random.get_state())
        states=[None]*world; dist.all_gather_object(states,state)
        if rank==0:
            checkpoint=dict(epoch=epoch,model=eval_model().state_dict(),student=model.module.state_dict() if a.semi else None,
                optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                best=best,history=history,rng=states,args=vars(a),world_size=world,protocol_sha256=protocol)
            atomic_save(checkpoint,out/'last.pt')
            if improved: atomic_save(checkpoint,out/'best_fullval.pt')
            if exhaustive is not None: atomic_save(checkpoint,out/f'epoch_{epoch:03d}.pt')
            (out/'history.json').write_text(json.dumps(history,indent=2)); plot_history(history,out)
            (out/'status.json').write_text(json.dumps(dict(status='running',last_completed_epoch=epoch,best_fullval_iou=best),indent=2))
            print(json.dumps(row),flush=True)
        dist.barrier()
    if rank==0:
        (out/'status.json').write_text(json.dumps(dict(status='completed_trial' if a.stop_after<a.epochs else 'completed',
            last_completed_epoch=min(a.epochs,a.stop_after),best_fullval_iou=best),indent=2))
    dist.destroy_process_group()

if __name__=='__main__': main()
