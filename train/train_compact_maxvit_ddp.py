"""Fresh M0 training on v2; epoch-boundary resumable, four-GPU DDP.

Historical MaxViT comparison is a reference, not an exact paired experiment:
the architecture/initialization, explicit FP32 loss and numerical execution differ.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from scipy.ndimage import gaussian_filter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build
from train.dataset import FaultVolumes, CATEGORIES
from train.train_old_recipe import dice_focal_loss, warmup_cosine_lr


def atomic_json(data, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)); tmp.replace(path)


def atomic_save(data, path):
    path = Path(path); tmp = path.with_suffix('.tmp.pt')
    torch.save(data, tmp); tmp.replace(path)


class SeededVolumes(Dataset):
    """Same seismic augmentation draws as GeometryVolumes, without geometry I/O."""
    def __init__(self, root, seed):
        self.base = FaultVolumes(root, 'train'); self.seed = seed; self.epoch = 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        name, cat = self.base.items[i]
        x = self.base._read('seismic', name, np.float32)
        y = (self.base._read('labels', name, np.uint8) > 0).astype(np.float32)
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + i)
        axes = [(0, 1), (0, 2), (1, 2)][rng.integers(3)]; k = int(rng.integers(4))
        x, y = np.rot90(x, k, axes).copy(), np.rot90(y, k, axes).copy()
        for axis in range(3):
            if rng.random() < .5:
                x, y = np.flip(x, axis).copy(), np.flip(y, axis).copy()
        if rng.random() < .35:
            x = gaussian_filter(x, sigma=rng.uniform(.4, 1.0))
        x = x * rng.uniform(.8, 1.2) + rng.uniform(-.15, .15)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(np.ascontiguousarray(x[None], dtype=np.float32)), torch.from_numpy(y[None]), CATEGORIES.index(cat)


@torch.no_grad()
def validate(module, loader, device):
    for buffer in module.buffers():
        dist.broadcast(buffer, 0)
    module.eval()
    stats = torch.zeros(2 + 3 * (1 + len(CATEGORIES)), dtype=torch.float64, device=device)
    for x, y, cat in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = module(x)
        loss = dice_focal_loss(logits.float(), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite validation loss')
        pred, truth = logits.float().sigmoid() > .5, y > .5
        counts = torch.stack([(pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()]).double()
        stats[0] += loss; stats[1] += 1; stats[2:5] += counts
        k = int(cat[0]); stats[5 + 3*k:8 + 3*k] += counts
    dist.all_reduce(stats)
    def score(c):
        tp, fp, fn = c.tolist()
        return dict(iou=tp/max(tp+fp+fn, 1), dice=2*tp/max(2*tp+fp+fn, 1),
                    precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1))
    result = dict(loss=float(stats[0]/stats[1]), **score(stats[2:5]))
    result.update({f'iou_{c}': score(stats[5+3*k:8+3*k])['iou'] for k, c in enumerate(CATEGORIES)})
    return result


def amp_step(module, optimizer, scaler, device):
    scaler.unscale_(optimizer)
    norms = torch._foreach_norm([p.grad for p in module.parameters() if p.grad is not None])
    norm = torch.stack(norms).double().norm()
    finite = torch.isfinite(norm).int(); dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        optimizer.zero_grad(set_to_none=True)
        scaler.update(new_scale=scaler.get_scale() * .5)
        state = scaler.state_dict(); state['_growth_tracker'] = 0; scaler.load_state_dict(state)
        return False
    scaler.step(optimizer); scaler.update()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--data', default='data/dataset_reproduction_v2')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--per-rank', type=int, default=2)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stop-after-epoch', type=int, default=0, help='Orderly epoch-boundary stop for resume checks')
    a = p.parse_args()
    local = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local)
    dist.init_process_group('nccl'); rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', local); torch.set_num_threads(4)
    random.seed(a.seed + rank); np.random.seed(a.seed + rank); torch.manual_seed(a.seed + rank)
    torch.backends.cudnn.benchmark = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pt').exists() or (out/'log.csv').exists():
        if not a.resume:
            raise RuntimeError('Existing run: use --resume or choose a new output directory')
    mkw = dict(base=36, drop_path_rate=.2, checkpoint_highres=True)
    module = build('compact_maxvit3d', **mkw).to(device)
    model = DDP(module, device_ids=[local], gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=.01, foreach=False)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    train = SeededVolumes(a.data, a.seed)
    val = FaultVolumes(a.data, 'val')
    val_indices = list(range(rank, 4 if a.smoke else len(val), world))
    vl = DataLoader(Subset(val, val_indices), batch_size=1, num_workers=min(a.workers, 2), pin_memory=True)
    sources = ['models/compact_maxvit3d.py', 'models/maxvit3d.py', 'train/train_compact_maxvit_ddp.py',
               'train/dataset.py', 'train/train_old_recipe.py']
    config = {**{k:v for k,v in vars(a).items() if k not in ('resume', 'stop_after_epoch')},
              'model': 'compact_maxvit3d', 'mkw': mkw, 'params': sum(x.numel() for x in module.parameters()),
              'world': world, 'global_batch': world*a.per_rank*a.accum,
              'init': 'random; no pretrained or previous-run weights',
              'loss': 'explicit FP32 0.6 Dice + 0.4 Focal(alpha=.75,gamma=2)',
              'initial_amp_scale': 1024, 'optimizer': 'AdamW wd=.01; no clipping',
              'schedule': '10 epoch warmup 1e-6 to peak, cosine to 1e-7; fixed 200 epoch budget',
              'selection': 'minimum validation segmentation loss, same criterion as historical baseline',
              'norm': 'InstanceNorm in detail/decoder, per-rank BatchNorm in MBConv, LayerNorm in attention',
              'manifest_sha256': hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest(),
              'source_hashes': {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
              'n_train':len(train), 'n_val':len(val), 'torch':torch.__version__,
              'reference':'runs/maxvit3d_v2_oldrecipe_all3', 'data_source_sha256':train.base.source_sha256}
    best, updates, skips, start = float('inf'), 0, 0, 1
    if a.resume:
        old = json.loads((out/'args.json').read_text())
        if old != config:
            raise RuntimeError('Resume configuration/source mismatch: '+str([k for k in config if old.get(k)!=config[k]]))
        ck = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        module.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        best, updates, skips, start = ck['best'], ck['updates'], ck['skips'], ck['epoch']+1
        rng = ck['rng_by_rank'][rank]
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'], device)
        # A crash after CSV write but before checkpoint rename may leave one extra row.
        if rank == 0:
            with (out/'log.csv').open() as f:
                reader = csv.DictReader(f); fields_old = reader.fieldnames
                keep = [r for r in reader if int(r['epoch']) < start]
            with (out/'log.csv').open('w', newline='') as f:
                w=csv.DictWriter(f, fieldnames=fields_old); w.writeheader(); w.writerows(keep)
            with (out/'resume_events.jsonl').open('a') as f:
                f.write(json.dumps({'time':time.time(), 'from_epoch':start-1, 'updates':updates,
                                    'optimizer_state_entries':len(optimizer.state), 'scaler':scaler.state_dict(),
                                    'restored_rng_ranks':len(ck['rng_by_rank'])})+'\n')
        del ck
    elif rank == 0:
        atomic_json(config, out/'args.json')
    fields = ['epoch','train_loss','train_seg','val_loss','val_iou','val_dice','val_precision','val_recall','lr','seconds',
              'updates','skipped_updates','amp_scale','peak_memory_mib']+[f'val_iou_{c}' for c in CATEGORIES]
    log = (out/'log.csv').open('a' if a.resume else 'w', newline='') if rank==0 else None
    writer = csv.DictWriter(log, fieldnames=fields) if rank==0 else None
    if rank==0:
        if not a.resume: writer.writeheader(); log.flush()
        print(f'M0 CompactMaxViT {config["params"]:,} parameters; global batch {config["global_batch"]}; start epoch {start}', flush=True)
    for epoch in range(start, a.epochs+1):
        lr = float(warmup_cosine_lr(epoch,a.epochs,10,1e-6,a.lr,1e-7))
        for g in optimizer.param_groups: g['lr']=lr
        train.epoch=epoch
        order=np.random.default_rng(a.seed+epoch).permutation(len(train)).tolist()
        if a.smoke: order=order[:config['global_batch']*3]
        assert len(order)%config['global_batch']==0
        subset=Subset(train,order)
        sampler=DistributedSampler(subset,num_replicas=world,rank=rank,shuffle=False,drop_last=False)
        dl=DataLoader(subset,sampler=sampler,batch_size=a.per_rank,num_workers=a.workers,pin_memory=True)
        model.train(); t0=time.time(); total=torch.zeros(2,device=device,dtype=torch.float64)
        torch.cuda.reset_peak_memory_stats(device)
        for step,(x,y,_) in enumerate(dl):
            if step%a.accum==0: optimizer.zero_grad(set_to_none=True)
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            last=(step+1)%a.accum==0
            with (contextlib.nullcontext() if last else model.no_sync()):
                with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
                loss=dice_focal_loss(logits.float(),y)
                finite=torch.isfinite(loss).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item(): raise RuntimeError(f'Nonfinite forward loss at {epoch}/{step}')
                scaler.scale(loss/a.accum).backward()
            if last:
                applied=amp_step(module,optimizer,scaler,device)
                updates+=int(applied);skips+=int(not applied)
                if rank==0 and not applied: print(f'AMP skip epoch={epoch} step={step+1} scale={scaler.get_scale()}',flush=True)
            total[0]+=loss.detach().double();total[1]+=1
            if rank==0 and (step==0 or (step+1)%10==0):
                atomic_json({'state':'training','epoch':epoch,'step':step+1,'steps':len(dl),'updated':time.time(),
                             'updates':updates,'skipped_updates':skips},out/'status.json')
                print(f'epoch={epoch:03d} step={step+1}/{len(dl)} loss={float(loss):.5f}',flush=True)
        dist.all_reduce(total)
        if rank==0: atomic_json({'state':'validating','epoch':epoch,'updated':time.time()},out/'status.json')
        v=validate(module,vl,device);dt=time.time()-t0
        improved=v['loss']<best
        if improved: best=v['loss']
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
             'cuda':torch.cuda.get_rng_state(device)}
        rngs=[None]*world if rank==0 else None
        dist.gather_object(rng,rngs,dst=0)
        if rank==0:
            row=dict(epoch=epoch,train_loss=float(total[0]/total[1]),train_seg=float(total[0]/total[1]),
                     val_loss=v['loss'],val_iou=v['iou'],val_dice=v['dice'],val_precision=v['precision'],
                     val_recall=v['recall'],lr=lr,seconds=dt,updates=updates,skipped_updates=skips,
                     amp_scale=scaler.get_scale(),peak_memory_mib=torch.cuda.max_memory_allocated(device)/2**20)
            row.update({f'val_iou_{c}':v[f'iou_{c}'] for c in CATEGORIES})
            # best.pt keeps the legacy evaluator's model/mkw format.
            ck_args={**config,'mkw':[f'{k}={int(val) if isinstance(val,bool) else val}' for k,val in mkw.items()]}
            state={'model':module.state_dict(),'epoch':epoch,'val':v,'args':ck_args,
                   'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng_by_rank':rngs,
                   'best':best,'updates':updates,'skips':skips,'schedule_epoch':epoch,'config':config}
            if improved: atomic_save(state,out/'best.pt')
            atomic_save(state,out/'last.pt')
            writer.writerow(row);log.flush()
            atomic_json({'state':'epoch_complete','epoch':epoch,'val':v,'updated':time.time(),
                         'updates':updates,'skipped_updates':skips},out/'status.json')
            print(f'EPOCH {epoch:03d}/{a.epochs}: train={row["train_loss"]:.5f} val={v["loss"]:.5f} '
                  f'IoU={v["iou"]:.6f} P={v["precision"]:.4f} R={v["recall"]:.4f} '
                  f'{dt:.1f}s peak={row["peak_memory_mib"]:.0f}MiB',flush=True)
        dist.barrier()
        if a.stop_after_epoch and epoch>=a.stop_after_epoch: break
    if rank==0:
        log.close();atomic_json({'state':'complete' if epoch==a.epochs else 'paused',
                                'epoch':epoch,'updated':time.time()},out/'status.json')
    dist.destroy_process_group()


if __name__=='__main__':
    main()
