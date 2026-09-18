"""Four-GPU controlled continuation with checkpoint/resume and final held-out test.

Each rank processes two volumes. DDP averages gradients across four ranks,
giving global batch 8 without accumulation. BatchNorm remains per-rank with
standard DDP buffer broadcasts; all compared variants use the same policy.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.maxvit3d_enhanced import EnhancedMaxViT3D
from train.dataset import FaultVolumes
from train.geometry_dataset import GeometryVolumes
from train.geometry_losses import segmentation_loss, profile_loss, trace_loss
from train.train import evaluate
from train.train_model_upgrade import atomic_save, sha


def safe_amp_step(module, optimizer, scaler, device):
    """Synchronously skip AMP overflow, reduce scale, and retain finite weights.

    Returns (stepped, norm, old_scale, new_scale). Also handles norm overflow
    conservatively instead of allowing a corrupted parameter update.
    """
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
    finite = torch.isfinite(norm).to(dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    old_scale = scaler.get_scale()
    if not finite.item():
        optimizer.zero_grad(set_to_none=True)
        scaler.update(new_scale=old_scale*.5)
        state = scaler.state_dict()
        state['_growth_tracker'] = 0
        scaler.load_state_dict(state)
        return False, float(norm), old_scale, scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return True, float(norm), old_scale, scaler.get_scale()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['baseline', 'architecture', 'loss', 'combined'], required=True)
    p.add_argument('--data', default='data/dataset_reproduction_v2')
    p.add_argument('--cache', default='data/geometry_model_upgrade_v1')
    p.add_argument('--init', default='runs/maxvit3d_v2_oldrecipe_all3/best.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--new-lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--resume-numerics-fix', action='store_true',
                   help='Allow only the documented trainer/supervisor AMP recovery patch on resume')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if (out/'log.csv').exists() and not a.resume:
            raise RuntimeError('Existing run; choose a new directory or use --resume')
    dist.barrier()
    use_arch, use_loss = a.variant in ['architecture', 'combined'], a.variant in ['loss', 'combined']
    mkw = dict(full_res_skip=True, img_size=128, drop_path_rate=.2, d2s=use_arch)
    module = EnhancedMaxViT3D(**mkw).to(device)
    initial = torch.load(a.init, map_location='cpu', weights_only=False)
    keys = module.load_state_dict(initial['model'], strict=False)
    assert not keys.unexpected_keys
    assert all(k.startswith(('fusion32.', 'fusion64.')) for k in keys.missing_keys)
    del initial
    model = DDP(module, device_ids=[local_rank], gradient_as_bucket_view=True, bucket_cap_mb=25)
    old = [p for n, p in module.named_parameters() if not n.startswith(('fusion32.', 'fusion64.'))]
    new = [p for n, p in module.named_parameters() if n.startswith(('fusion32.', 'fusion64.'))]
    groups = [{'params': old, 'lr': a.lr, 'peak_lr': a.lr}]
    if new:
        groups.append({'params': new, 'lr': a.new_lr, 'peak_lr': a.new_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01, foreach=False)
    scaler = torch.amp.GradScaler('cuda')
    data = GeometryVolumes(a.data, a.cache, seed=a.seed, geometry=use_loss)
    config = {**vars(a), 'kind': 'DDP warm-start pilot', 'world_size': world,
              'global_batch': a.batch*world, 'accumulation': 1, 'model_kwargs': mkw,
              'batchnorm': 'per-rank BN, DDP broadcast_buffers=True; not SyncBatchNorm',
              'initial_checkpoint_sha256': sha(a.init), 'manifest_sha256': sha(Path(a.data)/'manifest.json'),
              'n_train': len(data), 'n_val': 100, 'n_test': 100,
              'lambda_profile': .1 if use_loss else 0, 'lambda_trace': .05 if use_loss else 0,
              'selection': 'maximum validation voxel IoU at threshold .5, including epoch 0',
              'torch': torch.__version__, 'parameters': sum(t.numel() for t in module.parameters()),
              'source_hashes': {str(q.relative_to(ROOT)): sha(q) for folder in ['models', 'train', 'scripts']
                                for q in (ROOT/folder).glob('*.py')}}
    cache = json.loads((Path(a.cache)/'report.json').read_text())
    assert cache['manifest_sha'] == config['manifest_sha256']
    def validation(split):
        # Avoid DDP forward here: only rank zero evaluates, using unwrapped module.
        dist.barrier()
        obj = [None]
        if rank == 0:
            loader = DataLoader(FaultVolumes(a.data, split), batch_size=1,
                                num_workers=a.workers, pin_memory=True)
            obj[0] = evaluate(module, loader, device, amp=True)
        dist.broadcast_object_list(obj, src=0, device=device)
        return obj[0]
    start, best, updates = 1, -1., 0
    if a.resume:
        previous = json.loads((out/'config.json').read_text())
        for key in ['variant', 'epochs', 'world_size', 'batch', 'lr', 'new_lr', 'seed',
                    'initial_checkpoint_sha256', 'manifest_sha256']:
            assert config[key] == previous[key], f'Resume mismatch: {key}'
        changed = {key for key in set(config['source_hashes']) | set(previous['source_hashes'])
                   if config['source_hashes'].get(key) != previous['source_hashes'].get(key)}
        allowed = {'train/train_model_upgrade_ddp.py', 'scripts/run_model_upgrade_ddp.py'}
        if changed and not (a.resume_numerics_fix and changed <= allowed):
            raise RuntimeError(f'Resume source change not authorized by numerical migration: {changed}')
        ck = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        module.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        start, best, updates = ck['epoch']+1, ck['best_iou'], ck['updates']
        if rank == 0:
            with open(out/'resume_events.jsonl', 'a') as audit:
                audit.write(json.dumps({'unix': time.time(), 'from_epoch': ck['epoch'],
                                        'changed_sources': sorted(changed),
                                        'effective_config': config})+'\n')
        del ck
    else:
        if rank == 0:
            (out/'config.json').write_text(json.dumps(config, indent=2))
        if not a.smoke:
            v = validation('val'); best = v['iou']
            if rank == 0:
                atomic_save({'model': module.state_dict(), 'epoch': 0, 'val': v, 'config': config}, out/'best.pt')
                (out/'initial_val.json').write_text(json.dumps(v, indent=2))
                print(f'INITIAL {a.variant} val_iou={best:.6f}', flush=True)
    fields = ['epoch', 'loss', 'seg', 'profile', 'trace', 'valid_profiles', 'valid_windows',
              'aux_ramp', 'val_iou', 'val_dice', 'val_precision', 'val_recall', 'old_lr', 'seconds',
              'updates', 'peak_memory_mib', 'max_rank_parameter_difference']
    log = None
    if rank == 0:
        log = open(out/'log.csv', 'a' if a.resume else 'w', newline='')
        writer = csv.DictWriter(log, fieldnames=fields)
        if not a.resume:
            writer.writeheader()
        print(f'DDP {a.variant}: world={world}, global_batch={world*a.batch}, epochs={a.epochs}', flush=True)
    for epoch in range(start, a.epochs+1):
        dist.barrier(); t0 = time.time(); data.epoch = epoch
        order = np.random.default_rng(a.seed+epoch).permutation(len(data)).tolist()
        if a.smoke:
            order = order[:a.batch*world*3]
        subset = Subset(data, order)
        sampler = DistributedSampler(subset, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        loader = DataLoader(subset, sampler=sampler, batch_size=a.batch, num_workers=a.workers,
                            pin_memory=True, drop_last=False)
        assert len(order) % (world*a.batch) == 0
        ramp = min(max((epoch-2)/4, 0), 1) if not a.smoke else 1.
        factor = min(epoch/2, 1)*(.01+.99*.5*(1+math.cos(math.pi*(epoch-1)/max(a.epochs-1, 1))))
        for group in optimizer.param_groups:
            group['lr'] = group['peak_lr']*factor
        model.train(); torch.cuda.reset_peak_memory_stats(device)
        sums = np.zeros(6, np.float64)
        overflow_streak, overflow_count = 0, 0
        for step, (x, y, _, q, normal, valid, windows) in enumerate(loader):
            torch.manual_seed(a.seed+epoch*100003+step*world+rank)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x)
            with torch.autocast('cuda', enabled=False):
                seg = segmentation_loss(logits, y)
                pr, tr = logits.float().sum()*0, logits.float().sum()*0
                if use_loss:
                    prob = logits.float().sigmoid()
                    pr = profile_loss(prob, y, q.to(device), normal.to(device), valid.to(device))
                    tr = trace_loss(prob, y, windows)
                loss = seg+ramp*(.1*pr+.05*tr)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Rank {rank} nonfinite loss')
            scaler.scale(loss).backward()
            stepped, norm, old_scale, new_scale = safe_amp_step(module, optimizer, scaler, device)
            if stepped:
                updates += 1
                overflow_streak = 0
            else:
                overflow_streak += 1
                overflow_count += 1
                if rank == 0:
                    event = {'epoch': epoch, 'step': step+1, 'old_scale': old_scale,
                             'new_scale': new_scale, 'action': 'all ranks skipped parameter update',
                             'loss': loss.item(), 'norm': str(norm)}
                    with open(out/'amp_overflows.jsonl', 'a') as events:
                        events.write(json.dumps(event)+'\n')
                    print(f'AMP_RECOVERED {json.dumps(event)}', flush=True)
                if overflow_streak >= 4 or new_scale < 1:
                    raise FloatingPointError('Persistent nonfinite gradients after reducing AMP scale')
            sums += [loss.item(), seg.item(), pr.item(), tr.item(), valid.sum().item()/len(x),
                     (windows[:, :, 0] >= 0).sum().item()/len(x)]
            if rank == 0 and (step == 0 or (step+1) % 25 == 0):
                print(f'{a.variant} epoch={epoch} step={step+1}/{len(loader)} loss={loss.item():.5f} '
                      f'profile={pr.item():.5f} trace={tr.item():.5f}', flush=True)
        means = torch.tensor(sums/len(loader), device=device)
        dist.all_reduce(means); means /= world
        peak = torch.tensor(torch.cuda.max_memory_allocated(device)/2**20, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        # Check actual parameter synchronization, excluding intentionally local BN buffers.
        selected = torch.cat([p.detach().flatten()[:4].float() for p in module.parameters()])
        minimum, maximum = selected.clone(), selected.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        parameter_difference = (maximum-minimum).abs().max().item()
        assert parameter_difference < 1e-6, parameter_difference
        if a.smoke:
            if rank == 0:
                result = {'world_size': world, 'global_batch': world*a.batch, 'updates': updates,
                          'losses': means.tolist(), 'peak_memory_mib': peak.item(),
                          'seconds': time.time()-t0, 'max_rank_parameter_difference': parameter_difference}
                (out/'smoke.json').write_text(json.dumps(result, indent=2))
                print(f'DDP SMOKE PASS {json.dumps(result)}', flush=True)
                log.close()
            dist.destroy_process_group(); return
        v = validation('val')
        row = dict(zip(fields[:7], [epoch, *means.tolist()]))
        row.update(aux_ramp=ramp, val_iou=v['iou'], val_dice=v['dice'], val_precision=v['precision'],
                   val_recall=v['recall'], old_lr=optimizer.param_groups[0]['lr'], seconds=time.time()-t0,
                   updates=updates, peak_memory_mib=peak.item(), max_rank_parameter_difference=parameter_difference)
        improved = v['iou'] > best
        best = max(best, v['iou'])
        if rank == 0:
            writer.writerow(row); log.flush()
            if improved:
                atomic_save({'model': module.state_dict(), 'epoch': epoch, 'val': v,
                             'config': config}, out/'best.pt')
            atomic_save({'model': module.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict(), 'epoch': epoch, 'updates': updates,
                         'best_iou': best, 'config': config}, out/'last.pt')
            (out/'status.json').write_text(json.dumps({'status': 'training', 'epoch': epoch,
                                                       'best_val_iou': best, 'updates': updates,
                                                       'amp_overflows_this_epoch': overflow_count,
                                                       'amp_scale': scaler.get_scale()}, indent=2))
            print(f'EPOCH {a.variant} {json.dumps(row)} best={best:.6f}', flush=True)
    if rank == 0:
        log.close()
        ck = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
        module.load_state_dict(ck['model'])
    result = validation('test')
    if rank == 0:
        result.update(variant=a.variant, epoch=ck['epoch'], n_test=100,
                      reference_historical_test_iou=.8325868632244817,
                      difference_from_historical_iou=result['iou']-.8325868632244817,
                      checkpoint_sha256=sha(out/'best.pt'), selection=config['selection'])
        (out/'test.json').write_text(json.dumps(result, indent=2))
        (out/'status.json').write_text(json.dumps({'status': 'complete', 'test': result}, indent=2))
        print(f'COMPLETE {a.variant} {json.dumps(result)}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
