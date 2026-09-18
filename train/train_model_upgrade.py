"""Controlled warm-start pilot: baseline / architecture / loss / combined.

All variants use the same historical checkpoint, samples, augmentation, old
parameter LR, FP32 loss, validation selection and fixed optimization budget.
This is a continuation experiment, not a from-scratch paper comparison.
"""
import argparse
import csv
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.maxvit3d_enhanced import EnhancedMaxViT3D
from train.dataset import FaultVolumes
from train.geometry_dataset import GeometryVolumes
from train.geometry_losses import segmentation_loss, profile_loss, trace_loss
from train.train import evaluate


def atomic_save(obj, dest):
    temp = dest.with_suffix('.tmp.pt')
    torch.save(obj, temp)
    os.replace(temp, dest)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['baseline', 'architecture', 'loss', 'combined'], required=True)
    p.add_argument('--data', default='data/dataset_reproduction_v2')
    p.add_argument('--cache', default='data/geometry_model_upgrade_v1')
    p.add_argument('--init', default='runs/maxvit3d_v2_oldrecipe_all3/best.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--new-lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='two updates only; no test evaluation')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'log.csv').exists() and not a.resume:
        raise RuntimeError('Run already exists; use a new directory or --resume')
    device = torch.device(f'cuda:{a.gpu}')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    use_arch = a.variant in ['architecture', 'combined']
    use_loss = a.variant in ['loss', 'combined']
    model_kwargs = dict(full_res_skip=True, img_size=128, drop_path_rate=.2, d2s=use_arch)
    model = EnhancedMaxViT3D(**model_kwargs).to(device)
    initial = torch.load(a.init, map_location='cpu', weights_only=False)
    incompatible = model.load_state_dict(initial['model'], strict=False)
    if incompatible.unexpected_keys or any(not k.startswith(('fusion32.', 'fusion64.')) for k in incompatible.missing_keys):
        raise RuntimeError(str(incompatible))
    del initial
    old, new = [], []
    for name, param in model.named_parameters():
        (new if name.startswith(('fusion32.', 'fusion64.')) else old).append(param)
    groups = [{'params': old, 'lr': a.lr, 'peak_lr': a.lr}]
    if new:
        groups.append({'params': new, 'lr': a.new_lr, 'peak_lr': a.new_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    dataset = GeometryVolumes(a.data, a.cache, seed=a.seed, geometry=use_loss)
    cache_report = json.loads((Path(a.cache)/'report.json').read_text())
    if cache_report['manifest_sha'] != sha(Path(a.data)/'manifest.json'):
        raise RuntimeError('Geometry cache manifest mismatch')
    val = DataLoader(FaultVolumes(a.data, 'val'), batch_size=1, num_workers=a.workers, pin_memory=True)
    config = {**vars(a), 'kind': 'warm_start_controlled_pilot', 'model_kwargs': model_kwargs,
              'initial_checkpoint_sha256': sha(a.init), 'data_manifest_sha256': sha(Path(a.data)/'manifest.json'),
              'selection': 'maximum validation voxel IoU at threshold 0.5; includes initial epoch 0',
              'lambda_profile': .1 if use_loss else 0, 'lambda_trace': .05 if use_loss else 0,
              'n_train': len(dataset), 'n_val': len(val.dataset),
              'source_hashes': {str(q.relative_to(ROOT)): sha(q) for folder in ['models', 'train', 'scripts']
                                for q in (ROOT/folder).glob('*.py')},
              'torch': torch.__version__, 'gpu_name': torch.cuda.get_device_name(device),
              'parameters': sum(t.numel() for t in model.parameters())}
    start_epoch, best_iou, updates = 1, -1., 0
    if a.resume:
        previous_config = json.loads((out/'config.json').read_text())
        for k in ['variant', 'epochs', 'batch', 'accum', 'lr', 'new_lr', 'seed', 'model_kwargs',
                  'initial_checkpoint_sha256', 'data_manifest_sha256', 'source_hashes']:
            if config[k] != previous_config[k]:
                raise RuntimeError(f'Resume configuration changed: {k}')
        ck = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        start_epoch, best_iou, updates = ck['epoch']+1, ck['best_iou'], ck['updates']
        del ck
    else:
        (out/'config.json').write_text(json.dumps(config, indent=2))
        if not a.smoke:
            initial_val = evaluate(model, val, device, amp=True)
            best_iou = initial_val['iou']
            atomic_save({'model': model.state_dict(), 'epoch': 0, 'val': initial_val,
                         'config': config}, out/'best.pt')
            (out/'initial_val.json').write_text(json.dumps(initial_val, indent=2))
            print(f'INITIAL {a.variant}: val_iou={best_iou:.6f}', flush=True)
    print(f'CONFIG {json.dumps(config)}', flush=True)
    fields = ['epoch', 'loss', 'seg', 'profile', 'trace', 'valid_profiles', 'valid_windows', 'aux_ramp',
              'val_iou', 'val_dice', 'val_precision', 'val_recall', 'old_lr', 'seconds', 'peak_memory_mib']
    log = open(out/'log.csv', 'a' if a.resume else 'w', newline='')
    writer = csv.DictWriter(log, fieldnames=fields)
    if not a.resume:
        writer.writeheader()
    for epoch in range(start_epoch, a.epochs+1):
        t0 = time.time()
        dataset.epoch = epoch
        order = np.random.default_rng(a.seed+epoch).permutation(len(dataset)).tolist()
        if a.smoke:
            order = order[:a.batch*a.accum*2]
        loader = DataLoader(Subset(dataset, order), batch_size=a.batch, num_workers=a.workers,
                            pin_memory=True, shuffle=False)
        if len(loader) % a.accum:
            raise ValueError('Pilot requires complete accumulation groups')
        model.train(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        ramp = min(max((epoch-2)/4, 0), 1) if not a.smoke else 1.
        factor = min(epoch/2, 1)*(.01+.99*.5*(1+math.cos(math.pi*(epoch-1)/max(a.epochs-1, 1))))
        for group in optimizer.param_groups:
            group['lr'] = group['peak_lr']*factor
        sums = np.zeros(6, dtype=np.float64)
        for step, (x, y, _, q, normals, valid, windows) in enumerate(loader):
            # Same stochastic backbone seed and augmented inputs across all variants.
            torch.manual_seed(a.seed + epoch*100003 + step)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x)
            with torch.autocast('cuda', enabled=False):
                seg = segmentation_loss(logits, y)
                pr, tr = logits.float().sum()*0, logits.float().sum()*0
                if use_loss:
                    prob = logits.float().sigmoid()
                    pr = profile_loss(prob, y, q.to(device), normals.to(device), valid.to(device))
                    tr = trace_loss(prob, y, windows)
                loss = seg + ramp*(.1*pr+.05*tr)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}')
            scaler.scale(loss/a.accum).backward()
            if (step+1) % a.accum == 0:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient; refusing silently skipped AMP updates')
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                updates += 1
            sums += [loss.item(), seg.item(), pr.item(), tr.item(), valid.sum().item()/len(x),
                     (windows[:, :, 0] >= 0).sum().item()/len(x)]
            if step == 0 or (step+1) % 100 == 0:
                print(f'{a.variant} epoch={epoch} batch={step+1}/{len(loader)} loss={loss.item():.5f} '
                      f'seg={seg.item():.5f} profile={pr.item():.5f} trace={tr.item():.5f}', flush=True)
        peak = torch.cuda.max_memory_allocated(device)/2**20
        if a.smoke:
            report = {'variant': a.variant, 'updates': updates, 'losses': (sums/len(loader)).tolist(),
                      'peak_memory_mib': peak, 'seconds': time.time()-t0}
            (out/'smoke.json').write_text(json.dumps(report, indent=2))
            print(f'SMOKE PASS {json.dumps(report)}', flush=True)
            log.close()
            return
        result = evaluate(model, val, device, amp=True)
        row = dict(zip(fields[:7], [epoch, *(sums/len(loader))]))
        row.update(aux_ramp=ramp, val_iou=result['iou'], val_dice=result['dice'],
                   val_precision=result['precision'], val_recall=result['recall'],
                   old_lr=optimizer.param_groups[0]['lr'], seconds=time.time()-t0, peak_memory_mib=peak)
        writer.writerow(row); log.flush()
        if result['iou'] > best_iou:
            best_iou = result['iou']
            atomic_save({'model': model.state_dict(), 'epoch': epoch, 'val': result,
                         'config': config}, out/'best.pt')
        atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(), 'epoch': epoch, 'updates': updates,
                     'best_iou': best_iou, 'config': config}, out/'last.pt')
        print(f'EPOCH {a.variant} {json.dumps(row)} best={best_iou:.6f}', flush=True)
        (out/'status.json').write_text(json.dumps({'status': 'training', 'epoch': epoch,
                                                   'best_val_iou': best_iou, 'updates': updates}, indent=2))
    log.close()
    ck = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    test_loader = DataLoader(FaultVolumes(a.data, 'test'), batch_size=1, num_workers=a.workers, pin_memory=True)
    result = evaluate(model, test_loader, device, amp=True)
    result.update(epoch=ck['epoch'], variant=a.variant, n_test=len(test_loader.dataset),
                  initial_checkpoint_sha256=config['initial_checkpoint_sha256'],
                  reference_historical_test_iou=.8325868632244817,
                  difference_from_historical_iou=result['iou']-.8325868632244817,
                  checkpoint_sha256=sha(out/'best.pt'), selection=config['selection'])
    (out/'test.json').write_text(json.dumps(result, indent=2))
    (out/'status.json').write_text(json.dumps({'status': 'complete', 'test': result}, indent=2))
    print(f'COMPLETE {a.variant} {json.dumps(result)}', flush=True)


if __name__ == '__main__':
    main()
