"""Exact replication of the archived stage-1 training recipe, on dataset_v2.

Source: _archive_20260908_faultsyn_stage1/train_3dunet_resplit_batch16.py and
runs/*_faultvitnet_fullvol_200ep/launch_manual_ddp.sh.  Under that recipe, on
the OLD dataset, the ranking was MaxViT 0.611 > ResNet 0.578 > UNet 0.539
(test IoU); under our OneCycle/GDice recipe on v2 it is the reverse.  This run
separates the two variables: same data (v2), old recipe.

Replicated verbatim:
  loss          0.6 * soft Dice + 0.4 * focal (alpha=0.75, gamma=2)
  augmentation  random 90-deg rotation in a random plane, flips on all axes,
                Gaussian blur p=0.35 sigma 0.4-1.0, intensity scale 0.8-1.2 and
                shift +-0.15, THEN per-volume z-score
  optimiser     AdamW, PyTorch default weight_decay=0.01
  schedule      per-EPOCH linear warmup 1e-6 -> 1e-4 over 10 epochs, then cosine
                to 1e-7 over the 200-epoch budget
  batch         global 8.  The old run used 4-GPU DDP with 2 per GPU; here one
                GPU takes 2 per step and accumulates 4 steps, which is the same
                mean-of-8 gradient (InstanceNorm has no cross-sample statistics).
  selection     best = lowest val LOSS; early stop after 20 stale epochs
  seed          2026; epoch order = default_rng(seed + epoch).permutation
  AMP           fp16 autocast + GradScaler

    python train/train_old_recipe.py --model unet3d   --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build                              # noqa: E402
from train.dataset import FaultVolumes, CATEGORIES    # noqa: E402
from train.train import evaluate as evaluate_iou      # noqa: E402


class OldAugVolumes(Dataset):
    """v2 volumes with the archived project's augmentation, verbatim."""
    def __init__(self, base: FaultVolumes, train: bool):
        self.base, self.train = base, train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        name, cat = self.base.items[i]
        x = self.base._read("seismic", name, np.float32).astype(np.float32)
        y = (self.base._read(self.base.label_dir, name, np.uint8) > 0).astype(np.float32)
        if self.train:
            axes = random.choice(((0, 1), (0, 2), (1, 2))); k = random.choice((0, 1, 2, 3))
            x = np.rot90(x, k, axes).copy(); y = np.rot90(y, k, axes).copy()
            for axis in range(3):
                if random.random() < .5:
                    x = np.flip(x, axis).copy(); y = np.flip(y, axis).copy()
            if random.random() < .35:
                x = gaussian_filter(x, sigma=random.uniform(.4, 1.0)).astype(np.float32)
            x = x * random.uniform(.8, 1.2) + random.uniform(-.15, .15)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None]), CATEGORIES.index(cat)


def dice_focal_loss(logits, target, alpha=.75, gamma=2.0, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum((1, 2, 3, 4)); denom = probs.sum((1, 2, 3, 4)) + target.sum((1, 2, 3, 4))
    dice = (1 - (2 * inter + eps) / (denom + eps)).mean()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.where(target > 0.5, probs, 1 - probs)
    at = torch.where(target > 0.5, torch.as_tensor(alpha, device=target.device),
                     torch.as_tensor(1 - alpha, device=target.device))
    focal = (at * (1 - pt).pow(gamma) * bce).mean()
    return .6 * dice + .4 * focal


def focal_term(logits, target, alpha=.75, gamma=2.0):
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.where(target > 0.5, probs, 1 - probs)
    at = torch.where(target > 0.5, torch.as_tensor(alpha, device=target.device),
                     torch.as_tensor(1 - alpha, device=target.device))
    return (at * (1 - pt).pow(gamma) * bce).mean()


def dicesq_focal_loss(logits, target, alpha=.75, gamma=2.0, eps=1e-6):
    """0.6 * V-Net Dice (squared denominator, Milletari 2016) + 0.4 * focal.

    Same weights / focal as dice_focal_loss; the only change is sum(p) -> sum(p^2) in the
    Dice denominator.  With 2M background voxels at p ~ 3e-3 the linear Dice carries a
    "floor" term worth ~10% of the fault volume (logs/dice_decomposition.json); squaring
    makes it ~0.03%, so the loss can no longer be lowered by pushing the background floor.
    """
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum((1, 2, 3, 4))
    denom = probs.pow(2).sum((1, 2, 3, 4)) + target.sum((1, 2, 3, 4))      # target is binary: y^2 == y
    dice = (1 - (2 * inter + eps) / (denom + eps)).mean()
    return .6 * dice + .4 * focal_term(logits, target, alpha, gamma)


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jac = 1. - inter / union
    jac[1:] = jac[1:] - jac[:-1]
    return jac


def lovasz_hinge(logits, target):
    """Lovasz hinge (Berman et al. 2018): convex surrogate of 1 - IoU, per volume, averaged."""
    losses = []
    for lg, y in zip(logits, target):
        lg = lg.flatten(); y = y.flatten().float()
        signs = 2. * y - 1.
        errors = 1. - lg * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(F.relu(errors_sorted), _lovasz_grad(y[perm])))
    return torch.stack(losses).mean()


def bce_lovasz_loss(logits, target, alpha=.75):
    """Class-weighted BCE (alpha on faults) + Lovasz hinge.  Voxels beyond the margin
    (background with logit < -1, faults with logit > 1) contribute nothing to the hinge."""
    w = torch.where(target > 0.5, torch.as_tensor(alpha, device=target.device),
                    torch.as_tensor(1 - alpha, device=target.device))
    bce = (w * F.binary_cross_entropy_with_logits(logits, target, reduction='none')).mean() / alpha
    return bce + lovasz_hinge(logits, target)


LOSSES = {"dice_focal": dice_focal_loss, "dicesq_focal": dicesq_focal_loss, "bce_lovasz": bce_lovasz_loss}


def warmup_cosine_lr(epoch, total_epochs, warmup_epochs, warmup_start_lr, peak_lr, floor_lr):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return warmup_start_lr + (peak_lr - warmup_start_lr) * (epoch / max(1, warmup_epochs))
    t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return floor_lr + .5 * (peak_lr - floor_lr) * (1 + np.cos(np.pi * min(1.0, t)))


@torch.no_grad()
def val_loss_old(model, loader, device):
    model.eval(); s, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            s += dice_focal_loss(model(x), y).item() * x.shape[0]; n += x.shape[0]
    return s / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['unet3d', 'resnet3d', 'maxvit3d'])
    p.add_argument('--data', default='data/dataset_reproduction_v2')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--tag', default='v2_oldrecipe')
    p.add_argument('--epochs', type=int, default=200); p.add_argument('--patience', type=int, default=20)
    p.add_argument('--global-batch', type=int, default=8); p.add_argument('--per-step', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--warmup-epochs', type=int, default=10)
    p.add_argument('--warmup-start-lr', type=float, default=1e-6); p.add_argument('--min-lr', type=float, default=1e-7)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--mkw', nargs='*', default=[])
    p.add_argument('--resume-from', default=None,
                   help='continue from another run\'s last.pt: a SECOND cosine cycle over --epochs '
                        'starting at --lr, early stop on val loss as usual (diagnostic: does it keep improving?)')
    a = p.parse_args()
    assert a.global_batch % a.per_step == 0
    accum = a.global_batch // a.per_step

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{a.gpu}')
    out = Path(f'runs/{a.model}_{a.tag}'); out.mkdir(parents=True, exist_ok=True)

    tr_base = FaultVolumes(a.data, 'train'); va_base = FaultVolumes(a.data, 'val')
    tr = OldAugVolumes(tr_base, train=True); va = OldAugVolumes(va_base, train=False)
    assert len(tr) % a.global_batch == 0
    (out / 'args.json').write_text(json.dumps({**vars(a), 'recipe': 'archived stage-1 (dice_focal, warmup-cosine, old aug)',
                                               'data_source_sha256': tr_base.source_sha256,
                                               'n_train': len(tr), 'n_val': len(va)}, indent=2))
    vl = DataLoader(va, batch_size=1, num_workers=2, pin_memory=True)

    def _parse(v):
        for cast in (int, float):
            try: return cast(v)
            except ValueError: pass
        return {'true': True, 'false': False}.get(v.lower(), v)
    mkw = {k: _parse(v) for k, v in (s.split('=', 1) for s in a.mkw)}
    if 'full_res_skip' in mkw: mkw['full_res_skip'] = bool(mkw['full_res_skip'])
    model = build(a.model, **mkw).to(device)
    start_epoch, best_init = 1, float('inf')
    if a.resume_from:
        ck = torch.load(a.resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); start_epoch = 1
        bp = Path(a.resume_from).parent / 'best.pt'
        if bp.exists():
            best_init = float(torch.load(bp, map_location='cpu', weights_only=False).get('val_loss', float('inf')))
        a.warmup_epochs = 0                                   # second cycle: no warmup
        print(f'resumed weights from {a.resume_from} (epoch {ck.get("epoch")}); '
              f'best val loss to beat = {best_init:.5f}; new cosine cycle {a.lr:.0e} -> {a.min_lr:.0e} over {a.epochs} epochs', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.warmup_start_lr)      # default wd = 0.01
    scaler = torch.amp.GradScaler('cuda')
    print(f'model={a.model} params={sum(q.numel() for q in model.parameters())/1e6:.2f}M  train={len(tr)} val={len(va)} '
          f'global batch={a.global_batch} ({a.per_step} x {accum} accum)  lr warmup {a.warmup_start_lr:.0e}->{a.lr:.0e} '
          f'over {a.warmup_epochs} then cosine->{a.min_lr:.0e}  epochs={a.epochs} patience={a.patience}', flush=True)

    hist = open(out / 'log.csv', 'w', newline=''); wr = csv.writer(hist)
    wr.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_dice', 'val_precision', 'val_recall', 'lr', 'seconds']
                + [f'val_iou_{c}' for c in CATEGORIES])
    best, stale = best_init, 0
    for epoch in range(1, a.epochs + 1):
        lr_now = warmup_cosine_lr(epoch, a.epochs, a.warmup_epochs, a.warmup_start_lr, a.lr, a.min_lr)
        for g in opt.param_groups: g['lr'] = lr_now
        order = np.random.default_rng(a.seed + epoch).permutation(len(tr)).tolist()
        model.train(); t0 = time.time(); tot, cnt = 0.0, 0
        # loader over the seeded permutation, `per_step` volumes per forward
        dl = DataLoader(torch.utils.data.Subset(tr, order), batch_size=a.per_step, shuffle=False,
                        num_workers=a.workers, pin_memory=True, drop_last=False)
        opt.zero_grad(set_to_none=True)
        for j, (x, y, _) in enumerate(dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = dice_focal_loss(model(x), y)
            scaler.scale(loss * (x.shape[0] / a.global_batch)).backward()     # exact mean over the group of 8
            tot += loss.item() * x.shape[0]; cnt += x.shape[0]
            if (j + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        vloss = val_loss_old(model, vl, device)
        v = evaluate_iou(model, vl, device, amp=True)
        dt = time.time() - t0
        wr.writerow([epoch, f'{tot/cnt:.5f}', f'{vloss:.5f}', f'{v["iou"]:.4f}', f'{v["dice"]:.4f}',
                     f'{v["precision"]:.4f}', f'{v["recall"]:.4f}', f'{lr_now:.2e}', f'{dt:.0f}']
                    + [f'{v[f"iou_{c}"]:.4f}' for c in CATEGORIES]); hist.flush()
        print(f'epoch={epoch:03d} train={tot/cnt:.5f} val={vloss:.5f} IoU={v["iou"]:.4f} Dice={v["dice"]:.4f} '
              f'P={v["precision"]:.3f} R={v["recall"]:.3f} lr={lr_now:.2e} {dt:.0f}s', flush=True)
        if vloss < best:
            best, stale = vloss, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': best, 'val': v,
                        'args': {**vars(a), 'model': a.model}}, out / 'best.pt')
        else:
            stale += 1
            if stale >= a.patience:
                print(f'early stop at epoch {epoch}; best val loss {best:.5f}', flush=True); break
    torch.save({'model': model.state_dict(), 'epoch': epoch}, out / 'last.pt'); hist.close()


if __name__ == '__main__':
    main()
