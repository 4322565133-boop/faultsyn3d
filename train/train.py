"""Train one of the three baselines on dataset_v1.

    python train/train.py --model unet3d   --gpu 0
    python train/train.py --model resnet3d --gpu 1
    python train/train.py --model maxvit3d --gpu 2

One GPU per run so the three baselines train side by side.  Recipe is held
identical across models -- same loss, optimiser, schedule, augmentation, seed --
so the comparison is about architecture and nothing else.  The loss is the
archived project's GDice + BCE; the loss-function work comes later as its own
variable, on top of whichever backbone wins here.
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
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build                         # noqa: E402
from train.dataset import FaultVolumes, CATEGORIES   # noqa: E402


def gdice_bce(logits, target, eps=1e-6):
    """Generalised Dice + pos-weighted BCE, as in the archived stage-1 recipe."""
    probs = torch.sigmoid(logits)
    dims = (2, 3, 4)
    counts = torch.cat((target.sum(dims), (1 - target).sum(dims)), dim=1)
    w = 1.0 / (counts.square() + eps)
    pred = torch.cat((probs, 1 - probs), dim=1)
    truth = torch.cat((target, 1 - target), dim=1)
    inter = (pred * truth).sum(dims)
    denom = (pred + truth).sum(dims)
    gdice = 1 - (2 * (w * inter).sum(1) + eps) / ((w * denom).sum(1) + eps)
    pos = target.sum().clamp_min(1)
    pos_weight = ((target.numel() - target.sum()) / pos).clamp(1, 100).detach()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return gdice.mean() + bce


@torch.no_grad()
def evaluate(model, loader, device, amp, thr=0.5, categories=CATEGORIES):
    """Loss plus voxel IoU / Dice / precision / recall, overall and per category."""
    model.eval()
    loss_sum, n = 0.0, 0
    tp = np.zeros(len(categories)); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    # score histograms for a threshold-free AP: 2000 bins over [0,1], per category
    NB = 2000
    hpos = np.zeros((len(categories), NB)); hneg = np.zeros_like(hpos)
    for x, y, c in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = gdice_bce(out.float(), y)
        loss_sum += loss.item() * x.shape[0]; n += x.shape[0]
        prob = torch.sigmoid(out.float())
        pred = prob > thr
        yb = y > 0.5
        for b in range(x.shape[0]):
            k = int(c[b])
            tp[k] += (pred[b] & yb[b]).sum().item()
            fp[k] += (pred[b] & ~yb[b]).sum().item()
            fn[k] += (~pred[b] & yb[b]).sum().item()
            pb = (prob[b] * (NB - 1)).round().long().clamp_(0, NB - 1).flatten()
            m = yb[b].flatten()
            hpos[k] += torch.bincount(pb[m], minlength=NB).cpu().numpy()
            hneg[k] += torch.bincount(pb[~m], minlength=NB).cpu().numpy()
    def stats(tp, fp, fn):
        iou = tp / max(tp + fp + fn, 1); dice = 2 * tp / max(2 * tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        return dict(iou=iou, dice=dice, precision=prec, recall=rec)
    def ap(hp, hn):
        """Average precision from score histograms (step-wise, as sklearn does)."""
        # sweep threshold from high to low: cumulative counts of positives / negatives above it
        cp = np.cumsum(hp[::-1]); cn = np.cumsum(hn[::-1])
        P = cp / np.maximum(cp + cn, 1); R = cp / max(hp.sum(), 1)
        Rprev = np.concatenate([[0.0], R[:-1]])
        return float(np.sum((R - Rprev) * P))
    res = {"loss": loss_sum / max(n, 1), **stats(tp.sum(), fp.sum(), fn.sum()),
           "ap": ap(hpos.sum(0), hneg.sum(0))}
    for k, cat in enumerate(categories):
        res[f"iou_{cat}"] = stats(tp[k], fp[k], fn[k])["iou"]
        res[f"ap_{cat}"] = ap(hpos[k], hneg[k])
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["unet3d", "resnet3d", "maxvit3d"])
    p.add_argument("--data", default="data/dataset_reproduction_v2")
    p.add_argument("--out", default=None, help="default runs/<model>_<tag>")
    p.add_argument("--tag", default="v2")
    p.add_argument("--limit", type=int, default=None, help="use only the first N train volumes (overfit check)")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=4, help="grad accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--mkw", nargs="*", default=[],
                   help="model kwargs, key=value (e.g. full_res_skip=1 drop_path_rate=0.2 img_size=128)")
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{a.gpu}")
    amp = not a.no_amp
    out = Path(a.out or f"runs/{a.model}_{a.tag}"); out.mkdir(parents=True, exist_ok=True)

    tr = FaultVolumes(a.data, "train", augment=not a.no_aug, limit=a.limit)
    va = FaultVolumes(a.data, "val")
    # record which data this run saw, so a run can never be mistaken for another dataset's
    (out / "args.json").write_text(json.dumps({**vars(a), "data_source_sha256": tr.source_sha256,
                                               "n_train": len(tr), "n_val": len(va)}, indent=2))
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=min(a.workers, len(tr)),
                    pin_memory=True, drop_last=len(tr) >= a.batch * 4,
                    persistent_workers=min(a.workers, len(tr)) > 0)
    vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    print(f"train {len(tr)}  val {len(va)}  model {a.model}  gpu {a.gpu}", flush=True)

    def _parse(v):
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return {"true": True, "false": False}.get(v.lower(), v)
    mkw = {k: _parse(v) for k, v in (s.split("=", 1) for s in a.mkw)}
    if "full_res_skip" in mkw:
        mkw["full_res_skip"] = bool(mkw["full_res_skip"])
    model = build(a.model, **mkw).to(device)
    n_par = sum(t.numel() for t in model.parameters())
    print(f"params {n_par/1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = max(len(tl) // a.accum, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice",
                 "val_precision", "val_recall", "lr", "seconds"] +
                [f"val_iou_{c}" for c in CATEGORIES])
    best, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot, cnt = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = gdice_bce(model(x).float(), y) / a.accum
            scaler.scale(loss).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
            tot += loss.item() * a.accum; cnt += 1
        v = evaluate(model, vl, device, amp)
        dt = time.time() - t0
        wr.writerow([ep, f"{tot/max(cnt,1):.4f}", f"{v['loss']:.4f}", f"{v['iou']:.4f}",
                     f"{v['dice']:.4f}", f"{v['precision']:.4f}", f"{v['recall']:.4f}",
                     f"{opt.param_groups[0]['lr']:.2e}", f"{dt:.0f}"] +
                    [f"{v[f'iou_{c}']:.4f}" for c in CATEGORIES])
        log.flush()
        print(f"ep {ep:3d}  train {tot/max(cnt,1):.4f}  val {v['loss']:.4f}  "
              f"IoU {v['iou']:.4f}  Dice {v['dice']:.4f}  P {v['precision']:.3f} "
              f"R {v['recall']:.3f}  {dt:.0f}s", flush=True)
        if v["iou"] > best:
            best, bad = v["iou"], 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val": v,
                        "args": vars(a)}, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop at epoch {ep}; best val IoU {best:.4f}", flush=True)
                break
    torch.save({"model": model.state_dict(), "epoch": ep}, out / "last.pt")
    log.close()


if __name__ == "__main__":
    main()
