"""Per-volume zero-shot scores on Wu et al. 2019's 20 released validation volumes + paired bootstrap over volumes.

    python train/eval_wu_pervolume.py --gpu 2 --runs runs/maxvit3d_par_unet_lg_b30 runs/maxvit3d_par_unet_b30 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.analyze_loss_vs_iou import load        # noqa: E402
from train.eval_cross import FaultSeg3DVal        # noqa: E402


@torch.no_grad()
def per_volume(model, loader, device):
    rows = []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x).float())
        pr = p > .5; gt = y > .5
        tp = (pr & gt).sum().item(); fp = (pr & ~gt).sum().item(); fn = (~pr & gt).sum().item()
        yd = F.max_pool3d(y, 5, 1, 2) > .5; pd = F.max_pool3d(pr.float(), 5, 1, 2) > .5
        rows.append(dict(iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                         tol2_recall=(pd & gt).sum().item() / max(gt.sum().item(), 1), tol2_precision=(pr & yd).sum().item() / max(pr.sum().item(), 1)))
    return rows


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+", required=True); a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--out", default="logs/eval_wu_pervolume.json"); a.add_argument("--ref", default=None, help="run name to pair against (default: first)")
    a = a.parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    dl = DataLoader(FaultSeg3DVal(), batch_size=1, num_workers=2)
    res = {}
    for r in a.runs:
        m, ep = load(r, device); res[Path(r).name] = dict(epoch=ep, volumes=per_volume(m, dl, device)); del m; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1))
    names = list(res); ref = a.ref or names[0]; rng = np.random.default_rng(0)
    print(f'{"run":30s} {"ep":>3s} {"meanIoU":>8s} {"±sd":>6s} {"medIoU":>7s} {"R":>6s} {"t2R":>6s} | vs {ref}: {"dIoU":>6s} {"95% CI":>16s} {"wins":>5s}')
    for k in names:
        v = np.array([r["iou"] for r in res[k]["volumes"]]); R = np.mean([r["recall"] for r in res[k]["volumes"]]); t2 = np.mean([r["tol2_recall"] for r in res[k]["volumes"]])
        d = v - np.array([r["iou"] for r in res[ref]["volumes"]])
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f'{k:30s} {res[k]["epoch"]:3d} {v.mean():8.3f} {v.std(ddof=1):6.3f} {np.median(v):7.3f} {R:6.3f} {t2:6.3f} | {d.mean():+6.3f} [{lo:+.3f},{hi:+.3f}] {int((d>0).sum()):3d}/{len(d)}')


if __name__ == "__main__":
    main()
