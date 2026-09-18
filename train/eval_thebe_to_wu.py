"""Reverse zero-shot: Thebe-trained (spatial v3) checkpoints on Wu et al. 2019's 20 released synthetic volumes.

    python train/eval_thebe_to_wu.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny --device cpu

Same preprocessing as the models saw in training (depth-first, per-volume z-score).  Pooled IoU / Dice / AP / P / R,
2-voxel-tolerance P / R, per-volume IoU (mean +- sd) and a paired bootstrap between consecutive runs.
--device cpu lets it run while the GPUs are busy (20 volumes, a few minutes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build                                  # noqa: E402
from train.eval_cross import FaultSeg3DVal                # noqa: E402
from train.train_old_recipe_ddp import VARIANTS            # noqa: E402

NB = 2000


def load_v3(run, device, last=False):
    run = Path(run); cfg = json.loads((run / "config.json").read_text())
    name, mkw, _, _ = VARIANTS[cfg["args"].get("variant", "unet_l")]
    ck = torch.load(run / ("last.pt" if last else "best_fullval.pt"), map_location="cpu", weights_only=False)
    m = build(name, **mkw).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m, ck["epoch"], cfg["args"].get("variant", "unet_l")


@torch.no_grad()
def evaluate(model, loader, device):
    tp = fp = fn = 0.0; ttp = tfp = tfn = mgt = 0.0; hpos = np.zeros(NB); hneg = np.zeros(NB); per = []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model(x).float())
        else:
            p = torch.sigmoid(model(x).float())
        pr = p > .5; gt = y > .5
        a, b, c = (pr & gt).sum().item(), (pr & ~gt).sum().item(), (~pr & gt).sum().item()
        tp += a; fp += b; fn += c; per.append(a / max(a + b + c, 1))
        pb = (p * (NB - 1)).round().long().clamp_(0, NB - 1).flatten(); mk = gt.flatten()
        hpos += torch.bincount(pb[mk], minlength=NB).numpy(); hneg += torch.bincount(pb[~mk], minlength=NB).numpy()
        yd = F.max_pool3d(y, 5, 1, 2) > .5; pd = F.max_pool3d(pr.float(), 5, 1, 2) > .5
        ttp += (pr & yd).sum().item(); tfp += (pr & ~yd).sum().item(); mgt += (pd & gt).sum().item(); tfn += (~pd & gt).sum().item()
    cp = np.cumsum(hpos[::-1]); cn = np.cumsum(hneg[::-1])
    P = cp / np.maximum(cp + cn, 1); R = cp / max(hpos.sum(), 1); Rp = np.concatenate([[0.], R[:-1]])
    return dict(iou=tp / max(tp + fp + fn, 1), dice=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                ap=float(np.sum((R - Rp) * P)), tol2_precision=ttp / max(ttp + tfp, 1), tol2_recall=mgt / max(mgt + tfn, 1),
                pred_fraction=(tp + fp) / (len(per) * 128 ** 3), per_volume_iou=per)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+", required=True); a.add_argument("--device", default="cpu"); a.add_argument("--threads", type=int, default=16)
    a.add_argument("--last", action="store_true"); a.add_argument("--out", default="logs/eval_thebe_to_wu.json")
    a = a.parse_args()
    torch.set_num_threads(a.threads); device = torch.device(a.device)
    dl = DataLoader(FaultSeg3DVal(), batch_size=1, num_workers=2)
    out = Path(a.out); rows = json.loads(out.read_text()) if out.exists() else {}
    for r in a.runs:
        t0 = time.time(); m, ep, var = load_v3(r, device, a.last)
        res = evaluate(m, dl, device); res.update(epoch=ep, variant=var, seconds=int(time.time() - t0)); rows[Path(r).name] = res
        out.write_text(json.dumps(rows, indent=1))
        v = np.array(res["per_volume_iou"])
        print(f'{Path(r).name:34s} ep {ep:3d}  IoU {res["iou"]:.4f}  Dice {res["dice"]:.4f}  AP {res["ap"]:.4f}  P {res["precision"]:.4f}  R {res["recall"]:.4f}  '
              f'tol2 P/R {res["tol2_precision"]:.3f}/{res["tol2_recall"]:.3f}  pred-frac {res["pred_fraction"]:.3f}  per-vol IoU {v.mean():.3f}±{v.std(ddof=1):.3f}  ({res["seconds"]}s)', flush=True)
    names = [Path(r).name for r in a.runs]
    if len(names) >= 2:
        rng = np.random.default_rng(0)
        for i in range(1, len(names)):
            d = np.array(rows[names[i]]["per_volume_iou"]) - np.array(rows[names[0]]["per_volume_iou"])
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]; lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"paired {names[i]} - {names[0]}: dIoU {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] wins {(d > 0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
