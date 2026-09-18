"""Sealed-test evaluation under the thebe_spatial_v3 protocol (same rules as the trainer's validation).

    python train/eval_thebe_spatial.py --runs runs/thebe_spatial_v3_unet_l runs/thebe_spatial_v3_maxvit_tiny --gpu 0

Test region = sections [1100, 1803) (703 sections), 128^3 inputs with 32-voxel halo, only the central
64^3 core is scored, cores tile the region with stride 64 (each valid voxel scored exactly once),
signal-support mask, threshold 0.5.  Reports whole-region IoU / Dice / P / R exactly like the trainer,
plus AP (histogram PR over the masked voxels).  Uses <run>/best_fullval.pt (checkpoint selected on the
validation region) unless --last.  Requires data/thebe_spatial_v3 prepared with --with-test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build                                  # noqa: E402
from train.dataset_thebe_spatial import SpatialThebe, DEFAULT   # noqa: E402
from train.train_old_recipe_ddp import VARIANTS            # noqa: E402
from train.train_thebe_spatial_ddp import loss_fn          # noqa: E402

NB = 2000


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = n = pos = 0.0; loss = 0.0; count = 0
    hpos = np.zeros(NB); hneg = np.zeros(NB)
    t0 = time.time()
    for i, (x, y, m, _) in enumerate(loader):
        x, y, m = [a.to(device, non_blocking=True) for a in (x, y, m)]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        p = logits.float().sigmoid(); pr = p > .5; gt = y > .5; v = m > .5
        tp += (pr & gt & v).sum().item(); fp += (pr & ~gt & v).sum().item(); fn += (~pr & gt & v).sum().item()
        n += v.sum().item(); pos += (gt & v).sum().item()
        loss += loss_fn(logits, y, m).item(); count += 1
        pb = (p * (NB - 1)).round().long().clamp_(0, NB - 1)
        hpos += torch.bincount(pb[gt & v], minlength=NB).cpu().numpy(); hneg += torch.bincount(pb[~gt & v], minlength=NB).cpu().numpy()
        if i % 500 == 0:
            print(f"    {i}/{len(loader)} cores, {time.time()-t0:.0f}s", flush=True)
    cp = np.cumsum(hpos[::-1]); cn = np.cumsum(hneg[::-1])
    P = cp / np.maximum(cp + cn, 1); R = cp / max(hpos.sum(), 1); Rp = np.concatenate([[0.], R[:-1]])
    return dict(iou=tp / max(tp + fp + fn, 1), dice=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), ap=float(np.sum((R - Rp) * P)), valid_voxels=int(n), positive_voxels=int(pos),
                fault_fraction=pos / max(n, 1), loss=loss / max(count, 1), cores=count, seconds=int(time.time() - t0))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+", required=True); a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--split", default="test"); a.add_argument("--last", action="store_true", help="use last.pt instead of best_fullval.pt")
    a.add_argument("--workers", type=int, default=4)
    a = a.parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    ds = SpatialThebe(a.split, DEFAULT)
    print(f"{a.split}: {len(ds)} cores, support voxels {ds.record['support_voxels']:,}", flush=True)
    dl = DataLoader(ds, batch_size=1, num_workers=a.workers, pin_memory=True)
    rows = {}
    for r in a.runs:
        r = Path(r); cfg = json.loads((r / "config.json").read_text())
        name, mkw, _, _ = VARIANTS[cfg["args"].get("variant", "unet_l")]
        ck = torch.load(r / ("last.pt" if a.last else "best_fullval.pt"), map_location="cpu", weights_only=False)
        model = build(name, **mkw).to(device); model.load_state_dict(ck["model"])
        res = evaluate(model, dl, device); res.update(epoch=ck["epoch"], checkpoint="last" if a.last else "best_fullval")
        assert res["valid_voxels"] == ds.record["support_voxels"], (res["valid_voxels"], ds.record["support_voxels"])
        rows[r.name] = res
        (r / f"{a.split}_spatial_v3.json").write_text(json.dumps(res, indent=1))
        print(f"{r.name:32s} ep {res['epoch']:3d}  IoU {res['iou']:.4f}  Dice {res['dice']:.4f}  AP {res['ap']:.4f}  "
              f"P {res['precision']:.4f}  R {res['recall']:.4f}  ({res['seconds']}s)", flush=True)
        del model; torch.cuda.empty_cache()
    print(f'\n{"run":32s} {"ep":>3s} {"IoU":>7s} {"Dice":>7s} {"AP":>7s} {"P":>7s} {"R":>7s}')
    for k, v in rows.items():
        print(f'{k:32s} {v["epoch"]:3d} {v["iou"]:7.4f} {v["dice"]:7.4f} {v["ap"]:7.4f} {v["precision"]:7.4f} {v["recall"]:7.4f}')


if __name__ == "__main__":
    main()
