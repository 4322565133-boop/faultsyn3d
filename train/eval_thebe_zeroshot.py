"""Zero-shot evaluation of synthetic-trained (old-recipe) checkpoints on the Thebe spatial-v3 regions.

    python train/eval_thebe_zeroshot.py --runs runs/maxvit3d_par_unet_lg_b30 ... --split val --gpu 2

No fine-tuning: the cubes come from the same SpatialThebe loader as the Thebe-trained models (depth-first,
per-cube standardisation over supported voxels, 128^3 with 32-voxel halo, only the central 64^3 core scored,
signal-support mask).  Reports voxel IoU / Dice / P / R / AP and tolerance metrics (predictions and labels
dilated by k voxels, k = 2 and 5, because the Thebe labels are ~10-voxel bands while synthetic-trained models
predict thin faults).  Writes one JSON per split to logs/ (or --out).
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
from train.analyze_loss_vs_iou import load                       # noqa: E402  old-recipe best.pt loader
from train.dataset_thebe_spatial import SpatialThebe, DEFAULT     # noqa: E402

NB = 2000
TOLS = (2, 5)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = n = pos = 0.0
    tol = {k: dict(ttp=0.0, tfp=0.0, mgt=0.0, tfn=0.0) for k in TOLS}
    hpos = np.zeros(NB); hneg = np.zeros(NB); t0 = time.time()
    for i, (x, y, m, _) in enumerate(loader):
        x, y, m = [a.to(device, non_blocking=True) for a in (x, y, m)]
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x).float().sigmoid()
        pr = p > .5; gt = y > .5; v = m > .5
        tp += (pr & gt & v).sum().item(); fp += (pr & ~gt & v).sum().item(); fn += (~pr & gt & v).sum().item()
        n += v.sum().item(); pos += (gt & v).sum().item()
        pb = (p * (NB - 1)).round().long().clamp_(0, NB - 1)
        hpos += torch.bincount(pb[gt & v], minlength=NB).cpu().numpy(); hneg += torch.bincount(pb[~gt & v], minlength=NB).cpu().numpy()
        for k in TOLS:                                            # dilate on the full 128^3 (halo available), score in the core
            yd = F.max_pool3d(y, 2 * k + 1, 1, k) > .5; pd = F.max_pool3d(pr.float(), 2 * k + 1, 1, k) > .5
            t = tol[k]
            t["ttp"] += (pr & yd & v).sum().item(); t["tfp"] += (pr & ~yd & v).sum().item()
            t["mgt"] += (pd & gt & v).sum().item(); t["tfn"] += (~pd & gt & v).sum().item()
        if i % 400 == 0:
            print(f"    {i}/{len(loader)} batches, {time.time()-t0:.0f}s", flush=True)
    cp = np.cumsum(hpos[::-1]); cn = np.cumsum(hneg[::-1])
    P = cp / np.maximum(cp + cn, 1); R = cp / max(hpos.sum(), 1); Rp = np.concatenate([[0.], R[:-1]])
    out = dict(iou=tp / max(tp + fp + fn, 1), dice=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
               recall=tp / max(tp + fn, 1), ap=float(np.sum((R - Rp) * P)), valid_voxels=int(n), positive_voxels=int(pos),
               fault_fraction=pos / max(n, 1), seconds=int(time.time() - t0))
    for k in TOLS:
        t = tol[k]
        out[f"tol{k}_precision"] = t["ttp"] / max(t["ttp"] + t["tfp"], 1); out[f"tol{k}_recall"] = t["mgt"] / max(t["mgt"] + t["tfn"], 1)
        out[f"tol{k}_f1"] = 2 * out[f"tol{k}_precision"] * out[f"tol{k}_recall"] / max(out[f"tol{k}_precision"] + out[f"tol{k}_recall"], 1e-9)
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+", required=True); a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--split", default="val"); a.add_argument("--workers", type=int, default=4); a.add_argument("--batch", type=int, default=2)
    a.add_argument("--out", default=None)
    a = a.parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    ds = SpatialThebe(a.split, DEFAULT)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    out = Path(a.out or f"logs/eval_thebe_zeroshot_{a.split}.json")
    rows = json.loads(out.read_text()) if out.exists() else {}
    print(f"{a.split}: {len(ds)} cores, support voxels {ds.record['support_voxels']:,}", flush=True)
    for r in a.runs:
        model, ep = load(r, device); name = Path(r).name
        res = evaluate(model, dl, device); res.update(epoch=ep, split=a.split)
        assert res["valid_voxels"] == ds.record["support_voxels"], (res["valid_voxels"], ds.record["support_voxels"])
        rows[name] = res; out.write_text(json.dumps(rows, indent=1))
        print(f"{name:30s} ep {ep:3d}  IoU {res['iou']:.4f}  Dice {res['dice']:.4f}  AP {res['ap']:.4f}  P {res['precision']:.4f}  R {res['recall']:.4f}  "
              f"tol2 P/R {res['tol2_precision']:.3f}/{res['tol2_recall']:.3f}  tol5 P/R {res['tol5_precision']:.3f}/{res['tol5_recall']:.3f}  ({res['seconds']}s)", flush=True)
        del model; torch.cuda.empty_cache()
    print(f'\n{"run":30s} {"ep":>3s} {"IoU":>6s} {"Dice":>6s} {"AP":>6s} {"P":>6s} {"R":>6s} {"t2P":>6s} {"t2R":>6s} {"t5P":>6s} {"t5R":>6s} {"t5F1":>6s}')
    for k, v in rows.items():
        print(f'{k:30s} {v["epoch"]:3d} {v["iou"]:6.3f} {v["dice"]:6.3f} {v["ap"]:6.3f} {v["precision"]:6.3f} {v["recall"]:6.3f} '
              f'{v["tol2_precision"]:6.3f} {v["tol2_recall"]:6.3f} {v["tol5_precision"]:6.3f} {v["tol5_recall"]:6.3f} {v["tol5_f1"]:6.3f}')


if __name__ == "__main__":
    main()
