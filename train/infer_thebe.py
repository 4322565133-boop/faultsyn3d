"""Full-volume sliding-window inference on the Thebe test split (crosslines 1100-1802).

    python train/infer_thebe.py --runs runs/maxvit3d_thebe_maxvit_pico --gpu 0

Protocol (after An et al. 2021 / the 2026 3D global-attention paper): 128^3 windows, stride 64
(50 % overlap) on every axis, per-window standardisation as in training, fp16 forward, Gaussian
weighted blending of overlapping predictions.  Metrics are computed only on voxels that are
inside the survey (non-zero) and inside the interpreted depth band: voxel IoU / Dice / P / R
at 0.5, AP (histogram PR), and 2-voxel-tolerance P / R.  The blended probability volume is
saved as float16 (<run>/thebe_test_prob.npy, [sample, inline, crossline]).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.analyze_loss_vs_iou import load                    # noqa: E402
from synth.thebe_cubes import load_split, depth_band           # noqa: E402

N, STRIDE, NB = 128, 64, 2000


def gaussian_window(n=N, sigma=N / 4):
    g = np.exp(-0.5 * ((np.arange(n) - (n - 1) / 2) / sigma) ** 2).astype(np.float32)
    return g[:, None, None] * g[None, :, None] * g[None, None, :]


def starts(length, n=N, stride=STRIDE):
    s = list(range(0, max(length - n, 0) + 1, stride))
    if s[-1] != length - n:
        s.append(length - n)
    return s


@torch.no_grad()
def predict_volume(model, seis, valid, device, batch=2):
    Z, IL, XL = seis.shape
    prob = np.zeros(seis.shape, np.float32); wsum = np.zeros(seis.shape, np.float32)
    w = gaussian_window(); wt = torch.from_numpy(w).to(device)
    zs = starts(Z); ys = starts(IL); xs = starts(XL)
    wins = [(z, y, x) for z in zs for y in ys for x in xs if valid[z:z + N, y:y + N, x:x + N].any()]
    t0 = time.time()
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]; xb = []
        for z, y, x in chunk:
            s = seis[z:z + N, y:y + N, x:x + N]; s = (s - s.mean()) / (s.std() + 1e-6)
            xb.append(torch.from_numpy(np.ascontiguousarray(s)))
        xb = torch.stack(xb)[:, None].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(xb).float())[:, 0] * wt
        p = p.cpu().numpy()
        for (z, y, x), pk in zip(chunk, p):
            prob[z:z + N, y:y + N, x:x + N] += pk; wsum[z:z + N, y:y + N, x:x + N] += w
        if (i // batch) % 500 == 0:
            print(f"    {i}/{len(wins)} windows, {time.time() - t0:.0f}s", flush=True)
    prob /= np.maximum(wsum, 1e-6)
    return prob


def metrics(prob, fault, valid, device, chunk=64):
    tp = fp = fn = 0.0; ttp = tfp = tfn = 0.0; matched_gt = 0.0
    hpos = np.zeros(NB); hneg = np.zeros(NB)
    Z = prob.shape[0]
    for z0 in range(0, Z, chunk):
        lo, hi = max(z0 - 2, 0), min(z0 + chunk + 2, Z)              # halo for the tolerance dilation
        p = torch.from_numpy(prob[lo:hi]).to(device); y = torch.from_numpy(fault[lo:hi]).to(device)
        v = torch.from_numpy(valid[lo:hi]).to(device)
        # Unknown/padded voxels must not contribute matches across the mask boundary.
        y = y.bool() & v.bool()
        pr = (p > .5) & v.bool()
        yd = F.max_pool3d(y.float()[None, None], 5, 1, 2)[0, 0] > .5; pd = F.max_pool3d(pr.float()[None, None], 5, 1, 2)[0, 0] > .5
        a, b = z0 - lo, z0 - lo + min(chunk, Z - z0)
        pr, y, v, yd, pd, p = pr[a:b], y[a:b], v[a:b], yd[a:b], pd[a:b], p[a:b]
        tp += (pr & y & v).sum().item(); fp += (pr & ~y & v).sum().item(); fn += (~pr & y & v).sum().item()
        ttp += (pr & yd & v).sum().item(); tfp += (pr & ~yd & v).sum().item(); tfn += (~pd & y & v).sum().item()
        matched_gt += (pd & y & v).sum().item()
        pb = (p * (NB - 1)).round().long().clamp_(0, NB - 1)
        hpos += torch.bincount(pb[y & v], minlength=NB).cpu().numpy(); hneg += torch.bincount(pb[~y & v], minlength=NB).cpu().numpy()
    cp = np.cumsum(hpos[::-1]); cn = np.cumsum(hneg[::-1])
    P = cp / np.maximum(cp + cn, 1); R = cp / max(hpos.sum(), 1); Rp = np.concatenate([[0.], R[:-1]])
    return dict(iou=tp / max(tp + fp + fn, 1), dice=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), ap=float(np.sum((R - Rp) * P)),
                tol2_precision=ttp / max(ttp + tfp, 1), tol2_recall=matched_gt / max(matched_gt + tfn, 1),
                n_valid=int(valid.sum()), fault_fraction=float(fault[valid].mean()))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+", required=True); a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--split", default="test"); a.add_argument("--no-save", action="store_true")
    a = a.parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    t0 = time.time(); print(f"loading Thebe {a.split} ...", flush=True)
    seis, fault = load_split(a.split); band = depth_band(fault)
    valid = seis != 0; valid[:band[0]] = False; valid[band[1]:] = False
    print(f"  volume {seis.shape}, depth band {band}, valid voxels {valid.mean():.3f}, fault fraction (valid) {fault[valid].mean():.4f}, {time.time()-t0:.0f}s", flush=True)
    rows = {}
    for r in a.runs:
        r = Path(r); m, ep = load(r, device); t1 = time.time()
        prob = predict_volume(m, seis, valid, device)
        res = metrics(prob, fault, valid, device); res.update(epoch=ep, seconds=int(time.time() - t1))
        rows[r.name] = res
        (r / f"thebe_{a.split}_fullvol.json").write_text(json.dumps(res, indent=1))
        if not a.no_save:
            np.save(r / f"thebe_{a.split}_prob.npy", prob.astype(np.float16))
        print(f"{r.name:32s} ep {ep:3d}  IoU {res['iou']:.3f}  Dice {res['dice']:.3f}  AP {res['ap']:.3f}  P {res['precision']:.3f}  "
              f"R {res['recall']:.3f}  tol2 P {res['tol2_precision']:.3f} R {res['tol2_recall']:.3f}  ({res['seconds']}s)", flush=True)
        del m, prob; torch.cuda.empty_cache()
    print(f'\n{"run":32s} {"ep":>3s} {"IoU":>6s} {"Dice":>6s} {"AP":>6s} {"P":>6s} {"R":>6s} {"tol2P":>6s} {"tol2R":>6s}')
    for k, v in rows.items():
        print(f'{k:32s} {v["epoch"]:3d} {v["iou"]:6.3f} {v["dice"]:6.3f} {v["ap"]:6.3f} {v["precision"]:6.3f} {v["recall"]:6.3f} {v["tol2_precision"]:6.3f} {v["tol2_recall"]:6.3f}')


if __name__ == "__main__":
    main()
