"""Diagnostics that voxel IoU hides, for a set of runs on the val split.

    python train/analyze_runs.py --runs runs/maxvit3d_v2 runs/maxvit3d_v2_frs ... --gpu 3

Per run:
  * best val IoU / P / R and the train-val loss gap at that epoch (from log.csv)
  * tolerance-1 precision / recall / F1   (a predicted voxel within 1 voxel of a
    label voxel counts, and vice versa) -- the metric that removes the
    "drew it one voxel too thick" artefact which made up 87 % of all FPs
  * false positives binned by distance to the nearest label voxel
  * recall binned by the dataset's `confidence` proxy (vertical throw / lambda/4):
    low bins are the low-throw fault segments, the one place the baselines
    differed by 15 points
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build                              # noqa: E402
from train.dataset import FaultVolumes                 # noqa: E402

CONF_BINS = [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]
DIST_BINS = [(0, 2), (2, 4), (4, 8), (8, 1e9)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", default="data/dataset_reproduction_v2")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    va = FaultVolumes(a.data, "val"); nx, ny, nz = va.grid

    nets, info = {}, {}
    for r in a.runs:
        r = Path(r)
        ck = torch.load(r / "best.pt", map_location="cpu", weights_only=False)
        args = ck["args"]; name = args["model"]
        mkw = {}
        for s in args.get("mkw", []):
            k, v = s.split("=", 1)
            for cast in (int, float):
                try: v = cast(v); break
                except ValueError: pass
            mkw[k] = v
        if "full_res_skip" in mkw: mkw["full_res_skip"] = bool(mkw["full_res_skip"])
        net = build(name, **mkw).to(dev).eval(); net.load_state_dict(ck["model"]); nets[r.name] = net
        rows = list(csv.DictReader(open(r / "log.csv")))
        b = max(rows, key=lambda x: float(x["val_iou"]))
        info[r.name] = dict(model=name, mkw=mkw, epochs=len(rows), best_ep=int(b["epoch"]),
                            iou=float(b["val_iou"]), P=float(b["val_precision"]), R=float(b["val_recall"]),
                            gap=float(b["val_loss"]) - float(b["train_loss"]))

    acc = {k: dict(tol=np.zeros(4), fp=np.zeros(len(DIST_BINS)), fn=np.zeros(len(CONF_BINS)),
                   nconf=np.zeros(len(CONF_BINS))) for k in nets}
    for i in range(len(va)):
        x, y, _ = va[i]; name = va.items[i][0]
        yb = y[0].numpy() > 0.5
        conf = np.fromfile(va.root / "confidence" / f"{name}.dat", dtype=np.uint8).reshape(nz, ny, nx) / 255.0
        d_lab = distance_transform_edt(~yb); cf = conf[yb]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            xx = x[None].to(dev)
            for k, net in nets.items():
                pr = (torch.sigmoid(net(xx).float())[0, 0] > 0.5).cpu().numpy()
                d_pred = distance_transform_edt(~pr) if pr.any() else np.full(pr.shape, 99.)
                A = acc[k]
                A["tol"] += [(d_lab[pr] <= 1).sum(), pr.sum(), (d_pred[yb] <= 1).sum(), yb.sum()]
                dd = d_lab[pr & ~yb]
                for j, (lo, hi) in enumerate(DIST_BINS): A["fp"][j] += ((dd >= lo) & (dd < hi)).sum()
                fnm = (~pr)[yb]
                for j, (lo, hi) in enumerate(CONF_BINS):
                    sel = (cf >= lo) & (cf < hi); A["fn"][j] += fnm[sel].sum(); A["nconf"][j] += sel.sum()

    lines = []
    lines.append(f'{"run":22s} {"ep":>3s}/{"n":<3s} {"IoU":>6s} {"P":>6s} {"R":>6s} {"gap":>6s} | {"tol1 P":>6s} {"tol1 R":>6s} {"tol1 F1":>7s} | '
                 + " ".join(f"FP{lo}-{'∞' if hi > 1e8 else hi}".rjust(7) for lo, hi in DIST_BINS)
                 + " | " + " ".join(f"miss<{hi:.2f}".rjust(9) if hi < 1 else "miss≥.75".rjust(9) for lo, hi in CONF_BINS))
    lines.append("-" * len(lines[0]))
    for k in nets:
        I = info[k]; A = acc[k]
        Pt = A["tol"][0] / max(A["tol"][1], 1); Rt = A["tol"][2] / max(A["tol"][3], 1); F = 2 * Pt * Rt / max(Pt + Rt, 1e-9)
        fp_tot = max(A["fp"].sum(), 1)
        lines.append(f'{k:22s} {I["best_ep"]:3d}/{I["epochs"]:<3d} {I["iou"]:6.3f} {I["P"]:6.3f} {I["R"]:6.3f} {I["gap"]:+6.3f} | {Pt:6.3f} {Rt:6.3f} {F:7.3f} | '
                     + " ".join(f"{100*v/fp_tot:6.1f}%" for v in A["fp"]) + " | "
                     + " ".join(f"{100*A['fn'][j]/max(A['nconf'][j],1):8.1f}%" for j in range(len(CONF_BINS))))
    txt = "\n".join(lines); print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n\n" + json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
