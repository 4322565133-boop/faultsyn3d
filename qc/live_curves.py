"""Re-render a comparison curve PNG every N seconds while a run is training.

    python qc/live_curves.py --old runs/maxvit3d_v2_oldrecipe_all3 --new runs/maxvit3d_full_oldrecipe_ddp \
        --out runs/maxvit3d_full_oldrecipe_ddp/live_curves.png --every 60

Stops by itself once the new run's process is gone and the file has been drawn
one last time.  Open the PNG in the IDE; it refreshes when the file changes.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(p):
    p = Path(p)
    p = p if p.suffix == ".csv" else (p / "log.csv" if (p / "log.csv").exists() else p / "history.csv")
    if not p.exists():
        return []
    try:
        return list(csv.DictReader(open(p)))
    except Exception:
        return []


def col(r, k):
    e = [int(x["epoch"]) for x in r if k in x and x[k] not in ("", None)]
    v = [float(x[k]) for x in r if k in x and x[k] not in ("", None)]
    return e, v


def g(x, k, default=float("nan")):
    try:
        return float(x[k])
    except (KeyError, ValueError, TypeError):
        return default


def draw(old, new, out, labels):
    o, n = rows(old), rows(new)
    fig, ax = plt.subplots(2, 2, figsize=(16, 9.5))
    B, R = "#1f77b4", "#d62728"
    a = ax[0, 0]
    for r, c, lab, tk in ((o, B, labels[0], "train_loss"), (n, R, labels[1], "train_seg")):
        if r:
            a.plot(*col(r, "val_loss"), c, lw=2, label=f"{lab} val")
            a.plot(*col(r, tk), c, lw=1.2, ls="--", label=f"{lab} train(seg)")
    a.set_title("seg loss (dice+focal): solid = val, dashed = train"); a.set_ylim(0.08, 0.6)
    a = ax[0, 1]
    for r, c, lab in ((o, B, labels[0]), (n, R, labels[1])):
        if r:
            e, v = col(r, "val_iou"); a.plot(e, v, c, lw=2, label=lab)
            b = int(np.argmax(v)); a.plot(e[b], v[b], "o", c=c, ms=6)
            a.annotate(f"{v[b]:.3f}@{e[b]}", (e[b], v[b]), textcoords="offset points", xytext=(6, -14), color=c, fontsize=9)
    a.set_title("val IoU"); a.set_ylim(0, 0.9)
    a = ax[1, 0]
    for r, c, lab in ((o, B, labels[0]), (n, R, labels[1])):
        if r:
            a.plot(*col(r, "val_precision"), c, lw=1.2, ls="--", label=f"{lab} P")
            a.plot(*col(r, "val_recall"), c, lw=2, label=f"{lab} R")
    a.set_title("val precision (dashed) / recall (solid)"); a.set_ylim(0, 1)
    a = ax[1, 1]
    if n and "train_profile" in n[0]:
        a.plot(*col(n, "train_profile"), "#2ca02c", lw=2, label="profile (raw)")
        a.plot(*col(n, "train_trace"), "#9467bd", lw=2, label="trace (raw)")
        a.plot(*col(n, "ramp"), "#999", lw=1.2, ls=":", label="ramp")
    a.set_title(f"{labels[1]}: geometry losses"); a.set_ylim(0, 1)
    for x in ax.flat:
        x.grid(alpha=.3); x.legend(fontsize=8); x.set_xlim(0, 200); x.set_xlabel("epoch")
    # header
    s = f"updated {time.strftime('%H:%M:%S')}"
    if n:
        ln = n[-1]; same = next((x for x in o if x["epoch"] == ln["epoch"]), None)
        s += (f"  |  new: epoch {ln['epoch']}/200  val IoU {g(ln,'val_iou'):.4f}  P {g(ln,'val_precision'):.3f}  "
              f"R {g(ln,'val_recall'):.3f}  ({ln.get('seconds','?')}s/ep)")
        if same:
            s += (f"  |  old @ same epoch: IoU {g(same,'val_iou'):.4f}  F1 {g(same,'val_f1'):.3f}")
    if o:
        bo = max(o, key=lambda x: float(x["val_iou"]))
        s += f"  |  old final best {float(bo['val_iou']):.4f}@{bo['epoch']}"
    fig.suptitle(s, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    tmp = Path(str(out) + ".tmp.png"); plt.savefig(tmp, dpi=110); plt.close(fig); tmp.replace(out)


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True); p.add_argument("--new", required=True)
    p.add_argument("--out", required=True); p.add_argument("--every", type=int, default=60)
    p.add_argument("--labels", nargs=2, default=["old all3", "new full"])
    p.add_argument("--watch", default="train_old_recipe_ddp.py", help="pgrep pattern of the training process")
    a = p.parse_args()
    while True:
        draw(a.old, a.new, Path(a.out), a.labels)
        if not running(a.watch):
            draw(a.old, a.new, Path(a.out), a.labels)
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
