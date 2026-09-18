"""One panel: val IoU per epoch for several runs, re-drawn every N seconds.

    python qc/live_iou.py --runs runs/maxvit3d_unet_l_ddp runs/maxvit3d_unet_l_dicesq \
        --labels "UNet-L dice+focal" "UNet-L squared-dice+focal" --out runs/loss_ablation_live_iou.png
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]


def rows(run):
    p = Path(run)
    if p.is_dir() and (p / "history.json").exists():          # thebe_spatial_v3 runs
        try:
            h = json.loads((p / "history.json").read_text())
            # curve = whole-validation-region IoU (2433 cores) when the run computes it every epoch
            return [{"epoch": str(r["epoch"]), "val_iou": str((r["full_val"] or r["fast_val"])["iou"]), "full_iou": ""} for r in h]
        except Exception:
            return []
    p = p if p.suffix == ".csv" else (p / "log.csv" if (p / "log.csv").exists() else p / "history.csv")
    if not p.exists():
        return []
    try:
        return [r for r in csv.DictReader(open(p)) if r.get("val_iou")]
    except Exception:
        return []


def draw(runs, labels, out, ymin, xmax=200):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    head = [f"updated {time.strftime('%H:%M:%S')}"]
    for run, lab, c in zip(runs, labels, COLORS):
        r = rows(run)
        if not r:
            continue
        e = np.array([int(x["epoch"]) for x in r]); v = np.array([float(x["val_iou"]) for x in r])
        ax.plot(e, v, c, lw=2, label=lab)
        full = [(int(x["epoch"]), float(x["full_iou"])) for x in r if x.get("full_iou")]
        if full:                                               # full-validation points (every 10 epochs)
            fe, fv = zip(*full); ax.plot(fe, fv, "s", c=c, ms=8, mfc="white", mew=2, label=f"{lab} full val")
            for a_, b_ in full:
                ax.annotate(f"{b_:.3f}", (a_, b_), textcoords="offset points", xytext=(6, 6), color=c, fontsize=9, fontweight="bold")
        b = int(np.argmax(v)); ax.plot(e[b], v[b], "o", c=c, ms=6)
        ax.annotate(f"best {v[b]:.4f} @{e[b]}", (e[b], v[b]), textcoords="offset points", xytext=(6, -14), color=c, fontsize=9)
        ax.annotate(f"{v[-1]:.4f}", (e[-1], v[-1]), textcoords="offset points", xytext=(6, 4), color=c, fontsize=9)
        head.append(f"{lab}: ep {e[-1]}  IoU {v[-1]:.4f}  best {v[b]:.4f}@{e[b]}")
    ax.set_xlim(0, xmax); ax.set_ylim(ymin, 0.9); ax.grid(alpha=.3)
    ax.set_xlabel("epoch"); ax.set_ylabel("val IoU"); ax.legend(loc="lower right")
    ax.set_title("  |  ".join(head), fontsize=9)
    plt.tight_layout()
    tmp = Path(str(out) + ".tmp.png"); plt.savefig(tmp, dpi=110); plt.close(fig); tmp.replace(out)


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True); p.add_argument("--every", type=int, default=60)
    p.add_argument("--ymin", type=float, default=0.5); p.add_argument("--xmax", type=float, default=200)
    p.add_argument("--watch", default="train_old_recipe_ddp.py")
    a = p.parse_args()
    while True:
        draw(a.runs, a.labels, Path(a.out), a.ymin, a.xmax)
        if not running(a.watch):
            draw(a.runs, a.labels, Path(a.out), a.ymin, a.xmax)
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
