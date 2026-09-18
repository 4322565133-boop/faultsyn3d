"""Live PNG for thebe_spatial_v3 runs (history.json): fast-val IoU / P / R per epoch, full-val IoU
points, train loss and lr.  Re-drawn every N seconds while the run's status is "running".

    python qc/live_iou_spatial.py --runs runs/thebe_spatial_v3_unet_l_trial --labels "UNet-L 9.6M" \
        --out runs/thebe_spatial_v3_unet_l_trial/live_iou.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def load(run):
    p = Path(run) / "history.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def status(run):
    p = Path(run) / "status.json"
    try:
        return json.loads(p.read_text()).get("status", "?")
    except Exception:
        return "?"


def draw(runs, labels, out, xmax):
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    head = [f"updated {time.strftime('%H:%M:%S')}"]
    for run, lab, c in zip(runs, labels, COLORS):
        h = load(run)
        if not h:
            continue
        e = [r["epoch"] for r in h]
        fv = [r["fast_val"] for r in h]
        iou = [f["iou"] for f in fv]; P = [f["precision"] for f in fv]; R = [f["recall"] for f in fv]
        ax[0, 0].plot(e, iou, c, lw=2, label=f"{lab} fast-val (96 cores)")
        full = [(r["epoch"], r["full_val"]["iou"]) for r in h if r.get("full_val")]
        if full:
            ax[0, 0].plot(*zip(*full), "o--", c=c, ms=7, lw=1, label=f"{lab} FULL val (2433 cores)")
            for fe, fi in full:
                ax[0, 0].annotate(f"{fi:.3f}", (fe, fi), textcoords="offset points", xytext=(5, 6), color=c, fontsize=9)
        ax[0, 0].annotate(f"{iou[-1]:.3f}", (e[-1], iou[-1]), textcoords="offset points", xytext=(5, -12), color=c, fontsize=9)
        ax[0, 1].plot(e, P, c, lw=1.4, ls="--", label=f"{lab} P"); ax[0, 1].plot(e, R, c, lw=2, label=f"{lab} R")
        ax[1, 0].plot(e, [r["train_loss"] for r in h], c, lw=2, label=f"{lab} train loss")
        ax[1, 0].plot(e, [f["loss"] for f in fv], c, lw=1.4, ls="--", label=f"{lab} fast-val loss")
        ax[1, 1].plot(e, [r["lr"] for r in h], c, lw=2, label=f"{lab} lr")
        head.append(f"{lab}: ep {e[-1]} [{status(run)}]  fast IoU {iou[-1]:.4f} P {P[-1]:.3f} R {R[-1]:.3f}  "
                    f"({h[-1]['seconds']:.0f}s/ep)" + (f"  full-val best {max(f for _, f in full):.4f}" if full else ""))
    ax[0, 0].set_title("val IoU (masked, centre-64³ cores, thr 0.5)"); ax[0, 0].set_ylim(0, 0.8)
    ax[0, 1].set_title("fast-val precision (dashed) / recall (solid)"); ax[0, 1].set_ylim(0, 1)
    ax[1, 0].set_title("loss: 0.6 Dice + 0.4 Focal, masked"); ax[1, 0].set_ylim(0.2, 0.65)
    ax[1, 1].set_title("learning rate"); ax[1, 1].set_yscale("log")
    for a in ax.flat:
        a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlim(0, xmax); a.set_xlabel("epoch")
    fig.suptitle("  |  ".join(head), fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    tmp = Path(str(out) + ".tmp.png"); plt.savefig(tmp, dpi=110); plt.close(fig); tmp.replace(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True); p.add_argument("--every", type=int, default=60)
    p.add_argument("--xmax", type=float, default=100)
    a = p.parse_args()
    while True:
        draw(a.runs, a.labels, Path(a.out), a.xmax)
        if not any(status(r) == "running" for r in a.runs):
            draw(a.runs, a.labels, Path(a.out), a.xmax)
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
