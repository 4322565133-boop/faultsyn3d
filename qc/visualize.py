"""QC figures: seismic sections with the fault label overlaid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load(root: Path, name: str, grid):
    nx, ny, nz = grid
    s = np.fromfile(root / "seismic" / f"{name}.dat", dtype=np.float32).reshape(nz, ny, nx)
    l = np.fromfile(root / "labels" / f"{name}.dat", dtype=np.uint8).reshape(nz, ny, nx)
    return s, l


def overlay(ax, seis2d, lab2d, title):
    ax.imshow(seis2d, cmap="gray", aspect="auto",
              vmin=-2.5, vmax=2.5, interpolation="nearest")
    m = np.ma.masked_where(lab2d == 0, lab2d)
    ax.imshow(m, cmap="autumn", aspect="auto", alpha=0.85,
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def grid_figure(root: Path, names, grid, out_png: Path, n_sec=4):
    nx, ny, nz = grid
    fig, axes = plt.subplots(len(names), n_sec,
                             figsize=(3.0 * n_sec, 2.9 * len(names)))
    axes = np.atleast_2d(axes)
    for r, name in enumerate(names):
        s, l = load(root, name, grid)
        # two inline sections, one crossline, one time slice
        secs = [("inline", ny // 3), ("inline", 2 * ny // 3),
                ("crossline", nx // 2), ("time", nz // 2)]
        for c, (kind, i) in enumerate(secs[:n_sec]):
            if kind == "inline":
                s2, l2 = s[:, i, :], l[:, i, :]
            elif kind == "crossline":
                s2, l2 = s[:, :, i], l[:, :, i]
            else:
                s2, l2 = s[i, :, :], l[i, :, :]
            lab = name.split("_", 1)[1] if c == 0 else f"{kind} {i}"
            overlay(axes[r, c], s2, l2, lab)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/su_repro"))
    p.add_argument("--out", type=Path, default=Path("qc/out"))
    p.add_argument("--grid", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--per-category", type=int, default=1)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    names = sorted(x.stem for x in (a.root / "seismic").glob("*.dat"))
    bycat = {}
    for n in names:
        bycat.setdefault(n.split("_", 1)[1], []).append(n)

    picked = [n for cat in sorted(bycat) for n in bycat[cat][:a.per_category]]
    grid_figure(a.root, picked, tuple(a.grid), a.out / "categories_overview.png")
