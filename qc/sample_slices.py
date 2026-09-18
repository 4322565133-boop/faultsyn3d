"""Randomly sample 2-D slices from the generated volumes, per category.

Draws random (volume, orientation, index) triples so the sample is not biased
toward whatever section happens to cut the main fault.  Also reports how often
a uniformly random slice contains no fault at all, which is a useful sanity
number for downstream 2.5-D training.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORIENTS = ("inline", "crossline", "time")


def cut(vol: np.ndarray, orient: str, i: int) -> np.ndarray:
    """vol is [z, y, x]."""
    if orient == "inline":        # fix y -> (z, x)
        return vol[:, i, :]
    if orient == "crossline":     # fix x -> (z, y)
        return vol[:, :, i]
    return vol[i, :, :]           # fix z -> (y, x)


def load(root: Path, name: str, grid):
    nx, ny, nz = grid
    s = np.fromfile(root / "seismic" / f"{name}.dat", dtype=np.float32).reshape(nz, ny, nx)
    l = np.fromfile(root / "labels" / f"{name}.dat", dtype=np.uint8).reshape(nz, ny, nx)
    return s, l


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/su_repro"))
    p.add_argument("--out", type=Path, default=Path("qc/out/slices"))
    p.add_argument("--grid", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--n", type=int, default=20, help="slices per category")
    p.add_argument("--min-fault-px", type=int, default=25,
                   help="reject slices with fewer fault pixels than this")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = a.grid
    extent = {"inline": ny, "crossline": nx, "time": nz}

    names = sorted(x.stem for x in (a.root / "seismic").glob("*.dat"))
    bycat = collections.defaultdict(list)
    for n in names:
        bycat[n.split("_", 1)[1]].append(n)

    rng = np.random.default_rng(a.seed)
    report = {}

    for cat in sorted(bycat):
        cache = {n: load(a.root, n, tuple(a.grid)) for n in bycat[cat]}
        picks, tried, empty = [], 0, 0
        while len(picks) < a.n and tried < 4000:
            tried += 1
            name = bycat[cat][int(rng.integers(len(bycat[cat])))]
            orient = ORIENTS[int(rng.integers(3))]
            i = int(rng.integers(extent[orient]))
            s2 = cut(cache[name][0], orient, i)
            l2 = cut(cache[name][1], orient, i)
            npx = int(l2.sum())
            if npx == 0:
                empty += 1
            if npx < a.min_fault_px:
                continue
            picks.append((name, orient, i, s2, l2, npx))
        report[cat] = {"tried": tried, "empty": empty,
                       "empty_rate": empty / max(tried, 1), "got": len(picks)}

        rows, cols = 4, 5
        fig, axes = plt.subplots(rows, cols, figsize=(2.55 * cols, 2.55 * rows))
        for ax in axes.ravel():
            ax.axis("off")
        for k, (name, orient, i, s2, l2, npx) in enumerate(picks):
            ax = axes.ravel()[k]
            ax.axis("on")
            ax.imshow(s2, cmap="gray", aspect="auto", vmin=-2.5, vmax=2.5,
                      interpolation="nearest")
            m = np.ma.masked_where(l2 == 0, l2)
            ax.imshow(m, cmap="autumn", aspect="auto", alpha=0.8,
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(f"{name.split('_')[0]} {orient[:5]} {i}  ({npx}px)",
                         fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{cat}  —  {len(picks)} random slices "
                     f"(blank-slice rate {100*report[cat]['empty_rate']:.0f}%)",
                     fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, 0.975))
        out_png = a.out / f"{cat}_slices.png"
        plt.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"{cat:22s} kept={len(picks):2d}  tried={tried:4d}  "
              f"blank-slice rate={100*report[cat]['empty_rate']:5.1f}%  -> {out_png}")


if __name__ == "__main__":
    main()
