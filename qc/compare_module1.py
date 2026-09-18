"""Visual A/B for Module 1: same seed, geometry objective off vs on.

The numbers in the ablation only say the dips got steeper and the crossings got
rarer.  They cannot say whether the horsetail still splays or the flower still
opens upward.  This renders both arms side by side so that can be judged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patterns.su_patterns import CATEGORIES, build_case
from qc.render3d import best_slices, draw_cube, draw_surfaces
from su.model import GenConfig, generate_volume


def one(cat, k, use, grid, n_u=128):
    cfg = GenConfig(grid=grid, use_geometry_objective=use, n_u=n_u, n_v=n_u)
    rng = np.random.default_rng(2026 + 1000 * CATEGORIES.index(cat) + k)
    tree, main, sp = build_case(cat, cfg, rng)
    out = generate_volume(tree, main, sp, cfg, rng)
    surfs = [out["surfaces"][i] for i in sorted(out["surfaces"])]
    return out["seismic"], out["label"], surfs, out["meta"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("qc/out/module1"))
    p.add_argument("--grid", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    grid = tuple(a.grid)

    rows = len(a.categories)
    fig = plt.figure(figsize=(4.6 * 4, 4.0 * rows))
    for r, cat in enumerate(a.categories):
        for c, use in enumerate((False, True)):
            seis, lab, surfs, meta = one(cat, a.replicate, use, grid)
            dips = [f["dip"] % 180 for f in meta["faults"][1:]]
            dips = [d if d <= 90 else 180 - d for d in dips]

            ax = fig.add_subplot(rows, 4, r * 4 + c * 2 + 1, projection="3d")
            draw_surfaces(ax, surfs, grid)
            tag = "Su baseline" if not use else "+ Module 1"
            ax.set_title(f"{cat} — {tag}\nbranch dips "
                         f"{[round(d) for d in dips]}", fontsize=8)

            ax2 = fig.add_subplot(rows, 4, r * 4 + c * 2 + 2, projection="3d")
            draw_cube(ax2, seis, lab, grid, show_label=True,
                      at=best_slices(lab))
            ax2.set_title(f"{tag} — seismic + label", fontsize=8)
            print(f"{cat:22s} {tag:12s} dips={[round(d) for d in dips]} "
                  f"topo={meta['pso']['topology_penalty']:.2f} "
                  f"geo={meta['pso']['geometry_penalty']:.3f}", flush=True)

    plt.tight_layout()
    out_png = a.out / f"module1_ab_rep{a.replicate}.png"
    plt.savefig(out_png, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    main()
