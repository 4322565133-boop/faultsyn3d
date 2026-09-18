"""Generate the Su et al. (2026) MultiFaultStyle3D-style dataset.

Usage
-----
    python generate.py --per-category 10 --out data/su_repro
    python generate.py --per-category 1 --categories listric_assemblage --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patterns.su_patterns import CATEGORIES, build_case
from su.model import GenConfig, generate_volume


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/su_repro"))
    p.add_argument("--per-category", type=int, default=10)
    p.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    p.add_argument("--grid", type=int, nargs=3, default=[128, 128, 128],
                   metavar=("NX", "NY", "NZ"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true",
                   help="one volume, print diagnostics, write nothing")
    p.add_argument("--geometry", action="store_true",
                   help="Module 1: enable the geometry-aware objective (ours)")
    p.add_argument("--drop-term", nargs="*", default=[],
                   choices=["dip", "int", "dup"],
                   help="Module 1 per-term ablation: zero out these weights")
    a = p.parse_args()

    cfg = GenConfig(grid=tuple(a.grid), use_geometry_objective=a.geometry,
                    use_surface_repair=a.geometry)
    for t in a.drop_term:
        setattr(cfg, {"dip": "geo_w_dip", "int": "geo_w_int",
                      "dup": "geo_w_dup"}[t], 0.0)
    a.out.mkdir(parents=True, exist_ok=True)
    for sub in ("seismic", "labels", "labels_full", "metadata", "surfaces"):
        (a.out / sub).mkdir(exist_ok=True)

    index = 0
    summary = []
    for cat in a.categories:
        for k in range(a.per_category):
            rng = np.random.default_rng(a.seed + 1000 * CATEGORIES.index(cat) + k)
            t0 = time.perf_counter()
            tree, main, surf = build_case(cat, cfg, rng)
            out = generate_volume(tree, main, surf, cfg, rng)
            dt = time.perf_counter() - t0

            meta = out["meta"]
            meta.update(category=cat, index=index, seed=int(a.seed), replicate=k,
                        seconds=round(dt, 2), grid=list(a.grid))
            dips = [f["dip"] for f in meta["faults"]]
            summary.append({"category": cat, "k": k,
                            "fault_frac": meta["fault_fraction"],
                            "penalty": meta["pso"]["penalty"],
                            "iters": meta["pso"]["iterations"],
                            "dips": [round(d, 1) for d in dips],
                            "seconds": round(dt, 1)})

            print(f"[{cat:20s} {k:02d}] faults={meta['n_faults']} "
                  f"depth={meta['tree_depth']} "
                  f"penalty={meta['pso']['penalty']:.1f} "
                  f"iters={meta['pso']['iterations']:3d} "
                  f"fault%={100*meta['fault_fraction']:.2f} "
                  f"dips={[round(d) for d in dips]} "
                  f"{dt:.1f}s", flush=True)

            if a.smoke:
                return

            name = f"{index:06d}_{cat}"
            out["seismic"].astype(np.float32).tofile(a.out / "seismic" / f"{name}.dat")
            out["label"].astype(np.uint8).tofile(a.out / "labels" / f"{name}.dat")
            out["label_full"].astype(np.uint8).tofile(a.out / "labels_full" / f"{name}.dat")
            (a.out / "metadata" / f"{name}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False))
            np.savez_compressed(a.out / "surfaces" / f"{name}.npz",
                                **{f"f{fid}": pts for fid, pts in out["surfaces"].items()})
            index += 1

    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {index} volumes to {a.out}")


if __name__ == "__main__":
    main()
