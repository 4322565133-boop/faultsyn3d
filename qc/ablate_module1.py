"""Module 1 per-term ablation.

Six arms, all generated from identical seeds; only the objective differs.
Every metric below is computed from the generated data alone — no model is
trained.  In particular `semblance AUC` is a DATA-QUALITY measure (are label
voxels less coherent than background?), not a model score.

Each of Module 1's four terms carries a falsifiable claim.  Dropping a term
should degrade the metric that term is responsible for; if it does not, the
claim was wrong and the improvement came from somewhere else.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ARMS = [
    ("su_repro", "baseline (Su)"),
    ("module1", "all three terms"),
    ("m1_no_dip", "- G_dip"),
    ("m1_no_int", "- G_int"),
    ("m1_no_dup", "- G_dup"),
]


def wrap(d):
    d = float(d) % 180.0
    return d if d <= 90.0 else 180.0 - d


def semblance_c1(s, wz=9, w=3):
    """Neidell-Taner C1: sum laterally at each time sample, smooth vertically."""
    N = w * w
    ssum = uniform_filter(s, (1, w, w), mode="nearest") * N
    ssq = uniform_filter(s ** 2, (1, w, w), mode="nearest") * N
    num = uniform_filter(ssum ** 2, (wz, 1, 1), mode="nearest")
    den = uniform_filter(N * ssq, (wz, 1, 1), mode="nearest")
    return num / (den + 1e-8)


def arm_metrics(root: Path, grid, rng):
    nx, ny, nz = grid
    dips, mins, lows, tot = [], [], 0, 0
    within_sd = []
    gaps, aucs, fracs, topo = [], [], [], []
    aucs_by_cat = collections.defaultdict(list)

    for mp in sorted((root / "metadata").glob("*.json")):
        m = json.loads(mp.read_text())
        cat = m["category"]
        d = [wrap(f["dip"]) for f in m["faults"][1:]]
        if d:
            dips += d
            mins.append(min(d))
            lows += sum(1 for x in d if x < 20.0)
            tot += len(d)
            if len(d) > 1:
                within_sd.append(float(np.std(d)))
        gaps += m.get("min_surface_gap", [])
        fracs.append(m["fault_fraction"])
        topo.append(m["pso"].get("topology_penalty", m["pso"]["penalty"]))

        name = mp.stem
        y = np.fromfile(root / "labels" / f"{name}.dat",
                        dtype=np.uint8).reshape(nz, ny, nx) > 0
        s = np.fromfile(root / "seismic" / f"{name}.dat",
                        dtype=np.float32).reshape(nz, ny, nx)
        if not y.any():
            continue
        c = semblance_c1(s)
        bg = (~y) & (rng.random(y.shape) < 0.02)
        fv, bv = c[y], c[bg]
        k = min(len(fv), 20000)
        a = fv[rng.choice(len(fv), k, replace=False)]
        b = bv[rng.choice(len(bv), k, replace=True)]
        auc = float(np.mean(a < b) + 0.5 * np.mean(a == b))
        aucs.append(auc)
        aucs_by_cat[cat].append(auc)

    g = np.array(gaps, dtype=float)
    return {
        "topo": float(np.mean(topo)),
        "min_dip": float(np.mean(mins)) if mins else np.nan,
        "low_pct": 100.0 * lows / max(tot, 1),
        "dip_std": float(np.std(dips)) if dips else np.nan,
        "within_sd": float(np.median(within_sd)) if within_sd else np.nan,
        "collapsed": 100.0 * float(np.mean(np.array(within_sd) < 3.0))
                     if within_sd else np.nan,
        "cross_pct": 100.0 * float(np.mean(g < 1.0)) if g.size else np.nan,
        "auc": float(np.mean(aucs)),
        "auc_listric": float(np.mean(aucs_by_cat["listric_assemblage"]))
                       if aucs_by_cat["listric_assemblage"] else np.nan,
        "frac": 100.0 * float(np.mean(fracs)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--grid", type=int, nargs=3, default=[128, 128, 128])
    a = p.parse_args()
    grid = tuple(a.grid)

    print(f'{"arm":18s} {"topo":>6s} {"minDip":>7s} {"<20°":>6s} {"卷内SD":>7s} '
          f'{"塌缩%":>7s} {"cross":>7s} {"AUC":>7s} {"AUC_lis":>8s} {"fault%":>7s}')
    print("-" * 96)
    res = {}
    for d, label in ARMS:
        root = a.data / d
        if not (root / "metadata").is_dir():
            print(f"{label:18s}  (missing {root})")
            continue
        r = arm_metrics(root, grid, np.random.default_rng(0))
        res[d] = r
        print(f'{label:18s} {r["topo"]:6.2f} {r["min_dip"]:7.1f} '
              f'{r["low_pct"]:5.1f}% {r["within_sd"]:7.2f} {r["collapsed"]:6.1f}% '
              f'{r["cross_pct"]:6.1f}% {r["auc"]:7.3f} {r["auc_listric"]:8.3f} '
              f'{r["frac"]:6.2f}%')

    if "module1" in res:
        print("\n每项的贡献 (相对 all-four; 关掉该项后指标怎么变):")
        full = res["module1"]
        for d, label in ARMS[2:]:
            if d not in res:
                continue
            r = res[d]
            print(f'  {label:10s}  minDip {r["min_dip"]-full["min_dip"]:+6.1f}  '
                  f'卷内SD {r["within_sd"]-full["within_sd"]:+5.2f}  '
                  f'塌缩 {r["collapsed"]-full["collapsed"]:+5.1f}pp  '
                  f'cross {r["cross_pct"]-full["cross_pct"]:+6.1f}pp  '
                  f'AUC {r["auc"]-full["auc"]:+.3f}')


if __name__ == "__main__":
    main()
