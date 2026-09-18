"""Characterise the reproduced dataset: dip distribution, PSO convergence,
label statistics — and compare against the archived stage-1 generator.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label


ARCHIVE = Path("/hdd1/hukaixiao/projects/_archive_20260908_faultsyn_stage1"
               "/data/faults6_stratified_1100/train/metadata")


def wrap_dip(d: float) -> float:
    """Fold dip onto [0, 90]: 100 deg dipping one way == 80 deg the other."""
    d = d % 180.0
    return d if d <= 90.0 else 180.0 - d


def load_new(root: Path):
    by = collections.defaultdict(list)
    for f in sorted((root / "metadata").glob("*.json")):
        m = json.loads(f.read_text())
        by[m["category"]].append(m)
    return by


def load_archive():
    by = collections.defaultdict(list)
    if not ARCHIVE.is_dir():
        return by
    for f in sorted(ARCHIVE.glob("*.json")):
        m = json.loads(f.read_text())
        by[m["category"]].append(m)
    return by


def hist_row(vals, bins):
    v = np.asarray(vals)
    return [100.0 * np.mean((v >= lo) & (v < hi)) for lo, hi in bins]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/su_repro"))
    p.add_argument("--grid", type=int, nargs=3, default=[128, 128, 128])
    a = p.parse_args()

    new = load_new(a.root)
    old = load_archive()
    bins = [(0, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 90.01)]
    hdr = "  ".join(f"{lo}-{hi if hi < 90.1 else 90}" for lo, hi in bins)

    print("=" * 86)
    print("1. BRANCH-FAULT DIP  (main fault excluded — only optimizer-derived dips)")
    print("=" * 86)
    print(f"{'category':22s} {'n':>4s}  {'median':>7s}  {hdr}")
    all_new = []
    for cat in sorted(new):
        d = [wrap_dip(f["dip"]) for m in new[cat] for f in m["faults"] if f["id"] != 0]
        all_new += d
        row = "  ".join(f"{x:5.1f}" for x in hist_row(d, bins))
        print(f"{cat:22s} {len(d):4d}  {np.median(d):7.1f}  {row}")
    if all_new:
        print(f"{'-- ALL (new) --':22s} {len(all_new):4d}  {np.median(all_new):7.1f}  "
              + "  ".join(f"{x:5.1f}" for x in hist_row(all_new, bins)))

    if old:
        print()
        print(f"{'category':22s} {'n':>4s}  {'median':>7s}  {hdr}   <- ARCHIVED stage-1")
        all_old = []
        for cat in sorted(old):
            d = [wrap_dip(f["dip"]) for m in old[cat]
                 for f in m["faults"]["faults"] if f["id"] != "F00"]
            all_old += d
            row = "  ".join(f"{x:5.1f}" for x in hist_row(d, bins))
            print(f"{cat:22s} {len(d):4d}  {np.median(d):7.1f}  {row}")
        print(f"{'-- ALL (archived) --':22s} {len(all_old):4d}  {np.median(all_old):7.1f}  "
              + "  ".join(f"{x:5.1f}" for x in hist_row(all_old, bins)))

        print()
        print("Category/dip separability (how confounded dip is with category):")
        for name, data in (("new", new), ("archived", old)):
            meds = []
            for cat in sorted(data):
                if name == "new":
                    d = [wrap_dip(f["dip"]) for m in data[cat]
                         for f in m["faults"] if f["id"] != 0]
                else:
                    d = [wrap_dip(f["dip"]) for m in data[cat]
                         for f in m["faults"]["faults"] if f["id"] != "F00"]
                if d:
                    meds.append((cat, np.percentile(d, 10), np.percentile(d, 90)))
            # pairwise overlap of [p10, p90] intervals
            ov = 0, 0
            n_pair = n_ov = 0
            for i in range(len(meds)):
                for j in range(i + 1, len(meds)):
                    n_pair += 1
                    lo = max(meds[i][1], meds[j][1])
                    hi = min(meds[i][2], meds[j][2])
                    if hi > lo:
                        n_ov += 1
            print(f"  {name:9s}: {n_ov}/{n_pair} category pairs have overlapping "
                  f"dip ranges (p10-p90)")

    print()
    print("=" * 86)
    print("2. PSO CONVERGENCE")
    print("=" * 86)
    print(f"{'category':22s} {'depth':>6s} {'Nf':>4s} {'converged':>10s} "
          f"{'mean iters':>11s} {'mean sec':>9s}")
    for cat in sorted(new):
        ms = new[cat]
        conv = np.mean([m["pso"]["penalty"] <= 0 for m in ms])
        it = np.mean([m["pso"]["iterations"] for m in ms])
        sec = np.mean([m["seconds"] for m in ms])
        print(f"{cat:22s} {ms[0]['tree_depth']:6d} {ms[0]['n_faults']:4d} "
              f"{100*conv:9.0f}% {it:11.1f} {sec:9.1f}")

    print()
    print("=" * 86)
    print("3. LABEL STATISTICS")
    print("=" * 86)
    nx, ny, nz = a.grid
    print(f"{'category':22s} {'fault%':>8s} {'#comp':>7s} {'Nf':>4s}  "
          f"(#comp should match Nf for P-type, be lower for Y-type)")
    for cat in sorted(new):
        fr, nc, nf = [], [], []
        for m in new[cat]:
            name = f"{m['index']:06d}_{cat}"
            y = np.fromfile(a.root / "labels" / f"{name}.dat",
                            dtype=np.uint8).reshape(nz, ny, nx) > 0
            _, n = cc_label(y, structure=np.ones((3, 3, 3)))
            fr.append(y.mean()); nc.append(n); nf.append(m["n_faults"])
        print(f"{cat:22s} {100*np.mean(fr):7.2f}% {np.mean(nc):7.1f} "
              f"{np.mean(nf):4.1f}")


if __name__ == "__main__":
    main()
