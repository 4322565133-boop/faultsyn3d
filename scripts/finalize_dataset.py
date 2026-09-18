"""Turn the five per-category generator outputs into one finished dataset.

    python scripts/finalize_dataset.py --parts data/dataset_v1_parts --out data/dataset_v1

Checks every category delivered every volume with all five file kinds, then
builds a flat directory of relative symlinks (no copies: the .dat files are
8 MB each and the parts stay the source of truth), writes manifest.json with
the train/val/test split from configs/dataset_v1.json, and summary.json with
per-category statistics of the things we care about for training.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

KINDS = ("seismic", "labels", "labels_full", "metadata", "surfaces")
EXT = {"seismic": ".dat", "labels": ".dat", "labels_full": ".dat",
       "metadata": ".json", "surfaces": ".npz"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", type=Path, default=Path("data/dataset_v1_parts"))
    p.add_argument("--out", type=Path, default=Path("data/dataset_v1"))
    p.add_argument("--config", type=Path, default=Path("configs/dataset_v1.json"))
    a = p.parse_args()

    cfg = json.loads(a.config.read_text())
    cats = list(cfg["tree_specs"].keys())
    n_per = int(cfg["per_category"])
    split_of = lambda k: "train" if k < 160 else ("val" if k < 180 else "test")

    # ---- 1. completeness ------------------------------------------------
    problems = []
    for cat in cats:
        for k in range(n_per):
            name = f"{k:06d}_{cat}"
            for kind in KINDS:
                f = a.parts / cat / kind / f"{name}{EXT[kind]}"
                if not f.exists() or f.stat().st_size == 0:
                    problems.append(str(f))
    if problems:
        print(f"INCOMPLETE: {len(problems)} missing/empty files, e.g.")
        for s in problems[:8]:
            print("   ", s)
        raise SystemExit(1)
    print(f"complete: {len(cats)} categories x {n_per} volumes, all {len(KINDS)} kinds present")

    # ---- 2. flat layout via relative symlinks -----------------------------
    for kind in KINDS:
        (a.out / kind).mkdir(parents=True, exist_ok=True)
    manifest = []
    for cat in cats:
        for k in range(n_per):
            name = f"{k:06d}_{cat}"
            for kind in KINDS:
                dst = a.out / kind / f"{name}{EXT[kind]}"
                src = a.parts / cat / kind / f"{name}{EXT[kind]}"
                rel = os.path.relpath(src.resolve(), dst.parent.resolve())
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(rel)
            manifest.append({"name": name, "category": cat, "k": k,
                             "split": split_of(k)})
    (a.out / "manifest.json").write_text(json.dumps(
        {"config": str(a.config), "n": len(manifest), "volumes": manifest}, indent=1))
    counts = defaultdict(int)
    for m in manifest:
        counts[m["split"]] += 1
    print("split:", dict(counts))

    # ---- 3. summary statistics --------------------------------------------
    agg = defaultdict(lambda: defaultdict(list))
    for m in manifest:
        meta = json.loads((a.out / "metadata" / f"{m['name']}.json").read_text())
        d = agg[m["category"]]
        d["fault_fraction"].append(meta["fault_fraction"])
        d["fault_fraction_full"].append(meta["fault_fraction_full"])
        d["label_kept_frac"].append(meta["label_kept_frac"])
        d["lambda4_samples"].append(meta["lambda4_samples"])
        d["topology_penalty"].append(meta["pso"].get("topology_penalty", meta["pso"]["penalty"]))
        d["pso_iterations"].append(meta["pso"]["iterations"])
        d["n_faults"].append(meta["n_faults"])
        d["seconds"].append(meta["seconds"])
    summary = {}
    print(f'\n{"category":20s} {"fault%":>7s} {"full%":>6s} {"kept":>5s} {"λ/4":>5s} {"topo=0":>7s} {"iters":>6s} {"s/vol":>6s}')
    for cat in cats:
        d = agg[cat]
        s = {k: float(np.mean(v)) for k, v in d.items()}
        s["topology_zero_frac"] = float(np.mean(np.array(d["topology_penalty"]) == 0.0))
        summary[cat] = s
        print(f'{cat:20s} {100*s["fault_fraction"]:7.2f} {100*s["fault_fraction_full"]:6.2f} '
              f'{s["label_kept_frac"]:5.2f} {s["lambda4_samples"]:5.2f} '
              f'{100*s["topology_zero_frac"]:6.0f}% {s["pso_iterations"]:6.1f} {s["seconds"]:6.1f}')
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {a.out}/manifest.json and summary.json")


if __name__ == "__main__":
    main()
