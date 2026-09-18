"""Cut 128^3 training / validation / test cubes out of the Thebe field dataset (An et al. 2021).

Source files (Harvard Dataverse doi:10.7910/DVN/YBYGBK, npz): seis{train1-9,val1-2,test1-7}.npz,
each (100 crosslines, 3174 inlines, 1537 samples) float32; fault*.npz the same shape, bool
(expert polylines rasterised, faults with throw > 20 m, ~1 % of voxels).  Split follows the
data descriptor: crosslines 0-899 train, 900-1099 val, 1100-1802 test.

Cubes are written z-first [sample, inline, crossline] = 128^3.  The survey is not rectangular
(zero padding) and faults are only interpreted in a depth band, so a cube is kept only if
< 1 % of its voxels are exactly zero and its origin lies inside the labelled depth band.
Train / val cubes are random; test cubes sit on a regular stride-128 grid (no overlap).

    python synth/thebe_cubes.py --out data/thebe_cubes --n-train 2400 --n-val 240
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SRC = Path("data/thebe")
FILES = {"train": [f"train{i}" for i in range(1, 10)], "val": ["val1", "val2"], "test": [f"test{i}" for i in range(1, 8)]}


def load_split(split):
    seis, fault = [], []
    for k in FILES[split]:
        s = np.load(SRC / "seis" / f"seis{k}.npz")["arr_0"]; f = np.load(SRC / "fault" / f"fault{k}.npz")["arr_0"]
        seis.append(s.astype(np.float32, copy=False)); fault.append(f.astype(bool, copy=False))
        print(f"  {k}: {s.shape} {s.dtype}", flush=True)
    seis = np.concatenate(seis, 0); fault = np.concatenate(fault, 0)          # (XL, IL, Z)
    return np.ascontiguousarray(seis.transpose(2, 1, 0)), np.ascontiguousarray(fault.transpose(2, 1, 0))   # (Z, IL, XL)


def depth_band(fault, n=128, margin=8):
    """sample-index range that holds the interpreted faults (plus margin), clipped so cubes fit"""
    prof = fault.sum((1, 2)); idx = np.nonzero(prof > prof.max() * 0.002)[0]
    lo, hi = max(int(idx[0]) - margin, 0), min(int(idx[-1]) + margin, fault.shape[0])
    return lo, max(hi, lo + n)


def cut(seis, fault, n, rng, count, min_fault, keep_empty, band):
    Z, IL, XL = seis.shape; out = []; tries = 0
    while len(out) < count and tries < count * 200:
        tries += 1
        z = int(rng.integers(band[0], max(band[1] - n, band[0]) + 1)); y = int(rng.integers(0, IL - n + 1)); x = int(rng.integers(0, XL - n + 1))
        s = seis[z:z + n, y:y + n, x:x + n]
        if (s == 0).mean() > 0.01:
            continue
        f = fault[z:z + n, y:y + n, x:x + n]; fr = float(f.mean())
        if fr < min_fault and rng.random() > keep_empty:
            continue
        out.append(((z, y, x), s, f, fr))
    return out


def cut_stratified(seis, fault, n, rng, count, strata, band):
    """fixed validation set with a prescribed share of none / sparse / dense cubes (like the test grid)"""
    want = [int(round(count * f)) for f in strata]; got = [[], [], []]
    Z, IL, XL = seis.shape; tries = 0
    while any(len(g) < w for g, w in zip(got, want)) and tries < count * 400:
        tries += 1
        z = int(rng.integers(band[0], max(band[1] - n, band[0]) + 1)); y = int(rng.integers(0, IL - n + 1)); x = int(rng.integers(0, XL - n + 1))
        s = seis[z:z + n, y:y + n, x:x + n]
        if (s == 0).mean() > 0.01:
            continue
        f = fault[z:z + n, y:y + n, x:x + n]; fr = float(f.mean())
        k = 0 if fr < 0.005 else (1 if fr < 0.02 else 2)
        if len(got[k]) < want[k]:
            got[k].append(((z, y, x), s, f, fr))
    return got[0] + got[1] + got[2]


def grid(seis, fault, n, band):
    Z, IL, XL = seis.shape; out = []
    for z in range(band[0], band[1] - n + 1, n):
        for y in range(0, IL - n + 1, n):
            for x in range(0, XL - n + 1, n):
                s = seis[z:z + n, y:y + n, x:x + n]
                if (s == 0).mean() > 0.01:
                    continue
                f = fault[z:z + n, y:y + n, x:x + n]
                out.append(((z, y, x), s, f, float(f.mean())))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/thebe_cubes"); p.add_argument("--n", type=int, default=128)
    p.add_argument("--n-train", type=int, default=2400); p.add_argument("--n-val", type=int, default=240)
    p.add_argument("--min-fault", type=float, default=0.005, help="cubes below this fault fraction are kept with prob --keep-empty")
    p.add_argument("--keep-empty", type=float, default=0.15); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--val-strata", default="0.45,0.15,0.40", help="share of none / sparse / dense cubes in the fixed val set")
    p.add_argument("--export-train-volume", action="store_true",
                   help="also write the whole train volume as train_seis.npy / train_fault.npy for per-epoch random cubes")
    a = p.parse_args()
    out = Path(a.out); (out / "seismic").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    man = {"source": "Thebe (An et al. 2021), doi:10.7910/DVN/YBYGBK", "cube": a.n, "orientation": "[sample, inline, crossline]",
           "split_rule": "crosslines 0-899 train / 900-1099 val / 1100-1802 test (data descriptor)", "volumes": [],
           "train": [], "validation": [], "test": []}
    if (out / "manifest.json").exists():
        man = json.loads((out / "manifest.json").read_text())
    rng = np.random.default_rng(a.seed)
    for split in a.splits:
        t0 = time.time(); print(f"== {split}", flush=True)
        seis, fault = load_split(split); band = depth_band(fault, a.n)
        print(f"  volume {seis.shape}, fault fraction {fault.mean():.4f}, labelled depth band {band}, {time.time()-t0:.0f}s", flush=True)
        if split == "train" and a.export_train_volume:
            np.save(out / "train_seis.npy", seis); np.save(out / "train_fault.npy", fault)
            (out / "train_volume.json").write_text(json.dumps(dict(shape=list(seis.shape), band=list(band), dtype="float32/uint8",
                                                                      orientation="[sample, inline, crossline]", crosslines="0-899")))
            print(f"  exported train volume to {out}/train_seis.npy ({seis.nbytes/1e9:.1f} GB)", flush=True)
        if split == "test":
            cubes = grid(seis, fault, a.n, band)
        elif split == "val":
            cubes = cut_stratified(seis, fault, a.n, rng, a.n_val, [float(t) for t in a.val_strata.split(",")], band)
        else:
            cubes = cut(seis, fault, a.n, rng, a.n_train, a.min_fault, a.keep_empty, band)
        key = "validation" if split == "val" else split
        man["volumes"] = [v for v in man["volumes"] if v["split"] != key]; man[key] = []
        for i, ((z, y, x), s, f, fr) in enumerate(cubes):
            name = f"thebe_{split}_{i:04d}"
            np.save(out / "seismic" / f"{name}.npy", np.ascontiguousarray(s)); np.save(out / "labels" / f"{name}.npy", f.astype(np.uint8))
            man["volumes"].append(dict(name=name, split=key, origin=[z, y, x], fault_fraction=fr)); man[key].append(name)
        fr = np.array([c[3] for c in cubes])
        print(f"  {len(cubes)} cubes, fault fraction mean {fr.mean():.4f}, empty (<0.5%) {np.mean(fr < 0.005):.2f}, {time.time()-t0:.0f}s", flush=True)
        del seis, fault
        (out / "manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
