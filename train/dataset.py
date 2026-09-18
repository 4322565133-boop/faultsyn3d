"""Loader for a manifest-driven dataset (dataset_reproduction_v2 and later).

Layout, flat, with GLOBAL sample numbering:

    <root>/manifest.json                {"volumes": [{"name", "category", "k", "index", "split"}, ...]}
    <root>/seismic/<name>.dat           float32 [z, y, x]
    <root>/labels/<name>.dat            uint8   0/1, full geometric fault label
    <root>/instances/<name>.dat         uint8   0 background, 1..N fault instance   (optional)
    <root>/confidence/<name>.dat        uint8   displacement proxy, diagnostic only (optional)

The split is read from the manifest, never inferred from the file name: names
are numbered globally (000200_horsetail is the FIRST horsetail), so the old
per-category k-range rule would silently mis-split this layout.

The whole 128^3 volume is one sample.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CATEGORIES = ("en_echelon", "horsetail", "negative_flower",
              "positive_flower", "listric_assemblage")


class FaultVolumes(Dataset):
    def __init__(self, root: str | Path, split: str, augment: bool = False,
                 categories=CATEGORIES, limit: int | None = None,
                 label_dir: str = "labels"):
        self.root = Path(root)
        self.augment = augment
        self.label_dir = label_dir
        man = json.loads((self.root / "manifest.json").read_text())
        g = man.get("grid", [128, 128, 128])
        self.grid = tuple(int(v) for v in g)                 # (nx, ny, nz)
        vols = [v for v in man["volumes"]
                if v["split"] == split and v["category"] in categories]
        vols.sort(key=lambda v: v["name"])
        if limit is not None:
            # stratified: the first limit/5 of EACH category, so a learning-curve
            # subset keeps the class balance instead of being all en_echelon
            per = max(limit // len(categories), 1)
            vols = [v for cat in categories
                    for v in [w for w in vols if w["category"] == cat][:per]]
        if not vols:
            raise RuntimeError(f"no volumes for split={split} under {self.root}")
        self.items = [(v["name"], v["category"]) for v in vols]
        self.source_sha256 = man.get("source_sha256")

    def __len__(self):
        return len(self.items)

    def _read(self, sub: str, name: str, dtype):
        nx, ny, nz = self.grid
        return np.fromfile(self.root / sub / f"{name}.dat", dtype=dtype).reshape(nz, ny, nx)

    def __getitem__(self, i):
        name, cat = self.items[i]
        x = self._read("seismic", name, np.float32)
        y = self._read(self.label_dir, name, np.uint8)
        x = (x - x.mean()) / (x.std() + 1e-6)          # per-volume z-score
        y = (y > 0).astype(np.float32)
        if self.augment:
            for ax in range(3):
                if random.random() < 0.5:
                    x, y = np.flip(x, ax), np.flip(y, ax)
            if random.random() < 0.5:                    # inline <-> crossline
                x, y = np.swapaxes(x, 1, 2), np.swapaxes(y, 1, 2)
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        return (torch.from_numpy(x[None].astype(np.float32)),
                torch.from_numpy(y[None]),
                CATEGORIES.index(cat))
