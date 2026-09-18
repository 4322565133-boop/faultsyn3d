"""Loader for the archived stage-1 dataset (faults6_stratified_1100), 6 categories.

Reads the exact 880/110/110 split recorded by the archived MaxViT baseline run
(split_manifest.json, seed 2026), so results are comparable to that run's
test IoU 0.611.  Files are .npy volumes [z, y, x] = 128^3, float32 seismic and
uint8 labels; category is in the file name.

Returns the same 7-tuple as GeometryVolumes (x, y, cat, q, normals, valid,
windows) with empty geometry, so train_old_recipe_ddp.py can consume it with
the geometry losses off.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

LEGACY_CATEGORIES = ("en_echelon", "horsetail", "negative_flower",
                     "positive_flower", "listric_assemblage", "intersecting_conjugate")
ARCHIVE = Path("/hdd1/hukaixiao/projects/_archive_20260908_faultsyn_stage1")
DEFAULT_MANIFEST = ARCHIVE / "runs/maxvit_faultvitnet_fullvol_200ep/split_manifest.json"


class LegacyVolumes(Dataset):
    def __init__(self, split: str, manifest: str | Path = DEFAULT_MANIFEST, train: bool = False,
                 seed: int = 2026):
        man = json.loads(Path(manifest).read_text())
        key = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}[split]
        self.paths = [ARCHIVE / p for p in man[key]]
        self.train, self.seed, self.epoch = train, seed, 0
        self.items = [(p.stem, p.stem.split("_", 1)[1]) for p in self.paths]
        self.categories = LEGACY_CATEGORIES
        self.source_sha256 = f"legacy:{Path(manifest).name}:seed{man.get('seed')}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        x = np.load(p).astype(np.float32)
        y = (np.load(str(p).replace("/seismic/", "/labels/")) > 0).astype(np.float32)
        if self.train:
            rng = np.random.default_rng(self.seed + self.epoch * 100003 + i)   # same scheme as GeometryVolumes
            axes = [(0, 1), (0, 2), (1, 2)][rng.integers(3)]; k = int(rng.integers(4))
            x, y = np.rot90(x, k, axes).copy(), np.rot90(y, k, axes).copy()
            for ax in range(3):
                if rng.random() < .5:
                    x, y = np.flip(x, ax).copy(), np.flip(y, ax).copy()
            if rng.random() < .35:
                x = gaussian_filter(x, sigma=rng.uniform(.4, 1.0)).astype(np.float32)
            x = x * rng.uniform(.8, 1.2) + rng.uniform(-.15, .15)
        x = (x - x.mean()) / (x.std() + 1e-6)
        cat = LEGACY_CATEGORIES.index(self.items[i][1])
        q = np.zeros((1024, 3), np.float32); nrm = np.zeros_like(q); valid = np.zeros(1024, bool)
        windows = np.full((6, 4), -1, dtype=np.int64)
        return (torch.from_numpy(x[None].astype(np.float32)), torch.from_numpy(y[None]), cat,
                torch.from_numpy(q), torch.from_numpy(nrm), torch.from_numpy(valid), torch.from_numpy(windows))
