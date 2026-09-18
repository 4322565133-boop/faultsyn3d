"""Loader for the Wu-2019-style synthetic set written by synth/wu2019.py (data/wu2019_1000).

Same augmentation and 7-tuple interface as LegacyVolumes so train_old_recipe_ddp.py can
consume it with --wu.  Categories are noise-level bins (the one axis that varies most
across the released FaultSeg3D volumes), so per-category IoU reads as noise robustness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

WU_CATEGORIES = ("noise_low", "noise_mid", "noise_high")     # noise std / signal std: <0.33, 0.33-0.57, >0.57
DEFAULT_ROOT = Path("data/wu2019_1000")


def noise_bin(noise: float) -> int:
    return 0 if noise < 0.33 else (1 if noise < 0.57 else 2)


class WuVolumes(Dataset):
    def __init__(self, split: str, root: str | Path = DEFAULT_ROOT, train: bool = False, seed: int = 2026):
        self.root = Path(root)
        man = json.loads((self.root / "manifest.json").read_text())
        key = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}[split]
        self.names = list(man[key])
        self.train, self.seed, self.epoch = train, seed, 0
        self.noise = {}
        for nm in self.names:
            self.noise[nm] = float(json.loads((self.root / "params" / f"{nm}.json").read_text())["noise"])
        self.items = [(nm, WU_CATEGORIES[noise_bin(self.noise[nm])]) for nm in self.names]
        self.categories = WU_CATEGORIES
        self.source_sha256 = f"wu2019:{self.root.name}:seed{man.get('seed')}"

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        nm = self.names[i]
        x = np.load(self.root / "seismic" / f"{nm}.npy").astype(np.float32)
        y = (np.load(self.root / "labels" / f"{nm}.npy") > 0).astype(np.float32)
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
        cat = noise_bin(self.noise[nm])
        q = np.zeros((1024, 3), np.float32); nrm = np.zeros_like(q); valid = np.zeros(1024, bool)
        windows = np.full((6, 4), -1, dtype=np.int64)
        return (torch.from_numpy(x[None].astype(np.float32)), torch.from_numpy(y[None]), cat,
                torch.from_numpy(q), torch.from_numpy(nrm), torch.from_numpy(valid), torch.from_numpy(windows))


class WuXYC(Dataset):
    """(x, y, cat) view for train.train.evaluate / evaluate.py."""
    def __init__(self, split, root=DEFAULT_ROOT):
        self.ds = WuVolumes(split, root, train=False)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, c, *_ = self.ds[i]
        return x, y, c
