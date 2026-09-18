"""Loader for the Thebe field cubes written by synth/thebe_cubes.py (data/thebe_cubes).

Same 7-tuple interface as LegacyVolumes / WuVolumes.  Augmentation for field data keeps the
vertical axis vertical: rot90 only in the inline-crossline plane, flips on all axes (a flip in
depth mirrors dip direction, still a plausible section), blur / gain / offset as before.
Categories are fault-density bins of the cube (none / sparse / dense).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

THEBE_CATEGORIES = ("fault_none", "fault_sparse", "fault_dense")   # fault fraction < 0.5 %, 0.5-2 %, > 2 %
DEFAULT_ROOT = Path("data/thebe_cubes")


def density_bin(fr: float) -> int:
    return 0 if fr < 0.005 else (1 if fr < 0.02 else 2)


class ThebeCubes(Dataset):
    def __init__(self, split: str, root: str | Path = DEFAULT_ROOT, train: bool = False, seed: int = 2026):
        self.root = Path(root)
        man = json.loads((self.root / "manifest.json").read_text())
        key = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}[split]
        self.names = list(man[key])
        fr = {v["name"]: float(v["fault_fraction"]) for v in man["volumes"]}
        self.items = [(nm, THEBE_CATEGORIES[density_bin(fr[nm])]) for nm in self.names]
        self.cat = {nm: density_bin(fr[nm]) for nm in self.names}
        self.categories = THEBE_CATEGORIES
        self.train, self.seed, self.epoch = train, seed, 0
        self.source_sha256 = f"thebe:{self.root.name}:{len(self.names)}"

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        nm = self.names[i]
        x = np.load(self.root / "seismic" / f"{nm}.npy").astype(np.float32)
        y = (np.load(self.root / "labels" / f"{nm}.npy") > 0).astype(np.float32)
        if self.train:
            rng = np.random.default_rng(self.seed + self.epoch * 100003 + i)
            k = int(rng.integers(4))
            x, y = np.rot90(x, k, (1, 2)).copy(), np.rot90(y, k, (1, 2)).copy()      # horizontal plane only
            for ax in range(3):
                if rng.random() < .5:
                    x, y = np.flip(x, ax).copy(), np.flip(y, ax).copy()
            if rng.random() < .35:
                x = gaussian_filter(x, sigma=rng.uniform(.4, 1.0)).astype(np.float32)
            x = x * rng.uniform(.8, 1.2) + rng.uniform(-.15, .15)
        x = (x - x.mean()) / (x.std() + 1e-6)
        q = np.zeros((1024, 3), np.float32); nrm = np.zeros_like(q); valid = np.zeros(1024, bool)
        windows = np.full((6, 4), -1, dtype=np.int64)
        return (torch.from_numpy(x[None].astype(np.float32)), torch.from_numpy(y[None]), self.cat[nm],
                torch.from_numpy(q), torch.from_numpy(nrm), torch.from_numpy(valid), torch.from_numpy(windows))


class ThebeRandomCubes(Dataset):
    """Training cubes re-drawn every epoch from the whole train volume (memmap of train_seis.npy /
    train_fault.npy).  Index i of epoch e maps to a deterministic random origin, so DDP ranks agree
    and runs are reproducible; the same acceptance rules as the fixed cut (zero padding < 1 %, cubes
    with < min_fault kept with prob keep_empty).  Augmentation as ThebeCubes."""

    def __init__(self, root: str | Path = DEFAULT_ROOT, n_per_epoch: int = 2400, seed: int = 2026,
                 min_fault: float = 0.005, keep_empty: float = 0.7, n: int = 128):
        self.root = Path(root)
        meta = json.loads((self.root / "train_volume.json").read_text())
        self.shape, self.band, self.n = tuple(meta["shape"]), tuple(meta["band"]), n
        self.n_per_epoch, self.seed, self.epoch = n_per_epoch, seed, 0
        self.min_fault, self.keep_empty = min_fault, keep_empty
        self.items = [(f"rand_{i:04d}", "fault_dense") for i in range(n_per_epoch)]
        self.categories = THEBE_CATEGORIES
        self.source_sha256 = f"thebe:random-cubes:{n_per_epoch}/epoch"
        self._seis = self._fault = None

    def _open(self):
        if self._seis is None:                      # lazily per worker process
            self._seis = np.load(self.root / "train_seis.npy", mmap_mode="r")
            self._fault = np.load(self.root / "train_fault.npy", mmap_mode="r")

    def __len__(self):
        return self.n_per_epoch

    def _draw(self, rng):
        Z, IL, XL = self.shape; n = self.n
        for _ in range(200):
            z = int(rng.integers(self.band[0], max(self.band[1] - n, self.band[0]) + 1))
            y = int(rng.integers(0, IL - n + 1)); x = int(rng.integers(0, XL - n + 1))
            s = np.asarray(self._seis[z:z + n, y:y + n, x:x + n])
            if (s == 0).mean() > 0.01:
                continue
            f = np.asarray(self._fault[z:z + n, y:y + n, x:x + n])
            if f.mean() < self.min_fault and rng.random() > self.keep_empty:
                continue
            return s.astype(np.float32), f
        return s.astype(np.float32), f

    def __getitem__(self, i):
        self._open()
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + i)
        x, f = self._draw(rng)
        y = (f > 0).astype(np.float32); fr = float(y.mean())
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, (1, 2)).copy(), np.rot90(y, k, (1, 2)).copy()
        for ax in range(3):
            if rng.random() < .5:
                x, y = np.flip(x, ax).copy(), np.flip(y, ax).copy()
        if rng.random() < .35:
            x = gaussian_filter(x, sigma=rng.uniform(.4, 1.0)).astype(np.float32)
        x = x * rng.uniform(.8, 1.2) + rng.uniform(-.15, .15)
        x = (x - x.mean()) / (x.std() + 1e-6)
        q = np.zeros((1024, 3), np.float32); nrm = np.zeros_like(q); valid = np.zeros(1024, bool)
        windows = np.full((6, 4), -1, dtype=np.int64)
        return (torch.from_numpy(x[None].astype(np.float32)), torch.from_numpy(y[None]), density_bin(fr),
                torch.from_numpy(q), torch.from_numpy(nrm), torch.from_numpy(valid), torch.from_numpy(windows))


class ThebeXYC(Dataset):
    def __init__(self, split, root=DEFAULT_ROOT):
        self.ds = ThebeCubes(split, root, train=False)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, c, *_ = self.ds[i]
        return x, y, c
