"""Seeded common augmentation, with sparse geometry transformed identically."""
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, distance_transform_edt
from torch.utils.data import Dataset
from train.dataset import FaultVolumes, CATEGORIES


class GeometryVolumes(Dataset):
    def __init__(self, data, cache, seed=2026, geometry=True):
        self.base = FaultVolumes(data, 'train')
        self.cache, self.seed, self.geometry = Path(cache), seed, geometry
        self.epoch = 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + self.epoch*100003 + i)
        name, cat = self.base.items[i]
        x = self.base._read('seismic', name, np.float32).copy()
        y = (self.base._read('labels', name, np.uint8) > 0).astype(np.float32)
        with np.load(self.cache/f'{name}.npz') as z:
            q, normals = z['points'].copy(), z['normals'].copy()
        axes = [(0, 1), (0, 2), (1, 2)][rng.integers(3)]
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, axes).copy(), np.rot90(y, k, axes).copy()
        a, b = axes
        for _ in range(k):
            qa, na = q[:, a].copy(), normals[:, a].copy()
            q[:, a], q[:, b] = x.shape[a]-1-q[:, b], qa
            normals[:, a], normals[:, b] = -normals[:, b], na
        for axis in range(3):
            if rng.random() < .5:
                x, y = np.flip(x, axis).copy(), np.flip(y, axis).copy()
                q[:, axis] = x.shape[axis]-1-q[:, axis]
                normals[:, axis] *= -1
        if rng.random() < .35:
            x = gaussian_filter(x, sigma=rng.uniform(.4, 1.0))
        # Retained for fidelity to the historical augmentation (z-score cancels it).
        x = x*rng.uniform(.8, 1.2) + rng.uniform(-.15, .15)
        x = (x-x.mean())/(x.std()+1e-6)
        count = min(1024, len(q))
        ids = rng.choice(len(q), count, replace=False)
        points = np.zeros((1024, 3), np.float32)
        ns = np.zeros_like(points)
        valid = np.zeros(1024, bool)
        points[:count], ns[:count], valid[:count] = q[ids], normals[ids], True
        windows = np.full((6, 4), -1, dtype=np.int64)
        if self.geometry:
            for axis in range(3):
                other = [d for d in range(3) if d != axis]
                candidates = rng.permutation(np.flatnonzero(np.abs(normals[:, axis]) < .85))[:40]
                found = 0
                for j in candidates:
                    plane = int(q[j, axis])
                    row, col = np.clip(q[j, other].astype(int)-32, 0, 64)
                    sl = [slice(None)]*3
                    sl[axis] = plane
                    patch = y[tuple(sl)][row:row+64, col:col+64]
                    near = (np.abs(q[:, axis]-plane) < 1.1)
                    near &= (q[:, other[0]] >= row) & (q[:, other[0]] < row+64)
                    near &= (q[:, other[1]] >= col) & (q[:, other[1]] < col+64)
                    if near.sum() < 4 or (np.abs(normals[near, axis]) < .85).mean() < .8:
                        continue
                    if patch.mean() > .15 or distance_transform_edt(patch).max() > 3:
                        continue
                    if patch[22:-22, 22:-22].sum() < 4:
                        continue
                    window = [axis, plane, int(row), int(col)]
                    if any(np.array_equal(window, old) for old in windows):
                        continue
                    windows[2*axis+found] = window
                    found += 1
                    if found == 2:
                        break
        return (torch.from_numpy(np.ascontiguousarray(x[None])), torch.from_numpy(y[None]),
                CATEGORIES.index(cat), torch.from_numpy(points), torch.from_numpy(ns),
                torch.from_numpy(valid), torch.from_numpy(windows))
