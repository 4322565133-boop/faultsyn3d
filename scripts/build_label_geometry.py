"""Derive sparse, conservative geometry from TRAIN binary labels only."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

VERSION = 'label_pca_r4_k128_v1'


def process_one(job):
    data, out, name, shape, seed = job
    path = Path(data)/'labels'/f'{name}.dat'
    raw = path.read_bytes()
    label_sha = hashlib.sha256(raw).hexdigest()
    dest = Path(out)/f'{name}.npz'
    if dest.exists():
        with np.load(dest) as z:
            if str(z['label_sha']) != label_sha or str(z['version']) != VERSION:
                raise RuntimeError(f'Stale geometry: {dest}')
            return {'name': name, 'valid': len(z['points']), 'candidates': int(z['candidates'])}
    y = np.frombuffer(raw, np.uint8).reshape(shape) > 0
    coords = np.argwhere(y).astype(np.float32)
    eligible = coords[((coords >= 5) & (coords < np.asarray(shape)-5)).all(1)]
    rng = np.random.default_rng(seed)
    # Stratify by an 8-voxel grid before filling from remaining candidates.
    if len(eligible):
        perm = rng.permutation(len(eligible))
        _, first = np.unique((eligible[perm]//8).astype(int), axis=0, return_index=True)
        ids = perm[first]
        ids = np.concatenate([rng.permutation(ids), rng.permutation(np.setdiff1d(perm, ids))])[:2048]
        q = eligible[ids]
    else:
        q = np.zeros((0, 3), np.float32)
    candidates = len(q)
    normals = np.zeros_like(q)
    if len(q):
        tree = cKDTree(coords)
        dist, idx = tree.query(q, k=128, distance_upper_bound=4.01, workers=1)
        valid = np.isfinite(dist)
        local = coords[np.minimum(idx, len(coords)-1)]
        count = valid.sum(1)
        center = (local*valid[..., None]).sum(1)/np.maximum(count[:, None], 1)
        delta = (local-center[:, None])*valid[..., None]
        cov = np.einsum('nki,nkj->nij', delta, delta)/np.maximum(count[:, None, None], 1)
        values, vectors = np.linalg.eigh(cov)
        normals = vectors[:, :, 0].astype(np.float32)
        keep = ((count >= 16) & (values[:, 0]/(values[:, 1]+1e-6) < .25)
                & (values[:, 1]/(values[:, 2]+1e-6) > .2))
        # Reject sampled rays with separate foreground intervals (nearby surfaces).
        ray = q[:, None] + normals[:, None] * np.arange(-4, 4.01, .5)[None, :, None]
        signal = map_coordinates(y.astype(np.float32), ray.reshape(-1, 3).T,
                                 order=1, mode='constant').reshape(len(q), -1) > .4
        starts = signal[:, 0].astype(int) + ((~signal[:, :-1]) & signal[:, 1:]).sum(1)
        keep &= starts == 1
        q, normals = q[keep], normals[keep]
    tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, points=q, normals=normals, label_sha=label_sha,
                        version=VERSION, candidates=candidates)
    os.replace(tmp, dest)
    return {'name': name, 'valid': len(q), 'candidates': candidates}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/dataset_reproduction_v2')
    p.add_argument('--out', default='data/geometry_model_upgrade_v1')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data/'manifest.json').read_text())
    volumes = sorted([v for v in manifest['volumes'] if v['split'] == 'train'], key=lambda x: x['name'])
    shape = tuple(manifest.get('grid', [128]*3)[::-1])
    jobs = [(str(data), str(out), v['name'], shape, 2026+i) for i, v in enumerate(volumes)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(process_one, jobs):
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f'geometry {len(rows)}/{len(jobs)} mean_valid={np.mean([r["valid"] for r in rows]):.1f}', flush=True)
    report = {'version': VERSION, 'split': 'train', 'manifest_sha': hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
              'n': len(rows), 'mean_valid': float(np.mean([r['valid'] for r in rows])), 'volumes': rows}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    if any(r['valid'] < 32 for r in rows):
        raise RuntimeError('Insufficient valid geometry in at least one training volume; inspect report.')
    print('geometry cache complete', flush=True)


if __name__ == '__main__':
    main()
