"""Synthetic seismic + fault volumes after Wu, Liang, Shi & Fomel (2019), "FaultSeg3D",
Geophysics 84(3) IM35-IM45, section "Synthetic seismic and fault images".

Workflow (paper, Fig. 1):
  a) 1-D reflectivity r(z), random values in [-1, 1], laterally constant
  b) folding by a vertical shift  s1 = a0 + (1.5 z / zmax) * sum_k b_k exp(-((x-c_k)^2+(y-d_k)^2)/(2 sigma_k^2))
     (eq. 1; the 1.5 z/zmax factor damps the folding from below to above), sinc/cubic resampling
  c) planar shearing  s2 = e0 + f x + g y   (eq. 2), r(x, y, z + s1 + s2)
  d) planar faults, each with its own strike / dip / displacement; displacement along the plane is
     either Gaussian (decays from the fault centre in all directions) or linear (grows / shrinks
     in the dip direction: normal / reverse); max displacement per fault in (0, 40] samples;
     "more than five faults" in the model, "not too close to each other"
  e) 1-D convolution with a Ricker wavelet of random peak frequency, AFTER folding and faulting
  f) random noise; then a 128^3 crop from a larger model to avoid boundary artefacts
  labels: ones on the voxels adjacent to the fault plane on the hanging-wall and footwall sides

Numeric ranges the paper leaves as "predefined ranges" are calibrated on the released
validation volumes (data/wangjing/validation: fault fraction 4-11 %, 2-4 large fault
components per 128^3 crop, dominant frequency 0.07-0.16 cycles/sample, noise up to
signal level) and are all listed in DEFAULTS so they can be reported and varied.

Volumes are written z-first, [z, y, x] = [128, 128, 128], float32 seismic (per-volume
standardised) and uint8 labels, the convention of the rest of this project.
"""
from __future__ import annotations

import json
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, convolve1d

DEFAULTS = dict(
    n=128, pad=16,                      # crop size and margin on each side (model = 160^3)
    n_gauss=(3, 6),                     # N in eq. 1
    b=(6.0, 20.0),                      # |b_k| samples of fold amplitude (sign random)
    sigma=(12.0, 40.0),                 # sigma_k voxels
    a0=(-3.0, 3.0),
    fg=(-0.15, 0.15),                   # f, g of the planar shear (samples per voxel)
    e0=(-3.0, 3.0),
    n_faults=(3, 8),                    # faults placed in the 160^3 model; 2-6 usually intersect the crop
    strike_spread=12.0,                 # deg; faults of a volume share a strike family (released data: parallel sets)
    p_conjugate=0.3,                    # probability a fault dips the opposite way (X pattern in section)
    dip=(60.0, 86.0),                   # degrees from horizontal (released volumes: mostly steep)
    dmax=(4.0, 40.0),                   # maximum displacement, samples (paper: (0, 40])
    p_gauss=0.5,                        # Gaussian vs linear displacement distribution
    gauss_sigma=(30.0, 70.0),           # decay length of the Gaussian displacement along the plane
    min_sep=18.0,                       # minimum distance between anchor points of near-parallel faults
    fpeak=(0.08, 0.16),                 # Ricker peak frequency, cycles / sample
    noise=(0.1, 0.6),                   # white-noise std relative to the signal std
    noise_smooth=(0.0, 0.8),            # optional gaussian smoothing sigma of the noise (0 = white)
    label_halfwidth=1.0,                # |signed distance| <= this is labelled (~2 voxels thick)
    label_min_disp=1.0,                 # label only where the displacement exceeds this (finite fault patches)
)


def ricker(fpeak, n=None):
    if n is None:
        n = int(np.ceil(3.0 / fpeak)) * 2 + 1
    t = np.arange(n) - n // 2
    a = (np.pi * fpeak * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def generate(seed, cfg=None):
    c = {**DEFAULTS, **(cfg or {})}
    rng = np.random.default_rng(seed)
    U = lambda lo_hi: float(rng.uniform(*lo_hi))
    n, pad = c["n"], c["pad"]; N = n + 2 * pad
    z = np.arange(N, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, z, z, indexing="ij")            # [z, y, x]

    # (a) reflectivity, laterally constant.  Every deformation below is a coordinate mapping; we
    # compose them analytically (undo the faults from the last to the first, then undo the folding),
    # so the reflectivity is sampled exactly once and no resampling artefact can mark the faults.
    r1 = rng.uniform(-1, 1, N).astype(np.float32)

    # (b) folding, eq. 1 -- parameters
    ng = int(rng.integers(c["n_gauss"][0], c["n_gauss"][1] + 1))
    gauss = []
    for _ in range(ng):
        b = U(c["b"]) * (1 if rng.random() < .5 else -1); cx, cy = rng.uniform(0, N, 2); sg = U(c["sigma"])
        gauss.append(dict(b=b, c=float(cx), d=float(cy), sigma=sg))
    a0 = U(c["a0"])
    # (c) planar shear, eq. 2 -- parameters
    e0, f, g = U(c["e0"]), U(c["fg"]), U(c["fg"])

    # (d) planar faults: sample parameters first (youngest fault = last in the list)
    nf = int(rng.integers(c["n_faults"][0], c["n_faults"][1] + 1))
    strike0 = rng.uniform(0, 360)
    faults = []; anchors = []
    tries = 0
    while len(faults) < nf and tries < 200:
        tries += 1
        dip = np.deg2rad(U(c["dip"]))
        strike = np.deg2rad(strike0 + rng.normal(0, c["strike_spread"]) + (180.0 if rng.random() < c["p_conjugate"] else 0.0))
        # coordinates: x east, y north, z down.  strike direction t, down-dip direction dd, normal n
        ts = np.array([np.cos(strike), np.sin(strike), 0.0])
        dd = np.array([-np.sin(strike) * np.cos(dip), np.cos(strike) * np.cos(dip), np.sin(dip)])
        nrm = np.cross(ts, dd)
        p0 = rng.uniform(pad, N - pad, 3)                                                            # anchor (x, y, z)
        if any(np.linalg.norm(p0 - q) < c["min_sep"] and abs(np.dot(nrm, m)) > 0.85 for q, m in anchors):
            continue
        anchors.append((p0, nrm))
        dmax = U(c["dmax"]); gtype = rng.random() < c["p_gauss"]
        faults.append(dict(dip_deg=float(np.rad2deg(dip)), strike_deg=float(np.rad2deg(strike)), anchor=p0.tolist(),
                           dmax=dmax, kind="gaussian" if gtype else "linear",
                           gauss_sigma=U(c["gauss_sigma"]) if gtype else None,
                           sign=1.0 if rng.random() < .7 else -1.0,
                           _ts=ts, _dd=dd, _n=nrm))

    # undo the faults, youngest first: X, Y, Zc become the pre-faulting coordinates of every voxel;
    # a fault's plane (and label) sits wherever the partially-undone coordinates cross it, so older
    # faults are offset by younger ones exactly as their reflectors are.
    X, Y, Zc = xx.copy(), yy.copy(), zz.copy()
    lab = np.zeros((N, N, N), bool)
    for fp in reversed(faults):
        ts, dd, nrm = fp["_ts"], fp["_dd"], fp["_n"]; p0 = np.asarray(fp["anchor"])
        Px, Py, Pz = X - p0[0], Y - p0[1], Zc - p0[2]
        dist = Px * nrm[0] + Py * nrm[1] + Pz * nrm[2]
        u = Px * ts[0] + Py * ts[1]; v = Px * dd[0] + Py * dd[1] + Pz * dd[2]
        if fp["kind"] == "gaussian":
            disp = fp["dmax"] * np.exp(-(u ** 2 + v ** 2) / (2 * fp["gauss_sigma"] ** 2))
        else:                                                          # linear along dip, normal (+) or reverse (-)
            disp = np.clip(fp["dmax"] * (0.5 + fp["sign"] * v / N), 0, fp["dmax"])
        lab |= (np.abs(dist) <= c["label_halfwidth"]) & (disp > c["label_min_disp"])
        d3 = (disp * (dist > 0)).astype(np.float32)                    # hanging wall moved down-dip
        X = X - d3 * dd[0]; Y = Y - d3 * dd[1]; Zc = Zc - d3 * dd[2]
        del Px, Py, Pz, dist, u, v, disp, d3
    for fp in faults:
        for k in ("_ts", "_dd", "_n"): fp.pop(k)
    lab = lab.astype(np.uint8)

    # undo folding + shear at the pre-faulting coordinates: source depth of the flat model
    s1 = np.zeros((N, N, N), np.float32)
    for gk in gauss:
        s1 += gk["b"] * np.exp(-((X - gk["c"]) ** 2 + (Y - gk["d"]) ** 2) / (2 * gk["sigma"] ** 2))
    Z = Zc + a0 + (1.5 * Zc / (N - 1)) * s1 + e0 + f * X + g * Y
    del s1, X, Y, Zc
    r = map_coordinates(r1, [Z.ravel()], order=3, mode="nearest").reshape(N, N, N)   # folded + faulted reflectivity
    # (e) wavelet along z, after folding + faulting (blurs the discontinuities, as in the paper)
    fpeak = U(c["fpeak"]); w = ricker(fpeak).astype(np.float32)
    seis = convolve1d(r, w, axis=0, mode="nearest")
    # (f) noise, then crop
    seis = seis / (seis.std() + 1e-6)
    nz = rng.standard_normal(seis.shape).astype(np.float32); ns = U(c["noise_smooth"])
    if ns > 0:
        nz = gaussian_filter(nz, ns); nz /= (nz.std() + 1e-6)
    nlev = U(c["noise"]); seis = seis + nlev * nz
    sl = slice(pad, pad + n)
    seis = seis[sl, sl, sl]; lab = lab[sl, sl, sl]
    seis = ((seis - seis.mean()) / (seis.std() + 1e-6)).astype(np.float32)
    meta = dict(seed=seed, n_gauss=ng, gauss=gauss, a0=a0, e0=e0, f=f, g=g, faults=faults, fpeak=fpeak,
                noise=nlev, noise_smooth=ns, fault_fraction=float(lab.mean()))
    return seis, lab, meta


if __name__ == "__main__":
    import argparse, time
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/wu2019_1000"); p.add_argument("--count", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2019); p.add_argument("--workers", type=int, default=24)
    p.add_argument("--split", default="800,100,100")
    a = p.parse_args()
    out = Path(a.out); (out / "seismic").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True); (out / "params").mkdir(exist_ok=True)

    def job(i):
        s, l, m = generate(a.seed * 100000 + i)
        name = f"wu_{i:04d}"
        np.save(out / "seismic" / f"{name}.npy", s); np.save(out / "labels" / f"{name}.npy", l)
        (out / "params" / f"{name}.json").write_text(json.dumps(m))
        return name, m["fault_fraction"], len(m["faults"])

    t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(job, range(a.count)), 1):
            rows.append(res)
            if k % 100 == 0: print(f"{k}/{a.count}  {time.time()-t0:.0f}s", flush=True)
    ntr, nva, nte = map(int, a.split.split(","))
    names = [r[0] for r in rows]
    man = dict(generator="synth/wu2019.py", paper="Wu et al. 2019 FaultSeg3D, Geophysics 84(3)", seed=a.seed, defaults=DEFAULTS,
               volumes=[dict(name=r[0], fault_fraction=r[1], n_faults=r[2]) for r in rows],
               train=names[:ntr], validation=names[ntr:ntr + nva], test=names[ntr + nva:ntr + nva + nte])
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    fr = np.array([r[1] for r in rows]); print(f"done {len(rows)} volumes in {time.time()-t0:.0f}s; fault fraction mean {fr.mean():.3f} (min {fr.min():.3f} max {fr.max():.3f})")
