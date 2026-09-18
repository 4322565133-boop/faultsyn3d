"""Stratigraphy, folding and seismic forward modelling.

Su et al. (2026), Section III-D (Eq. 54-56) for the structural part, and
Section IV-B-1 for the 1-D Ricker convolution forward model inherited from
Wu et al. (2019, 2020).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Stratigraphic shift fields  (Eq. 54, 55)
# --------------------------------------------------------------------------

def linear_shift(x: np.ndarray, y: np.ndarray, a: float, b: float,
                 x0: float, y0: float) -> np.ndarray:
    """S_1 of Eq. (54).  c0 makes the centre trace unshifted."""
    c0 = -a * x0 - b * y0
    return a * x + b * y + c0


def fold_shift(x: np.ndarray, y: np.ndarray, z: np.ndarray,
               folds: list[dict], z_max: float) -> np.ndarray:
    """S_2 of Eq. (55): Gaussian folds whose amplitude grows with depth."""
    acc = np.zeros_like(x, dtype=np.float32)
    for f in folds:
        acc += np.float32(
            f["b"] * np.exp(-(((x - f["c"]) ** 2 + (y - f["d"]) ** 2)
                              / (2.0 * f["sigma"] ** 2)))
        )
    return np.float32(1.5 / z_max) * z.astype(np.float32) * acc


def random_folds(rng: np.random.Generator, n_folds: int,
                 nx: int, ny: int, amp: tuple[float, float],
                 sigma: tuple[float, float]) -> list[dict]:
    return [{
        "b": float(rng.uniform(*amp) * rng.choice([-1.0, 1.0])),
        "c": float(rng.uniform(-0.2 * nx, 1.2 * nx)),
        "d": float(rng.uniform(-0.2 * ny, 1.2 * ny)),
        "sigma": float(rng.uniform(*sigma)),
    } for _ in range(n_folds)]


# --------------------------------------------------------------------------
# Reflectivity
# --------------------------------------------------------------------------

def reflectivity_1d(nz: int, rng: np.random.Generator,
                    n_layers: tuple[int, int] = (40, 120)) -> np.ndarray:
    """1-D reflectivity series in [-1, 1], as in Wu et al. and Section III-D.

    Sampled on a finely oversampled depth axis so that it can be interpolated
    at non-integer deformed coordinates without aliasing.
    """
    k = int(rng.integers(*n_layers))
    edges = np.sort(rng.uniform(0, nz, size=k))
    values = rng.uniform(-1.0, 1.0, size=k + 1)
    axis = np.arange(nz, dtype=np.float32)
    idx = np.searchsorted(edges, axis)
    return values[idx].astype(np.float32)


def sample_reflectivity(ref: np.ndarray, z_def: np.ndarray) -> np.ndarray:
    """Look up the 1-D reflectivity at deformed depths (linear interpolation)."""
    nz = len(ref)
    zc = np.clip(z_def, 0.0, nz - 1.001)
    z0 = np.floor(zc).astype(np.int32)
    w = (zc - z0).astype(np.float32)
    return (1.0 - w) * ref[z0] + w * ref[z0 + 1]


# --------------------------------------------------------------------------
# Forward modelling
# --------------------------------------------------------------------------

def ricker(freq_hz: float, dt: float, length: float = 0.128) -> np.ndarray:
    """Ricker wavelet, peak frequency `freq_hz`."""
    n = int(round(length / dt))
    n += (n + 1) % 2                                    # force odd length
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * freq_hz * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float32)


def convolve_z(reflect: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    """1-D convolution down the depth axis (axis 0 of a [z, y, x] volume)."""
    from scipy.ndimage import convolve1d
    return convolve1d(reflect, wavelet, axis=0, mode="reflect").astype(np.float32)


def add_noise(seis: np.ndarray, snr_db: float,
              rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise at the requested SNR.

    Section IV-B-3 of the paper uses 5-22 dB for the synthetic test image.
    """
    p_sig = float(np.mean(seis.astype(np.float64) ** 2))
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    return (seis + rng.normal(0.0, np.sqrt(p_noise), seis.shape)).astype(np.float32)


def normalize(seis: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance, as the paper does before training."""
    m, s = float(seis.mean()), float(seis.std())
    return ((seis - m) / (s + 1e-8)).astype(np.float32)
