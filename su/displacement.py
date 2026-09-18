"""Fault displacement, drag and slip scaling — Su et al. (2026), Section III-A-3.

Two displacement models (Eq. 14-16):

* throughgoing faults        idealised elliptical field of Wu et al. (2019)
* non-throughgoing faults    separable Hermite-spline field, ksi = Dmax*Ds*Dd

The second is the paper's contribution here: because the profiles are defined
on the fault's own parametric grid they decay to exactly zero at the tips, which
removes the "spurious slip" artefact shown in Fig. 5 that the elliptical model
leaves on faults terminating inside the domain.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Hermite profiles  (Fig. 6a, 6b)
# --------------------------------------------------------------------------

def hermite_bump(t: np.ndarray) -> np.ndarray:
    """Smooth bump on [-1, 1]: 1 at the centre, 0 with zero slope at both tips.

    h(t) = 1 - 3t^2 + 2|t|^3 — the cubic Hermite basis, matching the shape of
    Ds and Dd plotted in Fig. 6(a) and 6(b).
    """
    a = np.clip(np.abs(np.asarray(t, dtype=float)), 0.0, 1.0)
    return 1.0 - 3.0 * a ** 2 + 2.0 * a ** 3


def surface_slip_field(n_u: int, n_v: int, d_max: float) -> np.ndarray:
    """ksi^(u,v) of Eq. (16) on the fault's (n_u, n_v) parametric grid.

    u runs along strike, v down dip; both are linearly normalised to [-1, 1]
    exactly as the paper prescribes for X^(u) and Y^(v).
    """
    xbar = np.linspace(-1.0, 1.0, n_u)
    ybar = np.linspace(-1.0, 1.0, n_v)
    return d_max * np.outer(hermite_bump(xbar), hermite_bump(ybar))


def surface_slip_field_elliptic(n_u: int, n_v: int, d_max: float) -> np.ndarray:
    """Elliptical field of Eq. (14)-(15), for throughgoing faults.

    Written on the same parametric grid, with L_X and L_Y taken as the half
    extents so that r is the normalised radial distance of Eq. (15).
    """
    X = np.linspace(-1.0, 1.0, n_u)[:, None]
    Y = np.linspace(-1.0, 1.0, n_v)[None, :]
    r = np.sqrt(X ** 2 + Y ** 2)
    r = np.clip(r, 0.0, 1.0)
    inner = np.maximum((1.0 + r) ** 2 / 4.0 - r ** 2, 0.0)
    return 2.0 * d_max * (1.0 - r) * np.sqrt(inner)


# --------------------------------------------------------------------------
# Slip decomposition  (Eq. 17-21)
# --------------------------------------------------------------------------

def decompose_slip(ksi: np.ndarray, phi_dis_deg: float) -> np.ndarray:
    """D_X, D_Y, D_Z in the local frame — Eq. (17)-(19).

    phi_dis = 0 deg is pure dip-slip, 90 deg is pure strike-slip.  D_Z is zero
    by construction; the out-of-plane motion is supplied later by the drag
    operator (Eq. 24-26).
    """
    p = np.deg2rad(phi_dis_deg)
    return np.stack([ksi * np.sin(p), ksi * np.cos(p), np.zeros_like(ksi)], axis=-1)


def slip_to_global(d_loc: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Eq. (20)-(21):  D_glob = R^T D_loc."""
    return d_loc @ R


# --------------------------------------------------------------------------
# Drag operator  (Eq. 22-26, Fig. 7)
# --------------------------------------------------------------------------

def drag_operator(d_norm: np.ndarray, mode: str = "normal",
                  asymmetry: float = 0.0) -> np.ndarray:
    """D_f applied to the signed, normalised distance field — Eq. (23).

    d_norm is the signed distance to the fault surface rescaled to [-1, 1],
    so |d_norm| = 1 is the outer limit of the drag zone.

    NOTE ON FIDELITY.  Fig. 7 plots the two operators but the paper gives no
    closed form for the splines.  The shapes below reproduce the plotted
    behaviour: `normal` drag carries full slip at the fault and decays
    monotonically outwards (antisymmetric across the surface); `reverse` drag
    additionally rolls the hanging wall over, i.e. the profile overshoots
    before decaying, which is what produces rollover anticlines above listric
    faults.
    """
    s = np.clip(np.asarray(d_norm, dtype=float), -1.0, 1.0)
    a = np.abs(s)
    side = np.sign(s)

    if mode == "normal":
        # Fig. 7(a): nonzero one-sided limits, near-field overshoot,
        # zero value and slope at the outer boundary. Cubic Hermite segment.
        prof = hermite_bump(a) + 2.0 * a * (1.0-a)**2
    elif mode == "reverse":
        # Fig. 7(b): opposite sense; maximum magnitude at the fault.
        prof = -hermite_bump(a)
    else:
        raise ValueError(f"unknown drag mode {mode!r}")

    if asymmetry:
        prof = prof * np.where(side > 0, 1.0 + asymmetry, 1.0 - asymmetry)
    return side * prof


# --------------------------------------------------------------------------
# Slip scaling law  (Eq. 27-29)
# --------------------------------------------------------------------------

def scale_branch_dmax(d_max_root: float, n_points_i: int, n_points_root: int) -> float:
    """Eq. (28):  Dmax_i = Dmax_0 * sqrt(N_i / N_0).

    Follows from Leonard's D ~ C sqrt(A) fault scaling (Eq. 27) together with
    A_i proportional to the voxel count N_i occupied by the fault surface.
    This is what keeps branch faults kinematically consistent with the main
    fault instead of independently randomised.
    """
    if n_points_root <= 0:
        return d_max_root
    return float(d_max_root * np.sqrt(max(n_points_i, 1) / n_points_root))
