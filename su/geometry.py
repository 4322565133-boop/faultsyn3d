"""Fault surface geometry — Su et al. (2026) IEEE TGRS, Section III-A.

Equation numbers refer to:
  Su, Zhang, Cai, Zhou, Yao, Hu, "A Fault Network Synthesis Optimization Model
  for Automatic Generation of Seismic Fault Datasets With Diverse Geological
  Patterns", IEEE TGRS vol. 64, 2026, DOI 10.1109/TGRS.2026.3699734.

Conventions used here
---------------------
Global frame (x, y, z): z is depth / time-sample index, increasing downward.
Voxel grid is specified as (nx, ny, nz) following the paper (Fig. 1), but
volumes are stored as numpy arrays indexed [z, y, x] (seismic convention).

Local frame (X, Y, Z) of a fault (Eq. 12-13):
  X  along strike
  Y  down dip (in plane)
  Z  plane normal;  the reference plane is Z = 0.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Local <-> global frame  (Eq. 12, 13)
# --------------------------------------------------------------------------

def rotation_matrix(strike_deg: float, dip_deg: float) -> np.ndarray:
    """Rotation matrix R of Eq. (13).

    Rows are the local X (strike), Y (dip) and Z (normal) axes expressed in
    global coordinates, so  local = R @ (global - centre).
    """
    phi = np.deg2rad(strike_deg)
    theta = np.deg2rad(dip_deg)
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([
        [sp,       -cp,       0.0],
        [cp * ct,   sp * ct,  st],
        [cp * st,   sp * st, -ct],
    ], dtype=np.float64)


def to_local(points_xyz: np.ndarray, centre: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Eq. (12).  points_xyz is (..., 3) in global coordinates."""
    return (points_xyz - centre) @ R.T


def to_global(points_XYZ: np.ndarray, centre: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Inverse of Eq. (12)."""
    return points_XYZ @ R + centre


# --------------------------------------------------------------------------
# Branch-fault orientation derived from three points  (Eq. 1, 2, 3)
# --------------------------------------------------------------------------

def normal_from_points(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Unit normal of the plane through three non-collinear points — Eq. (1)."""
    n = np.cross(P1 - P0, P2 - P0)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Degenerate (collinear) triple of points")
    return n / norm


def dip_strike_from_normal(n: np.ndarray) -> tuple[float, float]:
    """Dip theta and strike phi from a unit normal — Eq. (2), (3).

    Returned in degrees, wrapped to [0, 180) as the paper requires.
    """
    # Invert the actual Eq. 13 frame. The printed Eq. 2 uses ny rather
    # than the horizontal normal magnitude and fails away from phi=90 deg.
    # Canonicalise phi to [0,180); theta may exceed 90 to retain dip sense.
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    phi = float(np.degrees(np.arctan2(n[1], n[0])) % 360)
    if phi >= 180:
        n = -n
        phi -= 180
    theta = float(np.degrees(np.arctan2(np.hypot(n[0], n[1]), -n[2])))
    return theta, phi


# --------------------------------------------------------------------------
# Quadratic Bezier curve  (Eq. 4-9)
# --------------------------------------------------------------------------

def bezier_control_point(P0: np.ndarray, P2: np.ndarray,
                         alpha: float, beta: float) -> np.ndarray:
    """Control point P1 from (alpha, beta) — Eq. (5)-(9).

    alpha slides P1 along P0->P2; beta offsets it perpendicular to P0->P2.
    The auxiliary basis vector u (Eq. 8) is chosen along the axis in which the
    segment is shortest (Eq. 9), which keeps the cross product well conditioned.
    """
    v = P2 - P0                                             # Eq. (6)
    # Treat roundoff-level ties identically after a coordinate roundtrip.
    av = np.abs(v)
    tie_tol = 1e-12 * max(1., float(np.linalg.norm(v)))
    dim = int(np.flatnonzero(av <= av.min() + tie_tol)[0])   # Eq. (9)
    u = np.zeros(3)
    u[dim] = 1.0                                            # Eq. (8)
    w = np.cross(v, u)
    nw = np.linalg.norm(w)
    if nw < 1e-12:                       # v parallel to u: fall back to another axis
        u = np.zeros(3)
        u[(dim + 1) % 3] = 1.0
        w = np.cross(v, u)
        nw = np.linalg.norm(w)
    v_perp = w / nw                                         # Eq. (7)
    return P0 + 0.5 * alpha * v + beta * np.linalg.norm(v) * v_perp   # Eq. (5)


def bezier_curve(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray,
                 t: np.ndarray) -> np.ndarray:
    """Quadratic Bezier — Eq. (4).  Returns (len(t), 3)."""
    t = np.asarray(t, dtype=np.float64)[:, None]
    return (1 - t) ** 2 * P0 + 2 * t * (1 - t) * P1 + t ** 2 * P2


def bezier_curve_nth(control_pts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Bezier curve of arbitrary degree (de Casteljau).

    Not part of Su et al. — the paper uses quadratic curves only and lists
    higher-order surfaces (for S-shaped faults) as future work in Section V.
    Kept here so the same code path can serve the extension later.
    """
    pts = np.asarray(control_pts, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[:, None]
    cur = np.repeat(pts[:, None, :], len(t), axis=1)      # (n_ctrl, n_t, 3)
    while cur.shape[0] > 1:
        cur = (1 - t) * cur[:-1] + t * cur[1:]
    return cur[0]


# --------------------------------------------------------------------------
# Reference plane corners  (Fig. 4a)
# --------------------------------------------------------------------------

def reference_plane_corners(centre: np.ndarray, R: np.ndarray,
                            half_len: float, half_wid: float) -> dict:
    """The four boundary points of the reference plane, labelled as in Fig. 4(a).

    G_top are the two corners with the smallest global z (shallowest); within
    each layer G0 is the corner closer to the global origin and G2 the farther.
    """
    local = np.array([
        [-half_len, -half_wid, 0.0],
        [+half_len, -half_wid, 0.0],
        [-half_len, +half_wid, 0.0],
        [+half_len, +half_wid, 0.0],
    ])
    glob = to_global(local, centre, R)

    order = np.argsort(glob[:, 2])          # by depth
    top, bottom = glob[order[:2]], glob[order[2:]]

    def split(pair):
        d = np.linalg.norm(pair, axis=1)
        return (pair[0], pair[1]) if d[0] <= d[1] else (pair[1], pair[0])

    g0_top, g2_top = split(top)
    g0_bot, g2_bot = split(bottom)
    return {"G0_top": g0_top, "G2_top": g2_top,
            "G0_bottom": g0_bot, "G2_bottom": g2_bot}


# --------------------------------------------------------------------------
# Bezier fault surface  (Fig. 4b-d, Eq. 10, 11)
# --------------------------------------------------------------------------

def build_fault_surface(centre: np.ndarray, R: np.ndarray,
                        half_len: float, half_wid: float,
                        b_strike: dict, b_dip: dict,
                        n_u: int = 64, n_v: int = 64,
                        n_perturb: int = 0, eps_perturb: float = 0.0,
                        rng: np.random.Generator | None = None) -> dict:
    """Build the Bezier fault surface described in Fig. 4.

    Parameters
    ----------
    b_strike : {"alpha_top", "beta_top", "alpha_bottom", "beta_bottom"}
        B_strike of Eq. (10) — curvature of the two guide curves, i.e. the
        along-strike bending of the surface.
    b_dip : {"alpha": (m,), "beta": (m,)}
        B_dip of Eq. (11) — curvature of the m profile curves, i.e. the
        down-dip bending.  A single-signed beta profile here is what produces
        a listric (concave-up) fault.

    Returns a dict with the surface point grid, its parametric coordinates and
    the per-point unit normals.
    """
    rng = rng or np.random.default_rng()
    G = reference_plane_corners(centre, R, half_len, half_wid)

    # --- guide curves along strike, Eq. (10) / Fig. 4(b) -------------------
    u = np.linspace(0.0, 1.0, n_u)
    ctrl_top = bezier_control_point(G["G0_top"], G["G2_top"],
                                    b_strike["alpha_top"], b_strike["beta_top"])
    ctrl_bot = bezier_control_point(G["G0_bottom"], G["G2_bottom"],
                                    b_strike["alpha_bottom"], b_strike["beta_bottom"])
    g_top = bezier_curve(G["G0_top"], ctrl_top, G["G2_top"], u)
    g_bottom = bezier_curve(G["G0_bottom"], ctrl_bot, G["G2_bottom"], u)

    # --- profile curves along dip, Eq. (11) / Fig. 4(c) -------------------
    # The paper defines m profile curves; we interpolate their (alpha, beta)
    # onto the n_u guide samples so the surface is a full tensor-product grid.
    m = len(b_dip["alpha"])
    src = np.linspace(0.0, 1.0, m)
    alpha_k = np.interp(u, src, np.asarray(b_dip["alpha"], dtype=float))
    beta_k = np.interp(u, src, np.asarray(b_dip["beta"], dtype=float))

    v = np.linspace(0.0, 1.0, n_v)
    surf = np.empty((n_u, n_v, 3), dtype=np.float64)
    for i in range(n_u):
        ctrl = bezier_control_point(g_top[i], g_bottom[i], alpha_k[i], beta_k[i])
        surf[i] = bezier_curve(g_top[i], ctrl, g_bottom[i], v)

    # --- perturbation along local normals, Fig. 4(d) ----------------------
    normals = _surface_normals(surf)
    if n_perturb > 0 and eps_perturb > 0:
        flat = surf.reshape(-1, 3)
        nflat = normals.reshape(-1, 3)
        idx = rng.choice(len(flat), size=min(n_perturb, len(flat)), replace=False)
        amp = rng.uniform(-eps_perturb, eps_perturb, size=len(idx))
        # Smooth the perturbation over the parametric grid so the surface stays
        # differentiable; a raw per-point kick would alias at voxel resolution.
        field = np.zeros(len(flat))
        field[idx] = amp
        field = field.reshape(n_u, n_v)
        field = _smooth2d(field, sigma=max(1.0, n_u / 32.0))
        surf = surf + normals * field[..., None]
        normals = _surface_normals(surf)

    uu, vv = np.meshgrid(u, v, indexing="ij")
    return {"points": surf, "normals": normals, "u": uu, "v": vv,
            "corners": G, "g_top": g_top, "g_bottom": g_bottom}


def _surface_normals(surf: np.ndarray) -> np.ndarray:
    """Unit normals of a parametric surface grid via central differences."""
    du = np.gradient(surf, axis=0)
    dv = np.gradient(surf, axis=1)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


def _smooth2d(a: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, sigma=sigma, mode="nearest")
