"""Algorithm 4 — convex-hull constraint on the Bezier parameter domain.

Su et al. (2026), Section III-D, Eq. (53), Fig. 11.

Positioning the faults (Section III-B) only guarantees that the *reference
planes* respect the prescribed topology.  Once the planes are bent into Bezier
surfaces the bending can reintroduce intersections that the topology forbids
(Fig. 11b).  Algorithm 4 removes them in parameter space: by the convex-hull
property of Bezier curves, every point of a curve lies inside the hull of its
three control points, so if two such hulls are disjoint the curves cannot meet.

For node i the admissible domain starts as the full rectangle
(alpha, beta) in [0,1] x [-1,1] and is intersected, for every earlier node j
that is NOT connected to i by an X or Y edge in this direction, with the set of
(alpha_i, beta_i) whose hull misses j's hull for EVERY (alpha_j, beta_j) still
admissible for j.  That "for every" is what makes the result independent of the
order in which the parameters are finally drawn.

NOTE ON FIDELITY.  Exact hull-vs-hull disjointness needs the full separating
axis set (face normals of both bodies plus all edge-pair cross products).  We
test face normals only, which is a *sufficient* condition for disjointness:
some genuinely admissible parameter pairs are rejected, never the reverse.  The
domain is therefore slightly smaller than Algorithm 4's, which is the safe
direction — surfaces stay apart, and the cost is a little less curvature range.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .geometry import bezier_control_point, reference_plane_corners


# Which curve endpoints belong to which direction (Fig. 4a-c).
#   k = 1  strike direction: the two guide curves
#   k = 2  dip direction:    the two profile curves at the ends
_ENDPOINTS = {
    1: (("G0_top", "G2_top"), ("G0_bottom", "G2_bottom")),
    2: (("G0_top", "G0_bottom"), ("G2_top", "G2_bottom")),
}


def _hull_faces(points: np.ndarray):
    """Outward face planes (n, d) of conv(points), with n . x <= d inside.

    Falls back to a bounding-box hull for degenerate (coplanar) inputs.
    """
    pts = np.unique(np.round(points, 9), axis=0)
    if len(pts) < 4:
        return None
    try:
        h = ConvexHull(pts)
    except QhullError:
        return None
    # scipy stores each face as n . x + d <= 0
    return h.equations[:, :3], -h.equations[:, 3]


def _sweep_points(P0: np.ndarray, P2: np.ndarray,
                  alphas: np.ndarray, betas: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """All hull vertices node j can still reach: its endpoints plus every
    admissible control point.  Union of triangles over a convex parameter set
    equals the hull of the endpoints and the reachable control points."""
    ctrl = [bezier_control_point(P0, P2, float(a), float(b))
            for ia, a in enumerate(alphas)
            for ib, b in enumerate(betas) if mask[ia, ib]]
    if not ctrl:
        return np.stack([P0, P2])
    return np.vstack([P0[None, :], P2[None, :], np.asarray(ctrl)])


def _disjoint_mask(P0: np.ndarray, P2: np.ndarray,
                   alphas: np.ndarray, betas: np.ndarray,
                   other_pts: np.ndarray, clearance: float) -> np.ndarray:
    """Candidates whose triangle is provably separated from conv(other_pts).

    Two tests, both sufficient on their own:
      (a) a face plane of the other hull puts all three triangle vertices
          strictly outside it;
      (b) the triangle's own plane puts every vertex of the other hull
          strictly on one side.
    """
    na, nb = len(alphas), len(betas)
    ctrl = np.empty((na, nb, 3))
    for ia, a in enumerate(alphas):
        for ib, b in enumerate(betas):
            ctrl[ia, ib] = bezier_control_point(P0, P2, float(a), float(b))

    # tri[..., v, :] are the three triangle vertices per candidate
    tri = np.empty((na, nb, 3, 3))
    tri[..., 0, :] = P0
    tri[..., 1, :] = ctrl
    tri[..., 2, :] = P2

    ok = np.zeros((na, nb), dtype=bool)

    # (a) face normals of the other hull
    faces = _hull_faces(other_pts)
    if faces is not None:
        n_f, d_f = faces
        proj = np.einsum("abvi,fi->abvf", tri, n_f)          # (na, nb, 3, F)
        ok |= np.any(np.all(proj > (d_f + clearance), axis=2), axis=-1)

    # (b) the candidate triangle's own plane
    e1 = ctrl - P0
    e2 = P2 - P0
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=-1, keepdims=True)
    good = ln[..., 0] > 1e-9
    nrm = np.where(ln > 1e-9, nrm / np.maximum(ln, 1e-12), 0.0)
    off = np.einsum("abi,i->ab", nrm, P0)
    o_proj = np.einsum("abi,ji->abj", nrm, other_pts)         # (na, nb, M)
    sep = ((o_proj.min(axis=-1) > off + clearance) |
           (o_proj.max(axis=-1) < off - clearance))
    ok |= (sep & good)
    return ok


def constrain_bezier_domains(faults: list[dict], tree, cfg,
                             n_alpha: int = 13, n_beta: int = 25,
                             clearance: float = 1.5) -> dict:
    """Run Algorithm 4 for both subtrees.

    `faults` entries need "id", "centre", "R", "half_len", "half_wid",
    "in_edge" and "parent".  Returns {node_id: {k: (alphas, betas, mask)}}.
    """
    alphas = np.linspace(0.0, 1.0, n_alpha)
    betas = np.linspace(-1.0, 1.0, n_beta)

    corners = {f["id"]: reference_plane_corners(
        np.asarray(f["centre"], dtype=float), f["R"],
        f["half_len"], f["half_wid"]) for f in faults}

    order = [f["id"] for f in faults]
    by_id = {f["id"]: f for f in faults}
    out: dict[int, dict[int, tuple]] = {i: {} for i in order}

    for k in (1, 2):
        omega: dict[int, np.ndarray] = {}
        for i in order:
            mask = np.ones((n_alpha, n_beta), dtype=bool)
            fi = by_id[i]
            for j in order:
                if j == i:
                    break                      # only earlier nodes (line 5)
                # Algorithm 4 line 6 is exactly `if p_i != j` — the direct
                # parent is skipped regardless of edge type.  An earlier version
                # of this code only skipped parents joined by an X or Y edge,
                # which applied the constraint to strictly more pairs than the
                # paper does.
                if tree.direction_forest(k).nodes[i].parent_id == j:
                    continue

                # every endpoint pair of i against every endpoint pair of j
                acc = np.ones_like(mask)
                for ei in _ENDPOINTS[k]:
                    Pi0, Pi2 = corners[i][ei[0]], corners[i][ei[1]]
                    for ej_pair in _ENDPOINTS[k]:
                        Pj0, Pj2 = corners[j][ej_pair[0]], corners[j][ej_pair[1]]
                        pts_j = _sweep_points(Pj0, Pj2, alphas, betas, omega[j])
                        acc &= _disjoint_mask(Pi0, Pi2, alphas, betas,
                                              pts_j, clearance)
                mask &= acc
            if not mask.any():
                raise ValueError(f"empty Bezier domain: node={i}, direction={k}")
            omega[i] = mask
            out[i][k] = (alphas, betas, mask)
    return out


def project_to_domain(alpha: float, beta: float,
                      alphas: np.ndarray, betas: np.ndarray,
                      mask: np.ndarray) -> tuple[float, float]:
    """Nearest admissible (alpha, beta) to the one the pattern asked for.

    Keeps each category's geometric intent — a listric fault still gets the
    largest down-dip curvature it is allowed — while satisfying Eq. (53).
    """
    ia, ib = np.nonzero(mask)
    if len(ia) == 0:
        return alpha, beta
    da = (alphas[ia] - alpha) / max(float(np.ptp(alphas)), 1e-9)
    db = (betas[ib] - beta) / max(float(np.ptp(betas)), 1e-9)
    k = int(np.argmin(da ** 2 + db ** 2))
    return float(alphas[ia[k]]), float(betas[ib[k]])
