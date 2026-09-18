"""Fault network synthesis optimization — Su et al. (2026), Section III-B.

This is the heart of the paper.  Branch-fault positions are encoded as integer
offsets zeta along three axes normal to the main fault's reference plane; the
prescribed edge topology becomes a set of inequality constraints, those become
a quadratic penalty, and PSO minimises the total penalty to zero.

The three points that define a branch fault (and therefore ITS DIP AND STRIKE,
via Eq. 1-3) are recovered from zeta by Eq. (52).  Dip is an *output* of the
optimization, never a hand-set parameter — this is the property that makes the
generated geometry diverse.
"""

from __future__ import annotations

import numpy as np

from .geometry import rotation_matrix, to_local
from .topology import FaultTree


# --------------------------------------------------------------------------
# Axes  (Eq. 34-36)
# --------------------------------------------------------------------------

def axis_origins(centre: np.ndarray, R: np.ndarray, corners: dict) -> np.ndarray:
    """O_0, O_1, O_2 in the local frame — Eq. (34).

    Anchored at G0_top, G2_top and G0_bottom respectively.  Axis zeta_0 is the
    one that carries the combined strike+dip influence, and is the axis on
    which node positions must stay distinct (Eq. 39).
    """
    pts = np.stack([corners["G0_top"], corners["G2_top"], corners["G0_bottom"]])
    return to_local(pts, centre, R)          # (3, 3): rows are O_0, O_1, O_2


def axis_bounds(grid_shape_xyz, centre: np.ndarray, R: np.ndarray,
                origins: np.ndarray, d_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer bounds on zeta_k — Eq. (35), (36).

    Z in the paper is the set of local normal-direction coordinates spanned by
    the modelling domain; we take it from the eight corners of the voxel grid.

    NOTE ON FIDELITY.  As printed, Eq. (35) uses min(Z) and Eq. (36) uses
    max(Z), which makes l_k > m_k even though the text calls l_k the minimum.
    We therefore order the two bounds explicitly rather than trusting the
    printed labels.
    """
    nx, ny, nz = grid_shape_xyz
    cx, cy, cz = np.meshgrid([0, nx - 1], [0, ny - 1], [0, nz - 1], indexing="ij")
    corners_glob = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1).astype(float)
    Zloc = to_local(corners_glob, centre, R)[:, 2]

    a = (origins[:, 2] - Zloc.min()) / d_z
    b = (origins[:, 2] - Zloc.max()) / d_z
    lo = np.floor(np.minimum(a, b)).astype(int)
    hi = np.ceil(np.maximum(a, b)).astype(int)
    return lo, hi


def zeta_to_points(zeta: np.ndarray, centre: np.ndarray, R: np.ndarray,
                   origins: np.ndarray, d_z: float) -> np.ndarray:
    """Eq. (52): the three global points P_0, P_1, P_2 of one branch fault.

    zeta is (3,) — the node's discrete offsets along axes 0, 1, 2.
    """
    local = origins.copy().astype(float)          # rows O_0, O_1, O_2
    local[:, 2] = local[:, 2] + np.asarray(zeta, dtype=float) * d_z
    return local @ R + centre                     # inverse of Eq. (12)


# --------------------------------------------------------------------------
# Edge penalties  (Eq. 43-48)
# --------------------------------------------------------------------------

def _edge_penalty(d0: float, dk: float, attr: str) -> float:
    """e_X, e_Y, e_P of Eq. (46)-(48); constraints are Eq. (43)-(45)."""
    prod = d0 * dk
    if attr == "X":
        ok = prod < 0
    elif attr == "Y":
        ok = prod <= 0
    else:                                     # "P"
        ok = prod > 0
    return 0.0 if ok else float(d0 * d0 + dk * dk)


# --------------------------------------------------------------------------
# Algorithm 1 — pairwise penalty h_k^{i,j}
# --------------------------------------------------------------------------

def _pair_penalty(tree: FaultTree, k: int, i: int, j: int,
                  z: np.ndarray) -> float:
    """Algorithm 1.

    `k` is 1 (strike subtree) or 2 (dip subtree); the relevant edge attribute
    is in_edge[k-1].  `z` is the (Nf, 3) array of zeta values.

    NOTE ON FIDELITY.  Line 22 as printed reads
        min(zeta_0^j, zeta_0^{p_i}) < zeta_0^i < max(zeta_0^i, zeta_0^{p_i})
    which mixes i and j inconsistently with the symmetric test on line 25.
    We implement the symmetric reading: Case 3 asks whether one node falls
    inside the other node's Y-junction span.
    """
    ni, nj = tree.nodes[i], tree.nodes[j]
    p_i, p_j = ni.parent_id, nj.parent_id
    e_i = ni.in_edge[k - 1] if ni.in_edge else None
    e_j = nj.in_edge[k - 1] if nj.in_edge else None

    d0 = z[i, 0] - z[j, 0]                                        # Eq. (41)
    dk = z[i, k] - z[j, k]
    if d0 == 0:
        return 1e6                       # violates Eq. (39); rho would diverge
    # Separate directional subtrees carry a P relation; their Y-parent spans
    # are local to their own component and cannot waive cross-component P.
    if hasattr(tree, 'components') and tree.components[i] != tree.components[j]:
        return _edge_penalty(d0, dk, 'P')
    rho_ij = dk / d0                                              # Eq. (42)

    # ---- Case 1: i is the parent of j (lines 3-10) --------------------
    if p_j == i:
        t = 0.0
        if p_i is not None and ("Y" in (e_i, e_j)):
            d0_i_pi = z[i, 0] - z[p_i, 0]
            d0_j_pi = z[j, 0] - z[p_i, 0]
            if d0_i_pi * d0_j_pi <= 0:
                t += d0 * d0
            dk_i_pi = z[i, k] - z[p_i, k]
            rho_i_pi = dk_i_pi / d0_i_pi if d0_i_pi != 0 else np.inf
            if rho_ij < rho_i_pi:
                t += dk * dk
        if e_j == "X":
            return t + _edge_penalty(d0, dk, "X")
        if e_j == "Y":
            return t + _edge_penalty(d0, dk, "Y")
        return t + _edge_penalty(d0, dk, "P")

    # ---- Case 2: siblings (lines 11-19) -------------------------------
    if p_i is not None and p_i == p_j:
        d0_i_p = z[i, 0] - z[p_i, 0]
        d0_j_p = z[j, 0] - z[p_i, 0]
        if d0_i_p * d0_j_p <= 0:
            return _edge_penalty(d0, dk, "P")
        dk_i_p = z[i, k] - z[p_i, k]
        dk_j_p = z[j, k] - z[p_i, k]
        rho_p_i = dk_i_p / d0_i_p if d0_i_p != 0 else np.inf
        rho_p_j = dk_j_p / d0_j_p if d0_j_p != 0 else np.inf
        if e_i == "X" and e_j == "Y":
            return dk * dk if rho_p_i > rho_p_j else 0.0
        if e_i == "Y" and e_j == "X":
            return dk * dk if rho_p_i < rho_p_j else 0.0
        if e_i == "X" and e_j == "X":
            return _edge_penalty(d0, dk, "P")
        return 0.0

    # ---- Case 3: everything else (lines 20-29) ------------------------
    if e_i == "Y" and p_i is not None:
        lo, hi = sorted((z[i, 0], z[p_i, 0]))
        return d0 * d0 if lo < z[j, 0] < hi else 0.0
    if e_j == "Y" and p_j is not None:
        lo, hi = sorted((z[j, 0], z[p_j, 0]))
        return d0 * d0 if lo < z[i, 0] < hi else 0.0
    return _edge_penalty(d0, dk, "P")


def decode(zeta_flat: np.ndarray, n_f: int,
           lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Round a particle to integers and repair Eq. (39).

    Eq. (39) requires all zeta_0 to be distinct — it is what keeps Delta_0
    non-zero so that rho of Eq. (42) is defined.  Treating it as a soft penalty
    makes the landscape spiky (rounding produces collisions constantly), so we
    enforce it as a repair operator instead: colliding nodes are pushed to the
    nearest free integer.  Eq. (40) pins the root at zero.
    """
    z = np.zeros((n_f, 3))
    z[1:] = np.rint(zeta_flat.reshape(n_f - 1, 3))
    z[1:, 0] = np.clip(z[1:, 0], lo[0], hi[0])
    z[1:, 1] = np.clip(z[1:, 1], lo[1], hi[1])
    z[1:, 2] = np.clip(z[1:, 2], lo[2], hi[2])

    taken = {0.0}
    for i in range(1, n_f):
        v = z[i, 0]
        if v not in taken:
            taken.add(v)
            continue
        for step in range(1, int(hi[0] - lo[0]) + 2):
            for cand in (v + step, v - step):
                if lo[0] <= cand <= hi[0] and cand not in taken:
                    z[i, 0] = cand
                    break
            else:
                continue
            break
        taken.add(z[i, 0])
    return z


def objective(tree: FaultTree, z: np.ndarray, n_f: int) -> float:
    """F(s) of Eq. (49), evaluated on an already-decoded solution."""
    total = 0.0
    for k in (1, 2):
        forest = tree.direction_forest(k)
        for i in range(n_f - 1):
            for j in range(i + 1, n_f):
                total += _pair_penalty(forest, k, i, j, z)
    return total


def objective_batch(tree, z):
    """Vectorised Algorithm 1; z has shape (particles, faults, 3)."""
    total = np.zeros(len(z))
    def ep(d0, dk, attr):
        prod = d0*dk
        ok = prod < 0 if attr == 'X' else prod <= 0 if attr == 'Y' else prod > 0
        return np.where(ok, 0., d0*d0 + dk*dk)
    for k in (1, 2):
        f = tree.direction_forest(k)
        for i in range(z.shape[1]-1):
            for j in range(i+1, z.shape[1]):
                ni, nj = f.nodes[i], f.nodes[j]
                pi, pj = ni.parent_id, nj.parent_id
                ei = ni.in_edge[k-1] if ni.in_edge else None
                ej = nj.in_edge[k-1] if nj.in_edge else None
                d0, dk = z[:, i, 0]-z[:, j, 0], z[:, i, k]-z[:, j, k]
                safe = np.where(d0 == 0, 1., d0)
                if f.components[i] != f.components[j]:
                    h = ep(d0, dk, 'P')
                elif pj == i:
                    h = ep(d0, dk, ej)
                    if pi is not None and 'Y' in (ei, ej):
                        a = z[:, i, 0]-z[:, pi, 0]
                        b = z[:, j, 0]-z[:, pi, 0]
                        h += np.where(a*b <= 0, d0*d0, 0.)
                        rho = (z[:, i, k]-z[:, pi, k])/np.where(a == 0, 1., a)
                        h += np.where(dk/safe < rho, dk*dk, 0.)
                elif pi is not None and pi == pj:
                    a, b = z[:, i, 0]-z[:, pi, 0], z[:, j, 0]-z[:, pi, 0]
                    ra = (z[:, i, k]-z[:, pi, k])/np.where(a == 0, 1., a)
                    rb = (z[:, j, k]-z[:, pi, k])/np.where(b == 0, 1., b)
                    h = np.zeros(len(z))
                    if ei == 'X' and ej == 'Y': h = np.where(ra > rb, dk*dk, 0.)
                    elif ei == 'Y' and ej == 'X': h = np.where(ra < rb, dk*dk, 0.)
                    elif ei == 'X' and ej == 'X': h = ep(d0, dk, 'P')
                    h = np.where(a*b <= 0, ep(d0, dk, 'P'), h)
                elif ei == 'Y' and pi is not None:
                    a, b, c = z[:, i, 0], z[:, pi, 0], z[:, j, 0]
                    h = np.where((np.minimum(a,b)<c)&(c<np.maximum(a,b)), d0*d0, 0.)
                elif ej == 'Y' and pj is not None:
                    a, b, c = z[:, j, 0], z[:, pj, 0], z[:, i, 0]
                    h = np.where((np.minimum(a,b)<c)&(c<np.maximum(a,b)), d0*d0, 0.)
                else:
                    h = ep(d0, dk, 'P')
                total += np.where(d0 == 0, 1e6, h)
    return total


# --------------------------------------------------------------------------
# PSO  (Eq. 50, 51)
# --------------------------------------------------------------------------

def solve_pso(tree: FaultTree, lo: np.ndarray, hi: np.ndarray,
              rng: np.random.Generator,
              n_particles: int = 200, max_iter: int = 200,
              omega: float = 0.5, c1: float = 2.0, c2: float = 2.0,
              geo_fn=None, patience: int = 0) -> tuple[np.ndarray, dict]:
    """Minimise F(s) with particle swarm optimization.

    Paper settings (Section IV-A-1): N_par = 200, iter_max = 200, omega = 0.5,
    c1 = c2 = 2, terminating early when the penalty reaches zero.

    `geo_fn` is OUR extension (Module 1, `geo_objective.py`): an optional
    callable taking the decoded (n_f, 3) solution and returning a geometric
    penalty that is added to Su's topology penalty.  The two are tracked
    separately so we can confirm the topology term still reaches zero; early
    stopping requires BOTH to vanish.

    `patience` (ours) stops the swarm after that many iterations without any
    improvement of the global best.  With Su's topology-only penalty this is
    moot -- it reaches exactly zero and the paper's own stop rule fires.  The
    geometry terms are soft hinges that seldom reach exactly zero, so without
    it every geometry run consumed the full 200 iterations while the best
    solution had stopped changing by iteration ~40.  0 disables it.
    """
    n_f = len(tree)
    n_child = n_f - 1
    if n_child == 0:
        return np.zeros((0, 3)), {"penalty": 0.0, "iterations": 0}

    dim = 3 * n_child
    low = np.tile(lo.astype(float), n_child)
    high = np.tile(hi.astype(float), n_child)
    span = np.maximum(high - low, 1.0)

    pos = rng.uniform(low, high, size=(n_particles, dim))
    vel = rng.uniform(-span, span, size=(n_particles, dim)) * 0.1

    def parts(p):
        z = decode(p, n_f, lo, hi)
        topo = objective(tree, z, n_f)
        geo = float(geo_fn(z)) if geo_fn is not None else 0.0
        return topo, geo

    def score_all(p_all):
        z = np.stack([decode(p, n_f, lo, hi) for p in p_all])
        scores = objective_batch(tree, z)
        if geo_fn is not None:
            scores += np.array([geo_fn(v) for v in z])
        return scores

    scores = score_all(pos)
    pbest, pbest_score = pos.copy(), scores.copy()
    g = int(np.argmin(pbest_score))
    gbest, gbest_score = pbest[g].copy(), float(pbest_score[g])

    history = [gbest_score]
    iters = 0
    stale = 0
    for it in range(max_iter):
        iters = it + 1
        if gbest_score <= 0.0:
            break
        if patience and stale >= patience:
            break
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        vel = (omega * vel                                   # Eq. (50)
               + c1 * r1 * (pbest - pos)
               + c2 * r2 * (gbest[None, :] - pos))
        pos = np.clip(pos + vel, low, high)                  # Eq. (51)

        scores = score_all(pos)
        better = scores < pbest_score
        pbest[better], pbest_score[better] = pos[better], scores[better]
        g = int(np.argmin(pbest_score))
        if pbest_score[g] < gbest_score - 1e-9:
            gbest, gbest_score = pbest[g].copy(), float(pbest_score[g])
            stale = 0
        else:
            stale += 1
        history.append(gbest_score)

    zeta = decode(gbest, n_f, lo, hi)[1:].astype(int)
    topo, geo = parts(gbest)
    return zeta, {"penalty": gbest_score, "topology_penalty": topo,
                  "geometry_penalty": geo, "iterations": iters,
                  "particles": n_particles, "omega": omega, "c1": c1, "c2": c2,
                  "history": history}
