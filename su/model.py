"""Assemble one synthetic seismic / fault-label volume.

Pipeline (Su et al. 2026, Fig. 1):

  Step 1  fault network representation   -> topology.py
  Step 2  synthesis optimization (PSO)   -> optimize.py
  Step 3  geological attribute integration:
            Bezier surfaces              -> geometry.py
            slip + drag                  -> displacement.py
            stratigraphy + forward model -> stratigraphy.py

The single most important structural point: a branch fault's DIP AND STRIKE are
derived from the optimizer's solution (zeta -> three points -> Eq. 1-3), not
drawn from a hand-written range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

# cKDTree.query(workers=-1) grabs every core on the machine; with five
# generator processes in parallel that oversubscribed 96 cores 5x and roughly
# tripled the per-volume time.  Override with FAULTSYN_KD_WORKERS.
import os as _os
_KD_WORKERS = int(_os.environ.get("FAULTSYN_KD_WORKERS", "16"))

from . import stratigraphy as strat
from .displacement import (decompose_slip, drag_operator, scale_branch_dmax,
                           slip_to_global, surface_slip_field,
                           surface_slip_field_elliptic)
from .geometry import (build_fault_surface, dip_strike_from_normal,
                       normal_from_points, reference_plane_corners,
                       rotation_matrix)
from .geo_objective import (GeoContext, GeoWeights, crossing_pairs_from_tree,
                            disjoint_pairs_from_tree,
                            geometry_penalty)
from .hull_constraint import constrain_bezier_domains, project_to_domain
from .optimize import axis_bounds, axis_origins, solve_pso, zeta_to_points
from .topology import FaultTree


# --------------------------------------------------------------------------

@dataclass
class MainFault:
    """User-defined root attributes (Definition 3, case i = 0)."""
    strike_deg: float
    dip_deg: float
    centre: tuple[float, float, float]
    half_len: float
    half_wid: float
    d_max: float
    phi_dis_deg: float
    throughgoing: bool = False
    drag_mode: str = "normal"


@dataclass
class SurfaceParams:
    """G(i) of Eq. (32): the Bezier parameters plus the perturbation controls."""
    b_strike: dict
    b_dip: dict
    n_perturb: int = 0
    eps_perturb: float = 0.0


@dataclass
class GenConfig:
    grid: tuple[int, int, int] = (128, 128, 128)      # (nx, ny, nz)
    d_z: float = 4.0                  # zeta discretisation step, voxels
    # Surface sampling must stay finer than the voxel pitch or the rasterised
    # label breaks into fragments (128 gave 19 components on a 4-fault volume,
    # 192 gives the correct 3-4).
    n_u: int = 192                    # surface samples along strike
    n_v: int = 192                    # surface samples down dip
    drag_distance: float = 14.0       # voxels; sets the [-1,1] range of Eq. (23)
    label_half_thickness: float = 0.75
    # Su labels the whole fault SURFACE, whatever the slip on it.  Eq. (16)
    # tapers slip to zero at every tip and Eq. (28) halves it on branches, so a
    # large part of that surface moves the reflectors by less than the seismic
    # can resolve: on the reproduction 22-49 % of a main fault's area and
    # frequently 0 % of a branch's carried a vertical throw above lambda/4.
    # Those voxels are labelled "fault" on top of undisturbed reflectors.
    # With this on, the training label keeps only voxels whose ACTUAL vertical
    # offset across the surface (measured on the final displacement field, all
    # faults included) reaches `observable_frac` of the dominant wavelength;
    # 0.25 is the classical tuning thickness.  Su's full geometric label is
    # returned alongside as `label_full`, so nothing is discarded.
    observable_labels: bool = True
    observable_frac: float = 0.25
    # Definition 3: "Any one of these points serves as the reference point for
    # the branch fault."  Taken literally as "the branch is CENTRED on that
    # point", all three choices fail: O_0/O_1/O_2 sit on corners of the main
    # fault (G0_top, G2_top, G0_bottom), which straddle the domain boundary, so
    # a branch centred there falls almost entirely outside the grid.  Measured
    # fraction of branch surface inside a 128^3 volume:
    #     P0 14.5 %   P1 14.5 %   P2 20.9 %   centroid 79.3 %
    # and 8-15 % of branches vanish completely with P0/P1/P2 against 0 % with
    # the centroid.  Fig. 15 plainly shows branches contained in the volume, so
    # "reference point" must anchor the PLANE (supplying (x0,y0,z0) for Eq. 12)
    # rather than fix the branch's centre.  We centre on the centroid and record
    # the alternatives here so the choice stays auditable.
    branch_ref_point: str = "centroid"      # centroid | P0 | P1 | P2
    use_hull_constraint: bool = True   # Algorithm 4 / Eq. (53)
    hull_clearance: float = 1.5        # voxels of margin demanded between hulls
    # Branch extent is not specified anywhere in the paper.  Fig. 8, 10(e) and
    # 15 all show branches comparable in size to the main fault, filling the
    # box; our first choice of 0.45-0.85 left them visibly smaller than the
    # published figures.
    # In Fig. 15(b)-(e) the branches read as narrow slivers beside a broad main
    # fault: short along STRIKE but still spanning the full height in DIP.
    # Scaling both axes together, as we did first, turned them into slabs the
    # size of the main fault.
    branch_len_ratio: tuple[float, float] = (0.35, 0.60)
    branch_wid_ratio: tuple[float, float] = (0.85, 1.05)
    # Slide a branch inside its OWN plane until enough of it lies in the grid.
    # Also unspecified by the paper, and needed because the three axis origins
    # sit on corners of the main fault: a low-dip main fault (listric) puts them
    # near the domain boundary, and 35 % of listric branches ended up with less
    # than 30 % of their surface inside the volume.  Sliding in-plane leaves the
    # normal, dip, strike and therefore every topology relation untouched.
    # Kept for ablation but OFF: it maximised the fraction of a branch inside
    # the grid, which we now know is the wrong objective — the paper wants
    # faults to overrun the box so the volume samples their high-slip middle.
    # --- OURS, not Su et al.: post-bending repair (Step 3b) ---------------
    # This one is not an extra term in an objective, it is an extra STAGE in
    # the pipeline, so it stays off by default alongside the geometry objective
    # and the Su reproduction runs without it.  Leaving it on unconditionally
    # would have contaminated the baseline arm of every ablation.
    use_surface_repair: bool = False
    #: OURS: force every branch's Bezier curvature to a sign prescribed per
    #: pattern (patterns.su_patterns.BRANCH_CURV_SIGN).  The paper samples the
    #: control parameters of each curve independently inside the Algorithm-4
    #: domain, so this is off for the reproduction.
    inherit_branch_curvature: bool = False
    branch_recentre: bool = False
    branch_recentre_target: float = 0.55   # coverage floor before shrinking
    n_folds: tuple[int, int] = (3, 7)
    fold_amp: tuple[float, float] = (4.0, 22.0)
    fold_sigma: tuple[float, float] = (20.0, 70.0)
    tilt: float = 0.06                # slope coefficients a, b of Eq. (54)
    freq_hz: tuple[float, float] = (20.0, 34.0)
    dt: float = 0.004
    snr_db: tuple[float, float] = (5.0, 22.0)
    pso_particles: int = 200
    pso_iters: int = 200
    pso_patience: int = 30        # ours; only used with the geometry objective

    # --- Module 1 (ours, not Su et al.): geometry-aware objective ---------
    use_geometry_objective: bool = False
    geo_regime: str = "strike_slip"        # which Andersonian dip band applies
    geo_w_dip: float = 1.0
    geo_w_int: float = 2.0
    geo_w_dup: float = 3.0
    geo_w_conj: float = 2.0
    geo_w_present: float = 2.0
    geo_w_sep: float = 2.0
    #: (parent, child, "opp"|"same") edges, stamped on by the pattern module
    dip_relations: tuple = ()


# --------------------------------------------------------------------------

def _rasterise(surface_pts: np.ndarray, surface_nrm: np.ndarray,
               grid: tuple[int, int, int], reach: float):
    """Signed distance and nearest-surface-sample index for every voxel in reach.

    Returns (mask, signed_distance, flat_index) where `mask` selects the voxels
    that lie within `reach` of the surface and the other two arrays are given
    only on those voxels.  Volumes are indexed [z, y, x].
    """
    nx, ny, nz = grid
    flat = surface_pts.reshape(-1, 3)
    nflat = surface_nrm.reshape(-1, 3)

    lo = np.floor(flat.min(axis=0) - reach).astype(int)
    hi = np.ceil(flat.max(axis=0) + reach).astype(int)
    lo = np.maximum(lo, [0, 0, 0])
    hi = np.minimum(hi, [nx - 1, ny - 1, nz - 1])
    if np.any(hi < lo):
        return None

    xs = np.arange(lo[0], hi[0] + 1)
    ys = np.arange(lo[1], hi[1] + 1)
    zs = np.arange(lo[2], hi[2] + 1)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    query = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)

    tree = cKDTree(flat)
    dist, idx = tree.query(query, workers=_KD_WORKERS)

    keep = dist <= reach
    if not keep.any():
        return None
    query, dist, idx = query[keep], dist[keep], idx[keep]

    side = np.einsum("ij,ij->i", query - flat[idx], nflat[idx])
    signed = np.where(side >= 0, dist, -dist)

    vox = query.astype(int)
    return {"vox_xyz": vox, "signed": signed, "surf_idx": idx,
            "shape": (nz, ny, nx)}


def _inside_fraction(centre, R, hl, hw, grid, n=9):
    """Fraction of a fault rectangle's samples that fall inside the voxel grid."""
    nx, ny, nz = grid
    u = np.linspace(-hl, hl, n)
    v = np.linspace(-hw, hw, n)
    U, V = np.meshgrid(u, v, indexing="ij")
    loc = np.stack([U, V, np.zeros_like(U)], axis=-1).reshape(-1, 3)
    p = loc @ R + centre
    return float(((p[:, 0] >= 0) & (p[:, 0] <= nx - 1) &
                  (p[:, 1] >= 0) & (p[:, 1] <= ny - 1) &
                  (p[:, 2] >= 0) & (p[:, 2] <= nz - 1)).mean())


def _fit_into_grid(centre, normal, R, hl, hw, grid, floor=0.55, steps=12):
    """Place a branch so that enough of it lands in the voxel grid.

    Two moves, in order, neither of which touches the fault PLANE — so the
    normal, dip, strike and every topology relation the optimiser solved for
    are preserved exactly:

      1. slide the centre WITHIN the plane toward the domain centre's
         projection, taking whichever position maximises coverage;
      2. if coverage is still below `floor`, shrink the extent.

    Both are needed.  The three axis origins are pinned to corners of the main
    fault, and the main fault spans the box (measured full length 120-136 on a
    128 grid, with only 9-34 % of its corners inside), so a branch offset from
    those origins starts out badly placed.  Sliding alone was not enough: an
    earlier version that slid only when coverage fell below a target left the
    healthy categories untouched and made them worse once branches grew.
    """
    nx, ny, nz = grid
    gc = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0])
    proj = gc - normal * float((gc - centre) @ normal)      # stays in-plane

    best_c, best_f = centre, _inside_fraction(centre, R, hl, hw, grid)
    for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
        c = centre + t * (proj - centre)
        f = _inside_fraction(c, R, hl, hw, grid)
        if f > best_f:
            best_c, best_f = c, f
    centre = best_c

    scale = 1.0
    while best_f < floor and scale > 0.45:
        scale *= 0.9
        best_f = _inside_fraction(centre, R, hl * scale, hw * scale, grid)
    return centre, hl * scale, hw * scale


def _accumulate(vol: np.ndarray, vox_xyz: np.ndarray, values: np.ndarray):
    """Add `values` into a [z, y, x] volume at the given (x, y, z) voxels."""
    np.add.at(vol, (vox_xyz[:, 2], vox_xyz[:, 1], vox_xyz[:, 0]), values)


def generate_volume(tree: FaultTree, main: MainFault,
                    surf_params: dict[int, SurfaceParams],
                    cfg: GenConfig, rng: np.random.Generator,
                    branch_slip_mode: str = "hermite") -> dict:
    """Generate one (seismic, label) pair plus full provenance metadata."""
    nx, ny, nz = cfg.grid
    centre = np.asarray(main.centre, dtype=float)
    R = rotation_matrix(main.strike_deg, main.dip_deg)
    corners = reference_plane_corners(centre, R, main.half_len, main.half_wid)

    # ---- Step 2: optimise branch positions ---------------------------
    origins = axis_origins(centre, R, corners)
    lo, hi = axis_bounds(cfg.grid, centre, R, origins, cfg.d_z)

    # Branch extents do not depend on zeta, so they are drawn first: the
    # geometry objective needs them to estimate slip via Eq. (27)-(28).
    n_f = len(tree)
    hl_ratio = np.ones(n_f); hw_ratio = np.ones(n_f)
    for i in range(1, n_f):
        hl_ratio[i] = float(rng.uniform(*cfg.branch_len_ratio))
        hw_ratio[i] = float(rng.uniform(*cfg.branch_wid_ratio))
    hl_all = main.half_len * hl_ratio
    hw_all = main.half_wid * hw_ratio

    geo_fn = None
    geo_ctx = None
    if cfg.use_geometry_objective:
        # Analytic area proxy for Eq. (28): A is proportional to half_len*half_wid,
        # which avoids needing the rasterised surface before the surface exists.
        area = hl_all * hw_all
        d_max_proxy = main.d_max * np.sqrt(area / max(area[0], 1e-9))
        geo_ctx = GeoContext(
            centre=centre, R=R, origins=origins, d_z=cfg.d_z, grid=cfg.grid,
            ref_point=cfg.branch_ref_point,
            half_len=hl_all, half_wid=hw_all, d_max=d_max_proxy,
            regime=cfg.geo_regime,
            disjoint_pairs=disjoint_pairs_from_tree(tree),
            crossing_pairs=crossing_pairs_from_tree(tree),
            dip_relations=cfg.dip_relations)
        weights = GeoWeights(dip=cfg.geo_w_dip, intersect=cfg.geo_w_int,
                             duplicate=cfg.geo_w_dup, conj=cfg.geo_w_conj,
                             present=cfg.geo_w_present, separate=cfg.geo_w_sep)
        geo_fn = lambda z: geometry_penalty(z, geo_ctx, weights)[0]

    zeta, pso_info = solve_pso(tree, lo, hi, rng,
                               n_particles=cfg.pso_particles,
                               max_iter=cfg.pso_iters, geo_fn=geo_fn,
                               patience=cfg.pso_patience if geo_fn else 0)

    # ---- Step 3a-i: fault frames (no surfaces yet) ---------------------
    # Algorithm 4 needs every reference plane before any surface is bent, so
    # the frames are resolved in a first pass and the surfaces in a second.
    faults: list[dict] = []

    for i in range(n_f):
        if i == 0:
            f_centre, f_strike, f_dip = centre, main.strike_deg, main.dip_deg
            hl, hw = main.half_len, main.half_wid
        else:
            pts = zeta_to_points(zeta[i - 1], centre, R, origins, cfg.d_z)
            try:
                n_vec = normal_from_points(pts[0], pts[1], pts[2])   # Eq. (1)
            except ValueError:
                continue
            f_dip, f_strike = dip_strike_from_normal(n_vec)          # Eq. (2)(3)
            # Definition 3 restricts the reference point to {P0, P1, P2}; a
            # centroid is outside that set, so we take P0.  This matters: the
            # choice shifts a branch centre by ~70 voxels on average, more than
            # a fault's own half-length.  It leaves dip and strike untouched,
            # since those come from the normal of all three points (Eq. 1-3).
            f_centre = (pts.mean(axis=0) if cfg.branch_ref_point == "centroid"
                        else pts[int(cfg.branch_ref_point[1])])
            hl = main.half_len * float(hl_ratio[i])
            hw = main.half_wid * float(hw_ratio[i])
            if cfg.branch_recentre:
                f_centre, hl, hw = _fit_into_grid(
                    f_centre, n_vec, rotation_matrix(f_strike, f_dip),
                    hl, hw, cfg.grid, cfg.branch_recentre_target)


        faults.append({"id": i, "centre": f_centre, "strike": f_strike,
                       "dip": f_dip, "half_len": hl, "half_wid": hw,
                       "R": rotation_matrix(f_strike, f_dip),
                       "in_edge": tree.nodes[i].in_edge,
                       "parent": tree.nodes[i].parent_id})

    if not faults:
        raise RuntimeError("no valid fault surfaces were produced")

    # ---- Step 3a-ii: Algorithm 4, then build the surfaces --------------
    domains = None
    n_clamped = 0
    if cfg.use_hull_constraint and len(faults) > 1:
        domains = constrain_bezier_domains(faults, tree, cfg,
                                           clearance=cfg.hull_clearance)

    for f in faults:
        sp = surf_params[f["id"]]
        b_strike = dict(sp.b_strike)
        b_dip = {"alpha": np.asarray(sp.b_dip["alpha"], dtype=float).copy(),
                 "beta": np.asarray(sp.b_dip["beta"], dtype=float).copy()}

        if domains is not None:
            al1, be1, m1 = domains[f["id"]][1]
            for tag in ("top", "bottom"):
                a, b = project_to_domain(b_strike[f"alpha_{tag}"],
                                         b_strike[f"beta_{tag}"], al1, be1, m1)
                n_clamped += int(b != b_strike[f"beta_{tag}"])
                b_strike[f"alpha_{tag}"], b_strike[f"beta_{tag}"] = a, b

            al2, be2, m2 = domains[f["id"]][2]
            for q in range(len(b_dip["alpha"])):
                a, b = project_to_domain(float(b_dip["alpha"][q]),
                                         float(b_dip["beta"][q]), al2, be2, m2)
                n_clamped += int(b != b_dip["beta"][q])
                b_dip["alpha"][q], b_dip["beta"][q] = a, b

        f["surface"] = build_fault_surface(
            f["centre"], f["R"], f["half_len"], f["half_wid"],
            b_strike, b_dip, n_u=cfg.n_u, n_v=cfg.n_v,
            n_perturb=sp.n_perturb, eps_perturb=sp.eps_perturb, rng=rng)

    repairs = []
    if cfg.use_surface_repair:
        # ---- Step 3b: post-bending repair ---------------------------------
        # Su's pipeline optimises REFERENCE PLANES (Section III-B) and only then
        # bends them into Bezier surfaces (Section III-A-2), with no feedback from
        # the second stage to the first.  Algorithm 4 covers one direction of that
        # gap -- bending must not CREATE an intersection the topology forbids.
        # Nothing covers the other direction, and both halves of it bit us:
        #   * bending destroyed a required crossing.  positive_flower's reference
        #     planes met at 0.16 voxels; after bending the surfaces were 28.4 apart.
        #   * bending collapsed a required gap.  en_echelon's four (P,P) sheets
        #     ended 0.35-6.3 voxels apart, one thick smear at a 14-voxel drag width.
        # Two attempts to fix this inside the PSO objective failed for the same
        # reason: the objective can only see the reference planes, which were
        # already correct.  The repair therefore runs here, on the real surfaces.
        #
        # The only move used is a translation along the fault's own normal, which
        # leaves dip, strike and every Eq. (1)-(3) quantity bit-identical.
        _disjoint = set(disjoint_pairs_from_tree(tree))
        _MEET_TOL = 2.0            # voxels; "these two faults touch"
        _SEP_MIN = 16.0            # voxels; just over cfg.drag_distance

        def _in_box(pts, step):
            q = pts.reshape(-1, 3)[::step]
            return q[(q[:, 0] >= 0) & (q[:, 0] <= nx - 1) &
                     (q[:, 1] >= 0) & (q[:, 1] <= ny - 1) &
                     (q[:, 2] >= 0) & (q[:, 2] <= nz - 1)]

        def _gap(a, b, step=8):
            A, B = _in_box(a["surface"]["points"], step), _in_box(b["surface"]["points"], step)
            if len(A) == 0 or len(B) == 0:
                return None
            return float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min())

        def _shift(f, delta):
            n = f["R"][2]
            f["surface"]["points"] = f["surface"]["points"] + delta * n
            f["centre"] = np.asarray(f["centre"], dtype=float) + delta * n

        by_id = {f["id"]: f for f in faults}

        def _targets(f):
            """(other, want, mode) for every constraint this fault is under."""
            i = f["id"]
            node = tree.nodes[i]
            if node.parent_id is None:
                return []
            out = []
            if node.in_edge and (set(node.in_edge) & {"X", "Y"}):
                out.append((by_id[node.parent_id], _MEET_TOL, "meet"))
            else:
                out.append((by_id[node.parent_id], _SEP_MIN, "apart"))
            # The "stay apart" set is exactly the objective's, so the repair cannot
            # pull apart a horsetail splay or a flower junction: siblings that both
            # truncate against a common Y-edge parent converge by construction.
            for a, b in _disjoint:
                if a == i and b < i:
                    out.append((by_id[b], _SEP_MIN, "apart"))
                elif b == i and a < i:
                    out.append((by_id[a], _SEP_MIN, "apart"))
            return out

        def _cost(f, targets, step):
            """Squared violation summed over all of this fault's constraints.

            Greedy pair-at-a-time repair does not converge -- fixing one pair breaks
            another -- so each candidate move is scored against every constraint the
            fault is under, not just the one that triggered it.
            """
            c = 0.0
            for other, want, mode in targets:
                g = _gap(f, other, step)
                if g is None:
                    continue
                c += (max(0.0, g - want) ** 2 if mode == "meet"
                      else max(0.0, want - g) ** 2)
            return c

        _DELTAS = (3.0, -3.0, 6.0, -6.0, 12.0, -12.0, 20.0, -20.0, 32.0, -32.0)
        for _sweep in range(6):
            moved = False
            for f in faults:
                tg = _targets(f)
                if not tg:
                    continue
                c0 = _cost(f, tg, 24)
                if c0 < 1.0:
                    continue
                best_d, best_c = 0.0, c0
                for d in _DELTAS:
                    _shift(f, d)
                    c = _cost(f, tg, 24)
                    _shift(f, -d)
                    if c < best_c:
                        best_d, best_c = d, c
                if best_d != 0.0:
                    _shift(f, best_d)
                    repairs.append({"fault": f["id"], "shift": round(best_d, 1),
                                    "cost": [round(c0, 1), round(best_c, 1)]})
                    moved = True
            if not moved:
                break

    # ---- diagnostic: how much of each branch survives inside the grid? --
    # Added after a regression that no other metric caught: anchoring branches
    # on P0 instead of the centroid left only 14 % of each branch surface
    # inside the volume, and 8-15 % of branches vanished entirely.  Every
    # existing metric measured a property OF a fault (dip, crossing, collapse);
    # none measured how much fault was left in the box.
    def _inside_frac(pts):
        p = pts.reshape(-1, 3)
        return float(((p[:, 0] >= 0) & (p[:, 0] <= nx - 1) &
                      (p[:, 1] >= 0) & (p[:, 1] <= ny - 1) &
                      (p[:, 2] >= 0) & (p[:, 2] <= nz - 1)).mean())

    inside_frac = [_inside_frac(f["surface"]["points"]) for f in faults]

    # ---- diagnostic: do surfaces respect the prescribed topology? ------
    # Pairs NOT joined by an X or Y edge are supposed to stay apart (Eq. 53).
    gaps = []
    # Same exclusion rule as the geometry objective: parent/child joined by an
    # X or Y edge are meant to meet, and so are siblings that both truncate
    # against a common Y-edge parent — that convergence IS the horsetail /
    # flower morphology, not a defect.
    _disjoint = set(disjoint_pairs_from_tree(tree))
    for ii in range(len(faults)):
        for jj in range(ii + 1, len(faults)):
            fi, fj = faults[ii], faults[jj]
            if (fi["id"], fj["id"]) not in _disjoint:
                continue
            # Section III defines "P" as non-intersecting *in the voxel grid*,
            # so parts of a surface that leave the domain do not count.
            def _in_grid(p):
                return p[(p[:, 0] >= 0) & (p[:, 0] <= nx - 1) &
                         (p[:, 1] >= 0) & (p[:, 1] <= ny - 1) &
                         (p[:, 2] >= 0) & (p[:, 2] <= nz - 1)]
            a = _in_grid(fi["surface"]["points"].reshape(-1, 3)[::7])
            b = _in_grid(fj["surface"]["points"].reshape(-1, 3)[::7])
            if len(a) == 0 or len(b) == 0:
                continue
            d, _ = cKDTree(b).query(a, workers=_KD_WORKERS)
            gaps.append(float(d.min()))

    # ---- rasterise, then Y-truncate against the parent ----------------
    for f in faults:
        f["raster"] = _rasterise(f["surface"]["points"], f["surface"]["normals"],
                                 cfg.grid, cfg.drag_distance)

    for f in faults:
        f["valid_uv"] = np.ones((cfg.n_u, cfg.n_v), dtype=bool)

    by_id = {f["id"]: f for f in faults}
    for f in faults:
        e, pid = f["in_edge"], f["parent"]
        if e is None or pid is None or pid not in by_id:
            continue
        if "Y" not in e:
            continue
        parent = by_id[pid]
        ptree = cKDTree(parent["surface"]["points"].reshape(-1, 3))
        pn = parent["surface"]["normals"].reshape(-1, 3)
        pp = parent["surface"]["points"].reshape(-1, 3)
        pts = f["surface"]["points"].reshape(-1, 3)
        _, idx = ptree.query(pts, workers=_KD_WORKERS)
        side = np.einsum("ij,ij->i", pts - pp[idx], pn[idx])
        # Section III-D: keep the half of the child that lies on the same wall
        # as its reference point, so the child terminates against the parent.
        ref_side = np.sign(side[0]) if side[0] != 0 else 1.0
        f["valid_uv"] = (np.sign(side) == ref_side).reshape(cfg.n_u, cfg.n_v)
        if f["valid_uv"].sum() < 0.05 * f["valid_uv"].size:
            f["valid_uv"] = np.ones_like(f["valid_uv"])     # degenerate: keep all

    # ---- Eq. (28): slip scaling from surface area ---------------------
    def n_points(f):
        r = f["raster"]
        if r is None:
            return 0
        near = np.abs(r["signed"]) <= cfg.label_half_thickness
        return int(near.sum())

    n_root = max(n_points(by_id[0]), 1)
    for f in faults:
        f["d_max"] = (main.d_max if f["id"] == 0
                      else scale_branch_dmax(main.d_max, n_points(f), n_root))

    # ---- Step 3b: displacement fields ---------------------------------
    # The wavelet frequency is drawn here rather than with the forward model
    # because the observable-label rule below needs lambda/4 in samples.
    freq = float(rng.uniform(*cfg.freq_hz))
    lam4 = cfg.observable_frac / (freq * cfg.dt)      # samples

    disp = np.zeros((3, nz, ny, nx), dtype=np.float32)
    label_full = np.zeros((nz, ny, nx), dtype=np.uint8)
    lab_vox, lab_nrm = [], []

    for f in faults:
        r = f["raster"]
        if r is None:
            continue
        use_elliptic = (f["id"] == 0 and main.throughgoing)
        ksi_grid = (surface_slip_field_elliptic(cfg.n_u, cfg.n_v, f["d_max"])
                    if use_elliptic else
                    surface_slip_field(cfg.n_u, cfg.n_v, f["d_max"]))
        ksi_grid = ksi_grid * f["valid_uv"]

        ksi = ksi_grid.ravel()[r["surf_idx"]]
        d_loc = decompose_slip(ksi, main.phi_dis_deg)         # Eq. (17)-(19)
        d_glob = slip_to_global(d_loc, f["R"])                # Eq. (20)

        d_norm = r["signed"] / cfg.drag_distance
        df = drag_operator(d_norm, mode=main.drag_mode)       # Eq. (23)
        contrib = (df[:, None] * d_glob).astype(np.float32)

        for c in range(3):
            _accumulate(disp[c], r["vox_xyz"], contrib[:, c])

        on_surf = (np.abs(r["signed"]) <= cfg.label_half_thickness) & (ksi > 0)
        v = r["vox_xyz"][on_surf]
        label_full[v[:, 2], v[:, 1], v[:, 0]] = 1
        lab_vox.append(v)
        lab_nrm.append(f["surface"]["normals"].reshape(-1, 3)[r["surf_idx"][on_surf]])

    # ---- observable label: keep voxels the seismic can actually show ----
    label = label_full.copy()
    if cfg.observable_labels and lab_vox:
        V = np.concatenate(lab_vox); N = np.concatenate(lab_nrm)
        # vertical offset ACROSS the surface: disp_z two voxels either side.
        # Two, not one: the drag profile is ~0.94 of full slip at 2/14 of the
        # drag distance, so this reads the jump, not the ramp.
        def _dz(sign):
            q = np.rint(V + sign * 2.0 * N).astype(int)
            q[:, 0] = np.clip(q[:, 0], 0, nx - 1)
            q[:, 1] = np.clip(q[:, 1], 0, ny - 1)
            q[:, 2] = np.clip(q[:, 2], 0, nz - 1)
            return disp[2][q[:, 2], q[:, 1], q[:, 0]]
        throw = np.abs(_dz(+1.0) - _dz(-1.0))
        drop = V[throw < lam4]
        label[drop[:, 2], drop[:, 1], drop[:, 0]] = 0

    # ---- Step 3c: stratigraphy and forward model  (Eq. 54-56) ---------
    gz, gy, gx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    gx = gx.astype(np.float32); gy = gy.astype(np.float32); gz = gz.astype(np.float32)

    a = float(rng.uniform(-cfg.tilt, cfg.tilt))
    b = float(rng.uniform(-cfg.tilt, cfg.tilt))
    folds = strat.random_folds(rng, int(rng.integers(*cfg.n_folds)), nx, ny,
                               cfg.fold_amp, cfg.fold_sigma)
    s1 = strat.linear_shift(gx, gy, a, b, centre[0], centre[1])
    s2 = strat.fold_shift(gx, gy, gz, folds, float(nz))

    z_def = gz + s1 + s2 + disp[2]                            # Eq. (56)
    ref = strat.reflectivity_1d(nz, rng)
    model = strat.sample_reflectivity(ref, z_def)

    snr = float(rng.uniform(*cfg.snr_db))
    seis = strat.convolve_z(model, strat.ricker(freq, cfg.dt))
    seis = strat.normalize(strat.add_noise(seis, snr, rng))

    meta = {
        "pso": pso_info,
        "zeta": zeta.tolist(),
        "tree": [{"id": n.id, "parent": n.parent_id, "in_edge": n.in_edge}
                 for n in tree.nodes.values()],
        "n_faults": len(faults),
        "tree_depth": tree.depth(),
        "faults": [{"id": f["id"], "strike": round(f["strike"], 3),
                    "dip": round(f["dip"], 3), "d_max": round(f["d_max"], 3),
                    "half_len": round(f["half_len"], 2),
                    "half_wid": round(f["half_wid"], 2),
                    "in_edge": f["in_edge"]} for f in faults],
        "main": {"strike": main.strike_deg, "dip": main.dip_deg,
                 "phi_dis": main.phi_dis_deg, "d_max": main.d_max,
                 "throughgoing": main.throughgoing, "drag": main.drag_mode},
        "geometry_objective": bool(cfg.use_geometry_objective),
        "geometry_breakdown": (geometry_penalty(
            np.vstack([np.zeros((1, 3)), zeta]), geo_ctx,
            GeoWeights(cfg.geo_w_dip, cfg.geo_w_int, cfg.geo_w_dup,
                       cfg.geo_w_conj, cfg.geo_w_present, cfg.geo_w_sep))[1]
            if geo_ctx is not None else None),
        "hull_constraint": bool(cfg.use_hull_constraint),
        "bezier_params_clamped": int(n_clamped),
        "surface_repairs": repairs,
        "branch_inside_frac": [round(v, 4) for v in inside_frac[1:]],
        "min_branch_inside_frac": round(min(inside_frac[1:]), 4) if len(inside_frac) > 1 else 1.0,
        "disjoint_pairs": len(gaps),
        "min_surface_gap": [round(g, 3) for g in gaps],
        "folds": len(folds), "tilt": [a, b],
        "freq_hz": round(freq, 2), "snr_db": round(snr, 2),
        "fault_fraction": float(label.mean()),
        "fault_fraction_full": float(label_full.mean()),
        "observable_labels": bool(cfg.observable_labels),
        "lambda4_samples": round(lam4, 3),
        "label_kept_frac": (float(label.sum() / max(label_full.sum(), 1))),
    }
    # Downsampled parametric surfaces, for 3-D rendering (Fig. 15 style).
    # 48 x 48 put the samples 3.8 voxels apart, and since the surfaces overrun
    # the box the renderer clips them on that grid -- which is where the sawtooth
    # edges came from.  The surfaces themselves are smooth (median second
    # difference 0.15-0.26 voxels), so the fix is display resolution, not
    # smoothing: 96 x 96 halves the step to ~1.9 voxels.
    step = max(1, cfg.n_u // 96)
    surfaces = {f["id"]: f["surface"]["points"][::step, ::step].astype(np.float32)
                for f in faults}

    return {"seismic": seis, "label": label, "label_full": label_full,
            "meta": meta, "surfaces": surfaces}
