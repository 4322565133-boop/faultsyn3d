"""Module 1 — geometry-aware terms for the fault network objective.

OUR CONTRIBUTION, not part of Su et al. (2026).

Motivation, all measured on the faithful reproduction (see docs/step1_findings.md):

* **F2** the edge constraints of Eq. (43)-(45) are pure *sign* conditions on
  `Delta_0 . Delta_k`.  They carry no magnitude bound, so branch dip lands
  anywhere within about +-35 deg of the main fault; with a low-angle main fault
  ~20 % of branches come out below 20 deg, essentially bedding-parallel.
* **F7** "non-intersecting" is in the definition of the P edge but nothing in
  the objective enforces it: 23 % of pairs declared non-intersecting do cross
  inside the voxel grid, and Algorithm 4 cannot help because it only constrains
  the Bezier bending, not the placement.
* **F6** listric labels reach a semblance AUC of only 0.562 against background,
  i.e. close to chance — a low-dip fault has vertical throw `slip * sin(dip)`
  and leaves almost no reflector offset.

These share one cause: geometric plausibility is absent from the objective.
This module adds it:

    F'(s) = F_topology(s) + w_dip * G_dip + w_int * G_int + w_dup * G_dup

REMOVED AFTER ABLATION — an observability term.  We originally carried a fourth
term penalising vertical throw `d_max * sin(dip)` below the tuning thickness,
aimed at F6.  The per-term ablation (docs/module1.md) showed it inert: dropping
it moved overall semblance AUC by +0.000 and listric AUC by +0.002, both inside
noise.  The reason is visible in closed form — for a typical branch it demands
dip >= 13 deg while the dip band already demands >= 35 deg, so `G_dip`
subsumes it entirely.

That ablation also settled where F6 actually comes from.  Raising dip cannot
help, because the unobservable label voxels sit in the fault's TIP region where
the slip field itself tapers to zero; `xi * sin(dip)` stays ~0 whatever the dip
is.  Observability is a property of the LABELLING rule, not of the geometry,
and belongs in Module 6.

Every term is computed in closed form from the same decision variables zeta
that Su's topology penalty uses, so the optimiser is unchanged in structure and
the extra cost per evaluation is a few hundred flops.

Topology stays the primary term: the two are reported separately so we can
verify that adding geometry does not stop `F_topology` from reaching zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import dip_strike_from_normal, normal_from_points
from .optimize import zeta_to_points


# --------------------------------------------------------------------------
# Admissible dip bands, by Andersonian stress regime.
#
# Deliberately WIDE and OVERLAPPING.  The point is not to pin each pattern to a
# narrow dip range — that is exactly the confound that made the archived stage-1
# dataset teach a CNN a dip shortcut.  The point is only to exclude the
# geologically degenerate cases: faults so shallow they are indistinguishable
# from bedding, or so steep they cannot carry the prescribed slip sense.
# --------------------------------------------------------------------------

DIP_BANDS = {
    "extensional":   (35.0, 80.0),    # normal / listric faults, Anderson ~60
    "strike_slip":   (40.0, 90.0),    # wrench systems, Anderson ~90
    "compressional": (15.0, 55.0),    # thrusts, Anderson ~30
    # Fig. 15(e)'s caption is explicit: "listric assemblage formed by LOW-ANGLE
    # faults".  Running these under the extensional band cost 19.2 of penalty on
    # its own -- the band was fighting the (P, Y) topology, which wants the
    # splays to sole out onto the parent at a shallow angle.
    "listric":       (28.0, 62.0),
}

#: No discrete fault in seismic is meaningfully flatter than this.
DIP_FLOOR = 12.0


@dataclass
class GeoContext:
    """Per-volume quantities the geometry terms need, all independent of zeta."""
    centre: np.ndarray
    R: np.ndarray
    origins: np.ndarray
    d_z: float
    grid: tuple[int, int, int]
    half_len: np.ndarray          # (n_f,) per-fault half extent along strike
    half_wid: np.ndarray          # (n_f,) per-fault half extent down dip
    d_max: np.ndarray             # (n_f,) max slip, from Eq. (28)
    regime: str = "strike_slip"
    #: pairs (i, j) that the topology declares non-intersecting
    disjoint_pairs: tuple = ()
    #: (i, j, k) triples joined by an X edge; k = 1 strike, k = 2 dip
    crossing_pairs: tuple = ()
    #: (parent, child, "opp" | "same") for every edge of the tree
    dip_relations: tuple = ()
    ref_point: str = "centroid"   # must match GenConfig.branch_ref_point


# --------------------------------------------------------------------------

def branch_frames(z: np.ndarray, ctx: GeoContext):
    """Dip, strike, centre and unit normal of every fault, given a solution.

    z is the (n_f, 3) zeta array with the root pinned at zero (Eq. 40).
    """
    n_f = len(z)
    dips = np.empty(n_f)
    normals = np.empty((n_f, 3))
    centres = np.empty((n_f, 3))

    # root: its frame is user-specified, recovered from R's third row
    normals[0] = ctx.R[2]
    centres[0] = ctx.centre
    dips[0] = dip_strike_from_normal(ctx.R[2])[0]

    for i in range(1, n_f):
        pts = zeta_to_points(z[i], ctx.centre, ctx.R, ctx.origins, ctx.d_z)
        try:
            n = normal_from_points(pts[0], pts[1], pts[2])
        except ValueError:
            n = ctx.R[2]
        normals[i] = n
        centres[i] = (pts.mean(axis=0) if ctx.ref_point == "centroid"
                      else pts[int(ctx.ref_point[1])])
        dips[i] = dip_strike_from_normal(n)[0]

    # fold dip onto [0, 90]: 100 deg one way is 80 deg the other
    dips = np.where(dips > 90.0, 180.0 - dips, dips)
    return dips, normals, centres


# --------------------------------------------------------------------------
# Term 1 — dip plausibility
# --------------------------------------------------------------------------

def g_dip(dips: np.ndarray, ctx: GeoContext, scale: float = 10.0) -> float:
    """Quadratic hinge outside the admissible dip band, plus a hard floor.

    Branch faults only (the root's dip is the user's choice, not the
    optimiser's).
    """
    lo, hi = DIP_BANDS[ctx.regime]
    d = dips[1:]
    below = np.clip(lo - d, 0.0, None)
    above = np.clip(d - hi, 0.0, None)
    floor = np.clip(DIP_FLOOR - d, 0.0, None)
    return float((((below + above) / scale) ** 2).sum()
                 + 4.0 * ((floor / scale) ** 2).sum())


# --------------------------------------------------------------------------
# Term 3 — non-intersection actually enforced
# --------------------------------------------------------------------------

def _rect_corners(centre, normal, half_len, half_wid, R_hint):
    """Four corners of a finite fault rectangle, in global coordinates.

    Written without np.linalg.norm / np.cross: on 3-vectors their call
    overhead dominated, and this function was 60 % of a whole PSO run.
    """
    r0 = R_hint[0]
    a = r0 - normal * (r0[0] * normal[0] + r0[1] * normal[1] + r0[2] * normal[2])
    na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) ** 0.5
    if na < 1e-8:
        r1 = R_hint[1]
        a = r1 - normal * (r1[0] * normal[0] + r1[1] * normal[1] + r1[2] * normal[2])
        na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) ** 0.5
    a = a / max(na, 1e-12)
    b = np.array([normal[1] * a[2] - normal[2] * a[1],
                  normal[2] * a[0] - normal[0] * a[2],
                  normal[0] * a[1] - normal[1] * a[0]])
    la, lb = half_len * a, half_wid * b
    return np.stack([centre - la - lb, centre - la + lb,
                     centre + la - lb, centre + la + lb])


def _all_corners(normals, centres, ctx):
    """Corners of every fault, computed once per objective evaluation.

    `g_int` used to call `_rect_corners` twice per disjoint pair, i.e. 12 times
    per evaluation on a 4-fault network, for a quantity that only depends on
    the fault.  482 000 calls per PSO run; now n_f per evaluation.
    """
    return [_rect_corners(centres[k], normals[k], ctx.half_len[k],
                          ctx.half_wid[k], ctx.R) for k in range(len(normals))]


def g_int(normals: np.ndarray, centres: np.ndarray, ctx: GeoContext,
          corners_all=None) -> float:
    """Penalise pairs that the topology calls disjoint but which actually cross.

    Cheap test: take j's four corners, measure their signed distance to i's
    infinite plane.  Mixed signs mean j's rectangle straddles i's plane, and the
    straddle only matters if it happens near i's own footprint and inside the
    voxel grid.  Symmetric in i and j.
    """
    if not ctx.disjoint_pairs:
        return 0.0
    nx, ny, nz = ctx.grid
    if corners_all is None:
        corners_all = _all_corners(normals, centres, ctx)
    total = 0.0

    for i, j in ctx.disjoint_pairs:
        pen = 0.0
        for a, b in ((i, j), (j, i)):
            corners = corners_all[b]
            sd = (corners - centres[a]) @ normals[a]
            if sd.min() < 0.0 < sd.max():          # b straddles a's plane
                # how close the straddle is to a's own centre, normalised
                mid = corners[np.argsort(np.abs(sd))[:2]].mean(axis=0)
                if not (0 <= mid[0] <= nx - 1 and 0 <= mid[1] <= ny - 1
                        and 0 <= mid[2] <= nz - 1):
                    continue                        # crossing is outside the cube
                r = np.linalg.norm(mid - centres[a])
                reach = float(np.hypot(ctx.half_len[a], ctx.half_wid[a]))
                pen += float(np.clip(1.0 - r / max(reach, 1e-9), 0.0, 1.0) ** 2)
        total += pen
    return total / max(len(ctx.disjoint_pairs), 1)


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Term 4 — no near-duplicate faults
# --------------------------------------------------------------------------

def g_dup(normals: np.ndarray, centres: np.ndarray, ctx: GeoContext,
          cos_tol: float = 0.985, sep_frac: float = 0.35) -> float:
    """Penalise pairs of faults that are effectively the same fault.

    Terms 1-3 are hinges: inside the admissible band they are flat zero.  That
    leaves a large region where the total penalty is zero and PSO returns an
    arbitrary member of it — which can be degenerate.  Observed directly: with
    only terms 1-3 the horsetail's three splays collapsed onto dips
    [89, 89, 89], three near-coincident copies of the main fault, and the splay
    morphology disappeared.

    Admissibility is not sufficiency: the objective must also say that the
    faults should differ from one another.  A pair counts as duplicated when
    the normals are near-parallel AND the centres are close relative to the
    fault size.
    """
    n_f = len(normals)
    if n_f < 2:
        return 0.0
    reach = float(np.mean(np.hypot(ctx.half_len, ctx.half_wid)))
    total = 0.0
    for i in range(n_f):
        for j in range(i + 1, n_f):
            cosang = abs(float(normals[i] @ normals[j]))
            if cosang <= cos_tol:
                continue
            par = (cosang - cos_tol) / (1.0 - cos_tol)        # 0 -> 1
            d = float(np.linalg.norm(centres[i] - centres[j]))
            near = np.clip(1.0 - d / max(sep_frac * reach, 1e-9), 0.0, 1.0)
            total += float((par * near) ** 2)
    return total


# --------------------------------------------------------------------------
# Term 5 — the dip relation prescribed on every edge
#
# Replaces two earlier attempts, both of which measured the wrong object:
#   `g_cross`  demanded that the two reference RECTANGLES of an X edge meet
#              inside the grid.  They already did, at a gap of 0.16 voxels; the
#              term was inert, like `G_obs` before it.
#   `g_conj`   applied only to X edges.  But the dip relation is prescribed on
#              EVERY edge of Fig. 15, not just the crossing ones, and a horsetail
#              whose splays lean the wrong way is as wrong as a flower that does
#              not cross.
#
# Su's attribute set {X, Y, P} says only whether two faults MEET.  Which way a
# branch leans is unconstrained, and measured on the baseline it was a coin
# toss: dips opposed in 9 of 20 volumes.  This term supplies the missing half.
# --------------------------------------------------------------------------

#: How clearly the two dip directions must separate.  cos of the angle between
#: the horizontal components of the normals; 0.3 is about 72 deg apart.
DIP_REL_MARGIN = 0.30

#: An "opposite" pair also has to be visibly oblique to each other, or the
#: opposition is a technicality invisible on a section.
CONJ_MIN_ANGLE = 25.0


def _dip_from_normal(n) -> float:
    """Dip from horizontal, folded to [0, 90]."""
    return float(np.degrees(np.arccos(np.clip(abs(float(n[2])), 0.0, 1.0))))


def g_dip_rel(normals: np.ndarray, ctx: GeoContext, scale: float = 10.0) -> float:
    """Penalise edges whose child leans the wrong way relative to its parent."""
    if not ctx.dip_relations:
        return 0.0
    total = 0.0
    for i, j, rel in ctx.dip_relations:
        ni, nj = normals[i], normals[j]
        hv, hw = ni[:2], nj[:2]
        na, nb = np.linalg.norm(hv), np.linalg.norm(hw)
        if na < 1e-9 or nb < 1e-9:
            total += 1.0            # a horizontal plane has no dip direction
            continue
        align = float(hv @ hw) / (na * nb)        # +1 same sense, -1 opposed
        if rel == "opp":
            total += max(0.0, align + DIP_REL_MARGIN) ** 2
            ang = float(np.degrees(np.arccos(
                np.clip(abs(float(ni @ nj)), 0.0, 1.0))))
            total += (max(0.0, CONJ_MIN_ANGLE - ang) / scale) ** 2
        else:
            total += max(0.0, DIP_REL_MARGIN - align) ** 2
    return total / max(len(ctx.dip_relations), 1)


# --------------------------------------------------------------------------
# Term 7 — faults the topology separates must be visibly apart
#
# "P" in Definition 4 means non-intersecting, and Su's Eq. (45) enforces
# exactly that: the sign of Delta_0 * Delta_k, nothing more.  Measured on
# `en_echelon`, whose three edges are all (P, P), the four surfaces came out
# 0.35 to 6.28 voxels apart -- formally disjoint, and at a rasterised label
# half-thickness of 0.75 with a 14-voxel drag zone they are one thick smear.
# Fig. 15(a) shows four sheets stepping sideways with clear water between them.
#
# Non-intersection is a sign condition; "arrayed with a gap" is a magnitude
# condition, and it is the third time the same distinction has bitten us.
# --------------------------------------------------------------------------

#: Minimum perpendicular offset between two faults the topology separates.
#: Set just above `GenConfig.drag_distance` (14) so the two damage zones do not
#: merge into one deformation band.
SEP_MIN = 16.0


def g_sep(normals: np.ndarray, centres: np.ndarray, ctx: GeoContext,
          scale: float = 16.0) -> float:
    """Penalise disjoint pairs whose planes sit closer than SEP_MIN.

    The measure is the perpendicular offset of each centre from the other's
    plane -- for the near-parallel sheets of an en echelon array that is
    precisely the sideways step that makes them read as separate faults.
    """
    if not ctx.disjoint_pairs:
        return 0.0
    total = 0.0
    for i, j in ctx.disjoint_pairs:
        d = min(abs(float((centres[j] - centres[i]) @ normals[i])),
                abs(float((centres[i] - centres[j]) @ normals[j])))
        total += (max(0.0, SEP_MIN - d) / scale) ** 2
    return total / max(len(ctx.disjoint_pairs), 1)


# --------------------------------------------------------------------------
# Term 6 — every fault must be present in the observation window
# --------------------------------------------------------------------------

def g_present(centres: np.ndarray, ctx: GeoContext) -> float:
    """Penalise branches whose centre falls outside the voxel grid.

    NOT the same as maximising the fraction of a fault inside the box: Fig. 5
    defines a throughgoing fault as one whose tips lie OUTSIDE the domain, so
    overrunning the box is correct and an earlier attempt to maximise coverage
    was pushing against the paper.  This term only asks that the fault's centre
    — where the Hermite slip field peaks — be inside the window.  Measured on
    the reproduction, branches came out with 1.7 % and 14 % of their surface in
    the grid: those faults are not "throughgoing", they are absent.
    """
    nx, ny, nz = ctx.grid
    lim = np.array([nx - 1.0, ny - 1.0, nz - 1.0])
    c = centres[1:]
    if len(c) == 0:
        return 0.0
    out = np.maximum(np.maximum(-c, c - lim), 0.0) / lim
    return float((out ** 2).sum())


@dataclass
class GeoWeights:
    dip: float = 1.0
    intersect: float = 2.0
    duplicate: float = 3.0
    conj: float = 2.0
    separate: float = 2.0
    present: float = 2.0


def geometry_penalty(z: np.ndarray, ctx: GeoContext,
                     w: GeoWeights) -> tuple[float, dict]:
    """G(s) and its breakdown, for a solution already decoded to (n_f, 3)."""
    dips, normals, centres = branch_frames(z, ctx)
    corners_all = _all_corners(normals, centres, ctx)
    gd = g_dip(dips, ctx)
    gi = g_int(normals, centres, ctx, corners_all)
    gp = g_dup(normals, centres, ctx)
    gc = g_dip_rel(normals, ctx)
    gv = g_present(centres, ctx)
    gs = g_sep(normals, centres, ctx)
    total = (w.dip * gd + w.intersect * gi + w.duplicate * gp
             + w.conj * gc + w.present * gv + w.separate * gs)
    br = dips[1:]
    return total, {"dip": gd, "intersect": gi, "duplicate": gp,
                   "conj": gc, "present": gv, "separate": gs,
                   "min_branch_dip": float(br.min()) if len(br) else float("nan"),
                   "dip_spread": float(br.max() - br.min()) if len(br) > 1 else 0.0}


def disjoint_pairs_from_tree(tree) -> tuple:
    """Pairs the topology says should NOT meet.

    Two exclusions, not one:

    * a parent/child joined by an X or Y edge is *meant* to intersect;
    * **siblings that both truncate against the same Y-edge parent** converge at
      that junction by construction — that is what a horsetail splay or a
      flower structure IS.  Penalising their proximity would destroy the very
      morphology Su's Fig. 15 is built to produce.
    """
    ids = sorted(tree.nodes)
    out = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            ni, nj = tree.nodes[i], tree.nodes[j]
            if ((nj.parent_id == i and nj.in_edge
                 and set(nj.in_edge) & {"X", "Y"}) or
                    (ni.parent_id == j and ni.in_edge
                     and set(ni.in_edge) & {"X", "Y"})):
                continue
            if (ni.parent_id is not None and ni.parent_id == nj.parent_id
                    and ni.in_edge and nj.in_edge
                    and (set(ni.in_edge) & {"X", "Y"})
                    and (set(nj.in_edge) & {"X", "Y"})):
                continue
            out.append((i, j))
    return tuple(out)


def crossing_pairs_from_tree(tree) -> tuple:
    """Parent/child pairs joined by an X edge in either direction.

    Only direct edges: "X" is a statement about a fault and its parent, and the
    relation between two nodes that are not adjacent in the tree is whatever the
    other constraints leave it.
    """
    out = []
    for n in tree.nodes.values():
        if n.parent_id is None or not n.in_edge:
            continue
        for k in (1, 2):
            if n.in_edge[k - 1] == "X":
                out.append((n.parent_id, n.id, k))
    return tuple(out)
