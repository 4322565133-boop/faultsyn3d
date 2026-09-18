"""The five representative fault patterns of Su et al. (2026), Fig. 15.

Tree topologies are read straight off Fig. 15; edge attributes follow the
figure captions ("conjugate system in the strike direction" -> X in strike,
"in the dip direction" -> X in dip) together with the hierarchy rules of
Algorithm 3.

What is user-specified and what is not
--------------------------------------
Per Definition 3, ONLY the main fault (i = 0) gets a user-supplied strike, dip
and centre.  Every branch fault's dip and strike is produced by the optimizer
(zeta -> Eq. 52 -> Eq. 1 -> Eq. 2, 3).  Nothing below sets a branch dip.
"""

from __future__ import annotations

import numpy as np

from su.model import GenConfig, MainFault, SurfaceParams
from su.topology import tree_from_spec


CATEGORIES = ("en_echelon", "horsetail", "negative_flower",
              "positive_flower", "listric_assemblage")


# Fig. 15 tree structures: list of (parent_id, (e_strike, e_dip)) for nodes 1..n
# Transcribed from the trees drawn in Fig. 15 itself, not inferred from the
# captions.  Two of these were wrong in an earlier version: horsetail carried
# ("Y","Y") instead of ("Y","P"), and listric was a flat star instead of the
# depth-3 tree the figure shows.
TREE_SPECS = {
    # (a) en echelon arrays: overlapping, non-intersecting oblique faults
    "en_echelon":       [(0, ("P", "P")), (0, ("P", "P")), (0, ("P", "P"))],
    # (b) horsetail splay — branches converge on the main fault in STRIKE but
    # stay apart in DIP, which is what makes the splay fan out in map view.
    "horsetail":        [(0, ("Y", "P")), (0, ("Y", "P")), (0, ("Y", "P"))],
    # (c) conjugate in STRIKE -> negative flower;  0 -> 1 -> {2, 3}
    "negative_flower":  [(0, ("X", "P")), (1, ("Y", "Y")), (1, ("Y", "Y"))],
    # (d) conjugate in DIP -> positive flower;  0 -> 1 -> 2 -> 3 chain, and
    # every edge of that chain carries (P, X), not just the first: read off the
    # high-resolution figure.  Algorithm 3 admits the repetition — (P,X) is in
    # C_p^3, and (P,X) is not in C_c^3 = {(X,X), (X,P)}.
    "positive_flower":  [(0, ("P", "X")), (1, ("P", "X")), (2, ("P", "X"))],
    # (e) listric assemblage: 0 -> {1, 2, 3} with (P, Y), and 3 -> 4 with (Y,Y).
    # (P, Y) not (P, P): listric faults are non-intersecting in strike but SOLE
    # OUT onto the parent at depth, which is a Y junction in the dip direction.
    "listric_assemblage": [(0, ("P", "Y")), (0, ("P", "Y")),
                           (0, ("P", "Y")), (3, ("Y", "Y"))],
}


# Main-fault sampling ranges.  dip and phi_dis are the two knobs that separate
# the tectonic regimes: near-vertical + strike-slip-dominated phi_dis gives the
# wrench family, low dip + dip-slip gives the listric family.
# phi_dis: 0 deg is pure dip-slip, 90 deg pure strike-slip (Eq. 17-19).
# The wrench patterns were first given 68-88 deg, i.e. 93-100 % strike-slip,
# on the reasoning that en echelon arrays and horsetail splays ARE strike-slip
# structures.  That is geologically right and seismically useless: sliding
# flat-lying layers along strike leaves the section unchanged, so the labels sat
# on undisturbed reflectors.  Measured vertical throw d_max*cos(phi)*sin(dip)
# came to 3.3 samples against a tuning thickness lambda/4 = 2.1, and only 6.5 %
# of the labelled surface carried a throw above it.
# 35-55 deg is transtensional oblique slip: still strike-slip-dominated in
# character, but with a dip-slip component the seismic can record.  It lifts
# the observable area to ~40 %, which is where the Hermite taper caps out
# (even pure dip-slip only reaches 47 %).
# Strike is rotated 90 deg from our first choice (60-120).  It changes no
# geology -- a plane's orientation is strike mod 180 -- but it puts the fault
# traces on the block faces the way Fig. 15 and Fig. 16 show them.
MAIN_RANGES = {
    "en_echelon":       dict(strike=(150, 210), dip=(62, 80),  phi_dis=(35, 55),
                             throughgoing=False, drag="normal"),
    "horsetail":        dict(strike=(150, 210), dip=(66, 84),  phi_dis=(35, 55),
                             throughgoing=True,  drag="normal"),
    "negative_flower":  dict(strike=(150, 210), dip=(72, 88),  phi_dis=(35, 55),
                             throughgoing=True,  drag="normal"),
    "positive_flower":  dict(strike=(150, 210), dip=(72, 88),  phi_dis=(35, 55),
                             throughgoing=True,  drag="normal"),
    "listric_assemblage": dict(strike=(150, 210), dip=(32, 52), phi_dis=(0, 16),
                               throughgoing=False, drag="reverse"),
}


#: How a branch's BENDING relates to the main fault's, read off Fig. 15.
#:   -1  branch bows opposite to the main fault
#:   +1  branch bows the same way (nested shells)
#:
#: This is NOT what makes a conjugate pair cross.  "Conjugate" means the two
#: faults dip in OPPOSITE directions, which is a property of the normal and
#: therefore an output of the optimizer (Eq. 1-3); it has nothing to do with
#: the Bezier curvature sign.  Reading "opposite" as opposite BENDING for the
#: two flower patterns was a mistake with a measurable cost: the reference
#: planes of `positive_flower` cross at a gap of 0.16 voxels, and bending the
#: branch the other way pushed the final surfaces 28.4 voxels apart, deleting
#: the crossing the optimizer had correctly arranged.  The flowers therefore
#: take +1: the crossing comes from the dips, and the bending must not undo it.
BRANCH_CURV_SIGN = {
    # (a) Fig. 15(a) is four crescents of the SAME handedness, offset along
    # strike.  (An earlier -1 came from reading the branches as bowing
    # against the main fault; the high-resolution figure shows otherwise.)
    "en_echelon": +1.0,
    # (b) branches grow out of the main fault, so they inherit its bending
    "horsetail": +1.0,
    # (c)(d) the crossing comes from CONJUGATE DIPS (g_conj), not from bending.
    # Bending them the other way is what tore the crossing apart: the reference
    # planes met at 0.16 voxels and the final surfaces ended 28.4 apart.
    "negative_flower": +1.0,
    "positive_flower": +1.0,
    # (e) nested concave-up shells soling out onto the same detachment
    "listric_assemblage": +1.0,
}


#: Dip relation prescribed on every EDGE of the tree, parallel to TREE_SPECS.
#:   "opp"   child dips against its parent  (conjugate)
#:   "same"  child dips with its parent
#:
#: Su's edge attribute has three values (X, Y, P) and they are all statements
#: about whether two faults MEET.  Nothing in Eq. (43)-(45) says which way a
#: branch leans, so on the baseline the dip sense across an edge came out
#: opposed in only 9 of 20 volumes -- a coin toss.  Read off Fig. 15, the five
#: patterns each prescribe it, and it is what makes them recognisable:
#:   (a) three splays leaning against the master fault
#:   (b) three splays leaning with it
#:   (c) one child against the master, its two children with it
#:   (d) the same, carried one generation further down the chain
#:   (e) three splays with the master, one grandchild with its parent
DIP_RELATION = {
    "en_echelon":         ["opp", "opp", "opp"],
    "horsetail":          ["same", "same", "same"],
    "negative_flower":    ["opp", "same", "same"],
    "positive_flower":    ["opp", "same", "same"],
    "listric_assemblage": ["same", "same", "same", "same"],
}


def dip_relations(category: str, tree) -> tuple:
    """(parent, child, relation) for every edge, in TREE_SPECS order."""
    rel = DIP_RELATION[category]
    out = []
    for k, (parent, _edge) in enumerate(TREE_SPECS[category]):
        out.append((parent, k + 1, rel[k]))
    return tuple(out)


#: Andersonian stress regime per pattern — selects the admissible dip band
#: used by Module 1's geometry objective (our extension, not Su et al.).
REGIME = {
    "en_echelon": "strike_slip",
    "horsetail": "strike_slip",
    "negative_flower": "strike_slip",
    "positive_flower": "strike_slip",
    "listric_assemblage": "listric",
}


def sample_main_fault(category: str, cfg: GenConfig,
                      rng: np.random.Generator) -> MainFault:
    nx, ny, nz = cfg.grid
    r = MAIN_RANGES[category]
    return MainFault(
        strike_deg=float(rng.uniform(*r["strike"])),
        dip_deg=float(rng.uniform(*r["dip"])),
        centre=(nx / 2 + float(rng.uniform(-10, 10)),
                ny / 2 + float(rng.uniform(-10, 10)),
                nz / 2 + float(rng.uniform(-8, 8))),
        # Fig. 5 defines a "throughgoing" fault as one whose tips lie OUTSIDE
        # the modelling domain; a fault reduced to 80 % of that size becomes
        # non-throughgoing and its tips enter the volume.  Our first setting
        # (0.42-0.60, i.e. a full extent of 0.84-1.20 of the box) put the tips
        # right ON the boundary, which is the worst case: the outer faces then
        # sample precisely the tip region, where the Hermite slip field is zero,
        # so labels there sit on undisturbed reflectors.  Measured coherence
        # contrast by distance from the boundary: 0.016 at 0-5 voxels against
        # 0.275 at 50+, a 17x difference.  Fig. 16 shows clear offsets on the
        # block faces, so the main fault must overrun the box.
        half_len=float(rng.uniform(0.65, 0.85) * nx),
        half_wid=float(rng.uniform(0.65, 0.85) * nz),
        d_max=float(rng.uniform(8.0, 20.0)),
        phi_dis_deg=float(rng.uniform(*r["phi_dis"])),
        throughgoing=bool(r["throughgoing"]),
        drag_mode=r["drag"],
    )


def sample_surface_params(category: str, n_nodes: int, cfg: GenConfig,
                          rng: np.random.Generator) -> dict[int, SurfaceParams]:
    """B_strike (Eq. 10) and B_dip (Eq. 11) for every node.

    The down-dip curvature B_dip is what makes a fault listric: a consistently
    signed beta bends the surface concave-up so it flattens with depth.  For
    the wrench families the curvature is kept small and sign-varying.
    """
    m = 5                                   # number of profile curves
    out: dict[int, SurfaceParams] = {}

    # Curvature is drawn ONCE for the network, then inherited by every branch
    # with the same SIGN and a small jitter in magnitude.  In Fig. 15 the
    # branches read as copies of the main fault — (a) is four sheets bowing the
    # same way, (e) is a set of nested concentric shells.  Sampling the sign
    # independently per fault, as we did first, produces a jumble of surfaces
    # curving in different directions, which is what made our renders look
    # unlike the paper's.
    listric = category == "listric_assemblage"
    rel = (BRANCH_CURV_SIGN[category]
           if getattr(cfg, "inherit_branch_curvature", False) else None)
    if listric:
        sign_dip = 1.0
        base_dip = float(rng.uniform(0.40, 0.65))
        base_str = float(rng.uniform(-0.12, 0.12))
    else:
        sign_dip = float(rng.choice([-1.0, 1.0]))
        base_dip = float(rng.uniform(0.10, 0.25))
        base_str = float(rng.uniform(0.15, 0.35)) * float(rng.choice([-1.0, 1.0]))

    for i in range(n_nodes):
        # node 0 is the main fault; every branch takes the relationship above
        # Su et al. draw each curve's control parameters independently; the
        # per-pattern sign is ours and only applies when explicitly enabled.
        s_i = (1.0 if i == 0 else
               rel if rel is not None else float(rng.choice([-1.0, 1.0])))
        jitter = 1.0 if i == 0 else float(rng.uniform(0.80, 1.20))
        beta_dip = s_i * sign_dip * base_dip * jitter * np.ones(m)
        beta_dip = beta_dip * rng.uniform(0.92, 1.08, size=m)
        alpha_dip = rng.uniform(0.35, 0.65, size=m)
        bs = base_str * s_i * (1.0 if i == 0 else float(rng.uniform(0.80, 1.20)))
        b_strike = dict(alpha_top=float(rng.uniform(0.35, 0.65)),
                        beta_top=bs,
                        alpha_bottom=float(rng.uniform(0.35, 0.65)),
                        beta_bottom=bs * float(rng.uniform(0.85, 1.15)))
        out[i] = SurfaceParams(
            b_strike=b_strike,
            b_dip=dict(alpha=alpha_dip, beta=beta_dip),
            n_perturb=int(rng.integers(6, 20)),
            eps_perturb=float(rng.uniform(0.6, 2.0)),
        )
    return out


def build_case(category: str, cfg: GenConfig, rng: np.random.Generator):
    """Return (tree, main_fault, surface_params) for one volume.

    Also stamps the pattern's stress regime onto `cfg` so Module 1's dip band
    matches the tectonics; harmless when the geometry objective is disabled.
    """
    cfg.geo_regime = REGIME[category]
    tree_tmp = tree_from_spec(TREE_SPECS[category])
    cfg.dip_relations = dip_relations(category, tree_tmp)
    tree = tree_from_spec(TREE_SPECS[category])
    main = sample_main_fault(category, cfg, rng)
    surf = sample_surface_params(category, len(tree), cfg, rng)
    return tree, main, surf
