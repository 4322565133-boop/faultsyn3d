# faultsyn3d

Working repository for the TGRS-track project. Stage 1 (the previous
`synthetic_fault3d_su2026_final_vscode` tree) is preserved read-only at
`/hdd1/hukaixiao/projects/_archive_20260908_faultsyn_stage1`.

## Roadmap

| Step | Goal | Status |
|---|---|---|
| 1 | Faithful reproduction of Su et al. (2026) | **generator complete** |
| 2 | Dataset innovation (our contribution) | not started |
| 3 | Model comparison on the new dataset | not started |
| 4 | MaxViT upgrade (loss + architecture) | not started |
| 5 | Generalisation on real surveys vs U-Net and baselines | not started |

## Step 1 — reproducing Su et al. (2026)

> Su, Zhang, Cai, Zhou, Yao, Hu. *A Fault Network Synthesis Optimization Model
> for Automatic Generation of Seismic Fault Datasets With Diverse Geological
> Patterns.* IEEE TGRS **64**, 2026. DOI 10.1109/TGRS.2026.3699734

### What the method actually is

A fault network is a **rooted tree**. The root is the main fault (the only
fault whose strike/dip/centre the user chooses). Every branch fault is defined
by three points, and those three points come from the optimiser:

```
zeta (integer offsets on 3 axes)          <- PSO decision variables
   -> Eq. 52 -> three global points P0,P1,P2
   -> Eq. 1  -> unit normal
   -> Eq. 2,3 -> DIP and STRIKE of the branch fault
```

So **branch dip is an output of the optimisation, never a hand-set range.**
Edge attributes `(e_strike, e_dip) ∈ {X, Y, P}²` (7 admissible pairs) become
sign constraints on `Δ₀·Δₖ` (Eq. 43-45), those become quadratic penalties
(Eq. 46-48), and PSO drives the total penalty (Eq. 49) to zero.

### Module map

| File | Paper section | Contents |
|---|---|---|
| `su/geometry.py` | III-A-1, III-A-2 | Eq. 1-13; Bézier guide/profile surfaces (Fig. 4) |
| `su/topology.py` | Def. 4, 5; III-C | 7 edge types; Algorithms 2 and 3 |
| `su/optimize.py` | III-B | Eq. 34-52; Algorithm 1; PSO (Eq. 50-51) |
| `su/displacement.py` | III-A-3 | Eq. 14-29; Hermite slip, drag, scaling law |
| `su/stratigraphy.py` | III-D, IV-B-1 | Eq. 54-56; reflectivity, Ricker, noise |
| `su/model.py` | Fig. 1 | assembles one (seismic, label) pair |
| `patterns/su_patterns.py` | Fig. 15 | the five representative patterns |

### Run

```bash
PY=/hdd1/hukaixiao/projects/FAULTSEG3D/.venv/bin/python

$PY generate.py       --per-category 10 --out data/su_repro
$PY qc/visualize.py   --root data/su_repro --out qc/out      # 2-D section grid
$PY qc/sample_slices.py --root data/su_repro --n 20          # random slices
$PY qc/render3d.py    --root data/su_repro --per-category 1  # 3-D, paper style
$PY qc/analyze.py     --root data/su_repro                   # statistics
```

Output layout mirrors the archived project so downstream code can be reused:

```
seismic/*.dat     float32 [z, y, x]
labels/*.dat      uint8   [z, y, x]
metadata/*.json   tree, zeta, PSO penalty, per-fault dip/strike/throw
surfaces/*.npz    each fault's parametric Bézier surface, (n_u, n_v, 3)
```

### 3-D rendering

`qc/render3d.py` reproduces the two views the paper uses:

* **fault network** — every fault's Bézier surface, one colour each, in a
  wireframe box (Fig. 15 middle panel, Fig. 10e). Drawn from the exported
  parametric surfaces, not from an isosurface of the label, so it is the true
  geometry. Shading is computed here and passed as `facecolors`; matplotlib's
  own `shade=True` leaves visible facet seams on a dense grid.
* **seismic block** — three orthogonal faces with the label burnt on in red
  (Fig. 16, 19, 22). `at="outer"` gives the block faces; the default `"auto"`
  picks interior planes via `best_slices`.

`best_slices` scores a candidate plane by crisp trace pixels minus smeared
ones. Scoring by raw label count instead selects exactly the planes lying
nearly parallel to a fault, where the thin sheet smears into a blob. Some
smearing is unavoidable: with four or five faults at different attitudes no
single plane cuts all of them at a high angle.

### Fidelity notes

Everything below is a place where the published text is ambiguous or
incomplete. Each is marked in the source with `NOTE ON FIDELITY`.

1. **Algorithm 3, lines 13-15 are garbled in the PDF** — `C_c¹` is used before
   it is defined and two lines carry the same label. The sets in
   `topology.py` are reconstructed from the prose in Section III-C (X-type
   edges high in the tree, Y-type low). The reconstruction reproduces the
   failure mode the paper reports in Section V, which is evidence it is right.
2. **Eq. 35/36** label `l_k` the minimum and `m_k` the maximum but, as
   printed, give `l_k > m_k`. We order the two bounds explicitly.
3. **Algorithm 1, line 22** mixes `i` and `j` inconsistently with the
   symmetric test on line 25. We implement the symmetric reading.
4. **Fig. 7 drag operators** are plotted but never given in closed form. Our
   splines reproduce the plotted behaviour (normal drag decays monotonically
   from the fault; reverse drag overshoots, producing rollover).
5. **Branch reference point** — the paper says "any one of these points serves
   as the reference point". We use the centroid of the three, which is the
   only choice that does not bias branches toward one corner of the main fault.
6. **Branch extent** (`half_len`, `half_wid`) is not specified; we sample it
   as a ratio of the main fault.
7. **Label thickness** is not specified. We mark voxels within 0.75 voxel of
   the surface. Note the paper defines **no visible-throw threshold** — the
   archived stage-1 generator added one (`min_label_throw_abs`); that was a
   local addition, not part of Su et al.
8. **Algorithm 4** (convex-hull constraint on the Bézier parameter domain,
   Eq. 53) **is implemented** in `su/hull_constraint.py`, with one deliberate
   relaxation: hull disjointness is tested against face normals only, which is
   a *sufficient* condition, so the admissible domain is slightly smaller than
   the exact one. Measured effect is near zero, for a reason worth recording —
   see F7 in `docs/step1_findings.md`.
9. **Surface sampling density** must exceed the voxel pitch or the rasterised
   label fragments; `n_u = n_v = 192` is the tested default.

### Reproduction status

Every equation of Section III (Eq. 1-56) and all four algorithms are
implemented. Algorithm 2/3 are verified against the paper's own claim about
invalid configurations (F8); Algorithm 4 is implemented but has no measurable
effect for the reason given in F7. Two items remain open:

* **Algorithm 1** (fidelity note 3) — corrupted in print, reconstructed. Deep
  trees converge on 40-80% of runs instead of the paper's claimed always.
* **Unspecified constants** — Fig. 7 drag splines, branch extent ratio, label
  thickness, branch reference point, and the pattern mix of MultiFaultStyle3D
  are not given numerically anywhere in the paper.

### Open discrepancy: PSO convergence on deep trees

Replicating the Section IV-A-1 experiment (all edges `("Y","Y")`, 10 runs each):

| tree | ours: converged | ours: mean iters | paper: iters |
|---|---:|---:|---:|
| Tree 1 — 3 faults, depth 2 | 100% | 1.0 | 8 |
| Tree 2 — 4 faults, depth 2 | 100% | 1.6 | 13 |
| Tree 3 — 4 faults, depth 4 | 80% | 46.5 | 15 |
| Tree 4 — 7 faults, depth 3 | 40% | 126.1 | 29 |

Shallow trees reproduce cleanly and cheaply. Deep trees do not: the paper
reports that the optimiser "consistently converges to a zero-penalty solution",
while ours stalls on 20-60% of runs. Depth is exactly what drives Algorithm 1
into Case 1's grandparent test and Case 3's Y-span test — the two branches
whose printed form is ambiguous (fidelity notes 1 and 3). Resolving this needs
either the authors' code or a corrected Algorithm 1; until then, treat deep-tree
topologies as approximate.
