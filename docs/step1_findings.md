# Step 1 findings — reproducing Su et al. (2026)

All numbers come from `qc/analyze.py` on `data/su_repro` (10 volumes per
category, 128³, seed 2026), and from the two controlled sweeps in this
document. The archived stage-1 generator is at
`/hdd1/hukaixiao/projects/_archive_20260908_faultsyn_stage1`.

---

## F1. Branch dip is an output of the optimiser, not a parameter

This is the structural core of the method and the thing the stage-1 code got
wrong. In Su et al.:

```
zeta  --Eq.52-->  P0,P1,P2  --Eq.1-->  normal  --Eq.2,3-->  dip, strike
```

Only the **main fault** (i = 0) has a user-chosen dip. Consequence, measured
on the reproduction: branch dips inside one category span a wide, continuous
range, and the ranges of different categories **overlap**.

Branch-fault dip histogram (% of branches per bin, main fault excluded):

| category | median | 0-20° | 20-35° | 35-50° | 50-65° | 65-80° | 80-90° |
|---|---:|---:|---:|---:|---:|---:|---:|
| **reproduction** ||||||||
| en_echelon | 80.1 | 0 | 0 | 0 | 10.0 | 36.7 | 53.3 |
| horsetail | 57.2 | 0 | 10.0 | 20.0 | 33.3 | 30.0 | 6.7 |
| listric_assemblage | 36.4 | 20.0 | 25.0 | 30.0 | 22.5 | 2.5 | 0 |
| negative_flower | 71.4 | 0 | 3.3 | 6.7 | 30.0 | 16.7 | 43.3 |
| positive_flower | 42.3 | 0 | 3.3 | 60.0 | 23.3 | 13.3 | 0 |
| **archived stage-1** ||||||||
| en_echelon | 83.1 | 0 | 0 | 0 | 0 | 21.1 | 78.9 |
| horsetail | 75.3 | 0 | 0 | 0 | 0 | 74.8 | 25.2 |
| listric_assemblage | 36.8 | 0 | 42.4 | 57.6 | 0 | 0 | 0 |
| negative_flower | 67.0 | 0 | 0 | 0 | 42.7 | 57.3 | 0 |
| positive_flower | 54.8 | 0 | 0 | 30.6 | 58.8 | 10.6 | 0 |

Every archived category is confined to two adjacent bins; every reproduced
category spreads over four or five. Measuring the p10-p90 interval overlap
between category pairs:

* reproduction: **9 of 10** category pairs overlap
* archived stage-1: **6 of 15** pairs overlap

So the archived generator made dip almost a category label. The faithful
method does not.

The stage-1 generator instead called `_child(..., dip=rng.uniform(78, 88))`
with a hand-written interval per category, and discarded the PSO solution
entirely in 4 of its 6 categories. That is why its dip ranges were disjoint —
and why a CNN trained on it could learn a dip shortcut.

---

## F2. Branch dip ≈ main dip ± 35°, and nothing bounds it

Controlled sweep (main fault fixed, half-extent 60, 4 branches, `(P,P)` edges,
25 trials each):

| main fault dip | branch p5–p95 | branches < 20° | max deviation from main |
|---:|---:|---:|---:|
| 35° | 6.9 – 60.0° | **20.0 %** | 35.0° |
| 50° | 21.9 – 75.0° | 4.0 % | 35.0° |
| 65° | 38.4 – 86.8° | 0 % | 33.7° |
| 85° | 63.1 – 89.3° | 0 % | 28.1° |

The spread is a constant ±≈35°, set by (domain extent)/(fault extent) — the
lever arm between the three axis origins. It is **not** controlled by `d_Z`:
sweeping `d_Z` over 1, 2, 4, 8 leaves the dip distribution unchanged, because
Eq. (35)/(36) divide the bounds by `d_Z`, so the physical span of `zeta·d_Z`
is invariant. `d_Z` sets granularity only.

**Why this matters.** The edge constraints of Eq. (43)-(45) are pure *sign*
conditions on `Δ₀·Δₖ`; they carry no magnitude bound. So the topology can be
perfectly satisfied (penalty exactly 0) by a branch that is nearly parallel to
bedding. With a low-angle main fault, ~20 % of branches come out below 20° dip,
which is not a geologically meaningful discrete fault and leaves almost no
reflector offset. **Geometric plausibility is simply not part of the objective
function.** This is the clearest opening for our own contribution.

---

## F3. Topology behaves correctly — the label proves it

Connected-component count of the fault label, versus number of faults:

| category | edge type | faults | label components | fault % |
|---|---|---:|---:|---:|
| en_echelon | (P,P) | 4 | 2.5 | 2.24 |
| listric_assemblage | (P,P) | 5 | 2.9 | 2.42 |
| horsetail | (Y,Y) | 4 | 1.1 | 1.38 |
| negative_flower | (X,P) then (Y,Y) | 4 | 1.2 | 1.63 |
| positive_flower | (P,X) then (Y,Y) | 4 | 1.3 | 1.83 |

PSO convergence on the five patterns (10 volumes each):

| category | tree depth | converged | mean iters | sec/volume |
|---|---:|---:|---:|---:|
| en_echelon | 2 | 100% | 1.0 | 4.9 |
| horsetail | 2 | 100% | 1.4 | 5.8 |
| listric_assemblage | 2 | 100% | 1.4 | 7.0 |
| negative_flower | 3 | 90% | 29.8 | 5.4 |
| positive_flower | 4 | 90% | 35.2 | 6.2 |

`P` edges (non-intersecting) leave faults as separate components; `Y` edges
(branching junctions) merge them into one connected network. That is exactly
the intended semantics, and it is an independent check that the truncation
step of Section III-D works.

Fault fraction is 1.4–2.3 %, in line with FaultSeg3D-style data.

---

## F4. Deep trees do not converge as the paper claims

Replicating the Section IV-A-1 experiment (all edges `(Y,Y)`, 10 runs each):

| tree | ours converged | ours mean iters | paper iters |
|---|---:|---:|---:|
| Tree 1 — 3 faults, depth 2 | 100 % | 1.0 | 8 |
| Tree 2 — 4 faults, depth 2 | 100 % | 1.6 | 13 |
| Tree 3 — 4 faults, depth 4 | 80 % | 46.5 | 15 |
| Tree 4 — 7 faults, depth 3 | 40 % | 126.1 | 29 |

We reproduce the paper's *qualitative* conclusion — depth costs much more than
node count — but not its claim of consistent zero-penalty convergence. Depth is
precisely what pushes Algorithm 1 into Case 1's grandparent test and Case 3's
Y-span test, and those are the two branches whose printed form is corrupted in
the PDF. Treat deep-tree topologies as approximate until this is resolved.

Practical impact is limited: the five Fig. 15 patterns are depth 2–4 and
converge 70–100 % of the time; a failure means a slightly imperfect topology,
not a broken volume.

---

## F5. Su et al. define no visible-throw threshold

The label in this reproduction is simply the fault surface (within 0.75 voxel),
restricted to where the slip field is non-zero. There is no minimum-throw rule
anywhere in the paper.

The stage-1 generator added one —
`visible_thresh = max(min_label_throw_abs, min_label_throw_ratio * dmax)`
with 1.8 voxels / 0.20 — which was a local invention. Worth remembering when
comparing label statistics between the two datasets: they are not measuring
the same thing.

---

## F6. Slice-level QC — 20 random 2-D slices per category

`qc/sample_slices.py` draws random (volume, orientation, index) triples.
Figures in `qc/out/slices/`. Three things show up.

**Blank-slice rate** — fraction of uniformly random slices containing no fault
at all: horsetail 0%, listric 5%, positive_flower 5%, en_echelon 17%,
negative_flower 29%. Relevant if we ever train a 2-D/2.5-D model on this data.

**Label thickness is clean in 3-D, but slices can smear.** The 3-D label sheet
is thin everywhere — median thickness exactly 2.00 voxels, p99 2.2-2.9, max
3.4-4.2 across all categories. So the thick amorphous blobs visible in some
panels (e.g. `000019 horsetail inline 71`, 1816 px) are **not** a rasterisation
bug: they are the genuine intersection of a thin fault sheet with a slice plane
it happens to be nearly parallel to. Frequency, over fault-bearing slices:

| category | thickness > 6 px | > 12 px | area > 600 px |
|---|---:|---:|---:|
| en_echelon | 9.8% | 2.0% | 10.5% |
| horsetail | 4.3% | 0.8% | 5.2% |
| listric_assemblage | 4.0% | 0.8% | 16.5% |
| negative_flower | 7.4% | 2.6% | 8.7% |
| positive_flower | 7.1% | 1.8% | 8.9% |

**Ringing near faults is small.** Excess vertical high-frequency energy in a
1-5 voxel shell around the label, relative to the far field: 1.02 (en_echelon)
to 1.06 (listric). Highest for listric, as expected from its curvature, but
2-6% is not a numerical problem.

**Label observability — corrected measurement.** Several panels show a clean
yellow line lying on undisturbed reflectors, so we measured how well the labels
actually track seismic discontinuity.

*Method.* Neidell-Taner C1 semblance: sum amplitudes laterally over a 3x3 trace
window at each time sample, then smooth over 9 samples vertically;
`semblance = <(sum)^2> / <N * sum(a^2)>`, in [0, 1], where 1 means all traces in
the window are identical. Compare the semblance at label voxels against a
random 2% sample of non-label voxels from the same volume, and report the AUC:
the probability that a randomly chosen label voxel is less coherent than a
randomly chosen background voxel. 0.5 means the label carries no information
about seismic discontinuity.

| category | semblance at label | at background | **AUC** | below background p25 (null = 25%) |
|---|---:|---:|---:|---:|
| negative_flower | 0.636 | 0.883 | **0.823** | 75.5% |
| horsetail | 0.744 | 0.889 | **0.735** | 60.2% |
| en_echelon | 0.743 | 0.861 | **0.704** | 56.8% |
| positive_flower | 0.656 | 0.777 | **0.665** | 50.9% |
| listric_assemblage | 0.812 | 0.826 | **0.562** | 32.8% |

Four of five categories are solid — their labels are clearly tied to observable
reflector discontinuity. **`listric_assemblage` is the outlier at AUC 0.562,
barely above chance.** Its labels are close to unobservable in the seismic they
are paired with.

This is consistent with F2 by an independent route: listric faults are
low-angle, vertical throw is `slip x sin(dip)`, so a low-dip fault leaves a
proportionally weaker reflector offset. Two separate measurements — the dip
sweep and the semblance AUC — point at the same failure.

> **Correction.** An earlier version of this document reported "33-44% of
> labels have no seismic evidence" for every category. That used a
> non-standard coherence window that mixed vertical wavelet oscillation into
> the lateral-coherence estimate, and thresholded at the background *median*,
> whose null rate is 50% rather than 0. Both are fixed above. The conclusion
> narrowed considerably: the problem is specific to `listric_assemblage`, not
> general to the method.

---

## Implementation issues found and fixed

1. **Label fragmentation.** Rasterising against a surface sampled at
   `n_u = n_v = 128` broke a 4-fault label into 19 connected components,
   because sample spacing exceeded the voxel pitch. `192` gives the correct
   2–4. Fixed; now the default.
2. **Uniqueness constraint (Eq. 39).** Enforcing it as a soft penalty made the
   landscape spiky under integer rounding. Replaced with a repair operator
   (`optimize.decode`). Helps Tree 3 (65 → 46 iterations); does not fix F4.

---

## F7. Algorithm 4 is implemented — and it barely matters

`su/hull_constraint.py` implements Eq. (53) / Algorithm 4: per node, the
admissible Bézier domain (alpha, beta) is intersected with the set that keeps
its control-point hull disjoint from every earlier node not joined to it by an
X or Y edge in that direction. Enabled by `GenConfig.use_hull_constraint`;
28-35 curve parameters get clamped on a typical volume; cost is about +20 %
generation time.

Ablation, measuring the minimum gap between surface pairs the topology declares
non-intersecting, counting only points inside the voxel grid (Section III
defines "P" as non-intersecting *in the voxel grid*):

| category | Algorithm 4 | pairs | gap < 1 vox | gap < 2 vox | median gap |
|---|---|---:|---:|---:|---:|
| en_echelon | off | 48 | 22.9% | 25.0% | 13.43 |
| en_echelon | **on** | 48 | 22.9% | 25.0% | 12.93 |
| listric_assemblage | off | 76 | 23.7% | 26.3% | 27.47 |
| listric_assemblage | **on** | 76 | 23.7% | 26.3% | 27.79 |

No effect. The reason is not a bug: measuring the *unbent* reference planes
gives the same 25.0% / 26.7%, i.e. **the surfaces already intersect before any
Bézier bending is applied**. Algorithm 4 only constrains the bending (Fig. 11b),
so it cannot touch intersections that come from the placement stage. The
admissible domain is also not collapsing (215/325 cells survive for
en_echelon, 315/325 for listric), so the constraint is not over-tight either.

**Mechanism.** Two distinct pathologies, both admitted by the sign-only
constraints of Eq. (43)-(45):

| max abs(delta zeta) | pairs | gap < 1 vox | median gap |
|---:|---:|---:|---:|
| 0-2 | 1 | 100% | 0.50 |
| 2-5 | 5 | 60% | 0.19 |
| 5-10 | 9 | 0% | 13.28 |
| 10-20 | 65 | 26.2% | 17.74 |
| 20+ | 73 | 2.7% | 56.15 |

* **small offset** — `(P,P)` is satisfied by `delta = 1`, so a branch may sit a
  single `d_Z` (4 voxels) from its parent, essentially coincident. Eq. (39)
  only forces `zeta_0` to differ by one unit.
* **large differential offset** — very different offsets on the three axes tilt
  the branch strongly relative to the main fault, and two non-parallel planes
  always meet along a line; when that line falls inside the domain they cross.
  This is the same lever that produces the +-35 deg dip spread of F2.

Branch extent modulates it but does not explain it away (flat planes, no
bending): extent ratio 0.25 -> 8.4 % intersecting, 0.45 -> 12.0 %,
0.65 -> 15.0 %, 0.85 -> 15.7 %. Note the paper never specifies branch extent;
our 0.45-0.85 is a local choice and must be declared as such.

**For the paper:** Su et al. put "non-intersecting" in the definition of the
`P` edge, but nothing in the objective enforces it. About 23 % of pairs
declared non-intersecting do intersect inside the voxel grid. Together with
F2 this is the second independent piece of evidence that **magnitude bounds on
geometry are missing from the objective function**, which is exactly what our
Module 1 adds.

---

## F8. Algorithm 2/3 verified — and a latent bug it exposed

The tree-synthesis path (`build_tree` / `validate_edge`) existed but was never
executed, because dataset generation uses the hardcoded Fig. 15 topologies.
Exercising it immediately surfaced a real bug: Algorithm 2 line 8 shuffles the
edge **pool**, so the loop must iterate over edge *values*; our code iterated
over a permutation of indices into the original list, which goes out of range
as `pool.remove(e)` shrinks it. Fixed.

**Falsifiable test.** Section V of the paper states that geologically
inconsistent edge configurations "inevitably cause fault intersections" and
fail to reach zero penalty. If our reconstruction of `ValidateEdge` captures
the real rules, trees filtered through it should converge better than trees
built with the same edge pool and out-degree limit but no validation at all.

Raw comparison, 40 trees per cell:

| pool | builder | invariants held | PSO converged | mean iters | mean depth |
|---:|---|---:|---:|---:|---:|
| 3 | Algorithm 2 | 100% | 100% | 2.0 | 2.00 |
| 3 | no validation | 85% | 75% | 57.6 | 3.15 |
| 4 | Algorithm 2 | 100% | 88% | 31.2 | 3.00 |
| 4 | no validation | 92% | 50% | 108.2 | 3.38 |
| 6 | Algorithm 2 | 100% | 40% | 133.0 | 3.25 |
| 6 | no validation | 92% | 20% | 168.8 | 4.00 |

`invariants held` is checked by an independently written predicate, so 100%
confirms `validate_edge` enforces what it is meant to.

**Depth is a confound**, though: the unvalidated builder also produces deeper
trees, and depth independently hurts convergence (F4). Controlling for both
depth and node count over 260 paired trials:

| depth | nodes | Algorithm 2 | no validation |
|---:|---:|---:|---:|
| 2 | 4 | 100% (n=51) | 100% (n=5) |
| 3 | 5 | 81% (n=74) | 66% (n=50) |
| 3 | 6 | 54% (n=61) | 42% (n=19) |
| 3 | 7 | 31% (n=58) | 15% (n=20) |
| 4 | 6 | 67% (n=6) | 30% (n=37) |
| 4 | 7 | 30% (n=10) | 18% (n=34) |
| | **matched mean** | **60%** | **45%** |

The advantage survives the control — every non-saturated cell favours the
validated builder — but it is **smaller than the raw numbers suggest**: 1.33x
rather than ~2x. Part of the raw gap was the depth confound.

Conclusion: the Algorithm 3 reconstruction is directionally right. The residual
40-70% non-convergence at larger pools is F4's problem (Algorithm 1's corrupted
Case 1 / Case 3), not something `ValidateEdge` can fix.

---

## Still to do before Step 2
- Resolve F4, or document it as a limitation of the reproduction.
- Decide the grid size: the paper's own experiments use (256, 256, 128);
  we currently generate 128³ to match the archived benchmark.
