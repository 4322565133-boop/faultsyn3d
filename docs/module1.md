# Module 1 — geometry-aware objective

Our first contribution. Code in `su/geo_objective.py`; enabled with
`GenConfig.use_geometry_objective=True`, off by default so the Su reproduction
is untouched and the ablation is a single switch.

## What it adds

Su et al. optimise topology only. The edge constraints of Eq. (43)-(45) are
pure *sign* conditions on `Δ₀·Δₖ` — they constrain direction, never magnitude.
`Δ = 1` satisfies them as well as `Δ = 100`.

```
F(s)  = F_topology(s)                                    # Su, Eq. (49)
F'(s) = F_topology(s) + w_d·G_dip + w_i·G_int + w_p·G_dup
```

Every term is closed-form in the same decision variables `ζ`, via
Eq. (52) → Eq. (1) → Eq. (2,3). A few hundred flops per evaluation, negligible
against the topology penalty loop.

| term | targets | form | weight |
|---|---|---|---:|
| `G_dip` | **F2** — branch dip uncontrolled, ±35° around the main fault | quadratic hinge outside the Andersonian dip band, plus a hard 12° floor | 1 |
| `G_int` | **F7** — 23 % of "non-intersecting" pairs do intersect | corner-sign test: does j's rectangle straddle i's plane inside the grid | 2 |
| `G_dup` | degenerate solutions in the flat region the hinges leave | near-parallel normals **and** close centres | 3 |

Dip bands are deliberately **wide and overlapping** (extensional 35–80°,
strike-slip 55–90°, compressional 15–55°). Pinning each pattern to a narrow
range is exactly the confound that let the archived stage-1 dataset teach a CNN
a dip shortcut. The bands only exclude the geologically degenerate cases.

## Ablation — every surviving term earns its place

Five arms, 50 volumes each, identical seeds, only the objective differs.
All metrics are computed from the generated data; **no model is trained**.
`semblance AUC` is a data-quality measure (are label voxels less coherent than
background?), not a model score.

| arm | topo | min dip | <20° | within-vol SD | collapsed | crossing | AUC | AUC listric | fault % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (Su) | 0.08 | 48.2 | 5.0 % | 9.55 | 10.0 % | 17.3 % | 0.707 | 0.569 | 1.21 |
| **all three** | **0.04** | **57.5** | **0.0 %** | 8.65 | **10.0 %** | **8.4 %** | 0.726 | 0.587 | 1.18 |
| − `G_dip` | 0.08 | 46.6 | **6.2 %** | 10.38 | 8.0 % | 10.3 % | 0.709 | 0.583 | 1.20 |
| − `G_int` | 0.04 | 57.7 | 0.0 % | 8.36 | 8.0 % | **16.9 %** | 0.722 | 0.596 | 1.20 |
| − `G_dup` | 0.04 | 58.7 | 0.0 % | **6.75** | **18.0 %** | 11.4 % | 0.718 | 0.604 | 1.18 |

Each term, dropped alone, degrades the metric it claims responsibility for, and
lands back near the Su baseline:

* **`G_dip`** — sub-20° branches return (0.0 → 6.2 %, baseline 5.0 %), minimum
  dip falls 57.5 → 46.6 (baseline 48.2).
* **`G_int`** — crossings return (8.4 → 16.9 %, baseline 17.3 %).
* **`G_dup`** — collapsed volumes nearly double (10.0 → 18.0 %).

Topology is not sacrificed: 0.08 → 0.04, slightly better than the baseline.
Fault fraction stays in the 1.18–1.21 % band, so nothing degenerates.

### `collapsed` has to be measured within a volume

`G_dup` prevents a *within-volume* failure: several branches converging on one
attitude. An aggregate dip standard deviation pooled over all volumes cannot
see it — our first attempt used exactly that and concluded, wrongly, that
dropping `G_dup` *improved* things. The metric above is the median per-volume
standard deviation, plus the fraction of volumes whose branches fall inside
3°. Under that metric the sign reverses and matches the 3-D renders.

## The term we removed, and why it mattered

The module originally carried a fourth term, `G_obs`, penalising vertical
throw `d_max·sin(δ)` below the seismic tuning thickness λ/4. It was aimed
squarely at **F6** — listric labels reach a semblance AUC of only 0.562 against
background, essentially chance.

**Ablation killed it.** Dropping `G_obs` moved overall AUC by +0.000 and listric
AUC by +0.002, both inside noise. The redundancy is provable in closed form:
for a typical branch (`d_max ≈ 14`) it demands dip ≥ 19°, while the dip band
already demands ≥ 35°. `G_dip` strictly dominates it; the region `G_obs`
constrains is a subset of one already constrained.

Three things came out of that negative result.

**It located the real cause of F6.** Raising dip cannot fix observability,
because the unobservable label voxels sit in the fault's *tip* region, where the
slip field ξ itself tapers to zero. `ξ·sin(δ)` stays ≈ 0 whatever the dip is.
Observability is a property of the **labelling rule**, not of the geometry.
That moves Module 6 from optional to the only available fix — a conclusion we
reached by falsifying a wrong design, not by guessing.

**A redundant term does not merely add dead weight — it destroys the ablation's
diagnostic power.** With `G_obs` present, dropping `G_dip` left sub-20° branches
at 0.0 %, because `G_obs` silently covered for it. Only after removing `G_obs`
did dropping `G_dip` produce the 6.2 % that reveals its true contribution. The
four-term ablation could not measure its own components.

**It makes the surviving terms credible.** An ablation that never kills
anything is not evidence. This one killed a term we had designed, argued for,
and shipped.

## Reference point

Branch faults are anchored at `P0` (`GenConfig.branch_ref_point = 0`).
Definition 3 restricts the reference point to {P0, P1, P2}; our first
implementation used the centroid of the three, which is outside that set. The
choice shifts a branch centre by a median of 70.7 voxels — more than a fault's
own half-length — while leaving dip and strike bit-identical, since those come
from the normal of all three points. Switching to `P0` also visibly improved
the splay morphology: branches now emanate from one end of the main fault,
closer to Fig. 15, instead of sitting around its middle.

## Verified visually

`qc/compare_module1.py` renders both arms side by side from the same seed. This
check is not decorative: on the three-term-plus-`G_obs` version every number was
green — topology 0.00, sub-20° branches eliminated, crossings down — while the
render showed the horsetail's three splays collapsed onto dips **[89, 89, 89]**,
three near-coincident copies of the main fault. `G_dup` exists because of that
figure, and the numeric metric that detects it (`collapsed`) was designed after
the fact to match what the render showed.

> Feasibility constraints make each fault admissible. They say nothing about
> the faults differing from one another. **Admissible is not sufficient.**

The same criticism applies to Su's own objective: the baseline horsetail comes
out at [38, 36, 42], which for a near-vertical wrench system is equally wrong.
Both degenerate; they just degenerate differently.

## Limitations, stated

* **`G_dup` reduces collapse but does not eliminate it.** Collapsed volumes fall
  from 18 % (without the term) to 10 % (with it), not to zero. The horsetail
  example in `qc/out/module1/module1_ab_rep0.png` happens to be one of the
  residual cases: Module 1 correctly lifts its branches into the wrench band
  ([38,36,42] → [63,59,60]) but their within-volume spread is still only 1.7°.
  The honest claim is "significantly reduced", not "solved".
* **Weights `(1, 2, 3)` were set by hand and never tuned.** No sensitivity study
  has been run.
* **Sample size is 10 volumes per category per arm.** The `AUC listric` column
  is non-monotonic across arms (0.583 / 0.596 / 0.604 against 0.587 for the full
  objective) with no consistent pattern and a spread inside ±0.02; we read that
  as noise, not effect, and do not draw conclusions from it.
* **`positive_flower` is the one category where geometry and topology conflict**
  — in the four-term version its topology penalty rose 0.25 → 0.50. It is the
  depth-4 chain, the same tree F4 identifies as hardest for the optimiser.
* **The intersection metric embeds an interpretation.** It counts pairs with no
  direct X/Y edge and no shared Y-parent. For the flower categories that
  includes grandparent–grandchild pairs, whose relationship the paper handles
  through Algorithm 1's Case 3 — a Y-span test that forbids one node falling
  inside the other's span, not full non-intersection. Our criterion is
  therefore **stricter than the paper's**, and the flower baselines should not
  be quoted as "Su violates his own constraint". The en_echelon and listric
  numbers, whose pairs are genuine `(P,P)` siblings, carry no such caveat.
