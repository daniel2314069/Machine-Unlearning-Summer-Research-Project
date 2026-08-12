# Anchor-minimum exact orthogonal control

## Scope and answer

This matrix-only control reuses the frozen float64 C/D rows and the exact-feasible-family implementation from the preceding control. Variant E minimizes anchor feature movement over the full exact orthogonal family by calling the same constrained Procrustes solver with `S_x=YY^T`. The qualified dogs, fruits, and balls Joint settings, including the matched balls anchors, remain unchanged. No image, evaluator, checkpoint, AFR implementation, or production OCE change was created.

**Classification: Outcome E1 — Exact orthogonal mapping has an unavoidable anchor-feature cost.**

> Even after explicitly optimizing anchor preservation over the entire exact orthogonal feasible family, exact target-subspace alignment still requires substantial movement of the original anchor features. Therefore the previous D2 result is not an artifact of omitting anchors from the frozen preservation covariance. This closes the remaining algebraic caveat and provides a direct motivation for an anchor-fixed non-orthogonal relaxation.

Variant E is an oracle-like control, not a repaired OCE method or a proposed editor.

## Derivation and orientation

For `P = H Q G^T + H_perp Q_perp G_perp^T`, substituting into `tr(P^T S_x)` yields `tr(Q^T H^T S_x G) + tr(Q_perp^T H_perp^T S_x G_perp)`. Standard Procrustes therefore gives `Q=U1 V1^T` from `H^T S_x G=U1 Sigma1 V1^T` and `Q_perp=U2 V2^T` from `H_perp^T S_x G_perp=U2 Sigma2 V2^T`. With `S_x=YY^T`, orthogonality makes minimizing `||PY-Y||F^2` equivalent to this trace maximization. Because `Y` lies in `span(H)`, the complement block can be zero or rank-deficient; its SVD completion need not be unique and is not treated as a solver failure.

## Synthetic QA

All four float64 tests passed for `d=6`, `r=2`:

- exact feasibility: orthogonality `1.3270958e-15`, leakage `8.1684407e-31`, projector residual `8.0938807e-16`, mapping residual `5.2277094e-16`;
- anchor optimality: closed-form normalized drift `0.66036073` versus best of 256 random feasible maps `0.66036398`;
- closed-form value: direct trace `3.054343`, nuclear-norm sum `3.054343`, error `0`;
- known `G=H` case: normalized anchor drift `8.4569703e-31` and mapping residual `1.3609179e-15`.

## C/D/E group means

| Group | C anchor | D anchor | E anchor | E-C anchor | E/C anchor | E-D anchor | C frozen-S | D frozen-S | E frozen-S | E-C frozen-S | E/C frozen-S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dogs | 0.14672264 | 0.81160249 | 0.71371628 | 0.56699364 | 5.5309334 | -0.097886205 | 0.12014007 | 0.054788757 | 1.67655 | 1.5564099 | 19.052653 |
| fruits | 0.10031214 | 0.84634429 | 0.77264553 | 0.67233339 | 9.0599314 | -0.073698754 | 0.12455055 | 0.057132923 | 1.6751622 | 1.5506116 | 17.315937 |
| balls | 0.24355294 | 0.65593051 | 0.53560332 | 0.29205038 | 2.4115466 | -0.12032719 | 0.12379583 | 0.063533812 | 1.6590318 | 1.535236 | 17.229509 |

## C/D/E group medians

| Group | C anchor | D anchor | E anchor | E-C anchor | E/C anchor | E-D anchor | C frozen-S | D frozen-S | E frozen-S | E-C frozen-S | E/C frozen-S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dogs | 0.13976682 | 0.83530504 | 0.69351228 | 0.54145003 | 5.3794238 | -0.10038009 | 0.098083235 | 0.052507549 | 1.8113185 | 1.7039773 | 18.912637 |
| fruits | 0.097246397 | 0.81446989 | 0.7589267 | 0.62003128 | 6.0499547 | -0.07040558 | 0.10695471 | 0.057656493 | 1.776815 | 1.6552943 | 16.985375 |
| balls | 0.23975962 | 0.65574521 | 0.54018523 | 0.26253306 | 1.9564286 | -0.11957783 | 0.10946153 | 0.06551242 | 1.8006983 | 1.6912368 | 16.616788 |

## C/D/E overall aggregate

| Variant | Mean leakage | Median leakage | Mean anchor drift | Median anchor drift | Mean frozen-S distortion | Median frozen-S distortion |
|---|---:|---:|---:|---:|---:|---:|
| C_objective_faithful | 0.93841479 | 0.94455772 | 0.16352924 | 0.14833479 | 0.12282882 | 0.10186259 |
| D_exact_orthogonal | 1.6499912e-29 | 1.4731917e-29 | 0.77129243 | 0.78614579 | 0.058485164 | 0.058350919 |
| E_anchor_optimal_exact_orthogonal | 1.6598806e-29 | 1.4698259e-29 | 0.67398838 | 0.65837972 | 1.670248 | 1.8035917 |

Every `anchor` quantity means **anchor feature drift at the edited layer** only.

## Layer distributions

| Quantity | Min | Q25 | Median | Q75 | Max | Material layers |
|---|---:|---:|---:|---:|---:|---:|
| E true leakage | 5.7221879e-30 | 1.154281e-29 | 1.4698259e-29 | 2.0272804e-29 | 3.3099102e-29 | 48/48 checked |
| E normalized anchor drift | 0.18142967 | 0.35678303 | 0.65837972 | 0.98292713 | 1.2860394 | n/a |
| E-C anchor drift | -0.13668441 | 0.1187385 | 0.49684451 | 0.85090001 | 1.2134027 | 43/48 |
| E/C anchor drift | 0.5837212 | 1.7987982 | 3.933901 | 7.45276 | 20.473331 | n/a |
| E-D anchor drift | -0.1909888 | -0.12600017 | -0.096114187 | -0.073713514 | -0.01810514 | n/a |
| E normalized frozen-S distortion | 1.0440117 | 1.477624 | 1.8035917 | 1.94031 | 1.9812018 | n/a |
| E-C frozen-S distortion | 0.85172296 | 1.2941864 | 1.6906451 | 1.8709834 | 1.9097606 | 48/48 |
| E/C frozen-S distortion | 4.7274821 | 7.4219224 | 17.737065 | 27.746168 | 33.169718 | n/a |

## Q1 — Does E keep numerical-zero leakage in all 48 layers?

**Yes.** Maximum E leakage is `3.3099102e-29` under the `1e-10` fail-closed threshold. Maximum exact mapping residual is `2.2018884e-14` and maximum `||P^T P-I||F` is `2.203682e-11`. All 48 cases retain target rank = anchor rank = 12.

## Q2 — What is the minimum achievable anchor drift?

Overall E median is `0.65837972`. Overall E-C median is `0.49684451`, E/C median is `3.933901`, and E-D median is `-0.096114187`. Group medians: `dogs` E drift `0.69351228`, E-C `0.54145003`, E/C `5.3794238`; `fruits` E drift `0.7589267`, E-C `0.62003128`, E/C `6.0499547`; `balls` E drift `0.54018523`, E-C `0.26253306`, E/C `1.9564286`. The complete min/Q25/median/Q75/max distributions are above. E is materially above C in `43/48` layers using the frozen absolute (`0.05`) and ratio (`1.25`) thresholds.

## Q3 — Does minimizing anchor drift worsen frozen-S preservation?

Overall E normalized frozen-S median is `1.8035917`. Overall E-C median is `1.6906451` and E/C median is `17.737065`. Group medians: `dogs` E frozen-S `1.8113185`, E-C `1.7039773`, E/C `18.912637`; `fruits` E frozen-S `1.776815`, E-C `1.6552943`, E/C `16.985375`; `balls` E frozen-S `1.8006983`, E-C `1.6912368`, E/C `16.616788`. E is materially above C in `48/48` layers under the frozen preservation thresholds (`0.01`, `1.25`).

## Q4 — Was D's large anchor drift only an optimizer-choice artifact?

The answer follows from E, not from D: E is the closed-form minimum-anchor-drift member of the entire exact feasible family. Its residual anchor cost relative to C and its separately evaluated frozen-S cost are reported above. Under the preregistered decision rule, the primary result is `E1`.

## Q5 — Final classification

**Outcome E1 — Exact orthogonal mapping has an unavoidable anchor-feature cost.** Even after explicitly optimizing anchor preservation over the entire exact orthogonal feasible family, exact target-subspace alignment still requires substantial movement of the original anchor features. Therefore the previous D2 result is not an artifact of omitting anchors from the frozen preservation covariance. This closes the remaining algebraic caveat and provides a direct motivation for an anchor-fixed non-orthogonal relaxation.

**Algebraic gate passed. The next step is AFR implementation with a pure-projection ablation.**

## Optional fixed Pareto diagnostic

Not triggered because the primary C/D/E result was decisive.

Pareto triggered: `no`. Pareto sweet spot under the preregistered small-effect rule: `no`.

## Reproducibility and QA

- CSV: `results_anchor_min_control.csv` (144 rows = 48 cases x 3 controls)
- Optional Pareto CSV: `not created`
- Computation dtype: float64 after loading frozen production tensors
- Config SHA-256: `416ad7fd9e7666f8cd295ef6de4c6cf6af26d67502fd21d4027b6a81aa7e762b`
- Anchors SHA-256: `392802a728f5f726870718c2f9d0885ed57c8e16232899710f22690cee6c13b1`
- Qualification SHA-256: `b96514c7bd94e4d703079ddff238731c05ba76cde7fe6b0c2643d404bf7f043f`
- K0 SHA-256: `5d8ee50935eb3d22e1f9bc84947572afba386d05e063f70ea161c5cbf1e16235`
- Frozen C/D CSV SHA-256: `546789d6d939cccfa57551994630be7e93ebc4ff0a05527d68dfe32ba5722343`
- Rank relative-tolerance range: `7.1054274e-14` to `2.8421709e-13`
- Maximum anchor-support residual `||(I-HH^T)Y||F`: `1.225969e-12`
- Maximum anchor-objective nuclear-norm trace error: `3.3105607e-10`
- Endpoint optimality cross-checks passed: E anchor drift <= D anchor drift and D frozen-S loss <= E frozen-S loss in every layer (tolerance `1e-8`)
- Variant E maximum orthogonality residual: `2.203682e-11`
- Variant E determinant is metadata only; primary constraint is O(d)
- Runtime: `55.5` seconds
