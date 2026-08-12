# Exact orthogonal target-subspace mapping control

## Scope and answer

This final matrix-only control uses the frozen qualified Joint settings for `dogs`, `fruits`, and `balls` across the same 16 edited `attn2.to_v` layers. Targets, matched per-target anchors (including `basketball` and `baseball` for balls), prompt expansion, K0, local retain concepts, scales, and layers are unchanged. All audit linear algebra and metrics are float64. No image, image evaluator, editor checkpoint, AFR implementation, or production OCE change was created.

**Classification: Outcome D2 — Exact orthogonal mapping has a clear preservation cost.**

> Pure orthogonality can satisfy the geometric target-subspace mapping, but only at a substantial preservation / anchor-feature cost. This supports a genuine incompatibility between exact orthogonal target mapping and preservation, motivating the anchor-fixed non-orthogonal relaxation.

`D_exact_orthogonal` is an oracle-like best feasible orthogonal control, not a repaired OCE method or a proposed contribution.

## Derivation and orientation

For `P = H Q G^T + H_perp Q_perp G_perp^T`, orthonormal completion makes `P` orthogonal and gives `P G = H Q`. Cyclic trace expansion yields

`tr(P^T S) = tr(Q^T H^T S G) + tr(Q_perp^T H_perp^T S G_perp)`.

Thus the constrained maximization separates into two standard Procrustes problems. If `H^T S G = U1 Sigma1 V1^T` and `H_perp^T S G_perp = U2 Sigma2 V2^T`, the correct orientations are `Q*=U1 V1^T` and `Q_perp*=U2 V2^T`. The achieved maximum is the sum of the two nuclear norms. Because `P` is orthogonal and `S` symmetric, minimizing `tr[(P-I)S(P-I)^T]` is equivalent to maximizing `tr(P^T S)`.

## Synthetic tests

All three float64 tests passed for `d=4`, `r=2`:

- feasibility: orthogonality residual `2.8324215e-15`, true leakage `3.9024155e-32`, projector-mapping residual `4.9231929e-15`, and exact mapping residual `2.2048219e-15`;
- feasible-family optimality: closed-form preservation loss `2.5996332` versus best of 256 random feasible transforms `3.8643682`;
- closed-form value: direct trace `14.288118` versus nuclear-norm sum `14.288118`, absolute error `3.5527137e-15`.

## Float64 baseline reproduction

Variant C was rebuilt from frozen inputs as `-lambda_e (I-R*)R + S`, followed by standard O(d) Procrustes. Compared with the prior float32 CSV, maximum absolute leakage difference was `0.00030683194` and maximum relative raw-preservation difference was `0.00015944951`. Both passed the fail-closed thresholds (`0.01` and `0.01`). Float64 Variant C leakage has median `0.94455772` and minimum `0.87599542`; these values are reproduced rather than hard-coded.

## Group-level means

| Group | C leakage | D leakage | C norm. preserve | D norm. preserve | D-C preserve | Preserve ratio | C anchor drift | D anchor drift | D-C anchor drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dogs | 0.93989913 | 1.6877191e-29 | 0.12014007 | 0.054788757 | -0.065351316 | 0.5314065 | 0.14672264 | 0.81160249 | 0.66487984 |
| fruits | 0.94537227 | 1.6077071e-29 | 0.12455055 | 0.057132923 | -0.067417628 | 0.523104 | 0.10031214 | 0.84634429 | 0.74603215 |
| balls | 0.92997297 | 1.6545473e-29 | 0.12379583 | 0.063533812 | -0.06026202 | 0.58563929 | 0.24355294 | 0.65593051 | 0.41237757 |

## Group-level medians

| Group | C leakage | D leakage | C norm. preserve | D norm. preserve | D-C preserve | Preserve ratio | C anchor drift | D anchor drift | D-C anchor drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dogs | 0.9480986 | 1.7893903e-29 | 0.098083235 | 0.052507549 | -0.037190894 | 0.57858549 | 0.13976682 | 0.83530504 | 0.65640395 |
| fruits | 0.95160299 | 1.4055178e-29 | 0.10695471 | 0.057656493 | -0.044798088 | 0.54510439 | 0.097246397 | 0.81446989 | 0.67557447 |
| balls | 0.93928161 | 1.4731917e-29 | 0.10946153 | 0.06551242 | -0.038769613 | 0.61246738 | 0.23975962 | 0.65574521 | 0.37809304 |

`anchor drift` in these tables means **anchor feature drift at the edited layer** only.

## Layer distribution

| Quantity (D-C unless noted) | Min | Q25 | Median | Q75 | Max | Positive/material layers |
|---|---:|---:|---:|---:|---:|---:|
| Normalized preservation difference | -0.17528178 | -0.11488717 | -0.039761433 | -0.02227118 | -0.010977552 | 0/48 positive; 0/48 material |
| Normalized preservation ratio D/C | 0.24257507 | 0.39885074 | 0.57825612 | 0.68851211 | 0.83594028 | n/a |
| Anchor feature drift difference | -0.018645799 | 0.22062304 | 0.62214743 | 0.97695775 | 1.368868 | 46/48 positive; 45/48 material |
| D true leakage | 5.6399322e-30 | 1.1508546e-29 | 1.4731917e-29 | 2.0329061e-29 | 3.3240856e-29 | 48/48 checked |

## Q1 — Can all 48 layers reach numerical-zero leakage?

**Yes.** Maximum Variant D true leakage was `3.3240856e-29` under the preregistered `1e-10` threshold. All 48 layers had target rank = anchor rank = 12. Maximum exact mapping residual was `2.1896612e-14` and maximum `||P^T P-I||F` was `2.5513407e-11`. The projector leakage cross-check also passed in every layer.

## Q2 — What preservation cost does exact mapping require?

Overall median normalized D-C preservation distortion was `-0.039761433` and median ratio D/C was `0.57825612`. Group medians were `dogs` D-C `-0.037190894`, ratio `0.57858549`; `fruits` D-C `-0.044798088`, ratio `0.54510439`; `balls` D-C `-0.038769613`, ratio `0.61246738`. The full layer distribution is reported above; 0/48 layers increased and 0/48 crossed both the absolute (`0.01`) and ratio (`1.25`) materiality thresholds.

## Q3 — Does exact mapping increase anchor feature drift?

Overall median D-C anchor feature drift was `0.62214743`. Group medians were `dogs` D-C `0.65640395`; `fruits` D-C `0.67557447`; `balls` D-C `0.37809304`. Across layers, 46/48 increased and 45/48 crossed both the absolute (`0.05`) and ratio (`1.25`) materiality thresholds.

## Q4 — Interpretation

Statement A is confirmed: equal 12-dimensional ranks make exact orthogonal target-subspace mapping feasible, so the prior high leakage did **not** prove mathematical infeasibility of orthogonal mapping.

Statement B is the actual decision test: whether that exact mapping has a consistent, substantial preservation or anchor-feature cost. The preregistered rule assigns D2 only if a material penalty occurs in at least 36/48 layers and all three group medians; D1 requires clearly small effects, while heterogeneous intermediate evidence is D3. Under that rule, the result is **Outcome D2 — Exact orthogonal mapping has a clear preservation cost**.

The D2 trigger in these data is specifically **anchor-feature drift**, not the frozen `S` preservation loss: 45/48 layers have a material anchor-drift increase, whereas 0/48 have a material `S`-loss increase and the normalized `S` distortion decreases in all 48 layers. These findings are not contradictory because the frozen protocol sets `anchor_in_local_retain=false`, so the anchor feature matrix `Y=WC*` is not itself included in `S`. The control therefore shows that the `S`-optimal exact mapper moves the measured anchor features substantially. It does not solve a separate optimization that minimizes anchor drift over the exact-feasible family, so the D2 conclusion is scoped to the preregistered preservation/anchor-feature decision rule used here.

## Reproducibility and QA

- CSV: `results_exact_control.csv` (96 rows = 48 cases x 2 controls)
- Computation dtype: float64 after loading frozen production tensors
- Config SHA-256: `416ad7fd9e7666f8cd295ef6de4c6cf6af26d67502fd21d4027b6a81aa7e762b`
- Anchors SHA-256: `392802a728f5f726870718c2f9d0885ed57c8e16232899710f22690cee6c13b1`
- Qualification SHA-256: `b96514c7bd94e4d703079ddff238731c05ba76cde7fe6b0c2643d404bf7f043f`
- K0 SHA-256: `5d8ee50935eb3d22e1f9bc84947572afba386d05e063f70ea161c5cbf1e16235`
- Prior solver-audit CSV SHA-256: `330bdee50687b261d343063d11b7d4f52179ea4bb21e1e0f616ff45e9838b28b`
- Rank relative-tolerance range: `7.1054274e-14` to `2.8421709e-13`
- Variant C maximum orthogonality residual: `2.5882559e-11`
- Variant D maximum orthogonality residual: `2.5513407e-11`
- Variant D determinant is recorded only as numerical metadata, not used for interpretation
- Runtime: `255.9` seconds
