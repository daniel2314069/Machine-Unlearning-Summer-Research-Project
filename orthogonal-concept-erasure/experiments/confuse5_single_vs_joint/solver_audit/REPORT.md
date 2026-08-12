# OCE Confuse5 matrix-level solver audit

## Scope and result

This audit uses only the frozen, baseline-qualified Joint groups `dogs`, `fruits`, and `balls`, with targets, matched per-target anchors, prompt expansion, local retain concepts, K0, scales, and the 16 edited `attn2.to_v` layers resolved from the existing Confuse5 protocol. It generated no images and changed no production weights.

**Classification: Outcome B — Orthogonality gap remains.**

> The leakage cannot be explained away by QR rank inflation or Procrustes orientation. This provides a valid algebraic motivation for testing the proposed anchor-fixed non-orthogonal correction.

## Code and paper convention audit

- Paper main text Eq. 17 constructs `G = orth(W C1)`, `G* = orth(W C*)`, `R = G G^T`, and `R* = G* G*^T`. Eq. 18 states `min -||P R - (I-R*)||F^2 + preserve(P)`. Eq. 19 writes `max tr(P M_total)`, while Eq. 20 gives `M_total = -R(I-R*) + S`; the following sentence applies `P=U V^T` to the SVD of that matrix.
- Appendix A.2 instead uses the standard convention `max tr(P^T M_e)`, derives `M_e = -(I-R*)R`, adds the symmetric preservation matrix, and uses `P=U V^T`.
- Released `oce.py` and the frozen Confuse5 checkpoint path use reduced QR, `-R(I-R*) + S`, `U V^T`, and then flip the last column of the already-computed transform when its determinant is negative.
- Direct expansion gives `-lambda_e ||P R-A||F^2 + preserve(P) = const - 2 tr(P^T[-lambda_e A R+S])`, where `A=I-R*`. Thus Appendix A.2 is objective-faithful under standard Procrustes. Main-text `tr(P M)` with `M=-R A+S` is equivalent only when solved in that convention (its maximizer is `V U^T`); pairing that matrix with `U V^T` changes the solved objective.

The rank-revealing basis applies SVD to the same L2-normalized projected columns used by released QR and retains singular values satisfying `sigma_i > max(m,n) * eps(float32) * sigma_max` (zero absolute tolerance). All three variants are evaluated against the same rank-revealed `R`, `R*`, and exact Eq. 18 loss. Preservation includes the frozen weighted local-retain, K0, and repository regularizer quadratic terms.

## Synthetic checks

All checks passed. `tr(P^T M)` at `P=U V^T` was `4.9780005`, equal to the nuclear norm `4.9780005`. The objective-faithful synthetic Eq. 18 value was `-2.9901011` versus `-2.9901011` for the transposed-erasure orientation. The 2D projector test verified `P R P^T`, not `P R` (error `0`).

## Aggregate diagnostics

| Group | Variant | Layers | Mean paper objective | Median leakage | Max leakage | Mean anchor drift |
|---|---|---:|---:|---:|---:|---:|
| dogs | A_released_oce | 16 | -793355.97 | 0.9672236 | 0.98773766 | 0.14721465 |
| dogs | B_rank_corrected_released_oce | 16 | -793331.19 | 0.96722027 | 0.98779202 | 0.14773837 |
| dogs | C_objective_faithful_oce | 16 | -794430.29 | 0.94820575 | 0.97868903 | 0.14673406 |
| fruits | A_released_oce | 16 | -792593.55 | 0.96701443 | 0.99078703 | 0.10100712 |
| fruits | B_rank_corrected_released_oce | 16 | -792580.54 | 0.96700946 | 0.99081198 | 0.10104811 |
| fruits | C_objective_faithful_oce | 16 | -793756.09 | 0.95172695 | 0.98556709 | 0.10031962 |
| balls | A_released_oce | 16 | -791803.28 | 0.95762269 | 0.97433829 | 0.24386919 |
| balls | B_rank_corrected_released_oce | 16 | -791767.79 | 0.95762249 | 0.97433694 | 0.24385599 |
| balls | C_objective_faithful_oce | 16 | -793061.91 | 0.93939066 | 0.95769866 | 0.24356891 |

Leakage is `||(I-R*) P G||F^2 / r_t`. The cross-check `||(I-R*) P R P^T||F^2 / r_t` agrees within the configured numerical tolerance. Anchor drift means only **anchor feature drift at the edited layer**.

## Answers

### Q1 — Released QR rank inflation

**No.** Target QR included extra dependent directions in 0/48 group-layer matrices (maximum inflation 0); anchor QR did so in 0/48 (maximum inflation 0).

### Q2 — Effect of rank correction alone

The effect was **limited** by the pre-registered one-percentage-point leakage criterion. Across matched layers, median `A leakage - B leakage` was `-6.3578288e-07` and median `A objective - B objective` was `0`; positive values favor Variant B.

### Q3 — Effect of objective-faithful orientation

The leakage change was **material** by the same criterion. Holding the SVD bases fixed, median `B leakage - C leakage` was `0.015501539`, and median `B objective - C objective` was `1195.1875`; positive values favor objective-faithful Variant C. Variant C was also checked layer-by-layer not to have a worse Eq. 18 objective than Variant B beyond floating-point tolerance.

### Q4 — Residual leakage after both corrections

Residual objective-faithful leakage was **substantial across multiple layers**: median `0.94462446`, maximum `0.98556709`, with 48/48 group-layer matrices at or above 0.01.

## Reproducibility

- CSV: `results.csv` (144 rows = 3 groups x 16 layers x 3 variants)
- Config SHA-256: `416ad7fd9e7666f8cd295ef6de4c6cf6af26d67502fd21d4027b6a81aa7e762b`
- Anchors SHA-256: `392802a728f5f726870718c2f9d0885ed57c8e16232899710f22690cee6c13b1`
- Qualification SHA-256: `b96514c7bd94e4d703079ddff238731c05ba76cde7fe6b0c2643d404bf7f043f`
- K0 SHA-256: `5d8ee50935eb3d22e1f9bc84947572afba386d05e063f70ea161c5cbf1e16235`
- Variant A checkpoint agreement: all layers passed `atol=2e-05` and `rtol=2e-05`; maximum absolute edited-weight error `0`
- Leakage formula QA: maximum normalized absolute difference `0.00034268697`; minimum tolerance margin `0.00022236506`
- Numerical-rank relative tolerance range: `3.8146973e-05` to `0.00015258789`
- Paper: https://arxiv.org/abs/2605.28902
- Runtime estimate for one cached SD 1.4 GPU run: about 5–15 minutes; no image generation or image evaluator is involved.
