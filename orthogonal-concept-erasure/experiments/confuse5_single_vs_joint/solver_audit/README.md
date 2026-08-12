# OCE Confuse5 matrix-level solver audit

This directory is reserved for the small, matrix-only audit requested for the
baseline-qualified Confuse5 Joint groups. The executable is
[`../solver_audit.py`](../solver_audit.py). A successful GPU-server run writes
only:

- `results.csv`: 144 rows (3 groups x 16 edited layers x 3 solvers);
- `REPORT.md`: convention audit, aggregate diagnostics, Q1-Q4, and the required
  Outcome A/B/C interpretation.

No checkpoint is created or modified. No diffusion sample, ResNet result, CLIP
result, or FID result is produced.

## Frozen scope

The script loads `config.json`, `anchors.json`, the completed qualification
summary, and the existing primary joint-checkpoint metadata. It refuses to run
unless the qualified groups are exactly:

| Group | Targets | Matched anchors |
|---|---|---|
| dogs | golden retriever; labrador retriever | cocker spaniel; beagle |
| fruits | orange; lemon | banana; pineapple |
| balls | soccer ball; volleyball | basketball; baseball |

In particular, the balls anchors are not replaced by a shared `ball` anchor.
Prompt expansion, last-content-token extraction, local retain concepts, K0,
all four configured scales, and the 16 `attn2.to_v` layers come from the
existing protocol and checkpoint construction path.

## Solver definitions

Let `A = I - R*` and let `S` contain the weighted local-retain, K0, and
repository regularizer matrices.

- A (`A_released_oce`): normalized-column reduced QR,
  `M = -lambda_e R A + S`, `P = U V^T`, then the released post-product last
  column flip when `det(P) < 0`.
- B (`B_rank_corrected_released_oce`): the same matrix orientation, solver, and
  determinant behavior, changing only QR to a rank-revealing SVD basis.
- C (`C_objective_faithful_oce`): the same SVD bases as B,
  `M = -lambda_e A R + S`, and the standard O(d) solution `P = U V^T`, with no
  determinant correction.

The SVD rank rule is

```text
sigma_i > max(number_of_rows, number_of_columns) * eps(float32) * sigma_max
```

with zero absolute tolerance. SVD is applied to the same L2-normalized
projected columns received by released QR, so the comparison isolates basis
construction rather than silently changing OCE input normalization.

All three variants are evaluated with the same rank-revealed target and anchor
projectors, the literal weighted Eq. 18 loss, true `P R P^T` leakage, and
edited-layer anchor feature drift. The two algebraically equivalent leakage
expressions are checked on every row after rank normalization. That QA uses
the reproducible float32 tolerance
`8 * layer_dimension * eps(float32) * max(1, |leakage values|)`; this accounts
for dense projector-product roundoff and remains below the report's `0.01`
interpretation threshold even at the 1,280-dimensional layers.

## Convention audit

Directly expanding the erasure term gives

```text
-lambda_e ||P R - A||F^2
  = constant + 2 lambda_e tr(P^T A R).
```

After adding preservation, minimizing Eq. 18 is therefore equivalent to
maximizing

```text
tr(P^T [-lambda_e A R + S]).
```

For `M = U Sigma V^T`, standard `max tr(P^T M)` is solved by `P = U V^T`.
This is the convention in paper Appendix A.2. The paper main text instead
writes `max tr(P M_total)` with `M_total = -R A + S`, then states
`P = U V^T`; for that non-transposed trace, the matching solution would be
`V U^T`. Released `oce.py` and the Confuse5 checkpoint builder pair the
main-text matrix orientation with `U V^T`, then add the released determinant
correction.

## Safety and fail-closed checks

The full audit refuses to run on macOS or without CUDA. It loads the cached SD
1.4 components with `local_files_only=True`, uses no VAE, and never invokes the
pipeline for image generation. Before writing either output, it requires every
Variant A edited weight to match the corresponding completed production joint
checkpoint within `atol=2e-5`, `rtol=2e-5`. Any mismatch stops the audit and
writes no CSV/report.

The embedded synthetic tests first verify:

1. `U V^T` attains the nuclear-norm optimum for `tr(P^T M)`;
2. the objective-faithful orientation is not worse on an explicit Eq. 18
   instance;
3. a 2D target-to-anchor rotation transforms the projector as `P R P^T`, not
   as `P R`.

## Commands

On the GPU server, after activating the project environment:

```bash
conda activate MU
cd orthogonal-concept-erasure

# Optional tiny test only (no model/artifact load).
python experiments/confuse5_single_vs_joint/solver_audit.py --synthetic-only

# Full matrix audit; requires completed primary K0, qualification, and joint checkpoints.
python experiments/confuse5_single_vs_joint/solver_audit.py
```

The runner deliberately uses the active environment's `python`; it does not
hard-code an interpreter or Conda environment. With cached SD 1.4 artifacts,
the expected runtime is roughly 5-15 minutes on one GPU. The dominant work is
144 small-to-medium layer-level SVD solves, not diffusion inference.

The interpretation rule is preregistered in the generated report: leakage at
or above `0.01` in the median objective-faithful layer, or in at least 25% of
objective-faithful group-layer matrices, counts as a multi-layer residual gap;
`1e-4` is the near-zero threshold. A one-percentage-point median leakage change
is the threshold used for the words “material” in Q2/Q3. The CSV always retains
the unrounded values so these labels can be audited.

The follow-up float64 oracle-like feasibility control lives in
[`exact_orthogonal_control/`](exact_orthogonal_control/). It asks for the
preservation-optimal orthogonal transform under an exact target-subspace to
anchor-subspace mapping constraint; it is not an AFR implementation or a new
OCE variant.
