# Exact orthogonal target-subspace mapping control

This directory contains the final matrix-only control following the Confuse5
OCE solver audit. It asks how much preservation cost is required by the
preservation-optimal member of the orthogonal family that maps the target
subspace exactly onto the frozen pre-edit anchor subspace.

This is an oracle-like feasibility control. It is not AFR, a model checkpoint,
a repaired OCE variant, or a proposed method. It never generates images and
does not call ResNet, CLIP, or FID.

## Frozen scope

The runner resolves the same qualified Joint cases from the existing protocol:

| Group | Targets | Matched anchors |
|---|---|---|
| dogs | golden retriever; labrador retriever | cocker spaniel; beagle |
| fruits | orange; lemon | banana; pineapple |
| balls | soccer ball; volleyball | basketball; baseball |

It reuses the exact prompt expansion, last-content-token extraction, K0, local
retain concepts, repository regularizer, scales, and 16 `attn2.to_v` layers.
After loading the frozen production tensors, all projections, bases, SVDs,
Procrustes solves, and metrics are computed in float64.

## Mathematics

For equal-rank orthonormal bases `G` and `H`, every exact feasible transform
has the form

```text
P = H Q G^T + H_perp Q_perp G_perp^T,
Q in O(r), Q_perp in O(d-r).
```

Substitution into `max tr(P^T S)` gives two independent standard Procrustes
problems:

```text
M1 = H^T S G,                 Q*      = U1 V1^T
M2 = H_perp^T S G_perp,       Q_perp* = U2 V2^T.
```

The runner directly checks that the achieved trace equals
`||M1||_* + ||M2||_*` and that `P G = H Q*`. Primary control is `O(d)`; no
released determinant correction or SO(d) constraint is applied.

## Fail-closed controls

Before writing results, the runner requires:

- all synthetic feasibility, random-feasible optimality, and nuclear-norm
  value tests to pass;
- exactly the frozen dogs/fruits/balls Joint scope and 16 layers;
- target rank = anchor rank = 12 in every case;
- float64 Variant C leakage within `0.01` absolute and raw preservation within
  `1%` relative of the prior float32 audit row;
- Variant D true leakage and exact mapping residual at or below `1e-10`;
- both Variant C and D `||P^T P-I||F` at or below `1e-10`, with the float64
  leakage formula QA also passing.

If any check fails, the run stops before writing either requested output.

## Interpretation thresholds

Because the requested D1/D2/D3 language is qualitative, thresholds are frozen
in the script before observing this control's results:

- a material normalized preservation penalty requires both D-C >= `0.01` and
  D/C >= `1.25`;
- a material anchor-feature penalty requires both D-C >= `0.05` and D/C >=
  `1.25`;
- D2 requires one of those penalties in at least 36/48 layers and a material
  median in all three groups;
- D1 requires no more than 12/48 material layers for either metric, overall
  median preservation D/C <= `1.10`, and overall median anchor D/C <= `1.10`
  with small absolute medians;
- intermediate or group/layer-dependent evidence is D3.

Raw values and complete layer distributions remain in the outputs, so the
classification can be audited independently of these labels.

## Commands

On the GPU server:

```bash
conda activate MU
cd orthogonal-concept-erasure

python experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/run.py --synthetic-only
python experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/run.py
```

The full run writes only:

```text
experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/results_exact_control.csv
experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/REPORT_exact_control.md
```

Expected runtime with cached SD 1.4, K0, qualification, and primary artifacts
is approximately 15–45 minutes on one GPU. Float64 SVD of the large complement
block dominates runtime; there is no diffusion inference.

The anchor-minimum caveat-closing control lives in
[`anchor_min_control/`](anchor_min_control/). It uses the same exact feasible
family but directly minimizes anchor feature drift, while evaluating frozen-S
preservation separately.
