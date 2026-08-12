# Anchor-minimum exact orthogonal control

This directory contains the matrix-only follow-up that closes the remaining
caveat in the S-optimal exact orthogonal control. It asks for the minimum
possible **anchor feature drift at the edited layer** over the entire
orthogonal family that maps the target subspace exactly onto the matched
anchor subspace.

It is an oracle-like control, not AFR, an edited-model checkpoint, a repaired
OCE variant, or a proposed method. It does not generate images or call ResNet,
CLIP, or FID, and it does not modify production `oce.py`.

## Frozen inputs and reused implementation

The runner imports the preceding
[`../run.py`](../run.py) and directly reuses its float64 rank basis,
orthogonal completion, constrained Procrustes, frozen-S construction, metrics,
and protocol/checkpoint loaders. It imports the checked-in
[`../results_exact_control.csv`](../results_exact_control.csv) as the frozen
float64 C/D baseline instead of recomputing those controls.

Fail-closed SHA-256 guards fix the same config, registry, qualification, K0,
and C/D CSV. The qualified scope remains:

| Group | Targets | Matched anchors |
|---|---|---|
| dogs | golden retriever; labrador retriever | cocker spaniel; beagle |
| fruits | orange; lemon | banana; pineapple |
| balls | soccer ball; volleyball | basketball; baseball |

Prompt expansion, local retain, K0, regularizer, scales, and the same 16
edited layers are unchanged. Production prompt encoding is loaded in float32;
every audit tensor is explicitly cast to float64 before matrix operations.

## Variant E

Let `Y=WC*` be the expanded anchor feature matrix and `S_a=YY^T`. For the
exact feasible family

```text
P = H Q G^T + H_perp Q_perp G_perp^T,
```

minimizing `||PY-Y||F^2` is equivalent to maximizing `tr(P^T S_a)`. The same
closed-form solver from the preceding control is called with `S_x=S_a`:

```text
M1 = H^T S_a G,                 Q*      = U1 V1^T
M2 = H_perp^T S_a G_perp,       Q_perp* = U2 V2^T.
```

Because `Y` lies in `span(H)`, the second block can be zero or rank-deficient.
Its arbitrary O(d-r) SVD completion is allowed; the runner checks
orthogonality, exact mapping, anchor-loss trace identity, and attainment of
the nuclear-norm optimum instead of requiring a unique completion.

## Synthetic and real-data QA

Before the real matrices, the runner checks:

1. exact feasibility and numerical-zero leakage;
2. anchor loss no worse than 256 random feasible transforms;
3. equality with the closed-form nuclear-norm optimum;
4. zero normalized anchor drift in the known `G=H` case.

The full run then requires rank 12/12, exact leakage and mapping residual at or
below `1e-10`, and float64 orthogonality residual at or below `1e-10` in all
48 cases. Frozen-S preservation is evaluated with exactly the preceding
control's covariance. It also checks the two endpoint inequalities in every
layer: E cannot have higher anchor drift than feasible D, and frozen-S-optimal
D cannot have higher frozen-S loss than feasible E.

The E1/E2/E3 decision reuses the preceding material thresholds:

- anchor increase: E-C >= `0.05` and E/C >= `1.25`;
- frozen-S increase: E-C >= `0.01` and E/C >= `1.25`;
- a consistent effect requires at least 36/48 layers and a material median in
  all three groups.

Only an E3 or ambiguous primary result triggers the fixed closed-form Pareto
diagnostic for `lambda = {0, 0.1, 0.3, 1, 3, 10}`. E1 and E2 do not run it.

## Commands

On the GPU server:

```bash
conda activate MU
cd orthogonal-concept-erasure

python experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/anchor_min_control/run.py --synthetic-only
python experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/anchor_min_control/run.py
```

The full run always writes:

```text
experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/anchor_min_control/results_anchor_min_control.csv
experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/anchor_min_control/REPORT_anchor_min_control.md
```

It writes `results_anchor_pareto.csv` only when the primary result is E3 or
ambiguous. No other experiment tree or artifact is created.
