# ScaPre Informax alpha-channel controls

## Result

All integrity gates passed. Delta in this report is treatment minus baseline;
lower Unlearn and higher Preserve/Overall are favorable. The controls support
only conclusions about the final concept-max channel assignment in this fixed
official-empty-neutral configuration.

The channel-control comparison is mixed rather than a strict Pareto result. The next question is whether the trade-off is concentrated in particular target families, using the saved per-target deltas. Identity_B Pareto-dominates official on the five-seed means, so uniform all-one weighting remains plausible in this setting. These labels use strict
Pareto direction only; no significance or similarity threshold was invented.

## Exact intervention and isolation

The experiment-only runner replaced only `row_w_max` after the official
per-concept MI -> z-score -> sigmoid/power -> concept-max pipeline and before
the row-wise closed-form term `erase_scale * alpha_i * PiC`. The production
editor was byte-unchanged. Raw MI, pre-aggregate alpha, thresholds, Informax
noise tensors, UCE inputs, prompts, generation seeds, sampler, 50 steps, CFG
7.5, classifier, and row counts matched across variants.

## Smoke

Seed `20260820` produced finite checkpoints and complete small evaluations for
official, constant mean, three preregistered/non-selected shuffle salts, and
identity. Smoke was not used for tuning.

## Five-seed image metrics

Cells are `Unlearn / Preserve / Overall` percentages.

| Seed | official | constant_mean | shuffled | identity_B |
| --- | --- | --- | --- | --- |
| 20260820 | 19.75/40.61/53.93 | 19.33/40.89/54.27 | 19.67/40.61/53.95 | 19.33/40.94/54.32 |
| 20260821 | 19.00/40.44/53.95 | 19.08/40.78/54.23 | 19.42/40.50/53.91 | 19.25/40.39/53.85 |
| 20260822 | 19.75/40.22/53.59 | 18.92/40.72/54.22 | 19.17/40.28/53.77 | 19.17/40.78/54.21 |
| 20260823 | 18.83/40.78/54.28 | 19.75/40.72/54.03 | 18.42/40.94/54.52 | 19.42/40.67/54.05 |
| 20260824 | 19.00/40.22/53.75 | 18.83/40.94/54.43 | 19.00/41.00/54.44 | 19.00/40.61/54.10 |

## Required comparisons

| Comparison | Mean ΔU | Mean ΔP | Mean ΔO | U favorable | P favorable | O favorable |
| --- | --- | --- | --- | --- | --- | --- |
| official_vs_constant_mean | -0.08 | +0.36 | +0.33 | 3/5 | 4/5 | 4/5 |
| official_vs_shuffled | -0.13 | +0.21 | +0.22 | 3/5 | 4/5 | 4/5 |
| official_vs_identity_B | -0.03 | +0.22 | +0.20 | 2/5 | 3/5 | 3/5 |
| constant_mean_vs_identity_B | +0.05 | -0.13 | -0.13 | 1/5 | 2/5 | 2/5 |

### Official vs constant_mean

The corresponding row above is the mean-matched channel-identity control.

### Official vs shuffled

The corresponding row above preserves the exact per-matrix alpha multiset and
destroys only channel identity.

### Official vs identity_B

The corresponding row above is the separate all-one paper-limit control.

### Constant_mean vs identity_B

This row separates uniform alpha scale from channel differentiation.

### Optional no-Informax reference

Omitted because setting only final `B=0` leaves per-concept Informax-weighted
UCE accumulation active, while globally disabling Informax changes more than
the final-alpha intervention and is not a clean established No-Informax match.

## Per-target patterns

Mean target accuracy delta is treatment minus baseline; negative is stronger
unlearning. `Favorable seeds` counts negative deltas.

| Comparison | Target | Mean Δ accuracy | Favorable seeds |
| --- | --- | --- | --- |
| official_vs_constant_mean | golden retriever | +0.33 | 1/5 |
| official_vs_constant_mean | labrador retriever | +0.50 | 1/5 |
| official_vs_constant_mean | tabby | +0.00 | 3/5 |
| official_vs_constant_mean | tiger cat | -0.67 | 4/5 |
| official_vs_constant_mean | orange | -1.67 | 4/5 |
| official_vs_constant_mean | lemon | +0.33 | 2/5 |
| official_vs_constant_mean | yawl | +1.17 | 2/5 |
| official_vs_constant_mean | lifeboat | -0.83 | 4/5 |
| official_vs_constant_mean | soccer ball | +0.50 | 1/5 |
| official_vs_constant_mean | volleyball | -0.50 | 3/5 |
| official_vs_shuffled | golden retriever | +0.17 | 1/5 |
| official_vs_shuffled | labrador retriever | +0.33 | 2/5 |
| official_vs_shuffled | tabby | -0.33 | 3/5 |
| official_vs_shuffled | tiger cat | -0.33 | 2/5 |
| official_vs_shuffled | orange | -0.67 | 3/5 |
| official_vs_shuffled | lemon | -0.67 | 2/5 |
| official_vs_shuffled | yawl | +0.00 | 2/5 |
| official_vs_shuffled | lifeboat | +0.00 | 2/5 |
| official_vs_shuffled | soccer ball | -0.67 | 3/5 |
| official_vs_shuffled | volleyball | +0.83 | 1/5 |
| official_vs_identity_B | golden retriever | +2.33 | 0/5 |
| official_vs_identity_B | labrador retriever | +0.00 | 1/5 |
| official_vs_identity_B | tabby | +0.50 | 2/5 |
| official_vs_identity_B | tiger cat | +0.83 | 0/5 |
| official_vs_identity_B | orange | -1.00 | 4/5 |
| official_vs_identity_B | lemon | -0.83 | 3/5 |
| official_vs_identity_B | yawl | -0.17 | 2/5 |
| official_vs_identity_B | lifeboat | -0.83 | 4/5 |
| official_vs_identity_B | soccer ball | -2.00 | 4/5 |
| official_vs_identity_B | volleyball | +0.83 | 1/5 |
| constant_mean_vs_identity_B | golden retriever | +2.00 | 0/5 |
| constant_mean_vs_identity_B | labrador retriever | -0.50 | 3/5 |
| constant_mean_vs_identity_B | tabby | +0.50 | 1/5 |
| constant_mean_vs_identity_B | tiger cat | +1.50 | 0/5 |
| constant_mean_vs_identity_B | orange | +0.67 | 0/5 |
| constant_mean_vs_identity_B | lemon | -1.17 | 4/5 |
| constant_mean_vs_identity_B | yawl | -1.33 | 4/5 |
| constant_mean_vs_identity_B | lifeboat | -0.00 | 1/5 |
| constant_mean_vs_identity_B | soccer ball | -2.50 | 4/5 |
| constant_mean_vs_identity_B | volleyball | +1.33 | 0/5 |

All seed-level and per-target deltas are in `comparison_deltas.csv`; full
per-concept accuracies are in `per_target_metrics.csv`.

## Interpretation and limitations

Interpret mixed Unlearn/Preserve effects as trade-offs. `constant_mean`
preserves each matrix's mean alpha but not every nonlinear notion of edit
strength. `identity_B` is a separate paper-limit control with different scale.
The findings cover SD v1.5, this reconstructed Confuse5 protocol, official
empty-string neutral, fixed evaluator, and five edit seeds only. They do not
test a replacement relevance estimator or matched-retain neutral.

The next research question is the one stated in the result paragraph above; it
is selected by strict metric direction, not by tuning a threshold on this run.

## Reproducibility

- Git commit: `a393c649f54b532fdcd85fc3fe7ff3e30a1a5f0d` on `main`
- Edit seeds: `20260820, 20260821, 20260822, 20260823, 20260824`
- Per variant/seed: 25 concepts x 120 = 3,000 images
- Official image scores: reused byte-for-byte after archive, protocol, asset,
  evaluator, row-key, and score-hash validation
- Protocol SHA-256: `ab8ced3be82b1d8c6896eb24d32419bb12e8cf3e22d8ed0c46f92f4a5e8f0f2e`
- Actual config SHA-256: `c55a50602c987514720a9cb3c6e59585233a4ac94805c6763ce3772f836f440f`
- Base config SHA-256: `3ce815fe4bdaca02370e77e7d7d27b8db226e99638824a10ede3b7528341cddc`
- Total score rows: `60000`
- Working tree: clean at launch and immediately before aggregation; the worker
  also aborts unless it remains clean after aggregation and records the final
  empty status in `run_manifest.json`
