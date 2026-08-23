# ScaPre Informax superclass-neutral

## Technical summary

This five-edit-seed experiment changes only the Informax negative base from the empty prompt to the target superclass. Verified official score rows are reused; no 3,000-row official baseline evaluation is rerun. Mean ΔPreserve is `+0.51` pp, mean ΔUnlearn is `-0.03` pp, and mean ΔOverall is `+0.46` pp. The frozen decision gives **NOT SUPPORTED**.

## Official versus superclass-neutral

Delta is `superclass_neutral - official`; negative ΔUnlearn and positive ΔPreserve/ΔOverall are improvements.

| Edit Seed | Official U ↓ | Superclass U ↓ | ΔU ↓ | Official P ↑ | Superclass P ↑ | ΔP ↑ | ΔOverall ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20260820 | 19.75 | 19.58 | -0.17 | 40.61 | 41.28 | +0.67 | +0.62 |
| 20260821 | 19.00 | 19.33 | +0.33 | 40.44 | 41.00 | +0.56 | +0.42 |
| 20260822 | 19.75 | 19.67 | -0.08 | 40.22 | 41.44 | +1.22 | +1.09 |
| 20260823 | 18.83 | 18.58 | -0.25 | 40.78 | 40.44 | -0.33 | -0.24 |
| 20260824 | 19.00 | 19.00 | +0.00 | 40.22 | 40.67 | +0.44 | +0.40 |

## Across-seed stability

| Metric | Mean | Std | Median | Min | Max | Improving seeds |
| --- | --- | --- | --- | --- | --- | --- |
| delta_unlearn | -0.03 | 0.23 | -0.08 | -0.25 | 0.33 | 4/5 |
| delta_preserve | 0.51 | 0.56 | 0.56 | -0.33 | 1.22 | 4/5 |
| delta_overall | 0.46 | 0.48 | 0.42 | -0.24 | 1.09 | 4/5 |

## Group distribution

`5/5` groups have positive mean preservation delta.

| Group | Mean ΔU | Mean ΔP | Mean ΔOverall | ΔP > 0 | ΔOverall > 0 |
| --- | --- | --- | --- | --- | --- |
| dogs | +0.83 | +0.17 | -0.07 | 3/5 | 3/5 |
| cats | +0.58 | +0.61 | +0.71 | 4/5 | 4/5 |
| fruits | -0.75 | +0.83 | +1.20 | 4/5 | 4/5 |
| boats | -0.67 | +0.67 | +0.66 | 3/5 | 3/5 |
| balls | -0.17 | +0.28 | +0.34 | 3/5 | 3/5 |

## Target and retain distribution

`5/10` targets have lower mean residual accuracy and `10/15` retains have positive mean preservation delta; `10/15` retains improve in at least three seeds. Complete unselected results are in `per_concept_seed.csv`, `per_target_robustness.csv`, and `per_retain_robustness.csv`.

| Group | Target | Mean Δ accuracy | Erasure-improving seeds |
| --- | --- | --- | --- |
| dogs | golden retriever | +0.83 | 0/5 |
| dogs | labrador retriever | +0.83 | 1/5 |
| cats | tabby | +1.00 | 1/5 |
| cats | tiger cat | +0.17 | 3/5 |
| fruits | orange | -1.17 | 4/5 |
| fruits | lemon | -0.33 | 2/5 |
| boats | yawl | -0.17 | 2/5 |
| boats | lifeboat | -1.17 | 4/5 |
| balls | soccer ball | -0.83 | 2/5 |
| balls | volleyball | +0.50 | 2/5 |

| Group | Retain | Mean Δ accuracy | Preserve-improving seeds |
| --- | --- | --- | --- |
| dogs | german shepherd | +0.00 | 3/5 |
| dogs | chesapeake bay retriever | +0.17 | 3/5 |
| dogs | pug | +0.33 | 3/5 |
| cats | persian cat | +0.33 | 2/5 |
| cats | siamese cat | +1.50 | 4/5 |
| cats | egyptian cat | +0.00 | 1/5 |
| fruits | pomegranate | +1.50 | 3/5 |
| fruits | fig | -0.17 | 1/5 |
| fruits | granny smith | +1.17 | 4/5 |
| boats | speedboat | -0.17 | 0/5 |
| boats | catamaran | +0.00 | 2/5 |
| boats | schooner | +2.17 | 4/5 |
| balls | tennis ball | +1.33 | 4/5 |
| balls | rugby ball | -0.83 | 3/5 |
| balls | ping-pong ball | +0.33 | 3/5 |

## Context against matched-retain

This table is descriptive only; matched-retain scores are reused from the completed robustness experiment.

| Metric | Direction | Official mean | Matched mean | Superclass mean |
| --- | --- | --- | --- | --- |
| unlearn | ↓ | 19.27 | 17.15 | 19.23 |
| preserve | ↑ | 40.46 | 41.48 | 40.97 |
| overall | ↑ | 53.90 | 55.28 | 54.36 |

## Informax diagnostic

`informax_seed_diagnostics.csv` reports raw official and superclass MI summaries. These mechanism values do not determine success.

## Reproducibility and qualitative set

- Edit seeds: `20260820, 20260821, 20260822, 20260823, 20260824`; global non-Informax seed remains `20260820`.
- Five positive and five negative pseudo-samples are retained; superclass mode uses five noisy copies of exactly one mapped superclass embedding.
- All 15,000 new superclass score rows have the same 25 concepts × 120 prompts/seeds as official and matched-retain.
- Evaluator/classifier fingerprints are substantively identical across all 15 seed/variant observations.
- The predeclared qualitative set contains 90 images plus 30 side-by-side panels: both targets and one retain per group, sample indices 0 and 1, identical prompt and generation seed across all three variants.
- Official/matched qualitative pictures are the only baseline images regenerated. Their 60 predictions are rechecked against the recorded formal rows; no full baseline evaluation or metric recomputation is performed.
- Run commit: `4f45257fba7160ea6b8ef3ba9bb4115409b35a8a`.

## Final answer

**NOT SUPPORTED**

Superclass-neutral does not show a stable, non-trade-off image-level advantage over official.
