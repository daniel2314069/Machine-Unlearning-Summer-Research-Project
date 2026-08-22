# ScaPre Informax Specificity: Edit-Seed Robustness

## Technical summary

This run varies only the Informax pseudo-sample random stream across the fixed seeds `20260820, 20260821, 20260822, 20260823, 20260824`. The legacy `20260820` image-level scores are reused after integrity validation; no legacy images are regenerated. Mean ΔPreserve is `+1.02` points with `5/5` positive seeds (`4/4` among the newly generated seeds). Mean ΔUnlearn is `-2.12` points with `5/5` seeds maintaining or improving target erasure, and mean ΔOverall is `+1.38` with `5/5` positive seeds. The final judgment is **ROBUSTLY SUPPORTED**.

## Seed-level image results

Delta is always `matched_retain - official`; therefore negative ΔU and positive ΔP/ΔOverall are improvements.

| Edit Seed | Official U ↓ | Matched U ↓ | ΔU ↓ | Official P ↑ | Matched P ↑ | ΔP ↑ | ΔOverall ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20260820 | 19.75 | 17.42 | -2.33 | 40.61 | 41.72 | +1.11 | +1.51 |
| 20260821 | 19.00 | 17.25 | -1.75 | 40.44 | 40.83 | +0.39 | +0.73 |
| 20260822 | 19.75 | 17.25 | -2.50 | 40.22 | 41.83 | +1.61 | +1.99 |
| 20260823 | 18.83 | 17.08 | -1.75 | 40.78 | 41.33 | +0.56 | +0.88 |
| 20260824 | 19.00 | 16.75 | -2.25 | 40.22 | 41.67 | +1.44 | +1.78 |

## Effect magnitude and seed variance

Standard deviation is the sample standard deviation with an `n-1` denominator. No chart is used because the experiment has exactly five fixed observations and the exact values, signs, and extrema are more auditable in a table.

| Metric | Mean | Std | Median | Min | Max | Improving seeds |
| --- | --- | --- | --- | --- | --- | --- |
| delta_unlearn | -2.12 | 0.35 | -2.25 | -2.50 | -1.75 | 5/5 |
| delta_preserve | 1.02 | 0.54 | 1.11 | 0.39 | 1.61 | 5/5 |
| delta_overall | 1.38 | 0.55 | 1.51 | 0.73 | 1.99 | 5/5 |

## Group-level robustness

Positive mean Preserve deltas occur in `4/5` groups. This table shows whether the effect is distributed across semantic groups rather than being driven by only the original seed or one group.

| Group | Mean ΔU | Mean ΔP | Mean ΔOverall | ΔP > 0 | ΔOverall > 0 |
| --- | --- | --- | --- | --- | --- |
| dogs | -3.58 | +1.06 | +1.77 | 3/5 | 5/5 |
| cats | +2.17 | +2.17 | +2.45 | 5/5 | 5/5 |
| fruits | -1.75 | +1.72 | +2.48 | 4/5 | 5/5 |
| boats | -1.08 | -0.44 | +0.44 | 1/5 | 3/5 |
| balls | -6.33 | +0.61 | +1.51 | 4/5 | 4/5 |

## Retain-concept robustness

All `15` retain concepts are shown without selection. `9/15` have positive mean deltas and `9/15` improve in at least half of the evaluated seeds. The complete 25-concept × seed table, including all targets, is in `results/per_concept_seed.csv`.

| Group | Retain concept | Off 20260820 | Match 20260820 | Off 20260821 | Match 20260821 | Off 20260822 | Match 20260822 | Off 20260823 | Match 20260823 | Off 20260824 | Match 20260824 | Mean Δ | Improve seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dogs | german shepherd | 45.83 | 49.17 | 47.50 | 49.17 | 43.33 | 50.83 | 46.67 | 45.83 | 46.67 | 51.67 | +3.33 | 4/5 |
| dogs | chesapeake bay retriever | 10.00 | 13.33 | 6.67 | 6.67 | 10.00 | 11.67 | 11.67 | 12.50 | 6.67 | 10.00 | +1.83 | 4/5 |
| dogs | pug | 82.50 | 80.00 | 82.50 | 80.00 | 84.17 | 79.17 | 82.50 | 82.50 | 80.83 | 80.83 | -2.00 | 0/5 |
| cats | persian cat | 19.17 | 20.00 | 19.17 | 20.00 | 20.00 | 20.00 | 18.33 | 21.67 | 20.83 | 20.83 | +1.00 | 3/5 |
| cats | siamese cat | 38.33 | 44.17 | 35.83 | 37.50 | 35.00 | 43.33 | 36.67 | 43.33 | 36.67 | 42.50 | +5.67 | 5/5 |
| cats | egyptian cat | 0.83 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.17 | 0/5 |
| fruits | pomegranate | 17.50 | 20.00 | 18.33 | 20.00 | 19.17 | 21.67 | 16.67 | 21.67 | 16.67 | 18.33 | +2.67 | 5/5 |
| fruits | fig | 15.00 | 13.33 | 15.00 | 12.50 | 15.00 | 15.83 | 15.00 | 12.50 | 15.00 | 15.00 | -1.17 | 1/5 |
| fruits | granny smith | 23.33 | 22.50 | 20.00 | 23.33 | 20.83 | 25.00 | 18.33 | 25.00 | 20.83 | 25.83 | +3.67 | 4/5 |
| boats | speedboat | 98.33 | 99.17 | 98.33 | 97.50 | 98.33 | 98.33 | 98.33 | 96.67 | 98.33 | 97.50 | -0.50 | 1/5 |
| boats | catamaran | 88.33 | 90.00 | 89.17 | 92.50 | 88.33 | 90.00 | 92.50 | 91.67 | 90.83 | 91.67 | +1.33 | 4/5 |
| boats | schooner | 89.17 | 90.83 | 89.17 | 86.67 | 90.83 | 88.33 | 90.83 | 86.67 | 90.00 | 86.67 | -2.17 | 1/5 |
| balls | tennis ball | 19.17 | 19.17 | 17.50 | 20.83 | 16.67 | 20.83 | 19.17 | 17.50 | 20.00 | 21.67 | +1.50 | 3/5 |
| balls | rugby ball | 27.50 | 32.50 | 31.67 | 35.83 | 27.50 | 33.33 | 31.67 | 32.50 | 25.83 | 33.33 | +4.67 | 5/5 |
| balls | ping-pong ball | 34.17 | 31.67 | 35.83 | 30.00 | 34.17 | 29.17 | 33.33 | 30.00 | 34.17 | 29.17 | -4.33 | 0/5 |

## Experimental design and integrity

- Base method source: unchanged `scapre/edit/erase_scale.py` with SHA-256 `537fce48480ca8fa2233e28a3526c81ab6f9d848a055b317dea1eebb89724d33`.
- Fixed non-Informax/global edit seed: `20260820`.
- New seeds use the audited RNG wrapper: every legacy global Informax draw is still consumed to preserve all later non-Informax RNG positions, while only the tensor returned to Informax comes from the seed-specific stream.
- Generation prompts and generation seeds are identical across both variants and every edit seed.
- Formal denominator per variant/seed: 25 concepts × 120 images = 3,000 rows, comprising 1,200 target and 1,800 retain rows.
- Evaluator/classifier fingerprints are identical across all comparisons.
- Run commit: `9ca7b5e9c4ab626027fb8fe0bd32fca51e8faf89`; start and end working-tree status are both recorded in `run_manifest.json`.
- The protocol remains the project-established Confuse5 reconstruction, not an exact author-released Table 7 seed asset.

## Limitations and decision boundary

This is a deterministic five-seed robustness check over one fixed generation protocol, not an inferential population estimate or a hyperparameter sweep. Seed `20260820` comes from the legacy globally seeded execution; its controlled sources and raw outputs are verified, while the four new runs isolate Informax draws and preserve the legacy non-Informax RNG stream. Safety-checker substitutions remain part of the unchanged evaluator and can affect absolute classifier accuracy. No threshold, seed, group, concept, or method component is changed in response to observed results, and no new Informax formulation is proposed or implemented in this run.

## Informax mechanism diagnostic

These quantities are explanatory only and do not determine the judgment.

| Edit Seed | MI official | MI matched | Alpha Spearman | Top 1% overlap | Top 5% overlap | Top 10% overlap |
| --- | --- | --- | --- | --- | --- | --- |
| 20260820 | 0.68 | 0.38 | 0.06 | 0.40 | 0.10 | 0.17 |
| 20260821 | 0.68 | 0.38 | 0.06 | 0.40 | 0.11 | 0.18 |
| 20260822 | 0.68 | 0.38 | 0.06 | 0.40 | 0.11 | 0.18 |
| 20260823 | 0.68 | 0.38 | 0.06 | 0.40 | 0.11 | 0.18 |
| 20260824 | 0.68 | 0.38 | 0.07 | 0.40 | 0.11 | 0.18 |

## Final judgment

**ROBUSTLY SUPPORTED**

matched-retain improvement is reproducible across Informax pseudo-sample randomness.
