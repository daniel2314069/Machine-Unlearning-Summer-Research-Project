# Squared-cosine follow-up experiments

These experiments were added after the main ScaPre Informax analysis. They test whether a simple geometric channel score can replace the highly saturated binary MI in the earlier per-concept accumulation.

For target concept `k`, with official empty-string vector `c0`, define `d_k = c_k - c0`. For output row `w_i`, the score is

`r_i = (w_i^T d_k)^2 / ((||w_i||^2 + eps)(||d_k||^2 + eps))`,

which is approximately `cos^2(theta_i)`.

## Five-seed results

Deltas are treatment minus official. Lower Unlearn and higher Preserve/Overall are favorable.

| Variant | Delta Unlearn | Delta Preserve | Delta Overall | Favorable U/P/O |
| --- | ---: | ---: | ---: | --- |
| Transformed cos^2 | +0.5000 | +5.3889 | +4.4480 | 1/5, 5/5, 5/5 |
| Direct cos^2 | -0.1667 | +0.0556 | +0.0857 | 3/5, 3/5, 2/5 |
| Direct cos^2 with matched norm | +0.1000 | +0.2667 | +0.2139 | 1/5, 3/5, 3/5 |

### Transformed cos^2
The score was passed through the repository-style channel z-score and sigmoid/power transform. Preservation and Overall improved strongly, but target erasure became uneven. Tiger cat was the largest failure: mean Unlearn Accuracy worsened by about +40.17 percentage points and was unfavorable in all five edit seeds. Several other targets also worsened, so this was not an overall improvement.

Repository artifact path:
`experiments/scapre_informax_specificity/analysis/projection_accumulation/formal_results/`

### Direct cos^2
The raw score was used directly as the row weight. The extreme tiger-cat failure disappeared, but almost all of the preservation/Overall gain also disappeared. The result stayed close to official and Overall was favorable in only 2/5 seeds.

Repository artifact path:
`experiments/scapre_informax_specificity/analysis/projection_accumulation_direct_cos2/formal_results/`

### Direct cos^2 with matched norm
The raw score was multiplied by one scalar for each concept and matrix so that the Frobenius norm of the weighted accumulation contribution matched the official contribution. This removed the change in per-concept/per-matrix contribution norm, but the result still did not improve stably. Mean Unlearn Accuracy worsened by +0.10 points, and Overall was favorable in 3/5 seeds.

Repository artifact path:
`experiments/scapre_informax_specificity/analysis/projection_accumulation_budget_matched_cos2/formal_results/`

## Conclusion
The three experiments do not support simple squared-cosine alignment as a reliable replacement for ScaPre's per-concept Informax weighting. The transformed version shows that this geometric score can strongly change the erase-preserve trade-off, but the simpler and norm-controlled versions remain close to the official method. No COCO evaluation was run for these variants because none passed the continuation conditions.
