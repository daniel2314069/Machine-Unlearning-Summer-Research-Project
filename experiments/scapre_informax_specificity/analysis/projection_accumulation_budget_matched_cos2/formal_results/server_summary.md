# ScaPre projection accumulation — Confuse5

Delta is projection_accumulation_budget_matched_cos2 minus official. Lower Unlearn and higher Preserve/Overall are favorable.

| Variant | Unlearn ↓ | Preserve ↑ | Overall ↑ |
| --- | ---: | ---: | ---: |
| official five-seed mean | 19.2667 | 40.4556 | 53.9007 |
| projection_accumulation_budget_matched_cos2 five-seed mean | 19.3667 | 40.7222 | 54.1146 |
| treatment - official | +0.1000 | +0.2667 | +0.2139 |

Favorable seeds: Unlearn 1/5; Preserve 3/5; Overall 3/5.

Automatic directional conditions: FAIL
The per-group/per-target concentration condition is intentionally left for manual review; no numerical cutoff was invented.

## Group mean deltas

| Group | ΔUnlearn | ΔPreserve | ΔOverall |
| --- | ---: | ---: | ---: |
| dogs | +0.6667 | +0.4444 | +0.2005 |
| cats | +0.1667 | +0.5000 | +0.6150 |
| fruits | -0.0833 | +0.2778 | +0.3985 |
| boats | -0.7500 | -0.1667 | +0.3569 |
| balls | +0.5000 | +0.2778 | +0.2462 |

## All concept mean deltas

| Group | Role | Concept | Mean Δaccuracy | Favorable seeds |
| --- | --- | --- | ---: | ---: |
| dogs | target | golden retriever | +0.6667 | 2/5 |
| dogs | target | labrador retriever | +0.6667 | 1/5 |
| dogs | retain | german shepherd | +0.0000 | 2/5 |
| dogs | retain | chesapeake bay retriever | +1.5000 | 4/5 |
| dogs | retain | pug | -0.1667 | 2/5 |
| cats | target | tabby | +0.1667 | 2/5 |
| cats | target | tiger cat | +0.1667 | 1/5 |
| cats | retain | persian cat | -0.1667 | 1/5 |
| cats | retain | siamese cat | +1.6667 | 4/5 |
| cats | retain | egyptian cat | +0.0000 | 1/5 |
| fruits | target | orange | -0.5000 | 3/5 |
| fruits | target | lemon | +0.3333 | 1/5 |
| fruits | retain | pomegranate | +0.0000 | 2/5 |
| fruits | retain | fig | +1.6667 | 4/5 |
| fruits | retain | granny smith | -0.8333 | 2/5 |
| boats | target | yawl | -0.6667 | 3/5 |
| boats | target | lifeboat | -0.8333 | 4/5 |
| boats | retain | speedboat | -0.1667 | 0/5 |
| boats | retain | catamaran | +0.0000 | 2/5 |
| boats | retain | schooner | -0.3333 | 2/5 |
| balls | target | soccer ball | +0.0000 | 2/5 |
| balls | target | volleyball | +1.0000 | 0/5 |
| balls | retain | tennis ball | +0.1667 | 1/5 |
| balls | retain | rugby ball | +1.3333 | 4/5 |
| balls | retain | ping-pong ball | -0.6667 | 1/5 |
