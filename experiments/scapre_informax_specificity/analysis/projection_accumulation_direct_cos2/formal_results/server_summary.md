# ScaPre projection accumulation — Confuse5

Delta is projection_accumulation_direct_cos2 minus official. Lower Unlearn and higher Preserve/Overall are favorable.

| Variant | Unlearn ↓ | Preserve ↑ | Overall ↑ |
| --- | ---: | ---: | ---: |
| official five-seed mean | 19.2667 | 40.4556 | 53.9007 |
| projection_accumulation_direct_cos2 five-seed mean | 19.1000 | 40.5111 | 53.9864 |
| treatment - official | -0.1667 | +0.0556 | +0.0857 |

Favorable seeds: Unlearn 3/5; Preserve 3/5; Overall 2/5.

Automatic directional conditions: FAIL
The per-group/per-target concentration condition is intentionally left for manual review; no numerical cutoff was invented.

## Group mean deltas

| Group | ΔUnlearn | ΔPreserve | ΔOverall |
| --- | ---: | ---: | ---: |
| dogs | +0.2500 | -0.0000 | -0.0632 |
| cats | +0.2500 | -0.6111 | -0.7932 |
| fruits | -0.5833 | -0.0556 | -0.0420 |
| boats | -0.9167 | +0.3889 | +0.6958 |
| balls | +0.1667 | +0.5556 | +0.5925 |

## All concept mean deltas

| Group | Role | Concept | Mean Δaccuracy | Favorable seeds |
| --- | --- | --- | ---: | ---: |
| dogs | target | golden retriever | +0.3333 | 2/5 |
| dogs | target | labrador retriever | +0.1667 | 1/5 |
| dogs | retain | german shepherd | +0.5000 | 3/5 |
| dogs | retain | chesapeake bay retriever | -0.5000 | 1/5 |
| dogs | retain | pug | +0.0000 | 2/5 |
| cats | target | tabby | +0.3333 | 1/5 |
| cats | target | tiger cat | +0.1667 | 1/5 |
| cats | retain | persian cat | -1.3333 | 0/5 |
| cats | retain | siamese cat | -0.5000 | 1/5 |
| cats | retain | egyptian cat | +0.0000 | 1/5 |
| fruits | target | orange | -1.0000 | 3/5 |
| fruits | target | lemon | -0.1667 | 3/5 |
| fruits | retain | pomegranate | -0.8333 | 0/5 |
| fruits | retain | fig | +1.1667 | 3/5 |
| fruits | retain | granny smith | -0.5000 | 2/5 |
| boats | target | yawl | -1.0000 | 3/5 |
| boats | target | lifeboat | -0.8333 | 3/5 |
| boats | retain | speedboat | -0.1667 | 0/5 |
| boats | retain | catamaran | +1.1667 | 4/5 |
| boats | retain | schooner | +0.1667 | 3/5 |
| balls | target | soccer ball | -0.1667 | 2/5 |
| balls | target | volleyball | +0.5000 | 2/5 |
| balls | retain | tennis ball | +1.3333 | 4/5 |
| balls | retain | rugby ball | -0.0000 | 4/5 |
| balls | retain | ping-pong ball | +0.3333 | 3/5 |
