# ScaPre projection accumulation — Confuse5

Delta is projection_accumulation minus official. Lower Unlearn and higher Preserve/Overall are favorable.

| Variant | Unlearn ↓ | Preserve ↑ | Overall ↑ |
| --- | ---: | ---: | ---: |
| official five-seed mean | 19.2667 | 40.4556 | 53.9007 |
| projection_accumulation five-seed mean | 19.7667 | 45.8444 | 58.3487 |
| treatment - official | +0.5000 | +5.3889 | +4.4480 |

Favorable seeds: Unlearn 1/5; Preserve 5/5; Overall 5/5.

Automatic directional conditions: FAIL
The per-group/per-target concentration condition is intentionally left for manual review; no numerical cutoff was invented.

## Group mean deltas

| Group | ΔUnlearn | ΔPreserve | ΔOverall |
| --- | ---: | ---: | ---: |
| dogs | +1.6667 | +12.4444 | +8.9450 |
| cats | +20.1667 | +7.7222 | +5.3820 |
| fruits | -0.7500 | +7.1667 | +9.4746 |
| boats | -2.0833 | -0.5000 | +0.9823 |
| balls | -16.5000 | +0.1111 | +2.0765 |

## All concept mean deltas

| Group | Role | Concept | Mean Δaccuracy | Favorable seeds |
| --- | --- | --- | ---: | ---: |
| dogs | target | golden retriever | +1.3333 | 0/5 |
| dogs | target | labrador retriever | +2.0000 | 0/5 |
| dogs | retain | german shepherd | +27.0000 | 5/5 |
| dogs | retain | chesapeake bay retriever | -1.5000 | 1/5 |
| dogs | retain | pug | +11.8333 | 5/5 |
| cats | target | tabby | +0.1667 | 2/5 |
| cats | target | tiger cat | +40.1667 | 0/5 |
| cats | retain | persian cat | +1.8333 | 4/5 |
| cats | retain | siamese cat | +21.0000 | 5/5 |
| cats | retain | egyptian cat | +0.3333 | 3/5 |
| fruits | target | orange | -6.5000 | 5/5 |
| fruits | target | lemon | +5.0000 | 0/5 |
| fruits | retain | pomegranate | +11.6667 | 5/5 |
| fruits | retain | fig | +6.3333 | 5/5 |
| fruits | retain | granny smith | +3.5000 | 4/5 |
| boats | target | yawl | -6.3333 | 5/5 |
| boats | target | lifeboat | +2.1667 | 0/5 |
| boats | retain | speedboat | +0.0000 | 0/5 |
| boats | retain | catamaran | -3.6667 | 0/5 |
| boats | retain | schooner | +2.1667 | 4/5 |
| balls | target | soccer ball | -24.3333 | 5/5 |
| balls | target | volleyball | -8.6667 | 5/5 |
| balls | retain | tennis ball | +1.8333 | 4/5 |
| balls | retain | rugby ball | +7.6667 | 5/5 |
| balls | retain | ping-pong ball | -9.1667 | 0/5 |
