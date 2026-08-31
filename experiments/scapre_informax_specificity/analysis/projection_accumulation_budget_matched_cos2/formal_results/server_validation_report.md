# Budget-matched direct-cos2 Confuse5 validation report

This server-generated report is fail-closed through formal aggregation. The semantic-pattern condition remains for manual review; no numerical threshold is introduced here.

## Formal official comparison

| Variant | Role | Unlearn | Preserve | Overall | Delta U | Delta P | Delta O | Favorable U/P/O |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official | formal_baseline | 19.2667 | 40.4556 | 53.9007 | +0.0000 | +0.0000 | +0.0000 | - |
| projection_accumulation | historical_descriptive_only | 19.7667 | 45.8444 | 58.3487 | +0.5000 | +5.3889 | +4.4480 | 1/5, 5/5, 5/5 |
| projection_accumulation_direct_cos2 | historical_descriptive_only | 19.1000 | 40.5111 | 53.9864 | -0.1667 | +0.0556 | +0.0857 | 3/5, 3/5, 2/5 |
| projection_accumulation_budget_matched_cos2 | formal_treatment | 19.3667 | 40.7222 | 54.1146 | +0.1000 | +0.2667 | +0.2139 | 1/5, 3/5, 3/5 |

Only official versus budget-matched cos2 is the formal comparison. V1 and direct-cos2 are historical descriptive context.

## Automatic directional conditions

| Condition | Result |
| --- | --- |
| Mean Delta Unlearn < 0 | False |
| Mean Delta Preserve > 0 | True |
| Mean Delta Overall > 0 | True |
| Overall favorable >= 4/5 | False |
| Automatic conditions passed | False |
| Group/target semantic pattern | manual review required |

## Group mean deltas

| Group | Delta Unlearn | Delta Preserve | Delta Overall |
| --- | ---: | ---: | ---: |
| dogs | +0.6667 | +0.4444 | +0.2005 |
| cats | +0.1667 | +0.5000 | +0.6150 |
| fruits | -0.0833 | +0.2778 | +0.3985 |
| boats | -0.7500 | -0.1667 | +0.3569 |
| balls | +0.5000 | +0.2778 | +0.2462 |

## All concept mean deltas

| Group | Role | Concept | Mean delta accuracy | Favorable seeds |
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

## Decision contract

If any automatic directional condition fails, stop the geometric modification branch and do not run COCO. If all automatic conditions pass, stop for manual semantic-pattern review; COCO is never launched automatically.

Retrieval validation is produced only after the archive is downloaded and independently checked on the Mac.
