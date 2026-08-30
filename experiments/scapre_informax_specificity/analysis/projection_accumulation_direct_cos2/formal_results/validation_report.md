# Direct-cos2 accumulation Confuse5 validation report

## Overall assessment: valid, but does not qualify for COCO

The formal run is internally valid and all three five-seed means move in the
favorable direction. However, the changes are very small and Overall is
favorable in only `2/5` edit seeds, so the variant fails the predefined `4/5`
continuation requirement. The result is best characterized as an unstable,
near-official outcome rather than evidence of a robust improvement.

The server run covered 2026-08-29 12:58:28 UTC through 2026-08-30 04:12:52 UTC
(`15 h 14 m 24 s`). Treatment evaluation added 15,000 images. Historical
official scores were reused only after their protocol, assets, hashes, row
keys, and evaluator fingerprint passed validation. No COCO generation ran.

## Integrity and methodology review

- Qualification passed at edit seed `20260820`: direct scores/alphas were
  finite, non-constant, and exactly equal after the expected dtype cast; the
  treatment checkpoint was finite and differed from official.
- The production editor remained byte-identical at SHA-256
  `cc454407a70de5b403344f8e3d0372044fed156cf78a74fa04121473674ada20`.
- The experiment-only branch substituted exactly the two
  `for_mat1 * row_w_c` accumulation expressions. Official Informax still ran
  normally, and aggregate Informax/final `row_w_max` were not intercepted.
- Across all five seeds, official and treatment had identical Informax call
  signatures, ordering, random tensors, global RNG states, entropy positions,
  target/empty embeddings, accumulation inputs, geometry inputs, and
  bitwise-identical final `row_w_max`.
- Exact model snapshot, empty neutral, prompts, generation seeds, PNDM sampler,
  50 steps, CFG 7.5, 512x512 output, classifier, evaluator, scoring logic, and
  ordered row keys matched the integrity-validated official reference.
- Independent local validation parsed 68 JSON artifacts and recomputed all
  metrics from 30,000 raw score rows. It found no missing/duplicate keys, and
  all per-seed, per-group, per-concept, aggregate, delta, and favorable-seed
  values matched the packaged outputs.

## Five-seed result

Delta is direct-cos2 minus official. Lower Unlearn and higher Preserve/Overall
are favorable.

| Metric | Official | Direct cos2 | Delta | Favorable seeds |
| --- | ---: | ---: | ---: | ---: |
| Unlearn Accuracy | 19.2667 | 19.1000 | **-0.1667** | 3/5 |
| Preserve Accuracy | 40.4556 | 40.5111 | **+0.0556** | 3/5 |
| Overall Accuracy | 53.9007 | 53.9864 | **+0.0857** | 2/5 |

Per-seed deltas show why the positive Overall mean is not robust:

| Edit seed | Delta Unlearn | Delta Preserve | Delta Overall |
| --- | ---: | ---: | ---: |
| 20260820 | -0.0833 | -0.1111 | -0.0793 |
| 20260821 | -0.6667 | -0.1667 | -0.0022 |
| 20260822 | 0.0000 | +0.3333 | +0.2950 |
| 20260823 | +0.3333 | +0.0556 | -0.0257 |
| 20260824 | -0.4167 | +0.1667 | +0.2406 |

## Predefined decision rules

| Requirement | Result | Decision |
| --- | --- | --- |
| Mean Delta Unlearn < 0 | `-0.1667` | PASS |
| Mean Delta Preserve > 0 | `+0.0556` | PASS |
| Mean Delta Overall > 0 | `+0.0857` | PASS |
| Overall favorable in at least 4/5 seeds | `2/5` | **FAIL** |
| No obvious concentrated/repeated family degradation | Cat Overall worsened in all five seeds; manual pattern review below | Not clean |

The numerical continuation condition fails. Consequently, this result does
not authorize the COCO safeguard regardless of how the manual pattern criterion
is judged.

## Group and concept patterns

| Group | Delta Unlearn | Delta Preserve | Delta Overall | Pattern |
| --- | ---: | ---: | ---: | --- |
| dogs | +0.2500 | 0.0000 | -0.0632 | Approximately flat, slightly unfavorable. |
| cats | +0.2500 | -0.6111 | -0.7932 | Repeated Preserve/Overall degradation. |
| fruits | -0.5833 | -0.0556 | -0.0420 | Better forgetting, nearly flat elsewhere. |
| boats | -0.9167 | +0.3889 | +0.6958 | Most consistently favorable group. |
| balls | +0.1667 | +0.5556 | +0.5925 | Better preservation, slightly worse forgetting. |

The V1 tiger-cat failure did not recur: tiger-cat mean Delta Unlearn is only
`+0.1667`, not V1's `+40.1667`. But improvement is still heterogeneous. Among
the ten targets, five have favorable mean forgetting deltas and five have
unfavorable means; only four targets are favorable in at least three seeds.
The strongest target gains are orange and yawl (`-1.0000` each), while the
largest unfavorable target is volleyball (`+0.5000`).

For retain concepts, six of fifteen have favorable mean deltas, six are
unfavorable, and three tie. The cat group is the clearest repeated concern:
its Overall delta is negative in every seed, its Preserve delta is negative in
four seeds and tied once, and both cat targets have unfavorable mean forgetting
deltas. These magnitudes are modest, but the direction is repeated rather than
being caused by a single extreme target.

## What removing the V1 transform changed

Raw direct cos2 has mean `0.001318`, median `0.000510`, and maximum `0.059241`.
The V1 z-score/sigmoid-power transform had mean `0.068016`, median `0.000411`,
and maximum `1.0`; its mean weight was `51.62x` the raw-cos2 mean and its top
5% of rows carried `60.91%` of total alpha weight, versus `34.61%` for raw
cos2. Removing the transform therefore materially reduced and de-concentrated
the alpha.

The actual weighted contribution did not shrink by the full raw-alpha ratio
because alpha aligns non-uniformly with `for_mat1`: across 320 layer/concept
records, direct/official contribution norm had median `0.602` and mean `0.776`.
At the solved-matrix level, direct/official `V` norm had median `0.577` and mean
`0.712`. The edit was non-zero (checkpoint relative Frobenius delta `0.001027`)
but substantially closer to official than V1. This is consistent with the
formal metrics reverting almost completely to official.

For context only, direct cos2 versus V1 changed the five-seed means by
`-0.6667` Unlearn, `-5.3333` Preserve, and `-4.3623` Overall. Removing the
transform eliminated the catastrophic tiger-cat behavior, but it also removed
nearly all of V1's preservation/Overall gain.

## Conclusion

Direct cos2 is cleaner geometrically and avoided V1's extreme target failure,
but the formal data do not show a stable practical improvement over official.
The tiny favorable mean deltas coexist with only `2/5` favorable Overall seeds
and repeated cat-group degradation.

**Final decision:** accept this as a valid exploratory negative/near-null result,
stop the modification branch, and do not run COCO or tune another projection
formula from these results.
