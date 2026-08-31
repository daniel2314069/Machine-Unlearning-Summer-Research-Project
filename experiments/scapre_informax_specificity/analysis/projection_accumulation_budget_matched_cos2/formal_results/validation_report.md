# Budget-matched direct-cos2 Confuse5 validation report

## Overall assessment: valid, but does not qualify for COCO

The formal run is internally valid, and the treatment did exactly match each
official per-concept/per-matrix accumulation contribution's Frobenius norm.
The result nevertheless fails the predefined continuation rules: mean Unlearn
Accuracy worsened by `+0.1000`, and Overall was favorable in only `3/5` edit
seeds rather than the required `4/5`. Mean Preserve and Overall improved
slightly, but the changes are small and heterogeneous.

The server run covered 2026-08-30 08:39:18 UTC through 2026-08-30 23:54:25 UTC
(`15 h 15 m 7 s`). Treatment evaluation added 15,000 images. Historical
official scores were reused only after protocol, source hashes, ordered row
keys, and evaluator compatibility passed validation. No COCO generation ran.

## Integrity and qualification review

- The downloaded archive passed gzip integrity and safe-relative-path checks.
  Its local SHA-256 is
  `0999f4bf7af0fbc8d6350fc6411b41ade382aa7d2e218f3ed8ecb0ca5fba8890`.
  No server checksum sidecar accompanied the downloaded file, so transport
  checksum comparison is unavailable; internal manifests and hash chains did
  validate successfully.
- Qualification passed at edit seed `20260820`. All 320 concept/matrix records
  were finite and non-degenerate. Every recorded norm match passed the frozen
  `rtol=1e-5, atol=1e-7`; the independently checked
  `||C_new||_F / ||C_official||_F` range was
  `0.9999998808` to `1.0000001192`.
- Lambda was finite and positive in all 320 records: minimum `0.181495`, median
  `1.660029`, mean `2.901242`, p95 `10.719630`, p99 `19.244119`, and maximum
  `36.419144`. These values are descriptive only and were not used to tune the
  formula.
- The treatment checkpoint was finite, different from official, and non-zero:
  its checked projection-weight delta was `0.400695` in Frobenius norm,
  relative to official `0.001276`.
- The production editor remained byte-identical at SHA-256
  `cc454407a70de5b403344f8e3d0372044fed156cf78a74fa04121473674ada20`.
  The experiment-only path substituted only the two accumulation occurrences
  of `for_mat1 * row_w_c`.
- Across all five seeds, official and treatment independently matched on 1,280
  Informax random draws, 320 accumulation intercepts, 32 matrix records,
  Informax event-stream hashes, pre-draw and final RNG states, entropy
  positions, target/empty embeddings, `for_mat1`/`for_mat2`, S/R/geometry
  inputs, and bitwise final `row_w_max`.
- All ten checkpoint completion-to-evaluator-to-cleanup hash chains passed.
  Official and treatment used identical protocol rows and evaluator settings,
  with only checkpoint and variant identity differing.
- Independent local validation used Ruby standard-library CSV/JSON/Digest,
  not project Python. It parsed 70 JSON artifacts and ran 1,757 checks over
  30,000 raw score rows. There were no missing or duplicate generation keys;
  every per-seed, per-group, per-concept, aggregate, delta, and favorable-seed
  result matched the packaged outputs.

## Five-seed formal result

Delta is budget-matched direct cos2 minus official. Lower Unlearn and higher
Preserve/Overall are favorable.

| Metric | Official | Budget-matched cos2 | Delta | Favorable seeds |
| --- | ---: | ---: | ---: | ---: |
| Unlearn Accuracy | 19.2667 | 19.3667 | **+0.1000** | 1/5 |
| Preserve Accuracy | 40.4556 | 40.7222 | **+0.2667** | 3/5 |
| Overall Accuracy | 53.9007 | 54.1146 | **+0.2139** | 3/5 |

Per-seed deltas:

| Edit seed | Delta Unlearn | Delta Preserve | Delta Overall |
| --- | ---: | ---: | ---: |
| 20260820 | +0.1667 | -0.0556 | -0.0866 |
| 20260821 | +0.1667 | +0.4444 | +0.3564 |
| 20260822 | -0.0833 | +0.6111 | +0.5585 |
| 20260823 | +0.2500 | -0.2222 | -0.2529 |
| 20260824 | 0.0000 | +0.5556 | +0.4938 |

## Predefined decision rules

| Requirement | Result | Decision |
| --- | --- | --- |
| Mean Delta Unlearn < 0 | `+0.1000` | **FAIL** |
| Mean Delta Preserve > 0 | `+0.2667` | PASS |
| Mean Delta Overall > 0 | `+0.2139` | PASS |
| Overall favorable in at least 4/5 seeds | `3/5` | **FAIL** |
| No obvious concentrated/repeated family degradation | Manual pattern review below | Not used to rescue the failed automatic rules |

Because two automatic conditions fail, this experiment cannot advance to the
COCO safeguard regardless of the manual semantic-pattern assessment.

## Group and concept patterns

| Group | Delta Unlearn | Delta Preserve | Delta Overall | Pattern |
| --- | ---: | ---: | ---: | --- |
| dogs | +0.6667 | +0.4444 | +0.2005 | Preservation improves, but forgetting is worse in mean and favorable in only 1/5 seeds. |
| cats | +0.1667 | +0.5000 | +0.6150 | No V1-style tiger-cat outlier; forgetting is still slightly unfavorable. |
| fruits | -0.0833 | +0.2778 | +0.3985 | All three mean directions are favorable, with seed-level variation. |
| boats | -0.7500 | -0.1667 | +0.3569 | Strongest forgetting gain, offset by slightly worse preservation. |
| balls | +0.5000 | +0.2778 | +0.2462 | Forgetting is unfavorable and group Overall is favorable in only 2/5 seeds. |

Target behavior is not driven by one catastrophic outlier. Three of ten
targets have favorable mean forgetting deltas, six are unfavorable, and one
ties. The strongest gains are lifeboat (`-0.8333`, favorable in `4/5`), yawl
(`-0.6667`, `3/5`), and orange (`-0.5000`, `3/5`). The clearest repeated
unfavorable target is volleyball (`+1.0000`, favorable in `0/5`): it worsens in
two seeds and ties in three. Both dog targets average `+0.6667`; labrador is
favorable in only `1/5` seeds.

Among fifteen retain concepts, six have favorable mean deltas, six are
unfavorable, and three tie. The largest gains are siamese cat and fig
(`+1.6667` each) and chesapeake bay retriever (`+1.5000`); the largest losses
are granny smith (`-0.8333`) and ping-pong ball (`-0.6667`). These patterns are
mixed rather than a uniform semantic-family improvement.

## What budget matching resolved

The previous direct-cos2 variant changed both row allocation and contribution
strength. This treatment fixes each individual concept/matrix contribution
norm to official, so its formal comparison isolates row-wise allocation at
that scope. The match is exact to floating-point tolerance; no lambda clamp or
fallback occurred.

Per-concept norm matching does not require the sum across concepts to have the
same norm as official, because redistribution changes cross-concept alignment
and cancellation. Accordingly, solved-matrix treatment/official `V`-norm
ratios range from `0.7452` to `0.9947` (median `0.8969`). This is an expected
consequence of changing row allocation, not a failure of the registered
per-concept budget match.

For descriptive context only:

| Variant | Unlearn | Preserve | Overall | Delta U | Delta P | Delta O | Favorable U/P/O |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official | 19.2667 | 40.4556 | 53.9007 | 0.0000 | 0.0000 | 0.0000 | - |
| V1 zscore/sigmoid^8 | 19.7667 | 45.8444 | 58.3487 | +0.5000 | +5.3889 | +4.4480 | 1/5, 5/5, 5/5 |
| Direct cos2 | 19.1000 | 40.5111 | 53.9864 | -0.1667 | +0.0556 | +0.0857 | 3/5, 3/5, 2/5 |
| Budget-matched cos2 | 19.3667 | 40.7222 | 54.1146 | +0.1000 | +0.2667 | +0.2139 | 1/5, 3/5, 3/5 |

Restoring the official per-contribution budget changes direct cos2 by
`+0.2667` Unlearn, `+0.2111` Preserve, and `+0.1282` Overall. It modestly
recovers preservation/Overall but removes direct cos2's small forgetting gain;
neither direct variant shows the stable all-metric improvement required for
continuation.

## Conclusion

Budget matching successfully removed the intended per-concept contribution
strength confound, but row-wise cos2 reallocation alone does not stably improve
ScaPre. The result is a valid exploratory negative result: two registered
automatic conditions fail, and semantic effects remain heterogeneous.

**Final decision:** stop the geometric modification branch, do not design a
fourth cos2 variant, and do not run COCO.
