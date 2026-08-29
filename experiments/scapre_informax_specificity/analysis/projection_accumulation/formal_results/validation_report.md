# Projection accumulation Confuse5 validation report

## Overall assessment: ready to share as a negative result

The formal run is internally valid and answers the preregistered question, but
the modification does **not** qualify for the COCO safeguard. It failed the
required mean Unlearn direction and exhibited a clear repeated target-level
degradation in the cat family. Per the preregistration, this research branch
stops here: no COCO, no formula tuning, and no second projection variant.

The archive was retrieved and validated on 2026-08-29 Asia/Taipei. The server
run covered 2026-08-28 17:00:53 UTC through 2026-08-29 08:16:22 UTC
(`15.258` hours). The edited-model evaluation added 15,000 images. Historical
official scores were reused only after their hashes, protocol, assets, source
compatibility, row keys, and evaluator fingerprint passed validation.

## Integrity and methodology review

- Qualification passed at edit seed `20260820`: all projection scores and
  alphas were finite and non-constant; both checkpoints were finite and had
  different hashes.
- The production editor remained byte-identical at SHA-256
  `cc454407a70de5b403344f8e3d0372044fed156cf78a74fa04121473674ada20`.
- The experiment-only branch substituted exactly the two
  `for_mat1 * row_w_c` accumulation expressions. Aggregate Informax and final
  `row_w_max` were not intercepted.
- Across all five seeds, official and treatment had identical Informax random
  draw signatures/order/tensors, relevant RNG states, entropy positions,
  embeddings, accumulation inputs, geometry inputs, and bitwise-identical
  final `row_w_max`.
- The exact SD1.5 revision, prompts, generation seeds, PNDM sampler, 50 steps,
  CFG 7.5, 512x512 output, classifier, evaluator, scoring logic, and ordered
  row keys matched the official reference.
- Independent local validation recomputed all 30,000 score rows without using
  the server aggregation output. All per-seed, per-group, per-concept, mean,
  delta, and favorable-seed values matched.

## Five-seed result

Delta is projection_accumulation minus official. Lower Unlearn and higher
Preserve/Overall are favorable.

| Metric | Official | Projection | Delta | Favorable seeds |
| --- | ---: | ---: | ---: | ---: |
| Unlearn Accuracy | 19.2667 | 19.7667 | **+0.5000** | 1/5 |
| Preserve Accuracy | 40.4556 | 45.8444 | **+5.3889** | 5/5 |
| Overall Accuracy | 53.9007 | 58.3487 | **+4.4480** | 5/5 |

The Overall gain is real under the established harmonic metric, but it is
driven by preservation gains and does not rescue the failed forgetting
direction. Four of five seeds had worse Unlearn Accuracy; only seed `20260822`
was favorable (`-0.1667`).

## Preregistered decision rules

| Requirement | Result | Decision |
| --- | --- | --- |
| Mean Delta Unlearn < 0 | `+0.5000` | **FAIL** |
| Mean Delta Preserve > 0 | `+5.3889` | PASS |
| Mean Delta Overall > 0 | `+4.4480` | PASS |
| Overall favorable in at least 4/5 seeds | `5/5` | PASS |
| No obvious concentrated/repeated family degradation | Cat target Delta Unlearn `+20.1667`; tiger cat `+40.1667`, unfavorable 5/5 | **FAIL** |

The preregistered first condition fails numerically. The third condition also
fails under manual pattern review: the cat degradation is large, repeated in
every seed, and concentrated most strongly in tiger cat.

## Group patterns

| Group | Delta Unlearn | Delta Preserve | Delta Overall | Interpretation |
| --- | ---: | ---: | ---: | --- |
| dogs | +1.6667 | +12.4444 | +8.9450 | Preservation improves, forgetting worsens. |
| cats | +20.1667 | +7.7222 | +5.3820 | Severe repeated target degradation despite higher Overall. |
| fruits | -0.7500 | +7.1667 | +9.4746 | Favorable at group level, but lemon worsens 5/5. |
| boats | -2.0833 | -0.5000 | +0.9823 | Forgetting improves; preservation slightly worsens. |
| balls | -16.5000 | +0.1111 | +2.0765 | Strong forgetting gain, nearly flat preservation. |

The improvement is not uniform. Target behavior ranges from tiger cat
`+40.1667` (much worse forgetting) to soccer ball `-24.3333` (much better).
Among ten target concepts, four were favorable in all five seeds, five were
unfavorable in all five seeds, and tabby was mixed/tied. This is a systematic
semantic redistribution, not a small common improvement.

Preservation gains are also concentrated. German shepherd (`+27.0000`),
Siamese cat (`+21.0000`), pug (`+11.8333`), and pomegranate (`+11.6667`)
provide large gains, while ping-pong ball (`-9.1667`) and catamaran
(`-3.6667`) degrade in all five seeds. Ten of fifteen retain concepts were
favorable in at least four seeds; three had zero favorable seeds.

## Qualification diagnostics

Official accumulation alpha remained narrow and saturated:

- median `0.0074633`, mean `0.0074322`, std `0.0012661`, p99 `0.0096332`.

Projection alpha was sparse and heavy-tailed under the frozen transformation:

- median `0.0004114`, mean `0.0680158`, std `0.1954939`, p95 `0.5669813`,
  p99 `0.9770934`, maximum `1.0`.

The two masks were descriptively almost unrelated across the 320
layer/concept records: mean Pearson correlation `0.040971` and mean Spearman
correlation `0.047526`. This confirms that the intervention was material rather
than a no-op, but it does not establish that the new geometry is a better
relevance statistic.

## Caveats and conclusion

- This experiment was designed for directional decision-making, not a
  statistical-significance claim.
- The established evaluator includes its normal Stable Diffusion safety
  checker behavior; evaluator and protocol fingerprints match the official
  reference.
- The positive Overall result should not be reported alone because it hides
  the failed mean forgetting direction and the cat-family degradation.

**Final decision:** accept the negative modification result and terminate the
projection-accumulation branch. Do not execute `run_projection_coco.sh`.

