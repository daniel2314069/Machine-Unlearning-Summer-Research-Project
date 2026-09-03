# ScaPre paper-MI formal comparison: validation report

## Overall assessment: Ready to share with caveats

The fixed five-seed Confuse5 comparison completed successfully and its score
calculations were independently reproduced after download. Paper MI forgets
the target concepts more strongly, but its preservation and combined overall
score are substantially worse for every seed. It should not replace the
repository baseline under this evaluation.

No parameter search was performed, and no further COCO generation was
launched.

## Aggregate result

| Variant | Target/unlearn accuracy ↓ | Preserve accuracy ↑ | Overall accuracy ↑ |
| --- | ---: | ---: | ---: |
| Repository baseline | 19.2667 | 40.4556 | 53.9007 |
| Paper MI | 5.7167 | 20.4111 | 33.5565 |
| Paper − repository | **−13.5500** | **−20.0444** | **−20.3442** |

The paired sample standard deviations are 0.5089, 0.2761, and 0.3709 points
respectively. All five seeds favor paper MI on lower target accuracy; zero of
five favor it on preservation or overall accuracy.

Descriptive paired t intervals across the five fixed seeds are
`[-14.1819, -12.9181]` for target accuracy, `[-20.3873, -19.7016]` for
preservation, and `[-20.8048, -19.8836]` for overall. They summarize seed
sensitivity rather than a broader model/dataset population.

## Paper-alpha behavior

All 160 seed × projection × layer summaries are effectively uniform:

- global minimum alpha: `0.9999999404`;
- per-matrix mean alpha range: `0.9999999892` to `0.9999999978`;
- max-over-concept raw MI range: `0.6931471229` to `0.6931471825`.

The raw MI saturates near `ln(2)`. After max over concepts and channel
normalization, `B = diag(alpha)` is numerically almost the identity. The result
therefore shows that this raw binary-MI estimator/setup loses useful channel
selectivity under the fixed protocol.

## Validation evidence

- The archive contained only safe relative paths, and all 101 files named by
  its package manifest were present.
- Independent recomputation covered 30,000 score rows: 3,000 rows per variant
  per seed, with 1,200 target and 1,800 retain rows per paired evaluation.
- Prompt/generation keys matched exactly between variants for every seed.
- Score hashes, metrics, deltas, means, sample standard deviations, controlled
  provenance hashes, protocol, and evaluator fingerprints all matched.
- Every paper edit reported 320 raw-MI records, 32 aggregate records, 1,280
  controlled Informax draws, finite projection weights, and passed formula
  checks.
- The repository arm reused the checksum-pinned historical scores only after
  all five score hashes, protocol, evaluator, assets, and historical sources
  passed validation.

The paper run emitted 933 Diffusers safety-checker warnings among 15,000
generated images (6.22%). Matching historical official logs contain 1,107
warnings among 15,000 images (7.38%). The preservation loss is therefore not
explained by a higher aggregate safety-blackout count in the paper arm, though
the score CSV lacks a per-image NSFW flag for a stricter matched analysis.

## Caveats and decision

The local archive SHA-256 is
`f2d45cc264896dd03435fa9fa7704c01f922b32e31020f7e92b480b8b695f99a`.
The archive was readable and internally complete. This validation session
could not independently re-read the server checksum because SSH authentication
was rejected; `download_results.sh`, when used, performs that comparison before
reporting success.

This is the project-established 25-class Confuse5 reconstruction, not an exact
author-released paper seed asset. Treat paper MI as a negative fixed
comparison: stronger forgetting comes with unacceptable preservation/overall
loss, while alpha collapses to an almost uniform vector. Do not promote it over
the repository baseline or run first-1k/first-10k COCO unless the next explicit
research question is the raw-MI saturation itself.
