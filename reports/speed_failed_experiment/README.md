# SPEED negative experiment record

This directory preserves the compact evidence from the local SPEED experiments
after removing the large generated-image, log, and pretraining-data directories.
The original SPEED checkout remains recoverable from Git history.

## Main result

The experiment targeted erasure of `Snoopy` while using Mickey Mouse,
SpongeBob, Pikachu, Hello Kitty, and MSCOCO generation as preservation checks.

| Mode | Snoopy CLIP score | Avg. non-target FID | MSCOCO CLIP score | MSCOCO FID |
| --- | ---: | ---: | ---: | ---: |
| Original | 28.5403 | approximately 0 | 26.5021 | approximately 0 |
| SPEED | 23.4597 | 21.7929 | 26.4791 | 19.7026 |
| C-prime null, no IEC | 23.3071 | 28.7928 | 26.4445 | 26.8739 |
| C-prime direct equation, no null | 14.0491 | 333.5531 | 13.2808 | 289.4324 |

The direct-equation variant erased the target more strongly, but caused
catastrophic preservation and general-generation degradation. The less
aggressive variants only modestly reduced the target score.

The layer-group ablation also failed to provide a useful erasure/preservation
trade-off:

| Layer group | Snoopy CLIP score | Avg. non-target FID | MSCOCO FID |
| --- | ---: | ---: | ---: |
| Down only | 28.0853 | 19.9875 | 16.3659 |
| Mid only | 28.7447 | 14.3599 | 13.3855 |
| Up only | 26.3437 | 14.6567 | 11.4568 |

## Preserved artifacts

- `tables/`: final CSV/JSON tables and the layer map.
- `scripts/`: the local orchestration and aggregation scripts used for these
  runs. They depend on the original SPEED repository and are preserved for
  provenance, not as a standalone runnable package.
- `layer_group_report.tex`: compact experiment write-up.
- `figure/layer_group_grid.png`: representative qualitative comparison.

Raw generated images, verbose logs, downloaded pretraining data, and the paper
checkout were intentionally removed because they were large, reproducible, and
did not change the negative conclusion above.
