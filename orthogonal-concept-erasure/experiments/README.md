# Experiment infrastructure

Reusable Original-model preservation baselines are registered in
[`evaluation_references/`](evaluation_references/README.md). New evaluation
experiments must resolve a matching repository-level reference before
regenerating Original SD samples.

Experiment-specific checkpoints, edited generations and reports remain inside
their own experiment directories.

## Baseline diagnostics

- [`confuse5_single_vs_joint/`](confuse5_single_vs_joint/README.md): plans and
  launches matched group-wise single-target versus joint-target OCE edits using
  the official ImageNet-Confuse5 assignments from ScaPre Table 7.
