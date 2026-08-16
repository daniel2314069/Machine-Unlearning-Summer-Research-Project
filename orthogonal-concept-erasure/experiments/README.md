# Experiment infrastructure

## Abandoned experiments

- [`sequential_object_persistence/`](sequential_object_persistence/README.md)
  and [`sequential_object_followup/`](sequential_object_followup/README.md) are
  **abandoned and must not be used for paper claims**. The raw artifacts are kept
  only for reproducibility and audit. See the
  [methodology disposition](ABANDONED_SEQUENTIAL_OBJECT_EXPERIMENTS.md).

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
