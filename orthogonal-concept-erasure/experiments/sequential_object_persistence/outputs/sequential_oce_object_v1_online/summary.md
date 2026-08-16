# Sequential OCE object-persistence summary

> **RESEARCH STATUS: ABANDONED / NOT FOR CLAIMS.** Execution and raw-data audits
> completed, but the experiment does not identify the intended retain question
> and does not support a general causal claim about sequential edits. See
> [`../../../ABANDONED_SEQUENTIAL_OBJECT_EXPERIMENTS.md`](../../../ABANDONED_SEQUENTIAL_OBJECT_EXPERIMENTS.md).

- Selected X: **elephant**
- Formal scope: **141 concept-checkpoint cells / 14,100 images**
- Qualification overhead: **20 images**
- Material-change rule: absolute accuracy change of at least **0.10**
- Image retention: `delete-after-eval`

## Qualification

- `elephant`: top-1 accuracy 1.000; accepted = `True`

The evaluator is CLIP ViT-B/32. Qualification and X preservation use an 11-class context (the ten CIFAR-10 labels plus X); erased CIFAR targets keep the unchanged 10-class context.

## Answers

- **Retain persistence is unanswered:** elephant remained at 1.000 accuracy at
  W00 and every W01--W10 checkpoint under both conditions, but there is no
  Retain Never control and the elephant assay is saturated. The result cannot
  show that retaining once caused persistent protection or that retaining
  always is unnecessary.
- **Retain Once vs Retain Always:** no measurable difference appeared in this
  ceiling-limited assay. This is not evidence of equivalence between the two
  policies.
- **Previous-erasure trajectories:** dog and bird received higher CIFAR-10 CLIP
  target accuracy at later checkpoints in these exact fixed sequences. Because
  target, order, step, and retain history are confounded, this is a descriptive
  sequence-specific result rather than evidence of a general causal effect.

## Artifacts

- `raw/formal_per_image_predictions.csv`: all per-image evaluator outputs
- `tables/aggregated_cells.csv`: one row per formal cell
- `tables/previous_erasure_persistence_*.csv`: checkpoint × erased-target tables
- `tables/retain_persistence.csv`: X preservation comparison
- `tables/old_target_resurgence.csv`: per-target maximum later increase
- `figures/retain_persistence_curve.png`: X curve
- `figures/previous_erasure_persistence_heatmaps.png`: old-target heatmaps

Generation manifests retain every image index, seed, and original relative path. When `delete-after-eval` is active, PNG files are removed only after evaluator outputs and aggregate metrics for that cell have been validated and saved.
