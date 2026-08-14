# Sequential OCE object-persistence summary

- Selected X: **elephant**
- Formal scope: **141 concept-checkpoint cells / 14,100 images**
- Qualification overhead: **20 images**
- Material-change rule: absolute accuracy change of at least **0.10**
- Image retention: `delete-after-eval`

## Qualification

- `elephant`: top-1 accuracy 1.000; accepted = `True`

The evaluator is CLIP ViT-B/32. Qualification and X preservation use an 11-class context (the ten CIFAR-10 labels plus X); erased CIFAR targets keep the unchanged 10-class context.

## Answers

- **Retain persistence:** elephant remained at 1.000 accuracy at W00 and every W01--W10 checkpoint under both conditions. Retain Once therefore showed no degradation in this run.
- **Retain Once vs Retain Always:** no difference was observed. Repeating elephant in every explicit retain set provided no measured benefit over retaining it only at W01 in this ten-step setting. This conclusion is limited to elephant and the saturated 11-class CLIP assay used here.
- **Previous-erasure persistence:** material resurgence was observed. The clearest case was dog: 0.13 at its Retain Once erasure checkpoint to 0.85 at W08, and 0.11 to 0.79 under Retain Always. Bird also rose from 0.47 to 0.91 and from 0.41 to 0.91. Retain Always automobile rose only 0.68 to 0.79 and was already high immediately after erasure, so it is a threshold crossing rather than strong evidence of disappearance followed by reappearance.

## Artifacts

- `raw/formal_per_image_predictions.csv`: all per-image evaluator outputs
- `tables/aggregated_cells.csv`: one row per formal cell
- `tables/previous_erasure_persistence_*.csv`: checkpoint × erased-target tables
- `tables/retain_persistence.csv`: X preservation comparison
- `tables/old_target_resurgence.csv`: per-target maximum later increase
- `figures/retain_persistence_curve.png`: X curve
- `figures/previous_erasure_persistence_heatmaps.png`: old-target heatmaps

Generation manifests retain every image index, seed, and original relative path. When `delete-after-eval` is active, PNG files are removed only after evaluator outputs and aggregate metrics for that cell have been validated and saved.
