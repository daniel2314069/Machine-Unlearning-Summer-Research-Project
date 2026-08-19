# Long-Horizon Sequential OCE Celebrity Results

## Status

The fixed 10×10 sequential celebrity experiment completed its core GCD protocol. 
MS-COCO evaluation is explicitly deferred by the Lightning free-credit budget and is not silently substituted.

## Fixed protocol

- Budget profile: `profile_5`
- Orders: official order and its exact reverse
- Conditions: baseline and retain full history
- Introduction success threshold: raw GCD accuracy ≤ 10%
- GCD denominator: images with a detected face

## Paper checkpoint metrics

| Order | Condition | Erased | Acc_e ↓ | Acc_s ↑ | H_o ↑ |
|---|---|---:|---:|---:|---:|
| order_a | baseline | 10 | 0.0101 | 0.9468 | 0.9679 |
| order_a | baseline | 50 | 0.0295 | 0.9013 | 0.9346 |
| order_a | baseline | 100 | 0.0478 | 0.8854 | 0.9176 |
| order_a | retain_history | 10 | 0.0101 | 0.9468 | 0.9679 |
| order_a | retain_history | 50 | 0.0228 | 0.9161 | 0.9457 |
| order_a | retain_history | 100 | 0.0082 | 0.8542 | 0.9178 |
| order_b | baseline | 10 | 0.0061 | 0.9553 | 0.9742 |
| order_b | baseline | 50 | 0.0165 | 0.9402 | 0.9614 |
| order_b | baseline | 100 | 0.0367 | 0.8645 | 0.9112 |
| order_b | retain_history | 10 | 0.0061 | 0.9553 | 0.9742 |
| order_b | retain_history | 50 | 0.0102 | 0.9361 | 0.9622 |
| order_b | retain_history | 100 | 0.0082 | 0.8755 | 0.9300 |
| joint | joint_100 | 100 | 0.0467 | 0.9270 | 0.9400 |

## Immediate-erasure validity

| Order | Condition | Successfully erased at introduction | Failed at introduction |
|---|---|---:|---:|
| order_a | baseline | 93 | 7 |
| order_a | retain_history | 95 | 5 |
| order_b | baseline | 88 | 12 |
| order_b | retain_history | 95 | 5 |

## Interpretation rule

Only concepts that met the predeclared introduction threshold may be described as a successful erasure that later reappeared. If baseline trajectories do not show clear cumulative reappearance, this sequential direction is a negative result; retain-history is not a necessary improvement.

No celebrities, orders, metrics, or checkpoints may be selected after viewing results. The complete individual trajectories are in `trajectory_per_concept.csv`.

## Repository versus paper

The experiment followed the frozen current-repository edit and evaluator behavior. The manifest records the paper's `celebrity` anchor description versus the repository E10 guides, and the paper's `Melanie Grifftih` spelling versus the repository's `Melanie Griffith`; no silent reconciliation was made.

## Joint-100 reference

Joint-100 Acc_e=0.0467, Acc_s=0.9270, H_o=0.9400.

## Artifacts

Qualitative archive: `/teamspace/studios/this_studio/artifacts/sequential_oce_celebrity_long_horizon_v1/qualitative_samples.tar.gz`
