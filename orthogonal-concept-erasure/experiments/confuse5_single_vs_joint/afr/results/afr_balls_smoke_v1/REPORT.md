# AFR implementation, matrix QA, and conditional balls smoke

## Current result

**Matrix classification: AFR-GO.** All matrix gates passed; the conditional balls smoke is authorized.

This experiment-local implementation leaves production `oce.py`, frozen Confuse5 settings, and all existing checkpoints untouched. F is the pure residual projection ablation; G is full AFR with anchor-fixed orthogonal compensation. Primary alpha is exactly `1`.

## Closed form

For `D=I-R_e` and `P=HH^T+H_perp Q H_perp^T`, expanding `tr[(PD-I)S(PD-I)^T]` shows the variable-dependent term is `-2 tr(P^T S D)`. The anchor-complement block is therefore `M_perp=H_perp^T S D H_perp`. If `M_perp=U Sigma V^T`, standard O(d) Procrustes gives `Q=UV^T`. No determinant correction is used.

## Synthetic QA

- alpha=0 explicit no-op: `True`, transform error `0`;
- alpha=1 F/G target leakage: `7.9744613e-31` / `8.322887e-31`;
- alpha=1 F/G anchor error: `1.5620817e-30` / `1.2832588e-30`;
- normalized frozen-S F/G: `0.44095424` / `0.35451987`;
- compensation trace error `0`, Gram residual `1.5577092e-15`;
- closed-form AFR loss `21.870339` versus best of 256 random feasible compensations `32.455476`.

## Group means

| Group | C leakage | F leakage | G leakage | F anchor | G anchor | C frozen-S | F frozen-S | G frozen-S | F-G | Compensation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dogs | 0.60860554 | 1.5994192e-28 | 1.6072238e-28 | 1.4423913e-28 | 1.8712798e-28 | 0.12014007 | 0.082489932 | 0.019122123 | 0.063367809 | 0.54612004 |
| fruits | 0.63943658 | 8.6159211e-29 | 8.6788726e-29 | 6.6621635e-29 | 8.7833449e-29 | 0.12455055 | 0.066798031 | 0.017217333 | 0.049580698 | 0.54572359 |
| balls | 0.56237129 | 1.0515126e-28 | 1.0603877e-28 | 1.2504139e-28 | 1.542888e-28 | 0.12379583 | 0.080137637 | 0.021679385 | 0.058458252 | 0.5462913 |
| overall | 0.60347113 | 1.1708413e-28 | 1.1784996e-28 | 1.1196738e-28 | 1.4308341e-28 | 0.12282882 | 0.0764752 | 0.019339613 | 0.057135586 | 0.54604498 |

## Group medians

| Group | C leakage | F leakage | G leakage | F anchor | G anchor | C frozen-S | F frozen-S | G frozen-S | F-G | Compensation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dogs | 0.64256821 | 4.0237228e-29 | 4.1666524e-29 | 2.7746875e-29 | 5.3989865e-29 | 0.098083235 | 0.075849411 | 0.016309966 | 0.060687079 | 0.36439929 |
| fruits | 0.711417 | 4.6645018e-29 | 4.7012898e-29 | 1.1658031e-29 | 4.9236195e-29 | 0.10695471 | 0.072785447 | 0.016936548 | 0.055641926 | 0.36360628 |
| balls | 0.58502493 | 2.3155459e-29 | 2.4277893e-29 | 2.3691745e-29 | 3.2179335e-29 | 0.10946153 | 0.084474964 | 0.021267063 | 0.060497578 | 0.36356873 |
| overall | 0.62725931 | 3.3511846e-29 | 3.5228715e-29 | 2.1776553e-29 | 4.8753769e-29 | 0.10186259 | 0.077331554 | 0.017206805 | 0.058601928 | 0.3639574 |

## F versus G preservation ablation

- overall median F distortion: `0.077331554`;
- overall median G distortion: `0.017206805`;
- median F-G: `0.058601928`;
- G < F in `48/48` layers at the numerical measurable threshold `1e-10`;
- material F-G >= `0.01` in `48/48` layers;
- compensation magnitude distribution (min/Q25/median/Q75/max): `0.27223253 | 0.28231733 | 0.3639574 | 0.92439194 | 0.93411577`.

## Layer distributions

| Quantity | Min | Q25 | Median | Q75 | Max |
|---|---:|---:|---:|---:|---:|
| F target feature leakage | 3.8359002e-30 | 1.768587e-29 | 3.3511846e-29 | 1.0975394e-28 | 1.3790641e-27 |
| G target feature leakage | 4.6421927e-30 | 1.8111774e-29 | 3.5228715e-29 | 1.1161331e-28 | 1.3758989e-27 |
| F anchor feature error | 2.6732158e-30 | 7.0242198e-30 | 2.1776553e-29 | 8.7151953e-29 | 1.4194598e-27 |
| G anchor feature error | 5.3288384e-30 | 1.6626866e-29 | 4.8753769e-29 | 1.1896079e-28 | 1.6379179e-27 |
| F normalized frozen-S | 0.028879182 | 0.036878825 | 0.077331554 | 0.10219152 | 0.17379784 |
| G normalized frozen-S | 0.0093471947 | 0.013242623 | 0.017206805 | 0.024204299 | 0.04046774 |
| F-G normalized frozen-S | 0.018128143 | 0.024257958 | 0.058601928 | 0.078932266 | 0.1333301 |
| G Gram residual | 1.7860852e-12 | 1.8861812e-12 | 6.6471365e-12 | 2.4683735e-11 | 2.6060541e-11 |

The anchor guarantee means **exact preservation of the constrained anchor features at the edited layer** only; it is not a claim of invariant anchor generation.

## Matrix gate

- F/G exact leakage and anchor error: maximum `1.6379179e-27`;
- structural QA maximum residual: `2.6176405e-11`;
- G worse than F preservation: `0/48` layers;
- nonzero compensation: `48/48` layers;
- measurable preservation improvement: `48/48` layers;
- material preservation improvement: `48/48` layers.

## Balls image smoke

**Final classification: AFR-I0.**

New images: `1800`. Runtime: `6412.8` seconds. Original, Single, and released Joint context was read from completed artifacts; none was regenerated.

### Target erasure and literal target-to-anchor movement

| Variant | Target | Target accuracy | Target probability | Matched-anchor top-1 | Matched-anchor probability |
|---|---|---:|---:|---:|---:|
| C_objective_faithful | soccer ball | 0.15 | 0.055847057 | 0.31 | 0.10423773 |
| C_objective_faithful | volleyball | 0 | 0.0058464174 | 0.02 | 0.012025685 |
| F_pure_residual_projection | soccer ball | 0.92 | 0.41367923 | 0.02 | 0.011064424 |
| F_pure_residual_projection | volleyball | 0.09 | 0.036250176 | 0 | 0.0031155858 |
| G_full_afr | soccer ball | 0.87 | 0.41498579 | 0.06 | 0.016875413 |
| G_full_afr | volleyball | 0.09 | 0.041115977 | 0 | 0.0033044233 |

### Existing first-100 context (read only)

| Concept | Original | Released Joint | Matched Single / Single soccer | Single volleyball |
|---|---:|---:|---:|---:|
| soccer ball | 0.95 | 0.13 | 0.1 | n/a |
| volleyball | 0.59 | 0.02 | 0 | n/a |
| tennis ball | 0.99 | 0.99 | 0.98 | 0.99 |
| rugby ball | 0.74 | 0.81 | 0.81 | 0.81 |
| ping-pong ball | 0.8 | 0.77 | 0.83 | 0.8 |

### Similar non-target preservation

| Variant | tennis ball | rugby ball | ping-pong ball | Macro |
|---|---:|---:|---:|---:|
| C_objective_faithful | 0.99 | 0.72 | 0.79 | 0.83333333 |
| F_pure_residual_projection | 0.96 | 0.54 | 0.75 | 0.75 |
| G_full_afr | 0.98 | 0.6 | 0.86 | 0.81333333 |

### Anchor generation

| Variant | Anchor | Accuracy | Anchor probability | Original-vs-edited LPIPS (8 fixed seeds) |
|---|---|---:|---:|---:|
| C_objective_faithful | basketball | 0.96 | 0.427163 | 0.23581218 |
| C_objective_faithful | baseball | 0.92 | 0.46241405 | 0.2903013 |
| F_pure_residual_projection | basketball | 0.94 | 0.41206174 | 0.1362051 |
| F_pure_residual_projection | baseball | 0.94 | 0.46778448 | 0.15128503 |
| G_full_afr | basketball | 0.96 | 0.4066864 | 0.1027542 |
| G_full_afr | baseball | 0.92 | 0.46942542 | 0.11602453 |

### Smoke gates and F/G ablation

- C-to-F/G target success: `False` / `False`; G both targets improve: `False`.
- Target-to-anchor probabilities move for both targets under F/G: `False` / `False`.
- G similar preservation acceptable: `True`; G anchor generation acceptable: `True`.
- G-F similar macro accuracy: `0.063333333`; G-F anchor macro accuracy: `0`; F-G LPIPS: `0.034355701`.
- Compensation image benefit: `True`.

### Artifacts and provenance

- Command: `python experiments/confuse5_single_vs_joint/afr/runner.py all --skip-existing`
- Git commit: `51170e7082520324e9affe8bd944e7a05addc064`; dirty at completion: `True`
- Generation root: `/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/balls_smoke/images`
- Evaluator root: `/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/balls_smoke/evaluations`
- LPIPS output: `/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/balls_smoke/anchor_lpips.json`
- C_objective_faithful checkpoint: `/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/checkpoints/C_objective_faithful/weights.safetensors`; metadata: `/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/checkpoints/C_objective_faithful/metadata.json`
- F_pure_residual_projection checkpoint: `/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/checkpoints/F_pure_residual_projection/weights.safetensors`; metadata: `/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/checkpoints/F_pure_residual_projection/metadata.json`
- G_full_afr checkpoint: `/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/checkpoints/G_full_afr/weights.safetensors`; metadata: `/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/checkpoints/G_full_afr/metadata.json`

## Reproducibility

- Matrix CSV: `results_c_f_g.csv` (144 rows = 48 cases x C/F/G)
- GPU command: `python experiments/confuse5_single_vs_joint/afr/runner.py all --skip-existing`
- Git commit: `51170e7082520324e9affe8bd944e7a05addc064`; dirty at matrix start: `True`
- Config SHA-256: `416ad7fd9e7666f8cd295ef6de4c6cf6af26d67502fd21d4027b6a81aa7e762b`
- Anchors SHA-256: `392802a728f5f726870718c2f9d0885ed57c8e16232899710f22690cee6c13b1`
- Qualification SHA-256: `b96514c7bd94e4d703079ddff238731c05ba76cde7fe6b0c2643d404bf7f043f`
- K0 SHA-256: `5d8ee50935eb3d22e1f9bc84947572afba386d05e063f70ea161c5cbf1e16235`
- Frozen Variant C CSV SHA-256: `546789d6d939cccfa57551994630be7e93ebc4ff0a05527d68dfe32ba5722343`
- Matrix dtype: `torch.float64`; checkpoint dtype: `torch.float32`
- Matrix runtime: `260.9` seconds
