# Sequential OCE object follow-up experiments

This directory runs the two fixed empirical follow-ups to the completed
`sequential_object_persistence` experiment. It only orchestrates the repository's
existing single-concept OCE implementation.

## Hard generation gate

Before loading a generation pipeline, preflight writes:

- `inputs/cell_manifest.csv`
- `inputs/planned_generation.json`
- `inputs/formal_seeds_200.csv`

The run stops unless the plan resolves exactly to:

| Scope | Cells | New images | Final predictions/cell |
|---|---:|---:|---:|
| Experiment 1 direct single from W0 | 10 | 2,000 | 200 |
| Experiment 1 sequential own-step supplements | 20 | 2,000 | 200 (100 existing + 100 new) |
| Experiment 2 clean five-step chain | 15 | 3,000 | 200 |
| **Total new generations** | **45** | **7,000** | — |

Direct-single and clean-chain cells use seeds 42–241. Sequential supplementation
preserves the prior predictions at seeds 42–141 and adds seeds 142–241. Every
final cell is audited for exactly 200 unique seeds and exactly 200 evaluator rows.

The source result defaults to:

```text
../sequential_object_persistence/outputs/sequential_oce_object_v1_online
```

Its 20 own-step evaluator artifacts and server-side checkpoints must still be
present. Preflight stops before editing/generation if any required source artifact
or protocol setting is missing or inconsistent.

## Fixed experiments

Experiment 1 independently builds ten `W0 -> erase target` checkpoints using the
official mapping, then evaluates only that target at 200 images. It also evaluates
100 new non-overlapping seeds at each prior Retain Once/Retain Always own-step
checkpoint and combines them with the existing 100 predictions.

Experiment 2 is one genuinely sequential chain:

1. dog -> cat
2. bird -> cat
3. airplane -> sky
4. automobile -> truck
5. deer -> horse

At W1..W5 it evaluates all targets erased so far, yielding 15 cells. None of the
five anchors is a target in this chain. Dog and bird remain first and second.

Generation and evaluation match the prior run: SD v1.4, PNDM, 50 steps, CFG 7.5,
512x512, bfloat16 generation, `a photo of the {concept}`, and the unchanged
10-class CLIP ViT-B/32 evaluator.

## GPU server launch

After activating the server environment:

```bash
conda activate MU
cd orthogonal-concept-erasure/experiments/sequential_object_followup
bash run_server.sh
```

The launcher uses `nohup`, inherits the active environment's `python`, and returns
after submission. The terminal may then be closed. Check or follow status with:

```bash
bash status_server.sh
bash status_server.sh --follow
```

If model files are not cached but downloads are intentionally allowed:

```bash
bash run_server.sh --allow-downloads
```

The workflow resumes completed checkpoints and cells without overwriting them.
Use a new output directory for any protocol-affecting change.

The static plan command does not load models or require GPU artifacts:

```bash
python run_followup.py plan
```

## Outputs

```text
outputs/sequential_oce_object_followup_v1/
├── inputs/
│   ├── cell_manifest.csv
│   ├── planned_generation.json
│   └── formal_seeds_200.csv
├── checkpoints/
│   ├── direct_single_from_W0/
│   └── clean_five_step_chain/
├── raw/
│   ├── cells/
│   ├── experiment1_direct_per_image_predictions.csv
│   ├── experiment1_sequential_own_step_per_image_predictions_200.csv
│   └── experiment2_per_image_predictions.csv
├── tables/
│   ├── experiment1_comparison.csv
│   ├── experiment2_persistence.csv
│   └── experiment2_per_target_summary.csv
├── figures/
│   ├── experiment2_previous_erasure_heatmap.{png,pdf}
│   └── experiment2_trajectories.{png,pdf}
├── validation_report.md
├── summary.json
└── summary.md
```

With the default `delete-after-eval`, newly generated PNGs are removed only after
the corresponding evaluator output, aggregate metrics, 200-row count, and unique
seed audit have succeeded. Manifests and per-image predictions remain.
