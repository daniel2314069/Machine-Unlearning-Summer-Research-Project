# Sequential OCE object-persistence experiment

> **ABANDONED — NOT FOR PAPER CLAIMS.** The run completed, but the design does
> not answer the intended retain-persistence or general sequential-effect
> questions. Raw artifacts are retained only for audit and reproducibility. See
> [`../ABANDONED_SEQUENTIAL_OBJECT_EXPERIMENTS.md`](../ABANDONED_SEQUENTIAL_OBJECT_EXPERIMENTS.md).

Do not launch additional runs from this directory as a continuation of the
study. A replacement requires a separately named, redesigned protocol.

This directory implements two cumulative, ten-step single-concept OCE chains:

- `retain_once`: X is in the explicit local retain set at W01 only.
- `retain_always`: X is in the explicit local retain set at every step.

Each step starts from the preceding checkpoint in that condition. It does not
return to SD v1.4 and does not perform a joint multi-concept edit. Checkpoint
construction calls the current repository `oce.Orthogonal_Erase` function; the
runner only orchestrates state loading, persistence, generation, and evaluation.

## Fixed protocol

The ordered target-to-anchor mapping is the OCE object-erasure table:

| Step | Target | Anchor |
|---:|---|---|
| 1 | airplane | sky |
| 2 | automobile | truck |
| 3 | bird | cat |
| 4 | cat | dog |
| 5 | deer | horse |
| 6 | dog | cat |
| 7 | frog | bird |
| 8 | horse | deer |
| 9 | ship | airplane |
| 10 | truck | ship |

The object setting uses `erase_scale=1000`,
`preserve_global_scale=50`, `preserve_concept_scale=1`, float32 editing,
object prompt expansion, and the repository's existing `lamb=10` behavior.
The current anchor remains an explicit preserved concept at every step; X is the
only intended difference between the two conditions.

Generation uses the repository object-evaluation setup: SD v1.4, its default
PNDM scheduler, 50 steps, CFG 7.5, 512x512, bfloat16, and the prompt
`a photo of the {concept}`. Formal seeds are 42 through 141, shared by every
condition/checkpoint/concept cell.

Erased targets use the repository's ten-class CLIP ViT-B/32 classification
context. X uses an additional eleven-class context containing those ten labels
plus X. Qualification generates 20 SD v1.4 images per candidate and picks the
first candidate with at least 50% eleven-class top-1 accuracy. This threshold and
candidate order are recorded in `config.json`.

## Server execution

Run this only on the GPU server, after selecting its project environment:

```bash
conda activate MU
cd orthogonal-concept-erasure/experiments/sequential_object_persistence
bash run_server.sh
```

`run_server.sh` submits a detached `nohup` worker and returns after printing its
PID and log path. Once it reports that the job started, the SSH terminal may be
closed without sending a hangup signal to the experiment. The active Conda
environment and its resolved `python` executable are inherited by the worker.
The launcher refuses Conda `base`, a missing Conda environment, or a duplicate
active job.

Check progress after reconnecting:

```bash
cd orthogonal-concept-erasure/experiments/sequential_object_persistence
bash status_server.sh
```

Follow the current log continuously with:

```bash
bash status_server.sh --follow
```

Runtime PID, exit-code, and log-pointer files are kept under the ignored `.run/`
directory. A completed run reports `complete (exit 0)`; a nonzero exit is shown
as `failed` and the last log lines are printed. `nohup` protects against terminal
closure, but it cannot protect against a server reboot, administrator action, or
a cluster policy that kills all processes at logout. On a managed cluster with
a scheduler, submit the worker through that scheduler instead.

The repository-level `orthogonal-concept-erasure/Cg.pt` must already exist. If
it is absent on a new server checkout, create it once with the repository's
standard preparation step before launching this experiment:

```bash
cd orthogonal-concept-erasure
python compute_Cg.py
```

By default, model files must already be cached. If the server intentionally
needs to download missing SD/CLIP files, pass `--allow-downloads`:

```bash
bash run_server.sh --allow-downloads
```

The launcher uses the active environment's resolved `python`; it does not embed
a Conda environment name. The full command is restart-safe at completed cell and
checkpoint boundaries. A source/config fingerprint prevents accidental reuse
after protocol changes.

To keep generated PNG files instead of using the configured safe cleanup mode:

```bash
bash run_server.sh --image-retention keep \
  --output-dir outputs/sequential_oce_object_v1_keep
```

Use a distinct output directory when changing a protocol-affecting option.

For debugging, phases can also be run interactively in the foreground:

```bash
python run_sequential_oce.py preflight
python run_sequential_oce.py qualify
python run_sequential_oce.py build
python run_sequential_oce.py evaluate
python run_sequential_oce.py aggregate
```

`plan` prints the resolved static design without importing or loading models:

```bash
python run_sequential_oce.py plan
```

## Formal count

- Original W0: 10 CIFAR-10 concepts plus X = 11 cells.
- Each condition: sum of `(t erased targets + X)` from t=1..10 = 65 cells.
- Total: `11 + 2 * 65 = 141` cells.
- At 100 images per cell: 14,100 formal images.
- Qualification adds 20 images for each tried candidate and stops at the first
  acceptable candidate (normally 20 images if elephant qualifies).

## Output layout

```text
outputs/sequential_oce_object_v1/
├── resolved_protocol.json
├── run_state.json
├── events.jsonl
├── inputs/
│   ├── target_anchor_mapping.csv
│   └── formal_seeds.csv
├── qualification/
│   ├── results.csv
│   └── summary.json
├── checkpoints/
│   ├── retain_once/
│   └── retain_always/
├── images/                         # empty/absent after safe cleanup by default
├── raw/
│   ├── cells/<group>/<W>/<concept>/
│   │   ├── generation_manifest.json
│   │   ├── predictions.csv
│   │   ├── metrics.json
│   │   └── complete.json
│   └── formal_per_image_predictions.csv
├── tables/
│   ├── aggregated_cells.csv
│   ├── previous_erasure_persistence_retain_once.csv
│   ├── previous_erasure_persistence_retain_always.csv
│   ├── retain_persistence.csv
│   └── old_target_resurgence.csv
├── figures/
│   ├── retain_persistence_curve.{png,pdf}
│   └── previous_erasure_persistence_heatmaps.{png,pdf}
├── summary.json
└── summary.md
```

With `delete-after-eval`, a cell's PNGs are deleted only after all expected
predictions and aggregate metrics are written and re-read successfully. The
generation manifest and per-image evaluator row retain the image index, seed,
and original relative path. Checkpoints and all evaluator artifacts are kept.

The automated summary treats an absolute accuracy increase/drop of 0.10 as a
material change. Raw trajectories and exact values remain available in the CSV
tables for interpretation under another threshold.
