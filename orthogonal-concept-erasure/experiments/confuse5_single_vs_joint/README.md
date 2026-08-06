# Confuse5 single-vs-joint OCE baseline

This experiment compares independent Single OCE edits with group-wise Joint
OCE edits. The existing `run.py` remains the checkpoint-building backend and
calls `../../oce.py` without changing the OCE objective. `pipeline.py` adds
planning, image generation, ImageNet ResNet-50 evaluation, aggregation, and a
low-disk server workflow.

## Scientific protocol and current coverage

`config.json` records the five ImageNet-Confuse5 groups from Table 7 of
*Forget Many, Forget Right*. Each group has two targets and three visually
similar non-targets. Editing uses SD1.4, the 16 `attn2.to_v` modules, OCE's
default null-text object anchor, and the existing group retain policy.

The repository dataset `../../../scapre/eval/datasets/imagenet-15.csv`
contains 500 exact prompt/seed rows for the two targets and one preservation
class in each group. Ten official preservation classes are absent:

- Chesapeake Bay retriever, pug;
- Siamese cat, Egyptian cat;
- fig, Granny Smith;
- catamaran, schooner;
- rugby ball, ping-pong ball.

The pipeline never fabricates these rows. `--coverage partial` explicitly runs
the available 15 classes. `--coverage complete` is blocked until an official
CSV with all 25 classes and 500 rows per class is supplied with
`--dataset-csv`. Adding that CSV requires no code change.

Generation is paired by exact CSV row across Original SD1.4, the corresponding
Single checkpoint, and the group Joint checkpoint. Settings are explicit and
shared: PNDM, 50 steps, CFG 7.5, 512x512, bfloat16, one image per prompt, and a
fresh CPU generator seeded with the row's `evaluation_seed`.

The partial scale is 30,000 images:

- Original: 15 classes x 500 = 7,500;
- 10 Single checkpoints x 3 available group classes x 500 = 15,000;
- 5 Joint checkpoints x 3 available group classes x 500 = 7,500.

The complete scale is 50,000 images. A generation stage refuses to start
unless `--confirm-image-count` exactly matches its resolved plan.

## Low-disk lifecycle

The recommended `all` stage processes one model/concept job at a time. A
formal job has 500 images. It generates those images, writes a complete
per-image ResNet-50 result shard atomically, and only then deletes the 500 PNGs.
Peak normal retention is therefore 500 images rather than 30,000 or 50,000.

Manifest states are:

```text
planned -> generating -> generated -> evaluated -> purged
```

Purged manifests retain prompt, seed, path, image hash, classifier prediction,
and timestamps. Failed or interrupted jobs keep their images. Purging only
unlinks explicit manifest paths contained by this experiment's image root. A
purge failure stops the run instead of allowing disk use to grow silently.

`--skip-existing` treats a matching evaluated/purged result as terminal, even
though its PNGs are gone. `--overwrite` is the explicit replacement option and
is mutually exclusive with `--skip-existing`. Use `--keep-images` to opt out of
purging. A standalone `generate` command must include `--retain-images`, since
images cannot be deleted before a later evaluation.

## Server commands

Run from the OCE repository root after activating the machine's environment:

```bash
conda activate MU                  # GPU server
# conda activate py310            # WSL alternative

# Complete validation currently reports the ten missing classes and exits 2.
python experiments/confuse5_single_vs_joint/pipeline.py plan \
  --coverage complete

# Available partial plan: total 30,000, peak retained 500.
python experiments/confuse5_single_vs_joint/pipeline.py plan \
  --coverage partial

# Existing checkpoint workflow only.
./experiments/confuse5_single_vs_joint/launch_detached.sh edit \
  --skip-existing

# Real 12-image smoke test. It evaluates and purges successful PNGs, then
# verifies that resume skips the completed work without regeneration.
./experiments/confuse5_single_vs_joint/launch_detached.sh smoke \
  --group dogs \
  --single-target "golden retriever" \
  --rows-per-concept 2 \
  --purge-evaluated-images \
  --skip-existing

# Recommended partial run using the 15 completed checkpoints.
./experiments/confuse5_single_vs_joint/launch_detached.sh all \
  --start-at generate \
  --coverage partial \
  --confirm-image-count 30000 \
  --purge-evaluated-images \
  --skip-existing

# Complete run after obtaining the missing official rows.
./experiments/confuse5_single_vs_joint/launch_detached.sh all \
  --coverage complete \
  --dataset-csv /path/to/official-confuse5-25.csv \
  --confirm-image-count 50000 \
  --purge-evaluated-images \
  --skip-existing

# Generation only deliberately retains images.
./experiments/confuse5_single_vs_joint/launch_detached.sh generate \
  --coverage partial \
  --confirm-image-count 30000 \
  --retain-images \
  --skip-existing

# Evaluate retained images and purge them after durable results are written.
./experiments/confuse5_single_vs_joint/launch_detached.sh evaluate \
  --coverage partial \
  --purge-evaluated-images \
  --skip-existing

# Aggregate existing result shards only.
./experiments/confuse5_single_vs_joint/launch_detached.sh aggregate \
  --coverage partial

# Resume generation/evaluation with bounded disk use.
./experiments/confuse5_single_vs_joint/launch_detached.sh all \
  --start-at generate \
  --coverage partial \
  --confirm-image-count 30000 \
  --purge-evaluated-images \
  --skip-existing

# Resume from evaluation without generating.
./experiments/confuse5_single_vs_joint/launch_detached.sh all \
  --start-at evaluate \
  --coverage partial \
  --purge-evaluated-images \
  --skip-existing

# One partial group: 6,000 total images, peak retained 500.
./experiments/confuse5_single_vs_joint/launch_detached.sh all \
  --start-at generate \
  --coverage partial \
  --groups dogs \
  --confirm-image-count 6000 \
  --purge-evaluated-images \
  --skip-existing
```

`launch_detached.sh` uses the active Conda environment's Python. With no
arguments it preserves the old behavior and runs `edit --skip-existing`.
After it prints its PID, the terminal/VPN may be disconnected. Inspect it with:

```bash
./experiments/confuse5_single_vs_joint/status.sh
```

## Outputs and metrics

```text
outputs/evaluation/
  resolved_plan.json
  run_state.json
  logs/events.jsonl
  images/{original,single,joint}/...       # normally empty after success
  manifests/<job-id>.json
  evaluations/shards/<job-id>.json
  evaluations/per_image.csv
  evaluations/per_class.csv
  aggregates/summary.json
  aggregates/all_groups.csv
  aggregates/groups/<group>.csv
```

For every target, aggregation records Original, corresponding Single, and
Joint target accuracy plus measured differences. Similar-non-target averages
use only the three official preservation roles (one available in partial
mode). The other designated target in a Single run is reported separately as
`sibling_target_preservation`; it is not mixed into the official preservation
average. The tables do not automatically claim success or failure.

MSCOCO CLIP/FID is intentionally not part of this `all` stage. Any later
general-generation evaluation must continue to use the repository-level
evaluation reference registry and its first-1k screening gate.
