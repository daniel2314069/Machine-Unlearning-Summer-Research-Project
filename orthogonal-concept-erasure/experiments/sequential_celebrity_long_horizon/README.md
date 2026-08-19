# Long-horizon sequential OCE celebrity experiment

This directory implements the final qualification of the sequential direction. It performs ten legal repository E10 edits, each containing ten celebrities, instead of inventing a singleton configuration. The fixed 100-person repository list is used in its original order and exact reverse. Each order has an independent baseline trajectory and retain-full-history trajectory, all starting from the same frozen SD 1.4 snapshot.

The scientific question and outputs remain concept-level. A celebrity's introduction step is the 10-person batch in which it is first erased, and `current_position_age` is the number of later batch edits. The only algorithmic modification is that retain-history passes every earlier batch target to the repository retain input; because each call receives the current pre-edit checkpoint, this preserves the history concepts' current erased representation.

## Fixed protocol

- Order A: the exact `E100_LIST` order in `generate_celeb.py`.
- Order B: the exact reverse of Order A; no random or optimized third order.
- Conditions: `baseline` and `retain_history`.
- Steps: 10 batches × 10 celebrities, producing 40 sequential checkpoints total.
- Reference: one repository-configuration joint-100 checkpoint.
- No protocol-identical local joint result is assumed; the runner builds this reference once, then resumes/reuses it only by frozen fingerprint and checkpoint hash.
- Evaluation templates: the five exact repository celebrity templates frozen in `config.json`.
- Fixed retain evaluation: all 100 `PRESERVE_LIST` celebrities, five templates, seed 42, at every checkpoint (500 images).
- Immediate success rule: raw GCD accuracy at introduction must be at most 10%.
- Paper checkpoints: 10, 50, and 100 erased celebrities, each with 500 target and 500 retain images.
- GCD accuracy follows the repository evaluator's denominator: correctly detected celebrity among images where a face was detected. Raw no-face predictions and all top-5 predictions are retained.
- Qualitative positions are fixed before generation at 1, 10, 25, 50, 75, and 100 with seeds 42 and 43. Saved states are Original, introduction, introduction + 1 batch, introduction + 3 batches, and final, where distinct.

The exact 100 erasure celebrities and exact 100 fixed retain celebrities are present, in full and in order, in [`config.json`](config.json). Preflight independently parses the two current repository lists and refuses to proceed if either differs.

Repository-versus-paper differences that affect this protocol are recorded without silently reconciling them:

- The paper describes the celebrity anchor as `celebrity`; the current repository E10 configuration supplies `person`, `woman`, and `man`. This experiment follows the repository.
- The paper retain-list spelling `Melanie Grifftih` is `Melanie Griffith` in the repository. This experiment follows the repository string.

## Fixed sampling profiles and image totals

The 200-image benchmark is non-scoring and runs before the sample profile is locked. It measures the active Lightning GPU using the actual 512×512, PNDM-50, CFG 7.5 generation path. Its four fixed 50-image partitions test batch sizes 8, 16, 24, and 32; OOM trials are recorded, and the fastest successful batch is frozen into the active protocol before any formal sample is generated. The same 200 images also pass through the exact GCD path, so profile selection budgets measured generation and evaluator time rather than assuming scoring is free. GCD uses the pinned official detector, preprocessing, labels, recognizer, and top-5 outputs; only independent CPU face detections are scheduled through an ordered spawned-process pool, with the actual worker count frozen in the active manifest. Benchmark predictions and validation hashes remain, while benchmark PNGs are deleted only after prediction validation. Sample keys, prompts, and seeds do not depend on either execution choice. The runner then reserves 20% of the entered credits for editing, model loading, restarts, and variance, and chooses the largest profile whose measured end-to-end time fits the other 80%.

| Profile | Trajectory samples per celebrity/checkpoint | Formal GCD predictions | Extra qualitative/Original images | Core run | Including benchmark |
|---|---:|---:|---:|---:|---:|
| `profile_5` | 5 templates × 1 seed = 5 | 34,800 | 100 | 34,900 | 35,100 |
| `profile_10` | 5 templates × 2 seeds = 10 | 45,800 | 20 | 45,820 | 46,020 |

Both profiles still run the complete 500-image target and 500-image retain protocol at 10/50/100 and joint-100 at 100. Paper-scale target samples reproduce the repository's seed-42 generator stream and retain its sample indices; they are not approximated with seeds 42, 43, and so on. The profiles differ only in the paired per-step trajectory sampling. Concept lists, conditions, steps, milestones, metrics, and qualitative selection do not change.

For batch step `t`, each sequential cell has 500 retain rows. Target rows are `50 × t`, plus 450 official-stream rows at step 1 and 250 at step 5, for `profile_5`; they are `100 × t` plus the same 450/250 supplements for `profile_10`. The milestone's `paper_sample` subset is exactly 500 targets even when the larger trajectory profile has additional paired rows. Joint-100 has 500 target and 500 retain rows.

Do not estimate from a GPU name or a stale price table. On Lightning, enter the **remaining credit balance shown immediately before the run** and the **current credits/hour shown for the selected GPU**. For example, if the UI shows 12.4 credits remaining and 0.48 credits/hour:

```bash
./run_sequential_long_horizon.sh --benchmark \
  --remaining-credits 12.4 \
  --gpu-rate 0.48 \
  --artifact-root "$ARTIFACT_ROOT" \
  --gcd-project-root "$GCD_PROJECT_ROOT"
```

If neither fixed profile fits with the reserve, the runner stops before formal generation. It does not shrink samples after seeing results. The safer choice on free credits is whichever available GPU gives the largest benchmarked images-per-credit, not necessarily the largest raw GPU.

## Lightning AI setup and launch

The Mac is code-only. Perform every command in this section on the Lightning instance after the repository, OCE `Cg.pt`, GCD project/resources, and dependencies are present. Activate the project environment first; the shell runner rejects Conda `base` and does not hardcode an environment name.

This deployed Lightning Studio exposes exactly one platform-managed non-base Conda environment, `cloudspace`, and its `/commands/conda` wrapper refuses creation of a second environment. On this Studio, source `lightning_deployment.env`; all commands then resolve to `/home/zeus/miniconda3/envs/cloudspace/bin/python`. Other GPU servers continue to use the project `MU` environment.

```bash
cd /absolute/path/to/Machine-Unlearning-Summer-Research-Project/orthogonal-concept-erasure/experiments/sequential_celebrity_long_horizon
conda activate MU

export ARTIFACT_ROOT=/absolute/persistent/lightning/path/sequential_oce_celebrity_long_horizon_v1
export GCD_PROJECT_ROOT=/absolute/path/to/giphy-celebrity-detector
```

Use persistent Lightning storage for `ARTIFACT_ROOT`. The output directory defaults to this experiment directory's `outputs/sequential_oce_celebrity_long_horizon_v1`; it must also live on storage that survives a Lightning restart. If the repository workspace is not persistent, pass a persistent `--output-dir` to every command.

First inspect and freeze the protocol on a CPU session if desired. `--allow-downloads` permits only the requested model snapshot resolution; omit it when the snapshot is already cached.

```bash
./run_sequential_long_horizon.sh --plan

./run_sequential_long_horizon.sh --preflight \
  --artifact-root "$ARTIFACT_ROOT" \
  --gcd-project-root "$GCD_PROJECT_ROOT" \
  --allow-downloads
```

Then switch to the intended GPU, reactivate the same environment, read the live credit balance and GPU rate from Lightning, and run the benchmark exactly once:

```bash
conda activate MU

./run_sequential_long_horizon.sh --benchmark \
  --remaining-credits <LIVE_REMAINING_CREDITS> \
  --gpu-rate <LIVE_CREDITS_PER_HOUR> \
  --artifact-root "$ARTIFACT_ROOT" \
  --gcd-project-root "$GCD_PROJECT_ROOT"
```

Review `budget_selection.json`. If a profile was locked, launch the detached worker:

```bash
./run_sequential_long_horizon.sh --start
./run_sequential_long_horizon.sh --status
```

The terminal and SSH connection may be closed after `--start`; `nohup`, PID tracking, and an append-only log keep the process alive while the Lightning VM itself remains running. Lightning can still stop the VM because of session/runtime limits. After any platform stop, restart the instance, mount the same persistent storage, activate the environment, and issue the same `--start` command. Completed checkpoints and validated GCD cells are reused. Partially generated cells retain valid atomically-written images and continue with missing images only.

For the final H100 deployment, start the independent hard-deadline watchdog immediately after GPU verification and before the benchmark controller. It is a separate detached process, freezes the remaining deadline at 20,700 seconds (5 hours 45 minutes, so the earlier provisioning attempts plus the final run remain within the user's six-hour total cap), records its PID/deadline/log, covers both benchmark and formal execution, flushes files, and calls `lightning studio stop` for normal completion, a controller/runner failure, or the deadline. Profile selection uses at most 85% of this wall-clock window for measured generation plus GCD, leaving 15% for editing, model loads, aggregation, packaging, variance, and shutdown. `lightning_h100_controller.sh` runs the fixed benchmark detached and starts the separate formal runner only after profile/config locking succeeds. This prevents a benchmark or runner crash from leaving the Studio burning GPU credits.

If a non-default output was used, repeat it on start and status:

```bash
./run_sequential_long_horizon.sh --start --output-dir /absolute/persistent/output/path
./run_sequential_long_horizon.sh --status --output-dir /absolute/persistent/output/path
```

## Resume, validation, and cleanup guarantees

For each evaluation cell, the order is:

```text
generate -> GCD evaluate -> validate ordered prediction keys/counts
         -> save raw predictions, metrics, hashes -> delete formal images
```

No cell's formal images are deleted when GCD fails or predictions are incomplete. Checkpoint manifests freeze parent hashes, target batches, explicit retains, aligned guides, and the active protocol fingerprint. The standalone Ruby audit recomputes aggregate accuracies from raw prediction CSVs and verifies schedules, reverse order, parent chains, retain history, paired samples, row counts, hashes, and cleanup markers.

The normal successful run automatically performs editing, generation, GCD scoring, aggregation, contact sheets, archive creation, independent audit, cleanup, and report generation. A sequential GCD cell is completed and cleaned immediately after its checkpoint, before the next batch edit begins. Ruby must be available for the final standalone audit. Manual audit is available as:

```bash
./run_sequential_long_horizon.sh --audit
```

Final core artifacts include:

- `trajectory_per_concept.csv`
- `step_summary.csv`
- `paper_checkpoint_results.csv`
- `raw/all_gcd_predictions.csv`
- `independent_audit.json`
- `REPORT.md`
- `final_validation.json`

The qualitative archive is written to:

```text
<ARTIFACT_ROOT>/qualitative_samples.tar.gz
```

After the server path and SSH alias are known, copy it locally with:

```bash
scp <ssh-user-or-alias>:<ARTIFACT_ROOT>/qualitative_samples.tar.gz .
```

## COCO phase

MS-COCO CLIP/FID is deliberately deferred from the free-credit core run. The core command never generates COCO images. `--continue-coco` is currently a safety gate only: it validates completion of the core run and checks that the repository registry has portable, complete first-1k and first-10k Original reference artifacts. It refuses generation even when references exist until a separate budget revision explicitly implements and reviews the costly edited-model phase. It never regenerates Original SD baselines.

## No post-hoc rescue

If the complete 100-celebrity, ten-batch baseline does not show clear cumulative reappearance among celebrities that passed the fixed introduction threshold, the sequential/long-term-retention direction is a negative result. Do not select a favorable celebrity subset, add concepts, change orders, invent geometry diagnostics, or reinterpret tiny differences. If baseline has no meaningful failure, retain-history must not be declared a necessary improvement.

## Completed result

The completed Lightning run is stored under
[`results/sequential_oce_celebrity_long_horizon_v1`](results/sequential_oce_celebrity_long_horizon_v1/).
Start with its [`CHATGPT_HANDOFF.md`](results/sequential_oce_celebrity_long_horizon_v1/CHATGPT_HANDOFF.md)
for the evidence bundle and interpretation constraints. Model checkpoints and
generated images are intentionally excluded from the repository result.
