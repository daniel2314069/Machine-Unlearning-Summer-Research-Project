# Confuse5 single-vs-joint OCE baseline

This experiment wrapper plans and launches the existing SD OCE implementation
in `../../oce.py`. It does not change the OCE objective or editable modules.

## Scientific configuration

`config.json` contains the official ImageNet-Confuse5 assignments from Table 7
of *Forget Many, Forget Right*. The benchmark has five visually similar groups;
each has two unlearning targets and three similar non-target concepts. The local
ScaPre `eval/datasets/imagenet-15.csv` contains the first three classes of each
group, while Table 7 supplies the complete 25-class assignment. Source
provenance is recorded in the config.

The base model, anchor policy, and OCE hyperparameters follow the current
`oce.py` defaults. These are the OCE baseline settings, not ScaPre method
settings.

A real server edit also requires the existing OCE generic preservation artifact
at repository root `Cg.pt`. It remains intentionally untracked; the runner
checks for it before launching or loading the diffusion model. Dry-run reports
the expected path and whether the artifact is present.

Each group must have this shape:

```json
{
  "id": "authoritative-group-id",
  "concepts": ["all concepts in the group"],
  "targets": ["one or more designated targets"],
  "similar_non_targets": ["one or more non-targets"]
}
```

Every concept must have exactly one role. Targets and similar non-targets must
be non-empty, contained in `concepts`, unique after case/whitespace
normalization, and disjoint.

`shared.anchor_policy.kind` may be `oce_default`, which preserves `oce.py`'s
current object blank-anchor / art `art`-anchor behavior, or `per_target`, with an
`anchors` object mapping every target concept to one anchor string. The runner
resolves the same per-target anchors in single and joint runs. Anchor strings
are visible in dry-run; model-dependent embedding/matrix tensors are computed
only by `oce.py` during a real server run.

The default `group_similar_non_targets` retain policy passes the same three
official similar non-targets to every single and joint edit within a group. An
`explicit_global` policy is also supported when a future protocol needs one
fixed list across every group.
`similar_non_targets` is recorded for later within-group preservation
evaluation and is not silently added to the OCE preservation regularizer.
Each run also records `evaluation_non_target_concepts`: all group concepts not
targeted by that particular run. Thus, in single mode, the other designated
joint targets remain explicit non-targets for evaluation.

## Commands (run from the OCE repository root)

Activate the machine's project environment first. The server uses `MU`; the WSL
desktop uses `py310`. The runner propagates the active interpreter to `oce.py`.

```bash
# Server
conda activate MU

# Or WSL
conda activate py310

# Validate and print/save the full plan; never loads the diffusion model.
python experiments/confuse5_single_vs_joint/run.py \
  --config experiments/confuse5_single_vs_joint/config.json \
  --mode both --dry-run --plan-path experiments/confuse5_single_vs_joint/plan.json

# Server: all single, all joint, or both.
python experiments/confuse5_single_vs_joint/run.py --mode single --skip-completed
python experiments/confuse5_single_vs_joint/run.py --mode joint --skip-completed
python experiments/confuse5_single_vs_joint/run.py --mode both --skip-completed

# Server: one normalized group id only.
python experiments/confuse5_single_vs_joint/run.py \
  --mode both --groups dogs --skip-completed
```

Existing checkpoint/metadata collisions stop execution. `--skip-completed`
resumes by skipping a run only when both its checkpoint and a `status=complete`
metadata file exist. `--overwrite` is the explicit opt-in for replacement.

## Detached server execution

For a server reached through SSH/VPN, activate `MU` and use the committed
launcher. It resolves the active environment's Python, starts the runner with
`nohup`, disconnects stdin, and records its PID, log, and exit status under the
ignored `outputs/` directory.

```bash
conda activate MU
./experiments/confuse5_single_vs_joint/launch_detached.sh
```

Once the launcher prints its PID, the terminal and client VPN may be closed.
Later, reconnect and inspect progress with:

```bash
./experiments/confuse5_single_vs_joint/status.sh
```

A completed experiment reports exit status `SUCCESS (0)`, 15 checkpoints, and
15 `complete` metadata records. If a previous launch was interrupted, relaunch
with explicit replacement of only incomplete outputs; completed runs are still
skipped first:

```bash
./experiments/confuse5_single_vs_joint/launch_detached.sh --overwrite
```

Outputs follow the experiment-local convention:

```text
outputs/<group-id>/single/<target-slug>/weights.safetensors
outputs/<group-id>/single/<target-slug>/metadata.json
outputs/<group-id>/joint/weights.safetensors
outputs/<group-id>/joint/metadata.json
```
