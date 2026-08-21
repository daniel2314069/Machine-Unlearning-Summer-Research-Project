# ScaPre Informax Specificity: Edit-Seed Robustness

This directory tests whether the previous matched-retain preservation gain is
stable over the five fixed Informax seeds `20260820` through `20260824`.
No method or evaluation setting is tuned. Seed `20260820` is imported from the
verified previous formal run; only the other four seeds generate new images.

Read `AUDIT.md` before launching. In particular, changing the legacy
`--edit-seed` directly would also change entropy sampling. The dedicated runner
keeps all non-Informax randomness on the legacy `20260820` stream and changes
only the noise returned to the 5+5 Informax pseudo-samples.

## Fixed design

- variants: `official`, `matched_retain`;
- Informax seeds: `20260820`, `20260821`, `20260822`, `20260823`, `20260824`;
- fixed global/non-Informax edit seed: `20260820`;
- matched negatives: unchanged three same-group retains with fixed 2/2/1 order;
- one shared 25-concept protocol, 120 images per concept;
- one shared generation prompt/seed set across every seed and variant;
- 3,000 scores per variant/seed, yielding 30,000 final records;
- only 24,000 new images are generated.

The standard deviation in the final report is the sample standard deviation
with an `n-1` denominator. The three-way judgment is frozen in `config.json`.

## Server prerequisites

The seed-robustness lifecycle does not require `jq` or any newly installed
system command. JSON parsing and server-side SHA-256 use only the standard
library of the already active `MU` Python. Do not install `jq` for this
experiment.

Because the preceding formal run completed on the same server, its assets and
runtime are already present; do not rerun setup merely for this follow-up.
Activate the existing environment and launch the smoke test after pulling.
Only a genuinely fresh server without the parent setup marker or assets needs:

```bash
conda activate MU

experiments/scapre_informax_specificity/setup_server.sh
```

The seed launcher itself verifies active `MU`, CUDA, a clean `main`, performs
`git pull --ff-only origin main`, checks that commit `e157140...` is an
ancestor, and validates all controlled source hashes.

The previous formal worker wrote its generated summary into a tracked path. If
the first server pull reports only that known runtime modification, preserve it
recoverably before pulling:

```bash
git status --short
git stash push -m 'runtime summary from ScaPre seed 20260820' -- \
  experiments/scapre_informax_specificity/results/summary.md
git pull --ff-only origin main
```

Do not stash or discard any additional unexplained dirty path. Stop and inspect
it first. The previous summary and raw evidence are already retained in the
verified archive, but the stash provides an additional reversible copy.

For seed `20260820`, it first uses the exact completed previous run
`formal_20260820T163033Z`. If that directory is absent, it can safely extract
the pinned archive from `/home/tslin/Documents/jupyter_data/anLi/tmp`. It never
regenerates the legacy images.

## Required smoke test

Smoke runs only the dogs group, seed `20260821`, and two images per concept. It
tests the Informax-only RNG interception, both edit modes, evaluator,
aggregation, automatic packaging, and image cleanup. It is not a scientific
result.

```bash
experiments/scapre_informax_specificity/seed_robustness/run_server.sh smoke
experiments/scapre_informax_specificity/seed_robustness/status_server.sh
```

After the launch health check succeeds, the terminal, SSH connection, and local
computer may be closed safely.

## Formal robustness experiment

After smoke reports `completed`, exit code `0`, an archive, and image cleanup
`passed`:

```bash
experiments/scapre_informax_specificity/seed_robustness/run_server.sh formal
experiments/scapre_informax_specificity/seed_robustness/status_server.sh
```

Before detaching the formal worker, the launcher runs a lightweight synchronous
formal preflight over the frozen protocol, all 6,000 legacy score rows, prior
evaluator fingerprints, model/classifier asset provenance, and pinned source
hashes. It uses only the active `MU` Python standard library, does not load a
model, does not generate images, and does not download anything. A failed
preflight prevents the expensive formal run from starting.

To resume the same failed run after resolving an external problem:

```bash
experiments/scapre_informax_specificity/seed_robustness/run_server.sh \
  formal RUN_ID --resume
```

The formal run always completes all four new fixed seeds. It does not stop based
on intermediate results.

## Automatic packaging and image cleanup

The detached worker automatically packages results after all calculations and
integrity checks pass. The archive is written to:

```text
/home/tslin/Documents/jupyter_data/anLi/tmp
```

Only after that archive passes a full tar read and SHA-256 verification does
the worker delete generated PNGs. Formal cleanup deletes:

- all 24,000 newly generated robustness images;
- remaining images from previous runs `smoke_20260820T162319Z` and
  `formal_20260820T163033Z`, when those exact completed directories exist.

Scores, summaries, diagnostics, checkpoints, source snapshots, archives, model
weights, and all non-image results remain on the server. `cleanup_manifest.json`
records exact paths, counts, and released bytes. A cleanup failure makes the
overall run fail rather than silently claiming success.

`package_results.sh RUN_DIR` is idempotent and can be rerun after completion; it
reuses and revalidates the existing archive instead of creating duplicates.

## Download to the Mac

Use the exact archive path and SHA-256 printed by `status_server.sh`:

```bash
experiments/scapre_informax_specificity/seed_robustness/download_results.sh \
  'tslin@ACTUAL_SERVER_HOST' \
  '/home/tslin/Documents/jupyter_data/anLi/tmp/ACTUAL_ARCHIVE.tar.gz' \
  'ACTUAL_64_CHARACTER_SHA256'
```

The script downloads the archive and its cleanup-manifest sidecar to
`~/Downloads/`, refuses to overwrite either file, verifies SHA-256, and checks
that cleanup was recorded against that exact archive. It never deletes the
server archive.

## Required result outputs

The completed run archive contains:

```text
results/summary.md
results/per_seed.csv
results/per_group_seed.csv
results/per_concept_seed.csv
results/aggregate_across_seeds.csv
results/per_group_robustness.csv
results/per_retain_robustness.csv
results/informax_seed_diagnostics.csv
reproducibility/integrity_report.json
reproducibility/prior_seed_validation.json
```

It also includes all 30,000 raw score records, evaluator fingerprints,
per-seed controlled-ablation/RNG audits, configs, commands, logs, and the exact
source snapshot. Generated images, checkpoints, caches, weights, and raw tensor
diagnostics are intentionally excluded from the archive.
