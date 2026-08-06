# OCE E50 unique-anchor stress test

This directory contains one resumable runner for the current repository
implementation stress test. It compares a true one-column `celebrity` anchor
against the last 50 entries of the repository E100 erasure list.

The completed Table 11-style technical report is `table11_report.html`.
Its reviewed source rows and portable report definition are
`table11_report_data.json` and `table11_report_artifact.json`.

All Python commands must use the project `py310` Conda environment. GPU stages
also need the CUDA-wheel library path used by the repository's existing
`experiments/correspondence_diagnostic/scripts/run_py310.sh` wrapper:

```bash
experiments/correspondence_diagnostic/scripts/run_py310.sh \
  e50_unique_anchor_stress/run_experiment.py preflight
experiments/correspondence_diagnostic/scripts/run_py310.sh \
  e50_unique_anchor_stress/run_experiment.py weights
```

`run_all.sh` is resumable. It stops after the required first-1k COCO screening
unless `--continue-to-10k` is explicitly supplied. GCD is external to this
repository, so pass its checkout via `--gcd-project-root` (or set
`GCD_PROJECT_ROOT`).

For the current unique-anchor first-10k run, `watch_unique10k_then_gcd.sh`
provides a zero-GPU waiting path. It checks only the unique first-10k metrics
file once per minute. After that metrics file is complete and the COCO process
has exited, it downloads the pinned official GIPHY detector and model archive,
installs its Python-3.10 compatibility runtime under this experiment directory,
and evaluates all 3,000 already-generated celebrity images. It never calls
`finalize` and never deletes COCO or celebrity images.

The automatic GCD outputs are:

```text
gcd_metrics/metrics.json
gcd_metrics/predictions.csv
gcd_metrics/per_celebrity_accuracy.csv
gcd_metrics/summary.md
gcd_metrics/setup_manifest.json
gcd_metrics/automation_state.json
logs/gcd_after_unique10k.log
```

The explicit staged continuation sequence is:

```bash
# Completes only missing celebrity images.
experiments/correspondence_diagnostic/scripts/run_py310.sh \
  e50_unique_anchor_stress/run_experiment.py celebrity --batch-size 4

# Requires the external GIPHY Celebrity Detector checkout and its .env/resources.
experiments/correspondence_diagnostic/scripts/run_py310.sh \
  e50_unique_anchor_stress/run_experiment.py gcd \
  --gcd-project-root /absolute/path/to/giphy-celebrity-detector

# Required screening gate.
experiments/correspondence_diagnostic/scripts/run_py310.sh \
  e50_unique_anchor_stress/run_experiment.py coco \
  --coco-count 1000 --batch-size 4

# Run only after reviewing the first-1k metrics.
experiments/correspondence_diagnostic/scripts/run_py310.sh \
  e50_unique_anchor_stress/run_experiment.py coco \
  --coco-count 10000 --batch-size 4 --continue-to-10k

# Validates every acceptance condition before deleting exactly the 20k COCO PNGs.
experiments/correspondence_diagnostic/scripts/run_py310.sh \
  e50_unique_anchor_stress/run_experiment.py finalize
```

Cleanup is deliberately restricted to the two edited-model COCO image
directories. It runs only after checkpoints, smoke, celebrity/GCD, and COCO
first-10k metrics all pass final validation.
