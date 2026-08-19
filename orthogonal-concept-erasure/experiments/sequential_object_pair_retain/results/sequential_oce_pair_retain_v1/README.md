# Result artifact inventory

Primary report: [`REPORT.md`](REPORT.md).

The completed GPU run used source commit `7da1789974f4076fef42467b77b7d8cdd9e57257` and protocol fingerprint `abc9e13f90ecfce9c03458882654e63b7c4d16b3a9400bf7f02bce6e7adbeae8`.

## Included evidence

- `summary.csv`: requested per-order target and remaining-eight comparisons
- `per_class_results.csv`: all 310 class-level evaluation cells
- `stage1_paper_metrics.csv`: Stage-1 Acc_e, Acc_s, and H_o
- `order_effects.csv`: direct final-accuracy first-position minus second-position comparisons
- `recomputed_order_summary.csv`: independently reconstructed order summary
- `independent_audit.json`: raw-prediction and pipeline validation receipt
- `audit_results.rb`: standalone audit using only Ruby standard libraries
- `run_manifest.json`, `run_state.json`, `final_validation.json`, `qualitative_manifest.json`
- `pair_schedule.csv`, `formal_seeds.csv`, and complete `run.log`
- `artifact.json`: canonical Data Analytics report artifact
- `build_report_artifact.rb`: deterministic artifact builder

The 23 MB `raw/all_predictions.csv` is intentionally not committed to avoid unnecessary Git history growth. Its SHA-256 is recorded in `independent_audit.json`; it remains in the server output and in the downloaded results archive.

`run.log` is append-only and therefore retains the initial `cifar_class_text_template` failure from before commit `7da1789`. The later run resumed successfully; the controlling completion evidence is the final validation line, `run_state.json`/`run_manifest.json` status, and `final_validation.json`.

## Revalidation

Given an unpacked formal result directory:

```bash
ruby audit_results.rb /path/to/outputs/sequential_oce_pair_retain_v1 .
```

This checks row/cell completeness, seeds, prompts, labels, probabilities, saved aggregate reproduction, per-cell hashes, cleanup markers, and shared Stage-1 parents.

`artifact.json` is ready for the Data Analytics portable HTML builder. HTML was not generated on the local Mac because no Node executable is installed; `REPORT.md` is the readable report fallback.
