# Isolated codex_diverse changeset

This experiment is contained within `experiments/concept_description_clustering` and writes only to a new caller-selected output directory, normally `outputs/codex_diverse_overnight`.

It does not import or modify the OCE algorithm, does not edit W0, and does not write to `outputs/full_to_v`. The generated corpus has one provenance label only: `codex_diverse`.

The condition-specific implementation consists of:

- `concept_clustering/codex_diverse.py`: deterministic diverse corpus construction and out-of-fold TF-IDF-hard selection;
- `concept_clustering/overnight_runner.py`: deadline-aware resumable orchestration and reporting;
- `configs/codex_diverse_*.json`: pool, formal 4×50/4×100, and smoke configurations;
- `scripts/codex_diverse_job.sh`: detached launch, resume, status, and cached clustering entry points;
- `tests/test_codex_diverse.py`: balance, provenance, and OOF-output checks.

The shared generation CLI gained only opt-in deadline/stop-file handling used by the new runner. With no deadline environment variables, existing commands retain their previous behavior.
