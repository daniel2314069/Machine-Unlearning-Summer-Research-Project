# AFR balls smoke result import

This directory is the compact, Git-tracked import of the GPU result bundle
`afr_balls_smoke_results.tar.gz`.

- Bundle SHA-256: `2fe59c3751d91300be2bcd7f382ec377ceb0f30a5808915264c24cb97e41c26f`
- Implementation commit: `51170e7082520324e9affe8bd944e7a05addc064`
- Matrix classification: `AFR-GO`
- Final image-smoke classification: `AFR-I0`
- Matrix cases: 48 group-layer cases, 144 C/F/G rows
- Evaluator shards: 21/21
- Evaluated new images: 1,800
- LPIPS pairs: 48
- Checkpoint metadata records: 3

The import retains the report and compact machine-readable evidence: matrix
CSV/gate, resolved smoke plan, summary, LPIPS records, all evaluator shards,
and checkpoint metadata. PNGs and checkpoint weight files remain excluded.
Consequently their recorded SHA-256 values and all numerical summaries can be
audited, but the large bytes themselves cannot be re-hashed from this import.
The matrix CSV's CRLF record terminators were normalized to LF for the tracked
text artifact; its parsed rows and values are unchanged.

Cross-file validation performed before import confirmed:

- all 48 matrix keys are unique and cover dogs/fruits/balls x 16 layers;
- F/G numerical guarantees and the G-versus-F preservation gate recompute from
  the CSV without discrepancies;
- all 21 planned jobs have exactly one completed evaluator shard;
- all C/F/G counterparts use identical ordered cases, prompts, and seeds;
- the 1,800 per-image records and recorded image hashes are unique;
- summary accuracies, probabilities, macros, and LPIPS means recompute exactly
  from the raw evaluator and LPIPS records;
- frozen protocol hashes and AFR `runner.py`/`core.py` hashes match the tracked
  repository files at the implementation commit.

The run recorded `git_dirty=true`. A follow-up GPU-server `git status --short`
showed only the unrelated untracked file
`exact_orthogonal_control_results.patch`, while `git diff` was empty for
`run.py`, `pipeline.py`, `solver_audit.py`, and `qualified_primary.py`. No
tracked experiment-code difference was identified.
