# Files for GPT web analysis

This folder contains only lightweight, human-readable smoke evidence. Upload
the files individually. A useful order is:

1. `analysis_notes.md` - retrieval checks, compact findings, and the strict
   smoke-only decision boundary.
2. `implementation_audit.md` - exact intervention point and experimental design.
3. `integrity_report.json` and `controlled_ablation_check.json` - proof that the
   controls differ only in final alpha/B and that all gates passed.
4. `per_seed_metrics.csv` and `per_target_metrics.csv` - coarse established
   metrics and denominators.
5. `smoke_scores.csv` - all 60 classifier outputs, useful for inspecting the
   prediction-label changes hidden by tied accuracy.
6. `alpha_matrix_summary.csv` - 32 matrices x six variants, including alpha
   summaries, shuffle checks, `trace(B)`, and `||B||_F`.

Suggested prompt:

> Review this as an implementation smoke test, not a formal experiment. Verify
> the controlled-design evidence, explain why identical 10-image accuracy does
> not imply identical model behavior, inspect classifier-label and matrix-level
> patterns, and assess what the smoke does and does not tell us about whether a
> five-seed formal run is worth its GPU cost. Do not infer statistical or
> scientific equivalence from the tied smoke metrics.

Do not upload the `.pt` diagnostics from `.local_artifacts`: they require the
project's Torch environment, add little value to a web review, and are already
represented by the audited CSV/JSON summaries. The server log, provenance source
copies, and 3 MB archive are also unnecessary unless debugging a failure; this
smoke passed.
