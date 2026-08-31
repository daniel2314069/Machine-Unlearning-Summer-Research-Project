# Budget-matched direct-cos2 formal results

This directory contains the integrity-validated lightweight artifacts from the
five-seed ImageNet-Confuse5 formal experiment. The complete downloaded package
is retained outside Git under `.local_artifacts/scapre_informax/`; generated
images and regenerable checkpoints were excluded from the archive and remain
on the server.

The preregistered continuation decision is **stop**. Budget-matched direct cos2
improved mean Preserve by `+0.2667` and mean Overall by `+0.2139`, but mean
Unlearn worsened by `+0.1000`, and Overall was favorable in only `3/5` seeds.
It therefore fails two automatic continuation conditions. No COCO safeguard
was run, and the contract rules out a fourth cos2 variant.

Key files:

- `validation_report.md`: independent validation and scientific interpretation.
- `server_summary.md` and `server_validation_report.md`: server-generated reports.
- `per_seed_metrics.csv`, `per_group_metrics.csv`, and
  `per_target_metrics.csv`: complete formal breakdowns.
- `qualification_integrity_report.json` and
  `per_layer_concept_budget_matching.csv`: qualification and all 320
  contribution-norm matches.
- `per_layer_concept_weight_diagnostics.csv`,
  `per_layer_concept_correlations.csv`, and `per_matrix_edit_strength.csv`:
  descriptive intervention diagnostics.
- `seed_audits/`: all five RNG, source-substitution, and checkpoint isolation
  summaries.
- `retrieval_validation.json`: archive validation and independent recalculation.
- `result_manifest.json`: SHA-256 manifest for this curated result directory.
