# Direct-cos2 accumulation formal results

This directory contains the integrity-validated lightweight artifacts from the
five-seed ImageNet-Confuse5 formal experiment. The complete downloaded package
is retained outside Git under `.local_artifacts/scapre_informax/`; generated
images and regenerable checkpoints were excluded from the server archive and
remain on the server.

The preregistered continuation decision is **stop**. All three five-seed mean
deltas have the favorable sign, but the changes are very small and Overall was
favorable in only `2/5` seeds, failing the required `4/5`. The cat group also
showed repeated Preserve/Overall degradation. No COCO safeguard was run, and
this exploratory variant should not advance to COCO.

Key files:

- `validation_report.md`: independent validation and scientific interpretation.
- `server_summary.md`: server-generated aggregate report.
- `per_seed_metrics.csv`, `per_group_metrics.csv`, and
  `per_target_metrics.csv`: complete formal breakdowns.
- `qualification_integrity_report.json`: direct-cos2 qualification evidence.
- `per_layer_concept_weight_diagnostics.csv` and
  `per_matrix_edit_strength.csv`: intervention-strength diagnostics.
- `pre_analysis/`: frozen descriptive comparison of raw cos2 and the V1
  transformed alpha.
- `seed_audits/`: all five RNG, isolation, source-substitution, and checkpoint
  audit summaries.
- `retrieval_validation.json`: archive and independent recalculation record.
- `result_manifest.json`: SHA-256 manifest for this curated result directory.
