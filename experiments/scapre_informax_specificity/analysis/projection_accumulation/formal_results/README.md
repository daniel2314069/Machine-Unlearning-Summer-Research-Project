# Projection accumulation formal results

This directory contains the integrity-validated lightweight artifacts from the
five-seed ImageNet-Confuse5 formal experiment. The full downloaded package is
retained outside Git under `.local_artifacts/scapre_informax/`.

The preregistered decision is **stop**. Qualification and scientific isolation
passed, but the treatment worsened mean Unlearn Accuracy by `+0.5000` points.
It therefore failed the required `Delta Unlearn < 0` condition. The failure is
also substantively heterogeneous: tiger-cat Unlearn Accuracy worsened by
`+40.1667` points on average and was unfavorable in all five seeds.

No COCO safeguard was run, and this modification should not advance to COCO or
be tuned into a second projection variant.

Key files:

- `validation_report.md`: independently checked result interpretation.
- `server_summary.md`: server-generated aggregate report.
- `per_seed_metrics.csv`, `per_group_metrics.csv`, and
  `per_target_metrics.csv`: complete formal breakdowns.
- `qualification_integrity_report.json`: score/alpha distributions and seed
  `20260820` qualification evidence.
- `seed_audits/`: all five RNG, isolation, source-substitution, and checkpoint
  audit summaries.
- `retrieval_validation.json`: archive and independent recalculation record.

