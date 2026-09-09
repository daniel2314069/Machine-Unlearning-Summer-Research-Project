# ScaPre Informax report bundle

This bundle contains the completed report on MI saturation, the difference between the paper and repository implementations, five-seed final alpha-channel controls, the reference diagnostic, and two exploratory squared-cosine follow-up experiments.

## Main files
- `scapre_informax_report.pdf`: compiled final report.
- `scapre_informax_report.tex`: LaTeX source.
- `figures/`: figures used by the report.

## Result artifacts already bundled
- `results/server_summary.md`
- `results/per_seed_metrics.csv`
- `results/per_target_metrics.csv`
- `results/comparison_deltas.csv`

These files are the formal final-alpha channel-control artifacts that were already included in the earlier bundle.

## Supporting notes
- `notes/ScaPre_Informax_實驗總結.md`: matched-retain / superclass reference experiment summary.
- `notes/MI_channel_weighting_summary.md`: sample-size and concept-max diagnostics collected during the study.
- `notes/MI_implementation_audit.md`: paper-vs-repository Informax implementation audit.
- `notes/geometric_cos2_experiments_summary.md`: archived working notes from the geometric follow-up stage.

The full raw artifacts for the squared-cosine experiments are not duplicated in this bundle; they remain in the experiment repository paths recorded in the supporting note. The report uses only the validated aggregate results needed for the final analysis.

Matched-retain is presented only as a diagnostic. The squared-cosine experiments are also presented as exploratory follow-up tests, not as a successful replacement method. Neither reported variant passed the continuation conditions for COCO evaluation.
