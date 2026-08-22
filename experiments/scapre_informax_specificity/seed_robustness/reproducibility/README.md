# Reproducibility outputs

The completed formal run is `formal_20260821T081723Z`. This directory contains
its lightweight run/protocol manifests, integrity and prior-seed validation,
audited post-hoc finalization record, all ten evaluator fingerprints, verified
archive checksum, and image-cleanup manifest.

The verified archive is:

```text
scapre_informax_seed_robustness_formal_20260821T081723Z_20260822T092030Z.tar.gz
SHA-256: df0874fea7c0998bbaf52782c763025c4ce7968134e8334e0688adec95453708
```

It contains the 30,000 raw score records, per-seed audits, logs, commands, and
source snapshots. Those raw records and excluded tensor/image/checkpoint
artifacts are not duplicated in Git. `posthoc_finalization.json` proves the
final aggregation reused the existing edits and 24,000 newly generated images;
it did not rerun editing or generation. `cleanup_manifest.json` records deletion
of 30,020 PNGs only after archive verification.
