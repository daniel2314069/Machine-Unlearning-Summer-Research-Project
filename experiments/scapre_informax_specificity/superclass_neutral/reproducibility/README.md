# Reproducibility outputs

The completed formal run is `formal_20260822T175116Z`, executed from commit
`4f45257fba7160ea6b8ef3ba9bb4115409b35a8a`. This directory records its actual
configs, run/protocol manifests, formal preflight, baseline-reuse audit,
integrity report, verified archive checksum, and image-cleanup manifest.

The verified archive is:

```text
scapre_informax_superclass_neutral_formal_20260822T175116Z_20260823T090021Z.tar.gz
SHA-256: 5377f92ec2d154d94127bbebcafe3f38f3a3cac6e0b6e91ea29e543555037e0f
```

It contains all 45,000 raw score records, per-seed audits, commands, source
snapshots, the server log, 90 qualitative images, and 30 comparison panels.
Raw scores and server logs are retained in the archive rather than duplicated
in Git. `cleanup_manifest.json` confirms that exactly 15,000 full-evaluation
PNGs were deleted only after archive verification, while all qualitative files
were preserved.
