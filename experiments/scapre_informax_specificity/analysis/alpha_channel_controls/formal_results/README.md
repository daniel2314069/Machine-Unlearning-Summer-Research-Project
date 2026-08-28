# Formal alpha-channel control results

The five-seed formal run completed successfully at commit
`a393c649f54b532fdcd85fc3fe7ff3e30a1a5f0d`. The primary reader-facing
deliverable is `report.html`; `server_summary.md` is the server-generated
summary, while the CSV and JSON files retain the auditable aggregate evidence.

The full extracted archive, including 20 raw score files, 20 Torch diagnostics,
per-edit audits, logs, and source snapshots, is retained outside Git under:

```text
.local_artifacts/scapre_informax/scapre_informax_alpha_channel_controls_formal_20260825T201035Z_20260827T173500Z/
```

The CSV copies in this directory use LF line endings for Git and web-tool
compatibility. The server originals remain byte-for-byte unchanged in the
local artifact directory. Therefore, hashes in `result_manifest.json` refer to
the extracted server originals, not the normalized Git copies.

The 45,000-row generated-image manifest and the `.pt` diagnostics remain only
in the extracted archive. Their completeness and hashes are recorded by
`retrieval_validation.json` and `integrity_report.json` without adding large,
regenerable evidence to Git.

The local download did not include the archive cleanup sidecar. Calculation,
aggregation, packaging, and archive integrity are verified; server-side PNG
cleanup remains the only unverified lifecycle item.
