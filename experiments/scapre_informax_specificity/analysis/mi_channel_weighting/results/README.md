# Formal result files

These files come from completed server run `formal_20260825T092351Z` at code
commit `04181a15e1645c84941a2f6d44d02c731711ac02`.

Two tables are gzip-compressed without loss because their uncompressed sizes
are 252 MB and 42 MB:

```bash
gzip -dk max_mi_activation_summary.csv.gz
gzip -dk concept_count_repo_formula.csv.gz
```

The decompressed SHA-256 values match the server manifest:

- `max_mi_activation_summary.csv`:
  `e13a13d0b05f10f99c6e9d376fc7f9ac4936ea1c126e6bc5958366e93cda2312`
- `concept_count_repo_formula.csv`:
  `08fb51fa1c7283f6da2fcfde0555a7b42b998b144ecd2326e31c62db70e76bbf`

The checked-in sample-size CSVs differ bytewise from their server originals
only because `finalize_retrieved_results.rb` rewrites the
`enumerated_no_tie_unique_count` presentation field after tolerance-coalescing
float32-symmetric enumeration values. MI observations and all empirical
statistics are unchanged. `server_integrity_report.json` retains the original
server hashes; `integrity_report.json` records the normalized deliverables.

The server-generated repo-alpha PNG used a 0–1 x-axis and visually collapsed
the observed 0–0.012 distribution. The checked-in SVG replaces only that
presentation with the actual mean/median/upper-quantile series. The runner is
also corrected for future runs.
