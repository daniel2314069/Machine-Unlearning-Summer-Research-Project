# Report source notes

## Report contract

- Audience: technical.
- Delivery mode: portable HTML.
- Required technical sections mapped to visible report blocks: title, technical
  summary, key findings, scope/definitions, experiment design, validation,
  limitations/robustness, next steps, and further questions.
- The report uses one grouped bar chart because the primary comparison is three
  controls across the same three established delta metrics. Exact seed and
  target values remain in tables and CSVs.

## Chart map

- Section: channel identity did not improve the established metrics.
- Question: how do the three controls change five-seed mean Unlearn, Preserve,
  and Overall accuracy relative to official?
- Family/type: comparison / grouped bar.
- Fields: control, metric, delta percentage points; retained context includes
  favorable-seed count, baseline mean, treatment mean, and comparison id.
- Palette: relaxed three-category palette for the three established metrics;
  signed labels and a zero reference carry direction without red/green
  semantics.
- Caveat: lower Unlearn is favorable, while higher Preserve and Overall are
  favorable. The report states this immediately adjacent to the chart.

## Validation and transformations

- Result-file hashes were checked against the archived `result_manifest.json`.
- All 60,000 raw score rows were reread with Ruby CSV, and all 20 seed/variant
  metrics were independently recomputed with zero numerical discrepancy.
- Generation keys, role counts, concept denominators, score hashes, matrix
  gates, image-manifest uniqueness, worker completion, and final Git cleanliness
  were independently checked.
- Variant means and sample standard deviations are descriptive summaries across
  the five fixed edit seeds. No inferential population claim, combined score,
  or significance threshold is introduced.
- Family-level target and retain deltas are unweighted means of the existing
  per-concept accuracy deltas within each registered Confuse5 family.
- The canonical portable-report builder could not run because this Mac has no
  Node runtime. `report.html` was therefore rendered as dependency-free static
  HTML from the same checked values. Structural link/content checks were run,
  but automated browser rendering QA was unavailable.

## Omitted evidence

- The 45,000-row image manifest is omitted from Git because its verified SHA-256
  and row count suffice for the report; the full file remains in
  `.local_artifacts`.
- Raw score CSVs and Torch diagnostics remain in `.local_artifacts` because the
  committed aggregate evidence is enough for routine review.
- The archive cleanup sidecar was not present in Downloads. This does not affect
  calculation validity, but server-side deletion of regenerable PNGs is not yet
  independently verified.
