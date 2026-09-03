# ScaPre paper-MI formal result

Run `formal_20260902T151717Z` completed successfully on commit
`6f5e047179244b860f73623f88137ada24aff11a`. It used all five established
Informax seeds and reused the checksum-pinned, fully validated repository
baseline.

Paper MI reduced mean target/unlearn accuracy by 13.5500 percentage points,
but also reduced preservation by 20.0444 points and overall accuracy by
20.3442 points. The preservation and overall deltas were unfavorable for all
five seeds. The paper alpha was effectively uniform because max-over-concept
raw MI saturated near `ln(2)` for every channel.

This directory contains the lightweight, reviewable result record. Raw score
CSVs, diagnostics, logs, and other packaged metadata are retained locally at:

```text
.local_artifacts/scapre_informax/scapre_paper_mi_formal_20260902T151717Z_20260903T061852Z
```

The downloaded archive remains outside Git in `~/Downloads/`. Generated
images, checkpoints, and model caches remain on the server and were excluded
from the archive by design.

See [`validation_report.md`](validation_report.md) for the conclusion,
independent checks, and caveats.
