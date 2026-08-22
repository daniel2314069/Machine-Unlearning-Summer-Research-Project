#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <run-dir> <smoke|formal>" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$(cd "$SCRIPT_DIR/runs" 2>/dev/null && pwd || true)"
RUN_DIR="$(cd "$1" 2>/dev/null && pwd || true)"
PROFILE="$2"
if [[ -z "$RUNS_DIR" || -z "$RUN_DIR" || "$RUN_DIR" != "$RUNS_DIR"/* ]]; then
  echo "ERROR: cleanup target must be one explicit superclass-neutral run" >&2
  exit 2
fi
PYTHON_BIN="$(tr -d '\r\n' < "$RUN_DIR/python_path" 2>/dev/null || true)"
JSON_HELPER="$SCRIPT_DIR/../seed_robustness/json_stdlib.py"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { echo "ERROR: recorded MU Python is unavailable" >&2; exit 2; }
[[ "$PROFILE" == "smoke" || "$PROFILE" == "formal" ]] || { echo "ERROR: invalid profile" >&2; exit 2; }
[[ -f "$RUN_DIR/archive_manifest.json" ]] || { echo "ERROR: cleanup requires verified archive manifest" >&2; exit 2; }
ARCHIVE="$($PYTHON_BIN "$JSON_HELPER" get "$RUN_DIR/archive_manifest.json" archive)"
EXPECTED_SHA="$($PYTHON_BIN "$JSON_HELPER" get "$RUN_DIR/archive_manifest.json" sha256)"
[[ -f "$ARCHIVE" && "$($PYTHON_BIN "$JSON_HELPER" sha256 "$ARCHIVE")" == "$EXPECTED_SHA" ]] || {
  echo "ERROR: archive is missing or changed; no images deleted" >&2; exit 2;
}
if [[ -f "$RUN_DIR/cleanup_manifest.json" ]]; then
  [[ "$($PYTHON_BIN "$JSON_HELPER" get "$RUN_DIR/cleanup_manifest.json" status)" == "passed" ]] || exit 2
  echo "Full evaluation image cleanup already completed."
  exit 0
fi
RECORDS="$(mktemp "$RUN_DIR/.cleanup_records.XXXXXX")"
trap 'rm -f -- "$RECORDS"' EXIT
TOTAL_FILES=0
TOTAL_BYTES=0
if [[ "$PROFILE" == "formal" ]]; then SEEDS=(20260820 20260821 20260822 20260823 20260824); EXPECTED_LINES=3001; else SEEDS=(20260821); EXPECTED_LINES=11; fi
for seed in "${SEEDS[@]}"; do
  EVAL="$RUN_DIR/seeds/$seed/evaluation/superclass_neutral"
  IMAGES="$EVAL/images"
  if [[ ! -d "$IMAGES" ]]; then
    printf 'skipped\t%s\timage directory absent\t0\t0\n' "$IMAGES" >> "$RECORDS"
    continue
  fi
  [[ -f "$EVAL/COMPLETED" && -f "$EVAL/scores.csv" ]] || { echo "ERROR: incomplete evaluation: $EVAL" >&2; exit 2; }
  [[ "$(wc -l < "$EVAL/scores.csv" | tr -d ' ')" == "$EXPECTED_LINES" ]] || { echo "ERROR: score count changed: $EVAL" >&2; exit 2; }
  if find "$IMAGES" -type f ! -name '*.png' -print -quit | grep -q .; then
    echo "ERROR: non-PNG under full evaluation images: $IMAGES" >&2
    exit 2
  fi
  COUNT="$(find "$IMAGES" -type f -name '*.png' | wc -l | tr -d ' ')"
  [[ "$COUNT" == "$((EXPECTED_LINES - 1))" ]] || { echo "ERROR: generated image count changed: $COUNT" >&2; exit 2; }
  BYTES="$(du -sb "$IMAGES" | awk '{print $1}')"
  find "$IMAGES" -type f -name '*.png' -delete
  find "$IMAGES" -depth -type d -empty -delete
  TOTAL_FILES=$((TOTAL_FILES + COUNT)); TOTAL_BYTES=$((TOTAL_BYTES + BYTES))
  printf 'deleted\t%s\tfull superclass evaluation seed=%s\t%s\t%s\n' "$IMAGES" "$seed" "$COUNT" "$BYTES" >> "$RECORDS"
done
if [[ "$PROFILE" == "formal" ]]; then
  QUAL_COUNT="$(find "$RUN_DIR/qualitative/images" -type f -name '*.png' | wc -l | tr -d ' ')"
  PANEL_COUNT="$(find "$RUN_DIR/qualitative/comparisons" -type f -name '*.png' | wc -l | tr -d ' ')"
  [[ "$QUAL_COUNT" == "90" && "$PANEL_COUNT" == "30" ]] || { echo "ERROR: qualitative images are incomplete; cleanup stopped" >&2; exit 2; }
  printf 'preserved\t%s\tqualitative images and paired panels\t0\t0\n' "$RUN_DIR/qualitative" >> "$RECORDS"
fi
"$PYTHON_BIN" "$JSON_HELPER" cleanup-manifest \
  "$RUN_DIR/cleanup_manifest.json" "$PROFILE" "$ARCHIVE" "$EXPECTED_SHA" \
  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$TOTAL_FILES" "$TOTAL_BYTES" "$RECORDS"
cp "$RUN_DIR/cleanup_manifest.json" "$ARCHIVE.cleanup.json"
rm -f -- "$RECORDS"; trap - EXIT
echo "Full evaluation image cleanup completed."
echo "Deleted PNG files: $TOTAL_FILES"
echo "Preserved qualitative images: $([[ "$PROFILE" == "formal" ]] && echo 90 || echo 0)"
echo "Preserved side-by-side panels: $([[ "$PROFILE" == "formal" ]] && echo 30 || echo 0)"
echo "Manifest: $RUN_DIR/cleanup_manifest.json"
echo "Downloadable cleanup manifest: $ARCHIVE.cleanup.json"
