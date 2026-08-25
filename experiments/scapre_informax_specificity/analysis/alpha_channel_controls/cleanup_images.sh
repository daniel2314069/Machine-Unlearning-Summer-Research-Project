#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$(cd "$SCRIPT_DIR/runs" 2>/dev/null && pwd || true)"
RUN_DIR="$(cd "${1:-}" 2>/dev/null && pwd || true)"
[[ -n "$RUNS_DIR" && -n "$RUN_DIR" && "$RUN_DIR" == "$RUNS_DIR"/* ]] || {
  echo "ERROR: cleanup target must be one explicit alpha-control run" >&2
  exit 2
}
[[ -f "$RUN_DIR/archive_manifest.json" && -f "$RUN_DIR/results/generated_image_manifest.csv" ]] || {
  echo "ERROR: verified archive and image manifest are required before cleanup" >&2
  exit 2
}
ARCHIVE="$(sed -n 's/^  "archive": "\([^"]*\)".*/\1/p' "$RUN_DIR/archive_manifest.json")"
EXPECTED_SHA="$(sed -n 's/^  "sha256": "\([^"]*\)".*/\1/p' "$RUN_DIR/archive_manifest.json")"
[[ -f "$ARCHIVE" && "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$EXPECTED_SHA" ]] || {
  echo "ERROR: result archive failed verification; refusing image cleanup" >&2
  exit 2
}
if [[ -f "$RUN_DIR/cleanup_manifest.json" ]]; then
  echo "Image cleanup already recorded: $RUN_DIR/cleanup_manifest.json"
  exit 0
fi

EXPECTED_COUNT=$(( $(wc -l < "$RUN_DIR/results/generated_image_manifest.csv") - 1 ))
PROFILE="$(tr -d '\r\n' < "$RUN_DIR/profile")"
if [[ "$PROFILE" == "formal" ]]; then
  SEEDS=(20260820 20260821 20260822 20260823 20260824)
  VARIANTS=(constant_mean shuffled identity_B)
else
  SEEDS=(20260820)
  VARIANTS=(official constant_mean shuffled shuffled_alt1 shuffled_alt2 identity_B)
fi
FOUND=0
for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    IMAGE_DIR="$RUN_DIR/seeds/$seed/evaluation/$variant/images"
    [[ -d "$IMAGE_DIR" ]] || { echo "ERROR: image directory missing: $IMAGE_DIR" >&2; exit 2; }
    COUNT="$(find "$IMAGE_DIR" -type f -name '*.png' | wc -l | tr -d ' ')"
    FOUND=$((FOUND + COUNT))
  done
done
[[ "$FOUND" -eq "$EXPECTED_COUNT" ]] || {
  echo "ERROR: image count $FOUND differs from hashed manifest $EXPECTED_COUNT" >&2
  exit 2
}
for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    IMAGE_DIR="$RUN_DIR/seeds/$seed/evaluation/$variant/images"
    find "$IMAGE_DIR" -type f -name '*.png' -delete
  done
done
REMAINING="$(find "$RUN_DIR/seeds" -path '*/evaluation/*/images/*.png' -type f | wc -l | tr -d ' ')"
[[ "$REMAINING" -eq 0 ]] || { echo "ERROR: generated PNG cleanup incomplete" >&2; exit 2; }
{
  echo '{'
  echo '  "status": "passed",'
  echo "  \"deleted_png_files\": $FOUND,"
  echo "  \"image_manifest_sha256\": \"$(sha256sum "$RUN_DIR/results/generated_image_manifest.csv" | awk '{print $1}')\","
  echo "  \"verified_archive_sha256\": \"$EXPECTED_SHA\","
  echo "  \"completed_at_utc\": \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\""
  echo '}'
} > "$RUN_DIR/cleanup_manifest.json"
cp "$RUN_DIR/cleanup_manifest.json" "$ARCHIVE.cleanup.json"
echo "Deleted $FOUND regenerable PNG files after archive and image-hash verification."
echo "Cleanup record: $RUN_DIR/cleanup_manifest.json"
