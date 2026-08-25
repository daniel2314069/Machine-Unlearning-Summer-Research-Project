#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$(cd "$SCRIPT_DIR/runs" 2>/dev/null && pwd || true)"
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  [[ -f "$SCRIPT_DIR/.server/latest_run" ]] || { echo "ERROR: no recorded run" >&2; exit 2; }
  RUN_DIR="$(tr -d '\r\n' < "$SCRIPT_DIR/.server/latest_run")"
fi
RUN_DIR="$(cd "$RUN_DIR" 2>/dev/null && pwd || true)"
[[ -n "$RUNS_DIR" && -n "$RUN_DIR" && "$RUN_DIR" == "$RUNS_DIR"/* ]] || {
  echo "ERROR: package target must be one explicit alpha-control run" >&2
  exit 2
}
if [[ "${ALPHA_CONTROLS_INTERNAL_PACKAGE:-0}" == "1" ]]; then
  [[ -f "$RUN_DIR/CALCULATION_COMPLETED" && "$(tr -d '[:space:]' < "$RUN_DIR/calculation_exit_code")" == "0" ]] || {
    echo "ERROR: internal package requires successful calculation" >&2; exit 2;
  }
else
  [[ -f "$RUN_DIR/COMPLETED" && "$(tr -d '[:space:]' < "$RUN_DIR/exit_code")" == "0" ]] || {
    echo "ERROR: package requires completed successful run" >&2; exit 2;
  }
fi

if [[ -f "$RUN_DIR/archive_manifest.json" ]]; then
  ARCHIVE="$(sed -n 's/^  "archive": "\([^"]*\)".*/\1/p' "$RUN_DIR/archive_manifest.json")"
  EXPECTED="$(sed -n 's/^  "sha256": "\([^"]*\)".*/\1/p' "$RUN_DIR/archive_manifest.json")"
  [[ -f "$ARCHIVE" && "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$EXPECTED" ]] || {
    echo "ERROR: existing archive manifest does not verify" >&2; exit 2;
  }
  echo "Archive: $ARCHIVE"
  echo "Size: $(du -h "$ARCHIVE" | awk '{print $1}')"
  echo "SHA-256: $EXPECTED"
  exit 0
fi

PROFILE="$(tr -d '\r\n' < "$RUN_DIR/profile")"
if [[ "$PROFILE" == "formal" ]]; then
  SEEDS=(20260820 20260821 20260822 20260823 20260824)
  VARIANTS=(official constant_mean shuffled identity_B)
else
  SEEDS=(20260820)
  VARIANTS=(official constant_mean shuffled shuffled_alt1 shuffled_alt2 identity_B)
fi
FILES=(
  actual_config.json base_config.json protocol.csv protocol_manifest.json
  run_manifest.json worker_complete.json summary.md server.log profile python_path
  calculation_exit_code calculation_finished_at_utc CALCULATION_COMPLETED
  results/alpha_matrix_summary.csv results/per_seed_metrics.csv
  results/per_target_metrics.csv results/comparison_deltas.csv
  results/generated_image_manifest.csv
  results/integrity_report.json results/summary.md results/result_manifest.json
)
[[ "$PROFILE" == "formal" ]] && FILES+=(official_reference_validation.json)
for seed in "${SEEDS[@]}"; do
  FILES+=("seeds/$seed/controlled_ablation_check.json")
  for variant in "${VARIANTS[@]}"; do
    FILES+=(
      "seeds/$seed/alpha_audits/$variant.json"
      "seeds/$seed/diagnostics/$variant.pt"
      "seeds/$seed/stages/edit_$variant.completed.json"
      "seeds/$seed/stages/edit_$variant.command.json"
      "seeds/$seed/stages/checkpoint_$variant.cleanup.json"
      "seeds/$seed/evaluation/$variant/scores.csv"
      "seeds/$seed/evaluation/$variant/evaluation_manifest.json"
      "seeds/$seed/evaluation/$variant/COMPLETED"
    )
  done
done
while IFS= read -r source; do
  FILES+=("provenance/$source")
done < <(sed -n '/"source_sha256": {/,/^  }/s/^    "\([^"]*\)".*/\1/p' "$RUN_DIR/run_manifest.json")
for relative in "${FILES[@]}"; do
  [[ -f "$RUN_DIR/$relative" ]] || { echo "ERROR: required package file missing: $relative" >&2; exit 2; }
done
printf '%s\n' "${FILES[@]}" > "$RUN_DIR/package_file_manifest.txt"
FILES+=(package_file_manifest.txt)

ARCHIVE_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
mkdir -p "$ARCHIVE_DIR"
RUN_NAME="$(basename "$RUN_DIR")"
TIMESTAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
ARCHIVE="$ARCHIVE_DIR/scapre_informax_alpha_channel_controls_${RUN_NAME}_${TIMESTAMP}.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo "ERROR: archive already exists: $ARCHIVE" >&2; exit 2; }
tar -C "$RUN_DIR" -czf "$ARCHIVE" "${FILES[@]}"
tar -tzf "$ARCHIVE" >/dev/null
CHECKSUM="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
SIZE_BYTES="$(wc -c < "$ARCHIVE" | tr -d ' ')"
printf '%s  %s\n' "$CHECKSUM" "$ARCHIVE" > "$ARCHIVE.sha256"
{
  echo '{'
  echo "  \"archive\": \"$ARCHIVE\","
  echo "  \"sha256\": \"$CHECKSUM\","
  echo "  \"size_bytes\": $SIZE_BYTES,"
  echo "  \"profile\": \"$PROFILE\","
  echo "  \"created_at_utc\": \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\""
  echo '}'
} > "$RUN_DIR/archive_manifest.json"
echo "Archive: $ARCHIVE"
echo "Size: $(du -h "$ARCHIVE" | awk '{print $1}')"
echo "SHA-256: $CHECKSUM"
echo "Checksum file: $ARCHIVE.sha256"
echo "Generated images, checkpoints, model weights, and caches were excluded."
echo "Original server outputs were preserved."
