#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="$(cd "$SCRIPT_DIR/runs/coco" 2>/dev/null && pwd || true)"
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  [[ -f "$SCRIPT_DIR/.server/coco/latest_run" ]] || { echo "ERROR: no recorded COCO run" >&2; exit 2; }
  RUN_DIR="$(tr -d '\r\n' < "$SCRIPT_DIR/.server/coco/latest_run")"
fi
RUN_DIR="$(cd "$RUN_DIR" 2>/dev/null && pwd || true)"
[[ -n "$RUNS_ROOT" && -n "$RUN_DIR" && "$RUN_DIR" == "$RUNS_ROOT"/* ]] || {
  echo "ERROR: package target must be one explicit COCO run" >&2; exit 2;
}
[[ -f "$RUN_DIR/COMPLETED" && "$(tr -d '[:space:]' < "$RUN_DIR/exit_code")" == "0" ]] || {
  echo "ERROR: package requires a completed successful COCO run" >&2; exit 2;
}
INTEGRITY="$RUN_DIR/results/integrity_report.json"
[[ -f "$INTEGRITY" ]] || { echo "ERROR: COCO integrity report missing" >&2; exit 2; }
command -v jq >/dev/null || { echo "ERROR: COCO packaging requires jq" >&2; exit 2; }
jq -e '.status == "passed"' "$INTEGRITY" >/dev/null || { echo "ERROR: COCO integrity did not pass" >&2; exit 2; }

FILES=(
  COMPLETED exit_code finished_at_utc started_at_utc python_path mode logs/server.log status.json
  worker_complete.json results/metrics.json results/summary.md results/integrity_report.json
  reproducibility/protocol.json reproducibility/run_manifest.json
  reproducibility/edit_isolation.json reproducibility/generated_image_manifest.csv
)
PROMPT_MANIFEST="$(find "$RUN_DIR/reproducibility" -maxdepth 1 -name 'prompts_first*.csv' -type f)"
[[ "$(printf '%s\n' "$PROMPT_MANIFEST" | sed '/^$/d' | wc -l | tr -d ' ')" == "1" ]] || {
  echo "ERROR: expected exactly one COCO prompt manifest" >&2; exit 2;
}
FILES+=("reproducibility/$(basename "$PROMPT_MANIFEST")")
for relative in "${FILES[@]}"; do
  [[ -f "$RUN_DIR/$relative" ]] || { echo "ERROR: required COCO package file missing: $relative" >&2; exit 2; }
done
printf '%s\n' "${FILES[@]}" > "$RUN_DIR/package_file_manifest.txt"
FILES+=(package_file_manifest.txt)

ARCHIVE_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
mkdir -p "$ARCHIVE_DIR"
ARCHIVE="$ARCHIVE_DIR/scapre_projection_coco_$(basename "$RUN_DIR")_$(date -u +'%Y%m%dT%H%M%SZ').tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo "ERROR: archive already exists: $ARCHIVE" >&2; exit 2; }
tar -C "$RUN_DIR" -czf "$ARCHIVE" "${FILES[@]}"
tar -tzf "$ARCHIVE" >/dev/null
CHECKSUM="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
SIZE_BYTES="$(wc -c < "$ARCHIVE" | tr -d ' ')"
printf '%s  %s\n' "$CHECKSUM" "$ARCHIVE" > "$ARCHIVE.sha256"
echo "Archive: $ARCHIVE"
echo "Size: $(du -h "$ARCHIVE" | awk '{print $1}') ($SIZE_BYTES bytes)"
echo "SHA-256: $CHECKSUM"
echo "Mac download: scp 'tslin:$ARCHIVE' ~/Downloads/"
echo "Generated PNGs, checkpoints, model weights, and caches were excluded; server outputs were preserved."
