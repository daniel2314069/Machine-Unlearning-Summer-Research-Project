#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <seed-robustness-run-dir> <smoke|formal>" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$(cd "$SCRIPT_DIR/runs" 2>/dev/null && pwd || true)"
RUN_DIR="$(cd "$1" 2>/dev/null && pwd || true)"
PROFILE="$2"
PARENT_EXPERIMENT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$SCRIPT_DIR/config.json"

if [[ -z "$RUNS_DIR" || -z "$RUN_DIR" || "$RUN_DIR" != "$RUNS_DIR"/* ]]; then
  echo "ERROR: cleanup target must be one explicit seed-robustness run" >&2
  exit 2
fi
if [[ "$PROFILE" != "smoke" && "$PROFILE" != "formal" ]]; then
  echo "ERROR: cleanup profile must be smoke or formal" >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/archive_manifest.json" ]]; then
  echo "ERROR: image cleanup requires a verified archive manifest" >&2
  exit 2
fi
ARCHIVE="$(jq -r '.archive' "$RUN_DIR/archive_manifest.json")"
EXPECTED_SHA="$(jq -r '.sha256' "$RUN_DIR/archive_manifest.json")"
if [[ ! -f "$ARCHIVE" || "$(sha256sum "$ARCHIVE" | awk '{print $1}')" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: verified archive is missing or has changed; no images were deleted" >&2
  exit 2
fi
if [[ -f "$RUN_DIR/cleanup_manifest.json" ]]; then
  if [[ "$(jq -r '.status' "$RUN_DIR/cleanup_manifest.json")" == "passed" ]]; then
    echo "Image cleanup already completed: $RUN_DIR/cleanup_manifest.json"
    exit 0
  fi
  echo "ERROR: an incomplete cleanup manifest already exists" >&2
  exit 2
fi

RECORDS='[]'
TOTAL_FILES=0
TOTAL_BYTES=0

record_skip() {
  local path="$1"
  local reason="$2"
  RECORDS="$(jq -c --arg path "$path" --arg reason "$reason" '. + [{path:$path,status:"skipped",reason:$reason,deleted_files:0,deleted_bytes:0}]' <<<"$RECORDS")"
}

delete_evaluation_images() {
  local evaluation_dir="$1"
  local expected_lines="$2"
  local context="$3"
  local images_dir="$evaluation_dir/images"
  local scores="$evaluation_dir/scores.csv"
  if [[ ! -d "$images_dir" ]]; then
    record_skip "$images_dir" "image directory absent"
    return
  fi
  if [[ ! -f "$evaluation_dir/COMPLETED" || ! -f "$scores" ]]; then
    echo "ERROR: refusing cleanup for incomplete evaluation: $evaluation_dir" >&2
    exit 2
  fi
  local actual_lines
  actual_lines="$(wc -l < "$scores" | tr -d ' ')"
  if [[ "$actual_lines" != "$expected_lines" ]]; then
    echo "ERROR: score count changed for $context: $actual_lines != $expected_lines" >&2
    exit 2
  fi
  if find "$images_dir" -type f ! -name '*.png' -print -quit | grep -q .; then
    echo "ERROR: non-PNG file found under generated-image directory: $images_dir" >&2
    exit 2
  fi
  local file_count bytes
  file_count="$(find "$images_dir" -type f -name '*.png' | wc -l | tr -d ' ')"
  bytes="$(du -sb "$images_dir" | awk '{print $1}')"
  find "$images_dir" -type f -name '*.png' -delete
  find "$images_dir" -depth -type d -empty -delete
  TOTAL_FILES=$((TOTAL_FILES + file_count))
  TOTAL_BYTES=$((TOTAL_BYTES + bytes))
  RECORDS="$(jq -c \
    --arg path "$images_dir" --arg context "$context" \
    --argjson files "$file_count" --argjson bytes "$bytes" \
    '. + [{path:$path,status:"deleted",context:$context,deleted_files:$files,deleted_bytes:$bytes}]' \
    <<<"$RECORDS")"
}

if [[ "$PROFILE" == "formal" ]]; then
  NEW_SEEDS=(20260821 20260822 20260823 20260824)
  EXPECTED_LINES=3001
else
  NEW_SEEDS=(20260821)
  EXPECTED_LINES=11
fi
for seed in "${NEW_SEEDS[@]}"; do
  for variant in official matched_retain; do
    delete_evaluation_images \
      "$RUN_DIR/seeds/$seed/evaluation/$variant" \
      "$EXPECTED_LINES" \
      "seed robustness $PROFILE seed=$seed variant=$variant"
  done
done

if [[ "$PROFILE" == "formal" ]]; then
  while IFS= read -r prior_id; do
    prior_run="$PARENT_EXPERIMENT/runs/$prior_id"
    if [[ ! -d "$prior_run" ]]; then
      record_skip "$prior_run" "previous run directory absent"
      continue
    fi
    if [[ ! -f "$prior_run/COMPLETED" || ! -f "$prior_run/exit_code" || "$(tr -d '[:space:]' < "$prior_run/exit_code")" != "0" ]]; then
      echo "ERROR: refusing cleanup for previous run without verified completion: $prior_run" >&2
      exit 2
    fi
    prior_profile="${prior_id%%_*}"
    if [[ "$prior_profile" == "formal" ]]; then
      prior_lines=3001
    elif [[ "$prior_profile" == "smoke" ]]; then
      prior_lines=11
    else
      echo "ERROR: unexpected previous run id in cleanup config: $prior_id" >&2
      exit 2
    fi
    for variant in official matched_retain; do
      delete_evaluation_images \
        "$prior_run/evaluation/$variant" \
        "$prior_lines" \
        "previous specificity run=$prior_id variant=$variant"
    done
  done < <(jq -r '.cleanup.previous_specificity_run_ids[]' "$CONFIG")
fi

jq -n \
  --arg status "passed" \
  --arg profile "$PROFILE" \
  --arg archive "$ARCHIVE" \
  --arg archive_sha256 "$EXPECTED_SHA" \
  --arg completed_at_utc "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --argjson deleted_files "$TOTAL_FILES" \
  --argjson deleted_bytes "$TOTAL_BYTES" \
  --argjson records "$RECORDS" \
  '{status:$status,profile:$profile,archive:$archive,archive_sha256:$archive_sha256,completed_at_utc:$completed_at_utc,deleted_files:$deleted_files,deleted_bytes:$deleted_bytes,records:$records}' \
  > "$RUN_DIR/cleanup_manifest.json"
cp "$RUN_DIR/cleanup_manifest.json" "$ARCHIVE.cleanup.json"

echo "Image cleanup completed."
echo "Deleted PNG files: $TOTAL_FILES"
echo "Released bytes (directory-size estimate): $TOTAL_BYTES"
echo "Manifest: $RUN_DIR/cleanup_manifest.json"
echo "Downloadable cleanup manifest: $ARCHIVE.cleanup.json"
