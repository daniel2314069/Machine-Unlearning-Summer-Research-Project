#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
CONFIG="$SCRIPT_DIR/config.json"
PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python is unavailable in active Conda MU" >&2
  exit 2
fi
JSON_HELPER="$SCRIPT_DIR/json_stdlib.py"
ARCHIVE_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
RUN_ID="$("$PYTHON_BIN" "$JSON_HELPER" get "$CONFIG" prior_seed.run_id)"
ARCHIVE_BASENAME="$("$PYTHON_BIN" "$JSON_HELPER" get "$CONFIG" prior_seed.archive_basename)"
EXPECTED_SHA="$("$PYTHON_BIN" "$JSON_HELPER" get "$CONFIG" prior_seed.archive_sha256)"
PRIOR_RUN="$EXPERIMENT_DIR/runs/$RUN_ID"

required_prior_files() {
  local root="$1"
  local paths=(
    actual_config.json protocol.csv protocol_manifest.json run_manifest.json
    controlled_ablation_check.json exit_code COMPLETED
    results/aggregate.csv results/per_group.csv results/per_concept.csv
    results/informax_diagnostics.csv results/result_manifest.json
    evaluation/official/evaluation_manifest.json evaluation/official/scores.csv
    evaluation/matched_retain/evaluation_manifest.json
    evaluation/matched_retain/scores.csv
  )
  local relative
  for relative in "${paths[@]}"; do
    [[ -f "$root/$relative" ]] || return 1
  done
}

if [[ -d "$PRIOR_RUN" ]]; then
  if ! required_prior_files "$PRIOR_RUN"; then
    echo "ERROR: the expected prior run exists but is incomplete: $PRIOR_RUN" >&2
    exit 2
  fi
  printf '%s\n' "$PRIOR_RUN"
  exit 0
fi

EXTRACTED="$STATE_DIR/prior_seed_20260820"
if [[ -d "$EXTRACTED" ]]; then
  if ! required_prior_files "$EXTRACTED"; then
    echo "ERROR: the cached prior-seed extraction is incomplete: $EXTRACTED" >&2
    exit 2
  fi
  printf '%s\n' "$EXTRACTED"
  exit 0
fi

ARCHIVE="$ARCHIVE_DIR/$ARCHIVE_BASENAME"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "ERROR: seed 20260820 was not found as a completed run or verified archive" >&2
  echo "Expected run: $PRIOR_RUN" >&2
  echo "Expected archive: $ARCHIVE" >&2
  exit 2
fi
ACTUAL_SHA="$("$PYTHON_BIN" "$JSON_HELPER" sha256 "$ARCHIVE")"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: prior archive checksum mismatch" >&2
  exit 2
fi
if ! tar -tzf "$ARCHIVE" | awk '$0 ~ /^\// || $0 ~ /(^|\/)\.\.($|\/)/ {bad=1} END {exit bad ? 1 : 0}'; then
  echo "ERROR: prior archive contains an unsafe path" >&2
  exit 2
fi

mkdir -p "$STATE_DIR"
TEMP_DIR="$(mktemp -d "$STATE_DIR/prior_seed_extract.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
FILES=(
  actual_config.json protocol.csv protocol_manifest.json run_manifest.json
  controlled_ablation_check.json exit_code COMPLETED
  results/aggregate.csv results/per_group.csv results/per_concept.csv
  results/informax_diagnostics.csv results/result_manifest.json
  evaluation/official/evaluation_manifest.json evaluation/official/scores.csv
  evaluation/matched_retain/evaluation_manifest.json
  evaluation/matched_retain/scores.csv
)
tar -xzf "$ARCHIVE" -C "$TEMP_DIR" "${FILES[@]}"
if ! required_prior_files "$TEMP_DIR"; then
  echo "ERROR: verified prior archive did not yield the required files" >&2
  exit 2
fi
mv "$TEMP_DIR" "$EXTRACTED"
trap - EXIT
printf '%s\n' "$EXTRACTED"
