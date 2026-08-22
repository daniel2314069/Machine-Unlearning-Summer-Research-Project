#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROBUSTNESS_DIR="$PARENT/seed_robustness"
STATE_DIR="$SCRIPT_DIR/.server"
CONFIG="$SCRIPT_DIR/config.json"
JSON_HELPER="$ROBUSTNESS_DIR/json_stdlib.py"
PYTHON_BIN="$(command -v python || true)"
ARCHIVE_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python is unavailable in active Conda MU" >&2
  exit 2
fi
RUN_ID="$($PYTHON_BIN "$JSON_HELPER" get "$CONFIG" prior_robustness.run_id)"
ARCHIVE_BASENAME="$($PYTHON_BIN "$JSON_HELPER" get "$CONFIG" prior_robustness.archive_basename)"
EXPECTED_SHA="$($PYTHON_BIN "$JSON_HELPER" get "$CONFIG" prior_robustness.archive_sha256)"
PRIOR_RUN="$ROBUSTNESS_DIR/runs/$RUN_ID"

required_files() {
  local root="$1"
  local seed variant
  if [[ -f "$root/COMPLETED" && -f "$root/exit_code" ]]; then
    [[ "$(tr -d '[:space:]' < "$root/exit_code")" == "0" ]] || return 1
  elif [[ -f "$root/CALCULATION_COMPLETED" && -f "$root/calculation_exit_code" ]]; then
    [[ "$(tr -d '[:space:]' < "$root/calculation_exit_code")" == "0" ]] || return 1
  else
    return 1
  fi
  [[ -f "$root/run_manifest.json" ]] || return 1
  [[ -f "$root/reproducibility/integrity_report.json" && -f "$root/results/summary.md" ]] || return 1
  [[ -f "$root/results/result_manifest.json" ]] || return 1
  for seed in 20260820 20260821 20260822 20260823 20260824; do
    [[ -f "$root/seeds/$seed/results/informax_diagnostics.csv" ]] || return 1
    for variant in official matched_retain; do
      [[ -f "$root/seeds/$seed/evaluation/$variant/scores.csv" ]] || return 1
      [[ -f "$root/seeds/$seed/evaluation/$variant/evaluation_manifest.json" ]] || return 1
      [[ -f "$root/seeds/$seed/evaluation/$variant/COMPLETED" ]] || return 1
    done
  done
}

if [[ -d "$PRIOR_RUN" ]]; then
  if ! required_files "$PRIOR_RUN"; then
    echo "ERROR: expected robustness run exists but required lightweight outputs are incomplete" >&2
    exit 2
  fi
  if [[ -f "$PRIOR_RUN/archive_manifest.json" ]]; then
    RECORDED_SHA="$($PYTHON_BIN "$JSON_HELPER" get "$PRIOR_RUN/archive_manifest.json" sha256)"
    if [[ "$RECORDED_SHA" != "$EXPECTED_SHA" ]]; then
      echo "ERROR: prior run archive fingerprint differs from the pinned robustness archive" >&2
      exit 2
    fi
  fi
  printf '%s\n' "$PRIOR_RUN"
  exit 0
fi

EXTRACTED="$STATE_DIR/prior_robustness_$RUN_ID"
if [[ -d "$EXTRACTED" ]]; then
  if ! required_files "$EXTRACTED"; then
    echo "ERROR: cached robustness extraction is incomplete" >&2
    exit 2
  fi
  printf '%s\n' "$EXTRACTED"
  exit 0
fi

ARCHIVE="$ARCHIVE_DIR/$ARCHIVE_BASENAME"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "ERROR: verified robustness run/archive was not found" >&2
  echo "Expected run: $PRIOR_RUN" >&2
  echo "Expected archive: $ARCHIVE" >&2
  exit 2
fi
ACTUAL_SHA="$($PYTHON_BIN "$JSON_HELPER" sha256 "$ARCHIVE")"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: robustness archive checksum mismatch" >&2
  exit 2
fi
if ! tar -tzf "$ARCHIVE" | awk '$0 ~ /^\// || $0 ~ /(^|\/)\.\.($|\/)/ {bad=1} END {exit bad ? 1 : 0}'; then
  echo "ERROR: robustness archive contains an unsafe path" >&2
  exit 2
fi

mkdir -p "$STATE_DIR"
TEMP_DIR="$(mktemp -d "$STATE_DIR/prior_robustness_extract.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
FILES=(
  CALCULATION_COMPLETED calculation_exit_code run_manifest.json
  reproducibility/integrity_report.json results/summary.md results/result_manifest.json
)
for seed in 20260820 20260821 20260822 20260823 20260824; do
  FILES+=("seeds/$seed/results/informax_diagnostics.csv")
  for variant in official matched_retain; do
    FILES+=(
      "seeds/$seed/evaluation/$variant/scores.csv"
      "seeds/$seed/evaluation/$variant/evaluation_manifest.json"
      "seeds/$seed/evaluation/$variant/COMPLETED"
    )
  done
done
tar -xzf "$ARCHIVE" -C "$TEMP_DIR" "${FILES[@]}"
if ! required_files "$TEMP_DIR"; then
  echo "ERROR: verified archive did not yield the required prior files" >&2
  exit 2
fi
mv "$TEMP_DIR" "$EXTRACTED"
trap - EXIT
printf '%s\n' "$EXTRACTED"
