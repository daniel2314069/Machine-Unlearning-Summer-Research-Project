#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="$1"
RUN_DIR="$2"
ASSETS="$3"
PYTHON_BIN="$(tr -d '\r\n' < "$RUN_DIR/python_path")"

"$SCRIPT_DIR/ensure_assets.sh" "$RUN_DIR"
ASSET_EXIT=$?
if [[ "$ASSET_EXIT" -ne 0 ]]; then
  printf '%s\n' "$ASSET_EXIT" > "$RUN_DIR/calculation_exit_code"
  printf '%s\n' "$ASSET_EXIT" > "$RUN_DIR/exit_code"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/calculation_finished_at_utc"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at_utc"
  printf 'failed\n' > "$RUN_DIR/FAILED"
  exit "$ASSET_EXIT"
fi

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/worker.py"
  --profile "$PROFILE"
  --run-dir "$RUN_DIR"
  --assets "$ASSETS"
  --device cuda:0
)
if [[ "$PROFILE" == "formal" ]]; then
  OFFICIAL_REFERENCE="$($SCRIPT_DIR/resolve_official_reference.sh)"
  REFERENCE_EXIT=$?
  if [[ "$REFERENCE_EXIT" -eq 0 ]]; then
    COMMAND+=(--official-reference "$OFFICIAL_REFERENCE")
  elif [[ "$REFERENCE_EXIT" -eq 3 ]]; then
    echo "Verified historical baseline is unavailable; generating the repository baseline in this run."
  else
    printf '%s\n' "$REFERENCE_EXIT" > "$RUN_DIR/calculation_exit_code"
    printf '%s\n' "$REFERENCE_EXIT" > "$RUN_DIR/exit_code"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/calculation_finished_at_utc"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at_utc"
    printf 'failed\n' > "$RUN_DIR/FAILED"
    exit "$REFERENCE_EXIT"
  fi
fi

if [[ -f "$RUN_DIR/CALCULATION_COMPLETED" && -f "$RUN_DIR/worker_complete.json" && \
      -f "$RUN_DIR/calculation_exit_code" && \
      "$(tr -d '[:space:]' < "$RUN_DIR/calculation_exit_code")" == "0" ]]; then
  echo "[resume] calculation already completed; retrying final packaging only"
  CALCULATION_EXIT=0
else
  "${COMMAND[@]}"
  CALCULATION_EXIT=$?
fi
printf '%s\n' "$CALCULATION_EXIT" > "$RUN_DIR/calculation_exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/calculation_finished_at_utc"
FINAL_EXIT="$CALCULATION_EXIT"
printf '%s\n' "$FINAL_EXIT" > "$RUN_DIR/exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at_utc"
if [[ "$CALCULATION_EXIT" -eq 0 && -f "$RUN_DIR/worker_complete.json" ]]; then
  printf 'calculation completed\n' > "$RUN_DIR/CALCULATION_COMPLETED"
  if ! PAPER_MI_INTERNAL_PACKAGE=1 "$SCRIPT_DIR/package_results.sh" "$RUN_DIR"; then
    echo "ERROR: automatic result packaging failed" >&2
    FINAL_EXIT=3
  fi
fi
printf '%s\n' "$FINAL_EXIT" > "$RUN_DIR/exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at_utc"
if [[ "$FINAL_EXIT" -eq 0 ]]; then
  printf 'completed\n' > "$RUN_DIR/COMPLETED"
  if [[ "$PROFILE" == "smoke" ]]; then
    printf '%s\n' "$RUN_DIR" > "$SCRIPT_DIR/.server/latest_successful_smoke"
  fi
else
  printf 'failed\n' > "$RUN_DIR/FAILED"
fi
exit "$FINAL_EXIT"
