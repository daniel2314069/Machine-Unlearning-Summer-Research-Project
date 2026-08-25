#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="$1"
RUN_DIR="$2"
ASSETS="$3"
OFFICIAL_REFERENCE="${4:-}"
PYTHON_BIN="$(tr -d '\r\n' < "$RUN_DIR/python_path")"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/worker.py"
  --profile "$PROFILE"
  --run-dir "$RUN_DIR"
  --assets "$ASSETS"
  --device cuda:0
)
if [[ "$PROFILE" == "formal" ]]; then
  COMMAND+=(--official-reference "$OFFICIAL_REFERENCE")
fi

if [[ -f "$RUN_DIR/CALCULATION_COMPLETED" && -f "$RUN_DIR/worker_complete.json" && \
      -f "$RUN_DIR/calculation_exit_code" && \
      "$(tr -d '[:space:]' < "$RUN_DIR/calculation_exit_code")" == "0" ]]; then
  echo "[resume] calculation already completed; retrying finalization only"
  CALCULATION_EXIT=0
else
  "${COMMAND[@]}"
  CALCULATION_EXIT=$?
fi
printf '%s\n' "$CALCULATION_EXIT" > "$RUN_DIR/calculation_exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/calculation_finished_at_utc"
FINAL_EXIT="$CALCULATION_EXIT"
if [[ "$CALCULATION_EXIT" -eq 0 && -f "$RUN_DIR/worker_complete.json" ]]; then
  printf 'calculation completed\n' > "$RUN_DIR/CALCULATION_COMPLETED"
  if ! ALPHA_CONTROLS_INTERNAL_PACKAGE=1 "$SCRIPT_DIR/package_results.sh" "$RUN_DIR"; then
    echo "ERROR: automatic packaging failed" >&2
    FINAL_EXIT=3
  elif ! "$SCRIPT_DIR/cleanup_images.sh" "$RUN_DIR"; then
    echo "ERROR: post-archive image cleanup failed" >&2
    FINAL_EXIT=4
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
