#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <smoke|formal> <run-dir> <assets-manifest> <prior-run-or-empty>" >&2
  exit 2
fi
PROFILE="$1"
RUN_DIR="$2"
ASSETS="$3"
PRIOR_RUN="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python || true)"

CALCULATION_EXIT=0
if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" || -z "$PYTHON_BIN" ]]; then
  echo "ERROR: detached worker did not inherit active Conda MU" >&2
  CALCULATION_EXIT=2
else
  WORKER_ARGS=(
    --profile "$PROFILE"
    --run-dir "$RUN_DIR"
    --config "$SCRIPT_DIR/config.json"
    --assets "$ASSETS"
    --device cuda:0
  )
  if [[ -n "$PRIOR_RUN" ]]; then
    WORKER_ARGS+=(--prior-run "$PRIOR_RUN")
  fi
  if "$PYTHON_BIN" "$SCRIPT_DIR/worker.py" "${WORKER_ARGS[@]}"; then
    CALCULATION_EXIT=0
  else
    CALCULATION_EXIT=$?
  fi
fi

printf '%s\n' "$CALCULATION_EXIT" > "$RUN_DIR/calculation_exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/calculation_finished_at_utc"
FINAL_EXIT="$CALCULATION_EXIT"
if [[ "$CALCULATION_EXIT" -eq 0 && -f "$RUN_DIR/worker_complete.json" ]]; then
  printf 'calculation completed\n' > "$RUN_DIR/CALCULATION_COMPLETED"
  if ! SCAPRE_INTERNAL_FINALIZE=1 "$SCRIPT_DIR/package_results.sh" "$RUN_DIR"; then
    echo "ERROR: automatic result packaging failed" >&2
    FINAL_EXIT=3
  elif ! "$SCRIPT_DIR/cleanup_images.sh" "$RUN_DIR" "$PROFILE"; then
    echo "ERROR: verified-archive image cleanup failed" >&2
    FINAL_EXIT=4
  fi
fi

printf '%s\n' "$FINAL_EXIT" > "$RUN_DIR/exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at_utc"
if [[ "$FINAL_EXIT" -eq 0 ]]; then
  printf 'completed\n' > "$RUN_DIR/COMPLETED"
else
  printf 'failed\n' > "$RUN_DIR/FAILED"
fi
exit "$FINAL_EXIT"
