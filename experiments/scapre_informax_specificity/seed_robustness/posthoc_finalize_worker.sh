#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <failed-formal-run-dir>" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$1"
PYTHON_BIN="$(command -v python || true)"
FINAL_EXIT=0
POSTHOC_ALREADY_PASSED=0

if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" || -z "$PYTHON_BIN" ]]; then
  echo "ERROR: post-hoc worker did not inherit Conda MU" >&2
  FINAL_EXIT=2
elif [[ -f "$RUN_DIR/reproducibility/posthoc_finalization.json" && \
        -f "$RUN_DIR/archive_manifest.json" ]] && \
     [[ "$("$PYTHON_BIN" "$SCRIPT_DIR/json_stdlib.py" get \
          "$RUN_DIR/reproducibility/posthoc_finalization.json" status)" == "passed" ]]; then
  echo "Reusing the previously validated post-hoc aggregation and archive."
  POSTHOC_ALREADY_PASSED=1
else
  if "$PYTHON_BIN" "$SCRIPT_DIR/posthoc_finalize.py" \
      --run-dir "$RUN_DIR" \
      --config "$SCRIPT_DIR/config.json" \
      --base-config "$SCRIPT_DIR/../config.json"; then
    FINAL_EXIT=0
  else
    FINAL_EXIT=$?
  fi
fi

if [[ "$FINAL_EXIT" -eq 0 ]]; then
  ORIGINAL_FAILURE="$RUN_DIR/reproducibility/original_failure"
  mkdir -p "$ORIGINAL_FAILURE"
  for name in calculation_exit_code exit_code FAILED calculation_finished_at_utc finished_at_utc; do
    if [[ -f "$RUN_DIR/$name" && ! -f "$ORIGINAL_FAILURE/$name" ]]; then
      cp "$RUN_DIR/$name" "$ORIGINAL_FAILURE/$name"
    fi
  done
  rm -f -- "$RUN_DIR/FAILED"
  printf '0\n' > "$RUN_DIR/calculation_exit_code"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/calculation_finished_at_utc"
  printf 'post-hoc calculation completed\n' > "$RUN_DIR/CALCULATION_COMPLETED"
  if [[ "$POSTHOC_ALREADY_PASSED" -eq 1 ]]; then
    echo "No aggregation rerun is needed; continuing verified packaging/cleanup."
  fi
  if ! SCAPRE_INTERNAL_FINALIZE=1 "$SCRIPT_DIR/package_results.sh" "$RUN_DIR"; then
    echo "ERROR: post-hoc result packaging failed" >&2
    FINAL_EXIT=3
  elif ! "$SCRIPT_DIR/cleanup_images.sh" "$RUN_DIR" formal; then
    echo "ERROR: post-hoc verified-archive image cleanup failed" >&2
    FINAL_EXIT=4
  fi
fi

printf '%s\n' "$FINAL_EXIT" > "$RUN_DIR/posthoc_finalize_exit_code"
printf '%s\n' "$FINAL_EXIT" > "$RUN_DIR/exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at_utc"
rm -f -- "$RUN_DIR/FINALIZING"
if [[ "$FINAL_EXIT" -eq 0 ]]; then
  printf 'completed after audited post-hoc aggregation\n' > "$RUN_DIR/COMPLETED"
  rm -f -- "$RUN_DIR/FAILED"
else
  printf 'post-hoc finalization failed\n' > "$RUN_DIR/FAILED"
fi
exit "$FINAL_EXIT"
