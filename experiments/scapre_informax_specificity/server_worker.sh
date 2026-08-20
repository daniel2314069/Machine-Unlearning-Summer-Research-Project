#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <smoke|formal> <run-dir> <assets-manifest>" >&2
  exit 2
fi

PROFILE="$1"
RUN_DIR="$2"
ASSETS="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python || true)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" || -z "$PYTHON_BIN" ]]; then
  EXIT_CODE=2
  echo "ERROR: detached worker did not inherit active Conda MU" >&2
else
  "$PYTHON_BIN" "$SCRIPT_DIR/worker.py" \
    --profile "$PROFILE" \
    --run-dir "$RUN_DIR" \
    --config "$SCRIPT_DIR/config.json" \
    --assets "$ASSETS" \
    --device cuda:0
  EXIT_CODE=$?
fi

printf '%s\n' "$EXIT_CODE" > "$RUN_DIR/exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at_utc"
if [[ "$EXIT_CODE" -eq 0 && -f "$RUN_DIR/worker_complete.json" ]]; then
  printf 'completed\n' > "$RUN_DIR/COMPLETED"
else
  printf 'failed\n' > "$RUN_DIR/FAILED"
fi
exit "$EXIT_CODE"
