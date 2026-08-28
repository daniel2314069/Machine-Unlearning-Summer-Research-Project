#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="$1"
RUN_DIR="$2"
ASSETS="$3"
PYTHON_BIN="$(tr -d '\r\n' < "$RUN_DIR/python_path")"

"$PYTHON_BIN" "$SCRIPT_DIR/coco_worker.py" --mode "$MODE" --run-dir "$RUN_DIR" --assets "$ASSETS"
EXIT_CODE=$?
printf '%s\n' "$EXIT_CODE" > "$RUN_DIR/exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at_utc"
if [[ "$EXIT_CODE" -eq 0 && -f "$RUN_DIR/worker_complete.json" ]]; then
  printf 'completed\n' > "$RUN_DIR/COMPLETED"
  if [[ "$MODE" == "first-1k" ]]; then
    printf '%s\n' "$RUN_DIR" > "$SCRIPT_DIR/.server/coco/latest_successful_first1k"
  fi
else
  printf 'failed\n' > "$RUN_DIR/FAILED"
fi
exit "$EXIT_CODE"
