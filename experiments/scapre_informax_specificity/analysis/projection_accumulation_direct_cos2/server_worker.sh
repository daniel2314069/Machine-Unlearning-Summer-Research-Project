#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$SCRIPT_DIR/../projection_accumulation"
RUN_DIR="$1"
ASSETS="$2"
OFFICIAL_REFERENCE="$3"
V1_RUN_DIR="$4"
PYTHON_BIN="$(tr -d '\r\n' < "$RUN_DIR/python_path")"

"$PYTHON_BIN" "$SHARED_DIR/worker.py" \
  --experiment-dir "$SCRIPT_DIR" \
  --run-dir "$RUN_DIR" \
  --assets "$ASSETS" \
  --official-reference "$OFFICIAL_REFERENCE" \
  --v1-run-dir "$V1_RUN_DIR"
EXIT_CODE=$?
printf '%s\n' "$EXIT_CODE" > "$RUN_DIR/exit_code"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at_utc"
if [[ "$EXIT_CODE" -eq 0 && -f "$RUN_DIR/worker_complete.json" ]]; then
  printf 'completed\n' > "$RUN_DIR/COMPLETED"
else
  printf 'failed\n' > "$RUN_DIR/FAILED"
fi
exit "$EXIT_CODE"

