#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [run-directory]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  RUN_DIR="$1"
else
  LATEST_FILE="$SCRIPT_DIR/runs/latest_run"
  [[ -f "$LATEST_FILE" ]] || { echo "No run has been launched."; exit 1; }
  RUN_DIR="$(<"$LATEST_FILE")"
fi
[[ -d "$RUN_DIR" ]] || { echo "ERROR: run directory not found: $RUN_DIR" >&2; exit 1; }
PID="$(<"$RUN_DIR/pid" 2>/dev/null || printf 'unknown')"
LOG_PATH="$(<"$RUN_DIR/log_path" 2>/dev/null || printf '%s/server.log' "$RUN_DIR")"
OUTPUT_PATH="$(<"$RUN_DIR/output_path" 2>/dev/null || printf '%s/output' "$RUN_DIR")"
if [[ -f "$RUN_DIR/COMPLETED" && "$(<"$RUN_DIR/exit_code")" == "0" ]]; then
  STATE="completed"
elif [[ -f "$RUN_DIR/exit_code" ]]; then
  STATE="failed"
elif [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
  STATE="running"
else
  STATE="failed (worker absent without a recorded exit code)"
fi
echo "State: $STATE"
echo "Run: $RUN_DIR"
echo "PID: $PID"
echo "Exit code: $(<"$RUN_DIR/exit_code" 2>/dev/null || printf 'not recorded')"
echo "Log: $LOG_PATH"
echo "Output: $OUTPUT_PATH"
echo "Recent log:"
tail -n 40 "$LOG_PATH" 2>/dev/null || true
