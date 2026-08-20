#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$SCRIPT_DIR/.run"
PID_FILE="$RUN_DIR/last.pid"
EXIT_FILE="$RUN_DIR/last.exit"
LOG_POINTER="$RUN_DIR/last.log"

if [[ ! -s "$PID_FILE" || ! -s "$LOG_POINTER" ]]; then
    echo "No qualification run has been registered." >&2
    exit 2
fi
PID="$(tr -d '[:space:]' < "$PID_FILE")"
LOG_FILE="$(tr -d '\n' < "$LOG_POINTER")"

if kill -0 "$PID" 2>/dev/null; then
    echo "Status: running (PID $PID)"
else
    EXIT_CODE="$(tr -d '[:space:]' < "$EXIT_FILE")"
    echo "Status: finished (exit ${EXIT_CODE:-unknown})"
fi
echo "Log: $LOG_FILE"
tail -n 80 "$LOG_FILE"
