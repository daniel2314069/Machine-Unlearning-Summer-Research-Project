#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$SCRIPT_DIR/.run"
PID_FILE="$RUN_DIR/last.pid"
EXIT_FILE="$RUN_DIR/last.exit"
LOG_POINTER="$RUN_DIR/last.log"

if [[ ! -s "$PID_FILE" || ! -s "$LOG_POINTER" ]]; then
    echo "No submitted sequential OCE follow-up job was found."
    exit 1
fi

JOB_PID="$(tr -d '[:space:]' < "$PID_FILE")"
LOG_FILE="$(tr -d '\n' < "$LOG_POINTER")"
if [[ -s "$EXIT_FILE" ]]; then
    EXIT_CODE="$(tr -d '[:space:]' < "$EXIT_FILE")"
    if [[ "$EXIT_CODE" == "0" ]]; then
        echo "Status: complete (exit 0)"
    else
        echo "Status: failed (exit $EXIT_CODE)"
    fi
elif [[ "$JOB_PID" =~ ^[0-9]+$ ]] && kill -0 "$JOB_PID" 2>/dev/null; then
    echo "Status: running (PID $JOB_PID)"
else
    echo "Status: process ended before recording an exit code"
fi

echo "Log: $LOG_FILE"
if [[ "${1:-}" == "--follow" ]]; then
    exec tail -n 80 -f "$LOG_FILE"
fi
tail -n 60 "$LOG_FILE"
