#!/usr/bin/env bash
set -euo pipefail

TRANSFER_ROOT="/home/tslin/Documents/jupyter_data/anLi/tmp"
RUNS_ROOT="$TRANSFER_ROOT/storage_inventory_runs"
LATEST_FILE="$TRANSFER_ROOT/storage_inventory_state/latest_run"

if [[ ! -s "$LATEST_FILE" ]]; then
    echo "No storage inventory run has been registered." >&2
    exit 2
fi
RUN_DIR="$(tr -d '\n' < "$LATEST_FILE")"
if [[ "$RUN_DIR" != "$RUNS_ROOT"/* || ! -d "$RUN_DIR" ]]; then
    echo "Invalid latest-run pointer: $RUN_DIR" >&2
    exit 2
fi

PID="unknown"
if [[ -s "$RUN_DIR/pid" ]]; then
    PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
fi

STATUS="unknown"
STATUS_EXIT=0
if [[ -f "$RUN_DIR/COMPLETED" && -s "$RUN_DIR/exit_code" && "$(tr -d '[:space:]' < "$RUN_DIR/exit_code")" == "0" ]]; then
    STATUS="completed"
elif [[ -f "$RUN_DIR/FAILED" || -s "$RUN_DIR/exit_code" ]]; then
    STATUS="failed"
    STATUS_EXIT=1
elif [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
    STATUS="running"
else
    STATUS="stale-or-crashed"
    STATUS_EXIT=1
fi

echo "Status: $STATUS"
echo "PID: $PID"
echo "Output: $RUN_DIR"
echo "Log: $RUN_DIR/server.log"
if [[ -s "$RUN_DIR/stage" ]]; then
    echo "Stage: $(tr -d '\n' < "$RUN_DIR/stage")"
fi
if [[ -s "$RUN_DIR/exit_code" ]]; then
    echo "Exit code: $(tr -d '[:space:]' < "$RUN_DIR/exit_code")"
fi
if [[ -s "$RUN_DIR/progress.json" ]]; then
    echo
    echo "Progress:"
    sed -n '1,160p' "$RUN_DIR/progress.json"
fi
if [[ -s "$RUN_DIR/summary.md" ]]; then
    echo
    echo "Summary preview:"
    sed -n '1,80p' "$RUN_DIR/summary.md"
fi
echo
echo "Recent log:"
tail -n 60 "$RUN_DIR/server.log" || true
exit "$STATUS_EXIT"
