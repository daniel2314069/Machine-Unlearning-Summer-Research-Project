#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  if [[ ! -f "$STATE_DIR/latest_run" ]]; then
    echo "ERROR: no run path supplied and no latest run is recorded" >&2
    exit 2
  fi
  RUN_DIR="$(tr -d '\r\n' < "$STATE_DIR/latest_run")"
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run directory does not exist: $RUN_DIR" >&2
  exit 2
fi

PID="unknown"
[[ -f "$RUN_DIR/pid" ]] && PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
EXIT_CODE="pending"
[[ -f "$RUN_DIR/exit_code" ]] && EXIT_CODE="$(tr -d '[:space:]' < "$RUN_DIR/exit_code")"
if [[ -f "$RUN_DIR/COMPLETED" && "$EXIT_CODE" == "0" ]]; then
  STATUS="completed"
elif [[ -f "$RUN_DIR/FAILED" || "$EXIT_CODE" != "pending" ]]; then
  STATUS="failed"
elif [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
  STATUS="running"
else
  STATUS="unknown/stale"
fi

echo "status: $STATUS"
echo "pid: $PID"
echo "exit_code: $EXIT_CODE"
echo "output: $RUN_DIR"
echo "log: $RUN_DIR/server.log"
echo "summary: $RUN_DIR/results/summary.md"
if [[ -f "$RUN_DIR/started_at_utc" ]]; then
  echo "started_at_utc: $(tr -d '\r\n' < "$RUN_DIR/started_at_utc")"
fi
if [[ -f "$RUN_DIR/finished_at_utc" ]]; then
  echo "finished_at_utc: $(tr -d '\r\n' < "$RUN_DIR/finished_at_utc")"
fi
echo
echo "Recent log:"
tail -n 60 "$RUN_DIR/server.log" 2>/dev/null || echo "(log not created yet)"
