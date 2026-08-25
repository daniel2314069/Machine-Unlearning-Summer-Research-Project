#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  [[ -f "$STATE_DIR/latest_run" ]] || { echo "ERROR: no recorded run" >&2; exit 2; }
  RUN_DIR="$(tr -d '\r\n' < "$STATE_DIR/latest_run")"
fi
[[ -d "$RUN_DIR" ]] || { echo "ERROR: run directory missing: $RUN_DIR" >&2; exit 2; }
PID="unknown"; [[ -f "$RUN_DIR/pid" ]] && PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
EXIT_CODE="pending"; [[ -f "$RUN_DIR/exit_code" ]] && EXIT_CODE="$(tr -d '[:space:]' < "$RUN_DIR/exit_code")"
if [[ -f "$RUN_DIR/COMPLETED" && "$EXIT_CODE" == "0" ]]; then
  STATUS="completed"
elif [[ -f "$RUN_DIR/FAILED" || "$EXIT_CODE" != "pending" ]]; then
  STATUS="failed"
elif [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
  STATUS="running"
else
  STATUS="unknown/stale"
fi
PROFILE="$(tr -d '\r\n' < "$RUN_DIR/profile")"
if [[ "$PROFILE" == "formal" ]]; then EXPECTED=60000; else EXPECTED=60; fi
SCORED=0
while IFS= read -r scores; do
  rows=$(( $(wc -l < "$scores") - 1 ))
  (( rows > 0 )) && SCORED=$((SCORED + rows))
done < <(find "$RUN_DIR/seeds" -path '*/evaluation/*/scores.csv' -type f 2>/dev/null | sort)
echo "status: $STATUS"
echo "pid: $PID"
echo "exit_code: $EXIT_CODE"
echo "score_progress: $SCORED/$EXPECTED"
echo "output: $RUN_DIR"
echo "log: $RUN_DIR/server.log"
echo "summary: $RUN_DIR/results/summary.md"
if [[ -f "$RUN_DIR/archive_manifest.json" ]]; then
  sed -n 's/^  "archive": "\([^"]*\)".*/archive: \1/p; s/^  "sha256": "\([^"]*\)".*/archive_sha256: \1/p' "$RUN_DIR/archive_manifest.json"
else
  echo "archive: pending"
fi
if [[ -f "$RUN_DIR/cleanup_manifest.json" ]]; then
  sed -n 's/^  "status": "\([^"]*\)".*/image_cleanup: \1/p; s/^  "deleted_png_files": \([0-9]*\).*/deleted_png_files: \1/p' "$RUN_DIR/cleanup_manifest.json"
else
  echo "image_cleanup: pending (only after verified archive)"
fi
echo
echo "Recent log:"
tail -n 80 "$RUN_DIR/server.log" 2>/dev/null || echo "(log not created yet)"
