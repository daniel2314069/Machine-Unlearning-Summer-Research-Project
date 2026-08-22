#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  [[ -f "$STATE_DIR/latest_run" ]] || { echo "ERROR: no run is recorded" >&2; exit 2; }
  RUN_DIR="$(tr -d '\r\n' < "$STATE_DIR/latest_run")"
fi
[[ -d "$RUN_DIR" ]] || { echo "ERROR: run directory does not exist: $RUN_DIR" >&2; exit 2; }
json_string() {
  sed -n "s/^  \"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\"[[:space:]]*,\{0,1\}[[:space:]]*$/\1/p" "$2" | head -n 1
}
json_number() {
  sed -n "s/^  \"$1\"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\)[[:space:]]*,\{0,1\}[[:space:]]*$/\1/p" "$2" | head -n 1
}
PID="unknown"; [[ -f "$RUN_DIR/pid" ]] && PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
EXIT_CODE="pending"; [[ -f "$RUN_DIR/exit_code" ]] && EXIT_CODE="$(tr -d '[:space:]' < "$RUN_DIR/exit_code")"
CALCULATION_EXIT="pending"; [[ -f "$RUN_DIR/calculation_exit_code" ]] && CALCULATION_EXIT="$(tr -d '[:space:]' < "$RUN_DIR/calculation_exit_code")"
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
if [[ "$PROFILE" == "formal" ]]; then SEEDS=(20260820 20260821 20260822 20260823 20260824); EXPECTED=15000; else SEEDS=(20260821); EXPECTED=10; fi
PROGRESS=0
for seed in "${SEEDS[@]}"; do
  SCORE="$RUN_DIR/seeds/$seed/evaluation/superclass_neutral/scores.csv"
  if [[ -f "$SCORE" ]]; then
    ROWS=$(( $(wc -l < "$SCORE") - 1 )); (( ROWS > 0 )) && PROGRESS=$((PROGRESS + ROWS))
  fi
done
echo "status: $STATUS"
echo "pid: $PID"
echo "calculation_exit_code: $CALCULATION_EXIT"
echo "exit_code: $EXIT_CODE"
echo "output: $RUN_DIR"
echo "log: $RUN_DIR/server.log"
echo "summary: $RUN_DIR/results/summary.md"
echo "new_superclass_score_progress: $PROGRESS/$EXPECTED"
if [[ -f "$RUN_DIR/qualitative/manifest.csv" ]]; then echo "qualitative_manifest_rows: $(( $(wc -l < "$RUN_DIR/qualitative/manifest.csv") - 1 ))/90"; else echo "qualitative_manifest_rows: pending"; fi
if [[ -f "$RUN_DIR/archive_manifest.json" ]]; then
  echo "archive: $(json_string archive "$RUN_DIR/archive_manifest.json")"
  echo "archive_sha256: $(json_string sha256 "$RUN_DIR/archive_manifest.json")"
else
  echo "archive: pending"
fi
if [[ -f "$RUN_DIR/cleanup_manifest.json" ]]; then
  echo "full_evaluation_image_cleanup: $(json_string status "$RUN_DIR/cleanup_manifest.json")"
  echo "deleted_full_evaluation_images: $(json_number deleted_files "$RUN_DIR/cleanup_manifest.json")"
  echo "qualitative_images_preserved: yes"
else
  echo "full_evaluation_image_cleanup: pending (only after archive verification)"
fi
for name in started_at_utc calculation_finished_at_utc finished_at_utc; do
  [[ -f "$RUN_DIR/$name" ]] && echo "$name: $(tr -d '\r\n' < "$RUN_DIR/$name")"
done
echo
echo "Recent log:"
tail -n 80 "$RUN_DIR/server.log" 2>/dev/null || echo "(log not created yet)"
