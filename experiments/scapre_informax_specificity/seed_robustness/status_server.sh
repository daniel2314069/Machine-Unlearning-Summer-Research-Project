#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
json_string_field() {
  local key="$1"
  local file="$2"
  sed -n "s/^  \"$key\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\"[[:space:]]*,\{0,1\}[[:space:]]*$/\1/p" "$file" | head -n 1
}
json_number_field() {
  local key="$1"
  local file="$2"
  sed -n "s/^  \"$key\"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\)[[:space:]]*,\{0,1\}[[:space:]]*$/\1/p" "$file" | head -n 1
}
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
CALCULATION_EXIT="pending"
[[ -f "$RUN_DIR/calculation_exit_code" ]] && CALCULATION_EXIT="$(tr -d '[:space:]' < "$RUN_DIR/calculation_exit_code")"
if [[ -f "$RUN_DIR/FINALIZING" && "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
  STATUS="finalizing"
elif [[ -f "$RUN_DIR/COMPLETED" && "$EXIT_CODE" == "0" ]]; then
  STATUS="completed"
elif [[ -f "$RUN_DIR/FAILED" || "$EXIT_CODE" != "pending" ]]; then
  STATUS="failed"
elif [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
  STATUS="running"
else
  STATUS="unknown/stale"
fi

PROFILE="$(tr -d '\r\n' < "$RUN_DIR/profile")"
if [[ "$PROFILE" == "formal" ]]; then
  SEEDS=(20260821 20260822 20260823 20260824)
  EXPECTED=24000
else
  SEEDS=(20260821)
  EXPECTED=20
fi
GENERATED=0
for seed in "${SEEDS[@]}"; do
  for variant in official matched_retain; do
    scores="$RUN_DIR/seeds/$seed/evaluation/$variant/scores.csv"
    if [[ -f "$scores" ]]; then
      rows=$(( $(wc -l < "$scores") - 1 ))
      if (( rows > 0 )); then
        GENERATED=$((GENERATED + rows))
      fi
    fi
  done
done

echo "status: $STATUS"
echo "pid: $PID"
echo "calculation_exit_code: $CALCULATION_EXIT"
echo "exit_code: $EXIT_CODE"
echo "output: $RUN_DIR"
echo "log: $RUN_DIR/server.log"
echo "summary: $RUN_DIR/results/summary.md"
echo "new_image_score_progress: $GENERATED/$EXPECTED"
if [[ -f "$RUN_DIR/archive_manifest.json" ]]; then
  echo "archive: $(json_string_field archive "$RUN_DIR/archive_manifest.json")"
  echo "archive_sha256: $(json_string_field sha256 "$RUN_DIR/archive_manifest.json")"
else
  echo "archive: pending"
fi
if [[ -f "$RUN_DIR/cleanup_manifest.json" ]]; then
  echo "image_cleanup: $(json_string_field status "$RUN_DIR/cleanup_manifest.json")"
  echo "deleted_images: $(json_number_field deleted_files "$RUN_DIR/cleanup_manifest.json")"
  if [[ -f "$RUN_DIR/archive_manifest.json" ]]; then
    echo "cleanup_manifest_download: $(json_string_field archive "$RUN_DIR/archive_manifest.json").cleanup.json"
  fi
else
  echo "image_cleanup: pending (runs only after archive verification)"
fi
if [[ -f "$RUN_DIR/posthoc_finalize_exit_code" ]]; then
  echo "posthoc_finalize_exit_code: $(tr -d '[:space:]' < "$RUN_DIR/posthoc_finalize_exit_code")"
fi
for name in started_at_utc calculation_finished_at_utc finished_at_utc; do
  if [[ -f "$RUN_DIR/$name" ]]; then
    echo "$name: $(tr -d '\r\n' < "$RUN_DIR/$name")"
  fi
done
echo
echo "Recent log:"
tail -n 80 "$RUN_DIR/server.log" 2>/dev/null || echo "(log not created yet)"
