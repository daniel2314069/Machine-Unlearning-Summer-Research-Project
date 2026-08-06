#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIQUE_METRICS="$HERE/coco10k_metrics/methods/unique_anchor/first10000/metrics.json"
STATE="$HERE/gcd_metrics/automation_state.json"
LOG="$HERE/logs/gcd_after_unique10k.log"
LOCK="$HERE/logs/gcd_after_unique10k.lock"

mkdir -p "$HERE/gcd_metrics" "$HERE/logs"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "A GCD-after-10k watcher is already active." >&2
  exit 0
fi

write_state() {
  local status="$1"
  local detail="$2"
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    printf '{\n'
    printf '  "status": "%s",\n' "$status"
    printf '  "updated_at": "%s",\n' "$now"
    printf '  "watcher_pid": %s,\n' "$$"
    printf '  "detail": "%s",\n' "$detail"
    printf '  "trigger": "%s",\n' "$UNIQUE_METRICS"
    printf '  "log": "%s"\n' "$LOG"
    printf '}\n'
  } >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}

write_state "waiting_for_unique_first10k" \
  "Polling one metrics path every 60 seconds; no Python model and no GPU are loaded."
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Waiting for unique first-10k metrics." >>"$LOG"

while :; do
  if [[ -s "$UNIQUE_METRICS" ]] \
    && grep -q '"status": "complete"' "$UNIQUE_METRICS" \
    && grep -q '"prompt_count": 10000' "$UNIQUE_METRICS" \
    && grep -q '"unique_anchor": 10000' "$UNIQUE_METRICS"; then
    break
  fi
  sleep 60
done

write_state "waiting_for_coco_process_exit" \
  "Unique metrics exist; waiting for the COCO process to release its resources."
while pgrep -f '[r]un_experiment.py coco' >/dev/null; do
  sleep 10
done

write_state "running_gcd" \
  "COCO has exited; preparing the isolated official GCD runtime and evaluating 3000 celebrity images."
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] COCO complete; starting GCD." >>"$LOG"

if "$HERE/run_gcd_after_unique10k.sh" >>"$LOG" 2>&1; then
  write_state "complete" \
    "GCD outputs were written and validated: metrics JSON, prediction CSV, per-celebrity CSV, and Markdown summary."
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GCD complete and validated." >>"$LOG"
else
  exit_code=$?
  write_state "failed" \
    "GCD setup or evaluation failed; see the log. COCO artifacts were not modified or deleted."
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GCD failed with exit code $exit_code." >>"$LOG"
  exit "$exit_code"
fi
