#!/usr/bin/env bash
set -uo pipefail

OUTPUT_DIR="/teamspace/studios/this_studio/runs/sequential_oce_celebrity_long_horizon_v1"
RECOVERY_DIR="$OUTPUT_DIR/.joint_recovery"
LOG_FILE="$OUTPUT_DIR/logs/joint_recovery_watchdog.log"
LIGHTNING="/home/zeus/miniconda3/envs/cloudspace/bin/lightning"
STUDIO="mu"
TEAMSPACE="daniel941113-org/mu"
DEADLINE_SECONDS=3600

mkdir -p "$RECOVERY_DIR" "$(dirname "$LOG_FILE")"
printf '%s\n' "$$" > "$RECOVERY_DIR/watchdog.pid"
start_epoch="$(date +%s)"
deadline_epoch="$((start_epoch + DEADLINE_SECONDS))"
printf '[%s] recovery watchdog start pid=%s deadline_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
  "$(date -u -d "@$deadline_epoch" +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_FILE"

reason=""
while [[ -z "$reason" ]]; do
  worker_pid="$(tr -cd '0-9' < "$RECOVERY_DIR/worker.pid" 2>/dev/null || true)"
  if [[ -f "$OUTPUT_DIR/final_validation.json" ]] && \
      grep -q '"status"[[:space:]]*:[[:space:]]*"complete"' "$OUTPUT_DIR/final_validation.json"; then
    reason="normal_complete"
  elif [[ -f "$RECOVERY_DIR/worker.exit_code" ]] && ! kill -0 "$worker_pid" 2>/dev/null; then
    reason="worker_exit_$(tr -cd '0-9' < "$RECOVERY_DIR/worker.exit_code")"
  elif (( $(date +%s) >= deadline_epoch )); then
    reason="hard_deadline_${DEADLINE_SECONDS}_seconds"
    kill -TERM "$worker_pid" 2>/dev/null || true
    sleep 10
  else
    sleep 10
  fi
done

printf '[%s] stop reason=%s; flushing and stopping whole Studio\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$reason" >> "$LOG_FILE"
printf '%s\n' "$reason" > "$RECOVERY_DIR/watchdog.stop_reason"
sync
"$LIGHTNING" studio stop --name "$STUDIO" --teamspace "$TEAMSPACE" >> "$LOG_FILE" 2>&1
