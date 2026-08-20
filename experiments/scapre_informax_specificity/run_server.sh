#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
RUNS_DIR="$SCRIPT_DIR/runs"
ASSETS="$STATE_DIR/assets_manifest.json"
PROFILE="${1:-}"
RUN_ID="${2:-$(date -u +'%Y%m%dT%H%M%SZ')}"
RESUME="${3:-}"

if [[ "$PROFILE" != "smoke" && "$PROFILE" != "formal" ]]; then
  echo "usage: $0 <smoke|formal> [run-id] [--resume]" >&2
  exit 2
fi
if [[ -n "$RESUME" && "$RESUME" != "--resume" ]]; then
  echo "ERROR: third argument must be --resume when supplied" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
  echo "ERROR: run 'conda activate MU' before launching" >&2
  exit 2
fi
PYTHON_BIN="$(command -v python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python is unavailable in active MU" >&2
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: run-id may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
if [[ ! -f "$STATE_DIR/SETUP_COMPLETE" || ! -f "$ASSETS" ]]; then
  echo "ERROR: run setup_server.sh successfully before launching" >&2
  exit 2
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

mkdir -p "$RUNS_DIR"
RUN_DIR="$RUNS_DIR/${PROFILE}_${RUN_ID}"
if [[ "$RESUME" == "--resume" && ! -d "$RUN_DIR" ]]; then
  echo "ERROR: cannot resume a run directory that does not exist: $RUN_DIR" >&2
  exit 2
fi
if [[ -d "$RUN_DIR" && "$RESUME" != "--resume" ]]; then
  echo "ERROR: run directory already exists; use a new run-id or append --resume" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

if [[ -f "$RUN_DIR/pid" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: this run is already active with PID $OLD_PID" >&2
    exit 2
  fi
fi
if [[ -f "$RUN_DIR/COMPLETED" ]]; then
  echo "ERROR: completed run will not be relaunched: $RUN_DIR" >&2
  exit 2
fi

LOCK_DIR="$RUN_DIR/.launch_lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: another launcher is preparing this run" >&2
  exit 2
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

if [[ -f "$RUN_DIR/FAILED" ]]; then
  mv "$RUN_DIR/FAILED" "$RUN_DIR/FAILED.previous.$(date -u +'%Y%m%dT%H%M%SZ')"
fi
if [[ -f "$RUN_DIR/exit_code" ]]; then
  mv "$RUN_DIR/exit_code" "$RUN_DIR/exit_code.previous.$(date -u +'%Y%m%dT%H%M%SZ')"
fi

LOG_PATH="$RUN_DIR/server.log"
printf '%s\n' "$PROFILE" > "$RUN_DIR/profile"
printf '%s\n' "$PYTHON_BIN" > "$RUN_DIR/python_path"
printf '%s\n' "$RUN_DIR" > "$RUN_DIR/output_path"
printf '%s\n' "$LOG_PATH" > "$RUN_DIR/log_path"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/started_at_utc"
printf '%q ' "$SCRIPT_DIR/server_worker.sh" "$PROFILE" "$RUN_DIR" "$ASSETS" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

nohup "$SCRIPT_DIR/server_worker.sh" "$PROFILE" "$RUN_DIR" "$ASSETS" \
  </dev/null >>"$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/pid"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/latest_run"

sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: worker exited during launch health check" >&2
  tail -n 80 "$LOG_PATH" >&2 || true
  exit 1
fi

echo "Started $PROFILE run with PID $PID"
echo "Output: $RUN_DIR"
echo "Log: $LOG_PATH"
echo "The worker survived its launch health check; it is safe to disconnect."
echo "Status: $SCRIPT_DIR/status_server.sh '$RUN_DIR'"
