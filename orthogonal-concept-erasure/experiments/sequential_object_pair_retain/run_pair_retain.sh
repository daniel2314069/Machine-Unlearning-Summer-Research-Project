#!/usr/bin/env bash
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$HERE/.run"
PID_FILE="$RUN_DIR/pid"
EXIT_FILE="$RUN_DIR/exit_code"
LOG_FILE="$RUN_DIR/run.log"
RUNNER="$HERE/run_pair_retain.py"

usage() {
  echo "Usage: $0 [--start] [--allow-downloads] | --preflight [--allow-downloads] | --status | --worker [--allow-downloads]"
}

require_project_python() {
  if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    echo "Activate the correct project Conda environment before launch (GPU server: MU)." >&2
    exit 1
  fi
  python_bin="$(command -v python || true)"
  if [[ -z "$python_bin" ]]; then
    echo "No python executable in the active environment" >&2
    exit 1
  fi
}

preflight() {
  require_project_python
  "$python_bin" -u "$RUNNER" preflight "$@"
  echo "Preflight complete. No checkpoints were edited and no images were generated."
}

status() {
  echo "PID file: $PID_FILE"
  if [[ -f "$PID_FILE" ]]; then
    pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Process: running (PID $pid)"
    else
      echo "Process: not running (recorded PID ${pid:-invalid})"
    fi
  else
    echo "Process: not started"
  fi
  if [[ -f "$EXIT_FILE" ]]; then
    echo "Exit code: $(tr -d '[:space:]' < "$EXIT_FILE")"
  fi
  state="$HERE/outputs/sequential_oce_pair_retain_v1/run_state.json"
  if [[ -f "$state" ]]; then
    echo "State: $state"
    if command -v jq >/dev/null 2>&1; then
      jq '{status,phase,pair,order,variant,stage,current_class,completed_ordered_pairs,completed_generation_images,total_generation_images,completed_evaluators,total_evaluators,updated_at,error}' "$state"
    else
      sed -n -E '/"(status|phase|pair|order|variant|stage|current_class|completed_ordered_pairs|completed_generation_images|total_generation_images|completed_evaluators|total_evaluators|updated_at|error)"/p' "$state"
    fi
  else
    echo "State: not created"
  fi
  echo "Latest log: $LOG_FILE"
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 25 "$LOG_FILE"
  fi
}

start() {
  mkdir -p "$RUN_DIR"
  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Refusing duplicate launch: PID $old_pid is still running" >&2
      exit 1
    fi
  fi
  require_project_python
  rm -f "$EXIT_FILE"
  nohup "$0" --worker "$@" </dev/null >>"$LOG_FILE" 2>&1 &
  child_pid=$!
  echo "$child_pid" > "$PID_FILE"
  echo "Started detached run with PID $child_pid"
  echo "Log: $LOG_FILE"
  echo "Status: $0 --status"
  echo "The SSH terminal may now be closed; nohup protects the worker from hangup."
}

worker() {
  shift
  python_bin="$(command -v python || true)"
  if [[ -z "$python_bin" ]]; then
    echo "No python executable in inherited environment" >&2
    echo 127 > "$EXIT_FILE"
    exit 127
  fi
  "$python_bin" -u "$RUNNER" run "$@"
  rc=$?
  echo "$rc" > "$EXIT_FILE"
  exit "$rc"
}

mode="${1:---start}"
case "$mode" in
  --start)
    if [[ $# -gt 0 ]]; then
      shift
    fi
    start "$@"
    ;;
  --status)
    status
    ;;
  --preflight)
    shift
    preflight "$@"
    ;;
  --worker)
    worker "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
