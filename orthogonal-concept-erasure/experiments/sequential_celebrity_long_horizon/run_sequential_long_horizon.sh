#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_RUNNER="$SCRIPT_DIR/run_sequential_long_horizon.py"
DEFAULT_OUTPUT="$SCRIPT_DIR/outputs/sequential_oce_celebrity_long_horizon_v1"

mode="${1:---start}"
if [[ $# -gt 0 ]]; then
  shift
fi

resolve_output_dir() {
  local resolved="$DEFAULT_OUTPUT"
  local expect_value=0
  local argument
  for argument in "$@"; do
    if [[ "$expect_value" -eq 1 ]]; then
      resolved="$argument"
      expect_value=0
    elif [[ "$argument" == "--output-dir" ]]; then
      expect_value=1
    elif [[ "$argument" == --output-dir=* ]]; then
      resolved="${argument#--output-dir=}"
    fi
  done
  if [[ "$resolved" != /* ]]; then
    resolved="$(pwd)/$resolved"
  fi
  printf '%s\n' "$resolved"
}

OUTPUT_DIR="$(resolve_output_dir "$@")"
RUN_DIR="$OUTPUT_DIR/.run"
PID_FILE="$RUN_DIR/pid"
EXIT_FILE="$RUN_DIR/exit_code"
LOG_FILE="$OUTPUT_DIR/logs/run.log"
PROJECT_PYTHON=""

require_project_python() {
  if ! command -v python >/dev/null 2>&1; then
    printf 'ERROR: activate this machine\047s project Conda environment first; python is unavailable.\n' >&2
    exit 2
  fi
  if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    printf 'ERROR: activate this machine\047s project Conda environment first; base/system Python is forbidden.\n' >&2
    exit 2
  fi
  PROJECT_PYTHON="$(python -c 'import sys; print(sys.executable)')"
  if [[ -z "$PROJECT_PYTHON" || ! -x "$PROJECT_PYTHON" ]]; then
    printf 'ERROR: active environment did not resolve an executable sys.executable.\n' >&2
    exit 2
  fi
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local recorded_pid
  recorded_pid="$(tr -cd '0-9' < "$PID_FILE")"
  [[ -n "$recorded_pid" ]] || return 1
  kill -0 "$recorded_pid" 2>/dev/null || return 1
  [[ -r "/proc/$recorded_pid/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/$recorded_pid/cmdline")"
  [[ "$command_line" == *"run_sequential_long_horizon.sh"* && "$command_line" == *"--worker"* ]]
}

show_status() {
  local process_status="stopped"
  local recorded_pid="-"
  if [[ -f "$PID_FILE" ]]; then
    recorded_pid="$(tr -cd '0-9' < "$PID_FILE")"
  fi
  if is_running; then
    process_status="running"
  elif [[ -f "$OUTPUT_DIR/final_validation.json" ]] && grep -q '"status": "complete"' "$OUTPUT_DIR/final_validation.json"; then
    process_status="complete"
  elif [[ -f "$EXIT_FILE" ]]; then
    process_status="stopped/resumable (exit $(tr -cd '0-9' < "$EXIT_FILE"))"
  fi
  printf 'process: %s\n' "$process_status"
  printf 'pid: %s\n' "${recorded_pid:--}"
  printf 'output: %s\n' "$OUTPUT_DIR"
  if command -v python >/dev/null 2>&1 && [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && [[ "${CONDA_DEFAULT_ENV:-}" != "base" ]]; then
    "$(python -c 'import sys; print(sys.executable)')" "$PYTHON_RUNNER" status --output-dir "$OUTPUT_DIR" || true
  elif [[ -f "$OUTPUT_DIR/run_state.json" ]]; then
    printf 'run_state: %s\n' "$OUTPUT_DIR/run_state.json"
  fi
  printf 'latest log (%s):\n' "$LOG_FILE"
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 25 "$LOG_FILE"
  else
    printf '(no log yet)\n'
  fi
}

case "$mode" in
  --start)
    require_project_python
    mkdir -p "$RUN_DIR" "$(dirname "$LOG_FILE")"
    if is_running; then
      printf 'Already running with PID %s\n' "$(tr -cd '0-9' < "$PID_FILE")"
      exit 0
    fi
    printf '[%s] detached start requested\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_FILE"
    nohup "$0" --worker "$@" </dev/null >> "$LOG_FILE" 2>&1 &
    worker_pid=$!
    printf '%s\n' "$worker_pid" > "$PID_FILE"
    printf 'Started detached worker PID %s\n' "$worker_pid"
    printf 'Log: %s\n' "$LOG_FILE"
    printf 'You may close SSH. Platform shutdowns still stop the VM; rerun --start after reconnecting.\n'
    ;;
  --worker)
    require_project_python
    mkdir -p "$RUN_DIR" "$(dirname "$LOG_FILE")"
    printf '%s\n' "$$" > "$PID_FILE"
    set +e
    "$PROJECT_PYTHON" "$PYTHON_RUNNER" run "$@"
    exit_code=$?
    set -e
    printf '%s\n' "$exit_code" > "$EXIT_FILE"
    printf '[%s] worker exit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_code"
    exit "$exit_code"
    ;;
  --status)
    show_status
    ;;
  --plan)
    require_project_python
    "$PROJECT_PYTHON" "$PYTHON_RUNNER" plan "$@"
    ;;
  --preflight)
    require_project_python
    "$PROJECT_PYTHON" "$PYTHON_RUNNER" preflight "$@"
    ;;
  --benchmark)
    require_project_python
    "$PROJECT_PYTHON" "$PYTHON_RUNNER" benchmark "$@"
    ;;
  --audit)
    require_project_python
    "$PROJECT_PYTHON" "$PYTHON_RUNNER" audit "$@"
    ;;
  --continue-coco)
    require_project_python
    "$PROJECT_PYTHON" "$PYTHON_RUNNER" continue-coco "$@"
    ;;
  --help|-h)
    printf '%s\n' \
      'Usage:' \
      '  ./run_sequential_long_horizon.sh --plan' \
      '  ./run_sequential_long_horizon.sh --preflight [runner options]' \
      '  ./run_sequential_long_horizon.sh --benchmark --remaining-credits N --gpu-rate N [runner options]' \
      '  ./run_sequential_long_horizon.sh [--start] [runner options]' \
      '  ./run_sequential_long_horizon.sh --status [--output-dir PATH]' \
      '  ./run_sequential_long_horizon.sh --audit [--output-dir PATH]'
    ;;
  *)
    printf 'Unknown mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
