#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$SCRIPT_DIR/.run"
LOG_DIR="$RUN_DIR/logs"
PID_FILE="$RUN_DIR/last.pid"
EXIT_FILE="$RUN_DIR/last.exit"
LOG_POINTER="$RUN_DIR/last.log"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    echo "Refusing to start outside a project Conda environment." >&2
    echo "Activate the environment for this machine first (GPU server: MU; WSL: py310)." >&2
    exit 2
fi

PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "No python executable was found in the active environment." >&2
    exit 2
fi

mkdir -p "$LOG_DIR"

if [[ -s "$PID_FILE" && ! -s "$EXIT_FILE" ]]; then
    EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "A sequential OCE job is already running with PID $EXISTING_PID." >&2
        echo "Use: bash $SCRIPT_DIR/status_server.sh" >&2
        exit 3
    fi
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/sequential_oce_${TIMESTAMP}.log"
: > "$EXIT_FILE"

nohup bash "$SCRIPT_DIR/run_worker.sh" "$PYTHON_BIN" "$EXIT_FILE" "$@" \
    </dev/null >"$LOG_FILE" 2>&1 &
JOB_PID=$!

printf '%s\n' "$JOB_PID" > "$PID_FILE"
printf '%s\n' "$LOG_FILE" > "$LOG_POINTER"

# Catch immediate configuration/import failures while keeping the actual job detached.
sleep 1
if ! kill -0 "$JOB_PID" 2>/dev/null; then
    EXIT_CODE="$(tr -d '[:space:]' < "$EXIT_FILE")"
    echo "The job exited during startup (exit ${EXIT_CODE:-unknown})." >&2
    tail -n 40 "$LOG_FILE" >&2 || true
    exit "${EXIT_CODE:-1}"
fi

echo "Sequential OCE job started in the background."
echo "PID: $JOB_PID"
echo "Log: $LOG_FILE"
echo "You may now close this terminal."
echo "Status: bash $SCRIPT_DIR/status_server.sh"
echo "Follow log: bash $SCRIPT_DIR/status_server.sh --follow"
