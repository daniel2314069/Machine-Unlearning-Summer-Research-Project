#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$SCRIPT_DIR/.run"
OUTPUTS_DIR="$SCRIPT_DIR/outputs"

if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
    echo "Refusing to run: activate the GPU-server Conda environment first." >&2
    echo "Expected: conda activate MU" >&2
    exit 2
fi
PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "No python executable was found in the active MU environment." >&2
    exit 2
fi
if [[ ! -f "$OCE_ROOT/Cg.pt" ]]; then
    echo "Missing official OCE preservation statistic: $OCE_ROOT/Cg.pt" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$OUTPUTS_DIR"
PID_FILE="$RUN_DIR/last.pid"
EXIT_FILE="$RUN_DIR/last.exit"
LOG_POINTER="$RUN_DIR/last.log"
OUTPUT_POINTER="$RUN_DIR/last_output"

if [[ -s "$PID_FILE" ]]; then
    EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "An OCE qualification audit is already running (PID $EXISTING_PID)." >&2
        echo "Use: bash $SCRIPT_DIR/status_server.sh" >&2
        exit 3
    fi
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$OUTPUTS_DIR/run_${TIMESTAMP}"
LOG_FILE="$RUN_DIR/audit_${TIMESTAMP}.log"
: > "$EXIT_FILE"

nohup bash "$SCRIPT_DIR/run_worker.sh" \
    "$PYTHON_BIN" "$OUTPUT_DIR" "$EXIT_FILE" "$@" \
    </dev/null >"$LOG_FILE" 2>&1 &
JOB_PID=$!

printf '%s\n' "$JOB_PID" > "$PID_FILE"
printf '%s\n' "$LOG_FILE" > "$LOG_POINTER"
printf '%s\n' "$OUTPUT_DIR" > "$OUTPUT_POINTER"

sleep 1
if ! kill -0 "$JOB_PID" 2>/dev/null; then
    EXIT_CODE="$(tr -d '[:space:]' < "$EXIT_FILE")"
    echo "Audit exited during startup (exit ${EXIT_CODE:-unknown})." >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit "${EXIT_CODE:-1}"
fi

echo "Zero-generation OCE audit started in the background."
echo "PID: $JOB_PID"
echo "Project output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Status: bash $SCRIPT_DIR/status_server.sh"
