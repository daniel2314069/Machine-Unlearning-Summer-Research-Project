#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$HERE/pipeline.py"
OUTPUT_ROOT="$HERE/outputs/official_repo_primary_v1"
PID_FILE="$OUTPUT_ROOT/detached.pid"
EXIT_FILE="$OUTPUT_ROOT/detached.exit_code"
LATEST_FILE="$OUTPUT_ROOT/detached.latest"

if [[ "${1:-}" == "--worker" ]]; then
    shift
    worker_exit_file="$1"
    shift
    set +e
    "$@"
    worker_exit_code=$?
    printf '%s\n' "$worker_exit_code" > "$worker_exit_file"
    exit "$worker_exit_code"
fi

mkdir -p "$OUTPUT_ROOT"

if [[ -f "$PID_FILE" ]]; then
    previous_pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
        echo "A detached Confuse5 run is already active (PID $previous_pid)." >&2
        echo "Use $HERE/status.sh to inspect it." >&2
        exit 1
    fi
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active Conda environment. Activate MU on the server or py310 on WSL first." >&2
    exit 1
fi

PYTHON_BIN="$(command -v python)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "Could not resolve an executable Python from the active environment." >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$OUTPUT_ROOT/detached_${timestamp}.log"
printf 'running\n' > "$EXIT_FILE"

if [[ "$#" -eq 0 ]]; then
    pipeline_args=(all --skip-existing)
else
    pipeline_args=("$@")
fi
stage="${pipeline_args[0]}"

nohup bash "$0" --worker "$EXIT_FILE" \
    "$PYTHON_BIN" "$PIPELINE" \
    "${pipeline_args[@]}" \
    > "$LOG_FILE" 2>&1 < /dev/null &

worker_pid=$!
printf '%s\n' "$worker_pid" > "$PID_FILE"
{
    printf 'pid=%s\n' "$worker_pid"
    printf 'environment=%s\n' "${CONDA_DEFAULT_ENV:-unknown}"
    printf 'python=%s\n' "$PYTHON_BIN"
    printf 'stage=%s\n' "$stage"
    printf 'log=%s\n' "$LOG_FILE"
    printf 'started_utc=%s\n' "$timestamp"
} > "$LATEST_FILE"

echo "Detached Confuse5 run started."
echo "PID: $worker_pid"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Stage: $stage"
echo "Log: $LOG_FILE"
echo "Status: $HERE/status.sh"
echo "It is now safe to close this terminal or disconnect the VPN."
