#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
STATE_ROOT="$HERE/outputs/official_repo_primary_v1/baseline_qualified_primary_v1"
FORMAL_ROOT="$STATE_ROOT/formal"
PID_FILE="$STATE_ROOT/detached.pid"
LATEST_FILE="$STATE_ROOT/detached.latest"
EXIT_FILE="$STATE_ROOT/detached.exit_code"

require_mu() {
    if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
        echo "Activate the GPU server Conda environment MU first." >&2
        exit 1
    fi
    if [[ -z "$(command -v python)" ]]; then
        echo "Could not resolve python from the active MU environment." >&2
        exit 1
    fi
}

show_status() {
    local pid=""
    local latest=""
    local exit_code=""
    local completed_jobs="0"
    local retained_images="0"
    local current_job="unavailable"
    [[ -f "$PID_FILE" ]] && pid="$(tr -d '[:space:]' < "$PID_FILE")"
    [[ -f "$LATEST_FILE" ]] && latest="$(tr -d '\n' < "$LATEST_FILE")"
    [[ -f "$EXIT_FILE" ]] && exit_code="$(tr -d '[:space:]' < "$EXIT_FILE")"
    echo "PID: ${pid:-unknown}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Process status: RUNNING"
    else
        echo "Process status: NOT RUNNING"
    fi
    if [[ -n "$exit_code" ]]; then
        echo "Exit status: $exit_code"
    else
        echo "Exit status: pending or unavailable"
    fi
    echo "Log: ${latest:-unknown}"
    if [[ -d "$FORMAL_ROOT/evaluations/shards" ]]; then
        completed_jobs="$(find "$FORMAL_ROOT/evaluations/shards" -type f -name '*.json' | wc -l | tr -d '[:space:]')"
    fi
    if [[ -d "$FORMAL_ROOT/images" ]]; then
        retained_images="$(find "$FORMAL_ROOT/images" -type f -name '*.png' | wc -l | tr -d '[:space:]')"
    fi
    if [[ -f "$FORMAL_ROOT/progress.json" ]]; then
        current_job="$(sed -n 's/^[[:space:]]*"current_job": "\([^"]*\)",*[[:space:]]*$/\1/p' "$FORMAL_ROOT/progress.json" | head -n 1)"
        [[ -z "$current_job" ]] && current_job="aggregate or complete"
    fi
    echo "Completed jobs: $completed_jobs / 45"
    echo "Current job: $current_job"
    echo "Current job PNGs: $retained_images / 500"
    if [[ -f "$FORMAL_ROOT/aggregates/summary.json" ]]; then
        echo "Aggregate summary: COMPLETE"
    else
        echo "Aggregate summary: missing"
    fi
    if [[ -n "$latest" && -f "$latest" ]]; then
        echo "Latest log lines:"
        tail -n 20 "$latest"
    fi
}

run_worker() {
    "$(command -v python)" "$HERE/qualified_primary.py"
}

case "${1:-}" in
    --detached-worker)
        require_mu
        set +e
        run_worker
        worker_status="$?"
        printf '%s\n' "$worker_status" > "$EXIT_FILE"
        exit "$worker_status"
        ;;
    --foreground)
        require_mu
        run_worker
        ;;
    --status)
        show_status
        ;;
    "")
        require_mu
        mkdir -p "$STATE_ROOT"
        if [[ -f "$PID_FILE" ]]; then
            existing_pid="$(tr -d '[:space:]' < "$PID_FILE")"
            if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
                echo "Qualified primary run is already running with PID $existing_pid"
                show_status
                exit 0
            fi
        fi
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        log_path="$STATE_ROOT/detached_${stamp}.log"
        rm -f "$EXIT_FILE"
        printf '%s\n' "$log_path" > "$LATEST_FILE"
        nohup "$SCRIPT_PATH" --detached-worker > "$log_path" 2>&1 < /dev/null &
        child_pid="$!"
        printf '%s\n' "$child_pid" > "$PID_FILE"
        echo "Started detached baseline-qualified primary run."
        echo "PID: $child_pid"
        echo "Log: $log_path"
        echo "Expected workload: 45 jobs / 22,500 new edited images."
        echo "You may close SSH. Check later with:"
        echo "./experiments/confuse5_single_vs_joint/run_qualified_primary.sh --status"
        ;;
    *)
        echo "Usage: $0 [--status|--foreground]" >&2
        exit 2
        ;;
esac
