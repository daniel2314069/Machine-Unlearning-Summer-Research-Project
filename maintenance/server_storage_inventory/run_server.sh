#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
EXPECTED_ROOT="/home/tslin/Documents/jupyter_data/anLi/machine_unlearning"
TRANSFER_ROOT="/home/tslin/Documents/jupyter_data/anLi/tmp"
RUNS_ROOT="$TRANSFER_ROOT/storage_inventory_runs"
STATE_ROOT="$TRANSFER_ROOT/storage_inventory_state"
WORKER="$SCRIPT_DIR/inventory_worker.py"
ENTRYPOINT="$SCRIPT_DIR/worker_entrypoint.sh"

if [[ "$REPO_ROOT" != "$EXPECTED_ROOT" ]]; then
    echo "Refusing to scan an unexpected path: $REPO_ROOT" >&2
    echo "Expected GPU-server repository: $EXPECTED_ROOT" >&2
    exit 2
fi
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
for REQUIRED in "$WORKER" "$ENTRYPOINT"; do
    if [[ ! -f "$REQUIRED" ]]; then
        echo "Missing required inventory source: $REQUIRED" >&2
        exit 2
    fi
done

mkdir -p "$RUNS_ROOT" "$STATE_ROOT"
LATEST_FILE="$STATE_ROOT/latest_run"
if [[ -s "$LATEST_FILE" ]]; then
    PREVIOUS_RUN="$(tr -d '\n' < "$LATEST_FILE")"
    if [[ "$PREVIOUS_RUN" == "$RUNS_ROOT"/* && -s "$PREVIOUS_RUN/pid" && ! -f "$PREVIOUS_RUN/exit_code" ]]; then
        PREVIOUS_PID="$(tr -d '[:space:]' < "$PREVIOUS_RUN/pid")"
        if [[ "$PREVIOUS_PID" =~ ^[0-9]+$ ]] && kill -0 "$PREVIOUS_PID" 2>/dev/null; then
            echo "A storage inventory is already running (PID $PREVIOUS_PID)." >&2
            echo "Use: maintenance/server_storage_inventory/status_server.sh" >&2
            exit 3
        fi
    fi
fi

RUN_ID="inventory_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$RUNS_ROOT/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
    exit 3
fi
mkdir "$RUN_DIR"

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GIT_BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=normal > "$RUN_DIR/git_status_porcelain.txt"
{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'root=%s\n' "$REPO_ROOT"
    printf 'python=%s\n' "$PYTHON_BIN"
    printf 'conda_default_env=%s\n' "${CONDA_DEFAULT_ENV:-}"
    printf 'git_commit=%s\n' "$GIT_COMMIT"
    printf 'git_branch=%s\n' "$GIT_BRANCH"
    printf 'launcher=%s\n' "$SCRIPT_DIR/run_server.sh"
} > "$RUN_DIR/launch_metadata.txt"

LOG_FILE="$RUN_DIR/server.log"
: > "$LOG_FILE"
printf '%s\n' "$RUN_DIR" > "$LATEST_FILE"

nohup "$ENTRYPOINT" \
    "$PYTHON_BIN" \
    "$WORKER" \
    "$REPO_ROOT" \
    "$RUN_DIR" \
    "$RUN_ID" \
    "$GIT_COMMIT" \
    "$GIT_BRANCH" \
    </dev/null >"$LOG_FILE" 2>&1 &
JOB_PID=$!
printf '%s\n' "$JOB_PID" > "$RUN_DIR/pid"
printf '%s\n' "$LOG_FILE" > "$RUN_DIR/log_path"

sleep 2
if ! kill -0 "$JOB_PID" 2>/dev/null; then
    EXIT_CODE="unknown"
    if [[ -s "$RUN_DIR/exit_code" ]]; then
        EXIT_CODE="$(tr -d '[:space:]' < "$RUN_DIR/exit_code")"
    fi
    if [[ "$EXIT_CODE" == "0" && -f "$RUN_DIR/COMPLETED" ]]; then
        echo "Read-only storage inventory completed during the startup check."
        echo "Output: $RUN_DIR"
        echo "Package: maintenance/server_storage_inventory/audit_server_storage.sh --package"
        exit 0
    fi
    echo "Storage inventory exited during startup (exit $EXIT_CODE)." >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
fi

echo "Read-only full-repository storage inventory started in the background."
echo "PID: $JOB_PID"
echo "Scan root: $REPO_ROOT"
echo "Output: $RUN_DIR"
echo "Log: $LOG_FILE"
echo "You may safely disconnect from SSH."
echo "Status: maintenance/server_storage_inventory/status_server.sh"
