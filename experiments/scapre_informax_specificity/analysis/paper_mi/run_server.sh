#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARENT_EXPERIMENT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
RUNS_DIR="$SCRIPT_DIR/runs"
ASSETS="$PARENT_EXPERIMENT/.server/assets_manifest.json"
PROFILE="${1:-}"
RUN_ID="${2:-$(date -u +'%Y%m%dT%H%M%SZ')}"
RESUME="${3:-}"

if [[ "$PROFILE" != "smoke" && "$PROFILE" != "formal" ]]; then
  echo "usage: $0 <smoke|formal> [run-id] [--resume]" >&2
  exit 2
fi
if [[ -n "$RESUME" && "$RESUME" != "--resume" ]]; then
  echo "ERROR: third argument must be --resume" >&2
  exit 2
fi
[[ "${CONDA_DEFAULT_ENV:-}" == "MU" ]] || {
  echo "ERROR: run 'conda activate MU' before launching" >&2
  exit 2
}
PYTHON_BIN="$(command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: python is unavailable in Conda MU" >&2; exit 2; }
for command_name in git nohup tar sha256sum flock; do
  command -v "$command_name" >/dev/null || { echo "ERROR: missing command: $command_name" >&2; exit 2; }
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid run id" >&2; exit 2; }
for required in \
  "$SCRIPT_DIR/server_worker.sh" "$SCRIPT_DIR/ensure_assets.sh" \
  "$SCRIPT_DIR/worker.py" "$SCRIPT_DIR/config.json" \
  "$PARENT_EXPERIMENT/config.json" "$PARENT_EXPERIMENT/setup_server.sh"; do
  [[ -f "$required" ]] || { echo "ERROR: required experiment file is missing: $required" >&2; exit 2; }
done
cd "$REPO_ROOT"
[[ "$(git branch --show-current)" == "main" ]] || { echo "ERROR: launch from main" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "ERROR: working tree must be clean before launch" >&2
  git status --short >&2
  exit 2
}
git pull --ff-only origin main
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "ERROR: working tree became dirty after pull" >&2
  exit 2
}
mkdir -p "$STATE_DIR" "$RUNS_DIR"
if [[ -f "$STATE_DIR/active_run" ]]; then
  ACTIVE="$(tr -d '\r\n' < "$STATE_DIR/active_run")"
  if [[ -f "$ACTIVE/pid" && ! -f "$ACTIVE/COMPLETED" && ! -f "$ACTIVE/FAILED" ]]; then
    ACTIVE_PID="$(tr -d '[:space:]' < "$ACTIVE/pid")"
    if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
      echo "ERROR: another paper-MI run is active: $ACTIVE (PID $ACTIVE_PID)" >&2
      exit 2
    fi
  fi
fi

if [[ "$PROFILE" == "formal" ]]; then
  [[ -f "$STATE_DIR/latest_successful_smoke" ]] || {
    echo "ERROR: formal requires a completed paper-MI smoke run" >&2
    exit 2
  }
  SMOKE_DIR="$(tr -d '\r\n' < "$STATE_DIR/latest_successful_smoke")"
  [[ -f "$SMOKE_DIR/COMPLETED" && -f "$SMOKE_DIR/results/integrity_report.json" ]] || {
    echo "ERROR: recorded smoke run is incomplete" >&2
    exit 2
  }
  "$PYTHON_BIN" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "passed"' \
    "$SMOKE_DIR/results/integrity_report.json"
fi

RUN_DIR="$RUNS_DIR/${PROFILE}_${RUN_ID}"
if [[ "$RESUME" == "--resume" ]]; then
  [[ -d "$RUN_DIR" && ! -f "$RUN_DIR/COMPLETED" ]] || {
    echo "ERROR: run cannot be resumed: $RUN_DIR" >&2
    exit 2
  }
else
  [[ ! -e "$RUN_DIR" ]] || { echo "ERROR: run already exists: $RUN_DIR" >&2; exit 2; }
  mkdir -p "$RUN_DIR"
fi
if [[ -f "$RUN_DIR/pid" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: this run is already active with PID $OLD_PID" >&2
    exit 2
  fi
fi

if [[ "$RESUME" == "--resume" ]]; then
  ATTEMPT_DIR="$RUN_DIR/attempt_history/$(date -u +'%Y%m%dT%H%M%SZ')_$$"
  mkdir -p "$ATTEMPT_DIR"
  for previous in pid server.log started_at_utc FAILED exit_code finished_at_utc; do
    [[ ! -f "$RUN_DIR/$previous" ]] || mv "$RUN_DIR/$previous" "$ATTEMPT_DIR/$previous"
  done
  if [[ ! -f "$RUN_DIR/CALCULATION_COMPLETED" ]]; then
    for previous in calculation_exit_code calculation_finished_at_utc; do
      [[ ! -f "$RUN_DIR/$previous" ]] || mv "$RUN_DIR/$previous" "$ATTEMPT_DIR/$previous"
    done
  fi
fi

printf '%s\n' "$PYTHON_BIN" > "$RUN_DIR/python_path"
printf '%s\n' "$PROFILE" > "$RUN_DIR/profile"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/latest_run"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/active_run"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/started_at_utc"
nohup "$SCRIPT_DIR/server_worker.sh" "$PROFILE" "$RUN_DIR" "$ASSETS" \
  </dev/null >"$RUN_DIR/server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/pid"
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: worker exited during launch health check" >&2
  tail -n 80 "$RUN_DIR/server.log" >&2 || true
  exit 1
fi
echo "Started $PROFILE paper-MI comparison. Safe to disconnect."
echo "PID: $PID"
echo "Output: $RUN_DIR"
echo "Log: $RUN_DIR/server.log"
echo "Status: $SCRIPT_DIR/status_server.sh"
