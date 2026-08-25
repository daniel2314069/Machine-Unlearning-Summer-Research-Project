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
for command_name in git nohup tar sha256sum; do
  command -v "$command_name" >/dev/null || { echo "ERROR: missing command: $command_name" >&2; exit 2; }
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid run id" >&2; exit 2; }
[[ -f "$ASSETS" && -f "$PARENT_EXPERIMENT/.server/SETUP_COMPLETE" ]] || {
  echo "ERROR: parent ScaPre experiment assets are not set up" >&2
  exit 2
}

cd "$REPO_ROOT"
[[ "$(git branch --show-current)" == "main" ]] || { echo "ERROR: launch from main" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "ERROR: working tree is dirty before pull" >&2
  git status --short >&2
  exit 2
}
git pull --ff-only origin main
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "ERROR: working tree is dirty after pull" >&2; exit 2; }
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'

mkdir -p "$STATE_DIR" "$RUNS_DIR"
if [[ -f "$STATE_DIR/active_run" ]]; then
  ACTIVE_DIR="$(tr -d '\r\n' < "$STATE_DIR/active_run")"
  if [[ -f "$ACTIVE_DIR/pid" && ! -f "$ACTIVE_DIR/COMPLETED" && ! -f "$ACTIVE_DIR/FAILED" ]]; then
    ACTIVE_PID="$(tr -d '[:space:]' < "$ACTIVE_DIR/pid")"
    if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
      echo "ERROR: another alpha-control run is active: $ACTIVE_DIR (PID $ACTIVE_PID)" >&2
      exit 2
    fi
  fi
fi

OFFICIAL_REFERENCE=""
if [[ "$PROFILE" == "formal" ]]; then
  [[ -f "$STATE_DIR/latest_successful_smoke" ]] || {
    echo "ERROR: formal requires a completed smoke" >&2
    exit 2
  }
  SMOKE_DIR="$(tr -d '\r\n' < "$STATE_DIR/latest_successful_smoke")"
  [[ -f "$SMOKE_DIR/COMPLETED" && -f "$SMOKE_DIR/results/integrity_report.json" ]] || {
    echo "ERROR: recorded smoke is incomplete" >&2
    exit 2
  }
  "$PYTHON_BIN" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "passed"' "$SMOKE_DIR/results/integrity_report.json"
  OFFICIAL_REFERENCE="$($SCRIPT_DIR/resolve_official_reference.sh)"
fi

RUN_DIR="$RUNS_DIR/${PROFILE}_${RUN_ID}"
if [[ "$RESUME" == "--resume" ]]; then
  [[ -d "$RUN_DIR" ]] || { echo "ERROR: resume directory does not exist: $RUN_DIR" >&2; exit 2; }
else
  [[ ! -e "$RUN_DIR" ]] || { echo "ERROR: run directory already exists: $RUN_DIR" >&2; exit 2; }
  mkdir -p "$RUN_DIR"
fi
[[ ! -f "$RUN_DIR/COMPLETED" ]] || { echo "ERROR: completed run cannot be relaunched" >&2; exit 2; }
if [[ -f "$RUN_DIR/pid" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: run is already active with PID $OLD_PID" >&2
    exit 2
  fi
fi

printf '%s\n' "$PYTHON_BIN" > "$RUN_DIR/python_path"
printf '%s\n' "$PROFILE" > "$RUN_DIR/profile"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/latest_run"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/active_run"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/started_at_utc"
nohup "$SCRIPT_DIR/server_worker.sh" "$PROFILE" "$RUN_DIR" "$ASSETS" "$OFFICIAL_REFERENCE" \
  </dev/null >"$RUN_DIR/server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/pid"
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: worker exited during launch health check" >&2
  tail -n 80 "$RUN_DIR/server.log" >&2 || true
  exit 1
fi
if [[ "$PROFILE" == "smoke" ]]; then
  printf '%s\n' "$RUN_DIR" > "$STATE_DIR/latest_successful_smoke_candidate"
fi
echo "Started $PROFILE alpha-control run. Safe to disconnect."
echo "PID: $PID"
echo "Output: $RUN_DIR"
echo "Log: $RUN_DIR/server.log"
echo "Status: $SCRIPT_DIR/status_server.sh"
