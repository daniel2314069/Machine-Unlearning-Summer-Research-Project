#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PARENT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
RUNS_DIR="$SCRIPT_DIR/runs"
ASSETS="$PARENT/.server/assets_manifest.json"
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
if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
  echo "ERROR: run 'conda activate MU' before launching" >&2
  exit 2
fi
PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python is unavailable in active Conda MU" >&2
  exit 2
fi
for command_name in git tar nohup; do
  command -v "$command_name" >/dev/null || {
    echo "ERROR: required command is unavailable: $command_name" >&2
    exit 2
  }
done
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: run-id contains unsupported characters" >&2
  exit 2
fi

cd "$REPO_ROOT"
if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "ERROR: launch from branch main" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: working tree is dirty before git pull" >&2
  git status --short >&2
  exit 2
fi
git pull --ff-only origin main
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: working tree is dirty after git pull" >&2
  exit 2
fi
REQUIRED_COMMIT="$($PYTHON_BIN "$PARENT/seed_robustness/json_stdlib.py" get "$SCRIPT_DIR/config.json" required_ancestor_commit)"
git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD || {
  echo "ERROR: main does not contain required completed-results commit $REQUIRED_COMMIT" >&2
  exit 2
}
if [[ ! -f "$PARENT/.server/SETUP_COMPLETE" || ! -f "$ASSETS" ]]; then
  echo "ERROR: parent setup_server.sh has not completed" >&2
  exit 2
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

mkdir -p "$STATE_DIR" "$RUNS_DIR"
GLOBAL_LOCK="$STATE_DIR/.launch_lock"
if ! mkdir "$GLOBAL_LOCK" 2>/dev/null; then
  echo "ERROR: another superclass-neutral launcher is active" >&2
  exit 2
fi
trap 'rmdir "$GLOBAL_LOCK" 2>/dev/null || true' EXIT
PRIOR_RUN=""
if [[ "$PROFILE" == "formal" ]]; then
  if [[ -f "$STATE_DIR/latest_successful_smoke" ]]; then
    SMOKE_RUN="$(tr -d '\r\n' < "$STATE_DIR/latest_successful_smoke")"
  elif [[ -f "$STATE_DIR/latest_run" ]]; then
    SMOKE_RUN="$(tr -d '\r\n' < "$STATE_DIR/latest_run")"
  else
    echo "ERROR: run a successful superclass-neutral smoke test first" >&2
    exit 2
  fi
  if [[ "$SMOKE_RUN" != "$RUNS_DIR"/smoke_* || ! -f "$SMOKE_RUN/COMPLETED" || \
        "$(tr -d '[:space:]' < "$SMOKE_RUN/exit_code")" != "0" || \
        ! -f "$SMOKE_RUN/reproducibility/integrity_report.json" || \
        ! -f "$SMOKE_RUN/cleanup_manifest.json" ]]; then
    echo "ERROR: latest run is not a completed, packaged, cleaned smoke test" >&2
    exit 2
  fi
  if [[ "$($PYTHON_BIN "$PARENT/seed_robustness/json_stdlib.py" get "$SMOKE_RUN/reproducibility/integrity_report.json" status)" != "passed" || \
        "$($PYTHON_BIN "$PARENT/seed_robustness/json_stdlib.py" get "$SMOKE_RUN/cleanup_manifest.json" status)" != "passed" ]]; then
    echo "ERROR: smoke integrity or cleanup did not pass" >&2
    exit 2
  fi
  printf '%s\n' "$SMOKE_RUN" > "$STATE_DIR/latest_successful_smoke"
  PRIOR_RUN="$($SCRIPT_DIR/resolve_prior_robustness.sh)"
  "$PYTHON_BIN" "$SCRIPT_DIR/formal_preflight.py" \
    --config "$SCRIPT_DIR/config.json" --prior-run "$PRIOR_RUN" \
    --output "$STATE_DIR/formal_preflight_latest.json"
fi

if [[ -f "$STATE_DIR/active_run" ]]; then
  ACTIVE_DIR="$(tr -d '\r\n' < "$STATE_DIR/active_run")"
  if [[ ! -f "$ACTIVE_DIR/COMPLETED" && ! -f "$ACTIVE_DIR/FAILED" && -f "$ACTIVE_DIR/pid" ]]; then
    ACTIVE_PID="$(tr -d '[:space:]' < "$ACTIVE_DIR/pid")"
    if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
      echo "ERROR: another superclass-neutral run is active: $ACTIVE_DIR" >&2
      exit 2
    fi
  fi
fi

RUN_DIR="$RUNS_DIR/${PROFILE}_${RUN_ID}"
if [[ "$RESUME" == "--resume" && ! -d "$RUN_DIR" ]]; then
  echo "ERROR: resume directory does not exist: $RUN_DIR" >&2
  exit 2
fi
if [[ -d "$RUN_DIR" && "$RESUME" != "--resume" ]]; then
  echo "ERROR: run directory exists; use another id or --resume" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
if [[ -f "$RUN_DIR/pid" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: this run is already active" >&2
    exit 2
  fi
fi
if [[ -f "$RUN_DIR/COMPLETED" ]]; then
  echo "ERROR: completed run will not be relaunched" >&2
  exit 2
fi
for marker in FAILED exit_code calculation_exit_code; do
  if [[ -f "$RUN_DIR/$marker" ]]; then
    mv "$RUN_DIR/$marker" "$RUN_DIR/$marker.previous.$(date -u +'%Y%m%dT%H%M%SZ')"
  fi
done

LOG_PATH="$RUN_DIR/server.log"
printf '%s\n' "$PROFILE" > "$RUN_DIR/profile"
printf '%s\n' "$PYTHON_BIN" > "$RUN_DIR/python_path"
printf '%s\n' "$RUN_DIR" > "$RUN_DIR/output_path"
printf '%s\n' "$LOG_PATH" > "$RUN_DIR/log_path"
printf '%s\n' "$PRIOR_RUN" > "$RUN_DIR/prior_run_path"
if [[ "$PROFILE" == "formal" ]]; then
  cp "$STATE_DIR/formal_preflight_latest.json" "$RUN_DIR/formal_preflight.json"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/started_at_utc"
printf '%q ' "$SCRIPT_DIR/server_worker.sh" "$PROFILE" "$RUN_DIR" "$ASSETS" "$PRIOR_RUN" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

nohup "$SCRIPT_DIR/server_worker.sh" "$PROFILE" "$RUN_DIR" "$ASSETS" "$PRIOR_RUN" \
  </dev/null >>"$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/pid"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/latest_run"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/active_run"
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: worker exited during launch health check" >&2
  tail -n 100 "$LOG_PATH" >&2 || true
  exit 1
fi
echo "Started superclass-neutral $PROFILE run with PID $PID"
echo "Output: $RUN_DIR"
echo "Log: $LOG_PATH"
echo "The worker survived its launch health check; it is safe to disconnect."
echo "Status: $SCRIPT_DIR/status_server.sh '$RUN_DIR'"
