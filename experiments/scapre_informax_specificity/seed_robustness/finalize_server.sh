#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <failed-formal-run-dir>" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNS_DIR="$(cd "$SCRIPT_DIR/runs" 2>/dev/null && pwd || true)"
RUN_DIR="$(cd "$1" 2>/dev/null && pwd || true)"
ELIGIBLE_COMMIT="9ca7b5e9c4ab626027fb8fe0bd32fca51e8faf89"

if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
  echo "ERROR: run 'conda activate MU' before finalizing" >&2
  exit 2
fi
PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python is unavailable in active Conda MU" >&2
  exit 2
fi
if [[ -z "$RUNS_DIR" || -z "$RUN_DIR" || "$RUN_DIR" != "$RUNS_DIR"/formal_* ]]; then
  echo "ERROR: target must be one explicit formal run under $SCRIPT_DIR/runs" >&2
  exit 2
fi

cd "$REPO_ROOT"
if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "ERROR: finalization must run from branch main" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: working tree is dirty; refusing post-hoc finalization" >&2
  git status --short >&2
  exit 2
fi
git pull --ff-only origin main
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: working tree became dirty after pull" >&2
  exit 2
fi

if [[ ! -f "$RUN_DIR/FAILED" || ! -f "$RUN_DIR/exit_code" || \
      ! -f "$RUN_DIR/run_manifest.json" ]]; then
  echo "ERROR: target is not the expected failed aggregation run" >&2
  exit 2
fi
CURRENT_EXIT="$(tr -d '[:space:]' < "$RUN_DIR/exit_code")"
if [[ ! "$CURRENT_EXIT" =~ ^[0-9]+$ || "$CURRENT_EXIT" == "0" ]]; then
  echo "ERROR: target does not have a non-zero failed exit code" >&2
  exit 2
fi
GENERATION_COMMIT="$("$PYTHON_BIN" "$SCRIPT_DIR/json_stdlib.py" get "$RUN_DIR/run_manifest.json" git_commit)"
if [[ "$GENERATION_COMMIT" != "$ELIGIBLE_COMMIT" ]]; then
  echo "ERROR: run generation commit is not eligible: $GENERATION_COMMIT" >&2
  exit 2
fi
if [[ -f "$RUN_DIR/archive_manifest.json" && \
      ! -f "$RUN_DIR/reproducibility/posthoc_finalization.json" ]]; then
  echo "ERROR: an archive exists without audited post-hoc provenance" >&2
  exit 2
fi
if [[ -f "$RUN_DIR/FINALIZING" && -f "$RUN_DIR/pid" ]]; then
  ACTIVE_PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
  if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    echo "ERROR: post-hoc finalization is already running with PID $ACTIVE_PID" >&2
    exit 2
  fi
fi
if [[ -f "$RUN_DIR/pid" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "$RUN_DIR/pid")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: the original run process is still active with PID $OLD_PID" >&2
    exit 2
  fi
fi
for seed in 20260820 20260821 20260822 20260823 20260824; do
  for variant in official matched_retain; do
    SCORES="$RUN_DIR/seeds/$seed/evaluation/$variant/scores.csv"
    if [[ ! -f "$RUN_DIR/seeds/$seed/evaluation/$variant/COMPLETED" || \
          ! -f "$SCORES" || "$(wc -l < "$SCORES" | tr -d ' ')" != "3001" ]]; then
      echo "ERROR: incomplete evaluation for seed=$seed variant=$variant" >&2
      exit 2
    fi
  done
done

if [[ -f "$RUN_DIR/pid" && ! -f "$RUN_DIR/pid.original_failure" ]]; then
  cp "$RUN_DIR/pid" "$RUN_DIR/pid.original_failure"
fi
printf 'post-hoc finalization in progress\n' > "$RUN_DIR/FINALIZING"
printf '%q ' "$SCRIPT_DIR/posthoc_finalize_worker.sh" "$RUN_DIR" > "$RUN_DIR/posthoc_command.txt"
printf '\n' >> "$RUN_DIR/posthoc_command.txt"
nohup "$SCRIPT_DIR/posthoc_finalize_worker.sh" "$RUN_DIR" \
  </dev/null >>"$RUN_DIR/server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/pid"
printf '%s\n' "$RUN_DIR" > "$SCRIPT_DIR/.server/active_run"
printf '%s\n' "$RUN_DIR" > "$SCRIPT_DIR/.server/latest_run"

sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  if [[ -f "$RUN_DIR/COMPLETED" && "$(tr -d '[:space:]' < "$RUN_DIR/exit_code")" == "0" ]]; then
    echo "Post-hoc finalization completed during the launch check."
    echo "Status: $SCRIPT_DIR/status_server.sh '$RUN_DIR'"
    exit 0
  fi
  echo "ERROR: post-hoc worker exited during launch health check" >&2
  tail -n 120 "$RUN_DIR/server.log" >&2 || true
  exit 1
fi
echo "Started audited post-hoc finalization with PID $PID"
echo "No model editing or image generation will run."
echo "The worker survived its health check; it is safe to disconnect."
echo "Status: $SCRIPT_DIR/status_server.sh '$RUN_DIR'"
