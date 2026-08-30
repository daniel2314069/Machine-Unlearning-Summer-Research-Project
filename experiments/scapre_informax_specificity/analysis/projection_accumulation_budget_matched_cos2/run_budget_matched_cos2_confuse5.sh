#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARENT_EXPERIMENT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_EXPERIMENT="$SCRIPT_DIR/../projection_accumulation"
STATE_DIR="$SCRIPT_DIR/.server/confuse5"
RUNS_DIR="$SCRIPT_DIR/runs/confuse5"
ASSETS="$PARENT_EXPERIMENT/.server/assets_manifest.json"
TREATMENT="projection_accumulation_budget_matched_cos2"
ACTION="${1:-launch}"

status_run() {
  local run_dir="${1:-}"
  if [[ -z "$run_dir" ]]; then
    [[ -f "$STATE_DIR/latest_run" ]] || {
      echo "ERROR: no budget-matched-cos2 run recorded" >&2
      exit 2
    }
    run_dir="$(tr -d '\r\n' < "$STATE_DIR/latest_run")"
  fi
  [[ -d "$run_dir" ]] || { echo "ERROR: run directory missing: $run_dir" >&2; exit 2; }
  local pid="unknown" exit_code="pending" state="unknown/stale"
  [[ -f "$run_dir/pid" ]] && pid="$(tr -d '[:space:]' < "$run_dir/pid")"
  [[ -f "$run_dir/exit_code" ]] && exit_code="$(tr -d '[:space:]' < "$run_dir/exit_code")"
  if [[ -f "$run_dir/COMPLETED" && "$exit_code" == "0" ]]; then
    state="completed"
  elif [[ -f "$run_dir/FAILED" || "$exit_code" != "pending" ]]; then
    state="failed"
  elif [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    state="running"
  fi
  local stage="pending" seeds="[]" scored=0
  if [[ -f "$run_dir/status.json" ]]; then
    stage="$(sed -n 's/^  "stage": "\([^"]*\)".*/\1/p' "$run_dir/status.json")"
    seeds="$(sed -n '/"completed_seeds": \[/,/\]/p' "$run_dir/status.json" | tr '\n' ' ')"
  fi
  while IFS= read -r score_file; do
    local rows=$(( $(wc -l < "$score_file") - 1 ))
    (( rows > 0 )) && scored=$((scored + rows))
  done < <(find "$run_dir/seeds" -path "*/evaluation/$TREATMENT/scores.csv" -type f 2>/dev/null | sort)
  echo "status: $state"
  echo "stage: $stage"
  echo "pid: $pid"
  echo "exit_code: $exit_code"
  echo "completed_seeds: $seeds"
  echo "completed_budget_matched_cos2_images: $scored/15000"
  echo "output: $run_dir"
  echo "log: $run_dir/logs/server.log"
  echo "summary: $run_dir/results/validation_report.md"
  [[ -f "$run_dir/reproducibility/run_manifest.json" ]] && \
    sed -n 's/^  "git_commit": "\([^"]*\)".*/git_commit: \1/p; s/^  "git_status_start": \(.*\).*/git_status_start: \1/p' "$run_dir/reproducibility/run_manifest.json"
  echo "git_status_now:"
  git -C "$REPO_ROOT" status --short --untracked-files=all || true
  echo
  echo "Recent log:"
  tail -n 80 "$run_dir/logs/server.log" 2>/dev/null || echo "(log not created yet)"
}

if [[ "$ACTION" == "--status" ]]; then
  status_run "${2:-}"
  exit 0
fi
if [[ "$ACTION" == "--package" ]]; then
  exec "$SCRIPT_DIR/package_results.sh" "${2:-}"
fi
if [[ "$ACTION" != "launch" && "$ACTION" != "--resume" && "$ACTION" != "--preflight" ]]; then
  echo "usage: $0 [launch|--resume|--preflight|--status|--package] [run-id-or-run-dir]" >&2
  exit 2
fi

[[ "${CONDA_DEFAULT_ENV:-}" == "MU" ]] || {
  echo "ERROR: run 'conda activate MU' first" >&2
  exit 2
}
PYTHON_BIN="$(command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: python unavailable in Conda MU" >&2; exit 2; }
for command_name in git nohup sha256sum tar; do
  command -v "$command_name" >/dev/null || {
    echo "ERROR: missing command: $command_name" >&2
    exit 2
  }
done
[[ -f "$ASSETS" && -f "$PARENT_EXPERIMENT/.server/SETUP_COMPLETE" ]] || {
  echo "ERROR: ScaPre server assets are not prepared" >&2
  exit 2
}
cd "$REPO_ROOT"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "ERROR: formal launch requires a clean working tree" >&2
  git status --short >&2
  exit 2
}
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
"$PYTHON_BIN" -m py_compile \
  "$SHARED_EXPERIMENT/projection_runner.py" \
  "$SHARED_EXPERIMENT/evaluate_projection_runner.py" \
  "$SHARED_EXPERIMENT/aggregate_results.py" \
  "$SHARED_EXPERIMENT/worker.py" \
  "$SCRIPT_DIR/preflight_selftest.py"
"$PYTHON_BIN" "$SCRIPT_DIR/preflight_selftest.py"
OFFICIAL_REFERENCE="$("$PARENT_EXPERIMENT/analysis/alpha_channel_controls/resolve_official_reference.sh")"
[[ -d "$OFFICIAL_REFERENCE" ]] || {
  echo "ERROR: official reference resolution failed" >&2
  exit 2
}
if [[ "$ACTION" == "--preflight" ]]; then
  echo "Budget-matched-cos2 server preflight passed. No experiment was launched."
  echo "Python: $PYTHON_BIN"
  echo "Official reference: $OFFICIAL_REFERENCE"
  exit 0
fi

mkdir -p "$STATE_DIR" "$RUNS_DIR"
if [[ -f "$STATE_DIR/active_run" ]]; then
  ACTIVE="$(tr -d '\r\n' < "$STATE_DIR/active_run")"
  if [[ -f "$ACTIVE/pid" && ! -f "$ACTIVE/COMPLETED" && ! -f "$ACTIVE/FAILED" ]]; then
    ACTIVE_PID="$(tr -d '[:space:]' < "$ACTIVE/pid")"
    if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
      echo "ERROR: budget-matched-cos2 run already active: $ACTIVE (PID $ACTIVE_PID)" >&2
      exit 2
    fi
  fi
fi
if [[ "$ACTION" == "--resume" ]]; then
  if [[ -n "${2:-}" ]]; then
    RUN_ID="$2"
    [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid run id" >&2; exit 2; }
    RUN_DIR="$RUNS_DIR/formal_$RUN_ID"
  else
    [[ -f "$STATE_DIR/latest_run" ]] || { echo "ERROR: no run available to resume" >&2; exit 2; }
    RUN_DIR="$(tr -d '\r\n' < "$STATE_DIR/latest_run")"
  fi
  [[ -d "$RUN_DIR" && ! -f "$RUN_DIR/COMPLETED" ]] || {
    echo "ERROR: run cannot be resumed: $RUN_DIR" >&2
    exit 2
  }
else
  RUN_ID="${2:-$(date -u +'%Y%m%dT%H%M%SZ')}"
  [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid run id" >&2; exit 2; }
  RUN_DIR="$RUNS_DIR/formal_$RUN_ID"
  [[ ! -e "$RUN_DIR" ]] || { echo "ERROR: run already exists: $RUN_DIR" >&2; exit 2; }
  mkdir -p "$RUN_DIR/logs"
fi
[[ "$RUN_DIR" == "$RUNS_DIR"/formal_* ]] || {
  echo "ERROR: run path escaped runs root" >&2
  exit 2
}
mkdir -p "$RUN_DIR/logs"
printf '%s\n' "$PYTHON_BIN" > "$RUN_DIR/python_path"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/latest_run"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/active_run"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/started_at_utc"
nohup "$SCRIPT_DIR/server_worker.sh" "$RUN_DIR" "$ASSETS" "$OFFICIAL_REFERENCE" \
  </dev/null >"$RUN_DIR/logs/server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/pid"
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: worker exited during launch health check" >&2
  tail -n 80 "$RUN_DIR/logs/server.log" >&2 || true
  exit 1
fi
echo "Started qualification -> budget-matched-cos2 Confuse5 formal. Safe to disconnect."
echo "PID: $PID"
echo "Output: $RUN_DIR"
echo "Log: $RUN_DIR/logs/server.log"
echo "Status: $SCRIPT_DIR/run_budget_matched_cos2_confuse5.sh --status"
