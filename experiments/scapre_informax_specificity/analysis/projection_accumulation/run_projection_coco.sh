#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARENT_EXPERIMENT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATE_DIR="$SCRIPT_DIR/.server/coco"
RUNS_DIR="$SCRIPT_DIR/runs/coco"
ASSETS="$PARENT_EXPERIMENT/.server/assets_manifest.json"
ACTION="${1:-}"

status_run() {
  local run_dir="${1:-}"
  if [[ -z "$run_dir" ]]; then
    [[ -f "$STATE_DIR/latest_run" ]] || { echo "ERROR: no COCO run recorded" >&2; exit 2; }
    run_dir="$(tr -d '\r\n' < "$STATE_DIR/latest_run")"
  fi
  [[ -d "$run_dir" ]] || { echo "ERROR: run directory missing: $run_dir" >&2; exit 2; }
  local pid="unknown" exit_code="pending" state="unknown/stale" stage="pending"
  [[ -f "$run_dir/pid" ]] && pid="$(tr -d '[:space:]' < "$run_dir/pid")"
  [[ -f "$run_dir/exit_code" ]] && exit_code="$(tr -d '[:space:]' < "$run_dir/exit_code")"
  [[ -f "$run_dir/status.json" ]] && stage="$(sed -n 's/^  "stage": "\([^"]*\)".*/\1/p' "$run_dir/status.json")"
  if [[ -f "$run_dir/COMPLETED" && "$exit_code" == "0" ]]; then state="completed"
  elif [[ -f "$run_dir/FAILED" || "$exit_code" != "pending" ]]; then state="failed"
  elif [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then state="running"; fi
  local mode="unknown" original=0 official=0 projection=0
  [[ -f "$run_dir/mode" ]] && mode="$(tr -d '\r\n' < "$run_dir/mode")"
  [[ -d "$run_dir/images/original_sd15" ]] && original="$(find "$run_dir/images/original_sd15" -maxdepth 1 -name '*.png' -type f | wc -l | tr -d ' ')"
  [[ -d "$run_dir/images/official" ]] && official="$(find "$run_dir/images/official" -maxdepth 1 -name '*.png' -type f | wc -l | tr -d ' ')"
  [[ -d "$run_dir/images/projection_accumulation" ]] && projection="$(find "$run_dir/images/projection_accumulation" -maxdepth 1 -name '*.png' -type f | wc -l | tr -d ' ')"
  echo "status: $state"
  echo "stage: $stage"
  echo "mode: $mode"
  echo "pid: $pid"
  echo "exit_code: $exit_code"
  echo "completed_images: original_sd15=$original official=$official projection_accumulation=$projection"
  echo "output: $run_dir"
  echo "log: $run_dir/logs/server.log"
  [[ -f "$run_dir/reproducibility/run_manifest.json" ]] && \
    sed -n 's/^  "git_commit": "\([^"]*\)".*/git_commit: \1/p' "$run_dir/reproducibility/run_manifest.json"
  echo "git_status_now:"
  git -C "$REPO_ROOT" status --short --untracked-files=all || true
  echo
  echo "Recent log:"
  tail -n 80 "$run_dir/logs/server.log" 2>/dev/null || echo "(log not created yet)"
}

if [[ "$ACTION" == "--status" ]]; then status_run "${2:-}"; exit 0; fi
if [[ "$ACTION" == "--package" ]]; then
  exec "$SCRIPT_DIR/package_coco_results.sh" "${2:-}"
fi
if [[ "$ACTION" != "--first-1k" && "$ACTION" != "--first-10k" ]]; then
  echo "usage: $0 <--first-1k|--first-10k|--status|--package> [run-id-or-run-dir]" >&2
  exit 2
fi
MODE="first-1k"; COUNT=1000
if [[ "$ACTION" == "--first-10k" ]]; then MODE="first-10k"; COUNT=10000; fi
[[ "${CONDA_DEFAULT_ENV:-}" == "MU" ]] || { echo "ERROR: run 'conda activate MU' first" >&2; exit 2; }
PYTHON_BIN="$(command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: python unavailable in Conda MU" >&2; exit 2; }
[[ -f "$ASSETS" && -f "$PARENT_EXPERIMENT/.server/SETUP_COMPLETE" ]] || {
  echo "ERROR: ScaPre server assets are not prepared" >&2; exit 2;
}
cd "$REPO_ROOT"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "ERROR: COCO launch requires a clean working tree" >&2; git status --short >&2; exit 2;
}
"$PYTHON_BIN" -c 'import torch, torch_fidelity, pandas, transformers; assert torch.cuda.is_available(), "CUDA unavailable"'
mkdir -p "$STATE_DIR" "$RUNS_DIR"
if [[ "$ACTION" == "--first-10k" ]]; then
  [[ -f "$STATE_DIR/latest_successful_first1k" ]] || {
    echo "ERROR: first-10k requires a separately completed first-1k screening run" >&2; exit 2;
  }
  FIRST1K="$(tr -d '\r\n' < "$STATE_DIR/latest_successful_first1k")"
  [[ -f "$FIRST1K/COMPLETED" && -f "$FIRST1K/results/integrity_report.json" ]] || {
    echo "ERROR: recorded first-1k run is incomplete" >&2; exit 2;
  }
fi
if [[ -f "$STATE_DIR/active_run" ]]; then
  ACTIVE="$(tr -d '\r\n' < "$STATE_DIR/active_run")"
  if [[ -f "$ACTIVE/pid" && ! -f "$ACTIVE/COMPLETED" && ! -f "$ACTIVE/FAILED" ]]; then
    ACTIVE_PID="$(tr -d '[:space:]' < "$ACTIVE/pid")"
    if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
      echo "ERROR: COCO run already active: $ACTIVE (PID $ACTIVE_PID)" >&2; exit 2
    fi
  fi
fi
RUN_ID="${2:-$(date -u +'%Y%m%dT%H%M%SZ')}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid run id" >&2; exit 2; }
RUN_DIR="$RUNS_DIR/${MODE}_$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { echo "ERROR: run already exists: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR/logs"
printf '%s\n' "$PYTHON_BIN" > "$RUN_DIR/python_path"
printf '%s\n' "$MODE" > "$RUN_DIR/mode"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/latest_run"
printf '%s\n' "$RUN_DIR" > "$STATE_DIR/active_run"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/started_at_utc"
nohup "$SCRIPT_DIR/coco_server_worker.sh" "$MODE" "$RUN_DIR" "$ASSETS" \
  </dev/null >"$RUN_DIR/logs/server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/pid"
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: COCO worker exited during launch health check" >&2
  tail -n 80 "$RUN_DIR/logs/server.log" >&2 || true
  exit 1
fi
echo "Started COCO $MODE safeguard ($COUNT prompts per method). Safe to disconnect."
echo "This runner will not start any other COCO mode."
echo "PID: $PID"
echo "Output: $RUN_DIR"
echo "Log: $RUN_DIR/logs/server.log"
echo "Status: $SCRIPT_DIR/run_projection_coco.sh --status"
