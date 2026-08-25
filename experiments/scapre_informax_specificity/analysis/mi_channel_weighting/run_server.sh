#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [absolute-legacy-diagnostic.pt]" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
  echo "ERROR: activate the GPU server Conda environment first: conda activate MU" >&2
  exit 1
fi
PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python is unavailable in the active MU environment" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
[[ -f "$REPO_ROOT/scapre/edit/erase_scale.py" ]] || { echo "ERROR: repository root validation failed" >&2; exit 1; }
"$PYTHON_BIN" -c 'import pathlib,sys; compile(pathlib.Path(sys.argv[1]).read_text(), sys.argv[1], "exec")' \
  "$SCRIPT_DIR/run_diagnostics.py"
"$PYTHON_BIN" -c 'import diffusers, matplotlib, numpy, torch, transformers'
LEGACY_PATH="${1:-/home/tslin/Documents/jupyter_data/anLi/machine_unlearning/experiments/scapre_informax_specificity/runs/formal_20260820T163033Z/diagnostics/official.pt}"
[[ "$LEGACY_PATH" = /* && -f "$LEGACY_PATH" ]] || { echo "ERROR: legacy diagnostic is missing: $LEGACY_PATH" >&2; exit 1; }

RUNS_DIR="$SCRIPT_DIR/runs"
mkdir -p "$RUNS_DIR"
LATEST_FILE="$RUNS_DIR/latest_run"
if [[ -f "$LATEST_FILE" ]]; then
  PREVIOUS="$(<"$LATEST_FILE")"
  if [[ -f "$PREVIOUS/pid" && ! -f "$PREVIOUS/exit_code" ]]; then
    PREVIOUS_PID="$(<"$PREVIOUS/pid")"
    if kill -0 "$PREVIOUS_PID" 2>/dev/null; then
      echo "ERROR: an analysis run is already active: $PREVIOUS (PID $PREVIOUS_PID)" >&2
      exit 1
    fi
  fi
fi
RUN_ID="formal_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$RUNS_DIR/$RUN_ID"
mkdir "$RUN_DIR"
LOG_PATH="$RUN_DIR/server.log"
mkdir "$RUN_DIR/provenance"
cp "$SCRIPT_DIR/config.json" "$RUN_DIR/provenance/config.json"
cp "$SCRIPT_DIR/implementation_audit.md" "$RUN_DIR/provenance/implementation_audit.md"
cp "$SCRIPT_DIR/run_diagnostics.py" "$RUN_DIR/provenance/run_diagnostics.py"
cp "$SCRIPT_DIR/run_server.sh" "$RUN_DIR/provenance/run_server.sh"
cp "$SCRIPT_DIR/server_worker.sh" "$RUN_DIR/provenance/server_worker.sh"
cp "$SCRIPT_DIR/status_server.sh" "$RUN_DIR/provenance/status_server.sh"
cp "$SCRIPT_DIR/package_results.sh" "$RUN_DIR/provenance/package_results.sh"
printf '%s\n' "$RUN_DIR" > "$LATEST_FILE"
printf '%s\n' "$RUN_ID" > "$RUN_DIR/run_id"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/started_at_utc"
printf '%s\n' "$PYTHON_BIN" > "$RUN_DIR/python_executable"
printf '%s\n' "$LEGACY_PATH" > "$RUN_DIR/legacy_diagnostic"
printf '%s\n' "$LOG_PATH" > "$RUN_DIR/log_path"
printf '%s\n' "$RUN_DIR/output" > "$RUN_DIR/output_path"

cd "$REPO_ROOT"
nohup bash "$SCRIPT_DIR/server_worker.sh" "$RUN_DIR" "$PYTHON_BIN" "$LEGACY_PATH" \
  > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/pid"
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  for _ in {1..20}; do
    [[ -f "$RUN_DIR/exit_code" ]] && break
    sleep 0.1
  done
  if [[ -f "$RUN_DIR/exit_code" ]]; then
    EXIT_CODE="$(<"$RUN_DIR/exit_code")"
  else
    EXIT_CODE="unknown"
  fi
  echo "ERROR: worker exited during health check (exit=$EXIT_CODE); log: $LOG_PATH" >&2
  tail -n 40 "$LOG_PATH" >&2 || true
  exit 1
fi
echo "Started MI-only analysis: $RUN_ID"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Output: $RUN_DIR/output"
echo "The worker is detached; it is safe to close the terminal or SSH connection."
