#!/usr/bin/env bash
set -euo pipefail

action="${1:-status}"
root="${2:-outputs/codex_diverse_overnight}"
config="${CONFIG:-configs/codex_diverse_4x50.json}"
pool_config="${POOL_CONFIG:-configs/codex_diverse_pool.json}"
max_hours="${MAX_HOURS:-20}"

mkdir -p "$root/.matplotlib"
export MPLCONFIGDIR="$root/.matplotlib"

case "$action" in
  launch|resume)
    if [[ "$action" == "launch" && -e "$root/state.json" ]]; then
      echo "Refusing to replace existing state at $root/state.json; use resume."
      exit 2
    fi
    if [[ "$action" == "resume" && ! -e "$root/state.json" ]]; then
      echo "No state at $root/state.json; use launch."
      exit 2
    fi
    runner_action="run"
    if [[ "$action" == "resume" ]]; then
      runner_action="resume"
    fi
    nohup setsid ./scripts/run_py310.sh -m concept_clustering.overnight_runner "$runner_action" \
      --config "$config" \
      --pool-config "$pool_config" \
      --root "$root" \
      --max-hours "$max_hours" \
      </dev/null > "$root/runner_console.log" 2>&1 &
    runner_pid=$!
    disown "$runner_pid" 2>/dev/null || true
    echo "$runner_pid" > "$root/runner.pid"
    sleep 2
    if ! kill -0 "$runner_pid" 2>/dev/null; then
      echo "Runner exited during launch; inspect $root/runner_console.log"
      exit 1
    fi
    echo "Started PID $runner_pid; status: $root/heartbeat.json"
    ;;
  status)
    ./scripts/run_py310.sh -m concept_clustering.overnight_runner status --root "$root"
    ;;
  cluster-only)
    ./scripts/run_py310.sh -m concept_clustering.overnight_runner cluster-only \
      --config "$config" \
      --pool-config "$pool_config" \
      --root "$root"
    ;;
  *)
    echo "Usage: $0 {launch|resume|status|cluster-only} [output_root]"
    exit 2
    ;;
esac
