#!/usr/bin/env bash
# Detached H100 benchmark-to-formal-run controller.  The watchdog is separate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lightning_deployment.env"
RUNNER="$SCRIPT_DIR/run_sequential_long_horizon.sh"
mode="${1:---start}"
if [[ $# -gt 0 ]]; then shift; fi

remaining_credits=""
gpu_rate=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remaining-credits) remaining_credits="${2:-}"; shift 2 ;;
    --gpu-rate) gpu_rate="${2:-}"; shift 2 ;;
    *) printf 'Unknown controller argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

controller_pid_file="$SEQUENTIAL_OCE_OUTPUT/.run/controller_pid"
controller_log="$SEQUENTIAL_OCE_OUTPUT/logs/controller.log"
fatal_marker="$SEQUENTIAL_OCE_OUTPUT/.watchdog/fatal"

pid_matches_command() {
  local candidate="$1"
  local expected="$2"
  [[ -n "$candidate" ]] && kill -0 "$candidate" 2>/dev/null || return 1
  [[ -r "/proc/$candidate/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/$candidate/cmdline")"
  [[ "$command_line" == *"$expected"* && "$command_line" == *"--worker"* ]]
}

require_budget() {
  if [[ -z "$remaining_credits" || -z "$gpu_rate" ]]; then
    printf 'ERROR: --remaining-credits and --gpu-rate are required.\n' >&2
    exit 2
  fi
}

case "$mode" in
  --start)
    require_budget
    mkdir -p "$(dirname "$controller_pid_file")" "$(dirname "$controller_log")" "$(dirname "$fatal_marker")"
    if [[ -f "$controller_pid_file" ]]; then
      old_pid="$(tr -cd '0-9' < "$controller_pid_file")"
      if pid_matches_command "$old_pid" "lightning_h100_controller.sh"; then
        printf 'Controller already running with PID %s\n' "$old_pid"
        exit 0
      fi
    fi
    if [[ -f "$fatal_marker" ]]; then
      printf 'ERROR: fatal marker exists: %s\n' "$fatal_marker" >&2
      exit 2
    fi
    nohup "$0" --worker \
      --remaining-credits "$remaining_credits" --gpu-rate "$gpu_rate" \
      </dev/null >> "$controller_log" 2>&1 &
    controller_pid=$!
    printf '%s\n' "$controller_pid" > "$controller_pid_file"
    printf 'Started detached H100 controller PID %s\n' "$controller_pid"
    printf 'Log: %s\n' "$controller_log"
    ;;
  --worker)
    require_budget
    mkdir -p "$(dirname "$controller_pid_file")" "$(dirname "$controller_log")" "$(dirname "$fatal_marker")"
    printf '%s\n' "$$" > "$controller_pid_file"
    trap 'code=$?; if [[ $code -ne 0 ]]; then printf "%s\n" "controller_exit_$code" > "$fatal_marker"; sync; fi' EXIT
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -cd '0-9')"
    if [[ "$gpu_name" != *H100* || -z "$gpu_memory" || "$gpu_memory" -lt 80000 ]]; then
      printf 'ERROR: expected one H100 80GB, observed name=%s memory_mib=%s\n' "$gpu_name" "$gpu_memory" >&2
      exit 3
    fi
    printf '[%s] verified GPU name=%s memory_mib=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gpu_name" "$gpu_memory"
    hub_version="$("$SEQUENTIAL_OCE_PYTHON" -c 'import huggingface_hub; print(huggingface_hub.__version__)')"
    if [[ "$hub_version" != "0.36.2" ]]; then
      printf 'ERROR: frozen huggingface-hub version is 0.36.2, observed %s; repair the active environment before launch.\n' "$hub_version" >&2
      exit 5
    fi
    "$SEQUENTIAL_OCE_PYTHON" -c 'from diffusers import DiffusionPipeline' >/dev/null
    printf '[%s] verified Python=%s huggingface-hub=%s DiffusionPipeline=importable\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SEQUENTIAL_OCE_PYTHON" "$hub_version"
    cd "$SCRIPT_DIR"
    "$RUNNER" --benchmark \
      --remaining-credits "$remaining_credits" \
      --gpu-rate "$gpu_rate" \
      --output-dir "$SEQUENTIAL_OCE_OUTPUT" \
      --artifact-root "$SEQUENTIAL_OCE_ARTIFACT_ROOT" \
      --gcd-project-root "$GCD_PROJECT_ROOT"
    "$RUNNER" --start \
      --output-dir "$SEQUENTIAL_OCE_OUTPUT" \
      --artifact-root "$SEQUENTIAL_OCE_ARTIFACT_ROOT" \
      --gcd-project-root "$GCD_PROJECT_ROOT"
    sleep 2
    runner_pid="$(tr -cd '0-9' < "$SEQUENTIAL_OCE_OUTPUT/.run/pid")"
    if ! pid_matches_command "$runner_pid" "run_sequential_long_horizon.sh"; then
      printf 'ERROR: detached formal runner did not remain alive.\n' >&2
      exit 4
    fi
    printf '[%s] formal runner verified pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$runner_pid"
    trap - EXIT
    ;;
  *)
    printf 'Usage: %s --start --remaining-credits N --gpu-rate N\n' "$0" >&2
    exit 2
    ;;
esac
