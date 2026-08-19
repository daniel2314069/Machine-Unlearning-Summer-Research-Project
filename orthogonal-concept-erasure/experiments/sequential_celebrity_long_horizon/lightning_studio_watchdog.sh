#!/usr/bin/env bash
# Independent Lightning Studio shutdown watchdog for the formal H100 run.

set -uo pipefail

mode="${1:---status}"
if [[ $# -gt 0 ]]; then
  shift
fi

output_dir=""
studio_name=""
teamspace=""
deadline_seconds=20700
lightning_cli="/home/zeus/miniconda3/envs/cloudspace/bin/lightning"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) output_dir="${2:-}"; shift 2 ;;
    --studio-name) studio_name="${2:-}"; shift 2 ;;
    --teamspace) teamspace="${2:-}"; shift 2 ;;
    --deadline-seconds) deadline_seconds="${2:-}"; shift 2 ;;
    --lightning-cli) lightning_cli="${2:-}"; shift 2 ;;
    *) printf 'Unknown watchdog argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$output_dir" || "$output_dir" != /* || "$output_dir" == "/" ]]; then
  printf 'ERROR: --output-dir must be a safe absolute path.\n' >&2
  exit 2
fi
if [[ -z "$studio_name" || -z "$teamspace" ]]; then
  printf 'ERROR: --studio-name and --teamspace are required.\n' >&2
  exit 2
fi
if [[ "$deadline_seconds" != "20700" ]]; then
  printf 'ERROR: the final formal hard deadline is frozen at 20700 seconds.\n' >&2
  exit 2
fi
if [[ ! -x "$lightning_cli" ]]; then
  printf 'ERROR: Lightning CLI is not executable: %s\n' "$lightning_cli" >&2
  exit 2
fi

watchdog_dir="$output_dir/.watchdog"
pid_file="$watchdog_dir/pid"
state_file="$watchdog_dir/state.env"
log_file="$output_dir/logs/watchdog.log"
runner_pid_file="$output_dir/.run/pid"
runner_exit_file="$output_dir/.run/exit_code"
final_validation="$output_dir/final_validation.json"
fatal_marker="$watchdog_dir/fatal"

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

read_pid() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  tr -cd '0-9' < "$path"
}

pid_is_running() {
  local candidate="$1"
  [[ -n "$candidate" ]] && kill -0 "$candidate" 2>/dev/null
}

pid_matches_command() {
  local candidate="$1"
  local expected="$2"
  pid_is_running "$candidate" || return 1
  [[ -r "/proc/$candidate/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/$candidate/cmdline")"
  [[ "$command_line" == *"$expected"* && "$command_line" == *"--worker"* ]]
}

write_state() {
  local status="$1"
  local reason="$2"
  local started_at="$3"
  local deadline_epoch="$4"
  local temporary="$state_file.tmp"
  {
    printf 'status=%s\n' "$status"
    printf 'reason=%s\n' "$reason"
    printf 'watchdog_pid=%s\n' "$$"
    printf 'started_at_utc=%s\n' "$started_at"
    printf 'deadline_epoch=%s\n' "$deadline_epoch"
    printf 'deadline_utc=%s\n' "$(date -u -d "@$deadline_epoch" +%Y-%m-%dT%H:%M:%SZ)"
    printf 'updated_at_utc=%s\n' "$(utc_now)"
    printf 'studio_name=%s\n' "$studio_name"
    printf 'teamspace=%s\n' "$teamspace"
  } > "$temporary"
  mv "$temporary" "$state_file"
}

case "$mode" in
  --start)
    mkdir -p "$watchdog_dir" "$(dirname "$log_file")"
    existing_pid="$(read_pid "$pid_file" || true)"
    if pid_matches_command "$existing_pid" "lightning_studio_watchdog.sh"; then
      printf 'Watchdog already running with PID %s\n' "$existing_pid"
      exit 0
    fi
    if [[ -f "$fatal_marker" ]]; then
      printf 'ERROR: stale watchdog fatal marker exists: %s\n' "$fatal_marker" >&2
      exit 2
    fi
    runner_pid="$(read_pid "$runner_pid_file" || true)"
    printf '[%s] watchdog detached start requested runner_pid=%s deadline_seconds=%s\n' \
      "$(utc_now)" "${runner_pid:-awaiting_benchmark}" "$deadline_seconds" >> "$log_file"
    nohup "$0" --worker \
      --output-dir "$output_dir" \
      --studio-name "$studio_name" \
      --teamspace "$teamspace" \
      --deadline-seconds "$deadline_seconds" \
      --lightning-cli "$lightning_cli" \
      </dev/null >> "$log_file" 2>&1 &
    watchdog_pid=$!
    printf '%s\n' "$watchdog_pid" > "$pid_file"
    printf 'Started independent watchdog PID %s\n' "$watchdog_pid"
    printf 'Log: %s\n' "$log_file"
    ;;
  --worker)
    mkdir -p "$watchdog_dir" "$(dirname "$log_file")"
    printf '%s\n' "$$" > "$pid_file"
    started_at="$(utc_now)"
    deadline_epoch=$(($(date +%s) + deadline_seconds))
    write_state "running" "monitoring" "$started_at" "$deadline_epoch"
    printf '[%s] watchdog active pid=%s deadline_utc=%s\n' \
      "$started_at" "$$" "$(date -u -d "@$deadline_epoch" +%Y-%m-%dT%H:%M:%SZ)"

    stop_reason=""
    while [[ -z "$stop_reason" ]]; do
      now_epoch="$(date +%s)"
      runner_pid="$(read_pid "$runner_pid_file" || true)"
      if [[ -f "$fatal_marker" ]]; then
        stop_reason="controller_fatal_failure"
      elif [[ -f "$final_validation" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"complete"' "$final_validation"; then
        stop_reason="normal_complete"
      elif ! pid_matches_command "$runner_pid" "run_sequential_long_horizon.sh" && [[ -f "$runner_exit_file" ]]; then
        runner_exit="$(tr -cd '0-9' < "$runner_exit_file")"
        stop_reason="runner_exit_${runner_exit:-unknown}"
      elif (( now_epoch >= deadline_epoch )); then
        stop_reason="hard_deadline_20700_seconds"
      else
        sleep 15
      fi
    done

    write_state "stopping" "$stop_reason" "$started_at" "$deadline_epoch"
    printf '[%s] stop condition reason=%s; flushing persistent state\n' "$(utc_now)" "$stop_reason"
    if [[ "$stop_reason" == "hard_deadline_20700_seconds" ]]; then
      runner_pid="$(read_pid "$runner_pid_file" || true)"
      if pid_matches_command "$runner_pid" "run_sequential_long_horizon.sh"; then
        kill -TERM "$runner_pid" 2>/dev/null || true
        sleep 10
      fi
    fi
    sync
    printf '[%s] requesting whole-Studio stop name=%s teamspace=%s\n' \
      "$(utc_now)" "$studio_name" "$teamspace"
    write_state "stop_requested" "$stop_reason" "$started_at" "$deadline_epoch"
    "$lightning_cli" studio stop --name "$studio_name" --teamspace "$teamspace"
    stop_exit=$?
    printf '[%s] Lightning stop command exit=%s\n' "$(utc_now)" "$stop_exit"
    exit "$stop_exit"
    ;;
  --status)
    watchdog_pid="$(read_pid "$pid_file" || true)"
    if pid_matches_command "$watchdog_pid" "lightning_studio_watchdog.sh"; then
      printf 'watchdog: running\n'
    else
      printf 'watchdog: stopped\n'
    fi
    printf 'pid: %s\n' "${watchdog_pid:--}"
    if [[ -f "$state_file" ]]; then
      printf 'state:\n'
      sed 's/^/  /' "$state_file"
    fi
    printf 'latest log (%s):\n' "$log_file"
    if [[ -f "$log_file" ]]; then
      tail -n 25 "$log_file"
    else
      printf '(no log yet)\n'
    fi
    ;;
  *)
    printf 'Usage: %s --start|--status --output-dir PATH --studio-name NAME --teamspace OWNER/TEAMSPACE\n' "$0" >&2
    exit 2
    ;;
esac
