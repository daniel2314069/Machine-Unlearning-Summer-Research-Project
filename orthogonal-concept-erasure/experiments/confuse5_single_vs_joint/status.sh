#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$HERE/outputs"
PID_FILE="$OUTPUT_ROOT/detached.pid"
EXIT_FILE="$OUTPUT_ROOT/detached.exit_code"
LATEST_FILE="$OUTPUT_ROOT/detached.latest"

if [[ ! -f "$LATEST_FILE" ]]; then
    echo "No detached Confuse5 launch record found."
    exit 1
fi

source "$LATEST_FILE"
echo "PID: $pid"
echo "Environment: $environment"
echo "Python: $python"
echo "Started (UTC): $started_utc"
echo "Log: $log"

if kill -0 "$pid" 2>/dev/null; then
    echo "Process status: RUNNING"
else
    echo "Process status: NOT RUNNING"
fi

if [[ -f "$EXIT_FILE" ]]; then
    exit_value="$(tr -d '[:space:]' < "$EXIT_FILE")"
    if [[ "$exit_value" == "running" ]]; then
        echo "Exit status: not available yet"
    elif [[ "$exit_value" == "0" ]]; then
        echo "Exit status: SUCCESS (0)"
    else
        echo "Exit status: FAILED ($exit_value)"
    fi
fi

checkpoint_count="$(find "$OUTPUT_ROOT" -name weights.safetensors -type f | wc -l | tr -d '[:space:]')"
metadata_count="$(find "$OUTPUT_ROOT" -name metadata.json -type f | wc -l | tr -d '[:space:]')"
echo "Checkpoints: $checkpoint_count / 15"
echo "Metadata files: $metadata_count / 15"

if command -v jq >/dev/null 2>&1 && [[ "$metadata_count" != "0" ]]; then
    echo "Metadata statuses:"
    find "$OUTPUT_ROOT" -name metadata.json -type f -print0 \
        | xargs -0 jq -r '.status' \
        | sort \
        | uniq -c
fi

if [[ -f "$log" ]]; then
    echo "Latest log lines:"
    tail -n 20 "$log"
fi
