#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$1"
EXIT_FILE="$2"
shift 2

cd "$SCRIPT_DIR"

"$PYTHON_BIN" -u run_sequential_oce.py run "$@"
JOB_EXIT=$?
printf '%s\n' "$JOB_EXIT" > "$EXIT_FILE"
exit "$JOB_EXIT"
