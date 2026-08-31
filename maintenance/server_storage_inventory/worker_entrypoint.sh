#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
    echo "Usage: $0 PYTHON WORKER ROOT RUN_DIR RUN_ID GIT_COMMIT GIT_BRANCH" >&2
    exit 2
fi

PYTHON_BIN="$1"
WORKER="$2"
SCAN_ROOT="$3"
RUN_DIR="$4"
RUN_ID="$5"
GIT_COMMIT="$6"
GIT_BRANCH="$7"

printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/started_at_utc"
printf '%s\n' "starting" > "$RUN_DIR/stage"

set +e
"$PYTHON_BIN" "$WORKER" \
    --root "$SCAN_ROOT" \
    --output-dir "$RUN_DIR" \
    --run-id "$RUN_ID" \
    --git-commit "$GIT_COMMIT" \
    --git-branch "$GIT_BRANCH"
EXIT_CODE=$?
set -e

printf '%s\n' "$EXIT_CODE" > "$RUN_DIR/exit_code"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/finished_at_utc"
if [[ "$EXIT_CODE" -eq 0 ]]; then
    printf '%s\n' "completed" > "$RUN_DIR/COMPLETED"
else
    printf '%s\n' "failed" > "$RUN_DIR/FAILED"
fi
exit "$EXIT_CODE"
