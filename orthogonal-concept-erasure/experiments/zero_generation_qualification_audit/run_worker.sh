#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$1"
OUTPUT_DIR="$2"
EXIT_FILE="$3"
shift 3

cd "$OCE_ROOT"
"$PYTHON_BIN" -u "$SCRIPT_DIR/audit_oce.py" \
    --cg-path "$OCE_ROOT/Cg.pt" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
JOB_EXIT=$?
printf '%s\n' "$JOB_EXIT" > "$EXIT_FILE"

if [[ "$JOB_EXIT" -eq 0 ]]; then
    bash "$SCRIPT_DIR/package_results.sh" "$OUTPUT_DIR"
    PACKAGE_EXIT=$?
    if [[ "$PACKAGE_EXIT" -ne 0 ]]; then
        printf '%s\n' "$PACKAGE_EXIT" > "$EXIT_FILE"
        exit "$PACKAGE_EXIT"
    fi
fi
exit "$JOB_EXIT"
