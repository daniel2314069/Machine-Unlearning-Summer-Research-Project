#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$1"
OUTPUT_DIR="$2"
EXIT_FILE="$3"
shift 3

cd "$OCE_ROOT"

run_stage() {
    local stage="$1"
    shift
    echo "[stage] $stage"
    "$PYTHON_BIN" -u "$SCRIPT_DIR/run_qualification.py" \
        --config "$SCRIPT_DIR/config.json" \
        --output-dir "$OUTPUT_DIR" \
        "$stage" "$@"
}

run_stage preflight "$@" && \
run_stage prepare "$@" && \
run_stage d1-canonical-generate "$@" && \
run_stage d1-canonical-evaluate "$@" && \
run_stage d1-composition-generate "$@" && \
run_stage d1-composition-evaluate "$@" && \
run_stage d3-generate "$@" && \
run_stage d3-evaluate "$@" && \
run_stage report "$@"
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
