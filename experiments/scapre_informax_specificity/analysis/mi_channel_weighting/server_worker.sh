#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <run-dir> <python-executable> [legacy-diagnostic.pt]" >&2
  exit 2
fi
RUN_DIR="$1"
PYTHON_BIN="$2"
LEGACY_PATH="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set +e
if [[ -n "$LEGACY_PATH" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/run_diagnostics.py" \
    --config "$SCRIPT_DIR/config.json" \
    --legacy-diagnostic "$LEGACY_PATH" \
    --output "$RUN_DIR/output"
else
  "$PYTHON_BIN" "$SCRIPT_DIR/run_diagnostics.py" \
    --config "$SCRIPT_DIR/config.json" \
    --output "$RUN_DIR/output"
fi
EXIT_CODE=$?
set -e
printf '%s\n' "$EXIT_CODE" > "$RUN_DIR/exit_code"
if [[ "$EXIT_CODE" -eq 0 && -f "$RUN_DIR/output/COMPLETED" ]]; then
  printf 'completed\n' > "$RUN_DIR/COMPLETED"
else
  printf 'failed\n' > "$RUN_DIR/FAILED"
fi
exit "$EXIT_CODE"
