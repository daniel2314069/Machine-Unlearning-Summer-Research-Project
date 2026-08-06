#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$HERE/../experiments/correspondence_diagnostic/scripts/run_py310.sh"
RUNNER="$HERE/run_experiment.py"

"$WRAPPER" "$RUNNER" celebrity --batch-size 4
"$WRAPPER" "$RUNNER" coco \
  --coco-count 1000 \
  --batch-size 4 \
  --clip-batch-size 64
