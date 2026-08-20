#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
OUTPUT_DIR="${1:-}"

if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    echo "Usage: bash $0 /absolute/path/to/completed/project/output" >&2
    exit 2
fi
for REQUIRED in report.md audit_results.json run_manifest.json direction1_layers.csv direction3_layers.csv; do
    if [[ ! -f "$OUTPUT_DIR/$REQUIRED" ]]; then
        echo "Refusing to package incomplete output: missing $OUTPUT_DIR/$REQUIRED" >&2
        exit 2
    fi
done

mkdir -p "$TRANSFER_DIR"
RUN_NAME="$(basename "$OUTPUT_DIR")"
ARCHIVE="$TRANSFER_DIR/oce_zero_generation_qualification_${RUN_NAME}.tar.gz"
tar -C "$OUTPUT_DIR" -czf "$ARCHIVE" \
    report.md \
    audit_results.json \
    run_manifest.json \
    direction1_layers.csv \
    direction3_layers.csv

echo "Packaged only the return artifacts; project outputs remain in: $OUTPUT_DIR"
echo "Archive ready for manual scp: $ARCHIVE"
