#!/usr/bin/env bash
set -euo pipefail

TRANSFER_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
OUTPUT_DIR="${1:-}"
if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    echo "Usage: bash $0 /absolute/path/to/completed/output" >&2
    exit 2
fi
if [[ ! -f "$OUTPUT_DIR/completion.json" || ! -f "$OUTPUT_DIR/report.md" ]]; then
    echo "Refusing to package an incomplete qualification run." >&2
    exit 2
fi

mkdir -p "$TRANSFER_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$TRANSFER_DIR/oce_failure_image_qualification_v1_${TIMESTAMP}.tar.gz"
tar -C "$OUTPUT_DIR" -czf "$ARCHIVE" \
    report.md \
    completion.json \
    resolved_config.json \
    run_manifest.json \
    checkpoints/manifest.json \
    operator/head_mixing.csv \
    operator/orthogonality.csv \
    operator/summary.json \
    operator/d3_case_construction.json \
    d1/canonical_images.csv \
    d1/canonical_metrics.json \
    d1/composition_images.csv \
    d1/composition_metrics.json \
    d3/images.csv \
    d3/metrics.json

echo "Project images and checkpoints remain in: $OUTPUT_DIR"
echo "Return archive ready for manual scp: $ARCHIVE"
