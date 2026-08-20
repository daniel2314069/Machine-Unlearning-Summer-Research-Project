#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs/qualification_v1"
TRANSFER_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
D1_IMAGES="$OUTPUT_DIR/d1/images"
D3_IMAGES="$OUTPUT_DIR/d3/images"

if [[ ! -f "$OUTPUT_DIR/completion.json" ]]; then
    echo "Refusing to clean images from an incomplete run." >&2
    exit 2
fi
if ! compgen -G "$TRANSFER_DIR/oce_failure_image_review_v1_*.tar.gz" >/dev/null; then
    echo "Refusing to clean before a review-image archive exists in $TRANSFER_DIR." >&2
    echo "Run: bash $SCRIPT_DIR/package_review_images.sh" >&2
    exit 2
fi
if [[ ! -d "$D1_IMAGES" && ! -d "$D3_IMAGES" ]]; then
    echo "The experiment image directories are already absent."
    exit 0
fi

FILE_COUNT="$({ find "$D1_IMAGES" "$D3_IMAGES" -type f 2>/dev/null || true; } | wc -l | tr -d '[:space:]')"
DISK_USAGE="$({ du -sh "$D1_IMAGES" "$D3_IMAGES" 2>/dev/null || true; })"

rm -rf -- "$D1_IMAGES" "$D3_IMAGES"

if [[ -e "$D1_IMAGES" || -e "$D3_IMAGES" ]]; then
    echo "Image cleanup did not complete." >&2
    exit 1
fi

echo "Deleted $FILE_COUNT generated image files from this completed run."
echo "$DISK_USAGE"
echo "Reports, metrics, manifests, and checkpoints remain in: $OUTPUT_DIR"

