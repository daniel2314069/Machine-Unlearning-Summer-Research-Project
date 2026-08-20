#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs/qualification_v1"
TRANSFER_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
SELECTION_FILE="$OUTPUT_DIR/review_image_selection.txt"

if [[ ! -f "$OUTPUT_DIR/completion.json" ]]; then
    echo "Refusing to package review images from an incomplete run." >&2
    exit 2
fi

: > "$SELECTION_FILE"
printf '%s\n' "review_image_selection.txt" >> "$SELECTION_FILE"

# Fixed before visual inspection: one canonical prompt/seed per Original/OCE pair.
for target in airplane bird dog truck; do
    printf '%s\n' \
        "d1/images/canonical/original/$target/${target}_canonical_01/seed_101.png" \
        "d1/images/canonical/oce_$target/$target/${target}_canonical_01/seed_101.png" \
        >> "$SELECTION_FILE"
done

# Fixed seed 101 for every two-object prompt under Original and all four edits.
for condition in original oce_airplane oce_bird oce_dog oce_truck; do
    for prompt_id in \
        cat_bicycle car_horse elephant_bus bear_couch pizza_bottle sheep_motorcycle
    do
        printf '%s\n' \
            "d1/images/composition/$condition/$prompt_id/seed_101.png" \
            >> "$SELECTION_FILE"
    done
done

# Fixed target/non-target prompt and seed for both legal realizations in each case.
for target in airplane bird truck; do
    for variant in realization_a realization_b; do
        printf '%s\n' \
            "d3/images/$target/$variant/target/${target}_canonical_01/seed_101.png" \
            "d3/images/$target/$variant/non_target/cat/seed_101.png" \
            >> "$SELECTION_FILE"
    done
done

while IFS= read -r relative_path; do
    if [[ ! -f "$OUTPUT_DIR/$relative_path" ]]; then
        echo "Missing selected review file: $OUTPUT_DIR/$relative_path" >&2
        exit 2
    fi
done < "$SELECTION_FILE"

mkdir -p "$TRANSFER_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$TRANSFER_DIR/oce_failure_image_review_v1_${TIMESTAMP}.tar.gz"
tar -C "$OUTPUT_DIR" -czf "$ARCHIVE" -T "$SELECTION_FILE"

IMAGE_COUNT="$(($(wc -l < "$SELECTION_FILE") - 1))"
echo "Fixed-rule review package created with $IMAGE_COUNT images."
echo "Return archive ready for manual scp: $ARCHIVE"

