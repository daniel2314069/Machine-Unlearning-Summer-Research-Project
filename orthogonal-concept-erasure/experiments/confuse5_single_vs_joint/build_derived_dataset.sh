#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$HERE/../../../scapre/eval/datasets/imagenet-15.csv"
DESTINATION="$HERE/datasets/imagenet-confuse5-derived-25.csv"
EXPECTED_NORMALIZED_SOURCE_SHA256="f8bbff08a4139376b75d6be4023ce1ff50e87eb15f99242425e1d0ac2d666a64"

if command -v shasum >/dev/null 2>&1; then
    actual_source_sha256="$(tr -d '\r' < "$SOURCE" | shasum -a 256 | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
    actual_source_sha256="$(tr -d '\r' < "$SOURCE" | sha256sum | awk '{print $1}')"
else
    echo "Neither shasum nor sha256sum is available." >&2
    exit 1
fi

if [[ "$actual_source_sha256" != "$EXPECTED_NORMALIZED_SOURCE_SHA256" ]]; then
    echo "Source CSV hash changed; refusing to build an unreviewed derived dataset." >&2
    echo "Expected: $EXPECTED_NORMALIZED_SOURCE_SHA256" >&2
    echo "Actual:   $actual_source_sha256" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$DESTINATION")"
temporary="${DESTINATION}.tmp"
trap 'rm -f -- "$temporary"' EXIT
# Normalize the repository CSV's CRLF line endings to LF so the generated
# tracked artifact has a clean, platform-independent Git representation.
tr -d '\r' < "$SOURCE" > "$temporary"

append_class() {
    local source_class="$1"
    local derived_class="$2"
    local first_case_number="$3"
    awk -F',' \
        -v source_class="$source_class" \
        -v derived_class="$derived_class" \
        -v first_case_number="$first_case_number" \
        'BEGIN { generated = 0 }
         NR > 1 && $3 == source_class {
             seed = $4
             sub(/\r$/, "", seed)
             print first_case_number + generated ",an image of a " derived_class "," derived_class "," seed
             generated += 1
         }
         END {
             if (generated != 500) {
                 print "Expected 500 seed rows for " source_class ", found " generated > "/dev/stderr"
                 exit 1
             }
         }' "$SOURCE" >> "$temporary"
}

# Each missing preservation class reuses the ordered 500 seeds from the one
# preservation class already present in the same Confuse5 group.
append_class "german shepherd" "Chesapeake Bay retriever" 7500
append_class "german shepherd" "pug" 8000
append_class "persian cat" "Siamese cat" 8500
append_class "persian cat" "Egyptian cat" 9000
append_class "pomegranate" "fig" 9500
append_class "pomegranate" "Granny Smith" 10000
append_class "speedboat" "catamaran" 10500
append_class "speedboat" "schooner" 11000
append_class "tennis ball" "rugby ball" 11500
append_class "tennis ball" "ping-pong ball" 12000

line_count="$(wc -l < "$temporary" | tr -d '[:space:]')"
if [[ "$line_count" != "12501" ]]; then
    echo "Derived CSV has $line_count lines; expected 12501." >&2
    exit 1
fi

invalid_class_count="$(
    awk -F',' 'NR > 1 { counts[$3] += 1 }
        END {
            invalid = 0
            classes = 0
            for (name in counts) {
                classes += 1
                if (counts[name] != 500) invalid += 1
            }
            if (classes != 25) invalid += 1
            print invalid
        }' "$temporary"
)"
if [[ "$invalid_class_count" != "0" ]]; then
    echo "Derived CSV does not contain exactly 25 classes x 500 rows." >&2
    exit 1
fi

mv -- "$temporary" "$DESTINATION"
trap - EXIT
echo "Wrote $DESTINATION"
