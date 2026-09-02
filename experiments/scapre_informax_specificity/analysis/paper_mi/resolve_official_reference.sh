#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_EXPERIMENT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ALPHA_STATE="$PARENT_EXPERIMENT/analysis/alpha_channel_controls/.server"
STATE_DIR="$SCRIPT_DIR/.server"
ARCHIVE_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
RUN_ID="formal_20260821T081723Z"
ARCHIVE_NAME="scapre_informax_seed_robustness_formal_20260821T081723Z_20260822T092030Z.tar.gz"
EXPECTED_SHA="df0874fea7c0998bbaf52782c763025c4ce7968134e8334e0688adec95453708"
CACHE_NAME="official_reference_${RUN_ID}_${EXPECTED_SHA:0:16}"

reference_complete() {
  local root="$1"
  local seed
  [[ -f "$root/run_manifest.json" && -f "$root/protocol.csv" && -f "$root/protocol_manifest.json" ]] || return 1
  for seed in 20260820 20260821 20260822 20260823 20260824; do
    [[ -f "$root/seeds/$seed/evaluation/official/scores.csv" ]] || return 1
    [[ -f "$root/seeds/$seed/evaluation/official/evaluation_manifest.json" ]] || return 1
  done
}

validate_cache() {
  local cached="$1"
  reference_complete "$cached" || return 1
  [[ -f "$cached/.archive_sha256" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$cached/.archive_sha256")" == "$EXPECTED_SHA" ]]
}

# The storage cleanup allowlist removed runs and tmp archives, but not these
# checksum-bound extracted caches. Prefer them before looking for the archive.
for cached in "$STATE_DIR/$CACHE_NAME" "$ALPHA_STATE/$CACHE_NAME"; do
  if [[ -d "$cached" ]]; then
    if validate_cache "$cached"; then
      printf '%s\n' "$cached"
      exit 0
    fi
    echo "NOTICE: retained official-reference cache is invalid and will not be used: $cached" >&2
    exit 3
  fi
done

ARCHIVE="$ARCHIVE_DIR/$ARCHIVE_NAME"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "NOTICE: verified baseline archive/cache was cleaned; repository baseline will be regenerated." >&2
  exit 3
fi
ACTUAL_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || {
  echo "NOTICE: official reference archive SHA-256 mismatch; baseline will be regenerated" >&2
  exit 3
}
tar -tzf "$ARCHIVE" | awk '$0 ~ /^\// || $0 ~ /(^|\/)\.\.($|\/)/ {bad=1} END {exit bad ? 1 : 0}' || {
  echo "NOTICE: unsafe official reference archive will not be used; baseline will be regenerated" >&2
  exit 3
}

mkdir -p "$STATE_DIR"
TEMP_DIR="$(mktemp -d "$STATE_DIR/official_reference.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
FILES=(run_manifest.json protocol.csv protocol_manifest.json)
for seed in 20260820 20260821 20260822 20260823 20260824; do
  FILES+=(
    "seeds/$seed/evaluation/official/scores.csv"
    "seeds/$seed/evaluation/official/evaluation_manifest.json"
  )
done
if ! tar -xzf "$ARCHIVE" -C "$TEMP_DIR" "${FILES[@]}"; then
  echo "NOTICE: archive extraction failed; baseline will be regenerated" >&2
  exit 3
fi
reference_complete "$TEMP_DIR" || {
  echo "NOTICE: archive is incomplete; baseline will be regenerated" >&2
  exit 3
}
printf '%s\n' "$EXPECTED_SHA" > "$TEMP_DIR/.archive_sha256"
mv "$TEMP_DIR" "$STATE_DIR/$CACHE_NAME"
trap - EXIT
printf '%s\n' "$STATE_DIR/$CACHE_NAME"
