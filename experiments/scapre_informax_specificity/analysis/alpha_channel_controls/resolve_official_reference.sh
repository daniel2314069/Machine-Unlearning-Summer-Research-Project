#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
ARCHIVE_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
RUN_ID="formal_20260821T081723Z"
ARCHIVE_NAME="scapre_informax_seed_robustness_formal_20260821T081723Z_20260822T092030Z.tar.gz"
EXPECTED_SHA="df0874fea7c0998bbaf52782c763025c4ce7968134e8334e0688adec95453708"

reference_complete() {
  local root="$1"
  local seed
  [[ -f "$root/run_manifest.json" && -f "$root/protocol.csv" && -f "$root/protocol_manifest.json" ]] || return 1
  for seed in 20260820 20260821 20260822 20260823 20260824; do
    [[ -f "$root/seeds/$seed/evaluation/official/scores.csv" ]] || return 1
    [[ -f "$root/seeds/$seed/evaluation/official/evaluation_manifest.json" ]] || return 1
  done
}

ARCHIVE="$ARCHIVE_DIR/$ARCHIVE_NAME"
[[ -f "$ARCHIVE" ]] || {
  echo "ERROR: verified seed-robustness archive is missing: $ARCHIVE" >&2
  exit 2
}
ACTUAL_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || {
  echo "ERROR: official reference archive SHA-256 mismatch" >&2
  exit 2
}

# Never reuse the mutable historical run directory directly.  Resolve the
# official scores only from the checksum-pinned archive (or from a cache whose
# name and marker bind it to that exact archive).  worker.py performs the full
# protocol, source, asset, score, and evaluator-fingerprint validation before
# any GPU work starts.
CACHED="$STATE_DIR/official_reference_${RUN_ID}_${EXPECTED_SHA:0:16}"
if [[ -d "$CACHED" ]]; then
  reference_complete "$CACHED" || {
    echo "ERROR: cached official reference is incomplete: $CACHED" >&2
    exit 2
  }
  [[ -f "$CACHED/.archive_sha256" ]] || {
    echo "ERROR: cached official reference has no archive checksum marker: $CACHED" >&2
    exit 2
  }
  CACHED_SHA="$(tr -d '[:space:]' < "$CACHED/.archive_sha256")"
  [[ "$CACHED_SHA" == "$EXPECTED_SHA" ]] || {
    echo "ERROR: cached official reference checksum marker mismatch" >&2
    exit 2
  }
  printf '%s\n' "$CACHED"
  exit 0
fi

tar -tzf "$ARCHIVE" | awk '$0 ~ /^\// || $0 ~ /(^|\/)\.\.($|\/)/ {bad=1} END {exit bad ? 1 : 0}' || {
  echo "ERROR: official reference archive contains an unsafe path" >&2
  exit 2
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
tar -xzf "$ARCHIVE" -C "$TEMP_DIR" "${FILES[@]}"
reference_complete "$TEMP_DIR" || {
  echo "ERROR: verified archive did not yield complete official references" >&2
  exit 2
}
printf '%s\n' "$EXPECTED_SHA" > "$TEMP_DIR/.archive_sha256"
mv "$TEMP_DIR" "$CACHED"
trap - EXIT
printf '%s\n' "$CACHED"
