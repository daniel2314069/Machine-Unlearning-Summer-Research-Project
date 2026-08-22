#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <server-archive-absolute-path> <sha256> [ssh-host]" >&2
  exit 2
fi
ARCHIVE="$1"
EXPECTED_SHA="$2"
HOST="${3:-tslin}"
[[ "$ARCHIVE" == /home/tslin/Documents/jupyter_data/anLi/tmp/*.tar.gz ]] || {
  echo "ERROR: unexpected server archive path" >&2; exit 2;
}
[[ "$EXPECTED_SHA" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "ERROR: invalid SHA-256" >&2; exit 2; }
DESTINATION="$HOME/Downloads"
mkdir -p "$DESTINATION"
scp "$HOST:$ARCHIVE" "$DESTINATION/"
scp "$HOST:$ARCHIVE.sha256" "$DESTINATION/"
scp "$HOST:$ARCHIVE.cleanup.json" "$DESTINATION/"
LOCAL="$DESTINATION/$(basename "$ARCHIVE")"
if command -v shasum >/dev/null; then
  ACTUAL="$(shasum -a 256 "$LOCAL" | awk '{print $1}')"
else
  ACTUAL="$(sha256sum "$LOCAL" | awk '{print $1}')"
fi
[[ "$ACTUAL" == "$EXPECTED_SHA" ]] || { echo "ERROR: downloaded checksum mismatch" >&2; exit 1; }
echo "Verified download: $LOCAL"
echo "SHA-256: $ACTUAL"
