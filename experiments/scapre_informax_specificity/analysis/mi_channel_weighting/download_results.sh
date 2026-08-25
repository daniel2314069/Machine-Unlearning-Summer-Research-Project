#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <ssh-alias-or-user@host> <absolute-server-archive.tar.gz> <sha256>" >&2
  exit 2
fi
REMOTE="$1"
SERVER_ARCHIVE="$2"
EXPECTED_SHA="$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')"
[[ "$REMOTE" =~ ^[A-Za-z0-9._-]+(@[A-Za-z0-9._-]+)?$ ]] || {
  echo "ERROR: remote must be a simple SSH alias or user@host" >&2; exit 2;
}
[[ "$SERVER_ARCHIVE" == /home/tslin/Documents/jupyter_data/anLi/tmp/scapre_informax_mi_channel_weighting_*.tar.gz ]] || {
  echo "ERROR: archive path is outside the approved server package location" >&2; exit 2;
}
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: invalid SHA-256" >&2; exit 2; }
DESTINATION="$HOME/Downloads/$(basename "$SERVER_ARCHIVE")"
[[ ! -e "$DESTINATION" ]] || { echo "ERROR: refusing to overwrite: $DESTINATION" >&2; exit 1; }
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/scapre-mi-download.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
TEMP_ARCHIVE="$TEMP_DIR/$(basename "$SERVER_ARCHIVE")"
scp "$REMOTE:$SERVER_ARCHIVE" "$TEMP_ARCHIVE"
if command -v shasum >/dev/null; then
  ACTUAL_SHA="$(shasum -a 256 "$TEMP_ARCHIVE" | awk '{print $1}')"
else
  ACTUAL_SHA="$(sha256sum "$TEMP_ARCHIVE" | awk '{print $1}')"
fi
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || { echo "ERROR: downloaded archive checksum mismatch" >&2; exit 1; }
mv "$TEMP_ARCHIVE" "$DESTINATION"
trap - EXIT
rmdir "$TEMP_DIR"
echo "Downloaded: $DESTINATION"
echo "SHA-256 verified: $ACTUAL_SHA"
echo "Server archive and original outputs were not deleted."
