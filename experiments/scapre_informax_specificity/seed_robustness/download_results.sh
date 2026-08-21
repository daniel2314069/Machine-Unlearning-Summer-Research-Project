#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <user@server-host> <absolute-server-archive.tar.gz> <sha256>" >&2
  exit 2
fi
REMOTE="$1"
SERVER_ARCHIVE="$2"
EXPECTED_SHA="$3"
if [[ ! "$REMOTE" =~ ^[^/@:]+@[^/:]+$ ]]; then
  echo "ERROR: first argument must look like user@server-host" >&2
  exit 2
fi
if [[ "$SERVER_ARCHIVE" != /home/tslin/Documents/jupyter_data/anLi/tmp/scapre_informax_seed_robustness_*.tar.gz ]]; then
  echo "ERROR: archive must be an exact seed-robustness archive under the approved server tmp directory" >&2
  exit 2
fi
if [[ ! "$EXPECTED_SHA" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "ERROR: SHA-256 must contain exactly 64 hexadecimal characters" >&2
  exit 2
fi
DESTINATION="$HOME/Downloads/$(basename "$SERVER_ARCHIVE")"
DESTINATION_CLEANUP="$DESTINATION.cleanup.json"
if [[ -e "$DESTINATION" || -e "$DESTINATION_CLEANUP" ]]; then
  echo "ERROR: refusing to overwrite an existing archive or cleanup manifest" >&2
  exit 2
fi
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/scapre-seed-download.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
TEMP_ARCHIVE="$TEMP_DIR/$(basename "$SERVER_ARCHIVE")"
TEMP_CLEANUP="$TEMP_ARCHIVE.cleanup.json"
scp "$REMOTE:$SERVER_ARCHIVE" "$TEMP_ARCHIVE"
scp "$REMOTE:$SERVER_ARCHIVE.cleanup.json" "$TEMP_CLEANUP"
if command -v shasum >/dev/null; then
  ACTUAL_SHA="$(shasum -a 256 "$TEMP_ARCHIVE" | awk '{print $1}')"
else
  ACTUAL_SHA="$(sha256sum "$TEMP_ARCHIVE" | awk '{print $1}')"
fi
ACTUAL_SHA_LOWER="$(printf '%s' "$ACTUAL_SHA" | tr '[:upper:]' '[:lower:]')"
EXPECTED_SHA_LOWER="$(printf '%s' "$EXPECTED_SHA" | tr '[:upper:]' '[:lower:]')"
if [[ "$ACTUAL_SHA_LOWER" != "$EXPECTED_SHA_LOWER" ]]; then
  echo "ERROR: downloaded archive checksum mismatch" >&2
  exit 1
fi
MANIFEST_SHA="$(jq -r '.archive_sha256' "$TEMP_CLEANUP" | tr '[:upper:]' '[:lower:]')"
if [[ "$(jq -r '.status' "$TEMP_CLEANUP")" != "passed" || \
      "$MANIFEST_SHA" != "$EXPECTED_SHA_LOWER" ]]; then
  echo "ERROR: cleanup manifest does not confirm the verified archive" >&2
  exit 1
fi
mv "$TEMP_ARCHIVE" "$DESTINATION"
mv "$TEMP_CLEANUP" "$DESTINATION_CLEANUP"
trap - EXIT
rmdir "$TEMP_DIR"
echo "Downloaded: $DESTINATION"
echo "SHA-256 verified: $ACTUAL_SHA"
echo "Cleanup manifest: $DESTINATION_CLEANUP"
echo "Server archive and original non-image results were not deleted."
