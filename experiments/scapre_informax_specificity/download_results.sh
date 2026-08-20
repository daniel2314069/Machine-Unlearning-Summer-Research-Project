#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <user@server-host> <absolute-server-archive.tar.gz> <expected-sha256>" >&2
  exit 2
fi
SERVER="$1"
REMOTE_ARCHIVE="$2"
EXPECTED_SHA256="$3"
if [[ "$REMOTE_ARCHIVE" != /home/tslin/Documents/jupyter_data/anLi/tmp/*.tar.gz ]]; then
  echo "ERROR: archive must be the verified package_results.sh output" >&2
  exit 2
fi
if [[ ! "$EXPECTED_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "ERROR: expected SHA-256 must contain exactly 64 hexadecimal characters" >&2
  exit 2
fi

DESTINATION="$HOME/Downloads/$(basename "$REMOTE_ARCHIVE")"
if [[ -e "$DESTINATION" ]]; then
  echo "ERROR: local destination already exists: $DESTINATION" >&2
  exit 2
fi
scp "$SERVER:$REMOTE_ARCHIVE" "$DESTINATION"
ACTUAL_SHA256="$(shasum -a 256 "$DESTINATION" | awk '{print $1}')"
ACTUAL_NORMALIZED="$(printf '%s' "$ACTUAL_SHA256" | tr '[:upper:]' '[:lower:]')"
EXPECTED_NORMALIZED="$(printf '%s' "$EXPECTED_SHA256" | tr '[:upper:]' '[:lower:]')"
if [[ "$ACTUAL_NORMALIZED" != "$EXPECTED_NORMALIZED" ]]; then
  echo "ERROR: checksum mismatch for $DESTINATION" >&2
  exit 1
fi
echo "Downloaded and verified: $DESTINATION"
echo "SHA-256: $ACTUAL_SHA256"
