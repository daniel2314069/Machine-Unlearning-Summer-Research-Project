#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/tslin/Documents/jupyter_data/anLi/machine_unlearning"
REMOTE_SCRIPT="$REMOTE_ROOT/experiments/scapre_informax_specificity/analysis/alpha_channel_controls"
MANIFEST_CONTENT="$(ssh tslin "RUN=\$(tr -d '\\r\\n' < '$REMOTE_SCRIPT/.server/latest_run'); cat \"\$RUN/archive_manifest.json\"")"
ARCHIVE="$(printf '%s\n' "$MANIFEST_CONTENT" | sed -n 's/^  "archive": "\([^"]*\)".*/\1/p')"
EXPECTED="$(printf '%s\n' "$MANIFEST_CONTENT" | sed -n 's/^  "sha256": "\([^"]*\)".*/\1/p')"
[[ "$ARCHIVE" == /home/tslin/Documents/jupyter_data/anLi/tmp/scapre_informax_alpha_channel_controls_*.tar.gz ]] || {
  echo "ERROR: remote archive path is outside the approved result directory" >&2
  exit 2
}
[[ "$EXPECTED" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: invalid remote checksum" >&2; exit 2; }
DESTINATION="$HOME/Downloads/$(basename "$ARCHIVE")"
[[ ! -e "$DESTINATION" ]] || { echo "ERROR: destination already exists: $DESTINATION" >&2; exit 2; }
scp "tslin:$ARCHIVE" "$DESTINATION"
ACTUAL="$(shasum -a 256 "$DESTINATION" | awk '{print $1}')"
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo "ERROR: downloaded archive checksum mismatch" >&2; exit 1; }
if ssh tslin "test -f '$ARCHIVE.cleanup.json'"; then
  scp "tslin:$ARCHIVE.cleanup.json" "$DESTINATION.cleanup.json"
fi
echo "Downloaded: $DESTINATION"
echo "SHA-256: $ACTUAL (verified)"
[[ ! -f "$DESTINATION.cleanup.json" ]] || echo "Cleanup record: $DESTINATION.cleanup.json"
