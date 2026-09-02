#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_EXPERIMENT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="${1:-}"
ASSETS="$PARENT_EXPERIMENT/.server/assets_manifest.json"
BASE_CONFIG="$PARENT_EXPERIMENT/config.json"
REPORT="$RUN_DIR/asset_preflight.json"
MODE_FILE="$RUN_DIR/asset_provisioning_mode"

[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || { echo "ERROR: explicit run directory is required" >&2; exit 2; }
[[ "${CONDA_DEFAULT_ENV:-}" == "MU" ]] || { echo "ERROR: asset setup requires active Conda MU" >&2; exit 2; }
PYTHON_BIN="$(command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: python is unavailable in Conda MU" >&2; exit 2; }
[[ -f "$BASE_CONFIG" && -x "$PARENT_EXPERIMENT/setup_server.sh" ]] || {
  echo "ERROR: parent ScaPre setup files are missing" >&2
  exit 2
}

mkdir -p "$PARENT_EXPERIMENT/.server"
command -v flock >/dev/null || { echo "ERROR: required command is unavailable: flock" >&2; exit 2; }
LOCK_FILE="$PARENT_EXPERIMENT/.server/paper_mi_asset_setup.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "ERROR: another ScaPre asset setup is active" >&2; exit 2; }

if "$PYTHON_BIN" "$SCRIPT_DIR/validate_assets.py" \
    --manifest "$ASSETS" --base-config "$BASE_CONFIG" --report "$REPORT"; then
  printf 'reused_validated_cache\n' > "$MODE_FILE"
  echo "ScaPre model assets are complete; reusing the validated cache."
  exit 0
fi

echo "Recorded assets are absent or incomplete; provisioning them automatically."
echo "Existing Hugging Face and Torch cache files will be reused when available."
"$PARENT_EXPERIMENT/setup_server.sh"
"$PYTHON_BIN" "$SCRIPT_DIR/validate_assets.py" \
  --manifest "$ASSETS" --base-config "$BASE_CONFIG" --report "$REPORT"
printf 'automatically_provisioned\n' > "$MODE_FILE"
echo "Automatic ScaPre model-asset provisioning passed."
