#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.server"
CONFIG="$SCRIPT_DIR/config.json"
REQUIREMENTS="$SCRIPT_DIR/requirements_server.txt"

if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
  echo "ERROR: activate the GPU-server Conda environment first: conda activate MU" >&2
  exit 2
fi

PYTHON_BIN="$(command -v python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python is not available in the active MU environment" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" || ! -f "$REQUIREMENTS" ]]; then
  echo "ERROR: experiment config or server requirements are missing" >&2
  exit 2
fi

mkdir -p "$STATE_DIR"
echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" "$SCRIPT_DIR/build_protocol.py" \
  --config "$CONFIG" \
  --output "$STATE_DIR/formal_protocol_preflight.csv" \
  --profile formal > "$STATE_DIR/formal_protocol_preflight.json"

"$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS"
"$PYTHON_BIN" -m pip check

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"$PYTHON_BIN" "$SCRIPT_DIR/prefetch_assets.py" \
  --config "$CONFIG" \
  --manifest "$STATE_DIR/assets_manifest.json"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$STATE_DIR/SETUP_COMPLETE"
echo "Setup complete. Assets: $STATE_DIR/assets_manifest.json"
echo "Next: $SCRIPT_DIR/run_server.sh smoke"
