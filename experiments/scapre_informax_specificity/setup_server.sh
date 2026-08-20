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

if ! "$PYTHON_BIN" -c 'import importlib.metadata as m; assert m.version("torch").split("+")[0] == "2.6.0"; assert m.version("torchvision").split("+")[0] == "0.21.0"' 2>/dev/null; then
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0
fi
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS"

# MU is shared with unrelated notebook tooling, so a global `pip check` can
# fail on packages that this experiment never imports. Validate the pinned
# experiment runtime directly, including the PEFT version required by
# Diffusers when an older PEFT is already present in MU.
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata

expected = {
    "accelerate": "1.5.2",
    "diffusers": "0.35.0",
    "huggingface-hub": "0.34.4",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "peft": "0.17.1",
    "pillow": "11.1.0",
    "safetensors": "0.5.2",
    "scipy": "1.15.2",
    "sentencepiece": "0.2.0",
    "transformers": "4.49.0",
}
for package, required in expected.items():
    actual = metadata.version(package)
    if actual != required:
        raise RuntimeError(f"{package}: expected {required}, found {actual}")

import accelerate
import diffusers
import huggingface_hub
import peft
import torch
import torchvision
import transformers

if metadata.version("torch").split("+")[0] != "2.6.0":
    raise RuntimeError(f"unexpected torch version: {metadata.version('torch')}")
if metadata.version("torchvision").split("+")[0] != "0.21.0":
    raise RuntimeError(
        f"unexpected torchvision version: {metadata.version('torchvision')}"
    )
print("Experiment dependency validation passed")
PY

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"$PYTHON_BIN" "$SCRIPT_DIR/prefetch_assets.py" \
  --config "$CONFIG" \
  --manifest "$STATE_DIR/assets_manifest.json"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$STATE_DIR/SETUP_COMPLETE"
echo "Setup complete. Assets: $STATE_DIR/assets_manifest.json"
echo "Next: $SCRIPT_DIR/run_server.sh smoke"
