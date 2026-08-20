#!/usr/bin/env python3
"""Download and record every model asset needed by the server experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    from huggingface_hub import HfApi, snapshot_download
    from torchvision.models import ResNet50_Weights

    # The experiment config is authoritative. Environment overrides would make
    # nominally identical runs resolve different base checkpoints.
    repo_id = config["base_model"]
    requested_revision = config.get("base_model_revision") or None
    info = HfApi().model_info(repo_id, revision=requested_revision)
    resolved_revision = info.sha
    snapshot_path = Path(snapshot_download(repo_id, revision=resolved_revision))

    weights = ResNet50_Weights.DEFAULT
    weights.get_state_dict(progress=True, check_hash=True)
    packages = {}
    for name in (
        "torch", "torchvision", "diffusers", "transformers", "accelerate",
        "huggingface-hub", "numpy", "pandas", "scipy",
    ):
        packages[name] = importlib.metadata.version(name)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "base_model": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot_path),
        "resnet_weights": str(weights),
        "resnet_url": weights.url,
        "packages": packages,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
