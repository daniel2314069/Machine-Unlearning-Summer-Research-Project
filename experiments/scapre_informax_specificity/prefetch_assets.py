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


# StableDiffusionPipeline.from_pretrained() reads the componentized Diffusers
# snapshot below. Do not fetch root checkpoints, duplicate PyTorch .bin files,
# fp16 variants, non-EMA variants, ONNX, or Flax artifacts.
SD15_ALLOW_PATTERNS = (
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "safety_checker/config.json",
    "safety_checker/model.safetensors",
)

REQUIRED_MODEL_FILES = (
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "safety_checker/config.json",
    "safety_checker/model.safetensors",
)

# The required SD 1.5 component weights are about 5.5 GB. This guard catches
# an accidental return to downloading duplicate full-repository artifacts.
MAX_REQUIRED_MODEL_BYTES = 7 * 1024**3


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
    snapshot_path = Path(snapshot_download(
        repo_id,
        revision=resolved_revision,
        allow_patterns=list(SD15_ALLOW_PATTERNS),
    ))

    missing = [
        relative for relative in REQUIRED_MODEL_FILES
        if not (snapshot_path / relative).is_file()
    ]
    tokenizer_files = sorted(
        path for path in (snapshot_path / "tokenizer").glob("*") if path.is_file()
    )
    if missing or not tokenizer_files:
        raise RuntimeError(
            f"filtered SD 1.5 snapshot is incomplete: missing={missing}, "
            f"tokenizer_files={len(tokenizer_files)}"
        )

    recorded_paths = [snapshot_path / item for item in REQUIRED_MODEL_FILES]
    recorded_paths.extend(tokenizer_files)
    downloaded_files = []
    total_model_bytes = 0
    for path in sorted(set(recorded_paths)):
        size = path.stat().st_size
        total_model_bytes += size
        downloaded_files.append({
            "path": str(path.relative_to(snapshot_path)),
            "size_bytes": size,
        })
    if total_model_bytes > MAX_REQUIRED_MODEL_BYTES:
        raise RuntimeError(
            "required SD 1.5 assets exceeded the 7 GiB safety limit: "
            f"{total_model_bytes} bytes"
        )

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
        "snapshot_allow_patterns": list(SD15_ALLOW_PATTERNS),
        "downloaded_files": downloaded_files,
        "total_model_bytes": total_model_bytes,
        "maximum_model_bytes": MAX_REQUIRED_MODEL_BYTES,
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
