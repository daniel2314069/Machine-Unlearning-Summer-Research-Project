#!/usr/bin/env python3
"""Validate that the recorded ScaPre model assets still exist and are usable."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


REQUIRED_SD_FILES = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "MU":
        raise RuntimeError("asset validation requires active Conda MU")
    if not args.manifest.is_file():
        raise RuntimeError(f"asset manifest is missing: {args.manifest}")
    manifest = json.loads(args.manifest.read_text())
    config = json.loads(args.base_config.read_text())
    if manifest.get("base_model") != config["base_model"]:
        raise RuntimeError("asset manifest base model differs from the experiment config")
    if manifest.get("config_sha256") != sha256(args.base_config):
        raise RuntimeError("asset manifest config hash differs from the experiment config")
    snapshot = Path(manifest["snapshot_path"])
    if not snapshot.is_absolute() or not snapshot.is_dir():
        raise RuntimeError(f"recorded model snapshot is unavailable: {snapshot}")

    recorded = manifest.get("downloaded_files")
    if not isinstance(recorded, list) or not recorded:
        raise RuntimeError("asset manifest has no recorded model files")
    recorded_names = {item.get("path") for item in recorded}
    missing_required = sorted(REQUIRED_SD_FILES - recorded_names)
    if missing_required or not any(
        isinstance(name, str) and name.startswith("tokenizer/") for name in recorded_names
    ):
        raise RuntimeError(f"asset manifest lacks required SD components: {missing_required}")
    total_bytes = 0
    for item in recorded:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe model asset path: {relative}")
        path = snapshot / relative
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise RuntimeError(f"model asset is missing or has the wrong size: {path}")
        total_bytes += path.stat().st_size
    if total_bytes != int(manifest.get("total_model_bytes", -1)):
        raise RuntimeError("recorded model asset size total changed")
    if total_bytes > 7 * 1024**3:
        raise RuntimeError("recorded model assets exceed the 7 GiB safety limit")

    for package, expected in manifest.get("packages", {}).items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"package version changed: {package}")
    if Path(sys.executable).resolve() != Path(manifest["python_executable"]).resolve():
        raise RuntimeError("asset manifest was created by a different Python environment")

    import torch
    from torchvision.models import ResNet50_Weights

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    weights = ResNet50_Weights.DEFAULT
    if str(weights) != manifest.get("resnet_weights") or weights.url != manifest.get("resnet_url"):
        raise RuntimeError("recorded ResNet50 classifier weights changed")
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not checkpoint.is_file():
        raise RuntimeError(f"ResNet50 checkpoint is unavailable: {checkpoint}")
    expected_prefix = checkpoint.stem.rsplit("-", 1)[-1]
    if not sha256(checkpoint).startswith(expected_prefix):
        raise RuntimeError("ResNet50 checkpoint hash prefix mismatch")

    report = {
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "base_model": manifest["base_model"],
        "resolved_revision": manifest["resolved_revision"],
        "snapshot_path": str(snapshot),
        "model_file_count": len(recorded),
        "total_model_bytes": total_bytes,
        "resnet_checkpoint": str(checkpoint),
        "resnet_checkpoint_sha256": sha256(checkpoint),
        "cuda_device": torch.cuda.get_device_name(0),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
