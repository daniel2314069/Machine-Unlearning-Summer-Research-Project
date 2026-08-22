#!/usr/bin/env python3
"""Run the public ScaPre ResNet-50 evaluator semantics over a paired protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["official", "matched_retain", "superclass_neutral"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def load_existing(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {(row["concept"], int(row["sample_index"])) for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    assets = json.loads(args.assets.read_text())
    evaluation = config["evaluation"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    results_path = args.output_dir / "scores.csv"
    metadata_path = args.output_dir / "evaluation_manifest.json"

    fingerprint = {
        "variant": args.variant,
        "checkpoint_sha256": sha256(args.checkpoint),
        "protocol_sha256": sha256(args.protocol),
        "base_model": assets["base_model"],
        "resolved_revision": assets["resolved_revision"],
        "num_inference_steps": evaluation["num_inference_steps"],
        "guidance_scale": evaluation["guidance_scale"],
        "height": evaluation["height"],
        "width": evaluation["width"],
        "classifier": evaluation["classifier"],
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "diffusers": importlib.metadata.version("diffusers"),
    }
    previous = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        assets["snapshot_path"], torch_dtype=torch.float16, local_files_only=True
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    incompatible = pipe.unet.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint incompatibility: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights).to(args.device).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    fingerprint["scheduler_class"] = pipe.scheduler.__class__.__name__
    fingerprint["scheduler_config"] = json.loads(
        json.dumps(dict(pipe.scheduler.config), default=str)
    )
    if previous is not None:
        for key, value in fingerprint.items():
            if previous.get(key) != value:
                raise RuntimeError(f"evaluation fingerprint mismatch for {key}")
    metadata_path.write_text(json.dumps(fingerprint, indent=2, default=str) + "\n")

    with args.protocol.open(newline="") as handle:
        protocol_rows = list(csv.DictReader(handle))
    completed = load_existing(results_path)
    fields = [
        "variant", "group", "role", "concept", "sample_index", "prompt", "seed",
        "seed_source", "image_path", "prediction", "correct",
    ]
    write_header = not results_path.exists()
    with results_path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if write_header:
            writer.writeheader()
            output.flush()
        for index, row in enumerate(protocol_rows, 1):
            key = (row["concept"], int(row["sample_index"]))
            if key in completed:
                continue
            seed = int(row["seed"])
            image_path = images_dir / slug(row["concept"]) / f"{int(row['sample_index']):04d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
            else:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                image = pipe(
                    row["prompt"],
                    num_inference_steps=evaluation["num_inference_steps"],
                    guidance_scale=evaluation["guidance_scale"],
                    height=evaluation["height"],
                    width=evaluation["width"],
                ).images[0]
                image.save(image_path)

            inputs = preprocess(image).unsqueeze(0).to(args.device)
            with torch.no_grad():
                prediction_index = classifier(inputs).argmax(dim=-1).item()
            prediction = categories[prediction_index].lower()
            label = row["concept"].lower()
            correct = int(label in prediction or prediction in label)
            writer.writerow({
                "variant": args.variant,
                "group": row["group"],
                "role": row["role"],
                "concept": row["concept"],
                "sample_index": row["sample_index"],
                "prompt": row["prompt"],
                "seed": row["seed"],
                "seed_source": row["seed_source"],
                "image_path": str(image_path.resolve()),
                "prediction": prediction,
                "correct": correct,
            })
            output.flush()
            print(f"[{args.variant} {index}/{len(protocol_rows)}] {label}: {prediction} ({correct})", flush=True)

    with results_path.open(newline="") as handle:
        final_rows = list(csv.DictReader(handle))
    final_keys = [
        (row["concept"], int(row["sample_index"])) for row in final_rows
    ]
    if len(final_rows) != len(protocol_rows) or len(set(final_keys)) != len(protocol_rows):
        raise RuntimeError("evaluation did not produce exactly one result per protocol row")
    missing_images = [row["image_path"] for row in final_rows if not Path(row["image_path"]).is_file()]
    if missing_images:
        raise RuntimeError(f"evaluation is missing {len(missing_images)} generated images")
    (args.output_dir / "COMPLETED").write_text("ok\n")


if __name__ == "__main__":
    main()
