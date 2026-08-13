#!/usr/bin/env python
"""Run the server-side sequential OCE object persistence experiment.

The orchestration in this file does not change the OCE objective. Checkpoint
construction calls the repository's ``oce.Orthogonal_Erase`` implementation and
feeds each completed single-concept checkpoint into the next edit.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
OCE_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT = HERE / "outputs" / "sequential_oce_object_v1"
OCE_SOURCE = OCE_ROOT / "oce.py"

OFFICIAL_PAIRS = [
    ("airplane", "sky"),
    ("automobile", "truck"),
    ("bird", "cat"),
    ("cat", "dog"),
    ("deer", "horse"),
    ("dog", "cat"),
    ("frog", "bird"),
    ("horse", "deer"),
    ("ship", "airplane"),
    ("truck", "ship"),
]
CONDITIONS = ("retain_once", "retain_always")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def event(output_dir: Path, phase: str, message: str, **details: Any) -> None:
    row = {"timestamp": utc_now(), "phase": phase, "message": message, **details}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[{phase}] {message}{(' ' + suffix) if suffix else ''}", flush=True)


def update_state(output_dir: Path, phase: str, status: str, **details: Any) -> None:
    path = output_dir / "run_state.json"
    state = read_json(path) if path.is_file() else {}
    state[phase] = {"status": status, "updated_at": utc_now(), **details}
    write_json(path, state)


def load_config(path: Path) -> dict[str, Any]:
    return read_json(path.resolve())


def pair_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    return [dict(row) for row in config["targets"]]


def targets(config: Mapping[str, Any]) -> list[str]:
    return [row["target"] for row in pair_rows(config)]


def anchors(config: Mapping[str, Any]) -> list[str]:
    return [row["anchor"] for row in pair_rows(config)]


def resolved_cg_path(config_path: Path, config: Mapping[str, Any]) -> Path:
    return (config_path.resolve().parent / str(config["cg_path"])).resolve()


def validate_config(
    config_path: Path, config: Mapping[str, Any], require_artifacts: bool = True
) -> dict[str, Any]:
    configured_pairs = [
        (str(row["target"]).casefold(), str(row["anchor"]).casefold())
        for row in pair_rows(config)
    ]
    if configured_pairs != OFFICIAL_PAIRS:
        raise ValueError(
            "The ordered target-anchor mapping differs from the fixed official mapping"
        )
    target_set = set(targets(config))
    anchor_set = set(anchors(config))
    candidates = list(config["qualification"]["candidates"])
    if len(candidates) != len(set(candidates)):
        raise ValueError("Qualification candidates must be unique")
    overlap = sorted(set(candidates) & (target_set | anchor_set))
    if overlap:
        raise ValueError(f"Qualification candidates overlap targets/anchors: {overlap}")
    if int(config["qualification"]["images_per_candidate"]) != 20:
        raise ValueError("Qualification must use exactly 20 images per candidate")
    if int(config["generation"]["images_per_concept"]) != 100:
        raise ValueError("The first formal experiment must use 100 images per cell")
    if not bool(config["oce"]["always_preserve_current_anchor"]):
        raise ValueError("The fixed object protocol requires preserving each current anchor")
    if float(config["oce"]["preserve_concept_scale"]) <= 0:
        raise ValueError("Explicit local retain evaluation requires a positive retain weight")
    if config["storage"]["image_retention"] not in {"keep", "delete-after-eval"}:
        raise ValueError("Unsupported image retention policy")
    cg_path = resolved_cg_path(config_path, config)
    if require_artifacts and not cg_path.is_file():
        raise FileNotFoundError(f"Missing Cg.pt: {cg_path}")
    if cg_path != (OCE_ROOT / "Cg.pt").resolve():
        raise ValueError("Current oce.py requires the repository-level Cg.pt")
    formal_cells = 11 + 2 * sum(step + 1 for step in range(1, 11))
    formal_images = formal_cells * int(config["generation"]["images_per_concept"])
    if formal_cells != 141 or formal_images != 14100:
        raise AssertionError("Formal experiment count must resolve to 141 cells / 14,100 images")
    return {
        "targets": len(target_set),
        "anchors": len(anchor_set),
        "candidates": len(candidates),
        "formal_cells": formal_cells,
        "formal_images": formal_images,
        "qualification_images_per_candidate": 20,
        "sets_disjoint": True,
    }


def make_protocol(
    config_path: Path,
    output_dir: Path,
    allow_downloads: bool,
    image_retention_override: str | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    audit = validate_config(config_path, config, require_artifacts=True)
    effective_image_retention = (
        image_retention_override or config["storage"]["image_retention"]
    )
    sources = {
        "config.json": sha256_file(config_path.resolve()),
        "oce.py": sha256_file(OCE_SOURCE),
        "Cg.pt": sha256_file(resolved_cg_path(config_path, config)),
        "runner": sha256_file(Path(__file__).resolve()),
    }
    fingerprint_input = {
        "experiment": config["experiment_name"],
        "config": config,
        "source_hashes": sources,
        "local_files_only": not allow_downloads,
        "effective_image_retention": effective_image_retention,
    }
    return {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "base_model": config["model_id"],
        "target_anchor_mapping": pair_rows(config),
        "conditions": list(CONDITIONS),
        "planned_counts": audit,
        "source_hashes": sources,
        "protocol_fingerprint": stable_hash(fingerprint_input),
        "local_files_only": not allow_downloads,
        "effective_image_retention": effective_image_retention,
        "resolved_at": utc_now(),
        "software": {"python": sys.version, "platform": platform.platform()},
        "config": config,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    protocol = make_protocol(
        config_path, output_dir, args.allow_downloads, args.image_retention
    )
    existing_path = output_dir / "resolved_protocol.json"
    if existing_path.is_file():
        existing = read_json(existing_path)
        if existing.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise RuntimeError(
                "The output directory already contains a different protocol. "
                "Choose a new output directory so prior results are not mixed or overwritten."
            )
    update_state(output_dir, "preflight", "running")
    write_json(output_dir / "resolved_protocol.json", protocol)
    write_csv(
        output_dir / "inputs" / "target_anchor_mapping.csv",
        [
            {"step": index, **row}
            for index, row in enumerate(protocol["target_anchor_mapping"], start=1)
        ],
    )
    config = protocol["config"]
    seed_start = int(config["generation"]["seed_start"])
    n = int(config["generation"]["images_per_concept"])
    write_csv(
        output_dir / "inputs" / "formal_seeds.csv",
        [{"sample_index": i, "seed": seed_start + i} for i in range(n)],
    )
    event(
        output_dir,
        "preflight",
        "validated protocol",
        formal_cells=141,
        formal_images=14100,
    )
    update_state(
        output_dir,
        "preflight",
        "complete",
        protocol_fingerprint=protocol["protocol_fingerprint"],
    )
    return protocol


def require_protocol(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    expected = make_protocol(
        Path(args.config).resolve(),
        output_dir,
        args.allow_downloads,
        args.image_retention,
    )
    path = output_dir / "resolved_protocol.json"
    if not path.is_file():
        return preflight(args)
    current = read_json(path)
    if current.get("protocol_fingerprint") != expected["protocol_fingerprint"]:
        raise RuntimeError(
            "Protocol sources/configuration changed. Use a new output directory or "
            "remove the stale output after reviewing it."
        )
    return current


def torch_dtype(name: str) -> Any:
    import torch

    values = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in values:
        raise ValueError(f"Unsupported dtype: {name}")
    return values[name]


def release_cuda(*objects: Any) -> None:
    import torch

    for value in objects:
        movable = getattr(value, "model", value)
        if hasattr(movable, "to"):
            movable.to("cpu")
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pipeline(protocol: Mapping[str, Any], edit_only: bool = False) -> Any:
    from diffusers import DiffusionPipeline

    config = protocol["config"]
    dtype_name = (
        config["oce"]["edit_dtype"] if edit_only else config["generation"]["dtype"]
    )
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(dtype_name),
        "safety_checker": None,
        "local_files_only": bool(protocol["local_files_only"]),
    }
    if edit_only:
        kwargs["vae"] = None
    pipe = DiffusionPipeline.from_pretrained(config["model_id"], **kwargs).to(
        config["device"]
    )
    pipe.set_progress_bar_config(disable=True)
    if not edit_only:
        expected_scheduler = str(config["generation"]["scheduler"])
        observed_scheduler = type(pipe.scheduler).__name__
        if observed_scheduler != expected_scheduler:
            raise RuntimeError(
                f"Expected scheduler {expected_scheduler}, got {observed_scheduler}"
            )
    return pipe


def selected_projection_state(unet: Any) -> dict[str, Any]:
    state = {
        f"{name}.weight": module.weight.detach().cpu().clone()
        for name, module in unet.named_modules()
        if "attn2" in name and name.endswith("to_v")
    }
    if len(state) != 16:
        raise RuntimeError(f"Expected 16 selected tensors, got {len(state)}")
    return state


def apply_projection_state(unet: Any, state: Mapping[str, Any]) -> None:
    import torch

    modules = dict(unet.named_modules())
    expected = {
        f"{name}.weight"
        for name in modules
        if "attn2" in name and name.endswith("to_v")
    }
    if set(state) != expected:
        raise RuntimeError("Checkpoint keys differ from the repository-selected tensor set")
    with torch.no_grad():
        for key, value in state.items():
            module_name = key[: -len(".weight")]
            destination = modules[module_name].weight
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))


def expand_object_pair(target: str, anchor: str) -> tuple[list[str], list[str]]:
    edit_concepts = [target]
    guide_concepts = [anchor]
    suffixes = ("image of {}", "photo of {}", "portrait of {}", "picture of {}", "painting of {}")
    edit_concepts.extend(template.format(target) for template in suffixes)
    guide_concepts.extend(template.format(anchor) for template in suffixes)
    return edit_concepts, guide_concepts


def qualification_summary_path(output_dir: Path) -> Path:
    return output_dir / "qualification" / "summary.json"


def require_selected_x(output_dir: Path, protocol: Mapping[str, Any]) -> str:
    path = qualification_summary_path(output_dir)
    if not path.is_file():
        raise RuntimeError("Qualification is incomplete; run the qualify phase first")
    summary = read_json(path)
    if summary.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
        raise RuntimeError("Qualification protocol fingerprint mismatch")
    selected = summary.get("selected_concept")
    if not selected:
        raise RuntimeError("No qualification candidate met the configured acceptance rule")
    return str(selected)


class ClipClassifier:
    def __init__(self, model_id: str, device: str, local_files_only: bool):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(
            model_id, local_files_only=local_files_only
        ).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            model_id, local_files_only=local_files_only
        )

    def classify(
        self, image_paths: Sequence[Path], class_texts: Sequence[str], batch_size: int
    ) -> list[list[float]]:
        from PIL import Image

        all_probs: list[list[float]] = []
        with self.torch.inference_mode():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                images = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB").copy())
                inputs = self.processor(
                    text=list(class_texts),
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                probabilities = outputs.logits_per_image.softmax(dim=1).float().cpu()
                all_probs.extend(probabilities.tolist())
        return all_probs


def safe_label(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def cell_paths(
    output_dir: Path, group: str, checkpoint: str, concept: str
) -> tuple[Path, Path]:
    raw_dir = output_dir / "raw" / "cells" / group / checkpoint / concept
    image_dir = output_dir / "images" / group / checkpoint / concept
    return raw_dir, image_dir


def generation_index(
    output_dir: Path, image_dir: Path, seeds: Sequence[int]
) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": index,
            "seed": seed,
            "image_path": str(
                (image_dir / f"index_{index:03d}_seed_{seed}.png").relative_to(output_dir)
            ),
        }
        for index, seed in enumerate(seeds)
    ]


def generate_missing_images(
    pipe: Any,
    output_dir: Path,
    image_dir: Path,
    prompt: str,
    seeds: Sequence[int],
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    index = generation_index(output_dir, image_dir, seeds)
    missing = [row for row in index if not (output_dir / row["image_path"]).is_file()]
    batch_size = int(generation["batch_size"])
    image_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        generators = [
            torch.Generator(device=str(pipe.device)).manual_seed(int(row["seed"]))
            for row in batch
        ]
        with torch.inference_mode():
            images = pipe(
                prompt=[prompt] * len(batch),
                num_inference_steps=int(generation["num_inference_steps"]),
                guidance_scale=float(generation["guidance_scale"]),
                height=int(generation["height"]),
                width=int(generation["width"]),
                generator=generators,
            ).images
        if len(images) != len(batch):
            raise RuntimeError("Pipeline image count does not match the requested seed batch")
        for row, image in zip(batch, images):
            destination = output_dir / row["image_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination)
    absent = [row["image_path"] for row in index if not (output_dir / row["image_path"]).is_file()]
    if absent:
        raise RuntimeError(f"Generation incomplete; missing {len(absent)} images")
    return index


def delete_evaluated_images(
    output_dir: Path, index: Sequence[Mapping[str, Any]], image_dir: Path
) -> None:
    image_root = (output_dir / "images").resolve()
    for row in index:
        path = (output_dir / str(row["image_path"])).resolve()
        if image_root not in path.parents or path.suffix.lower() != ".png":
            raise RuntimeError(f"Refusing unsafe image deletion target: {path}")
        if path.is_file():
            path.unlink()
    try:
        image_dir.rmdir()
    except OSError:
        pass


def evaluate_cell(
    *,
    pipe: Any,
    classifier: ClipClassifier,
    protocol: Mapping[str, Any],
    output_dir: Path,
    group: str,
    checkpoint: str,
    concept: str,
    prompt: str,
    class_labels: Sequence[str],
    expected_label: str,
    seeds: Sequence[int],
    image_retention: str,
    extra_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    raw_dir, image_dir = cell_paths(output_dir, group, checkpoint, concept)
    complete_path = raw_dir / "complete.json"
    if complete_path.is_file():
        complete = read_json(complete_path)
        if complete.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise RuntimeError(f"Cell protocol mismatch: {raw_dir}")
        return read_json(raw_dir / "metrics.json")

    config = protocol["config"]
    generation = config["generation"]
    index = generate_missing_images(
        pipe, output_dir, image_dir, prompt, seeds, generation
    )
    write_json(
        raw_dir / "generation_manifest.json",
        {
            "status": "complete",
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "group": group,
            "checkpoint": checkpoint,
            "concept": concept,
            "prompt": prompt,
            "generation": generation,
            "images": index,
            **dict(extra_metadata),
        },
    )
    image_paths = [output_dir / row["image_path"] for row in index]
    template_key = (
        "cifar_class_text_template"
        if len(class_labels) == 10
        else "retain_class_text_template"
    )
    class_texts = [
        str(config["evaluation"][template_key]).format(concept=label)
        for label in class_labels
    ]
    probabilities = classifier.classify(
        image_paths,
        class_texts,
        int(config["evaluation"]["batch_size"]),
    )
    if len(probabilities) != len(index):
        raise RuntimeError("Evaluator output count does not match the generation index")
    rows: list[dict[str, Any]] = []
    correct = 0
    expected_index = list(class_labels).index(expected_label)
    for image_row, values in zip(index, probabilities):
        prediction_index = max(range(len(values)), key=values.__getitem__)
        predicted = class_labels[prediction_index]
        is_correct = predicted == expected_label
        correct += int(is_correct)
        row: dict[str, Any] = {
            "group": group,
            "checkpoint": checkpoint,
            "concept": concept,
            "prompt": prompt,
            "sample_index": image_row["sample_index"],
            "seed": image_row["seed"],
            "image_path": image_row["image_path"],
            "expected_label": expected_label,
            "predicted_label": predicted,
            "correct": is_correct,
            "expected_probability": values[expected_index],
            "image_retention": image_retention,
        }
        for label, value in zip(class_labels, values):
            row[f"prob_{safe_label(label)}"] = value
        rows.append(row)
    accuracy = correct / len(rows)
    metrics = {
        "status": "complete",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "group": group,
        "checkpoint": checkpoint,
        "concept": concept,
        "prompt": prompt,
        "expected_label": expected_label,
        "class_labels": list(class_labels),
        "class_texts": class_texts,
        "evaluator": config["clip_model_id"],
        "n_images": len(rows),
        "correct": correct,
        "accuracy": accuracy,
        "mean_expected_probability": sum(
            float(row["expected_probability"]) for row in rows
        )
        / len(rows),
        "image_retention": image_retention,
        **dict(extra_metadata),
    }
    write_csv(raw_dir / "predictions.csv", rows)
    write_json(raw_dir / "metrics.json", metrics)
    if len(rows) != len(seeds) or read_json(raw_dir / "metrics.json")["n_images"] != len(seeds):
        raise RuntimeError("Refusing cleanup because evaluator artifacts failed validation")
    if image_retention == "delete-after-eval":
        delete_evaluated_images(output_dir, index, image_dir)
        image_status = "deleted-after-successful-evaluation"
    else:
        image_status = "retained"
    write_json(
        complete_path,
        {
            "status": "complete",
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "completed_at": utc_now(),
            "n_images": len(rows),
            "image_status": image_status,
        },
    )
    return metrics


def effective_retention(args: argparse.Namespace, protocol: Mapping[str, Any]) -> str:
    del args
    return str(protocol["effective_image_retention"])


def qualify(args: argparse.Namespace) -> str:
    protocol = require_protocol(args)
    output_dir = Path(args.output_dir).resolve()
    existing = qualification_summary_path(output_dir)
    if existing.is_file():
        summary = read_json(existing)
        if summary.get("protocol_fingerprint") == protocol["protocol_fingerprint"]:
            selected = summary.get("selected_concept")
            if selected:
                event(output_dir, "qualification", "reuse selected concept", concept=selected)
                return str(selected)
    update_state(output_dir, "qualification", "running")
    config = protocol["config"]
    pipe = load_pipeline(protocol, edit_only=False)
    classifier = ClipClassifier(
        config["clip_model_id"], config["device"], bool(protocol["local_files_only"])
    )
    cifar = targets(config)
    n = int(config["qualification"]["images_per_candidate"])
    seed_start = int(config["generation"]["seed_start"])
    seeds = list(range(seed_start, seed_start + n))
    threshold = float(config["qualification"]["minimum_top1_accuracy"])
    retention = effective_retention(args, protocol)
    results: list[dict[str, Any]] = []
    selected: str | None = None
    for candidate in config["qualification"]["candidates"]:
        labels = cifar + [candidate]
        prompt = str(config["generation"]["prompt_template"]).format(
            concept=candidate
        )
        metrics = evaluate_cell(
            pipe=pipe,
            classifier=classifier,
            protocol=protocol,
            output_dir=output_dir,
            group="qualification",
            checkpoint="W00",
            concept=candidate,
            prompt=prompt,
            class_labels=labels,
            expected_label=candidate,
            seeds=seeds,
            image_retention=retention,
            extra_metadata={"phase": "qualification", "candidate": candidate},
        )
        accepted = float(metrics["accuracy"]) >= threshold
        result = {
            "candidate": candidate,
            "n_images": metrics["n_images"],
            "top1_accuracy": metrics["accuracy"],
            "mean_expected_probability": metrics["mean_expected_probability"],
            "minimum_top1_accuracy": threshold,
            "accepted": accepted,
        }
        results.append(result)
        event(
            output_dir,
            "qualification",
            "evaluated candidate",
            candidate=candidate,
            accuracy=f"{metrics['accuracy']:.4f}",
            accepted=accepted,
        )
        if accepted:
            selected = candidate
            break
    summary = {
        "status": "complete" if selected else "failed",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "rule": "first candidate with 11-class CLIP top-1 accuracy at or above threshold",
        "class_context": "ten CIFAR-10 labels plus the candidate",
        "selected_concept": selected,
        "results": results,
        "completed_at": utc_now(),
    }
    write_json(existing, summary)
    write_csv(output_dir / "qualification" / "results.csv", results)
    release_cuda(classifier, pipe)
    if not selected:
        update_state(output_dir, "qualification", "failed")
        raise RuntimeError(
            "No candidate passed qualification. Review qualification/results.csv "
            "and choose whether to lower the threshold or add a candidate."
        )
    update_state(output_dir, "qualification", "complete", selected_concept=selected)
    return selected


def checkpoint_path(output_dir: Path, condition: str, step: int, target: str) -> Path:
    return (
        output_dir
        / "checkpoints"
        / condition
        / f"W{step:02d}_{target}.safetensors"
    )


def checkpoint_manifest_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".manifest.json")


def validate_reusable_checkpoint(
    checkpoint: Path,
    protocol: Mapping[str, Any],
    selected_x: str,
    condition: str,
    step: int,
    parent_sha256: str | None,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    manifest_path = checkpoint_manifest_path(checkpoint)
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = read_json(manifest_path)
    checks = {
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "selected_x": selected_x,
        "condition": condition,
        "step": step,
        "parent_checkpoint_sha256": parent_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in checks.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint manifest mismatch at {checkpoint}: {mismatches}")
    if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    state = load_file(str(checkpoint))
    if len(state) != 16:
        raise RuntimeError(f"Unexpected checkpoint tensor count: {checkpoint}")
    return state


def build_checkpoints(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import load_file

    protocol = require_protocol(args)
    output_dir = Path(args.output_dir).resolve()
    selected_x = require_selected_x(output_dir, protocol)
    update_state(output_dir, "checkpoints", "running", selected_concept=selected_x)
    config = protocol["config"]
    if str(OCE_ROOT) not in sys.path:
        sys.path.insert(0, str(OCE_ROOT))
    import oce as oce_impl

    pipe = load_pipeline(protocol, edit_only=True)
    base_state = selected_projection_state(pipe.unet)
    oce_impl.device = config["device"]
    oce_impl.torch_dtype = torch.float32
    previous_cwd = Path.cwd()
    try:
        os.chdir(OCE_ROOT)
        for condition in CONDITIONS:
            apply_projection_state(pipe.unet, base_state)
            parent_sha256: str | None = None
            for step, pair in enumerate(pair_rows(config), start=1):
                target = pair["target"]
                anchor = pair["anchor"]
                checkpoint = checkpoint_path(output_dir, condition, step, target)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                try:
                    state = validate_reusable_checkpoint(
                        checkpoint,
                        protocol,
                        selected_x,
                        condition,
                        step,
                        parent_sha256,
                    )
                    apply_projection_state(pipe.unet, state)
                    parent_sha256 = sha256_file(checkpoint)
                    event(
                        output_dir,
                        "checkpoints",
                        "reuse checkpoint",
                        condition=condition,
                        step=step,
                        target=target,
                    )
                    continue
                except FileNotFoundError:
                    pass
                include_x = condition == "retain_always" or step == 1
                preserve = [anchor] + ([selected_x] if include_x else [])
                if bool(config["oce"]["expand_prompts"]):
                    edits, guides = expand_object_pair(target, anchor)
                else:
                    edits, guides = [target], [anchor]
                event(
                    output_dir,
                    "checkpoints",
                    "run single-concept edit",
                    condition=condition,
                    step=step,
                    target=target,
                    x_retained=include_x,
                )
                oce_impl.Orthogonal_Erase(
                    pipe,
                    edits,
                    guides,
                    preserve,
                    float(config["oce"]["erase_scale"]),
                    float(config["oce"]["preserve_global_scale"]),
                    float(config["oce"]["preserve_concept_scale"]),
                    float(config["oce"]["lamb"]),
                    str(checkpoint.parent),
                    checkpoint.stem,
                )
                state = load_file(str(checkpoint))
                apply_projection_state(pipe.unet, state)
                checkpoint_sha256 = sha256_file(checkpoint)
                write_json(
                    checkpoint_manifest_path(checkpoint),
                    {
                        "status": "complete",
                        "protocol_fingerprint": protocol["protocol_fingerprint"],
                        "condition": condition,
                        "step": step,
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": checkpoint_sha256,
                        "parent_checkpoint_sha256": parent_sha256,
                        "target": target,
                        "anchor": anchor,
                        "selected_x": selected_x,
                        "x_in_explicit_retain": include_x,
                        "explicit_retain_concepts": preserve,
                        "tensor_count": len(state),
                        "created_at": utc_now(),
                    },
                )
                parent_sha256 = checkpoint_sha256
    finally:
        os.chdir(previous_cwd)
        release_cuda(pipe)
    update_state(output_dir, "checkpoints", "complete", checkpoint_count=20)


def formal_seeds(config: Mapping[str, Any]) -> list[int]:
    start = int(config["generation"]["seed_start"])
    count = int(config["generation"]["images_per_concept"])
    return list(range(start, start + count))


def formal_cell_metrics_path(
    output_dir: Path, group: str, checkpoint: str, concept: str
) -> Path:
    return cell_paths(output_dir, group, checkpoint, concept)[0] / "metrics.json"


def evaluate_formal(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file

    protocol = require_protocol(args)
    output_dir = Path(args.output_dir).resolve()
    selected_x = require_selected_x(output_dir, protocol)
    config = protocol["config"]
    update_state(output_dir, "evaluation", "running", selected_concept=selected_x)
    pipe = load_pipeline(protocol, edit_only=False)
    base_state = selected_projection_state(pipe.unet)
    classifier = ClipClassifier(
        config["clip_model_id"], config["device"], bool(protocol["local_files_only"])
    )
    cifar = targets(config)
    seeds = formal_seeds(config)
    retention = effective_retention(args, protocol)
    prompt_template = str(config["generation"]["prompt_template"])

    apply_projection_state(pipe.unet, base_state)
    for concept in cifar + [selected_x]:
        class_labels = cifar if concept in cifar else cifar + [selected_x]
        metrics = evaluate_cell(
            pipe=pipe,
            classifier=classifier,
            protocol=protocol,
            output_dir=output_dir,
            group="original",
            checkpoint="W00",
            concept=concept,
            prompt=prompt_template.format(concept=concept),
            class_labels=class_labels,
            expected_label=concept,
            seeds=seeds,
            image_retention=retention,
            extra_metadata={"condition": "original", "step": 0, "role": "baseline"},
        )
        event(
            output_dir,
            "evaluation",
            "completed cell",
            condition="original",
            checkpoint="W00",
            concept=concept,
            accuracy=f"{metrics['accuracy']:.4f}",
        )

    for condition in CONDITIONS:
        for step, pair in enumerate(pair_rows(config), start=1):
            checkpoint = checkpoint_path(
                output_dir, condition, step, pair["target"]
            )
            parent_sha = (
                None
                if step == 1
                else sha256_file(
                    checkpoint_path(
                        output_dir,
                        condition,
                        step - 1,
                        pair_rows(config)[step - 2]["target"],
                    )
                )
            )
            state = validate_reusable_checkpoint(
                checkpoint,
                protocol,
                selected_x,
                condition,
                step,
                parent_sha,
            )
            apply_projection_state(pipe.unet, state)
            checkpoint_name = f"W{step:02d}"
            concepts = cifar[:step] + [selected_x]
            for concept in concepts:
                labels = cifar if concept in cifar else cifar + [selected_x]
                role = "erased_target" if concept in cifar else "explicit_retain_probe"
                metrics = evaluate_cell(
                    pipe=pipe,
                    classifier=classifier,
                    protocol=protocol,
                    output_dir=output_dir,
                    group=condition,
                    checkpoint=checkpoint_name,
                    concept=concept,
                    prompt=prompt_template.format(concept=concept),
                    class_labels=labels,
                    expected_label=concept,
                    seeds=seeds,
                    image_retention=retention,
                    extra_metadata={
                        "condition": condition,
                        "step": step,
                        "role": role,
                        "model_checkpoint": str(checkpoint.resolve()),
                        "model_checkpoint_sha256": sha256_file(checkpoint),
                    },
                )
                event(
                    output_dir,
                    "evaluation",
                    "completed cell",
                    condition=condition,
                    checkpoint=checkpoint_name,
                    concept=concept,
                    accuracy=f"{metrics['accuracy']:.4f}",
                )
    release_cuda(classifier, pipe)
    metrics_files = list((output_dir / "raw" / "cells").glob("**/metrics.json"))
    formal_files = [path for path in metrics_files if "qualification" not in path.parts]
    if len(formal_files) != 141:
        raise RuntimeError(f"Expected 141 completed formal cells, found {len(formal_files)}")
    update_state(
        output_dir,
        "evaluation",
        "complete",
        formal_cells=141,
        formal_images=14100,
        image_retention=retention,
    )


def collect_formal_metrics(
    output_dir: Path, protocol: Mapping[str, Any], selected_x: str
) -> list[dict[str, Any]]:
    config = protocol["config"]
    cifar = targets(config)
    paths: list[Path] = []
    for concept in cifar + [selected_x]:
        paths.append(formal_cell_metrics_path(output_dir, "original", "W00", concept))
    for condition in CONDITIONS:
        for step in range(1, 11):
            for concept in cifar[:step] + [selected_x]:
                paths.append(
                    formal_cell_metrics_path(
                        output_dir, condition, f"W{step:02d}", concept
                    )
                )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing formal metrics: {missing[:5]}")
    if len(paths) != 141:
        raise AssertionError(f"Expected 141 metric paths, got {len(paths)}")
    rows = [read_json(path) for path in paths]
    if any(row.get("protocol_fingerprint") != protocol["protocol_fingerprint"] for row in rows):
        raise RuntimeError("A formal cell has a mismatched protocol fingerprint")
    if sum(int(row["n_images"]) for row in rows) != 14100:
        raise RuntimeError("Formal evaluator outputs do not total 14,100 images")
    return rows


def metric_lookup(
    rows: Sequence[Mapping[str, Any]], group: str, checkpoint: str, concept: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["group"] == group
        and row["checkpoint"] == checkpoint
        and row["concept"] == concept
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one metric for {group}/{checkpoint}/{concept}, got {len(matches)}"
        )
    return matches[0]


def copy_raw_predictions(output_dir: Path) -> None:
    prediction_files = sorted((output_dir / "raw" / "cells").glob("**/predictions.csv"))
    rows: list[dict[str, Any]] = []
    for path in prediction_files:
        if "qualification" in path.parts:
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if len(rows) != 14100:
        raise RuntimeError(f"Expected 14,100 per-image rows, found {len(rows)}")
    write_csv(output_dir / "raw" / "formal_per_image_predictions.csv", rows)


def make_plots(
    output_dir: Path,
    config: Mapping[str, Any],
    selected_x: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cifar = targets(config)
    steps = list(range(11))
    baseline = float(metric_lookup(rows, "original", "W00", selected_x)["accuracy"])
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for condition, label in (
        ("retain_once", "Retain Once"),
        ("retain_always", "Retain Always"),
    ):
        values = [baseline] + [
            float(metric_lookup(rows, condition, f"W{step:02d}", selected_x)["accuracy"])
            for step in range(1, 11)
        ]
        axis.plot(steps, values, marker="o", linewidth=2, label=label)
    axis.set_xlabel("Sequential edit step")
    axis.set_ylabel(f"{selected_x} 11-class CLIP accuracy")
    axis.set_xticks(steps)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / "retain_persistence_curve.png", dpi=180)
    figure.savefig(plot_dir / "retain_persistence_curve.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    for axis, condition, title in zip(
        axes, CONDITIONS, ("Retain Once", "Retain Always")
    ):
        matrix = np.full((10, 10), np.nan, dtype=float)
        for step in range(1, 11):
            for concept_index, concept in enumerate(cifar[:step]):
                matrix[step - 1, concept_index] = float(
                    metric_lookup(rows, condition, f"W{step:02d}", concept)[
                        "accuracy"
                    ]
                )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("Erased target")
        axis.set_xticks(range(10), cifar, rotation=55, ha="right")
        axis.set_yticks(range(10), [f"W{step:02d}" for step in range(1, 11)])
        for row_index in range(10):
            for column_index in range(row_index + 1):
                value = matrix[row_index, column_index]
                color = "white" if value < 0.55 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
    axes[0].set_ylabel("Checkpoint")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="10-class CLIP accuracy")
    figure.subplots_adjust(left=0.08, right=0.92, bottom=0.24, wspace=0.08)
    figure.savefig(plot_dir / "previous_erasure_persistence_heatmaps.png", dpi=180)
    figure.savefig(plot_dir / "previous_erasure_persistence_heatmaps.pdf")
    plt.close(figure)


def aggregate(args: argparse.Namespace) -> None:
    protocol = require_protocol(args)
    output_dir = Path(args.output_dir).resolve()
    selected_x = require_selected_x(output_dir, protocol)
    config = protocol["config"]
    update_state(output_dir, "aggregation", "running")
    rows = collect_formal_metrics(output_dir, protocol, selected_x)
    cifar = targets(config)
    table_dir = output_dir / "tables"
    aggregate_rows = [
        {
            "condition": row["group"],
            "checkpoint": row["checkpoint"],
            "step": row.get("step", 0),
            "concept": row["concept"],
            "role": row.get("role", "baseline"),
            "n_images": row["n_images"],
            "accuracy": row["accuracy"],
            "mean_expected_probability": row["mean_expected_probability"],
            "class_count": len(row["class_labels"]),
            "image_retention": row["image_retention"],
        }
        for row in rows
    ]
    write_csv(table_dir / "aggregated_cells.csv", aggregate_rows)

    for condition in CONDITIONS:
        persistence_rows: list[dict[str, Any]] = []
        for step in range(1, 11):
            row: dict[str, Any] = {"checkpoint": f"W{step:02d}", "step": step}
            for index, concept in enumerate(cifar, start=1):
                row[concept] = (
                    metric_lookup(rows, condition, f"W{step:02d}", concept)["accuracy"]
                    if index <= step
                    else ""
                )
            persistence_rows.append(row)
        write_csv(
            table_dir / f"previous_erasure_persistence_{condition}.csv",
            persistence_rows,
        )

    baseline_x = float(metric_lookup(rows, "original", "W00", selected_x)["accuracy"])
    retain_rows: list[dict[str, Any]] = [
        {
            "checkpoint": "W00",
            "step": 0,
            "retain_once": baseline_x,
            "retain_always": baseline_x,
        }
    ]
    for step in range(1, 11):
        retain_rows.append(
            {
                "checkpoint": f"W{step:02d}",
                "step": step,
                "retain_once": metric_lookup(
                    rows, "retain_once", f"W{step:02d}", selected_x
                )["accuracy"],
                "retain_always": metric_lookup(
                    rows, "retain_always", f"W{step:02d}", selected_x
                )["accuracy"],
            }
        )
    write_csv(table_dir / "retain_persistence.csv", retain_rows)

    threshold = float(config["evaluation"]["material_change_threshold"])
    resurgence_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for erased_step, concept in enumerate(cifar, start=1):
            initial = float(
                metric_lookup(rows, condition, f"W{erased_step:02d}", concept)[
                    "accuracy"
                ]
            )
            later = [
                (
                    step,
                    float(
                        metric_lookup(rows, condition, f"W{step:02d}", concept)[
                            "accuracy"
                        ]
                    ),
                )
                for step in range(erased_step, 11)
            ]
            max_step, max_accuracy = max(later, key=lambda item: item[1])
            delta = max_accuracy - initial
            resurgence_rows.append(
                {
                    "condition": condition,
                    "target": concept,
                    "erased_at_step": erased_step,
                    "accuracy_at_erasure": initial,
                    "maximum_later_accuracy": max_accuracy,
                    "maximum_at_checkpoint": f"W{max_step:02d}",
                    "maximum_increase": delta,
                    "material_change_threshold": threshold,
                    "significant_resurgence": delta >= threshold,
                }
            )
    write_csv(table_dir / "old_target_resurgence.csv", resurgence_rows)

    make_plots(output_dir, config, selected_x, rows)
    copy_raw_predictions(output_dir)

    once = [float(row["retain_once"]) for row in retain_rows[1:]]
    always = [float(row["retain_always"]) for row in retain_rows[1:]]
    once_drop = once[0] - once[-1]
    always_drop = always[0] - always[-1]
    negative_once_steps = sum(
        next_value < value for value, next_value in zip(once, once[1:])
    )
    once_materially_degraded = once_drop >= threshold
    once_gradual = once_materially_degraded and negative_once_steps >= 5
    always_more_stable = (
        always[-1] - once[-1] >= threshold
        or once_drop - always_drop >= threshold
    )
    significant = [row for row in resurgence_rows if row["significant_resurgence"]]
    qualification = read_json(qualification_summary_path(output_dir))
    result_summary = {
        "status": "complete",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "selected_x": selected_x,
        "qualification": qualification,
        "formal_counts": {"cells": 141, "images": 14100},
        "qualification_images": sum(row["n_images"] for row in qualification["results"]),
        "image_retention": rows[0]["image_retention"],
        "material_change_threshold": threshold,
        "retain_once": {
            "W01_accuracy": once[0],
            "W10_accuracy": once[-1],
            "W01_to_W10_drop": once_drop,
            "negative_step_count": negative_once_steps,
            "material_degradation": once_materially_degraded,
            "gradual_degradation": once_gradual,
        },
        "retain_always": {
            "W01_accuracy": always[0],
            "W10_accuracy": always[-1],
            "W01_to_W10_drop": always_drop,
        },
        "retain_always_more_stable": always_more_stable,
        "previous_erasure_resurgence": {
            "observed": bool(significant),
            "significant_cases": significant,
        },
        "completed_at": utc_now(),
    }
    write_json(output_dir / "summary.json", result_summary)

    qualification_lines = [
        f"- `{row['candidate']}`: top-1 accuracy {float(row['top1_accuracy']):.3f}; "
        f"accepted = `{row['accepted']}`"
        for row in qualification["results"]
    ]
    if once_gradual:
        retain_once_answer = (
            f"X declined gradually under Retain Once: W01={once[0]:.3f}, "
            f"W10={once[-1]:.3f} (drop {once_drop:.3f})."
        )
    elif once_materially_degraded:
        retain_once_answer = (
            f"X was materially lower by W10 under Retain Once "
            f"(W01={once[0]:.3f}, W10={once[-1]:.3f}), but the path was not "
            "consistently gradual."
        )
    else:
        retain_once_answer = (
            f"X did not show material Retain Once degradation under the configured "
            f"{threshold:.2f} threshold (W01={once[0]:.3f}, W10={once[-1]:.3f})."
        )
    if always_more_stable:
        always_answer = (
            f"Retain Always was more stable by the configured criterion "
            f"(W10: {always[-1]:.3f} vs {once[-1]:.3f})."
        )
    else:
        always_answer = (
            "Retain Always did not show a material stability advantage over Retain Once "
            f"under the configured {threshold:.2f} threshold."
        )
    if significant:
        cases = ", ".join(
            f"{row['condition']}:{row['target']} (+{float(row['maximum_increase']):.3f})"
            for row in significant
        )
        resurgence_answer = f"Material previous-erasure resurgence was observed in: {cases}."
    else:
        resurgence_answer = (
            "在本 10-step sequential setting 中未觀察到明顯 previous-erasure resurgence。"
        )
    summary_lines = [
        "# Sequential OCE object-persistence summary",
        "",
        f"- Selected X: **{selected_x}**",
        f"- Formal scope: **141 concept-checkpoint cells / 14,100 images**",
        f"- Qualification overhead: **{result_summary['qualification_images']} images**",
        f"- Material-change rule: absolute accuracy change of at least **{threshold:.2f}**",
        f"- Image retention: `{result_summary['image_retention']}`",
        "",
        "## Qualification",
        "",
        *qualification_lines,
        "",
        "The evaluator is CLIP ViT-B/32. Qualification and X preservation use an "
        "11-class context (the ten CIFAR-10 labels plus X); erased CIFAR targets keep "
        "the unchanged 10-class context.",
        "",
        "## Answers",
        "",
        f"- {retain_once_answer}",
        f"- {always_answer}",
        f"- {resurgence_answer}",
        "",
        "## Artifacts",
        "",
        "- `raw/formal_per_image_predictions.csv`: all per-image evaluator outputs",
        "- `tables/aggregated_cells.csv`: one row per formal cell",
        "- `tables/previous_erasure_persistence_*.csv`: checkpoint × erased-target tables",
        "- `tables/retain_persistence.csv`: X preservation comparison",
        "- `tables/old_target_resurgence.csv`: per-target maximum later increase",
        "- `figures/retain_persistence_curve.png`: X curve",
        "- `figures/previous_erasure_persistence_heatmaps.png`: old-target heatmaps",
        "",
        "Generation manifests retain every image index, seed, and original relative path. "
        "When `delete-after-eval` is active, PNG files are removed only after evaluator "
        "outputs and aggregate metrics for that cell have been validated and saved.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    update_state(output_dir, "aggregation", "complete")
    event(output_dir, "aggregation", "wrote final summary", selected_x=selected_x)


def print_plan(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    audit = validate_config(config_path, config, require_artifacts=False)
    plan = {
        "config": str(config_path),
        "output_dir": str(Path(args.output_dir).resolve()),
        "target_order": targets(config),
        "target_anchor_mapping": pair_rows(config),
        "candidate_order": config["qualification"]["candidates"],
        "conditions": list(CONDITIONS),
        "counts": audit,
        "image_retention": args.image_retention
        or config["storage"]["image_retention"],
        "phases": [
            "preflight",
            "qualification",
            "sequential checkpoint construction",
            "141-cell generation/evaluation",
            "aggregation and plots",
        ],
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def run_all(args: argparse.Namespace) -> None:
    preflight(args)
    qualify(args)
    build_checkpoints(args)
    evaluate_formal(args)
    aggregate(args)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow model downloads instead of requiring the server cache.",
    )
    parser.add_argument(
        "--image-retention",
        choices=("keep", "delete-after-eval"),
        default=None,
        help="Override the config's generated-image retention policy.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    functions = {
        "plan": print_plan,
        "preflight": preflight,
        "qualify": qualify,
        "build": build_checkpoints,
        "evaluate": evaluate_formal,
        "aggregate": aggregate,
        "run": run_all,
    }
    for name, function in functions.items():
        command = subparsers.add_parser(name)
        add_common_arguments(command)
        command.set_defaults(function=function)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
