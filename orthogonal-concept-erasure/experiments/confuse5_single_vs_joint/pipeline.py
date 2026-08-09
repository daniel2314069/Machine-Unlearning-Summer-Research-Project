#!/usr/bin/env python3
"""Gated GPU-server pipeline for the primary Confuse5 OCE rerun.

The ``plan`` and ``status`` stages use only the standard library. Stages that
load Stable Diffusion or ResNet import their dependencies lazily and are not
permitted on the local Mac by the project AGENTS.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import protocol


HERE = Path(__file__).resolve().parent
COMPUTE_K0 = HERE / "compute_k0.py"
CHECKPOINT_BUILDER = HERE / "run.py"
REQUIRED_COLUMNS = ("case_number", "prompt", "class", "evaluation_seed")


class PipelineError(RuntimeError):
    pass


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _event(root: Path, event: str, **details: Any) -> None:
    path = root / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": protocol.utc_now(), "event": event, **details}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _state(root: Path, stage: str, status: str, **details: Any) -> None:
    path = root / "run_state.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        previous = protocol.read_json(path)
    protocol.write_json_atomic(path, {
        **previous,
        "stage": stage,
        "status": status,
        "updated_at": protocol.utc_now(),
        **details,
    })


def load_dataset(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    path = Path(config["_resolved"]["evaluation_dataset"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PipelineError(f"Dataset has no header: {path}")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise PipelineError(f"Dataset missing columns: {sorted(missing)}")
        raw_rows = list(reader)
    rows: list[dict[str, Any]] = []
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cases: set[int] = set()
    for line, raw in enumerate(raw_rows, start=2):
        try:
            case = int(raw["case_number"])
            seed = int(raw["evaluation_seed"])
        except ValueError as exc:
            raise PipelineError(f"Invalid case/seed at dataset line {line}") from exc
        concept = " ".join(raw["class"].split())
        prompt_text = raw["prompt"].strip()
        if not concept or not prompt_text or case in cases:
            raise PipelineError(f"Empty value or duplicate case at dataset line {line}")
        cases.add(case)
        row = {
            "case_number": case,
            "source_line": line,
            "prompt": prompt_text,
            "class": concept,
            "evaluation_seed": seed,
        }
        rows.append(row)
        by_class[protocol.normalize(concept)].append(row)
    rows.sort(key=lambda row: row["case_number"])
    for values in by_class.values():
        values.sort(key=lambda row: row["case_number"])
    evaluation = config["evaluation"]
    if len(rows) != int(evaluation["expected_total_rows"]):
        raise PipelineError(f"Expected {evaluation['expected_total_rows']} dataset rows, found {len(rows)}")
    expected = int(evaluation["expected_rows_per_class"])
    for group in config["groups"]:
        for concept in group["concepts"]:
            actual = len(by_class.get(protocol.normalize(concept), []))
            if actual != expected:
                raise PipelineError(f"Expected {expected} rows for {concept!r}, found {actual}")
    return rows, dict(by_class)


def _checkpoint_lookup(config: Mapping[str, Any], anchors: Mapping[str, str]) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    lookup: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for spec in protocol.checkpoint_specs(config, anchors):
        target = spec["targets"][0] if spec["mode"] == "single" else None
        lookup[(spec["group_id"], spec["mode"], target)] = spec
    return lookup


def _ordered_rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return protocol.fingerprint({"rows": [{key: row[key] for key in REQUIRED_COLUMNS} for row in rows]})


def build_plan(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config, anchors = protocol.load_protocol(config_path)
    rows, by_class = load_dataset(config)
    checkpoint_specs = protocol.checkpoint_specs(config, anchors)
    output_root = Path(config["_resolved"]["output_root"])
    plan = {
        "schema_version": 2,
        "created_at": protocol.utc_now(),
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path.resolve()),
        "config_sha256": protocol.sha256(config_path),
        "anchors_path": config["_resolved"]["anchors_path"],
        "anchors_sha256": protocol.sha256(Path(config["_resolved"]["anchors_path"])),
        "dataset_path": config["_resolved"]["evaluation_dataset"],
        "dataset_sha256": protocol.sha256(Path(config["_resolved"]["evaluation_dataset"])),
        "dataset_row_count": len(rows),
        "ordered_dataset_rows_sha256": _ordered_rows_hash(rows),
        "output_root": str(output_root),
        "legacy_reference_path": config["_resolved"]["legacy_original_reference"],
        "checkpoint_count": len(checkpoint_specs),
        "checkpoint_specs": checkpoint_specs,
        "anchor_mapping": anchors,
        "image_counts": {
            "anchor_sanity_original": 80,
            "smoke_original": 128,
            "smoke_single": 128,
            "smoke_joint": 128,
            "formal_original_regenerated": 0,
            "formal_single": 25000,
            "formal_joint": 12500,
            "formal_total_new_edited": 37500,
        },
        "hard_stops": [
            "anchor_sanity_failure",
            "original_reproduction_hash_mismatch",
            "any_single_smoke_drop_below_4_of_32",
        ],
        "source_hashes": protocol.source_hashes([Path(__file__), COMPUTE_K0, CHECKPOINT_BUILDER]),
        "resolved_config": config,
    }
    plan["plan_fingerprint"] = protocol.fingerprint({
        key: plan[key]
        for key in (
            "experiment_id", "config_sha256", "anchors_sha256", "dataset_sha256",
            "ordered_dataset_rows_sha256", "checkpoint_specs", "image_counts", "source_hashes",
        )
    })
    return plan, config, anchors


def _group_for_target(config: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    return protocol.group_for_target(config, target)


def _validate_checkpoint(spec: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = Path(spec["checkpoint_path"])
    metadata_path = Path(spec["metadata_path"])
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise PipelineError(f"Missing checkpoint or metadata: {checkpoint.parent}")
    metadata = protocol.read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise PipelineError(f"Checkpoint incomplete: {metadata_path}")
    if metadata.get("plan_fingerprint") != _checkpoint_plan_fingerprint(config):
        raise PipelineError(f"Checkpoint plan fingerprint mismatch: {metadata_path}")
    if metadata.get("checkpoint_sha256") != protocol.sha256(checkpoint):
        raise PipelineError(f"Checkpoint hash mismatch: {checkpoint}")
    if metadata.get("group_id") != spec["group_id"] or metadata.get("mode") != spec["mode"]:
        raise PipelineError(f"Checkpoint role mismatch: {checkpoint}")
    if [protocol.normalize(x) for x in metadata.get("targets", [])] != [protocol.normalize(x) for x in spec["targets"]]:
        raise PipelineError(f"Checkpoint targets mismatch: {checkpoint}")
    if [protocol.normalize(x) for x in metadata.get("anchors", [])] != [protocol.normalize(x) for x in spec["anchors"]]:
        raise PipelineError(f"Checkpoint anchors mismatch: {checkpoint}")
    return metadata


def _checkpoint_plan_fingerprint(config: Mapping[str, Any]) -> str:
    # Importing run.py is lightweight; model libraries are imported only inside execution functions.
    import run as checkpoint_builder

    plan, _, _ = checkpoint_builder.build_plan(Path(config["_resolved"]["config_path"]))
    return str(plan["plan_fingerprint"])


def _component_identity(component: Any) -> dict[str, Any]:
    component_config = getattr(component, "config", None)
    init_kwargs = getattr(component, "init_kwargs", {})
    return {
        "class": f"{type(component).__module__}.{type(component).__name__}",
        "name_or_path": getattr(component, "name_or_path", None)
        or getattr(component_config, "_name_or_path", None),
        "commit_hash": getattr(component_config, "_commit_hash", None)
        or (init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None),
    }


class GenerationRuntime:
    def __init__(self, config: Mapping[str, Any]):
        import torch
        from diffusers import DiffusionPipeline
        from safetensors.torch import load_file

        self.torch = torch
        self.load_file = load_file
        self.device = str(config["model"]["device"])
        generation = config["evaluation"]["generation"]
        if generation["scheduler"] != "PNDMScheduler":
            raise PipelineError("Only explicitly resolved PNDMScheduler is allowed")
        self.settings = generation
        self.pipe = DiffusionPipeline.from_pretrained(
            config["model"]["base_model"],
            torch_dtype=getattr(torch, config["model"]["generation_dtype"]),
            safety_checker=None,
        )
        if type(self.pipe.scheduler).__name__ != generation["scheduler"]:
            raise PipelineError("Resolved scheduler class differs from protocol")
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        scheduler_config = dict(self.pipe.scheduler.config)
        self.identity = {
            "requested_base_model": config["model"]["base_model"],
            "requested_dtype": config["model"]["generation_dtype"],
            "pipeline": _component_identity(self.pipe),
            "unet": _component_identity(self.pipe.unet),
            "text_encoder": _component_identity(self.pipe.text_encoder),
            "tokenizer": _component_identity(self.pipe.tokenizer),
            "scheduler": _component_identity(self.pipe.scheduler),
            "scheduler_config_sha256": protocol.fingerprint({"config": scheduler_config}),
            "settings": dict(generation),
            "runtime": protocol.runtime_provenance(),
        }
        self.modules = {
            f"{name}.weight": module
            for name, module in self.pipe.unet.named_modules()
            if "attn2" in name and name.endswith("to_v")
        }
        if len(self.modules) != int(config["evaluation"]["expected_checkpoint_keys"]):
            raise PipelineError(f"Expected 16 editable checkpoint keys, found {len(self.modules)}")
        self.original = {key: module.weight.detach().clone() for key, module in self.modules.items()}
        self.active_hash: str | None = None

    def activate_original(self) -> None:
        for key, value in self.original.items():
            self.modules[key].weight.data.copy_(value)
        self.active_hash = None

    def activate_checkpoint(self, spec: Mapping[str, Any]) -> str:
        checkpoint = Path(spec["checkpoint_path"])
        expected_hash = protocol.sha256(checkpoint)
        if self.active_hash == expected_hash:
            return expected_hash
        self.activate_original()
        state = self.load_file(str(checkpoint), device="cpu")
        if set(state) != set(self.modules):
            raise PipelineError(f"Checkpoint key mismatch: {checkpoint}")
        for key, value in state.items():
            self.modules[key].weight.data.copy_(value.to(device=self.device, dtype=self.modules[key].weight.dtype))
        self.active_hash = expected_hash
        return expected_hash

    def generate(self, prompt_text: str, seed: int) -> Any:
        generator = self.torch.Generator(device=self.settings["generator_device"]).manual_seed(seed)
        result = self.pipe(
            prompt_text,
            generator=generator,
            num_inference_steps=int(self.settings["num_inference_steps"]),
            guidance_scale=float(self.settings["guidance_scale"]),
            height=int(self.settings["height"]),
            width=int(self.settings["width"]),
            num_images_per_prompt=int(self.settings["images_per_prompt"]),
        )
        return result.images[0]


class EvaluationRuntime:
    def __init__(self, config: Mapping[str, Any]):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        classifier = config["evaluation"]["classifier"]
        weights = getattr(ResNet50_Weights, classifier["weights_enum"])
        if Path(weights.url).name != classifier["expected_weight_filename"]:
            raise PipelineError("Pinned ResNet weight filename differs from protocol")
        self.torch = torch
        self.device = str(config["model"]["device"])
        self.model = resnet50(weights=weights).to(self.device).eval()
        self.preprocess = weights.transforms()
        self.categories = list(weights.meta["categories"])
        self.normalized = [protocol.normalize(value) for value in self.categories]
        self.top_k = int(classifier["top_k"])
        self.batch_size = int(classifier["batch_size"])
        self.identity = {
            "implementation": classifier["implementation"],
            "weights_enum": classifier["weights_enum"],
            "weight_filename": Path(weights.url).name,
            "weight_url": weights.url,
            "categories_sha256": protocol.fingerprint({"categories": self.categories}),
        }

    def class_index(self, concept: str) -> int:
        needle = protocol.normalize(concept)
        matches = [index for index, value in enumerate(self.normalized) if value == needle]
        if len(matches) != 1:
            raise PipelineError(f"Expected one exact ImageNet category for {concept!r}, found {matches}")
        return matches[0]

    def evaluate(self, image_paths: Sequence[Path], expected_concept: str) -> list[dict[str, Any]]:
        from PIL import Image

        expected_index = self.class_index(expected_concept)
        output: list[dict[str, Any]] = []
        with self.torch.inference_mode():
            for start in range(0, len(image_paths), self.batch_size):
                chunk = image_paths[start:start + self.batch_size]
                tensors = []
                for path in chunk:
                    with Image.open(path) as image:
                        tensors.append(self.preprocess(image.convert("RGB")))
                logits = self.model(self.torch.stack(tensors).to(self.device))
                probabilities = logits.softmax(dim=1)
                top_probs, top_indices = probabilities.topk(self.top_k, dim=1)
                for offset, path in enumerate(chunk):
                    predicted = int(top_indices[offset, 0].item())
                    top: list[dict[str, Any]] = []
                    for rank in range(self.top_k):
                        index = int(top_indices[offset, rank].item())
                        top.append({
                            "rank": rank + 1,
                            "index": index,
                            "label": self.categories[index],
                            "probability": float(top_probs[offset, rank].item()),
                            "raw_logit": float(logits[offset, index].item()),
                        })
                    output.append({
                        "image_path": str(path.resolve()),
                        "image_sha256": protocol.sha256(path),
                        "expected_index": expected_index,
                        "expected_category": self.categories[expected_index],
                        "predicted_index": predicted,
                        "predicted_category": self.categories[predicted],
                        "correct": predicted == expected_index,
                        "target_probability": float(probabilities[offset, expected_index].item()),
                        "raw_target_logit": float(logits[offset, expected_index].item()),
                        "top5": top,
                    })
        return output


def _save_png(image: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)
    return protocol.sha256(path)


def _require_gate(root: Path, name: str) -> dict[str, Any]:
    path = root / name / "gate.json"
    if not path.is_file():
        raise PipelineError(f"Required gate is missing: {path}")
    gate = protocol.read_json(path)
    if gate.get("status") != "passed":
        raise PipelineError(f"Required gate did not pass: {path}")
    return gate


def anchor_sanity(config: Mapping[str, Any], anchors: Mapping[str, str], *, skip_existing: bool) -> dict[str, Any]:
    root = Path(config["_resolved"]["output_root"])
    gate_path = root / "anchor_sanity" / "gate.json"
    if skip_existing and gate_path.is_file():
        existing = protocol.read_json(gate_path)
        if existing.get("status") == "passed" and existing.get("protocol_fingerprint") == _stage_fingerprint(config, "anchor_sanity"):
            return existing
    if gate_path.exists():
        raise PipelineError(f"Anchor sanity output collision: {gate_path.parent}")
    _, by_class = load_dataset(config)
    generation = GenerationRuntime(config)
    evaluator = EvaluationRuntime(config)
    generation.activate_original()
    settings = config["anchor_sanity"]
    all_results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for target, anchor in anchors.items():
        rows = by_class[protocol.normalize(target)][:int(settings["seeds_per_anchor"])]
        paths: list[Path] = []
        for row in rows:
            path = root / "anchor_sanity" / "images" / protocol.slug(target) / f"case-{row['case_number']:06d}_seed-{row['evaluation_seed']}.png"
            image = generation.generate(settings["prompt_template"].format(anchor=anchor), row["evaluation_seed"])
            _save_png(image, path)
            paths.append(path)
        metrics = evaluator.evaluate(paths, target)
        collision_count = sum(bool(item["correct"]) for item in metrics)
        for row, item in zip(rows, metrics):
            all_results.append({
                **row,
                "source_target_prompt": row["prompt"],
                "prompt": settings["prompt_template"].format(anchor=anchor),
                "target": target,
                "anchor": anchor,
                **item,
            })
        summaries.append({
            "target": target,
            "anchor": anchor,
            "collision_count": collision_count,
            "total": len(rows),
            "collision_rate": collision_count / len(rows),
            "passed": collision_count < int(settings["target_collision_failure_count"]),
        })
        if not summaries[-1]["passed"]:
            gate = {
                "schema_version": 1,
                "status": "failed",
                "completed_at": protocol.utc_now(),
                "protocol_fingerprint": _stage_fingerprint(config, "anchor_sanity"),
                "failure_rule": f"target exact top-1 collisions >= {settings['target_collision_failure_count']} of {settings['seeds_per_anchor']}",
                "evaluator": evaluator.identity,
                "generation_runtime": generation.identity,
                "anchors": [summaries[-1]],
                "failed_anchors": [anchor],
                "images_retained": True,
                "images_root": str((root / "anchor_sanity" / "images").resolve()),
                "stopped_immediately": True,
            }
            protocol.write_json_atomic(root / "anchor_sanity" / "per_image.json", {
                "items": [item for item in all_results if item["anchor"] == anchor]
            })
            protocol.write_json_atomic(gate_path, gate)
            raise PipelineError(f"ANCHOR_SANITY_FAILURE: {anchor}")
    failed: list[dict[str, Any]] = []
    gate = {
        "schema_version": 1,
        "status": "failed" if failed else "passed",
        "completed_at": protocol.utc_now(),
        "protocol_fingerprint": _stage_fingerprint(config, "anchor_sanity"),
        "failure_rule": f"target exact top-1 collisions >= {settings['target_collision_failure_count']} of {settings['seeds_per_anchor']}",
        "evaluator": evaluator.identity,
        "generation_runtime": generation.identity,
        "anchors": summaries,
        "failed_anchors": [item["anchor"] for item in failed],
        "images_retained": True,
        "images_root": str((root / "anchor_sanity" / "images").resolve()),
    }
    protocol.write_json_atomic(root / "anchor_sanity" / "per_image.json", {"items": all_results})
    protocol.write_json_atomic(gate_path, gate)
    return gate


def _stage_fingerprint(config: Mapping[str, Any], stage: str) -> str:
    return protocol.fingerprint({
        "stage": stage,
        "config": config,
        "config_sha256": protocol.sha256(Path(config["_resolved"]["config_path"])),
        "anchors_sha256": protocol.sha256(Path(config["_resolved"]["anchors_path"])),
        "pipeline_sha256": protocol.sha256(Path(__file__)),
    })


def _legacy_reference(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    reference_path = Path(config["_resolved"]["legacy_original_reference"])
    reference = protocol.read_json(reference_path)
    if reference.get("status") != "conditional_reusable_original_reference":
        raise PipelineError("Legacy Original reference is not conditionally reusable")
    archive_root = reference_path.parent
    dataset_hash = protocol.sha256(Path(config["_resolved"]["evaluation_dataset"]))
    if reference.get("dataset_sha256") != dataset_hash:
        raise PipelineError("Legacy Original dataset hash differs from current protocol")
    if (
        reference.get("row_count") != config["evaluation"]["expected_total_rows"]
        or reference.get("class_count") != 25
        or reference.get("rows_per_class") != config["evaluation"]["expected_rows_per_class"]
        or reference.get("ordered_rows_verified_against_dataset") is not True
        or reference.get("images_purged") is not True
        or reference.get("reuse_forbidden_until_canary_passes") is not True
        or reference.get("invalid_edited_checkpoints_are_not_part_of_this_reference") is not True
    ):
        raise PipelineError("Legacy Original reference completeness/reuse policy is invalid")
    expected_generation = {
        "base_model": config["model"]["base_model"],
        "scheduler": config["evaluation"]["generation"]["scheduler"],
        "steps": config["evaluation"]["generation"]["num_inference_steps"],
        "cfg": config["evaluation"]["generation"]["guidance_scale"],
        "resolution": [
            config["evaluation"]["generation"]["width"],
            config["evaluation"]["generation"]["height"],
        ],
        "dtype": config["model"]["generation_dtype"],
        "images_per_prompt": config["evaluation"]["generation"]["images_per_prompt"],
        "generator_device": config["evaluation"]["generation"]["generator_device"],
    }
    if reference.get("generation") != expected_generation:
        raise PipelineError("Legacy Original generation protocol differs from current protocol")
    classifier = reference.get("classifier", {})
    expected_classifier = config["evaluation"]["classifier"]
    if (
        classifier.get("implementation") != expected_classifier["implementation"]
        or classifier.get("resolved_weights") != expected_classifier["weights_enum"]
        or classifier.get("weight_filename") != expected_classifier["expected_weight_filename"]
        or classifier.get("matching") != expected_classifier["matching"]
    ):
        raise PipelineError("Legacy Original evaluator protocol differs from current protocol")
    for relative_key, hash_key in (
        ("source_resolved_plan", "source_resolved_plan_sha256"),
        ("source_summary", "source_summary_sha256"),
    ):
        source = archive_root / reference[relative_key]
        if not source.is_file() or protocol.sha256(source) != reference[hash_key]:
            raise PipelineError(f"Legacy Original reference source hash mismatch: {source}")
    return reference, archive_root


def _legacy_original_manifest(archive_root: Path, group_id: str, concept: str) -> dict[str, Any]:
    path = archive_root / "evaluation" / "manifests" / f"original__{group_id}__{protocol.slug(concept)}.json"
    manifest = protocol.read_json(path)
    if manifest.get("model_type") != "original" or protocol.normalize(manifest.get("evaluated_concept", "")) != protocol.normalize(concept):
        raise PipelineError(f"Legacy Original manifest identity mismatch: {path}")
    return manifest


def _legacy_item_hashes(manifest: Mapping[str, Any]) -> dict[int, str]:
    output: dict[int, str] = {}
    for item in manifest.get("items", []):
        case = int(item["case_number"])
        value = item.get("image_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise PipelineError(f"Legacy Original manifest has invalid image hash for case {case}")
        output[case] = value
    return output


def smoke(config: Mapping[str, Any], anchors: Mapping[str, str], *, skip_existing: bool) -> dict[str, Any]:
    root = Path(config["_resolved"]["output_root"])
    gate_path = root / "smoke" / "gate.json"
    stage_fp = _stage_fingerprint(config, "smoke")
    if skip_existing and gate_path.is_file():
        existing = protocol.read_json(gate_path)
        if existing.get("status") == "passed" and existing.get("protocol_fingerprint") == stage_fp:
            return existing
    if gate_path.exists():
        raise PipelineError(f"Smoke output collision: {gate_path.parent}")
    _require_gate(root, "anchor_sanity")
    reference, archive_root = _legacy_reference(config)
    _, by_class = load_dataset(config)
    checkpoints = _checkpoint_lookup(config, anchors)
    for spec in checkpoints.values():
        _validate_checkpoint(spec, config)
    generation = GenerationRuntime(config)
    evaluator = EvaluationRuntime(config)
    per_image: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    smoke_config = config["smoke_gate"]
    count = int(smoke_config["ordered_rows_per_target"])

    # Canary phase is deliberately complete before any edited generation. A
    # single PNG mismatch stops immediately and prevents mixing legacy
    # Original results with newly rendered edited images.
    original_metrics: dict[str, list[dict[str, Any]]] = {}
    for target in smoke_config["targets"]:
        group = _group_for_target(config, target)
        rows = by_class[protocol.normalize(target)][:count]
        legacy_hashes = _legacy_item_hashes(_legacy_original_manifest(archive_root, group["id"], target))
        generation.activate_original()
        image_paths: list[Path] = []
        for row in rows:
            path = root / "smoke" / "images" / protocol.slug(target) / "original" / f"case-{row['case_number']:06d}_seed-{row['evaluation_seed']}.png"
            image = generation.generate(row["prompt"], row["evaluation_seed"])
            image_hash = _save_png(image, path)
            expected_hash = legacy_hashes.get(row["case_number"])
            if expected_hash != image_hash:
                mismatch = {
                    "target": target,
                    "case_number": row["case_number"],
                    "legacy_sha256": expected_hash,
                    "new_sha256": image_hash,
                    "image_path": str(path.resolve()),
                }
                canary = {
                    "schema_version": 1,
                    "status": "failed",
                    "checked_images_before_stop": sum(len(value) for value in original_metrics.values()) + len(image_paths) + 1,
                    "mismatch_count": 1,
                    "mismatches": [mismatch],
                    "legacy_reference": reference,
                    "generation_runtime": generation.identity,
                    "stopped_immediately": True,
                    "completed_at": protocol.utc_now(),
                }
                protocol.write_json_atomic(root / "original_canary" / "gate.json", canary)
                protocol.write_json_atomic(gate_path, {
                    "schema_version": 1,
                    "status": "blocked_by_original_canary",
                    "protocol_fingerprint": stage_fp,
                    "original_mismatch": mismatch,
                    "completed_at": protocol.utc_now(),
                })
                raise PipelineError(f"ORIGINAL_REPRODUCTION_HASH_MISMATCH: {target} case {row['case_number']}")
            image_paths.append(path)
        metrics = evaluator.evaluate(image_paths, target)
        original_metrics[target] = metrics
        for row, metric in zip(rows, metrics):
            per_image.append({
                "group_id": group["id"], "target": target, "model_type": "original",
                **row, "checkpoint_sha256": None, **metric,
            })
    canary = {
        "schema_version": 1,
        "status": "passed",
        "checked_images": len(smoke_config["targets"]) * count,
        "mismatch_count": 0,
        "mismatches": [],
        "legacy_reference": reference,
        "generation_runtime": generation.identity,
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(root / "original_canary" / "gate.json", canary)

    # Edited phase is target-ordered. Each Single is evaluated before its Joint;
    # a failed Single gate stops before any further edited generation.
    for target in smoke_config["targets"]:
        group = _group_for_target(config, target)
        rows = by_class[protocol.normalize(target)][:count]
        model_results: dict[str, list[dict[str, Any]]] = {"original": original_metrics[target]}
        single_spec = checkpoints[(group["id"], "single", target)]
        checkpoint_hash = generation.activate_checkpoint(single_spec)
        single_paths: list[Path] = []
        for row in rows:
            path = root / "smoke" / "images" / protocol.slug(target) / "single" / f"case-{row['case_number']:06d}_seed-{row['evaluation_seed']}.png"
            _save_png(generation.generate(row["prompt"], row["evaluation_seed"]), path)
            single_paths.append(path)
        single_metrics = evaluator.evaluate(single_paths, target)
        model_results["single"] = single_metrics
        for row, metric in zip(rows, single_metrics):
            per_image.append({
                "group_id": group["id"], "target": target, "model_type": "single",
                **row, "checkpoint_sha256": checkpoint_hash, **metric,
            })
        original_correct = sum(bool(item["correct"]) for item in model_results["original"])
        single_correct = sum(bool(item["correct"]) for item in model_results["single"])
        drop = original_correct - single_correct
        provisional = {
            "group_id": group["id"],
            "target": target,
            "total": count,
            "original_correct": original_correct,
            "original_accuracy": original_correct / count,
            "single_correct": single_correct,
            "single_accuracy": single_correct / count,
            "original_mean_target_probability": sum(item["target_probability"] for item in model_results["original"]) / count,
            "single_mean_target_probability": sum(item["target_probability"] for item in model_results["single"]) / count,
            "original_mean_raw_target_logit": sum(item["raw_target_logit"] for item in model_results["original"]) / count,
            "single_mean_raw_target_logit": sum(item["raw_target_logit"] for item in model_results["single"]) / count,
            "joint_correct": None,
            "joint_accuracy": None,
            "joint_mean_target_probability": None,
            "joint_mean_raw_target_logit": None,
            "original_minus_single_count": drop,
            "original_minus_single_percentage_points": 100.0 * drop / count,
            "passed": drop >= int(smoke_config["required_accuracy_drop_count"]),
        }
        if not provisional["passed"]:
            summaries.append(provisional)
            protocol.write_json_atomic(root / "smoke" / "per_image.json", {"items": per_image})
            gate = {
                "schema_version": 1,
                "status": "failed",
                "protocol_fingerprint": stage_fp,
                "rule": "all four Original exact top-1 minus Single exact top-1 counts >= 4 of 32",
                "joint_is_gate": False,
                "evaluator": evaluator.identity,
                "generation_runtime": generation.identity,
                "targets": [provisional],
                "failed_targets": [target],
                "images_retained": True,
                "images_root": str((root / "smoke" / "images").resolve()),
                "stopped_immediately": True,
                "completed_at": protocol.utc_now(),
            }
            protocol.write_json_atomic(gate_path, gate)
            raise PipelineError(f"SINGLE_SMOKE_GATE_FAILURE: {target}")

        joint_spec = checkpoints[(group["id"], "joint", None)]
        checkpoint_hash = generation.activate_checkpoint(joint_spec)
        joint_paths: list[Path] = []
        for row in rows:
            path = root / "smoke" / "images" / protocol.slug(target) / "joint" / f"case-{row['case_number']:06d}_seed-{row['evaluation_seed']}.png"
            _save_png(generation.generate(row["prompt"], row["evaluation_seed"]), path)
            joint_paths.append(path)
        joint_metrics = evaluator.evaluate(joint_paths, target)
        joint_correct = sum(bool(item["correct"]) for item in joint_metrics)
        for row, metric in zip(rows, joint_metrics):
            per_image.append({
                "group_id": group["id"], "target": target, "model_type": "joint",
                **row, "checkpoint_sha256": checkpoint_hash, **metric,
            })
        provisional["joint_correct"] = joint_correct
        provisional["joint_accuracy"] = joint_correct / count
        provisional["joint_mean_target_probability"] = sum(item["target_probability"] for item in joint_metrics) / count
        provisional["joint_mean_raw_target_logit"] = sum(item["raw_target_logit"] for item in joint_metrics) / count
        summaries.append(provisional)

    protocol.write_json_atomic(root / "smoke" / "per_image.json", {"items": per_image})
    gate = {
        "schema_version": 1,
        "status": "passed",
        "protocol_fingerprint": stage_fp,
        "rule": "all four Original exact top-1 minus Single exact top-1 counts >= 4 of 32",
        "joint_is_gate": False,
        "evaluator": evaluator.identity,
        "generation_runtime": generation.identity,
        "targets": summaries,
        "failed_targets": [],
        "images_retained": True,
        "images_root": str((root / "smoke" / "images").resolve()),
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(gate_path, gate)
    return gate


def _job_identity(mode: str, group: Mapping[str, Any], concept: str, single_target: str | None, spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_type": mode,
        "group_id": group["id"],
        "single_target": single_target,
        "targets": spec["targets"],
        "evaluated_concept": concept,
        "checkpoint_sha256": protocol.sha256(Path(spec["checkpoint_path"])),
        "rows_sha256": _ordered_rows_hash(rows),
        "generation": config["evaluation"]["generation"],
        "classifier": config["evaluation"]["classifier"],
    }


def _formal_jobs(config: Mapping[str, Any], anchors: Mapping[str, str]) -> list[dict[str, Any]]:
    _, by_class = load_dataset(config)
    checkpoints = _checkpoint_lookup(config, anchors)
    jobs: list[dict[str, Any]] = []
    root = Path(config["_resolved"]["output_root"]) / "formal"
    for group in config["groups"]:
        for target in group["targets"]:
            spec = checkpoints[(group["id"], "single", target)]
            for concept in group["concepts"]:
                rows = by_class[protocol.normalize(concept)]
                identity = _job_identity("single", group, concept, target, spec, rows, config)
                job_id = f"single__{group['id']}__{protocol.slug(target)}__{protocol.slug(concept)}"
                jobs.append({
                    **identity,
                    "job_id": job_id,
                    "job_fingerprint": protocol.fingerprint(identity),
                    "checkpoint_spec": spec,
                    "rows": rows,
                    "image_dir": str(root / "images" / "single" / group["id"] / protocol.slug(target) / protocol.slug(concept)),
                    "manifest_path": str(root / "manifests" / f"{job_id}.json"),
                    "result_path": str(root / "evaluations" / "shards" / f"{job_id}.json"),
                })
        spec = checkpoints[(group["id"], "joint", None)]
        for concept in group["concepts"]:
            rows = by_class[protocol.normalize(concept)]
            identity = _job_identity("joint", group, concept, None, spec, rows, config)
            job_id = f"joint__{group['id']}__{protocol.slug(concept)}"
            jobs.append({
                **identity,
                "job_id": job_id,
                "job_fingerprint": protocol.fingerprint(identity),
                "checkpoint_spec": spec,
                "rows": rows,
                "image_dir": str(root / "images" / "joint" / group["id"] / protocol.slug(concept)),
                "manifest_path": str(root / "manifests" / f"{job_id}.json"),
                "result_path": str(root / "evaluations" / "shards" / f"{job_id}.json"),
            })
    expected = int(config["evaluation"]["formal"]["total_new_edited_images"])
    actual = sum(len(job["rows"]) for job in jobs)
    if actual != expected:
        raise PipelineError(f"Formal edited plan resolves {actual} images, expected {expected}")
    return jobs


def _run_formal_job(job: Mapping[str, Any], generation: GenerationRuntime, evaluator: EvaluationRuntime, *, skip_existing: bool, purge: bool) -> dict[str, Any]:
    result_path = Path(job["result_path"])
    manifest_path = Path(job["manifest_path"])
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        manifest = protocol.read_json(manifest_path)
        if manifest.get("job_fingerprint") != job["job_fingerprint"]:
            raise PipelineError(f"Formal manifest fingerprint collision: {job['job_id']}")
    if result_path.is_file():
        existing = protocol.read_json(result_path)
        if existing.get("status") != "complete" or existing.get("job_fingerprint") != job["job_fingerprint"]:
            raise PipelineError(f"Formal result fingerprint collision: {job['job_id']}")
        if not skip_existing:
            raise PipelineError(f"Formal result already exists: {job['job_id']}")
        if manifest is None:
            raise PipelineError(f"Formal result is missing its manifest: {job['job_id']}")
        if purge and manifest.get("status") != "purged":
            for item in manifest.get("items", []):
                path = Path(item["image_path"])
                if path.is_file():
                    if protocol.sha256(path) != item["image_sha256"]:
                        raise PipelineError(f"Refusing to purge hash-mismatched image: {path}")
                    path.unlink()
            manifest["status"] = "purged"
            manifest["purged_at"] = protocol.utc_now()
            protocol.write_json_atomic(manifest_path, manifest)
        return existing
    if manifest is not None and not skip_existing:
        raise PipelineError(f"Partial formal manifest already exists: {job['job_id']}")
    if manifest is None and Path(job["image_dir"]).exists():
        raise PipelineError(f"Untracked formal image directory collision: {job['job_id']}")
    generation.activate_checkpoint(job["checkpoint_spec"])
    image_paths: list[Path] = []
    if manifest is None:
        manifest = {
            "schema_version": 2,
            "status": "generating",
            "job_id": job["job_id"],
            "job_fingerprint": job["job_fingerprint"],
            "model_type": job["model_type"],
            "single_target": job["single_target"],
            "targets": job["targets"],
            "evaluated_concept": job["evaluated_concept"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "generation_runtime": generation.identity,
            "evaluator": evaluator.identity,
            "items": [],
            "started_at": protocol.utc_now(),
        }
        protocol.write_json_atomic(manifest_path, manifest)
    recorded = {int(item["case_number"]): item for item in manifest.get("items", [])}
    for row in job["rows"]:
        path = Path(job["image_dir"]) / f"case-{row['case_number']:06d}_seed-{row['evaluation_seed']}.png"
        previous = recorded.get(int(row["case_number"]))
        if previous is not None and path.is_file():
            if previous.get("image_sha256") != protocol.sha256(path):
                raise PipelineError(f"Existing formal image hash mismatch: {path}")
            image_paths.append(path)
            continue
        if previous is not None or path.exists():
            raise PipelineError(f"Incomplete/untracked formal image collision: {path}")
        image = generation.generate(row["prompt"], row["evaluation_seed"])
        image_hash = _save_png(image, path)
        image_paths.append(path)
        recorded[int(row["case_number"])] = {
            **row,
            "image_path": str(path.resolve()),
            "image_sha256": image_hash,
            "generated_at": protocol.utc_now(),
        }
        manifest["items"] = [recorded[key] for key in sorted(recorded)]
        protocol.write_json_atomic(manifest_path, manifest)
    if len(image_paths) != len(job["rows"]) or len(recorded) != len(job["rows"]):
        raise PipelineError(f"Formal generation count mismatch: {job['job_id']}")
    manifest["status"] = "generated"
    manifest["generation_completed_at"] = protocol.utc_now()
    protocol.write_json_atomic(manifest_path, manifest)
    metrics = evaluator.evaluate(image_paths, job["evaluated_concept"])
    items = [{**row, **metric} for row, metric in zip(job["rows"], metrics)]
    correct = sum(bool(item["correct"]) for item in items)
    result = {
        "schema_version": 2,
        "status": "complete",
        "job_id": job["job_id"],
        "job_fingerprint": job["job_fingerprint"],
        "group_id": job["group_id"],
        "model_type": job["model_type"],
        "single_target": job["single_target"],
        "targets": job["targets"],
        "evaluated_concept": job["evaluated_concept"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "generation_runtime": generation.identity,
        "evaluator": evaluator.identity,
        "correct": correct,
        "total": len(items),
        "accuracy": correct / len(items),
        "mean_target_probability": sum(item["target_probability"] for item in items) / len(items),
        "mean_raw_target_logit": sum(item["raw_target_logit"] for item in items) / len(items),
        "items": items,
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(result_path, result)
    manifest["status"] = "evaluated"
    if purge:
        for path in image_paths:
            path.unlink()
        manifest["status"] = "purged"
        manifest["purged_at"] = protocol.utc_now()
    protocol.write_json_atomic(manifest_path, manifest)
    return result


def formal(config: Mapping[str, Any], anchors: Mapping[str, str], *, skip_existing: bool) -> dict[str, Any]:
    root = Path(config["_resolved"]["output_root"])
    _require_gate(root, "anchor_sanity")
    _require_gate(root, "original_canary")
    _require_gate(root, "smoke")
    jobs = _formal_jobs(config, anchors)
    for spec in protocol.checkpoint_specs(config, anchors):
        _validate_checkpoint(spec, config)
    formal_root = root / "formal"
    generation = GenerationRuntime(config)
    evaluator = EvaluationRuntime(config)
    protocol.write_json_atomic(formal_root / "resolved_plan.json", {
        "schema_version": 2,
        "status": "resolved",
        "new_edited_image_count": sum(len(job["rows"]) for job in jobs),
        "original_generation_count": 0,
        "generation_runtime": generation.identity,
        "evaluator": evaluator.identity,
        "jobs": [{key: value for key, value in job.items() if key not in {"rows", "checkpoint_spec"}} for job in jobs],
    })
    for index, job in enumerate(jobs, start=1):
        _state(root, "formal", "running", current_job=job["job_id"], job_index=index, total_jobs=len(jobs))
        _event(root, "formal_job_started", job_id=job["job_id"], index=index, total=len(jobs))
        _run_formal_job(
            job,
            generation,
            evaluator,
            skip_existing=skip_existing,
            purge=bool(config["evaluation"]["formal"]["purge_edited_images_after_durable_evaluation"]),
        )
        _event(root, "formal_job_complete", job_id=job["job_id"], index=index, total=len(jobs))
    complete = {
        "schema_version": 1,
        "status": "complete",
        "job_count": len(jobs),
        "new_edited_image_count": sum(len(job["rows"]) for job in jobs),
        "original_generation_count": 0,
        "generation_runtime": generation.identity,
        "evaluator": evaluator.identity,
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(formal_root / "completion.json", complete)
    return complete


def _load_legacy_original_results(config: Mapping[str, Any], archive_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    _, by_class = load_dataset(config)
    for group in config["groups"]:
        for concept in group["concepts"]:
            path = archive_root / "evaluation" / "evaluations" / "shards" / f"original__{group['id']}__{protocol.slug(concept)}.json"
            row = protocol.read_json(path)
            expected_rows = by_class[protocol.normalize(concept)]
            items = row.get("items", [])
            if (
                row.get("model_type") != "original"
                or row.get("checkpoint_path") is not None
                or row.get("checkpoint_sha256") is not None
                or row.get("total") != config["evaluation"]["expected_rows_per_class"]
                or len(items) != len(expected_rows)
            ):
                raise PipelineError(f"Invalid legacy Original evaluation shard: {path}")
            manifest_hashes = _legacy_item_hashes(
                _legacy_original_manifest(archive_root, group["id"], concept)
            )
            for expected, item in zip(expected_rows, items):
                if (
                    int(item.get("case_number", -1)) != expected["case_number"]
                    or item.get("prompt") != expected["prompt"]
                    or int(item.get("seed", -1)) != expected["evaluation_seed"]
                    or protocol.normalize(item.get("expected_category", "")) != protocol.normalize(concept)
                    or item.get("image_sha256") != manifest_hashes.get(expected["case_number"])
                ):
                    raise PipelineError(
                        f"Legacy Original ordered row/hash mismatch: {path} case {expected['case_number']}"
                    )
            output[(group["id"], protocol.normalize(concept))] = row
    if len(output) != 25:
        raise PipelineError(f"Expected 25 legacy Original shards, found {len(output)}")
    return output


def aggregate(config: Mapping[str, Any], anchors: Mapping[str, str]) -> dict[str, Any]:
    root = Path(config["_resolved"]["output_root"])
    _require_gate(root, "anchor_sanity")
    _require_gate(root, "original_canary")
    _require_gate(root, "smoke")
    completion = protocol.read_json(root / "formal" / "completion.json")
    if completion.get("status") != "complete" or completion.get("new_edited_image_count") != 37500:
        raise PipelineError("Formal edited generation/evaluation is incomplete")
    _, archive_root = _legacy_reference(config)
    originals = _load_legacy_original_results(config, archive_root)
    jobs = _formal_jobs(config, anchors)
    edited: dict[tuple[str, str, str | None, str], dict[str, Any]] = {}
    per_class: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for job in jobs:
        result = protocol.read_json(Path(job["result_path"]))
        if result.get("status") != "complete" or result.get("job_fingerprint") != job["job_fingerprint"]:
            raise PipelineError(f"Formal result fingerprint mismatch: {job['result_path']}")
        if result.get("total") != 500 or len(result.get("items", [])) != 500:
            raise PipelineError(f"Formal result row count mismatch: {job['result_path']}")
        key = (job["group_id"], job["model_type"], job["single_target"], protocol.normalize(job["evaluated_concept"]))
        edited[key] = result
        per_class.append({
            "group_id": job["group_id"],
            "model_type": job["model_type"],
            "single_target": job["single_target"] or "",
            "evaluated_concept": job["evaluated_concept"],
            "correct": result["correct"],
            "total": result["total"],
            "accuracy": result["accuracy"],
            "mean_target_probability": result["mean_target_probability"],
            "mean_raw_target_logit": result["mean_raw_target_logit"],
        })
        for item in result["items"]:
            per_image.append({
                "group_id": job["group_id"],
                "model_type": job["model_type"],
                "single_target": job["single_target"] or "",
                "evaluated_concept": job["evaluated_concept"],
                **item,
            })
    if len(edited) != 75 or len(per_image) != 37500:
        raise PipelineError(
            f"Expected 75 edited shards / 37500 rows, found {len(edited)} / {len(per_image)}"
        )
    comparisons: list[dict[str, Any]] = []
    preservation: list[dict[str, Any]] = []
    sibling: list[dict[str, Any]] = []
    for group in config["groups"]:
        group_id = group["id"]
        for target in group["targets"]:
            norm_target = protocol.normalize(target)
            original = originals[(group_id, norm_target)]
            single = edited[(group_id, "single", target, norm_target)]
            joint = edited[(group_id, "joint", None, norm_target)]
            comparisons.append({
                "group_id": group_id,
                "target": target,
                "anchor": anchors[target],
                "original_target_correct": original["correct"],
                "original_target_total": original["total"],
                "original_target_residual_accuracy": original["accuracy"],
                "single_target_residual_accuracy": single["accuracy"],
                "joint_target_residual_accuracy": joint["accuracy"],
                "joint_minus_single_target_residual": joint["accuracy"] - single["accuracy"],
                "original_auxiliary_metrics": "unavailable_for_legacy_original_full_baseline",
                "single_mean_target_probability": single["mean_target_probability"],
                "joint_mean_target_probability": joint["mean_target_probability"],
                "single_mean_raw_target_logit": single["mean_raw_target_logit"],
                "joint_mean_raw_target_logit": joint["mean_raw_target_logit"],
            })
            preserve_rows: list[dict[str, Any]] = []
            for concept in group["similar_non_targets"]:
                concept_norm = protocol.normalize(concept)
                original_p = originals[(group_id, concept_norm)]
                single_p = edited[(group_id, "single", target, concept_norm)]
                joint_p = edited[(group_id, "joint", None, concept_norm)]
                row = {
                    "group_id": group_id,
                    "single_target": target,
                    "preservation_concept": concept,
                    "original_accuracy": original_p["accuracy"],
                    "single_preservation_accuracy": single_p["accuracy"],
                    "joint_preservation_accuracy": joint_p["accuracy"],
                    "joint_minus_single_preservation": joint_p["accuracy"] - single_p["accuracy"],
                    "original_auxiliary_metrics": "unavailable_for_legacy_original_full_baseline",
                    "single_mean_target_probability": single_p["mean_target_probability"],
                    "joint_mean_target_probability": joint_p["mean_target_probability"],
                    "single_mean_raw_target_logit": single_p["mean_raw_target_logit"],
                    "joint_mean_raw_target_logit": joint_p["mean_raw_target_logit"],
                }
                preservation.append(row)
                preserve_rows.append(row)
            comparisons[-1].update({
                "single_preservation_macro_accuracy": sum(row["single_preservation_accuracy"] for row in preserve_rows) / 3,
                "joint_preservation_macro_accuracy": sum(row["joint_preservation_accuracy"] for row in preserve_rows) / 3,
                "joint_minus_single_preservation_macro": sum(row["joint_minus_single_preservation"] for row in preserve_rows) / 3,
            })
            other = next(value for value in group["targets"] if protocol.normalize(value) != norm_target)
            sibling_result = edited[(group_id, "single", target, protocol.normalize(other))]
            sibling.append({
                "group_id": group_id,
                "single_target": target,
                "sibling_target": other,
                "single_sibling_accuracy": sibling_result["accuracy"],
                "role": "secondary_diagnostic_only",
            })
    aggregate_root = root / "formal" / "aggregates"
    _write_csv(aggregate_root / "target_residual.csv", comparisons, list(comparisons[0]))
    _write_csv(aggregate_root / "similar_non_target_preservation.csv", preservation, list(preservation[0]))
    _write_csv(aggregate_root / "sibling_target_secondary.csv", sibling, list(sibling[0]))
    _write_csv(root / "formal" / "evaluations" / "per_class.csv", per_class, list(per_class[0]))
    _write_csv(root / "formal" / "evaluations" / "per_image.csv", per_image, [
        "group_id", "model_type", "single_target", "evaluated_concept", "case_number", "source_line",
        "prompt", "class", "evaluation_seed", "image_path", "image_sha256", "expected_index",
        "expected_category", "predicted_index", "predicted_category", "correct", "target_probability",
        "raw_target_logit", "top5",
    ])
    summary = {
        "schema_version": 2,
        "status": "complete",
        "research_question": "Does moving from effective Single OCE erasure to a two-target semantically overlapping Joint subspace introduce extra target-erasure failure or similar-non-target collateral damage?",
        "primary_behavior": "official released repository behavior",
        "formal_original_source": "conditional_reusable_original_reference",
        "formal_original_auxiliary_metrics": "unavailable_for_legacy_original_full_baseline",
        "formal_original_regenerated_images": 0,
        "formal_new_edited_images": 37500,
        "interpretation": {
            "joint_minus_single_target_residual": "positive means Joint erasure is worse",
            "joint_minus_single_preservation": "negative means additional Joint collateral damage",
        },
        "target_rows": comparisons,
        "preservation_rows": preservation,
        "sibling_target_rows": sibling,
        "provenance_categories": config["provenance_notes"],
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(aggregate_root / "summary.json", summary)
    return summary


def run_subprocess(script: Path, config_path: Path, *, skip_existing: bool) -> None:
    command = [sys.executable, str(script), "--config", str(config_path)]
    if skip_existing:
        command.append("--skip-existing")
    subprocess.run(command, cwd=HERE, check=True)


def status(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["_resolved"]["output_root"])
    paths = {
        "k0": root / "artifacts" / config["k0"]["metadata_filename"],
        "anchor_sanity": root / "anchor_sanity" / "gate.json",
        "checkpoints": root / "checkpoint_summary.json",
        "original_canary": root / "original_canary" / "gate.json",
        "smoke": root / "smoke" / "gate.json",
        "formal": root / "formal" / "completion.json",
        "aggregate": root / "formal" / "aggregates" / "summary.json",
    }
    states: dict[str, Any] = {}
    for name, path in paths.items():
        states[name] = protocol.read_json(path).get("status") if path.is_file() else "missing"
    retained_pngs = list(root.rglob("*.png")) if root.exists() else []
    return {
        "output_root": str(root),
        "stages": states,
        "retained_png_count": len(retained_pngs),
        "retained_png_bytes": sum(path.stat().st_size for path in retained_pngs),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "k0", "anchor-sanity", "checkpoints", "smoke", "formal", "aggregate", "all", "status"))
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    try:
        plan, config, anchors = build_plan(config_path)
        root = Path(config["_resolved"]["output_root"])
        if args.stage == "plan":
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        if args.stage == "status":
            print(json.dumps(status(config), indent=2, ensure_ascii=False))
            return 0
        root.mkdir(parents=True, exist_ok=True)
        protocol.write_json_atomic(root / "resolved_pipeline_plan.json", plan)
        _state(root, args.stage, "running", plan_fingerprint=plan["plan_fingerprint"])
        if args.stage == "k0":
            run_subprocess(COMPUTE_K0, config_path, skip_existing=args.skip_existing)
        elif args.stage == "anchor-sanity":
            anchor_sanity(config, anchors, skip_existing=args.skip_existing)
        elif args.stage == "checkpoints":
            _require_gate(root, "anchor_sanity")
            run_subprocess(CHECKPOINT_BUILDER, config_path, skip_existing=args.skip_existing)
        elif args.stage == "smoke":
            smoke(config, anchors, skip_existing=args.skip_existing)
        elif args.stage == "formal":
            formal(config, anchors, skip_existing=args.skip_existing)
        elif args.stage == "aggregate":
            aggregate(config, anchors)
        elif args.stage == "all":
            run_subprocess(COMPUTE_K0, config_path, skip_existing=args.skip_existing)
            anchor_sanity(config, anchors, skip_existing=args.skip_existing)
            run_subprocess(CHECKPOINT_BUILDER, config_path, skip_existing=args.skip_existing)
            smoke(config, anchors, skip_existing=args.skip_existing)
            formal(config, anchors, skip_existing=args.skip_existing)
            aggregate(config, anchors)
        _state(root, args.stage, "complete", finished_at=protocol.utc_now())
        return 0
    except Exception as exc:
        try:
            if "config" in locals():
                _state(Path(config["_resolved"]["output_root"]), args.stage, "failed", error=str(exc))
        except Exception:
            pass
        print(f"pipeline error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
