#!/usr/bin/env python3
"""Shared protocol resolution for the primary Confuse5 OCE rerun.

This module intentionally imports only the Python standard library. Model
libraries are imported lazily by the server-only execution stages.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_ANCHORS = HERE / "anchors.json"
ARCHIVE_ROOT = HERE / "archives" / "invalid_for_primary__pilot_default_config"


class ProtocolError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", normalize(value)).strip("-")
    if not result:
        raise ProtocolError(f"Cannot form a safe slug from {value!r}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object in {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def resolve_relative(raw: str | Path, relative_to: Path = HERE) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _strings(value: Any, field: str, expected_count: int | None = None) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProtocolError(f"{field} must be a list of strings")
    cleaned = [" ".join(item.split()) for item in value]
    if any(not item for item in cleaned):
        raise ProtocolError(f"{field} contains an empty value")
    if len({normalize(item) for item in cleaned}) != len(cleaned):
        raise ProtocolError(f"{field} contains normalized duplicates")
    if expected_count is not None and len(cleaned) != expected_count:
        raise ProtocolError(f"{field} must contain {expected_count} values")
    return cleaned


def load_protocol(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, str]]:
    config = read_json(config_path)
    if config.get("schema_version") != 2:
        raise ProtocolError("Primary Confuse5 config schema_version must be 2")
    if config.get("experiment_id") != "confuse5_single_vs_joint__official_repo_primary_v1":
        raise ProtocolError("Unexpected primary experiment_id")
    if config.get("source_precedence", [])[-1:] != ["parser_defaults_forbidden"]:
        raise ProtocolError("Parser defaults must be explicitly forbidden")

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ProtocolError("paths must be an object")
    anchor_path = resolve_relative(paths.get("anchors", ""), config_path.parent)
    anchor_payload = read_json(anchor_path)
    raw_anchors = anchor_payload.get("anchors")
    if not isinstance(raw_anchors, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_anchors.items()
    ):
        raise ProtocolError("anchors.json anchors must be a string mapping")
    anchors = {normalize(key): " ".join(value.split()) for key, value in raw_anchors.items()}

    model = config.get("model", {})
    required_model = {
        "base_model": "CompVis/stable-diffusion-v1-4",
        "concept_type": "object",
        "editing_dtype": "float32",
        "generation_dtype": "bfloat16",
        "safety_checker": None,
    }
    for key, expected in required_model.items():
        if model.get(key) != expected:
            raise ProtocolError(f"model.{key} must be {expected!r}")
    if not isinstance(model.get("device"), str) or not model["device"].strip():
        raise ProtocolError("model.device must be explicit")

    oce = config.get("oce", {})
    locked_oce = {
        "lambda_e": 1000.0,
        "lambda_0": 50.0,
        "lambda_r": 1.0,
        "lamb_repo_regularizer": 10.0,
        "expand_prompts": True,
        "expansion_order": "all_bare_then_per_concept_extras",
        "normalization_eps": 1e-8,
        "qr_mode": "reduced",
        "erasure_matrix_order": "-R(I-R_star)",
        "embedding_token": "last_non_special_token_attention_mask_sum_minus_2",
        "subspace_input_normalization": "l2",
        "svd": "torch.linalg.svd_full_matrices_false",
        "determinant_correction": "official_repo_column_flip_after_UVt_when_det_negative",
        "local_retain_policy": "three_designated_similar_non_targets",
        "anchor_in_local_retain": False,
        "guide_alignment_seed": 42,
    }
    for key, expected in locked_oce.items():
        if oce.get(key) != expected:
            raise ProtocolError(f"oce.{key} must be locked to {expected!r}")
    templates = _strings(oce.get("expansion_templates"), "oce.expansion_templates", 6)
    if templates[0] != "{concept}" or any("{concept}" not in item for item in templates):
        raise ProtocolError("Every expansion template must contain {concept}; bare concept first")

    groups = config.get("groups")
    if not isinstance(groups, list) or len(groups) != 5:
        raise ProtocolError("groups must contain exactly five Confuse5 groups")
    target_names: list[str] = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ProtocolError(f"groups[{index}] must be an object")
        group_id = slug(str(group.get("id", "")))
        targets = _strings(group.get("targets"), f"groups[{index}].targets", 2)
        retains = _strings(
            group.get("similar_non_targets"),
            f"groups[{index}].similar_non_targets",
            3,
        )
        group_norms = {normalize(value) for value in targets + retains}
        if len(group_norms) != 5:
            raise ProtocolError(f"Group {group_id} target/retain roles overlap")
        for target in targets:
            target_norm = normalize(target)
            if target_norm not in anchors:
                raise ProtocolError(f"Missing anchor for {target!r}")
            if normalize(anchors[target_norm]) in group_norms:
                raise ProtocolError(f"Anchor for {target!r} overlaps its Confuse5 group")
        group["id"] = group_id
        group["targets"] = targets
        group["similar_non_targets"] = retains
        group["concepts"] = targets + retains
        target_names.extend(targets)
    if set(anchors) != {normalize(value) for value in target_names}:
        raise ProtocolError("anchors.json keys must exactly equal the ten targets")

    smoke = config.get("smoke_gate", {})
    smoke_targets = _strings(smoke.get("targets"), "smoke_gate.targets", 4)
    if not {normalize(value) for value in smoke_targets} <= set(anchors):
        raise ProtocolError("Smoke targets must be configured target concepts")
    locked_smoke = {
        "ordered_rows_per_target": 32,
        "required_accuracy_drop_count": 4,
        "required_accuracy_drop": 0.125,
        "all_targets_must_pass": True,
        "joint_is_gate": False,
        "keep_images": True,
        "original_png_sha256_must_match_legacy_manifest": True,
    }
    for key, expected in locked_smoke.items():
        if smoke.get(key) != expected:
            raise ProtocolError(f"smoke_gate.{key} must be {expected!r}")

    anchor_sanity = config.get("anchor_sanity", {})
    if anchor_sanity.get("seeds_per_anchor") != 8:
        raise ProtocolError("anchor_sanity.seeds_per_anchor must be 8")
    if anchor_sanity.get("target_collision_failure_count") != 4:
        raise ProtocolError("anchor collision gate must fail at 4/8 exact target labels")
    if anchor_sanity.get("prompt_template") != "an image of a {anchor}":
        raise ProtocolError("anchor_sanity.prompt_template must be explicit and locked")
    if anchor_sanity.get("keep_images") is not True:
        raise ProtocolError("Anchor sanity PNGs must be retained")

    k0 = config.get("k0", {})
    locked_k0 = {
        "definition": "mean_outer_product_of_all_nonpadding_last_hidden_state_tokens",
        "dataset_column": "prompt",
        "batch_size": 64,
        "accumulation_dtype": "float32",
        "output_dtype": "float32",
        "process_all_rows": True,
        "flush_final_partial_batch": True,
        "resume_allowed": False,
    }
    for key, expected in locked_k0.items():
        if k0.get(key) != expected:
            raise ProtocolError(f"k0.{key} must be locked to {expected!r}")

    checkpointing = config.get("checkpointing", {})
    if checkpointing.get("single_count") != 10 or checkpointing.get("joint_count") != 5:
        raise ProtocolError("Checkpoint namespace must contain 10 Single and 5 Joint runs")
    if checkpointing.get("overwrite_allowed") is not False:
        raise ProtocolError("Primary checkpoint overwrite must remain forbidden")

    diagnostic = config.get("paper_repo_diagnostic", {})
    if diagnostic.get("primary") != {
        "lamb_repo_regularizer": 10.0, "determinant_correction": True
    }:
        raise ProtocolError("Primary diagnostic behavior must be repo lamb=10/correction=ON")
    if diagnostic.get("paper_literal") != {
        "lamb_repo_regularizer": 0.0, "determinant_correction": False
    }:
        raise ProtocolError("Paper-literal diagnostic must be lamb=0/correction=OFF")
    if diagnostic.get("checkpoint_level_only") is not True or diagnostic.get("second_full_generation_forbidden") is not True:
        raise ProtocolError("Paper/repo diagnostic must remain checkpoint-only")

    evaluation = config.get("evaluation", {})
    if evaluation.get("expected_rows_per_class") != 500 or evaluation.get("expected_total_rows") != 12500:
        raise ProtocolError("Evaluation dataset must be exactly 25 x 500 rows")
    if evaluation.get("expected_checkpoint_keys") != 16:
        raise ProtocolError("Expected checkpoint key count must be 16")
    generation = evaluation.get("generation", {})
    locked_generation = {
        "scheduler": "PNDMScheduler",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "images_per_prompt": 1,
        "generator_device": "cpu",
    }
    for key, expected in locked_generation.items():
        if generation.get(key) != expected:
            raise ProtocolError(f"evaluation.generation.{key} must be {expected!r}")
    classifier = evaluation.get("classifier", {})
    locked_classifier = {
        "implementation": "torchvision_resnet50",
        "weights_enum": "IMAGENET1K_V2",
        "expected_weight_filename": "resnet50-11ad3fa6.pth",
        "matching": "exact_normalized_category",
        "top_k": 5,
        "save_target_probability": True,
        "save_raw_target_logit": True,
    }
    for key, expected in locked_classifier.items():
        if classifier.get(key) != expected:
            raise ProtocolError(f"evaluation.classifier.{key} must be {expected!r}")
    if not isinstance(classifier.get("batch_size"), int) or classifier["batch_size"] < 1:
        raise ProtocolError("Classifier batch size must be explicitly positive")
    formal = evaluation.get("formal", {})
    locked_formal = {
        "reuse_legacy_original_top1": True,
        "legacy_original_auxiliary_metrics": "unavailable",
        "single_images": 25000,
        "joint_images": 12500,
        "total_new_edited_images": 37500,
        "evaluate_sibling_target_secondary": True,
        "primary_preservation_roles": "three_designated_similar_non_targets",
        "purge_edited_images_after_durable_evaluation": True,
    }
    for key, expected in locked_formal.items():
        if formal.get(key) != expected:
            raise ProtocolError(f"evaluation.formal.{key} must be {expected!r}")

    output_root = resolve_relative(paths["output_namespace"], config_path.parent)
    archive_root = ARCHIVE_ROOT.resolve()
    try:
        output_root.relative_to(HERE.resolve())
    except ValueError as exc:
        raise ProtocolError("Primary output namespace must be inside this experiment") from exc
    if output_root == archive_root or archive_root in output_root.parents:
        raise ProtocolError("Primary output namespace may not be inside the invalid archive")

    config["_resolved"] = {
        "config_path": str(config_path.resolve()),
        "anchors_path": str(anchor_path),
        "output_root": str(output_root),
        "k0_dataset": str(resolve_relative(paths["k0_dataset"], config_path.parent)),
        "evaluation_dataset": str(
            resolve_relative(paths["evaluation_dataset"], config_path.parent)
        ),
        "legacy_original_reference": str(
            resolve_relative(paths["legacy_original_reference"], config_path.parent)
        ),
    }
    return config, {target: anchors[normalize(target)] for target in target_names}


def group_for_target(config: Mapping[str, Any], target: str) -> dict[str, Any]:
    target_norm = normalize(target)
    for group in config["groups"]:
        if target_norm in {normalize(value) for value in group["targets"]}:
            return dict(group)
    raise ProtocolError(f"No group contains target {target!r}")


def checkpoint_specs(
    config: Mapping[str, Any], anchors: Mapping[str, str]
) -> list[dict[str, Any]]:
    root = Path(config["_resolved"]["output_root"]) / "checkpoints"
    specs: list[dict[str, Any]] = []
    for group in config["groups"]:
        for target in group["targets"]:
            run_dir = root / group["id"] / "single" / slug(target)
            specs.append({
                "mode": "single",
                "group_id": group["id"],
                "targets": [target],
                "anchors": [anchors[target]],
                "retain_concepts": list(group["similar_non_targets"]),
                "checkpoint_path": str(run_dir / config["checkpointing"]["filename"]),
                "metadata_path": str(run_dir / config["checkpointing"]["metadata_filename"]),
            })
        run_dir = root / group["id"] / "joint"
        specs.append({
            "mode": "joint",
            "group_id": group["id"],
            "targets": list(group["targets"]),
            "anchors": [anchors[target] for target in group["targets"]],
            "retain_concepts": list(group["similar_non_targets"]),
            "checkpoint_path": str(run_dir / config["checkpointing"]["filename"]),
            "metadata_path": str(run_dir / config["checkpointing"]["metadata_filename"]),
        })
    if len(specs) != 15:
        raise ProtocolError(f"Expected 15 checkpoints, resolved {len(specs)}")
    return specs


def git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        check=False, capture_output=True, text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
        check=False, capture_output=True, text=True,
    )
    return {
        "git_hash": head.stdout.strip() if head.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_provenance() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(
            ["torch", "torchvision", "diffusers", "transformers", "safetensors", "pandas", "Pillow"]
        ),
        **git_state(),
    }


def source_hashes(extra: Iterable[Path] = ()) -> dict[str, str]:
    candidates = [
        DEFAULT_CONFIG,
        DEFAULT_ANCHORS,
        HERE / "PROVENANCE.md",
        Path(__file__),
        REPO_ROOT / "oce.py",
        REPO_ROOT / "trainscripts" / "object.sh",
        *extra,
    ]
    return {
        str(path.resolve().relative_to(PROJECT_ROOT.resolve())): sha256(path)
        for path in candidates
        if path.is_file()
    }


def expanded_prompts(concepts: Iterable[str], config: Mapping[str, Any]) -> list[str]:
    templates = config["oce"]["expansion_templates"]
    ordered_concepts = list(concepts)
    return [
        *ordered_concepts,
        *[
            template.format(concept=concept)
            for concept in ordered_concepts
            for template in templates[1:]
        ],
    ]
