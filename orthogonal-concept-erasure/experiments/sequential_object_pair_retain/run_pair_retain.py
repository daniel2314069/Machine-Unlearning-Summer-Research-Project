#!/usr/bin/env python
"""Matched two-concept sequential OCE experiment with previous-target retain.

This runner intentionally studies only image-level two-step behavior.  It uses
the repository OCE implementation without changing its objective.  In the one
modified condition, the first erased target is appended to the repository
object protocol's existing local retain list during the second edit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
OCE_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT = HERE / "outputs" / "sequential_oce_pair_retain_v1"
OCE_SOURCE = OCE_ROOT / "oce.py"
EVALUATOR_SOURCE = OCE_ROOT / "metrics" / "eval_clip_acc.py"
COMMON_SOURCE = HERE.parent / "sequential_object_persistence" / "run_sequential_oce.py"

CIFAR10 = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
OFFICIAL_ANCHORS = {
    "airplane": "sky", "automobile": "truck", "bird": "cat",
    "cat": "dog", "deer": "horse", "dog": "cat", "frog": "bird",
    "horse": "deer", "ship": "airplane", "truck": "ship",
}
FIXED_UNORDERED_PAIRS = [
    ("airplane", "cat"),
    ("automobile", "bird"),
    ("deer", "dog"),
    ("frog", "ship"),
    ("horse", "truck"),
]
SECOND_VARIANTS = ("baseline_second", "retain_previous_second")
EXPECTED_EDITED_MODULES = [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v.weight",
]


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
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def load_common() -> Any:
    specification = importlib.util.spec_from_file_location("oce_sequential_common", COMMON_SOURCE)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import shared runner: {COMMON_SOURCE}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_config(path: Path) -> dict[str, Any]:
    return read_json(path.resolve())


def ordered_pairs(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs = [tuple(pair) for pair in config["unordered_pairs"]]
    return [(a, b) for a, b in pairs for a, b in ((a, b), (b, a))]


def pair_name(first: str, second: str) -> str:
    configured = next(
        pair for pair in FIXED_UNORDERED_PAIRS if {first, second} == set(pair)
    )
    return "|".join(configured)


def order_name(first: str, second: str) -> str:
    return f"{first}->{second}"


def order_slug(first: str, second: str) -> str:
    return f"{first}_then_{second}"


def resolved_cg_path(config_path: Path, config: Mapping[str, Any]) -> Path:
    return (config_path.resolve().parent / str(config["cg_path"])).resolve()


def validate_config(
    config_path: Path, config: Mapping[str, Any], require_artifacts: bool
) -> dict[str, Any]:
    if config["target_anchor_mapping"] != OFFICIAL_ANCHORS:
        raise ValueError("Target-anchor mapping differs from the fixed repository mapping")
    configured_pairs = [tuple(pair) for pair in config["unordered_pairs"]]
    if configured_pairs != FIXED_UNORDERED_PAIRS:
        raise ValueError("Unordered pair schedule differs from the preregistered five pairs")
    orders = ordered_pairs(config)
    if len(orders) != 10 or len(set(orders)) != 10:
        raise ValueError("The pair schedule must produce ten unique ordered sequences")
    if sorted(first for first, _ in orders) != sorted(CIFAR10):
        raise ValueError("Every CIFAR-10 class must appear exactly once as the first target")
    for first, second in orders:
        if OFFICIAL_ANCHORS[first] == second or OFFICIAL_ANCHORS[second] == first:
            raise ValueError(f"Pair has a target-anchor collision: {first}, {second}")
    generation = config["generation"]
    if int(generation["images_per_class"]) != 200:
        raise ValueError("Paper-level object evaluation requires 200 images per class")
    if list(generation["qualitative_seeds"]) != [42, 43]:
        raise ValueError("Qualitative seeds are preregistered as 42 and 43")
    seeds = list(range(int(generation["seed_start"]), int(generation["seed_start"]) + 200))
    if seeds != list(range(42, 242)):
        raise ValueError("Formal deterministic seeds must be exactly 42..241")
    oce = config["oce"]
    expected_oce = {
        "erase_scale": 1000.0,
        "preserve_global_scale": 50.0,
        "preserve_concept_scale": 1.0,
        "lamb": 10.0,
        "expand_prompts": True,
        "always_preserve_current_anchor": True,
        "edit_dtype": "float32",
    }
    if dict(oce) != expected_oce:
        raise ValueError("OCE settings differ from the repository object protocol")
    if config["model_id"] != "CompVis/stable-diffusion-v1-4":
        raise ValueError("Base model must be Stable Diffusion v1.4")
    if config["clip_model_id"] != "openai/clip-vit-base-patch32":
        raise ValueError("Evaluator must use the repository CLIP ViT-B/32 model")
    if generation["scheduler"] != "PNDMScheduler":
        raise ValueError("Repository SD v1.4 generation must resolve to PNDMScheduler")
    if (
        int(generation["num_inference_steps"]) != 50
        or float(generation["guidance_scale"]) != 7.5
        or int(generation["height"]) != 512
        or int(generation["width"]) != 512
    ):
        raise ValueError("Generation settings differ from the repository object protocol")
    cg_path = resolved_cg_path(config_path, config)
    if require_artifacts and not cg_path.is_file():
        raise FileNotFoundError(f"Missing repository K0/Cg artifact: {cg_path}")
    if cg_path != (OCE_ROOT / "Cg.pt").resolve():
        raise ValueError("oce.py requires the repository-level Cg.pt")
    artifact_root = Path(str(config["artifact_root"]))
    if not artifact_root.is_absolute() or artifact_root.name != config["experiment_name"]:
        raise ValueError("artifact_root must be a dedicated absolute experiment directory")
    if artifact_root.parent != Path("/home/tslin/Documents/jupyter_data/anLi/tmp"):
        raise ValueError("Compressed artifacts must stay in the configured server anLi/tmp")
    formal_cells = 10 + 10 * 3 * 10
    formal_images = formal_cells * 200
    if formal_cells != 310 or formal_images != 62000:
        raise AssertionError("Formal count must be 310 cells / 62,000 generated images")
    return {
        "cifar_classes": 10,
        "unordered_pairs": 5,
        "ordered_pairs": 10,
        "checkpoints": 30,
        "formal_cells": formal_cells,
        "formal_images": formal_images,
        "qualitative_raw_images": 140,
        "qualitative_contact_sheets": 20,
    }


def git_capture(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=OCE_ROOT.parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout.strip()


def package_versions() -> dict[str, str]:
    names = ["torch", "diffusers", "transformers", "safetensors", "Pillow"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def resolve_snapshot(model_id: str, allow_downloads: bool) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(repo_id=model_id, local_files_only=not allow_downloads)
    ).resolve()
    revision = path.name if path.parent.name == "snapshots" else "unresolved"
    return {"model_id": model_id, "snapshot_path": str(path), "revision": revision}


def event(output_dir: Path, phase: str, message: str, **details: Any) -> None:
    row = {"timestamp": utc_now(), "phase": phase, "message": message, **details}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[{phase}] {message}{(' ' + suffix) if suffix else ''}", flush=True)


def completed_cell_markers(output_dir: Path) -> list[Path]:
    return list((output_dir / "raw" / "cells").glob("**/pair_experiment_complete.json"))


def update_progress(output_dir: Path, **details: Any) -> None:
    path = output_dir / "run_state.json"
    state = read_json(path) if path.is_file() else {}
    state.update(details)
    state["updated_at"] = utc_now()
    markers = completed_cell_markers(output_dir)
    state["completed_evaluators"] = len(markers)
    state["completed_generation_images"] = len(markers) * 200
    completed_orders = 0
    for first, second in ordered_pairs(state.get("config", load_config(DEFAULT_CONFIG))):
        slug = order_slug(first, second)
        expected = [
            output_dir / "raw" / "cells" / slug / variant / concept / "pair_experiment_complete.json"
            for variant in ("stage1", *SECOND_VARIANTS)
            for concept in CIFAR10
        ]
        completed_orders += int(all(path.is_file() for path in expected))
    state["completed_ordered_pairs"] = completed_orders
    write_json(path, state)


def make_protocol(
    config_path: Path, output_dir: Path, allow_downloads: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    counts = validate_config(config_path, config, require_artifacts=True)
    sources = {
        "config.json": sha256_file(config_path.resolve()),
        "oce.py": sha256_file(OCE_SOURCE),
        "object_evaluator.py": sha256_file(EVALUATOR_SOURCE),
        "shared_generation_evaluator_runner.py": sha256_file(COMMON_SOURCE),
        "runner": sha256_file(Path(__file__).resolve()),
        "Cg.pt": sha256_file(resolved_cg_path(config_path, config)),
    }
    model_snapshot = resolve_snapshot(config["model_id"], allow_downloads)
    clip_snapshot = resolve_snapshot(config["clip_model_id"], allow_downloads)
    fingerprint_input = {
        "config": config,
        "sources": sources,
        "model_snapshot": model_snapshot,
        "clip_snapshot": clip_snapshot,
    }
    commit = git_capture("rev-parse", "HEAD")
    dirty = bool(git_capture("status", "--porcelain"))
    return {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "protocol_fingerprint": stable_hash(fingerprint_input),
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "base_checkpoint": model_snapshot,
        "clip_evaluator_checkpoint": clip_snapshot,
        "paper_repo_configuration_source": {
            "paper": config["paper"],
            "repository_commit": commit,
            "repository_files": sources,
            "authority": "current repository implementation and object scripts",
        },
        "target_anchor_mapping": config["target_anchor_mapping"],
        "unordered_pairs": config["unordered_pairs"],
        "ordered_pairs": [list(pair) for pair in ordered_pairs(config)],
        "planned_counts": counts,
        "checkpoint_save_load": {
            "format": "safetensors",
            "selected_tensor_count": 16,
            "stage2_parent_rule": "both variants load the identical Stage-1 checkpoint hash",
        },
        "edited_modules": EXPECTED_EDITED_MODULES,
        "text_embedding_readout": (
            "DiffusionPipeline.encode_prompt output at tokenizer attention_mask.sum()-2; "
            "the final content token immediately before EOS"
        ),
        "K0": {
            "repository_name": "Cg.pt",
            "implementation": "compute_Cg.py second moment over non-padding COCO-30k token hidden states",
            "sha256": sources["Cg.pt"],
        },
        "official_local_neighbor_retain": (
            "the current target's official anchor only, matching trainscripts/object.sh; "
            "retain prompt expansion is not applied"
        ),
        "previous_target_retain_semantics": (
            "oce.Orthogonal_Erase forms each local retain target as W_current @ c; "
            "Stage-2 therefore preserves the post-first-edit representation"
        ),
        "seeds": list(range(42, 242)),
        "qualitative_seeds": config["generation"]["qualitative_seeds"],
        "package_versions": package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "local_files_only": not allow_downloads,
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "log_file": str((HERE / ".run" / "run.log").resolve()),
        "qualitative_tarball": str(
            Path(config["artifact_root"]) / config["storage"]["qualitative_tarball_name"]
        ),
        "config": config,
        "resolved_at": utc_now(),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    protocol = make_protocol(config_path, output_dir, args.allow_downloads)
    existing_path = output_dir / "run_manifest.json"
    if existing_path.is_file():
        existing = read_json(existing_path)
        if existing.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise RuntimeError(
                "Output directory contains another protocol; refusing to overwrite it"
            )
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(protocol["config"]["artifact_root"]).mkdir(parents=True, exist_ok=True)
    write_json(existing_path, protocol)
    write_csv(
        output_dir / "inputs" / "formal_seeds.csv",
        [{"seed_set_identifier": "cifar10_fixed_42_241", "sample_index": i, "seed": seed}
         for i, seed in enumerate(protocol["seeds"])],
    )
    pair_rows = []
    for pair_index, (a, b) in enumerate(FIXED_UNORDERED_PAIRS, start=1):
        for order_index, (first, second) in enumerate(((a, b), (b, a)), start=1):
            pair_rows.append({
                "pair_index": pair_index, "order_index_within_pair": order_index,
                "pair": pair_name(first, second), "order": order_name(first, second),
                "first_target": first, "first_anchor": OFFICIAL_ANCHORS[first],
                "second_target": second, "second_anchor": OFFICIAL_ANCHORS[second],
            })
    write_csv(output_dir / "inputs" / "pair_schedule.csv", pair_rows)
    update_progress(
        output_dir, status="ready", phase="preflight", pair="-", order="-",
        variant="-", stage="preflight", total_ordered_pairs=10,
        total_generation_images=62000, total_evaluators=310, config=protocol["config"],
    )
    event(output_dir, "preflight", "validated matched pair protocol", images=62000, cells=310)
    return protocol


def require_protocol(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    path = output_dir / "run_manifest.json"
    if not path.is_file():
        return preflight(args)
    current = read_json(path)
    expected = make_protocol(Path(args.config).resolve(), output_dir, args.allow_downloads)
    if current.get("protocol_fingerprint") != expected.get("protocol_fingerprint"):
        raise RuntimeError("Protocol/source fingerprint changed; choose a new output directory")
    return current


def runtime_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(protocol))
    value["config"]["model_id"] = protocol["base_checkpoint"]["snapshot_path"]
    return value


def checkpoint_path(output_dir: Path, first: str, second: str | None, variant: str) -> Path:
    if variant == "stage1":
        return output_dir / "checkpoints" / "stage1" / first / f"after_{first}.safetensors"
    if second is None:
        raise ValueError("Second target is required for Stage-2 checkpoints")
    return output_dir / "checkpoints" / "stage2" / order_slug(first, second) / f"{variant}.safetensors"


def checkpoint_manifest_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".manifest.json")


def validate_checkpoint(
    checkpoint: Path, protocol: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    from safetensors.torch import load_file

    manifest_path = checkpoint_manifest_path(checkpoint)
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = read_json(manifest_path)
    checks = {"protocol_fingerprint": protocol["protocol_fingerprint"], **dict(expected)}
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in checks.items() if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint manifest mismatch at {checkpoint}: {mismatches}")
    if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    state = load_file(str(checkpoint))
    if len(state) != 16:
        raise RuntimeError(f"Expected 16 edited tensors in {checkpoint}")
    if sorted(state) != sorted(EXPECTED_EDITED_MODULES):
        raise RuntimeError(f"Edited module names differ from repository authority: {checkpoint}")
    return state


def save_checkpoint_manifest(
    checkpoint: Path, protocol: Mapping[str, Any], **metadata: Any
) -> str:
    from safetensors.torch import load_file

    state = load_file(str(checkpoint))
    if len(state) != 16:
        raise RuntimeError("Saved checkpoint does not contain the selected tensor set")
    digest = sha256_file(checkpoint)
    write_json(
        checkpoint_manifest_path(checkpoint),
        {
            "status": "complete",
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": digest,
            "tensor_count": len(state),
            "created_at": utc_now(),
            **metadata,
        },
    )
    return digest


def build_stage1(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    import torch
    from safetensors.torch import load_file

    common = load_common()
    output_dir = Path(args.output_dir).resolve()
    run_protocol = runtime_protocol(protocol)
    config = protocol["config"]
    if str(OCE_ROOT) not in sys.path:
        sys.path.insert(0, str(OCE_ROOT))
    import oce as oce_impl

    pipe = common.load_pipeline(run_protocol, edit_only=True)
    base_state = common.selected_projection_state(pipe.unet)
    oce_impl.device = config["device"]
    oce_impl.torch_dtype = torch.float32
    previous_cwd = Path.cwd()
    try:
        os.chdir(OCE_ROOT)
        for first, _ in ordered_pairs(config):
            checkpoint = checkpoint_path(output_dir, first, None, "stage1")
            expected = {"stage": "stage1", "first_target": first, "parent_checkpoint_sha256": None}
            try:
                validate_checkpoint(checkpoint, protocol, expected)
                event(output_dir, "checkpoints", "reuse Stage-1 checkpoint", target=first)
                continue
            except FileNotFoundError:
                pass
            common.apply_projection_state(pipe.unet, base_state)
            anchor = OFFICIAL_ANCHORS[first]
            edits, guides = common.expand_object_pair(first, anchor)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            update_progress(
                output_dir, status="running", phase="checkpoints", pair=first,
                order=first, variant="stage1", stage="build_stage1",
            )
            oce_impl.Orthogonal_Erase(
                pipe, edits, guides, [anchor],
                float(config["oce"]["erase_scale"]),
                float(config["oce"]["preserve_global_scale"]),
                float(config["oce"]["preserve_concept_scale"]),
                float(config["oce"]["lamb"]), str(checkpoint.parent), checkpoint.stem,
            )
            state = load_file(str(checkpoint))
            if sorted(state) != sorted(EXPECTED_EDITED_MODULES):
                raise RuntimeError("Stage-1 edited module names differ from repository authority")
            common.apply_projection_state(pipe.unet, state)
            save_checkpoint_manifest(
                checkpoint, protocol, stage="stage1", first_target=first,
                first_anchor=anchor, target=first, anchor=anchor,
                parent_checkpoint_sha256=None, explicit_retain_concepts=[anchor],
                local_retain_reference="original/current W0 for Stage-1",
            )
    finally:
        os.chdir(previous_cwd)
        common.release_cuda(pipe)


def build_stage2(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    import torch
    from safetensors.torch import load_file

    common = load_common()
    output_dir = Path(args.output_dir).resolve()
    run_protocol = runtime_protocol(protocol)
    config = protocol["config"]
    if str(OCE_ROOT) not in sys.path:
        sys.path.insert(0, str(OCE_ROOT))
    import oce as oce_impl

    pipe = common.load_pipeline(run_protocol, edit_only=True)
    oce_impl.device = config["device"]
    oce_impl.torch_dtype = torch.float32
    previous_cwd = Path.cwd()
    try:
        os.chdir(OCE_ROOT)
        for first, second in ordered_pairs(config):
            stage1 = checkpoint_path(output_dir, first, None, "stage1")
            stage1_state = validate_checkpoint(
                stage1, protocol,
                {"stage": "stage1", "first_target": first, "parent_checkpoint_sha256": None},
            )
            stage1_sha = sha256_file(stage1)
            for variant in SECOND_VARIANTS:
                checkpoint = checkpoint_path(output_dir, first, second, variant)
                expected = {
                    "stage": "stage2", "variant": variant,
                    "first_target": first, "second_target": second,
                    "parent_checkpoint_sha256": stage1_sha,
                }
                try:
                    validate_checkpoint(checkpoint, protocol, expected)
                    event(output_dir, "checkpoints", "reuse Stage-2 checkpoint", order=order_name(first, second), variant=variant)
                    continue
                except FileNotFoundError:
                    pass
                common.apply_projection_state(pipe.unet, stage1_state)
                anchor = OFFICIAL_ANCHORS[second]
                preserve = [anchor]
                if variant == "retain_previous_second":
                    preserve.append(first)
                edits, guides = common.expand_object_pair(second, anchor)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                update_progress(
                    output_dir, status="running", phase="checkpoints",
                    pair=pair_name(first, second), order=order_name(first, second),
                    variant=variant, stage="build_stage2",
                )
                oce_impl.Orthogonal_Erase(
                    pipe, edits, guides, preserve,
                    float(config["oce"]["erase_scale"]),
                    float(config["oce"]["preserve_global_scale"]),
                    float(config["oce"]["preserve_concept_scale"]),
                    float(config["oce"]["lamb"]), str(checkpoint.parent), checkpoint.stem,
                )
                state = load_file(str(checkpoint))
                if sorted(state) != sorted(EXPECTED_EDITED_MODULES):
                    raise RuntimeError("Stage-2 edited module names differ from repository authority")
                common.apply_projection_state(pipe.unet, state)
                save_checkpoint_manifest(
                    checkpoint, protocol, stage="stage2", variant=variant,
                    first_target=first, first_anchor=OFFICIAL_ANCHORS[first],
                    second_target=second, second_anchor=anchor,
                    target=second, anchor=anchor,
                    parent_checkpoint=str(stage1.resolve()),
                    parent_checkpoint_sha256=stage1_sha,
                    explicit_retain_concepts=preserve,
                    previous_target_in_retain=variant == "retain_previous_second",
                    local_retain_reference=(
                        "post-first-edit W_A @ c_A from the loaded Stage-1 checkpoint"
                        if variant == "retain_previous_second"
                        else "repository baseline local retain only"
                    ),
                )
    finally:
        os.chdir(previous_cwd)
        common.release_cuda(pipe)


def formal_seeds(protocol: Mapping[str, Any]) -> list[int]:
    return [int(value) for value in protocol["seeds"]]


def qualitative_destination(
    output_dir: Path, first: str | None, second: str | None,
    variant: str, concept: str, seed: int,
) -> Path:
    if variant == "original":
        return output_dir / "qualitative" / "raw" / "original" / concept / f"seed_{seed}.png"
    if first is None or second is None:
        raise ValueError("Ordered pair required for edited qualitative images")
    return (
        output_dir / "qualitative" / "raw" / order_slug(first, second)
        / variant / concept / f"seed_{seed}.png"
    )


def cell_paths(
    output_dir: Path, first: str | None, second: str | None,
    variant: str, concept: str,
) -> tuple[Path, Path, str]:
    group = "original" if variant == "original" else order_slug(str(first), str(second))
    checkpoint_name = variant
    raw_dir = output_dir / "raw" / "cells" / group / checkpoint_name / concept
    image_dir = output_dir / "images" / group / checkpoint_name / concept
    return raw_dir, image_dir, group


def evaluate_one_cell(
    *, common: Any, pipe: Any, classifier: Any, protocol: Mapping[str, Any],
    output_dir: Path, first: str | None, second: str | None, variant: str,
    stage: int, concept: str, checkpoint_identifier: str,
    qualitative_concepts: set[str],
) -> dict[str, Any]:
    raw_dir, image_dir, group = cell_paths(output_dir, first, second, variant, concept)
    final_marker = raw_dir / "pair_experiment_complete.json"
    if final_marker.is_file():
        marker = read_json(final_marker)
        if marker.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise RuntimeError(f"Completed cell protocol mismatch: {raw_dir}")
        return read_json(raw_dir / "metrics.json")
    pair_value = "all" if variant == "original" else pair_name(str(first), str(second))
    order_value = "original" if variant == "original" else order_name(str(first), str(second))
    update_progress(
        output_dir, status="running", phase="evaluation", pair=pair_value,
        order=order_value, variant=variant, stage=stage, current_class=concept,
    )
    prompt = str(protocol["config"]["generation"]["prompt_template"]).format(concept=concept)
    metadata = {
        "pair": pair_value,
        "order": order_value,
        "variant": variant,
        "stage": stage,
        "seed_set_identifier": "cifar10_fixed_42_241",
        "checkpoint_identifier": checkpoint_identifier,
    }
    metrics = common.evaluate_cell(
        pipe=pipe, classifier=classifier, protocol=runtime_protocol(protocol),
        output_dir=output_dir, group=group, checkpoint=variant, concept=concept,
        prompt=prompt, class_labels=CIFAR10, expected_label=concept,
        seeds=formal_seeds(protocol), image_retention="keep", extra_metadata=metadata,
    )
    predictions = read_csv(raw_dir / "predictions.csv")
    if len(predictions) != 200 or int(metrics["n_images"]) != 200:
        raise RuntimeError(f"Evaluator count mismatch in {raw_dir}")
    observed_seeds = [int(row["seed"]) for row in predictions]
    if observed_seeds != formal_seeds(protocol) or len(set(observed_seeds)) != 200:
        raise RuntimeError(f"Seed mismatch in {raw_dir}")
    correct = sum(str(row["correct"]).casefold() == "true" for row in predictions)
    if abs(float(metrics["accuracy"]) - correct / 200) > 1e-12:
        raise RuntimeError(f"Saved accuracy cannot be reproduced from predictions: {raw_dir}")
    if concept in qualitative_concepts:
        index = read_json(raw_dir / "generation_manifest.json")["images"]
        by_seed = {int(row["seed"]): output_dir / row["image_path"] for row in index}
        for seed in protocol["qualitative_seeds"]:
            source = by_seed[int(seed)]
            destination = qualitative_destination(
                output_dir, first, second, variant, concept, int(seed)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                shutil.copy2(source, destination)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError(f"Qualitative copy failed: {destination}")
    index = read_json(raw_dir / "generation_manifest.json")["images"]
    common.delete_evaluated_images(output_dir, index, image_dir)
    write_json(
        final_marker,
        {
            "status": "complete",
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "n_images": 200,
            "prediction_sha256": sha256_file(raw_dir / "predictions.csv"),
            "metrics_sha256": sha256_file(raw_dir / "metrics.json"),
            "formal_images": "deleted-after-successful-evaluation-and-qualitative-copy",
            "completed_at": utc_now(),
        },
    )
    update_progress(output_dir)
    return metrics


def load_generation_stack(protocol: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    common = load_common()
    run_protocol = runtime_protocol(protocol)
    pipe = common.load_pipeline(run_protocol, edit_only=False)
    classifier = common.ClipClassifier(
        protocol["clip_evaluator_checkpoint"]["snapshot_path"],
        protocol["config"]["device"], True,
    )
    return common, pipe, classifier


def evaluate_original(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    output_dir = Path(args.output_dir).resolve()
    common, pipe, classifier = load_generation_stack(protocol)
    try:
        for concept in CIFAR10:
            evaluate_one_cell(
                common=common, pipe=pipe, classifier=classifier, protocol=protocol,
                output_dir=output_dir, first=None, second=None, variant="original",
                stage=0, concept=concept,
                checkpoint_identifier=protocol["base_checkpoint"]["revision"],
                qualitative_concepts=set(CIFAR10),
            )
    finally:
        common.release_cuda(classifier, pipe)


def evaluate_stage(
    args: argparse.Namespace, protocol: Mapping[str, Any], variants: Sequence[str]
) -> None:
    from safetensors.torch import load_file

    output_dir = Path(args.output_dir).resolve()
    common, pipe, classifier = load_generation_stack(protocol)
    try:
        for first, second in ordered_pairs(protocol["config"]):
            for variant in variants:
                stage = 1 if variant == "stage1" else 2
                checkpoint = checkpoint_path(
                    output_dir, first, None if variant == "stage1" else second, variant
                )
                expected = (
                    {"stage": "stage1", "first_target": first, "parent_checkpoint_sha256": None}
                    if variant == "stage1"
                    else {
                        "stage": "stage2", "variant": variant,
                        "first_target": first, "second_target": second,
                        "parent_checkpoint_sha256": sha256_file(
                            checkpoint_path(output_dir, first, None, "stage1")
                        ),
                    }
                )
                state = validate_checkpoint(checkpoint, protocol, expected)
                common.apply_projection_state(pipe.unet, state)
                identifier = sha256_file(checkpoint)
                for concept in CIFAR10:
                    evaluate_one_cell(
                        common=common, pipe=pipe, classifier=classifier,
                        protocol=protocol, output_dir=output_dir,
                        first=first, second=second, variant=variant, stage=stage,
                        concept=concept, checkpoint_identifier=identifier,
                        qualitative_concepts={first, second},
                    )
    finally:
        common.release_cuda(classifier, pipe)


def metric_path(
    output_dir: Path, first: str | None, second: str | None,
    variant: str, concept: str,
) -> Path:
    return cell_paths(output_dir, first, second, variant, concept)[0] / "metrics.json"


def collect_metrics(output_dir: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = [metric_path(output_dir, None, None, "original", concept) for concept in CIFAR10]
    for first, second in ordered_pairs(protocol["config"]):
        for variant in ("stage1", *SECOND_VARIANTS):
            paths.extend(metric_path(output_dir, first, second, variant, concept) for concept in CIFAR10)
    if len(paths) != 310 or any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise RuntimeError(f"Expected 310 evaluator cells; missing {missing[:5]}")
    rows = [read_json(path) for path in paths]
    if sum(int(row["n_images"]) for row in rows) != 62000:
        raise RuntimeError("Evaluator metrics do not total 62,000 images")
    return rows


def lookup(
    rows: Sequence[Mapping[str, Any]], first: str | None, second: str | None,
    variant: str, concept: str,
) -> Mapping[str, Any]:
    pair_value = "all" if variant == "original" else pair_name(str(first), str(second))
    order_value = "original" if variant == "original" else order_name(str(first), str(second))
    matches = [
        row for row in rows
        if row.get("pair") == pair_value and row.get("order") == order_value
        and row.get("variant") == variant and row.get("concept") == concept
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one metric row for {order_value}/{variant}/{concept}")
    return matches[0]


def harmonic_object(acc_e: float, acc_s: float) -> float:
    erase_success = 1.0 - acc_e
    if erase_success <= 0.0 or acc_s <= 0.0:
        return 0.0
    return 2.0 / (1.0 / erase_success + 1.0 / acc_s)


def build_contact_sheets(output_dir: Path, protocol: Mapping[str, Any]) -> None:
    from PIL import Image, ImageDraw

    labels = ["Original", "After first erase", "After normal second", "After retain previous"]
    variants = ["original", "stage1", "baseline_second", "retain_previous_second"]
    sheet_rows: list[dict[str, Any]] = []
    for first, second in ordered_pairs(protocol["config"]):
        for concept in (first, second):
            tiles: list[list[Image.Image]] = []
            for seed in protocol["qualitative_seeds"]:
                row: list[Image.Image] = []
                for variant in variants:
                    source = qualitative_destination(
                        output_dir,
                        None if variant == "original" else first,
                        None if variant == "original" else second,
                        variant, concept, int(seed),
                    )
                    if not source.is_file():
                        raise FileNotFoundError(f"Missing preregistered qualitative image: {source}")
                    with Image.open(source) as image:
                        row.append(image.convert("RGB").copy())
                tiles.append(row)
            width, height = tiles[0][0].size
            header = 44
            canvas = Image.new("RGB", (4 * width, header + 2 * height), "white")
            draw = ImageDraw.Draw(canvas)
            for column, label in enumerate(labels):
                draw.text((column * width + 8, 8), label, fill="black")
            for row_index, row in enumerate(tiles):
                for column, tile in enumerate(row):
                    canvas.paste(tile, (column * width, header + row_index * height))
                draw.text((8, header + row_index * height + 8), f"seed {protocol['qualitative_seeds'][row_index]}", fill="white", stroke_width=2, stroke_fill="black")
            destination = (
                output_dir / "qualitative" / "contact_sheets" / order_slug(first, second)
                / f"{concept}.png"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(destination)
            sheet_rows.append({
                "pair": pair_name(first, second), "order": order_name(first, second),
                "concept": concept, "seeds": protocol["qualitative_seeds"],
                "contact_sheet": str(destination.relative_to(output_dir)),
            })
    write_json(
        output_dir / "qualitative" / "qualitative_manifest.json",
        {
            "status": "complete",
            "selection_timing": "fixed before generation",
            "seeds": protocol["qualitative_seeds"],
            "raw_image_count": 140,
            "contact_sheet_count": 20,
            "conditions": variants,
            "sheets": sheet_rows,
        },
    )


def package_qualitative(output_dir: Path, protocol: Mapping[str, Any]) -> Path:
    artifact_root = Path(protocol["config"]["artifact_root"])
    destination = artifact_root / protocol["config"]["storage"]["qualitative_tarball_name"]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    artifact_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(output_dir / "qualitative" / "raw", arcname="raw")
        archive.add(output_dir / "qualitative" / "contact_sheets", arcname="contact_sheets")
        archive.add(
            output_dir / "qualitative" / "qualitative_manifest.json",
            arcname="qualitative_manifest.json",
        )
    temporary.replace(destination)
    return destination


def aggregate(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    output_dir = Path(args.output_dir).resolve()
    rows = collect_metrics(output_dir, protocol)
    per_class_rows: list[dict[str, Any]] = []
    for row in rows:
        per_class_rows.append({
            "pair": row["pair"], "order": row["order"],
            "variant": row["variant"], "stage": row["stage"],
            "class": row["concept"], "number_of_generated_samples": row["n_images"],
            "accuracy": row["accuracy"],
            "seed_set_identifier": row["seed_set_identifier"],
            "checkpoint_identifier": row["checkpoint_identifier"],
        })
    write_csv(output_dir / "per_class_results.csv", per_class_rows)

    summary_rows: list[dict[str, Any]] = []
    stage1_rows: list[dict[str, Any]] = []
    for first, second in ordered_pairs(protocol["config"]):
        first_after_first = float(lookup(rows, first, second, "stage1", first)["accuracy"])
        first_after_baseline = float(lookup(rows, first, second, "baseline_second", first)["accuracy"])
        first_after_retain = float(lookup(rows, first, second, "retain_previous_second", first)["accuracy"])
        second_baseline = float(lookup(rows, first, second, "baseline_second", second)["accuracy"])
        second_retain = float(lookup(rows, first, second, "retain_previous_second", second)["accuracy"])
        remaining = [concept for concept in CIFAR10 if concept not in {first, second}]
        remaining_baseline = sum(
            float(lookup(rows, first, second, "baseline_second", concept)["accuracy"])
            for concept in remaining
        ) / 8
        remaining_retain = sum(
            float(lookup(rows, first, second, "retain_previous_second", concept)["accuracy"])
            for concept in remaining
        ) / 8
        stage1_others = [concept for concept in CIFAR10 if concept != first]
        stage1_acc_s = sum(
            float(lookup(rows, first, second, "stage1", concept)["accuracy"])
            for concept in stage1_others
        ) / 9
        stage1_h = harmonic_object(first_after_first, stage1_acc_s)
        summary_rows.append({
            "pair": pair_name(first, second), "order": order_name(first, second),
            "first_target": first, "second_target": second,
            "first_target_accuracy_after_first_edit": first_after_first,
            "first_target_accuracy_after_second_baseline": first_after_baseline,
            "first_target_accuracy_after_retain_previous_second": first_after_retain,
            "first_target_raw_delta_after_second_baseline": first_after_baseline - first_after_first,
            "first_target_raw_delta_after_retain_previous_second": first_after_retain - first_after_first,
            "second_target_final_accuracy_baseline": second_baseline,
            "second_target_final_accuracy_retain_previous": second_retain,
            "second_target_raw_difference_retain_minus_baseline": second_retain - second_baseline,
            "remaining_8_mean_accuracy_baseline": remaining_baseline,
            "remaining_8_mean_accuracy_retain_previous": remaining_retain,
            "remaining_8_mean_raw_difference_retain_minus_baseline": remaining_retain - remaining_baseline,
            "stage1_Acc_e": first_after_first, "stage1_Acc_s": stage1_acc_s,
            "stage1_H_o": stage1_h,
        })
        stage1_rows.append({
            "pair": pair_name(first, second), "order": order_name(first, second),
            "target": first, "Acc_e": first_after_first,
            "Acc_s": stage1_acc_s, "H_o": stage1_h,
        })
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "stage1_paper_metrics.csv", stage1_rows)

    prediction_rows: list[dict[str, str]] = []
    for path in sorted((output_dir / "raw" / "cells").glob("**/predictions.csv")):
        prediction_rows.extend(read_csv(path))
    if len(prediction_rows) != 62000:
        raise RuntimeError(f"Expected 62,000 saved predictions, found {len(prediction_rows)}")
    write_csv(output_dir / "raw" / "all_predictions.csv", prediction_rows)
    build_contact_sheets(output_dir, protocol)
    tarball = package_qualitative(output_dir, protocol)
    final_validation(args, protocol, tarball)


def final_validation(
    args: argparse.Namespace, protocol: Mapping[str, Any], tarball: Path | None = None
) -> None:
    output_dir = Path(args.output_dir).resolve()
    rows = collect_metrics(output_dir, protocol)
    markers = completed_cell_markers(output_dir)
    if len(markers) != 310:
        raise RuntimeError(f"Expected 310 completed evaluator markers, found {len(markers)}")
    prediction_total = 0
    for marker in markers:
        raw_dir = marker.parent
        predictions = read_csv(raw_dir / "predictions.csv")
        metrics = read_json(raw_dir / "metrics.json")
        marker_data = read_json(marker)
        if len(predictions) != 200:
            raise RuntimeError(f"Prediction count mismatch: {raw_dir}")
        seeds = [int(row["seed"]) for row in predictions]
        if seeds != formal_seeds(protocol) or len(set(seeds)) != 200:
            raise RuntimeError(f"Final seed audit mismatch: {raw_dir}")
        if metrics.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise RuntimeError(f"Metric protocol fingerprint mismatch: {raw_dir}")
        if marker_data.get("prediction_sha256") != sha256_file(raw_dir / "predictions.csv"):
            raise RuntimeError(f"Prediction hash mismatch: {raw_dir}")
        if marker_data.get("metrics_sha256") != sha256_file(raw_dir / "metrics.json"):
            raise RuntimeError(f"Metric hash mismatch: {raw_dir}")
        correct = sum(str(row["correct"]).casefold() == "true" for row in predictions)
        if abs(correct / 200 - float(metrics["accuracy"])) > 1e-12:
            raise RuntimeError(f"Accuracy recomputation mismatch: {raw_dir}")
        prediction_total += len(predictions)
    remaining_formal_images = list((output_dir / "images").glob("**/*.png"))
    if remaining_formal_images:
        raise RuntimeError(f"Formal cleanup incomplete: {len(remaining_formal_images)} PNGs remain")
    stage2_manifests = [
        read_json(path) for path in (output_dir / "checkpoints" / "stage2").glob("**/*.manifest.json")
    ]
    stage1_manifests = [
        read_json(path) for path in (output_dir / "checkpoints" / "stage1").glob("**/*.manifest.json")
    ]
    if len(stage1_manifests) != 10:
        raise RuntimeError("Expected ten unique Stage-1 checkpoint manifests")
    if len(stage2_manifests) != 20:
        raise RuntimeError("Expected twenty Stage-2 checkpoint manifests")
    for first, second in ordered_pairs(protocol["config"]):
        relevant = [
            row for row in stage2_manifests
            if row["first_target"] == first and row["second_target"] == second
        ]
        if len(relevant) != 2 or len({row["parent_checkpoint_sha256"] for row in relevant}) != 1:
            raise RuntimeError(f"Stage-2 variants do not share one Stage-1 parent: {first}->{second}")
        by_variant = {row["variant"]: row for row in relevant}
        expected_baseline = [OFFICIAL_ANCHORS[second]]
        expected_modified = [OFFICIAL_ANCHORS[second], first]
        if by_variant["baseline_second"]["explicit_retain_concepts"] != expected_baseline:
            raise RuntimeError(f"Baseline retain list mismatch: {first}->{second}")
        if by_variant["retain_previous_second"]["explicit_retain_concepts"] != expected_modified:
            raise RuntimeError(f"Modified retain list mismatch: {first}->{second}")
    if tarball is None:
        tarball = Path(protocol["qualitative_tarball"])
    if not tarball.is_file() or tarball.stat().st_size == 0:
        raise RuntimeError(f"Qualitative tarball missing: {tarball}")
    validation = {
        "status": "complete",
        "validated_at": utc_now(),
        "classes": len(CIFAR10), "unordered_pairs": 5, "ordered_pairs": 10,
        "evaluator_cells": len(rows), "prediction_rows": prediction_total,
        "generated_images": 62000,
        "same_seed_list_all_conditions": True,
        "stage2_shared_parent_check": "passed",
        "saved_predictions_recompute_metrics": "passed",
        "formal_image_cleanup": "complete",
        "qualitative_raw_images": len(list((output_dir / "qualitative" / "raw").glob("**/*.png"))),
        "qualitative_contact_sheets": len(list((output_dir / "qualitative" / "contact_sheets").glob("**/*.png"))),
        "qualitative_tarball": str(tarball.resolve()),
        "qualitative_tarball_sha256": sha256_file(tarball),
    }
    if validation["qualitative_raw_images"] != 140 or validation["qualitative_contact_sheets"] != 20:
        raise RuntimeError("Qualitative artifact count mismatch")
    write_json(output_dir / "final_validation.json", validation)
    manifest = read_json(output_dir / "run_manifest.json")
    manifest.update({
        "status": "complete", "completed_at": utc_now(),
        "generated_image_counts": {"formal": 62000, "qualitative_extra": 0},
        "cleanup_status": "all formal PNGs deleted after validated evaluation",
        "results": {
            "per_class_results": str((output_dir / "per_class_results.csv").resolve()),
            "summary": str((output_dir / "summary.csv").resolve()),
            "stage1_paper_metrics": str((output_dir / "stage1_paper_metrics.csv").resolve()),
            "predictions": str((output_dir / "raw" / "all_predictions.csv").resolve()),
            "validation": str((output_dir / "final_validation.json").resolve()),
        },
        "qualitative_tarball": str(tarball.resolve()),
    })
    write_json(output_dir / "run_manifest.json", manifest)
    update_progress(
        output_dir, status="complete", phase="complete", pair="-", order="-",
        variant="-", stage="complete", current_class="-",
    )
    event(output_dir, "validation", "formal run passed final validation", images=62000, predictions=prediction_total)


def print_plan(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    counts = validate_config(Path(args.config), config, require_artifacts=False)
    print(json.dumps({
        "config": str(Path(args.config).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "ordered_pairs": [order_name(*pair) for pair in ordered_pairs(config)],
        "conditions": ["original", "stage1", *SECOND_VARIANTS],
        "counts": counts,
        "qualitative_tarball": str(
            Path(config["artifact_root"]) / config["storage"]["qualitative_tarball_name"]
        ),
        "launches_models": False,
    }, indent=2, ensure_ascii=False))


def run_all(args: argparse.Namespace) -> None:
    protocol = preflight(args)
    output_dir = Path(args.output_dir).resolve()
    try:
        update_progress(output_dir, status="running", phase="evaluation", stage="original")
        evaluate_original(args, protocol)
        update_progress(output_dir, status="running", phase="checkpoints", stage="build_stage1")
        build_stage1(args, protocol)
        update_progress(output_dir, status="running", phase="evaluation", stage="stage1")
        evaluate_stage(args, protocol, ["stage1"])
        update_progress(output_dir, status="running", phase="checkpoints", stage="build_stage2")
        build_stage2(args, protocol)
        update_progress(output_dir, status="running", phase="evaluation", stage="stage2")
        evaluate_stage(args, protocol, list(SECOND_VARIANTS))
        update_progress(output_dir, status="running", phase="aggregation", stage="aggregate")
        aggregate(args, protocol)
    except BaseException as error:
        update_progress(output_dir, status="failed", error=repr(error))
        raise


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-downloads", action="store_true",
        help="Allow missing model snapshots to be downloaded during preflight",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    functions = {
        "plan": print_plan,
        "preflight": preflight,
        "run": run_all,
        "validate": lambda args: final_validation(args, require_protocol(args)),
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
