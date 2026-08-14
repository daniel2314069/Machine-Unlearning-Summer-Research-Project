#!/usr/bin/env python
"""Run the two 200-image sequential OCE object-persistence follow-ups.

This file is experiment orchestration only. It calls the repository's existing
single-concept OCE implementation without changing its objective or algorithm.
"""

from __future__ import annotations

import argparse
import csv
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
SOURCE_EXPERIMENT = HERE.parent / "sequential_object_persistence"
SOURCE_RUNNER = SOURCE_EXPERIMENT / "run_sequential_oce.py"
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT = HERE / "outputs" / "sequential_oce_object_followup_v1"
OCE_SOURCE = OCE_ROOT / "oce.py"

if str(SOURCE_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_EXPERIMENT))
import run_sequential_oce as common  # noqa: E402


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
CLEAN_PAIRS = [
    ("dog", "cat"),
    ("bird", "cat"),
    ("airplane", "sky"),
    ("automobile", "truck"),
    ("deer", "horse"),
]
CONDITIONS = ("retain_once", "retain_always")
EXPECTED_NEW_IMAGES = 7000


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
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


def clean_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    return [dict(row) for row in config["clean_chain"]]


def targets(config: Mapping[str, Any]) -> list[str]:
    return [row["target"] for row in pair_rows(config)]


def resolved_path(config_path: Path, value: str) -> Path:
    return (config_path.resolve().parent / value).resolve()


def source_output_path(config_path: Path, config: Mapping[str, Any]) -> Path:
    return resolved_path(config_path, str(config["source_output_dir"]))


def cg_path(config_path: Path, config: Mapping[str, Any]) -> Path:
    return resolved_path(config_path, str(config["cg_path"]))


def full_seeds(config: Mapping[str, Any]) -> list[int]:
    generation = config["generation"]
    start = int(generation["seed_start"])
    return list(range(start, start + int(generation["images_per_cell"])))


def source_seeds(config: Mapping[str, Any]) -> list[int]:
    generation = config["generation"]
    start = int(generation["seed_start"])
    return list(range(start, start + int(generation["source_images_per_cell"])))


def supplement_seeds(config: Mapping[str, Any]) -> list[int]:
    generation = config["generation"]
    start = int(generation["supplement_seed_start"])
    return list(range(start, start + int(generation["supplement_images_per_cell"])))


def build_cell_manifest(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_seeds = full_seeds(config)
    old_seeds = source_seeds(config)
    new_seeds = supplement_seeds(config)
    for pair in pair_rows(config):
        target = pair["target"]
        rows.append(
            {
                "experiment": "experiment1",
                "part": "direct_single_from_W0",
                "condition": "direct_single",
                "checkpoint": f"W0_erase_{target}",
                "target": target,
                "anchor": pair["anchor"],
                "existing_images": 0,
                "new_images": 200,
                "final_predictions": 200,
                "seed_start": all_seeds[0],
                "seed_end": all_seeds[-1],
                "seed_sha256": stable_hash(all_seeds),
            }
        )
    for condition in CONDITIONS:
        for step, pair in enumerate(pair_rows(config), start=1):
            rows.append(
                {
                    "experiment": "experiment1",
                    "part": "sequential_own_step_supplement",
                    "condition": condition,
                    "checkpoint": f"W{step:02d}",
                    "target": pair["target"],
                    "anchor": pair["anchor"],
                    "existing_images": 100,
                    "new_images": 100,
                    "final_predictions": 200,
                    "existing_seed_start": old_seeds[0],
                    "existing_seed_end": old_seeds[-1],
                    "new_seed_start": new_seeds[0],
                    "new_seed_end": new_seeds[-1],
                    "combined_seed_sha256": stable_hash(old_seeds + new_seeds),
                }
            )
    erased: list[str] = []
    for step, pair in enumerate(clean_rows(config), start=1):
        erased.append(pair["target"])
        for target in erased:
            target_anchor = dict(OFFICIAL_PAIRS)[target]
            rows.append(
                {
                    "experiment": "experiment2",
                    "part": "clean_five_step_persistence",
                    "condition": "clean_chain",
                    "checkpoint": f"W{step}",
                    "target": target,
                    "anchor": target_anchor,
                    "existing_images": 0,
                    "new_images": 200,
                    "final_predictions": 200,
                    "seed_start": all_seeds[0],
                    "seed_end": all_seeds[-1],
                    "seed_sha256": stable_hash(all_seeds),
                }
            )
    return rows


def planned_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    direct = [row for row in rows if row["part"] == "direct_single_from_W0"]
    supplement = [
        row for row in rows if row["part"] == "sequential_own_step_supplement"
    ]
    clean = [row for row in rows if row["part"] == "clean_five_step_persistence"]
    counts = {
        "experiment1": {
            "direct_single_cells": len(direct),
            "direct_single_new_images": sum(int(row["new_images"]) for row in direct),
            "sequential_supplement_cells": len(supplement),
            "sequential_supplement_new_images": sum(
                int(row["new_images"]) for row in supplement
            ),
            "new_images": sum(
                int(row["new_images"]) for row in direct + supplement
            ),
        },
        "experiment2": {
            "cells": len(clean),
            "new_images": sum(int(row["new_images"]) for row in clean),
        },
    }
    counts["total"] = {
        "new_formal_cells": len(rows),
        "new_formal_images": sum(int(row["new_images"]) for row in rows),
        "final_prediction_rows_in_followup_cells": sum(
            int(row["final_predictions"]) for row in rows
        ),
    }
    return counts


def validate_config(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    pairs = [(row["target"], row["anchor"]) for row in pair_rows(config)]
    clean = [(row["target"], row["anchor"]) for row in clean_rows(config)]
    if pairs != OFFICIAL_PAIRS:
        raise ValueError("Target-anchor mapping must exactly match OCE Table 7")
    if clean != CLEAN_PAIRS:
        raise ValueError("Clean-chain order/mapping differs from the fixed five-step design")
    clean_targets = {target for target, _ in clean}
    clean_anchors = {anchor for _, anchor in clean}
    if clean_targets & clean_anchors:
        raise ValueError("A clean-chain anchor is also a clean-chain target")
    generation = config["generation"]
    required = {
        "images_per_cell": 200,
        "source_images_per_cell": 100,
        "supplement_images_per_cell": 100,
        "seed_start": 42,
        "supplement_seed_start": 142,
    }
    mismatches = {
        key: {"expected": value, "observed": generation.get(key)}
        for key, value in required.items()
        if int(generation.get(key, -1)) != value
    }
    if mismatches:
        raise ValueError(f"Hard image-count/seed requirements failed: {mismatches}")
    if source_seeds(config) != list(range(42, 142)):
        raise AssertionError("Source seeds must be 42..141")
    if supplement_seeds(config) != list(range(142, 242)):
        raise AssertionError("Supplement seeds must be 142..241")
    if len(set(full_seeds(config))) != 200:
        raise AssertionError("Formal cell seeds must contain exactly 200 unique values")
    if config["storage"]["image_retention"] not in {"keep", "delete-after-eval"}:
        raise ValueError("Unsupported image retention policy")
    manifest = build_cell_manifest(config)
    counts = planned_counts(manifest)
    if counts != {
        "experiment1": {
            "direct_single_cells": 10,
            "direct_single_new_images": 2000,
            "sequential_supplement_cells": 20,
            "sequential_supplement_new_images": 2000,
            "new_images": 4000,
        },
        "experiment2": {"cells": 15, "new_images": 3000},
        "total": {
            "new_formal_cells": 45,
            "new_formal_images": 7000,
            "final_prediction_rows_in_followup_cells": 9000,
        },
    }:
        raise AssertionError(f"Formal plan is not the required 7,000-image plan: {counts}")
    return counts


def source_cell_dir(
    source_output: Path, condition: str, step: int, target: str
) -> Path:
    return source_output / "raw" / "cells" / condition / f"W{step:02d}" / target


def source_checkpoint(
    source_output: Path, condition: str, step: int, target: str
) -> Path:
    return (
        source_output
        / "checkpoints"
        / condition
        / f"W{step:02d}_{target}.safetensors"
    )


def validate_source_artifacts(
    config_path: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    source_output = source_output_path(config_path, config)
    protocol_path = source_output / "resolved_protocol.json"
    summary_path = source_output / "summary.json"
    if not protocol_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Prior completed result is missing under {source_output}"
        )
    source_protocol = read_json(protocol_path)
    source_summary = read_json(summary_path)
    if source_summary.get("status") != "complete":
        raise RuntimeError("Prior sequential result is not marked complete")
    source_config = source_protocol["config"]
    comparisons = {
        "model_id": (config["model_id"], source_config["model_id"]),
        "clip_model_id": (config["clip_model_id"], source_config["clip_model_id"]),
        "targets": (pair_rows(config), source_config["targets"]),
        "oce": (config["oce"], source_config["oce"]),
    }
    generation_keys = (
        "seed_start",
        "num_inference_steps",
        "guidance_scale",
        "height",
        "width",
        "dtype",
        "batch_size",
        "prompt_template",
        "scheduler",
    )
    for key in generation_keys:
        comparisons[f"generation.{key}"] = (
            config["generation"][key],
            source_config["generation"][key],
        )
    comparisons["evaluation.class_text_template"] = (
        config["evaluation"]["cifar_class_text_template"],
        source_config["evaluation"]["cifar_class_text_template"],
    )
    comparisons["evaluation.batch_size"] = (
        config["evaluation"]["batch_size"],
        source_config["evaluation"]["batch_size"],
    )
    mismatches = {
        key: {"expected": expected, "observed": observed}
        for key, (expected, observed) in comparisons.items()
        if expected != observed
    }
    if mismatches:
        raise RuntimeError(f"Follow-up differs from prior protocol: {mismatches}")
    expected_old_seeds = source_seeds(config)
    source_fingerprint = source_protocol["protocol_fingerprint"]
    audited_cells = 0
    for condition in CONDITIONS:
        parent_checkpoint_sha: str | None = None
        for step, pair in enumerate(pair_rows(config), start=1):
            target = pair["target"]
            cell = source_cell_dir(source_output, condition, step, target)
            required = [
                cell / "generation_manifest.json",
                cell / "predictions.csv",
                cell / "metrics.json",
                cell / "complete.json",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Prior own-step artifacts missing: {missing}")
            rows = read_csv(cell / "predictions.csv")
            seeds = [int(row["seed"]) for row in rows]
            metrics = read_json(cell / "metrics.json")
            generation_manifest = read_json(cell / "generation_manifest.json")
            complete = read_json(cell / "complete.json")
            manifest_seeds = [
                int(row["seed"]) for row in generation_manifest.get("images", [])
            ]
            correct = sum(bool_value(row["correct"]) for row in rows)
            if len(rows) != 100 or seeds != expected_old_seeds or len(set(seeds)) != 100:
                raise RuntimeError(f"Prior seeds/count invalid at {cell}")
            if (
                metrics.get("protocol_fingerprint") != source_fingerprint
                or int(metrics.get("n_images", -1)) != 100
                or int(metrics.get("correct", -1)) != correct
                or abs(float(metrics.get("accuracy", -1)) - correct / 100) > 1e-12
                or metrics.get("concept") != target
                or len(generation_manifest.get("images", [])) != 100
                or manifest_seeds != expected_old_seeds
                or complete.get("status") != "complete"
                or complete.get("protocol_fingerprint") != source_fingerprint
            ):
                raise RuntimeError(f"Prior evaluator artifacts invalid at {cell}")
            checkpoint = source_checkpoint(source_output, condition, step, target)
            checkpoint_manifest = checkpoint.with_suffix(".manifest.json")
            if not checkpoint.is_file() or not checkpoint_manifest.is_file():
                raise FileNotFoundError(
                    f"Prior server checkpoint required for supplementation: {checkpoint}"
                )
            checkpoint_meta = read_json(checkpoint_manifest)
            observed_checkpoint_sha = sha256_file(checkpoint)
            if (
                checkpoint_meta.get("protocol_fingerprint") != source_fingerprint
                or checkpoint_meta.get("condition") != condition
                or int(checkpoint_meta.get("step", -1)) != step
                or checkpoint_meta.get("target") != target
                or checkpoint_meta.get("checkpoint_sha256")
                != observed_checkpoint_sha
                or checkpoint_meta.get("parent_checkpoint_sha256")
                != parent_checkpoint_sha
            ):
                raise RuntimeError(f"Prior checkpoint manifest invalid: {checkpoint}")
            parent_checkpoint_sha = observed_checkpoint_sha
            audited_cells += 1
    return {
        "source_output_dir": str(source_output),
        "source_protocol_fingerprint": source_fingerprint,
        "source_protocol_sha256": sha256_file(protocol_path),
        "source_summary_sha256": sha256_file(summary_path),
        "audited_own_step_cells": audited_cells,
        "selected_x_in_source_only": source_summary.get("selected_x"),
    }


def make_protocol(
    config_path: Path,
    output_dir: Path,
    allow_downloads: bool,
    retention_override: str | None,
    require_artifacts: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    counts = validate_config(config_path, config)
    source_audit = (
        validate_source_artifacts(config_path, config)
        if require_artifacts
        else {
            "source_output_dir": str(source_output_path(config_path, config)),
            "source_protocol_fingerprint": "validated-during-preflight",
        }
    )
    if require_artifacts and not cg_path(config_path, config).is_file():
        raise FileNotFoundError(f"Missing repository Cg.pt: {cg_path(config_path, config)}")
    retention = retention_override or config["storage"]["image_retention"]
    sources = {
        "config.json": sha256_file(config_path),
        "followup_runner": sha256_file(Path(__file__).resolve()),
        "source_runner": sha256_file(SOURCE_RUNNER),
        "oce.py": sha256_file(OCE_SOURCE),
    }
    if require_artifacts:
        sources["Cg.pt"] = sha256_file(cg_path(config_path, config))
    fingerprint_input = {
        "experiment": config["experiment_name"],
        "config": config,
        "sources": sources,
        "source_protocol_fingerprint": source_audit["source_protocol_fingerprint"],
        "local_files_only": not allow_downloads,
        "image_retention": retention,
    }
    return {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "base_model": config["model_id"],
        "target_anchor_mapping": pair_rows(config),
        "clean_chain": clean_rows(config),
        "planned_counts": counts,
        "source_audit": source_audit,
        "source_hashes": sources,
        "protocol_fingerprint": stable_hash(fingerprint_input),
        "local_files_only": not allow_downloads,
        "effective_image_retention": retention,
        "resolved_at": utc_now(),
        "software": {"python": sys.version, "platform": platform.platform()},
        "config": config,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    protocol = make_protocol(
        config_path,
        output_dir,
        args.allow_downloads,
        args.image_retention,
        require_artifacts=True,
    )
    rows = build_cell_manifest(protocol["config"])
    counts = planned_counts(rows)
    if int(counts["total"]["new_formal_images"]) != EXPECTED_NEW_IMAGES:
        raise RuntimeError(
            f"HARD STOP: planned new images are not {EXPECTED_NEW_IMAGES}: {counts}"
        )
    existing_path = output_dir / "resolved_protocol.json"
    if existing_path.is_file():
        existing = read_json(existing_path)
        if existing.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise RuntimeError(
                "Output contains a different protocol; use a new output directory"
            )
    update_state(output_dir, "preflight", "running")
    write_json(output_dir / "resolved_protocol.json", protocol)
    write_csv(output_dir / "inputs" / "cell_manifest.csv", rows)
    write_json(
        output_dir / "inputs" / "planned_generation.json",
        {
            "status": "approved-for-generation",
            "hard_required_new_images": EXPECTED_NEW_IMAGES,
            "counts": counts,
            "cells": rows,
            "full_seeds": full_seeds(protocol["config"]),
            "source_seeds": source_seeds(protocol["config"]),
            "supplement_seeds": supplement_seeds(protocol["config"]),
        },
    )
    write_csv(
        output_dir / "inputs" / "formal_seeds_200.csv",
        [
            {"sample_index": index, "seed": seed}
            for index, seed in enumerate(full_seeds(protocol["config"]))
        ],
    )
    print(
        json.dumps(
            {
                "generation_hard_gate": "PASSED",
                "counts": counts,
                "cell_manifest": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    update_state(
        output_dir,
        "preflight",
        "complete",
        protocol_fingerprint=protocol["protocol_fingerprint"],
        planned_new_images=EXPECTED_NEW_IMAGES,
    )
    event(
        output_dir,
        "preflight",
        "hard generation-count gate passed",
        cells=len(rows),
        new_images=EXPECTED_NEW_IMAGES,
    )
    return protocol


def require_protocol(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    expected = make_protocol(
        Path(args.config).resolve(),
        output_dir,
        args.allow_downloads,
        args.image_retention,
        require_artifacts=True,
    )
    path = output_dir / "resolved_protocol.json"
    if not path.is_file():
        return preflight(args)
    current = read_json(path)
    if current.get("protocol_fingerprint") != expected["protocol_fingerprint"]:
        raise RuntimeError("Protocol fingerprint changed; refusing mixed output")
    planned = read_json(output_dir / "inputs" / "planned_generation.json")
    if int(planned["counts"]["total"]["new_formal_images"]) != EXPECTED_NEW_IMAGES:
        raise RuntimeError("HARD STOP: saved generation plan is not exactly 7,000 images")
    return current


def checkpoint_path(output_dir: Path, family: str, step: int, target: str) -> Path:
    name = f"W{step:02d}_{target}.safetensors" if step else f"W0_{target}.safetensors"
    return output_dir / "checkpoints" / family / name


def checkpoint_manifest_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".manifest.json")


def validate_new_checkpoint(
    checkpoint: Path,
    protocol: Mapping[str, Any],
    family: str,
    step: int,
    target: str,
    parent_sha256: str | None,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    manifest_path = checkpoint_manifest_path(checkpoint)
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = read_json(manifest_path)
    checks = {
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "family": family,
        "step": step,
        "target": target,
        "parent_checkpoint_sha256": parent_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in checks.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint manifest mismatch at {checkpoint}: {mismatches}")
    observed_sha = sha256_file(checkpoint)
    if manifest.get("checkpoint_sha256") != observed_sha:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    state = load_file(str(checkpoint))
    if len(state) != 16:
        raise RuntimeError(f"Unexpected checkpoint tensor count: {checkpoint}")
    return state


def save_checkpoint_manifest(
    checkpoint: Path,
    protocol: Mapping[str, Any],
    family: str,
    step: int,
    pair: Mapping[str, str],
    parent_sha256: str | None,
    tensor_count: int,
) -> str:
    checkpoint_sha = sha256_file(checkpoint)
    write_json(
        checkpoint_manifest_path(checkpoint),
        {
            "status": "complete",
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "family": family,
            "step": step,
            "target": pair["target"],
            "anchor": pair["anchor"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "parent_checkpoint_sha256": parent_sha256,
            "explicit_retain_concepts": [pair["anchor"]],
            "tensor_count": tensor_count,
            "created_at": utc_now(),
        },
    )
    return checkpoint_sha


def run_single_edit(
    pipe: Any,
    oce_impl: Any,
    protocol: Mapping[str, Any],
    pair: Mapping[str, str],
    checkpoint: Path,
) -> None:
    config = protocol["config"]
    if bool(config["oce"]["expand_prompts"]):
        edits, guides = common.expand_object_pair(pair["target"], pair["anchor"])
    else:
        edits, guides = [pair["target"]], [pair["anchor"]]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    oce_impl.Orthogonal_Erase(
        pipe,
        edits,
        guides,
        [pair["anchor"]],
        float(config["oce"]["erase_scale"]),
        float(config["oce"]["preserve_global_scale"]),
        float(config["oce"]["preserve_concept_scale"]),
        float(config["oce"]["lamb"]),
        str(checkpoint.parent),
        checkpoint.stem,
    )


def build_checkpoints(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import load_file

    protocol = require_protocol(args)
    output_dir = Path(args.output_dir).resolve()
    config = protocol["config"]
    update_state(output_dir, "checkpoints", "running")
    if str(OCE_ROOT) not in sys.path:
        sys.path.insert(0, str(OCE_ROOT))
    import oce as oce_impl

    pipe = common.load_pipeline(protocol, edit_only=True)
    base_state = common.selected_projection_state(pipe.unet)
    oce_impl.device = config["device"]
    oce_impl.torch_dtype = torch.float32
    previous_cwd = Path.cwd()
    try:
        os.chdir(OCE_ROOT)
        for pair in pair_rows(config):
            common.apply_projection_state(pipe.unet, base_state)
            checkpoint = checkpoint_path(
                output_dir, "direct_single_from_W0", 0, pair["target"]
            )
            try:
                validate_new_checkpoint(
                    checkpoint,
                    protocol,
                    "direct_single_from_W0",
                    0,
                    pair["target"],
                    None,
                )
                event(output_dir, "checkpoints", "reuse direct checkpoint", target=pair["target"])
                continue
            except FileNotFoundError:
                pass
            event(output_dir, "checkpoints", "run direct single edit", target=pair["target"])
            run_single_edit(pipe, oce_impl, protocol, pair, checkpoint)
            state = load_file(str(checkpoint))
            common.apply_projection_state(pipe.unet, state)
            save_checkpoint_manifest(
                checkpoint,
                protocol,
                "direct_single_from_W0",
                0,
                pair,
                None,
                len(state),
            )

        common.apply_projection_state(pipe.unet, base_state)
        parent_sha: str | None = None
        for step, pair in enumerate(clean_rows(config), start=1):
            checkpoint = checkpoint_path(
                output_dir, "clean_five_step_chain", step, pair["target"]
            )
            try:
                state = validate_new_checkpoint(
                    checkpoint,
                    protocol,
                    "clean_five_step_chain",
                    step,
                    pair["target"],
                    parent_sha,
                )
                common.apply_projection_state(pipe.unet, state)
                parent_sha = sha256_file(checkpoint)
                event(output_dir, "checkpoints", "reuse clean-chain checkpoint", step=step, target=pair["target"])
                continue
            except FileNotFoundError:
                pass
            event(output_dir, "checkpoints", "run clean-chain single edit", step=step, target=pair["target"])
            run_single_edit(pipe, oce_impl, protocol, pair, checkpoint)
            state = load_file(str(checkpoint))
            common.apply_projection_state(pipe.unet, state)
            parent_sha = save_checkpoint_manifest(
                checkpoint,
                protocol,
                "clean_five_step_chain",
                step,
                pair,
                parent_sha,
                len(state),
            )
    finally:
        os.chdir(previous_cwd)
        common.release_cuda(pipe)
    update_state(output_dir, "checkpoints", "complete", checkpoint_count=15)


def source_checkpoint_state(
    source_output: Path,
    source_fingerprint: str,
    condition: str,
    step: int,
    target: str,
) -> tuple[dict[str, Any], str]:
    from safetensors.torch import load_file

    checkpoint = source_checkpoint(source_output, condition, step, target)
    manifest = read_json(checkpoint_manifest_path(checkpoint))
    observed_sha = sha256_file(checkpoint)
    checks = {
        "protocol_fingerprint": source_fingerprint,
        "condition": condition,
        "step": step,
        "target": target,
        "checkpoint_sha256": observed_sha,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in checks.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Source checkpoint mismatch at {checkpoint}: {mismatches}")
    state = load_file(str(checkpoint))
    if len(state) != 16:
        raise RuntimeError(f"Unexpected source checkpoint tensor count: {checkpoint}")
    return state, observed_sha


def supplement_paths(
    output_dir: Path, condition: str, checkpoint: str, target: str
) -> tuple[Path, Path]:
    raw_dir = (
        output_dir
        / "raw"
        / "cells"
        / "experiment1_sequential_own_step"
        / condition
        / checkpoint
        / target
    )
    image_dir = (
        output_dir
        / "images"
        / "experiment1_sequential_own_step"
        / condition
        / checkpoint
        / target
    )
    return raw_dir, image_dir


def generate_supplement_images(
    pipe: Any,
    output_dir: Path,
    image_dir: Path,
    prompt: str,
    seeds: Sequence[int],
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    index = [
        {
            "sample_index": 100 + offset,
            "seed": seed,
            "image_path": str(
                (image_dir / f"index_{100 + offset:03d}_seed_{seed}.png").relative_to(output_dir)
            ),
            "sample_origin": "new_followup_supplement",
        }
        for offset, seed in enumerate(seeds)
    ]
    missing = [row for row in index if not (output_dir / row["image_path"]).is_file()]
    image_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(generation["batch_size"])
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
            raise RuntimeError("Supplement generation batch size mismatch")
        for row, image in zip(batch, images):
            image.save(output_dir / row["image_path"])
    absent = [row for row in index if not (output_dir / row["image_path"]).is_file()]
    if absent:
        raise RuntimeError(f"Supplement generation incomplete: {len(absent)} images")
    return index


def bool_value(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def validate_final_cell(
    raw_dir: Path,
    expected_seeds: Sequence[int],
    protocol_fingerprint: str,
) -> dict[str, Any]:
    required = [
        raw_dir / "generation_manifest.json",
        raw_dir / "predictions.csv",
        raw_dir / "metrics.json",
        raw_dir / "complete.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final cell artifacts missing: {missing}")
    rows = read_csv(raw_dir / "predictions.csv")
    metrics = read_json(raw_dir / "metrics.json")
    complete = read_json(raw_dir / "complete.json")
    manifest = read_json(raw_dir / "generation_manifest.json")
    seeds = [int(row["seed"]) for row in rows]
    manifest_seeds = [int(row["seed"]) for row in manifest.get("images", [])]
    correct = sum(bool_value(row["correct"]) for row in rows)
    if (
        seeds != list(expected_seeds)
        or manifest_seeds != list(expected_seeds)
        or len(rows) != 200
        or len(set(seeds)) != 200
        or int(metrics.get("n_images", -1)) != 200
        or int(metrics.get("correct", -1)) != correct
        or abs(float(metrics.get("accuracy", -1)) - correct / 200) > 1e-12
        or metrics.get("protocol_fingerprint") != protocol_fingerprint
        or complete.get("status") != "complete"
        or complete.get("protocol_fingerprint") != protocol_fingerprint
        or int(complete.get("n_images", -1)) != 200
        or manifest.get("protocol_fingerprint") != protocol_fingerprint
        or len(manifest.get("images", [])) != 200
    ):
        raise RuntimeError(f"Final 200-image cell validation failed: {raw_dir}")
    return metrics


def evaluate_supplement_cell(
    *,
    pipe: Any,
    classifier: Any,
    protocol: Mapping[str, Any],
    output_dir: Path,
    source_output: Path,
    condition: str,
    step: int,
    target: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    checkpoint_name = f"W{step:02d}"
    raw_dir, image_dir = supplement_paths(
        output_dir, condition, checkpoint_name, target
    )
    complete_path = raw_dir / "complete.json"
    all_expected_seeds = full_seeds(protocol["config"])
    if complete_path.is_file():
        return validate_final_cell(
            raw_dir, all_expected_seeds, protocol["protocol_fingerprint"]
        )
    config = protocol["config"]
    old_cell = source_cell_dir(source_output, condition, step, target)
    old_rows = read_csv(old_cell / "predictions.csv")
    old_manifest = read_json(old_cell / "generation_manifest.json")
    expected_old = source_seeds(config)
    if [int(row["seed"]) for row in old_rows] != expected_old:
        raise RuntimeError(f"Existing source predictions changed: {old_cell}")
    prompt = str(config["generation"]["prompt_template"]).format(concept=target)
    new_seed_values = supplement_seeds(config)
    new_index = generate_supplement_images(
        pipe,
        output_dir,
        image_dir,
        prompt,
        new_seed_values,
        config["generation"],
    )
    class_labels = targets(config)
    class_texts = [
        str(config["evaluation"]["cifar_class_text_template"]).format(concept=label)
        for label in class_labels
    ]
    probabilities = classifier.classify(
        [output_dir / row["image_path"] for row in new_index],
        class_texts,
        int(config["evaluation"]["batch_size"]),
    )
    if len(probabilities) != 100:
        raise RuntimeError("Supplement evaluator must return exactly 100 predictions")
    expected_index = class_labels.index(target)
    new_rows: list[dict[str, Any]] = []
    for image_row, values in zip(new_index, probabilities):
        prediction_index = max(range(len(values)), key=values.__getitem__)
        predicted = class_labels[prediction_index]
        row: dict[str, Any] = {
            "group": condition,
            "checkpoint": checkpoint_name,
            "concept": target,
            "prompt": prompt,
            "sample_index": image_row["sample_index"],
            "seed": image_row["seed"],
            "image_path": image_row["image_path"],
            "expected_label": target,
            "predicted_label": predicted,
            "correct": predicted == target,
            "expected_probability": values[expected_index],
            "image_retention": protocol["effective_image_retention"],
            "sample_origin": "new_followup_supplement",
            "artifact_root": str(output_dir),
        }
        for label, value in zip(class_labels, values):
            row[f"prob_{common.safe_label(label)}"] = value
        new_rows.append(row)
    normalized_old: list[dict[str, Any]] = []
    for row in old_rows:
        copied: dict[str, Any] = dict(row)
        copied["sample_origin"] = "existing_v1_100"
        copied["artifact_root"] = str(source_output)
        normalized_old.append(copied)
    combined = normalized_old + new_rows
    combined.sort(key=lambda row: int(row["sample_index"]))
    seeds = [int(row["seed"]) for row in combined]
    if seeds != all_expected_seeds or len(set(seeds)) != 200:
        raise RuntimeError("Combined own-step cell does not have seeds 42..241 exactly once")
    correct = sum(bool_value(row["correct"]) for row in combined)
    expected_probabilities = [float(row["expected_probability"]) for row in combined]
    images = [
        {
            **dict(row),
            "sample_origin": "existing_v1_100",
            "artifact_root": str(source_output),
        }
        for row in old_manifest["images"]
    ] + [dict(row) | {"artifact_root": str(output_dir)} for row in new_index]
    write_json(
        raw_dir / "generation_manifest.json",
        {
            "status": "complete",
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "source_protocol_fingerprint": protocol["source_audit"]["source_protocol_fingerprint"],
            "condition": condition,
            "checkpoint": checkpoint_name,
            "concept": target,
            "prompt": prompt,
            "existing_prediction_count": 100,
            "new_generation_count": 100,
            "final_prediction_count": 200,
            "source_predictions": str((old_cell / "predictions.csv").resolve()),
            "images": images,
        },
    )
    write_csv(raw_dir / "supplement_predictions.csv", new_rows)
    write_csv(raw_dir / "predictions.csv", combined)
    metrics = {
        "status": "complete",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "source_protocol_fingerprint": protocol["source_audit"]["source_protocol_fingerprint"],
        "group": condition,
        "checkpoint": checkpoint_name,
        "step": step,
        "concept": target,
        "role": "sequential_own_step_200",
        "prompt": prompt,
        "expected_label": target,
        "class_labels": class_labels,
        "class_texts": class_texts,
        "evaluator": config["clip_model_id"],
        "n_images": 200,
        "existing_predictions": 100,
        "new_predictions": 100,
        "correct": correct,
        "accuracy": correct / 200,
        "mean_expected_probability": sum(expected_probabilities) / 200,
        "model_checkpoint_sha256": checkpoint_sha256,
        "image_retention": protocol["effective_image_retention"],
    }
    write_json(raw_dir / "metrics.json", metrics)
    reread = read_csv(raw_dir / "predictions.csv")
    if len(reread) != 200 or len({int(row["seed"]) for row in reread}) != 200:
        raise RuntimeError("Refusing image cleanup: combined evaluator audit failed")
    if protocol["effective_image_retention"] == "delete-after-eval":
        common.delete_evaluated_images(output_dir, new_index, image_dir)
        image_status = "new-100-deleted-after-successful-evaluation"
    else:
        image_status = "new-100-retained"
    write_json(
        complete_path,
        {
            "status": "complete",
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "completed_at": utc_now(),
            "n_images": 200,
            "unique_seeds": 200,
            "image_status": image_status,
        },
    )
    return validate_final_cell(
        raw_dir, all_expected_seeds, protocol["protocol_fingerprint"]
    )


def evaluate_formal(args: argparse.Namespace) -> None:
    protocol = require_protocol(args)
    output_dir = Path(args.output_dir).resolve()
    config = protocol["config"]
    source_output = source_output_path(Path(args.config).resolve(), config)
    source_fingerprint = protocol["source_audit"]["source_protocol_fingerprint"]
    update_state(output_dir, "evaluation", "running", planned_new_images=7000)
    pipe = common.load_pipeline(protocol, edit_only=False)
    classifier = common.ClipClassifier(
        config["clip_model_id"], config["device"], bool(protocol["local_files_only"])
    )
    retention = str(protocol["effective_image_retention"])
    prompt_template = str(config["generation"]["prompt_template"])
    labels = targets(config)
    seeds_200 = full_seeds(config)

    for pair in pair_rows(config):
        checkpoint = checkpoint_path(
            output_dir, "direct_single_from_W0", 0, pair["target"]
        )
        state = validate_new_checkpoint(
            checkpoint,
            protocol,
            "direct_single_from_W0",
            0,
            pair["target"],
            None,
        )
        common.apply_projection_state(pipe.unet, state)
        metrics = common.evaluate_cell(
            pipe=pipe,
            classifier=classifier,
            protocol=protocol,
            output_dir=output_dir,
            group="experiment1_direct_single",
            checkpoint=f"W0_erase_{pair['target']}",
            concept=pair["target"],
            prompt=prompt_template.format(concept=pair["target"]),
            class_labels=labels,
            expected_label=pair["target"],
            seeds=seeds_200,
            image_retention=retention,
            extra_metadata={
                "experiment": "experiment1",
                "role": "direct_single_from_W0",
                "anchor": pair["anchor"],
                "model_checkpoint": str(checkpoint.resolve()),
                "model_checkpoint_sha256": sha256_file(checkpoint),
            },
        )
        event(output_dir, "evaluation", "completed direct cell", target=pair["target"], n=metrics["n_images"], accuracy=f"{metrics['accuracy']:.4f}")

    for condition in CONDITIONS:
        for step, pair in enumerate(pair_rows(config), start=1):
            state, checkpoint_sha = source_checkpoint_state(
                source_output,
                source_fingerprint,
                condition,
                step,
                pair["target"],
            )
            common.apply_projection_state(pipe.unet, state)
            metrics = evaluate_supplement_cell(
                pipe=pipe,
                classifier=classifier,
                protocol=protocol,
                output_dir=output_dir,
                source_output=source_output,
                condition=condition,
                step=step,
                target=pair["target"],
                checkpoint_sha256=checkpoint_sha,
            )
            event(output_dir, "evaluation", "completed own-step 200 cell", condition=condition, step=step, target=pair["target"], accuracy=f"{metrics['accuracy']:.4f}")

    parent_sha: str | None = None
    erased: list[str] = []
    for step, pair in enumerate(clean_rows(config), start=1):
        checkpoint = checkpoint_path(
            output_dir, "clean_five_step_chain", step, pair["target"]
        )
        state = validate_new_checkpoint(
            checkpoint,
            protocol,
            "clean_five_step_chain",
            step,
            pair["target"],
            parent_sha,
        )
        common.apply_projection_state(pipe.unet, state)
        parent_sha = sha256_file(checkpoint)
        erased.append(pair["target"])
        for target in erased:
            metrics = common.evaluate_cell(
                pipe=pipe,
                classifier=classifier,
                protocol=protocol,
                output_dir=output_dir,
                group="experiment2_clean_chain",
                checkpoint=f"W{step}",
                concept=target,
                prompt=prompt_template.format(concept=target),
                class_labels=labels,
                expected_label=target,
                seeds=seeds_200,
                image_retention=retention,
                extra_metadata={
                    "experiment": "experiment2",
                    "role": "previous_erasure_persistence",
                    "step": step,
                    "anchor": dict(OFFICIAL_PAIRS)[target],
                    "model_checkpoint": str(checkpoint.resolve()),
                    "model_checkpoint_sha256": parent_sha,
                },
            )
            event(output_dir, "evaluation", "completed clean-chain cell", checkpoint=f"W{step}", target=target, accuracy=f"{metrics['accuracy']:.4f}")
    common.release_cuda(classifier, pipe)
    audit_completed_cells(output_dir, protocol)
    update_state(
        output_dir,
        "evaluation",
        "complete",
        new_formal_images=7000,
        final_predictions=9000,
        cells=45,
        image_retention=retention,
    )


def direct_raw_dir(output_dir: Path, target: str) -> Path:
    return common.cell_paths(
        output_dir,
        "experiment1_direct_single",
        f"W0_erase_{target}",
        target,
    )[0]


def clean_raw_dir(output_dir: Path, step: int, target: str) -> Path:
    return common.cell_paths(
        output_dir, "experiment2_clean_chain", f"W{step}", target
    )[0]


def audit_completed_cells(
    output_dir: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    config = protocol["config"]
    expected = full_seeds(config)
    new_generation_total = 0
    final_prediction_total = 0
    cells = 0
    for pair in pair_rows(config):
        validate_final_cell(
            direct_raw_dir(output_dir, pair["target"]),
            expected,
            protocol["protocol_fingerprint"],
        )
        new_generation_total += 200
        final_prediction_total += 200
        cells += 1
    for condition in CONDITIONS:
        for step, pair in enumerate(pair_rows(config), start=1):
            raw_dir = supplement_paths(
                output_dir, condition, f"W{step:02d}", pair["target"]
            )[0]
            validate_final_cell(
                raw_dir, expected, protocol["protocol_fingerprint"]
            )
            new_generation_total += 100
            final_prediction_total += 200
            cells += 1
    erased: list[str] = []
    for step, pair in enumerate(clean_rows(config), start=1):
        erased.append(pair["target"])
        for target in erased:
            validate_final_cell(
                clean_raw_dir(output_dir, step, target),
                expected,
                protocol["protocol_fingerprint"],
            )
            new_generation_total += 200
            final_prediction_total += 200
            cells += 1
    if (cells, new_generation_total, final_prediction_total) != (45, 7000, 9000):
        raise RuntimeError(
            "Completed-cell audit did not resolve to 45 cells / 7,000 new / 9,000 final rows"
        )
    return {
        "cells": cells,
        "new_generation_images": new_generation_total,
        "final_prediction_rows": final_prediction_total,
        "unique_seeds_per_final_cell": 200,
    }


def metrics(path: Path) -> dict[str, Any]:
    return read_json(path / "metrics.json")


def combine_predictions(
    paths: Sequence[Path], output_path: Path, expected_count: int
) -> None:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_csv(path / "predictions.csv"))
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} combined predictions, found {len(rows)}"
        )
    write_csv(output_path, rows)


def make_plots(
    output_dir: Path,
    persistence_rows: Sequence[Mapping[str, Any]],
    per_target_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    concepts = [target for target, _ in CLEAN_PAIRS]
    matrix = np.full((5, 5), np.nan, dtype=float)
    for row_index, row in enumerate(persistence_rows):
        for column_index, concept in enumerate(concepts):
            if row.get(concept, "") != "":
                matrix[row_index, column_index] = float(row[concept])
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
    axis.set_title("Clean five-step previous-erasure persistence")
    axis.set_xlabel("Erased target")
    axis.set_ylabel("Checkpoint")
    axis.set_xticks(range(5), concepts)
    axis.set_yticks(range(5), [f"W{step}" for step in range(1, 6)])
    for row_index in range(5):
        for column_index in range(row_index + 1):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label="10-class CLIP accuracy", pad=0.03)
    figure.tight_layout()
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / "experiment2_previous_erasure_heatmap.png", dpi=180)
    figure.savefig(plot_dir / "experiment2_previous_erasure_heatmap.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    styles = {
        "dog": {"linewidth": 3.0, "marker": "o", "zorder": 5},
        "bird": {"linewidth": 3.0, "marker": "s", "zorder": 5},
    }
    for row in per_target_rows:
        target = str(row["target"])
        own_step = int(row["erased_at_step"])
        steps = list(range(own_step, 6))
        values = [float(row[f"accuracy_W{step}"]) for step in steps]
        kwargs = styles.get(
            target,
            {"linewidth": 1.4, "marker": ".", "alpha": 0.55, "zorder": 2},
        )
        axis.plot(steps, values, label=target, **kwargs)
    axis.set_title("Clean-chain target trajectories (200 images/cell)")
    axis.set_xlabel("Sequential checkpoint")
    axis.set_ylabel("10-class CLIP target accuracy")
    axis.set_xticks(range(1, 6), [f"W{step}" for step in range(1, 6)])
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(plot_dir / "experiment2_trajectories.png", dpi=180)
    figure.savefig(plot_dir / "experiment2_trajectories.pdf")
    plt.close(figure)


def aggregate(args: argparse.Namespace) -> None:
    protocol = require_protocol(args)
    output_dir = Path(args.output_dir).resolve()
    config = protocol["config"]
    update_state(output_dir, "aggregation", "running")
    audit = audit_completed_cells(output_dir, protocol)
    table_dir = output_dir / "tables"

    comparison_rows: list[dict[str, Any]] = []
    direct_prediction_dirs: list[Path] = []
    supplement_prediction_dirs: list[Path] = []
    for step, pair in enumerate(pair_rows(config), start=1):
        target = pair["target"]
        direct_dir = direct_raw_dir(output_dir, target)
        once_dir = supplement_paths(
            output_dir, "retain_once", f"W{step:02d}", target
        )[0]
        always_dir = supplement_paths(
            output_dir, "retain_always", f"W{step:02d}", target
        )[0]
        direct = float(metrics(direct_dir)["accuracy"])
        once = float(metrics(once_dir)["accuracy"])
        always = float(metrics(always_dir)["accuracy"])
        comparison_rows.append(
            {
                "target": target,
                "anchor": pair["anchor"],
                "sequential_own_step": f"W{step:02d}",
                "direct_single_from_W0_accuracy_200": direct,
                "sequential_retain_once_own_step_accuracy_200": once,
                "sequential_retain_always_own_step_accuracy_200": always,
                "retain_once_sequential_minus_direct": once - direct,
                "retain_always_sequential_minus_direct": always - direct,
            }
        )
        direct_prediction_dirs.append(direct_dir)
        supplement_prediction_dirs.extend([once_dir, always_dir])
    write_csv(table_dir / "experiment1_comparison.csv", comparison_rows)

    persistence_rows: list[dict[str, Any]] = []
    clean_prediction_dirs: list[Path] = []
    clean_targets = [target for target, _ in CLEAN_PAIRS]
    for step in range(1, 6):
        row: dict[str, Any] = {"checkpoint": f"W{step}", "step": step}
        for target_index, target in enumerate(clean_targets, start=1):
            if target_index <= step:
                cell = clean_raw_dir(output_dir, step, target)
                row[target] = metrics(cell)["accuracy"]
                clean_prediction_dirs.append(cell)
            else:
                row[target] = ""
        persistence_rows.append(row)
    write_csv(table_dir / "experiment2_persistence.csv", persistence_rows)

    per_target_rows: list[dict[str, Any]] = []
    for own_step, target in enumerate(clean_targets, start=1):
        trajectory = {
            step: float(metrics(clean_raw_dir(output_dir, step, target))["accuracy"])
            for step in range(own_step, 6)
        }
        maximum_step, maximum_accuracy = max(
            trajectory.items(), key=lambda item: item[1]
        )
        row: dict[str, Any] = {
            "target": target,
            "anchor": dict(OFFICIAL_PAIRS)[target],
            "erased_at_step": own_step,
            "accuracy_immediately_after_own_erasure": trajectory[own_step],
        }
        for step in range(1, 6):
            row[f"accuracy_W{step}"] = trajectory.get(step, "")
        row.update(
            {
                "maximum_post_erasure_accuracy": maximum_accuracy,
                "maximum_at_checkpoint": f"W{maximum_step}",
                "raw_increase_from_own_step": maximum_accuracy - trajectory[own_step],
            }
        )
        per_target_rows.append(row)
    write_csv(table_dir / "experiment2_per_target_summary.csv", per_target_rows)

    combine_predictions(
        direct_prediction_dirs,
        output_dir / "raw" / "experiment1_direct_per_image_predictions.csv",
        2000,
    )
    combine_predictions(
        supplement_prediction_dirs,
        output_dir
        / "raw"
        / "experiment1_sequential_own_step_per_image_predictions_200.csv",
        4000,
    )
    combine_predictions(
        clean_prediction_dirs,
        output_dir / "raw" / "experiment2_per_image_predictions.csv",
        3000,
    )
    make_plots(output_dir, persistence_rows, per_target_rows)

    dog = next(row for row in per_target_rows if row["target"] == "dog")
    bird = next(row for row in per_target_rows if row["target"] == "bird")
    both_higher = [
        row["target"]
        for row in comparison_rows
        if float(row["retain_once_sequential_minus_direct"]) > 0
        and float(row["retain_always_sequential_minus_direct"]) > 0
    ]
    mixed_or_not_higher = [
        row["target"] for row in comparison_rows if row["target"] not in both_higher
    ]
    dog_up = float(dog["raw_increase_from_own_step"]) > 0
    bird_up = float(bird["raw_increase_from_own_step"]) > 0
    result = {
        "status": "complete",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "formal_counts": audit,
        "experiment1": {
            "targets_with_both_sequential_own_step_accuracies_above_direct": both_higher,
            "other_targets": mixed_or_not_higher,
            "comparison_rows": comparison_rows,
        },
        "experiment2": {
            "dog": dog,
            "bird": bird,
            "dog_later_maximum_above_own_step": dog_up,
            "bird_later_maximum_above_own_step": bird_up,
            "per_target_rows": per_target_rows,
        },
        "completed_at": utc_now(),
    }
    write_json(output_dir / "summary.json", result)

    comparison_lines = [
        f"- `{row['target']}`: direct={float(row['direct_single_from_W0_accuracy_200']):.3f}, "
        f"Retain Once own-step={float(row['sequential_retain_once_own_step_accuracy_200']):.3f} "
        f"(difference {float(row['retain_once_sequential_minus_direct']):+.3f}), "
        f"Retain Always own-step={float(row['sequential_retain_always_own_step_accuracy_200']):.3f} "
        f"(difference {float(row['retain_always_sequential_minus_direct']):+.3f})"
        for row in comparison_rows
    ]
    if both_higher:
        direction_answer = (
            "Both sequential own-step values were above direct-single for: "
            + ", ".join(both_higher)
            + "."
        )
    else:
        direction_answer = "No target had both sequential own-step values above direct-single."
    if mixed_or_not_higher:
        consistency_answer = (
            "The direction was not uniform across all targets; the remaining targets were: "
            + ", ".join(mixed_or_not_higher)
            + "."
        )
    else:
        consistency_answer = "All ten targets had both sequential own-step values above direct-single."
    dog_line = (
        f"Dog: W1={float(dog['accuracy_immediately_after_own_erasure']):.3f}, "
        f"maximum through W5={float(dog['maximum_post_erasure_accuracy']):.3f} "
        f"at {dog['maximum_at_checkpoint']} "
        f"(raw increase {float(dog['raw_increase_from_own_step']):+.3f})."
    )
    bird_line = (
        f"Bird: W2={float(bird['accuracy_immediately_after_own_erasure']):.3f}, "
        f"maximum through W5={float(bird['maximum_post_erasure_accuracy']):.3f} "
        f"at {bird['maximum_at_checkpoint']} "
        f"(raw increase {float(bird['raw_increase_from_own_step']):+.3f})."
    )
    if dog_up or bird_up:
        overlap_answer = (
            "At least one focal trajectory rose in the clean chain, where none of the "
            "five anchors is a later erase target. Therefore the original upward pattern "
            "cannot be explained only by the anchor itself later being erased. This design "
            "does not exclude every other anchor interaction."
        )
    else:
        overlap_answer = (
            "Neither focal trajectory rose in this clean-chain repeat. The new result does "
            "not reproduce the original resurgence signal and is compatible with a strong "
            "target-anchor dependency in the original order."
        )
    summary_lines = [
        "# Sequential OCE object follow-up summary",
        "",
        "- New formal generations: **7,000** exactly.",
        "- Every final formal cell: **200 unique predictions**.",
        "- Evaluator: unchanged CIFAR-10 10-class CLIP ViT-B/32 protocol.",
        "- No artificial significance threshold is applied; all comparisons below are raw accuracies and absolute differences.",
        "",
        "## Experiment 1 — direct single from W0 vs sequential own-step",
        "",
        *comparison_lines,
        "",
        direction_answer,
        consistency_answer,
        "The table should be read as empirical direction and magnitude; inconsistent directions are not converted into a universal claim.",
        "",
        "## Experiment 2 — clean five-step persistence",
        "",
        dog_line,
        bird_line,
        overlap_answer,
        "",
        "The persistence table and per-target summary report every 200-image checkpoint accuracy. Conclusions are limited to this fixed repeat.",
        "",
        "## Artifacts",
        "",
        "- `inputs/cell_manifest.csv` and `inputs/planned_generation.json`",
        "- `raw/*per_image_predictions*.csv` and per-cell evaluator outputs",
        "- `tables/experiment1_comparison.csv`",
        "- `tables/experiment2_persistence.csv`",
        "- `tables/experiment2_per_target_summary.csv`",
        "- `figures/experiment2_previous_erasure_heatmap.*`",
        "- `figures/experiment2_trajectories.*`",
        "",
        "Generated PNGs are removed only after the corresponding 200-row evaluator artifacts pass the final seed/count audit when delete-after-eval is active.",
        "",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    update_state(output_dir, "aggregation", "complete", **audit)
    event(output_dir, "aggregation", "wrote final follow-up results", new_images=7000)


def print_plan(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    counts = validate_config(config_path, config)
    rows = build_cell_manifest(config)
    print(
        json.dumps(
            {
                "config": str(config_path),
                "output_dir": str(Path(args.output_dir).resolve()),
                "hard_gate": {
                    "required_new_images": EXPECTED_NEW_IMAGES,
                    "planned_new_images": counts["total"]["new_formal_images"],
                    "passes": counts["total"]["new_formal_images"] == EXPECTED_NEW_IMAGES,
                },
                "counts": counts,
                "cells": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_all(args: argparse.Namespace) -> None:
    preflight(args)
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
        help="Override generated-image retention; evaluator artifacts are always kept.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    functions = {
        "plan": print_plan,
        "preflight": preflight,
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
