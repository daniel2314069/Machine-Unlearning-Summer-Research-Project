"""Resumable image-level qualification for two OCE implementation hypotheses.

The runner reproduces the unchanged official OCE object-edit path.  It does not
implement head-local OCE and does not repair the determinant correction.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from safetensors.torch import load_file, save_file


SCRIPT_DIR = Path(__file__).resolve().parent
OCE_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = OCE_ROOT.parent
ZERO_AUDIT_DIR = SCRIPT_DIR.parent / "zero_generation_qualification_audit"
ZERO_AUDIT_RESULTS = (
    ZERO_AUDIT_DIR
    / "results"
    / "run_20260820T111545Z"
    / "audit_results.json"
)
sys.path.insert(0, str(ZERO_AUDIT_DIR))

from audit_oce import (  # noqa: E402
    build_subspace,
    collect_layers,
    encode_prompts,
    expanded_object_prompts,
    numerical_rank,
    official_correction,
    sha256_file,
    tensor_sha256,
    verify_official_object_protocol,
)


STAGES = (
    "preflight",
    "prepare",
    "d1-canonical-generate",
    "d1-canonical-evaluate",
    "d1-composition-generate",
    "d1-composition-evaluate",
    "d3-generate",
    "d3-evaluate",
    "report",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("stage", choices=STAGES)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def stage_marker(output_dir: Path, stage: str) -> Path:
    return output_dir / "stages" / f"{stage}.json"


def stage_is_complete(output_dir: Path, stage: str) -> bool:
    path = stage_marker(output_dir, stage)
    return path.is_file() and read_json(path).get("status") == "complete"


def mark_stage(output_dir: Path, stage: str, details: Mapping[str, Any]) -> None:
    write_json(
        stage_marker(output_dir, stage),
        {
            "status": "complete",
            "stage": stage,
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **details,
        },
    )


def require_stage(output_dir: Path, stage: str) -> None:
    if not stage_is_complete(output_dir, stage):
        raise RuntimeError(f"Required stage is incomplete: {stage}")


def initialize(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_config.json"
    if resolved_path.exists():
        if read_json(resolved_path) != config:
            raise RuntimeError(
                f"Output config mismatch; use a new output directory: {resolved_path}"
            )
    else:
        write_json(resolved_path, config)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    official = config["official_object_protocol"]
    verified = verify_official_object_protocol(OCE_ROOT / "trainscripts" / "object.sh")
    expected = {
        "MODEL_ID": config["model_id"],
        "GUIDE_CONCEPTS": official["guide"],
        "PRESERVE_CONCEPTS": official["preserve"],
        "CONCEPT_TYPE": "object",
        "ERASE_SCALE": str(int(official["erase_scale"])),
        "PRESERVE_GLOBAL_SCALE": str(int(official["preserve_global_scale"])),
        "PRESERVE_CONCEPT_SCALE": str(int(official["preserve_concept_scale"])),
        "LAMB": str(int(official["lambda"])),
    }
    if verified != expected:
        raise RuntimeError(f"Frozen object protocol mismatch: {verified} != {expected}")
    targets = set(config["direction1"]["targets"])
    for item in config["direction1"]["two_object_prompts"]:
        words = set(str(item["prompt"]).casefold().split())
        if targets & words:
            raise RuntimeError(f"Two-object prompt contains erased target: {item}")
        if len(item["objects"]) != 2:
            raise RuntimeError(f"Two-object prompt does not name two labels: {item}")
    d3 = config["direction3"]
    if set(d3["targets"]) != set(d3["pre_registered_cases"]):
        raise RuntimeError("Direction 3 targets/cases do not match")
    if int(d3["gate"]["required_meaningful_cases"]) != 2:
        raise RuntimeError("Direction 3 majority gate must require two of three cases")
    if not ZERO_AUDIT_RESULTS.is_file():
        raise FileNotFoundError(f"Missing source zero-generation audit: {ZERO_AUDIT_RESULTS}")
    prior_rows = read_json(ZERO_AUDIT_RESULTS)["direction3_layers"]
    for target, selected_layer in d3["pre_registered_cases"].items():
        triggered_1280 = [
            row["layer"]
            for row in prior_rows
            if row["target"] == target
            and row["correction_triggered"]
            and row["objective_shape"] == [1280, 1280]
        ]
        if not triggered_1280 or triggered_1280[0] != selected_layer:
            raise RuntimeError(
                f"Direction 3 case is not the first prior triggered 1280 layer for "
                f"{target}: {selected_layer} vs {triggered_1280[:1]}"
            )


def release_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dtype_from_name(name: str) -> torch.dtype:
    options = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in options:
        raise ValueError(f"Unsupported dtype: {name}")
    return options[name]


def projection_modules(unet: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    modules = [
        (name, module)
        for name, module in unet.named_modules()
        if "attn2" in name and name.endswith("to_v")
    ]
    if len(modules) != 16:
        raise RuntimeError(f"Expected 16 official attn2.to_v layers, got {len(modules)}")
    return modules


def clone_projection_state(unet: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name + ".weight": module.weight.detach().cpu().clone()
        for name, module in projection_modules(unet)
    }


def apply_projection_state(
    unet: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    for name, module in projection_modules(unet):
        key = name + ".weight"
        if key not in state:
            raise KeyError(f"Checkpoint missing {key}")
        module.weight.data.copy_(
            state[key].to(device=module.weight.device, dtype=module.weight.dtype)
        )


def save_state(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.safetensors")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state.items()},
        str(temporary),
    )
    os.replace(temporary, path)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def relative_frobenius(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.matrix_norm(reference).clamp_min(1e-30)
    return float((torch.linalg.matrix_norm(actual - reference) / denominator).item())


def require_unique_image_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    paths = [str(row["image_path"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Image manifest contains duplicate paths")


def build_objective(
    *,
    weight: torch.Tensor,
    erase_embeddings: Sequence[torch.Tensor],
    guide_embeddings: Sequence[torch.Tensor],
    preserve_embeddings: Sequence[torch.Tensor],
    cg: torch.Tensor,
    protocol: Mapping[str, Any],
) -> torch.Tensor:
    guide = build_subspace(weight, list(guide_embeddings))
    erase = build_subspace(weight, list(erase_embeddings))
    guide_projector = guide @ guide.T
    erase_projector = erase @ erase.T
    identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
    objective = torch.zeros(
        weight.shape[0], weight.shape[0], device=weight.device, dtype=weight.dtype
    )
    objective += -float(protocol["erase_scale"]) * erase_projector @ (
        identity - guide_projector
    )
    for embedding in preserve_embeddings:
        value = weight @ embedding
        objective += float(protocol["preserve_concept_scale"]) * (
            value.unsqueeze(1) @ value.unsqueeze(0)
        )
    objective += float(protocol["preserve_global_scale"]) * (
        weight @ cg @ weight.T
    )
    objective += float(protocol["lambda"]) * (weight @ weight.T)
    return objective


def head_mixing_rows(
    target: str,
    layer: Mapping[str, Any],
    rotation: torch.Tensor,
) -> list[dict[str, Any]]:
    rotation64 = rotation.detach().cpu().double()
    heads = int(layer["head_count"])
    head_dim = int(layer["head_dim"])
    rows = []
    for source_head in range(heads):
        start = source_head * head_dim
        stop = start + head_dim
        source_columns = rotation64[:, start:stop]
        diagonal_block = rotation64[start:stop, start:stop]
        total_energy = float((source_columns * source_columns).sum().item())
        stay_energy = float((diagonal_block * diagonal_block).sum().item())
        leaked_energy = max(0.0, total_energy - stay_energy)
        rows.append(
            {
                "target": target,
                "layer": layer["name"],
                "layer_shape": str(layer["shape"]),
                "head_count": heads,
                "head_dim": head_dim,
                "source_head": source_head,
                "source_total_energy": total_energy,
                "same_head_energy": stay_energy,
                "other_head_energy": leaked_energy,
                "m_g": 1.0 - stay_energy / head_dim,
                "normalized_leak_fraction": leaked_energy / max(total_energy, 1e-30),
            }
        )
    return rows


def orthogonality_row(
    *,
    target: str,
    layer: Mapping[str, Any],
    raw_rotation: torch.Tensor,
    final_rotation: torch.Tensor,
    raw_determinant_float32: float,
    correction_triggered: bool,
) -> dict[str, Any]:
    raw64 = raw_rotation.detach().cpu().double()
    final64 = final_rotation.detach().cpu().double()
    identity = torch.eye(raw64.shape[0], dtype=torch.float64)
    residual = raw64.T @ raw64 - identity
    residual_fro = float(torch.linalg.matrix_norm(residual).item())
    singular_values = torch.linalg.svdvals(raw64)
    raw_det64 = float(torch.linalg.det(raw64).item())
    final_det64 = float(torch.linalg.det(final64).item())
    return {
        "target": target,
        "layer": layer["name"],
        "dimension": int(raw64.shape[0]),
        "raw_determinant_float32": raw_determinant_float32,
        "correction_triggered": correction_triggered,
        "raw_determinant_cpu_float64": raw_det64,
        "final_determinant_cpu_float64": final_det64,
        "rt_r_minus_i_fro": residual_fro,
        "rt_r_minus_i_relative": residual_fro / math.sqrt(raw64.shape[0]),
        "rotation_min_singular_value": float(singular_values[-1].item()),
        "rotation_max_singular_value": float(singular_values[0].item()),
    }


def make_d3_realizations(
    *, objective: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    objective64 = objective.detach().cpu().double()
    weight64 = weight.detach().cpu().double()
    u, singular, vh = torch.linalg.svd(objective64, full_matrices=False)
    rank, tolerance = numerical_rank(
        singular,
        objective64.shape[0],
        objective64.shape[1],
        torch.finfo(torch.float32).eps,
    )
    if rank >= objective64.shape[0]:
        raise RuntimeError("Pre-registered Direction 3 case is not numerically deficient")
    truncated = singular.clone()
    truncated[rank:] = 0
    objective_truncated = (u * truncated.unsqueeze(0)) @ vh
    u_alternative = u.clone()
    u_alternative[:, rank] *= -1
    alternative_reconstruction = (u_alternative * truncated.unsqueeze(0)) @ vh

    raw_a = u @ vh
    raw_b = u_alternative @ vh
    rotation_a, determinant_a, triggered_a = official_correction(raw_a)
    rotation_b, determinant_b, triggered_b = official_correction(raw_b)
    weight_a = (rotation_a @ weight64).float()
    weight_b = (rotation_b @ weight64).float()
    objective_norm = torch.linalg.matrix_norm(objective64).clamp_min(1e-30)
    details = {
        "numerical_rank": rank,
        "numerical_nullity": int(objective64.shape[0] - rank),
        "float32_resolution_tolerance_in_float64": tolerance,
        "rank_truncation_relative_to_objective": float(
            (
                torch.linalg.matrix_norm(objective_truncated - objective64)
                / objective_norm
            ).item()
        ),
        "alternative_reconstruction_relative_to_same_truncated_objective": float(
            (
                torch.linalg.matrix_norm(
                    alternative_reconstruction - objective_truncated
                )
                / torch.linalg.matrix_norm(objective_truncated).clamp_min(1e-30)
            ).item()
        ),
        "flipped_null_index": rank,
        "variant_a_raw_determinant": determinant_a,
        "variant_a_correction_triggered": triggered_a,
        "variant_b_raw_determinant": determinant_b,
        "variant_b_correction_triggered": triggered_b,
        "rotation_relative_difference": relative_frobenius(rotation_b, rotation_a),
        "edited_weight_relative_difference": relative_frobenius(weight_b, weight_a),
        "variant_a_weight_sha256": tensor_sha256(weight_a),
        "variant_b_weight_sha256": tensor_sha256(weight_b),
    }
    return weight_a, weight_b, details


@torch.inference_mode()
def stage_preflight(
    config: Mapping[str, Any], output_dir: Path, allow_network: bool
) -> None:
    if stage_is_complete(output_dir, "preflight"):
        print("[skip] preflight already complete")
        return
    device = torch.device(str(config["device"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on tslin")
    if not (OCE_ROOT / "Cg.pt").is_file():
        raise FileNotFoundError(OCE_ROOT / "Cg.pt")

    print("[preflight] diffusion model", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
        local_files_only=not allow_network,
    )
    if len(projection_modules(pipe.unet)) != 16:
        raise RuntimeError("Unexpected OCE layer count")
    del pipe
    gc.collect()

    print("[preflight] COCO detector", flush=True)
    import torchvision
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    detector_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    detector_checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(detector_weights.url).name
    )
    if not detector_checkpoint.is_file() and not allow_network:
        raise FileNotFoundError(
            "The COCO detector checkpoint is not cached; rerun the server launcher "
            "with --allow-network"
        )
    detector = fasterrcnn_resnet50_fpn_v2(weights=detector_weights).eval()
    if not detector_checkpoint.is_file():
        raise FileNotFoundError(
            f"Resolved detector checkpoint was not found: {detector_checkpoint}"
        )
    categories = list(detector_weights.meta["categories"])
    required_categories = {
        target for target in config["direction1"]["targets"]
    } | {
        label
        for item in config["direction1"]["two_object_prompts"]
        for label in item["objects"]
    }
    missing = sorted(required_categories - set(categories))
    if missing:
        raise RuntimeError(f"Detector lacks required COCO labels: {missing}")
    del detector
    gc.collect()

    print("[preflight] CLIP classifier", flush=True)
    from transformers import CLIPModel, CLIPProcessor, __version__ as transformers_version

    classifier_id = str(config["direction3"]["classifier"]["model_id"])
    classifier = CLIPModel.from_pretrained(
        classifier_id, local_files_only=not allow_network
    ).eval()
    processor = CLIPProcessor.from_pretrained(
        classifier_id, local_files_only=not allow_network
    )
    del classifier, processor
    gc.collect()

    print("[preflight] LPIPS", flush=True)
    import lpips

    installed_lpips = importlib.metadata.version("lpips")
    expected_lpips = str(config["direction3"]["lpips"]["version"])
    if installed_lpips != expected_lpips:
        raise RuntimeError(
            f"LPIPS version mismatch: expected {expected_lpips}, got {installed_lpips}"
        )
    lpips_model = lpips.LPIPS(
        net=str(config["direction3"]["lpips"]["network"])
    ).eval()
    del lpips_model
    gc.collect()

    manifest = {
        "experiment_id": config["experiment_id"],
        "repository_git_revision": git_revision(),
        "official_oce_sha256": sha256_file(OCE_ROOT / "oce.py"),
        "official_object_script_sha256": sha256_file(
            OCE_ROOT / "trainscripts" / "object.sh"
        ),
        "cg_sha256": sha256_file(OCE_ROOT / "Cg.pt"),
        "source_zero_generation_audit": {
            "path": str(ZERO_AUDIT_RESULTS),
            "sha256": sha256_file(ZERO_AUDIT_RESULTS),
        },
        "generation_before_preflight": False,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torchvision": torchvision.__version__,
            "transformers": transformers_version,
            "lpips": installed_lpips,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
        "detector": {
            "implementation": config["direction1"]["detector"]["implementation"],
            "weights": config["direction1"]["detector"]["weights"],
            "weights_url": detector_weights.url,
            "checkpoint_path": str(detector_checkpoint),
            "checkpoint_sha256": sha256_file(detector_checkpoint),
            "categories": categories,
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    mark_stage(output_dir, "preflight", {"run_manifest": "run_manifest.json"})


@torch.inference_mode()
def stage_prepare(config: Mapping[str, Any], output_dir: Path, allow_network: bool) -> None:
    require_stage(output_dir, "preflight")
    if stage_is_complete(output_dir, "prepare"):
        print("[skip] prepare already complete")
        return
    device = torch.device(str(config["device"]))
    print("[prepare] loading official float32 edit stack", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
        local_files_only=not allow_network,
    ).to(device)
    layers, implementation_records = collect_layers(pipe.unet)
    targets = list(config["direction1"]["targets"])
    prompt_sets = {target: expanded_object_prompts(target) for target in targets}
    prompts = sorted(
        {
            prompt
            for edit, guide, preserve in prompt_sets.values()
            for prompt in edit + guide + preserve
        }
    )
    embeddings = encode_prompts(pipe, prompts, device)
    cg_payload = torch.load(OCE_ROOT / "Cg.pt", map_location=device)
    cg = cg_payload["C"].to(device=device, dtype=torch.float32)
    protocol = config["official_object_protocol"]
    pre_registered = dict(config["direction3"]["pre_registered_cases"])

    mixing: list[dict[str, Any]] = []
    orthogonality: list[dict[str, Any]] = []
    d3_construction: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []

    for target_index, target in enumerate(targets, start=1):
        print(f"[prepare target {target_index}/{len(targets)}] {target}", flush=True)
        edit_prompts, guide_prompts, preserve_prompts = prompt_sets[target]
        erase_embeddings = [embeddings[prompt] for prompt in edit_prompts]
        guide_embeddings = [embeddings[prompt] for prompt in guide_prompts]
        preserve_embeddings = [embeddings[prompt] for prompt in preserve_prompts]
        official_state: dict[str, torch.Tensor] = {}
        selected_payload: tuple[dict[str, Any], torch.Tensor, torch.Tensor, bool] | None = None

        for layer_index, layer in enumerate(layers, start=1):
            print(
                f"  [layer {layer_index}/{len(layers)}] {layer['name']}", flush=True
            )
            weight = layer["module"].weight.detach().float()
            objective = build_objective(
                weight=weight,
                erase_embeddings=erase_embeddings,
                guide_embeddings=guide_embeddings,
                preserve_embeddings=preserve_embeddings,
                cg=cg,
                protocol=protocol,
            )
            u, singular, vh = torch.linalg.svd(objective, full_matrices=False)
            raw_rotation = u @ vh
            final_rotation, raw_det, triggered = official_correction(raw_rotation)
            final_weight = final_rotation @ weight
            official_state[layer["name"] + ".weight"] = final_weight.detach().cpu()
            mixing.extend(head_mixing_rows(target, layer, final_rotation))
            orthogonality.append(
                orthogonality_row(
                    target=target,
                    layer=layer,
                    raw_rotation=raw_rotation,
                    final_rotation=final_rotation,
                    raw_determinant_float32=raw_det,
                    correction_triggered=triggered,
                )
            )
            if target in pre_registered and layer["name"] == pre_registered[target]:
                selected_payload = (
                    layer,
                    objective.detach().cpu(),
                    weight.detach().cpu(),
                    triggered,
                )
            del objective, u, singular, vh, raw_rotation, final_rotation, final_weight
            if device.type == "cuda":
                torch.cuda.empty_cache()

        d1_path = output_dir / "checkpoints" / "d1_official" / f"{target}.safetensors"
        save_state(d1_path, official_state)
        checkpoint_records.append(
            {
                "kind": "direction1_official_oce",
                "target": target,
                "path": str(d1_path.relative_to(output_dir)),
                "state_sha256": state_sha256(official_state),
                "layers": len(official_state),
            }
        )

        if target in pre_registered:
            if selected_payload is None:
                raise RuntimeError(f"Pre-registered layer was not found for {target}")
            selected_layer, selected_objective, selected_weight, selected_triggered = (
                selected_payload
            )
            if selected_layer["shape"][0] != 1280 or not selected_triggered:
                raise RuntimeError(
                    f"Pre-registered case is not a triggered 1280 layer: {target} / "
                    f"{selected_layer['name']}"
                )
            weight_a, weight_b, construction = make_d3_realizations(
                objective=selected_objective, weight=selected_weight
            )
            key = selected_layer["name"] + ".weight"
            state_a = dict(official_state)
            state_b = dict(official_state)
            state_a[key] = weight_a
            state_b[key] = weight_b
            unequal_other_layers = [
                candidate
                for candidate in official_state
                if candidate != key and not torch.equal(state_a[candidate], state_b[candidate])
            ]
            if unequal_other_layers:
                raise RuntimeError(f"D3 variants differ outside selected layer: {unequal_other_layers}")
            path_a = output_dir / "checkpoints" / "d3" / target / "realization_a.safetensors"
            path_b = output_dir / "checkpoints" / "d3" / target / "realization_b.safetensors"
            save_state(path_a, state_a)
            save_state(path_b, state_b)
            construction.update(
                {
                    "target": target,
                    "selected_layer": selected_layer["name"],
                    "selection_policy": config["direction3"]["selection_policy"],
                    "selected_layer_official_correction_triggered": selected_triggered,
                    "all_other_layers_bitwise_identical": True,
                    "variant_a_checkpoint": str(path_a.relative_to(output_dir)),
                    "variant_b_checkpoint": str(path_b.relative_to(output_dir)),
                    "variant_a_state_sha256": state_sha256(state_a),
                    "variant_b_state_sha256": state_sha256(state_b),
                    "variant_a_vs_official_selected_weight_relative_difference": relative_frobenius(
                        weight_a.double(), official_state[key].double()
                    ),
                    "variant_b_vs_official_selected_weight_relative_difference": relative_frobenius(
                        weight_b.double(), official_state[key].double()
                    ),
                }
            )
            d3_construction.append(construction)
            for variant, path, state in [
                ("realization_a", path_a, state_a),
                ("realization_b", path_b, state_b),
            ]:
                checkpoint_records.append(
                    {
                        "kind": "direction3_legal_svd_realization",
                        "target": target,
                        "selected_layer": selected_layer["name"],
                        "variant": variant,
                        "path": str(path.relative_to(output_dir)),
                        "state_sha256": state_sha256(state),
                        "layers": len(state),
                    }
                )

    write_csv(
        output_dir / "operator" / "head_mixing.csv",
        mixing,
        [
            "target",
            "layer",
            "layer_shape",
            "head_count",
            "head_dim",
            "source_head",
            "source_total_energy",
            "same_head_energy",
            "other_head_energy",
            "m_g",
            "normalized_leak_fraction",
        ],
    )
    write_csv(
        output_dir / "operator" / "orthogonality.csv",
        orthogonality,
        [
            "target",
            "layer",
            "dimension",
            "raw_determinant_float32",
            "correction_triggered",
            "raw_determinant_cpu_float64",
            "final_determinant_cpu_float64",
            "rt_r_minus_i_fro",
            "rt_r_minus_i_relative",
            "rotation_min_singular_value",
            "rotation_max_singular_value",
        ],
    )
    write_json(output_dir / "operator" / "d3_case_construction.json", d3_construction)
    expected_mixing_rows = len(targets) * sum(
        int(layer["head_count"]) for layer in layers
    )
    expected_orthogonality_rows = len(targets) * len(layers)
    if (
        len(mixing) != expected_mixing_rows
        or len(orthogonality) != expected_orthogonality_rows
    ):
        raise RuntimeError(
            f"Incomplete operator audit: mixing={len(mixing)}/"
            f"{expected_mixing_rows}, orthogonality={len(orthogonality)}/"
            f"{expected_orthogonality_rows}"
        )
    if len(d3_construction) != 3 or len(checkpoint_records) != 10:
        raise RuntimeError(
            f"Incomplete checkpoint construction: cases={len(d3_construction)}, "
            f"records={len(checkpoint_records)}"
        )
    write_json(
        output_dir / "checkpoints" / "manifest.json",
        {
            "official_oce_sha256": sha256_file(OCE_ROOT / "oce.py"),
            "attention_implementation": implementation_records,
            "records": checkpoint_records,
        },
    )
    mixing_values = [float(row["m_g"]) for row in mixing]
    triggered_orthogonality = [row for row in orthogonality if row["correction_triggered"]]
    summary = {
        "direction1_head_mixing": {
            "rows": len(mixing),
            "minimum_m_g": min(mixing_values),
            "median_m_g": statistics.median(mixing_values),
            "maximum_m_g": max(mixing_values),
            "target_medians": {
                target: statistics.median(
                    float(row["m_g"]) for row in mixing if row["target"] == target
                )
                for target in targets
            },
        },
        "direction3_orthogonality": {
            "rows": len(orthogonality),
            "triggered_rows": len(triggered_orthogonality),
            "maximum_rt_r_minus_i_fro": max(
                float(row["rt_r_minus_i_fro"]) for row in orthogonality
            ),
            "maximum_rt_r_minus_i_relative": max(
                float(row["rt_r_minus_i_relative"]) for row in orthogonality
            ),
            "minimum_rotation_singular_value": min(
                float(row["rotation_min_singular_value"]) for row in orthogonality
            ),
            "maximum_rotation_singular_value": max(
                float(row["rotation_max_singular_value"]) for row in orthogonality
            ),
            "triggered_float32_determinant_range": [
                min(float(row["raw_determinant_float32"]) for row in triggered_orthogonality),
                max(float(row["raw_determinant_float32"]) for row in triggered_orthogonality),
            ],
            "triggered_cpu_float64_determinant_range": [
                min(
                    float(row["raw_determinant_cpu_float64"])
                    for row in triggered_orthogonality
                ),
                max(
                    float(row["raw_determinant_cpu_float64"])
                    for row in triggered_orthogonality
                ),
            ],
        },
    }
    write_json(output_dir / "operator" / "summary.json", summary)
    mark_stage(
        output_dir,
        "prepare",
        {
            "head_mixing_rows": len(mixing),
            "orthogonality_rows": len(orthogonality),
            "d3_cases": len(d3_construction),
            "checkpoint_records": len(checkpoint_records),
        },
    )
    release_cuda(pipe)


def article_for(target: str) -> str:
    return "an" if target[0].casefold() in "aeiou" else "a"


def canonical_prompt_rows(config: Mapping[str, Any], target: str) -> list[dict[str, str]]:
    article = article_for(target)
    return [
        {
            "prompt_id": f"{target}_canonical_{index:02d}",
            "prompt": str(template).format(article=article, target=target),
            "expected": target,
        }
        for index, template in enumerate(
            config["direction1"]["canonical_prompt_templates"], start=1
        )
    ]


def image_sha256(path: Path) -> str:
    return sha256_file(path)


def verify_existing_image(path: Path, width: int, height: int) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.size != (width, height):
            raise RuntimeError(f"Existing image has wrong size: {path} / {image.size}")


@torch.inference_mode()
def generate_one(
    pipe: DiffusionPipeline,
    *,
    prompt: str,
    seed: int,
    destination: Path,
    generation: Mapping[str, Any],
    device: torch.device,
) -> None:
    width = int(generation["width"])
    height = int(generation["height"])
    if destination.exists():
        verify_existing_image(destination, width, height)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = pipe(
        prompt=prompt,
        num_inference_steps=int(generation["num_inference_steps"]),
        guidance_scale=float(generation["guidance_scale"]),
        width=width,
        height=height,
        generator=torch.Generator(device=device).manual_seed(int(seed)),
    ).images[0]
    temporary = destination.with_suffix(".tmp.png")
    image.save(temporary)
    os.replace(temporary, destination)


def load_generation_pipe(
    config: Mapping[str, Any], allow_network: bool
) -> tuple[DiffusionPipeline, torch.device, dict[str, torch.Tensor]]:
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=dtype_from_name(str(config["generation"]["dtype"])),
        safety_checker=None,
        local_files_only=not allow_network,
    ).to(device)
    return pipe, device, clone_projection_state(pipe.unet)


@torch.inference_mode()
def stage_d1_canonical_generate(
    config: Mapping[str, Any], output_dir: Path, allow_network: bool
) -> None:
    require_stage(output_dir, "prepare")
    stage = "d1-canonical-generate"
    if stage_is_complete(output_dir, stage):
        print(f"[skip] {stage} already complete")
        return
    pipe, device, original_state = load_generation_pipe(config, allow_network)
    generation = config["generation"]
    seeds = [int(seed) for seed in generation["seeds"]]
    rows: list[dict[str, Any]] = []
    targets = list(config["direction1"]["targets"])
    for condition_target in [None, *targets]:
        if condition_target is None:
            condition = "original"
            apply_projection_state(pipe.unet, original_state)
            active_targets = targets
        else:
            condition = f"oce_{condition_target}"
            checkpoint = (
                output_dir
                / "checkpoints"
                / "d1_official"
                / f"{condition_target}.safetensors"
            )
            apply_projection_state(pipe.unet, load_file(str(checkpoint)))
            active_targets = [condition_target]
        for target in active_targets:
            for prompt_row in canonical_prompt_rows(config, target):
                for seed in seeds:
                    path = (
                        output_dir
                        / "d1"
                        / "images"
                        / "canonical"
                        / condition
                        / target
                        / prompt_row["prompt_id"]
                        / f"seed_{seed}.png"
                    )
                    print(f"[generate] {condition} / {target} / {prompt_row['prompt_id']} / {seed}")
                    generate_one(
                        pipe,
                        prompt=prompt_row["prompt"],
                        seed=seed,
                        destination=path,
                        generation=generation,
                        device=device,
                    )
                    rows.append(
                        {
                            "condition": condition,
                            "target": target,
                            **prompt_row,
                            "seed": seed,
                            "image_path": str(path.resolve()),
                            "image_sha256": image_sha256(path),
                        }
                    )
    require_unique_image_rows(rows)
    if len(rows) != 128:
        raise RuntimeError(f"Expected 128 canonical images, got {len(rows)}")
    write_csv(
        output_dir / "d1" / "canonical_images.csv",
        rows,
        [
            "condition",
            "target",
            "prompt_id",
            "prompt",
            "expected",
            "seed",
            "image_path",
            "image_sha256",
        ],
    )
    mark_stage(output_dir, stage, {"images": len(rows)})
    release_cuda(pipe)


def load_detector(config: Mapping[str, Any]) -> tuple[Any, Any, list[str], torch.device]:
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    device = torch.device(str(config["device"]))
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)
    return detector, weights.transforms(), list(weights.meta["categories"]), device


@torch.inference_mode()
def detector_scores(
    image_rows: Sequence[Mapping[str, Any]],
    labels_needed: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    detector, transform, categories, device = load_detector(config)
    batch_size = int(config["direction1"]["detector"]["batch_size"])
    threshold = float(config["direction1"]["detector"]["score_threshold"])
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(image_rows), batch_size):
        batch_rows = image_rows[start : start + batch_size]
        tensors = []
        for row in batch_rows:
            with Image.open(str(row["image_path"])) as image:
                tensors.append(transform(image.convert("RGB")).to(device))
        predictions = detector(tensors)
        for row, prediction in zip(batch_rows, predictions):
            scores = prediction["scores"].detach().cpu().tolist()
            labels = prediction["labels"].detach().cpu().tolist()
            best_by_label: dict[str, float] = defaultdict(float)
            for label_index, score in zip(labels, scores):
                label = categories[int(label_index)]
                best_by_label[label] = max(best_by_label[label], float(score))
            expected = list(labels_needed[str(row["prompt_id"])])
            expected_scores = {label: best_by_label[label] for label in expected}
            present = {label: score >= threshold for label, score in expected_scores.items()}
            outputs.append(
                {
                    **dict(row),
                    "expected_labels": expected,
                    "expected_scores": expected_scores,
                    "expected_present": present,
                    "correct": all(present.values()),
                    "object_recall": sum(present.values()) / len(present),
                }
            )
    release_cuda(detector)
    return outputs


@torch.inference_mode()
def stage_d1_canonical_evaluate(config: Mapping[str, Any], output_dir: Path) -> None:
    require_stage(output_dir, "d1-canonical-generate")
    stage = "d1-canonical-evaluate"
    if stage_is_complete(output_dir, stage):
        print(f"[skip] {stage} already complete")
        return
    image_rows = read_csv(output_dir / "d1" / "canonical_images.csv")
    labels_needed = {
        row["prompt_id"]: [row["expected"]] for row in image_rows
    }
    items = detector_scores(image_rows, labels_needed, config)
    gate = config["direction1"]["gate"]
    summaries = []
    for target in config["direction1"]["targets"]:
        original = [
            item for item in items if item["target"] == target and item["condition"] == "original"
        ]
        edited = [
            item
            for item in items
            if item["target"] == target and item["condition"] == f"oce_{target}"
        ]
        if len(original) != 16 or len(edited) != 16:
            raise RuntimeError(
                f"Incomplete canonical evaluation for {target}: "
                f"original={len(original)}, edited={len(edited)}"
            )
        original_accuracy = sum(item["correct"] for item in original) / len(original)
        edited_accuracy = sum(item["correct"] for item in edited) / len(edited)
        drop = original_accuracy - edited_accuracy
        qualified = (
            original_accuracy >= float(gate["canonical_original_accuracy_floor"])
            and drop >= float(gate["canonical_min_detection_drop"])
            and edited_accuracy <= float(gate["canonical_oce_accuracy_ceiling"])
        )
        summaries.append(
            {
                "target": target,
                "images_per_condition": len(original),
                "original_target_detection_accuracy": original_accuracy,
                "official_oce_target_detection_accuracy": edited_accuracy,
                "target_detection_drop": drop,
                "canonical_erasure_qualified": qualified,
            }
        )
    payload = {
        "evaluator": config["direction1"]["detector"],
        "gate": gate,
        "qualified_targets": [
            row["target"] for row in summaries if row["canonical_erasure_qualified"]
        ],
        "summaries": summaries,
        "items": items,
    }
    write_json(output_dir / "d1" / "canonical_metrics.json", payload)
    mark_stage(
        output_dir,
        stage,
        {"qualified_targets": payload["qualified_targets"], "evaluated_images": len(items)},
    )


@torch.inference_mode()
def stage_d1_composition_generate(
    config: Mapping[str, Any], output_dir: Path, allow_network: bool
) -> None:
    require_stage(output_dir, "d1-canonical-evaluate")
    stage = "d1-composition-generate"
    if stage_is_complete(output_dir, stage):
        print(f"[skip] {stage} already complete")
        return
    canonical = read_json(output_dir / "d1" / "canonical_metrics.json")
    qualified = list(canonical["qualified_targets"])
    rows: list[dict[str, Any]] = []
    if qualified:
        pipe, device, original_state = load_generation_pipe(config, allow_network)
        generation = config["generation"]
        seeds = [int(seed) for seed in generation["seeds"]]
        prompts = list(config["direction1"]["two_object_prompts"])
        for condition_target in [None, *qualified]:
            if condition_target is None:
                condition = "original"
                apply_projection_state(pipe.unet, original_state)
            else:
                condition = f"oce_{condition_target}"
                checkpoint = (
                    output_dir / "checkpoints" / "d1_official" / f"{condition_target}.safetensors"
                )
                apply_projection_state(pipe.unet, load_file(str(checkpoint)))
            for prompt_row in prompts:
                for seed in seeds:
                    path = (
                        output_dir
                        / "d1"
                        / "images"
                        / "composition"
                        / condition
                        / str(prompt_row["id"])
                        / f"seed_{seed}.png"
                    )
                    print(f"[generate] {condition} / {prompt_row['id']} / {seed}")
                    generate_one(
                        pipe,
                        prompt=str(prompt_row["prompt"]),
                        seed=seed,
                        destination=path,
                        generation=generation,
                        device=device,
                    )
                    rows.append(
                        {
                            "condition": condition,
                            "edited_target": condition_target or "",
                            "prompt_id": prompt_row["id"],
                            "prompt": prompt_row["prompt"],
                            "object_1": prompt_row["objects"][0],
                            "object_2": prompt_row["objects"][1],
                            "seed": seed,
                            "image_path": str(path.resolve()),
                            "image_sha256": image_sha256(path),
                        }
                    )
        release_cuda(pipe)
    require_unique_image_rows(rows)
    expected_images = 24 * (1 + len(qualified)) if qualified else 0
    if len(rows) != expected_images:
        raise RuntimeError(
            f"Expected {expected_images} composition images, got {len(rows)}"
        )
    write_csv(
        output_dir / "d1" / "composition_images.csv",
        rows,
        [
            "condition",
            "edited_target",
            "prompt_id",
            "prompt",
            "object_1",
            "object_2",
            "seed",
            "image_path",
            "image_sha256",
        ],
    )
    mark_stage(output_dir, stage, {"qualified_targets": qualified, "images": len(rows)})


@torch.inference_mode()
def stage_d1_composition_evaluate(config: Mapping[str, Any], output_dir: Path) -> None:
    require_stage(output_dir, "d1-composition-generate")
    stage = "d1-composition-evaluate"
    if stage_is_complete(output_dir, stage):
        print(f"[skip] {stage} already complete")
        return
    image_rows = read_csv(output_dir / "d1" / "composition_images.csv")
    canonical = read_json(output_dir / "d1" / "canonical_metrics.json")
    qualified = list(canonical["qualified_targets"])
    gate = config["direction1"]["gate"]
    items: list[dict[str, Any]] = []
    if image_rows:
        labels_needed = {
            row["prompt_id"]: [row["object_1"], row["object_2"]]
            for row in image_rows
        }
        items = detector_scores(image_rows, labels_needed, config)
    original = [item for item in items if item["condition"] == "original"]
    if qualified and len(original) != 24:
        raise RuntimeError(f"Expected 24 Original composition images, got {len(original)}")
    original_accuracy = (
        sum(item["correct"] for item in original) / len(original) if original else 0.0
    )
    target_summaries = []
    for target in config["direction1"]["targets"]:
        edited = [item for item in items if item["condition"] == f"oce_{target}"]
        if target not in qualified or not edited:
            target_summaries.append(
                {
                    "target": target,
                    "canonical_erasure_qualified": target in qualified,
                    "composition_evaluated": False,
                    "stable_compositional_degradation": False,
                }
            )
            continue
        if len(edited) != 24:
            raise RuntimeError(
                f"Expected 24 composition images for {target}, got {len(edited)}"
            )
        edited_accuracy = sum(item["correct"] for item in edited) / len(edited)
        accuracy_drop = original_accuracy - edited_accuracy
        prompt_deltas = []
        for prompt in config["direction1"]["two_object_prompts"]:
            prompt_id = str(prompt["id"])
            original_prompt = [item for item in original if item["prompt_id"] == prompt_id]
            edited_prompt = [item for item in edited if item["prompt_id"] == prompt_id]
            original_prompt_accuracy = sum(item["correct"] for item in original_prompt) / len(
                original_prompt
            )
            edited_prompt_accuracy = sum(item["correct"] for item in edited_prompt) / len(
                edited_prompt
            )
            prompt_deltas.append(
                {
                    "prompt_id": prompt_id,
                    "original_accuracy": original_prompt_accuracy,
                    "official_oce_accuracy": edited_prompt_accuracy,
                    "accuracy_drop": original_prompt_accuracy - edited_prompt_accuracy,
                }
            )
        degraded_families = sum(row["accuracy_drop"] > 0 for row in prompt_deltas)
        stable = (
            original_accuracy >= float(gate["composition_original_accuracy_floor"])
            and accuracy_drop >= float(gate["composition_min_accuracy_drop"])
            and degraded_families
            >= int(gate["composition_min_degraded_prompt_families"])
        )
        target_summaries.append(
            {
                "target": target,
                "canonical_erasure_qualified": True,
                "composition_evaluated": True,
                "images_per_condition": len(edited),
                "original_two_object_accuracy": original_accuracy,
                "official_oce_two_object_accuracy": edited_accuracy,
                "two_object_accuracy_drop": accuracy_drop,
                "degraded_prompt_families": degraded_families,
                "prompt_families": prompt_deltas,
                "stable_compositional_degradation": stable,
            }
        )
    stable_count = sum(
        row["stable_compositional_degradation"] for row in target_summaries
    )
    decision = (
        "GO"
        if stable_count >= int(gate["required_stably_degraded_targets"])
        else "NO-GO"
    )
    payload = {
        "decision": decision,
        "reason": (
            f"{stable_count}/{len(config['direction1']['targets'])} targets showed "
            "pre-registered stable two-object degradation"
        ),
        "evaluator": config["direction1"]["detector"],
        "gate": gate,
        "canonical_qualified_targets": qualified,
        "original_two_object_accuracy": original_accuracy,
        "stable_degradation_count": stable_count,
        "target_summaries": target_summaries,
        "items": items,
    }
    write_json(output_dir / "d1" / "composition_metrics.json", payload)
    mark_stage(output_dir, stage, {"decision": decision, "evaluated_images": len(items)})


@torch.inference_mode()
def stage_d3_generate(
    config: Mapping[str, Any], output_dir: Path, allow_network: bool
) -> None:
    require_stage(output_dir, "prepare")
    stage = "d3-generate"
    if stage_is_complete(output_dir, stage):
        print(f"[skip] {stage} already complete")
        return
    pipe, device, _ = load_generation_pipe(config, allow_network)
    generation = config["generation"]
    seeds = [int(seed) for seed in generation["seeds"]]
    rows: list[dict[str, Any]] = []
    for target in config["direction3"]["targets"]:
        prompt_rows = [
            {**row, "role": "target", "expected": target}
            for row in canonical_prompt_rows(config, target)
        ] + [
            {
                "prompt_id": row["id"],
                "prompt": row["prompt"],
                "expected": row["expected"],
                "role": "non_target",
            }
            for row in config["direction3"]["non_target_prompts"]
        ]
        for variant in ["realization_a", "realization_b"]:
            checkpoint = output_dir / "checkpoints" / "d3" / target / f"{variant}.safetensors"
            apply_projection_state(pipe.unet, load_file(str(checkpoint)))
            for prompt_row in prompt_rows:
                for seed in seeds:
                    path = (
                        output_dir
                        / "d3"
                        / "images"
                        / target
                        / variant
                        / str(prompt_row["role"])
                        / str(prompt_row["prompt_id"])
                        / f"seed_{seed}.png"
                    )
                    print(
                        f"[generate] D3 {target} / {variant} / "
                        f"{prompt_row['prompt_id']} / {seed}"
                    )
                    generate_one(
                        pipe,
                        prompt=str(prompt_row["prompt"]),
                        seed=seed,
                        destination=path,
                        generation=generation,
                        device=device,
                    )
                    rows.append(
                        {
                            "case_target": target,
                            "selected_layer": config["direction3"]["pre_registered_cases"][target],
                            "variant": variant,
                            "role": prompt_row["role"],
                            "prompt_id": prompt_row["prompt_id"],
                            "prompt": prompt_row["prompt"],
                            "expected": prompt_row["expected"],
                            "seed": seed,
                            "image_path": str(path.resolve()),
                            "image_sha256": image_sha256(path),
                        }
                    )
    require_unique_image_rows(rows)
    if len(rows) != 192:
        raise RuntimeError(f"Expected 192 Direction 3 images, got {len(rows)}")
    write_csv(
        output_dir / "d3" / "images.csv",
        rows,
        [
            "case_target",
            "selected_layer",
            "variant",
            "role",
            "prompt_id",
            "prompt",
            "expected",
            "seed",
            "image_path",
            "image_sha256",
        ],
    )
    mark_stage(output_dir, stage, {"images": len(rows), "cases": 3})
    release_cuda(pipe)


@torch.inference_mode()
def clip_classify(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    from transformers import CLIPModel, CLIPProcessor

    classifier_config = config["direction3"]["classifier"]
    device = torch.device(str(config["device"]))
    model_id = str(classifier_config["model_id"])
    model = CLIPModel.from_pretrained(model_id, local_files_only=True).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    labels = list(classifier_config["labels"])
    templates = list(classifier_config["templates"])
    text_vectors = []
    for label in labels:
        texts = [str(template).format(label) for template in templates]
        inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
        vectors = model.get_text_features(**inputs)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        mean = vectors.mean(dim=0)
        text_vectors.append(mean / mean.norm().clamp_min(1e-12))
    text_matrix = torch.stack(text_vectors, dim=0)
    batch_size = int(classifier_config["batch_size"])
    output = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = []
        for row in batch_rows:
            with Image.open(str(row["image_path"])) as image:
                images.append(image.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt").to(device)
        image_vectors = model.get_image_features(**inputs)
        image_vectors = image_vectors / image_vectors.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        probabilities = (
            model.logit_scale.exp() * image_vectors @ text_matrix.T
        ).softmax(dim=-1)
        for row, probability in zip(batch_rows, probabilities.detach().cpu()):
            predicted_index = int(probability.argmax().item())
            expected_index = labels.index(str(row["expected"]))
            output.append(
                {
                    **dict(row),
                    "predicted": labels[predicted_index],
                    "correct": predicted_index == expected_index,
                    "expected_probability": float(probability[expected_index].item()),
                    "top1_probability": float(probability[predicted_index].item()),
                }
            )
    release_cuda(model)
    return output


def lpips_tensor(path: Path, device: torch.device) -> torch.Tensor:
    from torchvision.transforms.functional import pil_to_tensor

    with Image.open(path) as image:
        tensor = pil_to_tensor(image.convert("RGB")).to(
            device=device, dtype=torch.float32
        )
    return tensor.unsqueeze(0) / 127.5 - 1.0


@torch.inference_mode()
def paired_lpips(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    import lpips

    device = torch.device(str(config["device"]))
    model = lpips.LPIPS(
        net=str(config["direction3"]["lpips"]["network"])
    ).eval().to(device)
    lookup = {
        (
            str(row["case_target"]),
            str(row["variant"]),
            str(row["role"]),
            str(row["prompt_id"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    output = []
    for target in config["direction3"]["targets"]:
        keys = sorted(
            key
            for key in lookup
            if key[0] == target and key[1] == "realization_a"
        )
        for key_a in keys:
            _, _, role, prompt_id, seed = key_a
            key_b = (target, "realization_b", role, prompt_id, seed)
            if key_b not in lookup:
                raise RuntimeError(f"Missing paired Direction 3 image: {key_b}")
            row_a = lookup[key_a]
            row_b = lookup[key_b]
            value = float(
                model(
                    lpips_tensor(Path(str(row_a["image_path"])), device),
                    lpips_tensor(Path(str(row_b["image_path"])), device),
                ).reshape(()).item()
            )
            output.append(
                {
                    "case_target": target,
                    "role": role,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "variant_a_path": row_a["image_path"],
                    "variant_b_path": row_b["image_path"],
                    "lpips": value,
                }
            )
    release_cuda(model)
    return output


@torch.inference_mode()
def stage_d3_evaluate(config: Mapping[str, Any], output_dir: Path) -> None:
    require_stage(output_dir, "d3-generate")
    stage = "d3-evaluate"
    if stage_is_complete(output_dir, stage):
        print(f"[skip] {stage} already complete")
        return
    rows = read_csv(output_dir / "d3" / "images.csv")
    normalized_rows = [{**row, "seed": int(row["seed"])} for row in rows]
    classifications = clip_classify(normalized_rows, config)
    lpips_rows = paired_lpips(normalized_rows, config)
    gate = config["direction3"]["gate"]
    case_summaries = []
    for target in config["direction3"]["targets"]:
        variant_metrics: dict[str, Any] = {}
        for variant in ["realization_a", "realization_b"]:
            subset = [
                row
                for row in classifications
                if row["case_target"] == target and row["variant"] == variant
            ]
            target_rows = [row for row in subset if row["role"] == "target"]
            non_target_rows = [row for row in subset if row["role"] == "non_target"]
            if len(target_rows) != 16 or len(non_target_rows) != 16:
                raise RuntimeError(
                    f"Incomplete D3 classification rows for {target}/{variant}: "
                    f"target={len(target_rows)}, non_target={len(non_target_rows)}"
                )
            variant_metrics[variant] = {
                "target_classification_accuracy": sum(
                    row["correct"] for row in target_rows
                )
                / len(target_rows),
                "non_target_classification_accuracy": sum(
                    row["correct"] for row in non_target_rows
                )
                / len(non_target_rows),
                "target_expected_probability_mean": statistics.mean(
                    float(row["expected_probability"]) for row in target_rows
                ),
                "non_target_expected_probability_mean": statistics.mean(
                    float(row["expected_probability"]) for row in non_target_rows
                ),
                "target_images": len(target_rows),
                "non_target_images": len(non_target_rows),
            }
        target_delta = (
            variant_metrics["realization_b"]["target_classification_accuracy"]
            - variant_metrics["realization_a"]["target_classification_accuracy"]
        )
        non_target_delta = (
            variant_metrics["realization_b"]["non_target_classification_accuracy"]
            - variant_metrics["realization_a"]["non_target_classification_accuracy"]
        )
        case_lpips = [
            float(row["lpips"]) for row in lpips_rows if row["case_target"] == target
        ]
        if len(case_lpips) != 32:
            raise RuntimeError(
                f"Expected 32 paired LPIPS rows for {target}, got {len(case_lpips)}"
            )
        mean_lpips = statistics.mean(case_lpips)
        meaningful = (
            max(abs(target_delta), abs(non_target_delta))
            >= float(gate["meaningful_accuracy_delta"])
            and mean_lpips >= float(gate["minimum_mean_paired_lpips"])
        )
        case_summaries.append(
            {
                "case_target": target,
                "selected_layer": config["direction3"]["pre_registered_cases"][target],
                "variant_metrics": variant_metrics,
                "target_accuracy_delta_b_minus_a": target_delta,
                "non_target_accuracy_delta_b_minus_a": non_target_delta,
                "paired_lpips_mean": mean_lpips,
                "paired_lpips_median": statistics.median(case_lpips),
                "paired_lpips_min": min(case_lpips),
                "paired_lpips_max": max(case_lpips),
                "meaningful_image_level_difference": meaningful,
            }
        )
    meaningful_count = sum(
        row["meaningful_image_level_difference"] for row in case_summaries
    )
    decision = (
        "GO"
        if meaningful_count >= int(gate["required_meaningful_cases"])
        else "NO-GO"
    )
    payload = {
        "decision": decision,
        "reason": f"{meaningful_count}/3 pre-registered cases met the image-difference gate",
        "classifier": config["direction3"]["classifier"],
        "classifier_limitation": (
            "CLIP ViT-B/32 is a closed-set, uncalibrated classifier; paired accuracy "
            "and probability outputs are interpreted only as a smoke qualification."
        ),
        "lpips": config["direction3"]["lpips"],
        "gate": gate,
        "meaningful_case_count": meaningful_count,
        "case_summaries": case_summaries,
        "classification_items": classifications,
        "lpips_items": lpips_rows,
    }
    write_json(output_dir / "d3" / "metrics.json", payload)
    mark_stage(output_dir, stage, {"decision": decision, "evaluated_images": len(rows)})


def format_float(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def stage_report(config: Mapping[str, Any], output_dir: Path) -> None:
    require_stage(output_dir, "d1-composition-evaluate")
    require_stage(output_dir, "d3-evaluate")
    stage = "report"
    if stage_is_complete(output_dir, stage):
        print(f"[skip] {stage} already complete")
        return
    operator = read_json(output_dir / "operator" / "summary.json")
    canonical = read_json(output_dir / "d1" / "canonical_metrics.json")
    composition = read_json(output_dir / "d1" / "composition_metrics.json")
    d3 = read_json(output_dir / "d3" / "metrics.json")
    construction = read_json(output_dir / "operator" / "d3_case_construction.json")

    canonical_lines = [
        "| Target | Original detection | Official OCE detection | Drop | Qualified |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in canonical["summaries"]:
        canonical_lines.append(
            "| {target} | {original} | {edited} | {drop} | {qualified} |".format(
                target=row["target"],
                original=format_float(row["original_target_detection_accuracy"]),
                edited=format_float(row["official_oce_target_detection_accuracy"]),
                drop=format_float(row["target_detection_drop"]),
                qualified="yes" if row["canonical_erasure_qualified"] else "no",
            )
        )
    composition_lines = [
        "| Edited target | Original two-object | Official OCE | Drop | Degraded prompt families | Stable failure |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in composition["target_summaries"]:
        if not row["composition_evaluated"]:
            composition_lines.append(
                f"| {row['target']} | — | — | — | — | no (canonical unqualified) |"
            )
            continue
        composition_lines.append(
            "| {target} | {original} | {edited} | {drop} | {families}/6 | {stable} |".format(
                target=row["target"],
                original=format_float(row["original_two_object_accuracy"]),
                edited=format_float(row["official_oce_two_object_accuracy"]),
                drop=format_float(row["two_object_accuracy_drop"]),
                families=row["degraded_prompt_families"],
                stable="yes" if row["stable_compositional_degradation"] else "no",
            )
        )
    d3_lines = [
        "| Case | Layer | Target acc A/B | Non-target acc A/B | Mean paired LPIPS | Meaningful |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in d3["case_summaries"]:
        a = row["variant_metrics"]["realization_a"]
        b = row["variant_metrics"]["realization_b"]
        d3_lines.append(
            "| {target} | `{layer}` | {ta}/{tb} | {na}/{nb} | {lpips} | {meaningful} |".format(
                target=row["case_target"],
                layer=row["selected_layer"],
                ta=format_float(a["target_classification_accuracy"]),
                tb=format_float(b["target_classification_accuracy"]),
                na=format_float(a["non_target_classification_accuracy"]),
                nb=format_float(b["non_target_classification_accuracy"]),
                lpips=format_float(row["paired_lpips_mean"]),
                meaningful="yes" if row["meaningful_image_level_difference"] else "no",
            )
        )
    mixing = operator["direction1_head_mixing"]
    orth = operator["direction3_orthogonality"]
    mixing_target_medians = ", ".join(
        f"{target}={value:.6f}"
        for target, value in mixing["target_medians"].items()
    )
    construction_diffs = ", ".join(
        f"{row['target']}={row['edited_weight_relative_difference']:.6f}"
        for row in construction
    )
    report = f"""# OCE failure image qualification

No OCE method or determinant correction was modified.

## Direction 1: {composition['decision']}

The per-head source-coordinate leakage statistic `m_g` was computed for all
{mixing['rows']} target/layer/head rows: median
`{mixing['median_m_g']:.6f}`, range
`[{mixing['minimum_m_g']:.6f}, {mixing['maximum_m_g']:.6f}]`. Target medians:
{mixing_target_medians}.

### Canonical target erasure

{chr(10).join(canonical_lines)}

### GenEval-style two-object smoke

Primary evaluator: COCO Faster R-CNN v2 at score threshold
`{config['direction1']['detector']['score_threshold']}`. Both named objects must
be detected for a correct image. Prompts and seeds are paired across Original
SD and official OCE.

{chr(10).join(composition_lines)}

Decision reason: {composition['reason']}.

## Direction 3: {d3['decision']}

Across official rotations, maximum `||R^T R-I||_F` was
`{orth['maximum_rt_r_minus_i_fro']:.6e}` and maximum relative residual was
`{orth['maximum_rt_r_minus_i_relative']:.6e}`. Rotation singular values ranged
from `{orth['minimum_rotation_singular_value']:.9f}` to
`{orth['maximum_rotation_singular_value']:.9f}`. For triggered rows, the raw
float32 determinant range was
`[{orth['triggered_float32_determinant_range'][0]:.6f}, {orth['triggered_float32_determinant_range'][1]:.6f}]`;
re-evaluating the same stored rotations on CPU float64 gave
`[{orth['triggered_cpu_float64_determinant_range'][0]:.6f}, {orth['triggered_cpu_float64_determinant_range'][1]:.6f}]`.

The three cases were pre-registered by official module order, not effect size.
All non-selected layers are bitwise identical between each A/B checkpoint.
Operator-level A/B edited-weight relative differences: {construction_diffs}.

{chr(10).join(d3_lines)}

Decision reason: {d3['reason']}.

This was a tiny fixed-prompt, fixed-seed qualification smoke. The detector and
closed-set CLIP classifier are automated readouts rather than human ground
truth. No expansion to head-local OCE or all triggered Direction 3 cases is
authorized unless the corresponding gate is GO.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    completion = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "direction1_decision": composition["decision"],
        "direction3_decision": d3["decision"],
        "no_oce_method_modification": True,
        "no_determinant_correction_fix": True,
        "report_sha256": sha256_file(output_dir / "report.md"),
        "canonical_metrics_sha256": sha256_file(
            output_dir / "d1" / "canonical_metrics.json"
        ),
        "composition_metrics_sha256": sha256_file(
            output_dir / "d1" / "composition_metrics.json"
        ),
        "direction3_metrics_sha256": sha256_file(output_dir / "d3" / "metrics.json"),
    }
    write_json(output_dir / "completion.json", completion)
    mark_stage(output_dir, stage, completion)
    print(report)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    config = initialize(args.config, output_dir)
    dispatch = {
        "preflight": lambda: stage_preflight(config, output_dir, args.allow_network),
        "prepare": lambda: stage_prepare(config, output_dir, args.allow_network),
        "d1-canonical-generate": lambda: stage_d1_canonical_generate(
            config, output_dir, args.allow_network
        ),
        "d1-canonical-evaluate": lambda: stage_d1_canonical_evaluate(
            config, output_dir
        ),
        "d1-composition-generate": lambda: stage_d1_composition_generate(
            config, output_dir, args.allow_network
        ),
        "d1-composition-evaluate": lambda: stage_d1_composition_evaluate(
            config, output_dir
        ),
        "d3-generate": lambda: stage_d3_generate(
            config, output_dir, args.allow_network
        ),
        "d3-evaluate": lambda: stage_d3_evaluate(config, output_dir),
        "report": lambda: stage_report(config, output_dir),
    }
    dispatch[args.stage]()


if __name__ == "__main__":
    main()
