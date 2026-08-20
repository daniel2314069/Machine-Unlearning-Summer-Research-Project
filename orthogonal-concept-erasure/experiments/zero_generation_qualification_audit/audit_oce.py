"""Zero-generation qualification audits for the official OCE implementation.

This script loads the official SD 1.4 OCE inputs and evaluates the exact
closed-form edit path in ``oce.py``.  It never calls the UNet forward pass or a
diffusion pipeline generation method, and it does not save edited weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import diffusers
import torch
import transformers
from diffusers import DiffusionPipeline


TARGETS = ("airplane", "bird", "dog", "truck")
MODEL_ID = "CompVis/stable-diffusion-v1-4"
GUIDE = "sky"
PRESERVE = "sky"
ERASE_SCALE = 2000.0
PRESERVE_GLOBAL_SCALE = 10.0
PRESERVE_CONCEPT_SCALE = 0.0
LAMBDA = 10.0

# Conservative qualification gates.  A direction only advances when the
# implementation-level effect is both substantial and stable.
D1_SUBSTANTIAL_OFFDIAG_FRACTION = 0.10
D1_REQUIRED_ROW_FRACTION = 0.75
D3_MIN_TRIGGER_FRACTION = 0.10
D3_MEANINGFUL_WEIGHT_RELATIVE_DIFFERENCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cg-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--max-float64-cases", type=int, default=2)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow model files not already present in the local HF cache.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def source_provenance(obj: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "class": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
    }
    try:
        source_file = inspect.getsourcefile(obj.__class__)
        source = inspect.getsource(obj.__class__)
    except (OSError, TypeError):
        source_file = None
        source = None
    result["source_file"] = source_file
    result["source_sha256"] = (
        hashlib.sha256(source.encode("utf-8")).hexdigest() if source else None
    )
    return result


def git_revision(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def verify_official_object_protocol(script_path: Path) -> dict[str, str]:
    expected = {
        "MODEL_ID": MODEL_ID,
        "GUIDE_CONCEPTS": GUIDE,
        "PRESERVE_CONCEPTS": PRESERVE,
        "CONCEPT_TYPE": "object",
        "ERASE_SCALE": "2000",
        "PRESERVE_GLOBAL_SCALE": "10",
        "PRESERVE_CONCEPT_SCALE": "0",
        "LAMB": "10",
    }
    assignments: dict[str, str] = {}
    pattern = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")
    for line in script_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        value = match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        assignments[match.group(1)] = value
    mismatches = {
        key: {"expected": value, "actual": assignments.get(key)}
        for key, value in expected.items()
        if assignments.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Audit constants no longer match trainscripts/object.sh: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if '--expand_prompts "true"' not in script_path.read_text(encoding="utf-8"):
        raise RuntimeError("Official object.sh no longer enables prompt expansion")
    return {key: assignments[key] for key in expected}


def expanded_object_prompts(target: str) -> tuple[list[str], list[str], list[str]]:
    edit = [target]
    guide = [GUIDE]
    preserve = [PRESERVE]
    edit.extend(
        [
            f"image of {target}",
            f"photo of {target}",
            f"portrait of {target}",
            f"picture of {target}",
            f"painting of {target}",
        ]
    )
    guide.extend(
        [
            f"image of {GUIDE}",
            f"photo of {GUIDE}",
            f"portrait of {GUIDE}",
            f"picture of {GUIDE}",
            f"painting of {GUIDE}",
        ]
    )
    return edit, guide, preserve


@torch.inference_mode()
def encode_prompts(
    pipe: DiffusionPipeline, prompts: list[str], device: torch.device
) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    for index, prompt in enumerate(prompts, start=1):
        print(f"[embed {index}/{len(prompts)}] {prompt!r}", flush=True)
        encoded = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        attention_mask = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["attention_mask"]
        last_index = int(attention_mask.sum().item()) - 2
        cache[prompt] = (
            encoded[0][:, last_index, :]
            .squeeze(0)
            .to(device=device, dtype=torch.float32)
        )
    return cache


def build_subspace(weight: torch.Tensor, embeddings: list[torch.Tensor]) -> torch.Tensor:
    vectors = []
    for embedding in embeddings:
        vector = weight @ embedding
        vector = vector / (vector.norm() + 1e-8)
        vectors.append(vector)
    stacked = torch.stack(vectors, dim=1)
    orthogonal, _ = torch.linalg.qr(stacked, mode="reduced")
    return orthogonal


def collect_layers(unet: torch.nn.Module) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layers: list[dict[str, Any]] = []
    implementation_records: dict[tuple[str, str | None], dict[str, Any]] = {}
    for name, module in unet.named_modules():
        if "attn2" not in name or not name.endswith("to_v"):
            continue
        parent_name = name.rsplit(".", 1)[0]
        attention = unet.get_submodule(parent_name)
        weight = module.weight.detach()
        out_dim, in_dim = map(int, weight.shape)
        heads = int(attention.heads)
        inner_dim = int(attention.inner_dim)
        if inner_dim != out_dim:
            raise RuntimeError(
                f"{name}: attention.inner_dim={inner_dim}, to_v.out_dim={out_dim}"
            )
        if out_dim % heads:
            raise RuntimeError(f"{name}: output dimension {out_dim} not divisible by {heads}")
        head_dim = out_dim // heads
        processor = attention.processor
        attn_source = source_provenance(attention)
        proc_source = source_provenance(processor)
        implementation_records[(attn_source["class"], attn_source["source_sha256"])] = (
            attn_source
        )
        implementation_records[(proc_source["class"], proc_source["source_sha256"])] = (
            proc_source
        )
        layers.append(
            {
                "name": name,
                "module": module,
                "attention": attention,
                "shape": [out_dim, in_dim],
                "head_count": heads,
                "head_dim": head_dim,
                "runtime_attention_inner_dim": inner_dim,
                "runtime_attention_heads": heads,
                "runtime_attention_class": attn_source["class"],
                "runtime_processor_class": proc_source["class"],
            }
        )
    if not layers:
        raise RuntimeError("No attn2.to_v layers were found")
    return layers, list(implementation_records.values())


def numerical_rank(
    singular_values: torch.Tensor, rows: int, columns: int, eps: float
) -> tuple[int, float]:
    tolerance = max(rows, columns) * eps * float(singular_values[0].item())
    rank = int((singular_values > tolerance).sum().item())
    return rank, tolerance


def relative_frobenius(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.matrix_norm(reference).clamp_min(1e-30)
    return float((torch.linalg.matrix_norm(actual - reference) / denominator).item())


def official_correction(raw_rotation: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
    # This intentionally reproduces oce.py exactly.  In particular, it flips
    # the last *column of R*, rather than modifying an SVD factor.
    determinant = float(torch.det(raw_rotation).item())
    corrected = raw_rotation.clone()
    triggered = determinant < 0
    if triggered:
        corrected[:, -1] *= -1
    return corrected, determinant, triggered


def cross_head_metrics(
    rotation: torch.Tensor, heads: int, head_dim: int
) -> dict[str, Any]:
    dimension = int(rotation.shape[0])
    delta = rotation - torch.eye(dimension, device=rotation.device, dtype=rotation.dtype)
    blocks = delta.reshape(heads, head_dim, heads, head_dim).permute(0, 2, 1, 3)
    block_energy = (blocks * blocks).sum(dim=(-1, -2))
    total_energy = float(block_energy.sum().item())
    diagonal_energy = float(torch.diagonal(block_energy).sum().item())
    off_diagonal_energy = max(0.0, total_energy - diagonal_energy)
    denominator = max(total_energy, 1e-30)
    return {
        "r_minus_i_fro": math.sqrt(max(total_energy, 0.0)),
        "total_update_energy_sq": total_energy,
        "diagonal_block_update_energy_sq": diagonal_energy,
        "off_diagonal_cross_head_update_energy_sq": off_diagonal_energy,
        "diagonal_block_update_fro": math.sqrt(max(diagonal_energy, 0.0)),
        "off_diagonal_cross_head_update_fro": math.sqrt(
            max(off_diagonal_energy, 0.0)
        ),
        "off_diagonal_energy_fraction": off_diagonal_energy / denominator,
        "block_energy_sq": block_energy.detach().cpu().tolist(),
    }


def float64_nullspace_audit(
    *,
    target: str,
    layer: dict[str, Any],
    objective: torch.Tensor,
    weight: torch.Tensor,
    official_rotation: torch.Tensor,
) -> dict[str, Any]:
    print(f"[float64 CPU] {target} / {layer['name']}", flush=True)
    objective64 = objective.detach().to(device="cpu", dtype=torch.float64)
    weight64 = weight.detach().to(device="cpu", dtype=torch.float64)
    started = time.time()
    u64, singular64, vh64 = torch.linalg.svd(objective64, full_matrices=False)
    elapsed = time.time() - started

    native_rank, native_tolerance = numerical_rank(
        singular64,
        objective64.shape[0],
        objective64.shape[1],
        torch.finfo(torch.float64).eps,
    )
    protocol_rank, protocol_tolerance = numerical_rank(
        singular64,
        objective64.shape[0],
        objective64.shape[1],
        torch.finfo(torch.float32).eps,
    )
    if protocol_rank >= objective64.shape[0]:
        raise RuntimeError("Selected float64 case has no float32-resolution null space")

    raw64 = u64 @ vh64
    corrected64, determinant64, triggered64 = official_correction(raw64)
    official_cpu = official_rotation.detach().to(device="cpu", dtype=torch.float64)
    weight_official_gpu = official_cpu @ weight64
    weight_cpu64 = corrected64 @ weight64

    # Form the numerical rank-r objective.  Flipping one column of U in its
    # zero-singular-value subspace yields a second exact SVD realization of the
    # same truncated matrix.  The two decompositions reconstruct that same
    # matrix, while U@Vh is allowed to differ on the null space.
    singular_truncated = singular64.clone()
    singular_truncated[protocol_rank:] = 0
    objective_truncated = (u64 * singular_truncated.unsqueeze(0)) @ vh64
    null_index = protocol_rank
    u_alternative = u64.clone()
    u_alternative[:, null_index] *= -1
    alternative_reconstruction = (
        u_alternative * singular_truncated.unsqueeze(0)
    ) @ vh64
    raw_alternative = u_alternative @ vh64
    corrected_alternative, determinant_alternative, triggered_alternative = (
        official_correction(raw_alternative)
    )

    raw_weight64 = raw64 @ weight64
    raw_weight_alternative = raw_alternative @ weight64
    corrected_weight_alternative = corrected_alternative @ weight64
    objective_norm = torch.linalg.matrix_norm(objective64).clamp_min(1e-30)

    return {
        "target": target,
        "layer": layer["name"],
        "shape": layer["shape"],
        "elapsed_seconds": elapsed,
        "float64_native_rank": native_rank,
        "float64_native_nullity": int(objective64.shape[0] - native_rank),
        "float64_native_tolerance": native_tolerance,
        "float32_resolution_rank_in_float64": protocol_rank,
        "float32_resolution_nullity_in_float64": int(
            objective64.shape[0] - protocol_rank
        ),
        "float32_resolution_tolerance_in_float64": protocol_tolerance,
        "rank_truncation_relative_to_objective": float(
            (torch.linalg.matrix_norm(objective_truncated - objective64) / objective_norm).item()
        ),
        "alternative_reconstruction_relative_to_same_truncated_objective": float(
            (
                torch.linalg.matrix_norm(
                    alternative_reconstruction - objective_truncated
                )
                / torch.linalg.matrix_norm(objective_truncated).clamp_min(1e-30)
            ).item()
        ),
        "flipped_null_index": null_index,
        "base_raw_determinant": determinant64,
        "base_correction_triggered": triggered64,
        "alternative_raw_determinant": determinant_alternative,
        "alternative_correction_triggered": triggered_alternative,
        "cpu64_vs_official_gpu_rotation_relative_difference": relative_frobenius(
            corrected64, official_cpu
        ),
        "cpu64_vs_official_gpu_weight_relative_difference": relative_frobenius(
            weight_cpu64, weight_official_gpu
        ),
        "legal_realizations_raw_rotation_relative_difference": relative_frobenius(
            raw_alternative, raw64
        ),
        "legal_realizations_raw_weight_relative_difference": relative_frobenius(
            raw_weight_alternative, raw_weight64
        ),
        "legal_realizations_corrected_rotation_relative_difference": relative_frobenius(
            corrected_alternative, corrected64
        ),
        "legal_realizations_corrected_weight_relative_difference": relative_frobenius(
            corrected_weight_alternative, weight_cpu64
        ),
        "base_rotation_sha256": tensor_sha256(corrected64),
        "alternative_rotation_sha256": tensor_sha256(corrected_alternative),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_direction1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fractions = [float(row["off_diagonal_energy_fraction"]) for row in rows]
    by_target: dict[str, list[float]] = defaultdict(list)
    by_shape: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_target[str(row["target"])].append(
            float(row["off_diagonal_energy_fraction"])
        )
        by_shape["x".join(map(str, row["layer_shape"]))].append(
            float(row["off_diagonal_energy_fraction"])
        )
    substantial_fraction = sum(
        value >= D1_SUBSTANTIAL_OFFDIAG_FRACTION for value in fractions
    ) / len(fractions)
    target_medians = {
        target: statistics.median(values) for target, values in sorted(by_target.items())
    }
    go = (
        substantial_fraction >= D1_REQUIRED_ROW_FRACTION
        and all(
            median >= D1_SUBSTANTIAL_OFFDIAG_FRACTION
            for median in target_medians.values()
        )
    )
    return {
        "decision": "GO" if go else "NO-GO",
        "qualification_rule": {
            "substantial_off_diagonal_energy_fraction": D1_SUBSTANTIAL_OFFDIAG_FRACTION,
            "required_fraction_of_rows": D1_REQUIRED_ROW_FRACTION,
            "each_target_median_must_be_substantial": True,
        },
        "rows": len(rows),
        "minimum": min(fractions),
        "median": statistics.median(fractions),
        "p95": percentile(fractions, 0.95),
        "maximum": max(fractions),
        "substantial_row_fraction": substantial_fraction,
        "target_medians": target_medians,
        "head_layouts": sorted(
            {
                (
                    "x".join(map(str, row["layer_shape"])),
                    int(row["head_count"]),
                    int(row["head_dim"]),
                )
                for row in rows
            }
        ),
        "shape_medians": {
            shape: statistics.median(values) for shape, values in sorted(by_shape.items())
        },
    }


def summarize_direction3(
    rows: list[dict[str, Any]], float64_cases: list[dict[str, Any]]
) -> dict[str, Any]:
    triggered = [row for row in rows if row["correction_triggered"]]
    rank_deficient = [row for row in rows if row["numerical_nullity"] > 0]
    triggered_rank_deficient = [
        row
        for row in rows
        if row["correction_triggered"] and row["numerical_nullity"] > 0
    ]
    trigger_fraction = len(triggered) / len(rows)
    maximum_correction_weight_effect = max(
        (
            float(row["correction_weight_relative_difference"])
            for row in triggered_rank_deficient
        ),
        default=0.0,
    )
    maximum_legal_realization_weight_effect = max(
        (
            float(case["legal_realizations_corrected_weight_relative_difference"])
            for case in float64_cases
        ),
        default=0.0,
    )
    target_trigger_counts = {
        target: {
            "triggered": sum(
                row["correction_triggered"] for row in rows if row["target"] == target
            ),
            "rows": sum(row["target"] == target for row in rows),
        }
        for target in sorted({str(row["target"]) for row in rows})
    }
    ranks_by_shape: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        shape = "x".join(map(str, row["objective_shape"]))
        ranks_by_shape[shape].append(
            (int(row["numerical_rank"]), int(row["numerical_nullity"]))
        )
    go = (
        trigger_fraction >= D3_MIN_TRIGGER_FRACTION
        and bool(triggered_rank_deficient)
        and maximum_correction_weight_effect
        >= D3_MEANINGFUL_WEIGHT_RELATIVE_DIFFERENCE
        and maximum_legal_realization_weight_effect
        >= D3_MEANINGFUL_WEIGHT_RELATIVE_DIFFERENCE
    )
    return {
        "decision": "GO" if go else "NO-GO",
        "qualification_rule": {
            "minimum_correction_trigger_fraction": D3_MIN_TRIGGER_FRACTION,
            "meaningful_weight_relative_difference": D3_MEANINGFUL_WEIGHT_RELATIVE_DIFFERENCE,
            "requires_rank_deficient_triggered_case": True,
            "requires_meaningful_legal_svd_realization_effect": True,
        },
        "rows": len(rows),
        "correction_trigger_count": len(triggered),
        "correction_trigger_fraction": trigger_fraction,
        "rank_deficient_count": len(rank_deficient),
        "rank_deficient_trigger_count": len(triggered_rank_deficient),
        "target_trigger_counts": target_trigger_counts,
        "rank_nullity_ranges_by_shape": {
            shape: {
                "rank_min": min(rank for rank, _ in values),
                "rank_max": max(rank for rank, _ in values),
                "nullity_min": min(nullity for _, nullity in values),
                "nullity_max": max(nullity for _, nullity in values),
            }
            for shape, values in sorted(ranks_by_shape.items())
        },
        "maximum_correction_weight_relative_difference": maximum_correction_weight_effect,
        "float64_case_count": len(float64_cases),
        "maximum_legal_realization_corrected_weight_relative_difference": (
            maximum_legal_realization_weight_effect
        ),
        "maximum_cpu64_vs_official_gpu_weight_relative_difference": max(
            (
                float(case["cpu64_vs_official_gpu_weight_relative_difference"])
                for case in float64_cases
            ),
            default=0.0,
        ),
    }


def render_report(
    d1: dict[str, Any], d3: dict[str, Any], float64_cases: list[dict[str, Any]]
) -> str:
    target_text = ", ".join(
        f"{target}={value:.6f}" for target, value in d1["target_medians"].items()
    )
    head_layout_text = ", ".join(
        f"{shape}: {heads} heads x {head_dim}"
        for shape, heads, head_dim in d1["head_layouts"]
    )
    target_trigger_text = ", ".join(
        f"{target}={counts['triggered']}/{counts['rows']}"
        for target, counts in d3["target_trigger_counts"].items()
    )
    rank_text = ", ".join(
        f"{shape}: rank {values['rank_min']}-{values['rank_max']}, "
        f"nullity {values['nullity_min']}-{values['nullity_max']}"
        for shape, values in d3["rank_nullity_ranges_by_shape"].items()
    )
    lines = [
        "# OCE zero-generation qualification audit",
        "",
        (
            f"1. **Direction 1: {d1['decision']}.** Off-diagonal update-energy "
            f"fraction across {d1['rows']} target/layer rows: median "
            f"{d1['median']:.6f}, p95 {d1['p95']:.6f}, range "
            f"[{d1['minimum']:.6f}, {d1['maximum']:.6f}]. The fraction of rows "
            f"at or above 0.10 is {d1['substantial_row_fraction']:.3f}. "
            f"Runtime head layouts: {head_layout_text}. Target medians: {target_text}."
        ),
        "",
        (
            f"2. **Direction 3: {d3['decision']}.** The official determinant "
            f"correction triggered in {d3['correction_trigger_count']}/{d3['rows']} "
            f"rows ({d3['correction_trigger_fraction']:.3f}); "
            f"per target: {target_trigger_text}. M rank/nullity: {rank_text}. "
            f"{d3['rank_deficient_trigger_count']} triggered rows were numerically "
            f"rank-deficient. Maximum correction-induced edited-weight relative "
            f"difference was {d3['maximum_correction_weight_relative_difference']:.6e}. "
            f"Across {d3['float64_case_count']} CPU float64 case(s), the maximum "
            f"difference between two legal numerical-null-space SVD realizations "
            f"after the official correction was "
            f"{d3['maximum_legal_realization_corrected_weight_relative_difference']:.6e}."
        ),
        "",
    ]
    go_directions = []
    if d1["decision"] == "GO":
        go_directions.append(
            "Direction 1: use one representative low/mid/high-resolution layer group "
            "and a tiny fixed-seed target-prompt set to compare the official edit with "
            "a head-local matched control."
        )
    if d3["decision"] == "GO":
        go_directions.append(
            "Direction 3: use the smallest triggered rank-deficient case and a tiny "
            "fixed-seed target/preservation prompt set to compare weights from two "
            "recorded legal SVD realizations."
        )
    if go_directions:
        lines.append("3. **Next minimum image-level qualification (not run):**")
        lines.append("")
        lines.extend(f"   - {item}" for item in go_directions)
    else:
        lines.append("3. **Next image-level qualification:** none; both directions failed the gate.")
    lines.extend(
        [
            "",
            "No images were generated. Full per-layer evidence is in "
            "`direction1_layers.csv`, `direction3_layers.csv`, and `audit_results.json`.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.max_float64_cases < 1:
        raise ValueError("--max-float64-cases must be at least 1")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cg_path = args.cg_path.resolve()
    if not cg_path.is_file():
        raise FileNotFoundError(f"Missing official Cg.pt: {cg_path}")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")

    script_path = Path(__file__).resolve()
    oce_root = script_path.parents[2]
    repo_root = oce_root.parent
    official_oce_path = oce_root / "oce.py"
    official_object_script = oce_root / "trainscripts" / "object.sh"
    verified_object_protocol = verify_official_object_protocol(official_object_script)
    device = torch.device(args.device)
    started = time.time()

    print(f"Loading {args.model_id} without VAE; no generation will be performed.", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
        local_files_only=not args.allow_network,
    ).to(device)
    layers, implementation_records = collect_layers(pipe.unet)

    prompt_sets = {
        target: expanded_object_prompts(target) for target in args.targets
    }
    all_prompts = sorted(
        {
            prompt
            for edit, guide, preserve in prompt_sets.values()
            for prompt in edit + guide + preserve
        }
    )
    embeddings = encode_prompts(pipe, all_prompts, device)
    cg_payload = torch.load(cg_path, map_location=device)
    cg = cg_payload["C"].to(device=device, dtype=torch.float32)
    if cg.ndim != 2 or cg.shape[0] != cg.shape[1]:
        raise RuntimeError(f"Cg matrix is not square: {tuple(cg.shape)}")

    manifest = {
        "audit_kind": "zero_generation_qualification",
        "generation_calls": 0,
        "official_oce_path": str(official_oce_path),
        "official_oce_sha256": sha256_file(official_oce_path),
        "official_object_script_path": str(official_object_script),
        "official_object_script_sha256": sha256_file(official_object_script),
        "verified_object_script_assignments": verified_object_protocol,
        "repository_git_revision": git_revision(repo_root),
        "model_id": args.model_id,
        "local_files_only": not args.allow_network,
        "targets": args.targets,
        "object_protocol": {
            "guide": GUIDE,
            "preserve": PRESERVE,
            "expand_prompts": True,
            "erase_scale": ERASE_SCALE,
            "preserve_global_scale": PRESERVE_GLOBAL_SCALE,
            "preserve_concept_scale": PRESERVE_CONCEPT_SCALE,
            "lambda": LAMBDA,
        },
        "cg": {
            "path": str(cg_path),
            "sha256": sha256_file(cg_path),
            "count": json_ready(cg_payload.get("count")),
            "shape": list(cg.shape),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "unet_config": json_ready(dict(pipe.unet.config)),
        "attention_implementation": implementation_records,
        "layers": [
            {key: value for key, value in layer.items() if key not in {"module", "attention"}}
            for layer in layers
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    d1_rows: list[dict[str, Any]] = []
    d3_rows: list[dict[str, Any]] = []
    float64_cases: list[dict[str, Any]] = []
    for target_index, target in enumerate(args.targets, start=1):
        edit_prompts, guide_prompts, preserve_prompts = prompt_sets[target]
        erase_embeddings = [embeddings[prompt] for prompt in edit_prompts]
        guide_embeddings = [embeddings[prompt] for prompt in guide_prompts]
        preserve_embeddings = [embeddings[prompt] for prompt in preserve_prompts]
        for layer_index, layer in enumerate(layers, start=1):
            print(
                f"[target {target_index}/{len(args.targets)}; layer "
                f"{layer_index}/{len(layers)}] {target} / {layer['name']}",
                flush=True,
            )
            weight = layer["module"].weight.detach().to(
                device=device, dtype=torch.float32
            )
            out_dim, in_dim = weight.shape
            guide_subspace = build_subspace(weight, guide_embeddings)
            guide_projector = guide_subspace @ guide_subspace.T
            erase_subspace = build_subspace(weight, erase_embeddings)
            erase_projector = erase_subspace @ erase_subspace.T
            identity = torch.eye(out_dim, device=device, dtype=torch.float32)
            objective = torch.zeros(
                out_dim, out_dim, device=device, dtype=torch.float32
            )
            objective += -ERASE_SCALE * erase_projector @ (
                identity - guide_projector
            )
            for preserve_embedding in preserve_embeddings:
                value = weight @ preserve_embedding
                objective += PRESERVE_CONCEPT_SCALE * (
                    value.unsqueeze(1) @ value.unsqueeze(0)
                )
            objective += PRESERVE_GLOBAL_SCALE * (weight @ cg @ weight.T)
            objective += LAMBDA * (weight @ weight.T)

            u, singular_values, vh = torch.linalg.svd(
                objective, full_matrices=False
            )
            raw_rotation = u @ vh
            final_rotation, raw_determinant, correction_triggered = (
                official_correction(raw_rotation)
            )
            raw_weight = raw_rotation @ weight
            final_weight = final_rotation @ weight
            rank, rank_tolerance = numerical_rank(
                singular_values,
                int(out_dim),
                int(out_dim),
                torch.finfo(torch.float32).eps,
            )

            head_metrics = cross_head_metrics(
                final_rotation, layer["head_count"], layer["head_dim"]
            )
            d1_rows.append(
                {
                    "target": target,
                    "layer": layer["name"],
                    "layer_shape": [int(out_dim), int(in_dim)],
                    "head_count": layer["head_count"],
                    "head_dim": layer["head_dim"],
                    "rotation_sha256": tensor_sha256(final_rotation),
                    **head_metrics,
                }
            )
            correction_rotation_fro = float(
                torch.linalg.matrix_norm(final_rotation - raw_rotation).item()
            )
            correction_weight_fro = float(
                torch.linalg.matrix_norm(final_weight - raw_weight).item()
            )
            d3_row = {
                "target": target,
                "layer": layer["name"],
                "objective_shape": [int(out_dim), int(out_dim)],
                "weight_shape": [int(out_dim), int(in_dim)],
                "numerical_rank": rank,
                "numerical_nullity": int(out_dim) - rank,
                "rank_tolerance": rank_tolerance,
                "largest_singular_value": float(singular_values[0].item()),
                "smallest_singular_value": float(singular_values[-1].item()),
                "raw_determinant": raw_determinant,
                "correction_triggered": correction_triggered,
                "correction_rotation_fro": correction_rotation_fro,
                "correction_rotation_relative_difference": (
                    correction_rotation_fro
                    / float(torch.linalg.matrix_norm(raw_rotation).item())
                ),
                "correction_weight_fro": correction_weight_fro,
                "correction_weight_relative_difference": (
                    correction_weight_fro
                    / float(torch.linalg.matrix_norm(raw_weight).item())
                ),
                "objective_sha256": tensor_sha256(objective),
                "raw_rotation_sha256": tensor_sha256(raw_rotation),
                "final_rotation_sha256": tensor_sha256(final_rotation),
                "raw_weight_sha256": tensor_sha256(raw_weight),
                "final_weight_sha256": tensor_sha256(final_weight),
            }
            d3_rows.append(d3_row)

            if (
                len(float64_cases) < args.max_float64_cases
                and correction_triggered
                and rank < int(out_dim)
            ):
                float64_cases.append(
                    float64_nullspace_audit(
                        target=target,
                        layer=layer,
                        objective=objective,
                        weight=weight,
                        official_rotation=final_rotation,
                    )
                )

            del u, vh, raw_rotation, final_rotation, raw_weight, final_weight
            del objective, singular_values, guide_subspace, guide_projector
            del erase_subspace, erase_projector
            if device.type == "cuda":
                torch.cuda.empty_cache()

    d1_summary = summarize_direction1(d1_rows)
    d3_summary = summarize_direction3(d3_rows, float64_cases)
    result = {
        "manifest": manifest,
        "direction1_summary": d1_summary,
        "direction3_summary": d3_summary,
        "direction1_layers": d1_rows,
        "direction3_layers": d3_rows,
        "float64_nullspace_cases": float64_cases,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "audit_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_csv(
        output_dir / "direction1_layers.csv",
        d1_rows,
        [
            "target",
            "layer",
            "layer_shape",
            "head_count",
            "head_dim",
            "r_minus_i_fro",
            "total_update_energy_sq",
            "diagonal_block_update_energy_sq",
            "off_diagonal_cross_head_update_energy_sq",
            "diagonal_block_update_fro",
            "off_diagonal_cross_head_update_fro",
            "off_diagonal_energy_fraction",
            "rotation_sha256",
        ],
    )
    write_csv(
        output_dir / "direction3_layers.csv",
        d3_rows,
        [
            "target",
            "layer",
            "objective_shape",
            "weight_shape",
            "numerical_rank",
            "numerical_nullity",
            "rank_tolerance",
            "largest_singular_value",
            "smallest_singular_value",
            "raw_determinant",
            "correction_triggered",
            "correction_rotation_fro",
            "correction_rotation_relative_difference",
            "correction_weight_fro",
            "correction_weight_relative_difference",
            "objective_sha256",
            "raw_rotation_sha256",
            "final_rotation_sha256",
            "raw_weight_sha256",
            "final_weight_sha256",
        ],
    )
    (output_dir / "report.md").write_text(
        render_report(d1_summary, d3_summary, float64_cases), encoding="utf-8"
    )
    print(json.dumps({"direction1": d1_summary, "direction3": d3_summary}, indent=2))
    print(f"Saved zero-generation audit to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
