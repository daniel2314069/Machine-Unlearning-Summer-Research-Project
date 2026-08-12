#!/usr/bin/env python3
"""Run AFR matrix QA and, conditionally, the frozen balls image smoke.

This is an experiment-local editor and ablation runner.  It does not modify
production oce.py or any existing checkpoint/result namespace.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import pipeline  # noqa: E402
import protocol  # noqa: E402
import solver_audit as prior_audit  # noqa: E402
from core import (  # noqa: E402
    AFRError,
    build_afr_transforms,
    preservation_loss,
    procrustes,
    rank_basis,
    scalar,
    sqnorm,
    structural_qa,
    synthetic_unit_tests,
    transform_metrics,
)


def _load_checkpoint_builder() -> Any:
    """Load the parent runner under a collision-proof module name."""
    module_name = "_confuse5_afr_checkpoint_builder"
    source = EXPERIMENT_ROOT / "run.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load checkpoint builder from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


checkpoint_builder = _load_checkpoint_builder()


VARIANT_C = "C_objective_faithful"
VARIANT_F = "F_pure_residual_projection"
VARIANT_G = "G_full_afr"
VARIANTS = (VARIANT_C, VARIANT_F, VARIANT_G)
EXPECTED_GROUPS = ("dogs", "fruits", "balls")
EXPECTED_CASES = 48
EXPECTED_LAYERS = 16
EXPECTED_TARGET_RANK = 12
ALPHA = 1.0

ZERO_TOLERANCE = 1e-10
C_REPRODUCTION_TOLERANCE = 1e-6
PRESERVATION_OPTIMALITY_TOLERANCE = 1e-10
COMPENSATION_NONZERO_TOLERANCE = 1e-8
MEASURABLE_IMPROVEMENT_TOLERANCE = 1e-10
MEASURABLE_LAYER_COUNT = 12
MATERIAL_PRESERVATION_IMPROVEMENT = 0.01

TARGET_SMOKE_IMPROVEMENT = 0.05
PRESERVATION_COLLAPSE_TOLERANCE = 0.10
IMAGE_ABLATION_ACCURACY_BENEFIT = 0.05
IMAGE_ABLATION_LPIPS_BENEFIT = 0.02

FROZEN_CONFIG_SHA256 = (
    "416ad7fd9e7666f8cd295ef6de4c6cf6af26d67502fd21d4027b6a81aa7e762b"
)
FROZEN_ANCHORS_SHA256 = (
    "392802a728f5f726870718c2f9d0885ed57c8e16232899710f22690cee6c13b1"
)
FROZEN_QUALIFICATION_SHA256 = (
    "b96514c7bd94e4d703079ddff238731c05ba76cde7fe6b0c2643d404bf7f043f"
)
FROZEN_K0_SHA256 = (
    "5d8ee50935eb3d22e1f9bc84947572afba386d05e063f70ea161c5cbf1e16235"
)
FROZEN_EXACT_RESULTS_SHA256 = (
    "546789d6d939cccfa57551994630be7e93ebc4ff0a05527d68dfe32ba5722343"
)

OUTPUT_ROOT = HERE / "outputs" / "afr_balls_smoke_v1"
MATRIX_CSV = OUTPUT_ROOT / "matrix" / "results_c_f_g.csv"
MATRIX_GATE = OUTPUT_ROOT / "matrix" / "gate.json"
MATRIX_REPORT = OUTPUT_ROOT / "REPORT.md"
IMAGE_ROOT = OUTPUT_ROOT / "balls_smoke"
IMAGE_SUMMARY = IMAGE_ROOT / "summary.json"

MATRIX_FIELDS = [
    "group",
    "target_names",
    "anchor_names",
    "layer",
    "variant",
    "alpha",
    "target_rank",
    "anchor_rank",
    "residual_rank",
    "target_feature_leakage",
    "target_basis_leakage",
    "anchor_feature_error",
    "raw_frozen_s_distortion",
    "normalized_frozen_s_distortion",
    "f_minus_g_normalized_preservation",
    "compensation_magnitude",
    "compensation_orthogonality_residual",
    "anchor_pointwise_fix_residual",
    "anchor_residual_basis_orthogonality",
    "gram_residual",
    "projection_gram_match_residual",
    "compensation_trace_error",
]


class AFRRunError(RuntimeError):
    """Raised when protocol, artifact, or gate invariants fail."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("synthetic", "matrix", "all"), nargs="?", default="all"
    )
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def _sha_guard(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise AFRRunError(
            f"Frozen {label} hash mismatch: expected {expected}, got {actual}"
        )


def _atomic_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        ""
                        if row.get(field) is None
                        else f"{row[field]:.17g}"
                        if isinstance(row.get(field), float)
                        else row.get(field)
                    )
                    for field in fields
                }
            )
    temporary.replace(path)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise AFRRunError("Cannot average an empty collection")
    return float(statistics.fmean(materialized))


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise AFRRunError("Cannot take an empty median")
    return float(statistics.median(materialized))


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AFRRunError("Cannot take an empty quantile")
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8g}"


def _distribution(values: Sequence[float]) -> str:
    return " | ".join(
        _fmt(value)
        for value in (
            min(values),
            _quantile(values, 0.25),
            _median(values),
            _quantile(values, 0.75),
            max(values),
        )
    )


def _projected_columns(
    weight: Any, embeddings: Sequence[Any], normalization_eps: float
) -> Any:
    import torch

    columns = []
    for embedding in embeddings:
        vector = weight @ embedding
        columns.append(
            vector / (torch.linalg.vector_norm(vector) + normalization_eps)
        )
    return torch.stack(columns, dim=1)


def _preservation_matrix(
    weight: Any,
    retain_embeddings: Sequence[Any],
    k0: Any,
    oce: Mapping[str, Any],
) -> Any:
    import torch

    local = torch.zeros(
        weight.shape[0], weight.shape[0], device=weight.device, dtype=weight.dtype
    )
    for embedding in retain_embeddings:
        vector = weight @ embedding
        local.add_(float(oce["lambda_r"]) * torch.outer(vector, vector))
    preservation = (
        local
        + float(oce["lambda_0"]) * (weight @ k0 @ weight.T)
        + float(oce["lamb_repo_regularizer"]) * (weight @ weight.T)
    )
    symmetry = scalar(torch.linalg.matrix_norm(preservation - preservation.T))
    tolerance = (
        256.0
        * weight.shape[0]
        * torch.finfo(torch.float64).eps
        * max(1.0, scalar(torch.linalg.matrix_norm(preservation)))
    )
    if symmetry > tolerance:
        raise AFRRunError(
            f"Frozen S is not symmetric: residual={symmetry:.9g}, tolerance={tolerance:.9g}"
        )
    return 0.5 * (preservation + preservation.T)


def _load_frozen_c(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not path.is_file():
        raise AFRRunError(f"Frozen exact-control CSV is missing: {path}")
    _sha_guard("exact-control CSV", protocol.sha256(path), FROZEN_EXACT_RESULTS_SHA256)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("variant") == VARIANT_C
        ]
    if len(rows) != EXPECTED_CASES:
        raise AFRRunError(f"Expected 48 frozen C rows, found {len(rows)}")
    lookup = {}
    for row in rows:
        key = (row["group"], row["layer"])
        if key in lookup:
            raise AFRRunError(f"Duplicate frozen C key: {key}")
        lookup[key] = {
            "target_basis_leakage": float(row["true_leakage"]),
            "anchor_feature_error": float(row["anchor_feature_drift"]),
            "normalized_frozen_s_distortion": float(
                row["normalized_preservation_distortion"]
            ),
        }
    return lookup


def _validate_frozen_scope(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    list[dict[str, Any]],
    Path,
    dict[str, Any],
    Any,
]:
    _sha_guard("config", protocol.sha256(config_path), FROZEN_CONFIG_SHA256)
    config, anchors = protocol.load_protocol(config_path)
    _sha_guard(
        "anchors",
        protocol.sha256(Path(config["_resolved"]["anchors_path"])),
        FROZEN_ANCHORS_SHA256,
    )
    specs, _, qualification_path = prior_audit._validated_joint_specs(config, anchors)
    if tuple(spec["group_id"] for spec in specs) != EXPECTED_GROUPS:
        raise AFRRunError("Frozen AFR groups are not dogs/fruits/balls")
    _sha_guard(
        "qualification", protocol.sha256(qualification_path), FROZEN_QUALIFICATION_SHA256
    )
    plan, _, _ = checkpoint_builder.build_plan(config_path)
    k0_matrix, k0_metadata = checkpoint_builder.validate_k0(plan, config)
    _sha_guard("K0", k0_metadata["artifact_sha256"], FROZEN_K0_SHA256)
    balls = next(spec for spec in specs if spec["group_id"] == "balls")
    if balls["targets"] != ["soccer ball", "volleyball"]:
        raise AFRRunError("Frozen balls targets changed")
    if balls["anchors"] != ["basketball", "baseball"]:
        raise AFRRunError("Frozen matched balls anchors changed")
    return config, anchors, specs, qualification_path, k0_metadata, k0_matrix


def _matrix_row(
    common: Mapping[str, Any],
    variant: str,
    metrics: Mapping[str, float],
    *,
    residual_rank: int | None,
    f_minus_g: float | None,
    qa: Mapping[str, float] | None,
    trace_error: float | None,
) -> dict[str, Any]:
    return {
        **common,
        "variant": variant,
        "alpha": ALPHA if variant in (VARIANT_F, VARIANT_G) else None,
        "residual_rank": residual_rank,
        **metrics,
        "f_minus_g_normalized_preservation": f_minus_g,
        "compensation_magnitude": qa.get("compensation_magnitude") if qa else None,
        "compensation_orthogonality_residual": qa.get(
            "compensation_orthogonality_residual"
        )
        if qa
        else None,
        "anchor_pointwise_fix_residual": qa.get("anchor_pointwise_fix_residual")
        if qa
        else None,
        "anchor_residual_basis_orthogonality": qa.get(
            "anchor_residual_basis_orthogonality"
        )
        if qa
        else None,
        "gram_residual": qa.get("gram_residual") if qa else None,
        "projection_gram_match_residual": qa.get("projection_gram_match_residual")
        if qa
        else None,
        "compensation_trace_error": trace_error,
    }


def _paired(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["group"], row["layer"], row["variant"]): row for row in rows
    }
    keys = sorted({(row["group"], row["layer"]) for row in rows})
    pairs = [
        {
            "group": group,
            "layer": layer,
            "c": lookup[(group, layer, VARIANT_C)],
            "f": lookup[(group, layer, VARIANT_F)],
            "g": lookup[(group, layer, VARIANT_G)],
        }
        for group, layer in keys
    ]
    if len(pairs) != EXPECTED_CASES:
        raise AFRRunError(f"Expected 48 C/F/G cases, got {len(pairs)}")
    return pairs


def _classify_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    pairs = _paired(rows)
    exact_values = [
        float(pair[key][metric])
        for pair in pairs
        for key in ("f", "g")
        for metric in (
            "target_feature_leakage",
            "target_basis_leakage",
            "anchor_feature_error",
        )
    ]
    structural = [
        float(pair["g"][metric])
        for pair in pairs
        for metric in (
            "compensation_orthogonality_residual",
            "anchor_pointwise_fix_residual",
            "anchor_residual_basis_orthogonality",
            "gram_residual",
            "projection_gram_match_residual",
        )
    ]
    optimality_failures = [
        pair
        for pair in pairs
        if float(pair["g"]["normalized_frozen_s_distortion"])
        > float(pair["f"]["normalized_frozen_s_distortion"])
        + PRESERVATION_OPTIMALITY_TOLERANCE
    ]
    improvements = [
        float(pair["f"]["normalized_frozen_s_distortion"])
        - float(pair["g"]["normalized_frozen_s_distortion"])
        for pair in pairs
    ]
    magnitudes = [float(pair["g"]["compensation_magnitude"]) for pair in pairs]
    measurable = sum(value > MEASURABLE_IMPROVEMENT_TOLERANCE for value in improvements)
    nonzero = sum(value > COMPENSATION_NONZERO_TOLERANCE for value in magnitudes)
    material = sum(value >= MATERIAL_PRESERVATION_IMPROVEMENT for value in improvements)
    details = {
        "max_exact_guarantee_residual": max(exact_values),
        "max_structural_residual": max(structural),
        "optimality_failure_count": len(optimality_failures),
        "measurable_improvement_layers": measurable,
        "nonzero_compensation_layers": nonzero,
        "material_improvement_layers": material,
        "median_f_minus_g": _median(improvements),
        "median_compensation_magnitude": _median(magnitudes),
    }
    if (
        max(exact_values) > ZERO_TOLERANCE
        or max(structural) > ZERO_TOLERANCE
        or optimality_failures
    ):
        return "AFR-M0", details
    if nonzero == 0 or measurable < MEASURABLE_LAYER_COUNT:
        return "AFR-M1", details
    return "AFR-GO", details


def _matrix_table(pairs: Sequence[Mapping[str, Any]], aggregate: Any) -> str:
    lines = [
        "| Group | C leakage | F leakage | G leakage | F anchor | G anchor | C frozen-S | F frozen-S | G frozen-S | F-G | Compensation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in (*EXPECTED_GROUPS, "overall"):
        selected = (
            list(pairs)
            if group == "overall"
            else [pair for pair in pairs if pair["group"] == group]
        )
        fields = [
            group,
            _fmt(aggregate(float(p["c"]["target_feature_leakage"]) for p in selected)),
            _fmt(aggregate(float(p["f"]["target_feature_leakage"]) for p in selected)),
            _fmt(aggregate(float(p["g"]["target_feature_leakage"]) for p in selected)),
            _fmt(aggregate(float(p["f"]["anchor_feature_error"]) for p in selected)),
            _fmt(aggregate(float(p["g"]["anchor_feature_error"]) for p in selected)),
            _fmt(aggregate(float(p["c"]["normalized_frozen_s_distortion"]) for p in selected)),
            _fmt(aggregate(float(p["f"]["normalized_frozen_s_distortion"]) for p in selected)),
            _fmt(aggregate(float(p["g"]["normalized_frozen_s_distortion"]) for p in selected)),
            _fmt(aggregate(float(p["g"]["f_minus_g_normalized_preservation"]) for p in selected)),
            _fmt(aggregate(float(p["g"]["compensation_magnitude"]) for p in selected)),
        ]
        lines.append("| " + " | ".join(fields) + " |")
    return "\n".join(lines)


def _render_matrix_report(
    rows: Sequence[Mapping[str, Any]],
    synthetic: Mapping[str, Any],
    gate: Mapping[str, Any],
    run_info: Mapping[str, Any],
    image_section: str = "Image smoke has not run.",
) -> str:
    pairs = _paired(rows)
    improvements = [
        float(pair["g"]["f_minus_g_normalized_preservation"]) for pair in pairs
    ]
    f_distortion = [
        float(pair["f"]["normalized_frozen_s_distortion"]) for pair in pairs
    ]
    g_distortion = [
        float(pair["g"]["normalized_frozen_s_distortion"]) for pair in pairs
    ]
    magnitudes = [float(pair["g"]["compensation_magnitude"]) for pair in pairs]
    classification = gate["classification"]
    if classification == "AFR-M0":
        conclusion = "Implementation/algebra failed; no image was generated."
    elif classification == "AFR-M1":
        conclusion = (
            "Full compensation is algebraically redundant under the frozen objective; "
            "no image was generated."
        )
    else:
        conclusion = "All matrix gates passed; the conditional balls smoke is authorized."
    return f"""# AFR implementation, matrix QA, and conditional balls smoke

## Current result

**Matrix classification: {classification}.** {conclusion}

This experiment-local implementation leaves production `oce.py`, frozen Confuse5 settings, and all existing checkpoints untouched. F is the pure residual projection ablation; G is full AFR with anchor-fixed orthogonal compensation. Primary alpha is exactly `1`.

## Closed form

For `D=I-R_e` and `P=HH^T+H_perp Q H_perp^T`, expanding `tr[(PD-I)S(PD-I)^T]` shows the variable-dependent term is `-2 tr(P^T S D)`. The anchor-complement block is therefore `M_perp=H_perp^T S D H_perp`. If `M_perp=U Sigma V^T`, standard O(d) Procrustes gives `Q=UV^T`. No determinant correction is used.

## Synthetic QA

- alpha=0 explicit no-op: `{synthetic['alpha_zero_explicit_noop']}`, transform error `{_fmt(float(synthetic['alpha_zero_transform_error']))}`;
- alpha=1 F/G target leakage: `{_fmt(float(synthetic['alpha_one_projection_target_leakage']))}` / `{_fmt(float(synthetic['alpha_one_afr_target_leakage']))}`;
- alpha=1 F/G anchor error: `{_fmt(float(synthetic['alpha_one_projection_anchor_error']))}` / `{_fmt(float(synthetic['alpha_one_afr_anchor_error']))}`;
- normalized frozen-S F/G: `{_fmt(float(synthetic['projection_normalized_frozen_s_distortion']))}` / `{_fmt(float(synthetic['afr_normalized_frozen_s_distortion']))}`;
- compensation trace error `{_fmt(float(synthetic['compensation_trace_error']))}`, Gram residual `{_fmt(float(synthetic['gram_residual']))}`;
- closed-form AFR loss `{_fmt(float(synthetic['closed_form_optimal_loss']))}` versus best of 256 random feasible compensations `{_fmt(float(synthetic['best_of_256_random_feasible_losses']))}`.

## Group means

{_matrix_table(pairs, _mean)}

## Group medians

{_matrix_table(pairs, _median)}

## F versus G preservation ablation

- overall median F distortion: `{_fmt(_median(f_distortion))}`;
- overall median G distortion: `{_fmt(_median(g_distortion))}`;
- median F-G: `{_fmt(_median(improvements))}`;
- G < F in `{sum(value > MEASURABLE_IMPROVEMENT_TOLERANCE for value in improvements)}/48` layers at the numerical measurable threshold `{MEASURABLE_IMPROVEMENT_TOLERANCE}`;
- material F-G >= `{MATERIAL_PRESERVATION_IMPROVEMENT}` in `{sum(value >= MATERIAL_PRESERVATION_IMPROVEMENT for value in improvements)}/48` layers;
- compensation magnitude distribution (min/Q25/median/Q75/max): `{_distribution(magnitudes)}`.

## Layer distributions

| Quantity | Min | Q25 | Median | Q75 | Max |
|---|---:|---:|---:|---:|---:|
| F target feature leakage | {_distribution([float(p['f']['target_feature_leakage']) for p in pairs])} |
| G target feature leakage | {_distribution([float(p['g']['target_feature_leakage']) for p in pairs])} |
| F anchor feature error | {_distribution([float(p['f']['anchor_feature_error']) for p in pairs])} |
| G anchor feature error | {_distribution([float(p['g']['anchor_feature_error']) for p in pairs])} |
| F normalized frozen-S | {_distribution(f_distortion)} |
| G normalized frozen-S | {_distribution(g_distortion)} |
| F-G normalized frozen-S | {_distribution(improvements)} |
| G Gram residual | {_distribution([float(p['g']['gram_residual']) for p in pairs])} |

The anchor guarantee means **exact preservation of the constrained anchor features at the edited layer** only; it is not a claim of invariant anchor generation.

## Matrix gate

- F/G exact leakage and anchor error: maximum `{_fmt(float(gate['max_exact_guarantee_residual']))}`;
- structural QA maximum residual: `{_fmt(float(gate['max_structural_residual']))}`;
- G worse than F preservation: `{gate['optimality_failure_count']}/48` layers;
- nonzero compensation: `{gate['nonzero_compensation_layers']}/48` layers;
- measurable preservation improvement: `{gate['measurable_improvement_layers']}/48` layers;
- material preservation improvement: `{gate['material_improvement_layers']}/48` layers.

## Balls image smoke

{image_section}

## Reproducibility

- Matrix CSV: `{MATRIX_CSV.name}` (144 rows = 48 cases x C/F/G)
- GPU command: `{run_info['command']}`
- Git commit: `{run_info['git_hash']}`; dirty at matrix start: `{run_info['git_dirty']}`
- Config SHA-256: `{run_info['config_sha256']}`
- Anchors SHA-256: `{run_info['anchors_sha256']}`
- Qualification SHA-256: `{run_info['qualification_sha256']}`
- K0 SHA-256: `{run_info['k0_sha256']}`
- Frozen Variant C CSV SHA-256: `{run_info['frozen_c_sha256']}`
- Matrix dtype: `torch.float64`; checkpoint dtype: `{run_info['checkpoint_dtype']}`
- Matrix runtime: `{run_info['matrix_runtime_seconds']:.1f}` seconds
"""


def run_matrix(
    config_path: Path, output_root: Path
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    if platform.system() == "Darwin":
        raise AFRRunError("Full AFR matrix QA is forbidden on the local Mac")
    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        raise AFRRunError("Full AFR matrix QA requires the configured GPU server")
    started = time.monotonic()
    synthetic = synthetic_unit_tests()
    config, anchors, specs, qualification_path, k0_metadata, k0_matrix = (
        _validate_frozen_scope(config_path)
    )
    frozen_path = (
        EXPERIMENT_ROOT
        / "solver_audit"
        / "exact_orthogonal_control"
        / "results_exact_control.csv"
    )
    frozen_c = _load_frozen_c(frozen_path)

    device = str(config["model"]["device"])
    pipe = DiffusionPipeline.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)
    modules = checkpoint_builder._projection_modules(pipe.unet)
    if len(modules) != EXPECTED_LAYERS:
        raise AFRRunError(f"Expected 16 edited layers, found {len(modules)}")

    prompt_order: list[str] = []
    for spec in specs:
        prompt_order.extend(protocol.expanded_prompts(spec["targets"], config))
        prompt_order.extend(protocol.expanded_prompts(spec["anchors"], config))
        prompt_order.extend(spec["retain_concepts"])
    prompt_order = list(dict.fromkeys(prompt_order))
    encoded = {
        prompt: checkpoint_builder._encode_prompt(
            pipe, prompt, device, torch.float32
        )
        for prompt in prompt_order
    }
    embeddings = {
        key: value.to(device=device, dtype=torch.float64)
        for key, value in encoded.items()
    }
    k0 = k0_matrix.to(device=device, dtype=torch.float64)
    oce = config["oce"]
    production_dtype = getattr(torch, str(config["model"]["editing_dtype"]))
    rows: list[dict[str, Any]] = []
    balls_states: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    max_c_difference = 0.0

    with torch.inference_mode():
        for spec in specs:
            target_prompts = protocol.expanded_prompts(spec["targets"], config)
            anchor_prompts = protocol.expanded_prompts(spec["anchors"], config)
            target_embeddings = [embeddings[value] for value in target_prompts]
            anchor_embeddings = [embeddings[value] for value in anchor_prompts]
            retain_embeddings = [embeddings[value] for value in spec["retain_concepts"]]
            for layer_index, (layer, module) in enumerate(modules, start=1):
                weight = module.weight.detach().to(device=device, dtype=torch.float64)
                target_features = torch.stack(
                    [weight @ embedding for embedding in target_embeddings], dim=1
                )
                anchor_features = torch.stack(
                    [weight @ embedding for embedding in anchor_embeddings], dim=1
                )
                target_columns = _projected_columns(
                    weight, target_embeddings, float(oce["normalization_eps"])
                )
                target_basis, target_rank, _, _ = rank_basis(target_columns)
                if target_rank != EXPECTED_TARGET_RANK:
                    raise AFRRunError(
                        f"Unexpected target rank at {spec['group_id']} {layer}: {target_rank}"
                    )
                preservation = _preservation_matrix(weight, retain_embeddings, k0, oce)
                parts = build_afr_transforms(
                    target_features,
                    anchor_features,
                    preservation,
                    alpha=ALPHA,
                )
                if parts["anchor_rank"] != EXPECTED_TARGET_RANK:
                    raise AFRRunError(
                        f"Unexpected anchor rank at {spec['group_id']} {layer}: "
                        f"{parts['anchor_rank']}"
                    )
                identity = torch.eye(
                    weight.shape[0], device=device, dtype=torch.float64
                )
                target_projector = target_basis @ target_basis.T
                anchor_projector = parts["anchor_basis"] @ parts["anchor_basis"].T
                faithful_matrix = (
                    -float(oce["lambda_e"])
                    * (identity - anchor_projector)
                    @ target_projector
                    + preservation
                )
                faithful, _ = procrustes(faithful_matrix)
                transforms = {
                    VARIANT_C: faithful,
                    VARIANT_F: parts["T_projection"],
                    VARIANT_G: parts["T_afr"],
                }
                metrics = {
                    variant: transform_metrics(
                        transform,
                        target_features,
                        target_basis,
                        anchor_features,
                        parts["anchor_basis"],
                        preservation,
                    )
                    for variant, transform in transforms.items()
                }
                frozen = frozen_c[(spec["group_id"], layer)]
                for field in (
                    "target_basis_leakage",
                    "anchor_feature_error",
                    "normalized_frozen_s_distortion",
                ):
                    difference = abs(metrics[VARIANT_C][field] - frozen[field])
                    max_c_difference = max(max_c_difference, difference)
                    if difference > C_REPRODUCTION_TOLERANCE:
                        raise AFRRunError(
                            f"Variant C reproduction failed at {spec['group_id']} "
                            f"{layer} {field}: difference={difference:.9g}"
                        )
                qa = structural_qa(parts)
                improvement = (
                    metrics[VARIANT_F]["normalized_frozen_s_distortion"]
                    - metrics[VARIANT_G]["normalized_frozen_s_distortion"]
                )
                common = {
                    "group": spec["group_id"],
                    "target_names": json.dumps(spec["targets"], ensure_ascii=False),
                    "anchor_names": json.dumps(spec["anchors"], ensure_ascii=False),
                    "layer": layer,
                    "target_rank": target_rank,
                    "anchor_rank": parts["anchor_rank"],
                }
                rows.extend(
                    [
                        _matrix_row(
                            common,
                            VARIANT_C,
                            metrics[VARIANT_C],
                            residual_rank=None,
                            f_minus_g=None,
                            qa=None,
                            trace_error=None,
                        ),
                        _matrix_row(
                            common,
                            VARIANT_F,
                            metrics[VARIANT_F],
                            residual_rank=parts["residual_rank"],
                            f_minus_g=improvement,
                            qa=qa,
                            trace_error=parts["compensation_trace_error"],
                        ),
                        _matrix_row(
                            common,
                            VARIANT_G,
                            metrics[VARIANT_G],
                            residual_rank=parts["residual_rank"],
                            f_minus_g=improvement,
                            qa=qa,
                            trace_error=parts["compensation_trace_error"],
                        ),
                    ]
                )
                if spec["group_id"] == "balls":
                    key = f"{layer}.weight"
                    for variant, transform in transforms.items():
                        balls_states[variant][key] = (
                            transform @ weight
                        ).to(dtype=production_dtype, device="cpu")
                print(
                    f"[AFR matrix] {spec['group_id']} layer "
                    f"{layer_index}/{len(modules)}",
                    flush=True,
                )

    if len(rows) != EXPECTED_CASES * 3:
        raise AFRRunError(f"Expected 144 matrix rows, found {len(rows)}")
    classification, details = _classify_matrix(rows)
    gate = {
        "schema_version": 1,
        "status": "passed" if classification == "AFR-GO" else "stopped",
        "classification": classification,
        "alpha": ALPHA,
        "synthetic_passed": bool(synthetic["passed"]),
        **details,
        "thresholds": {
            "zero_tolerance": ZERO_TOLERANCE,
            "preservation_optimality_tolerance": PRESERVATION_OPTIMALITY_TOLERANCE,
            "compensation_nonzero_tolerance": COMPENSATION_NONZERO_TOLERANCE,
            "measurable_improvement_tolerance": MEASURABLE_IMPROVEMENT_TOLERANCE,
            "measurable_layer_count": MEASURABLE_LAYER_COUNT,
            "material_preservation_improvement": MATERIAL_PRESERVATION_IMPROVEMENT,
        },
        "variant_c_max_reproduction_difference": max_c_difference,
    }
    gate["gate_fingerprint"] = protocol.fingerprint(gate)
    gate["completed_at"] = protocol.utc_now()
    run_info = {
        "config_sha256": protocol.sha256(config_path),
        "anchors_sha256": protocol.sha256(Path(config["_resolved"]["anchors_path"])),
        "qualification_sha256": protocol.sha256(qualification_path),
        "k0_sha256": k0_metadata["artifact_sha256"],
        "frozen_c_sha256": protocol.sha256(frozen_path),
        "matrix_runtime_seconds": time.monotonic() - started,
        "checkpoint_dtype": str(production_dtype),
        "command": (
            "python experiments/confuse5_single_vs_joint/afr/runner.py all "
            "--skip-existing"
        ),
        **protocol.git_state(),
    }
    matrix_csv = output_root / "matrix" / "results_c_f_g.csv"
    matrix_gate = output_root / "matrix" / "gate.json"
    report_path = output_root / "REPORT.md"
    _atomic_csv(matrix_csv, MATRIX_FIELDS, rows)
    protocol.write_json_atomic(matrix_gate, gate)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_matrix_report(rows, synthetic, gate, run_info), encoding="utf-8"
    )
    context = {
        "config": config,
        "anchors": anchors,
        "specs": specs,
        "qualification_path": qualification_path,
        "k0_metadata": k0_metadata,
        "synthetic": synthetic,
        "run_info": run_info,
    }
    return rows, gate, context, balls_states


def _checkpoint_paths(output_root: Path, variant: str) -> tuple[Path, Path]:
    root = output_root / "checkpoints" / variant
    return root / "weights.safetensors", root / "metadata.json"


def _write_balls_checkpoints(
    output_root: Path,
    states: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
    matrix_gate_path: Path,
    matrix_gate: Mapping[str, Any],
    *,
    skip_existing: bool,
) -> dict[str, dict[str, Any]]:
    from safetensors.torch import save_file

    config = context["config"]
    balls = next(spec for spec in context["specs"] if spec["group_id"] == "balls")
    expected_keys = int(config["evaluation"]["expected_checkpoint_keys"])
    output: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        state = dict(states[variant])
        if len(state) != expected_keys:
            raise AFRRunError(
                f"{variant} checkpoint has {len(state)} keys, expected {expected_keys}"
            )
        checkpoint_path, metadata_path = _checkpoint_paths(output_root, variant)
        identity = {
            "experiment": "afr_balls_smoke_v1",
            "variant": variant,
            "alpha": ALPHA if variant in (VARIANT_F, VARIANT_G) else None,
            "group_id": "balls",
            "targets": list(balls["targets"]),
            "anchors": list(balls["anchors"]),
            "retain_concepts": list(balls["retain_concepts"]),
            "config_sha256": context["run_info"]["config_sha256"],
            "anchors_sha256": context["run_info"]["anchors_sha256"],
            "qualification_sha256": context["run_info"]["qualification_sha256"],
            "k0_sha256": context["run_info"]["k0_sha256"],
            "matrix_gate_fingerprint": matrix_gate["gate_fingerprint"],
            "module_names": sorted(state),
            "implementation_sha256": {
                "runner": protocol.sha256(Path(__file__)),
                "core": protocol.sha256(HERE / "core.py"),
            },
            "production_oce_modified": False,
        }
        fingerprint = protocol.fingerprint(identity)
        if checkpoint_path.is_file() or metadata_path.is_file():
            if not (
                skip_existing and checkpoint_path.is_file() and metadata_path.is_file()
            ):
                raise AFRRunError(
                    f"AFR checkpoint output collision: {checkpoint_path.parent}"
                )
            existing = protocol.read_json(metadata_path)
            if (
                existing.get("status") != "complete"
                or existing.get("checkpoint_fingerprint") != fingerprint
                or existing.get("checkpoint_sha256") != protocol.sha256(checkpoint_path)
            ):
                raise AFRRunError(
                    f"Existing AFR checkpoint failed validation: {checkpoint_path}"
                )
            output[variant] = existing
            continue
        checkpoint_path.parent.mkdir(parents=True, exist_ok=False)
        running = {
            "schema_version": 1,
            "status": "running",
            "checkpoint_fingerprint": fingerprint,
            **identity,
            "started_at": protocol.utc_now(),
            "runtime": protocol.runtime_provenance(),
            "source_hashes": protocol.source_hashes([Path(__file__), HERE / "core.py"]),
            "matrix_gate_path": str(matrix_gate_path.resolve()),
            "matrix_gate_sha256": protocol.sha256(matrix_gate_path),
        }
        protocol.write_json_atomic(metadata_path, running)
        save_file(state, str(checkpoint_path))
        complete = {
            **running,
            "status": "complete",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": protocol.sha256(checkpoint_path),
            "production_dtype": context["run_info"]["checkpoint_dtype"],
            "finished_at": protocol.utc_now(),
        }
        protocol.write_json_atomic(metadata_path, complete)
        output[variant] = complete
    return output


def _evaluate_with_classes(
    evaluator: Any,
    image_paths: Sequence[Path],
    expected_concept: str,
    extra_concepts: Sequence[str],
) -> list[dict[str, Any]]:
    """Evaluate exact top-1 plus probabilities for a fixed small label set."""
    from PIL import Image

    labels = list(dict.fromkeys([expected_concept, *extra_concepts]))
    indices = {label: evaluator.class_index(label) for label in labels}
    output: list[dict[str, Any]] = []
    with evaluator.torch.inference_mode():
        for start in range(0, len(image_paths), evaluator.batch_size):
            chunk = image_paths[start : start + evaluator.batch_size]
            tensors = []
            for path in chunk:
                with Image.open(path) as image:
                    tensors.append(evaluator.preprocess(image.convert("RGB")))
            logits = evaluator.model(
                evaluator.torch.stack(tensors).to(evaluator.device)
            )
            probabilities = logits.softmax(dim=1)
            top_probs, top_indices = probabilities.topk(evaluator.top_k, dim=1)
            expected_index = indices[expected_concept]
            for offset, path in enumerate(chunk):
                predicted = int(top_indices[offset, 0].item())
                top = []
                for rank in range(evaluator.top_k):
                    index = int(top_indices[offset, rank].item())
                    top.append(
                        {
                            "rank": rank + 1,
                            "index": index,
                            "label": evaluator.categories[index],
                            "probability": float(top_probs[offset, rank].item()),
                            "raw_logit": float(logits[offset, index].item()),
                        }
                    )
                output.append(
                    {
                        "image_path": str(path.resolve()),
                        "image_sha256": protocol.sha256(path),
                        "expected_index": expected_index,
                        "expected_category": evaluator.categories[expected_index],
                        "predicted_index": predicted,
                        "predicted_category": evaluator.categories[predicted],
                        "correct": predicted == expected_index,
                        "expected_probability": float(
                            probabilities[offset, expected_index].item()
                        ),
                        "class_probabilities": {
                            label: float(probabilities[offset, index].item())
                            for label, index in indices.items()
                        },
                        "top5": top,
                    }
                )
    return output


def _image_job_identity(
    variant: str,
    role: str,
    concept: str,
    rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    extra_labels: Sequence[str],
) -> dict[str, Any]:
    return {
        "experiment": "afr_balls_smoke_v1",
        "variant": variant,
        "role": role,
        "concept": concept,
        "extra_labels": list(extra_labels),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "ordered_rows_sha256": protocol.fingerprint(
            {
                "rows": [
                    {
                        "case_number": row["case_number"],
                        "prompt": row["prompt"],
                        "evaluation_seed": row["evaluation_seed"],
                    }
                    for row in rows
                ]
            }
        ),
        "generation": config["evaluation"]["generation"],
        "classifier": config["evaluation"]["classifier"],
    }


def _run_image_job(
    job: Mapping[str, Any],
    generation: Any,
    evaluator: Any,
    *,
    skip_existing: bool,
) -> dict[str, Any]:
    manifest_path = Path(job["manifest_path"])
    result_path = Path(job["result_path"])
    fingerprint = str(job["job_fingerprint"])
    if result_path.is_file():
        if not skip_existing:
            raise AFRRunError(f"Image result already exists: {result_path}")
        existing = protocol.read_json(result_path)
        if (
            existing.get("status") != "complete"
            or existing.get("job_fingerprint") != fingerprint
            or existing.get("total") != len(job["rows"])
            or len(existing.get("items", [])) != len(job["rows"])
        ):
            raise AFRRunError(f"Existing image result failed validation: {result_path}")
        for item in existing["items"]:
            image_path = Path(item["image_path"])
            if (
                not image_path.is_file()
                or item.get("image_sha256") != protocol.sha256(image_path)
            ):
                raise AFRRunError(
                    f"Existing image result has a missing/hash-mismatched PNG: {image_path}"
                )
        return existing
    manifest: dict[str, Any]
    if manifest_path.is_file():
        manifest = protocol.read_json(manifest_path)
        if manifest.get("job_fingerprint") != fingerprint:
            raise AFRRunError(f"Image manifest fingerprint collision: {manifest_path}")
    else:
        if Path(job["image_dir"]).exists():
            raise AFRRunError(f"Untracked image directory collision: {job['image_dir']}")
        manifest = {
            "schema_version": 1,
            "status": "generating",
            "job_id": job["job_id"],
            "job_fingerprint": fingerprint,
            "variant": job["variant"],
            "role": job["role"],
            "concept": job["concept"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "generation_runtime": generation.identity,
            "evaluator": evaluator.identity,
            "items": [],
            "started_at": protocol.utc_now(),
        }
        protocol.write_json_atomic(manifest_path, manifest)
    generation.activate_checkpoint(job["checkpoint_spec"])
    recorded = {int(item["case_number"]): item for item in manifest.get("items", [])}
    paths: list[Path] = []
    for row in job["rows"]:
        path = Path(job["image_dir"]) / (
            f"case-{int(row['case_number']):06d}_seed-{int(row['evaluation_seed'])}.png"
        )
        previous = recorded.get(int(row["case_number"]))
        if previous is not None and path.is_file():
            if previous.get("image_sha256") != protocol.sha256(path):
                raise AFRRunError(f"Resumed image hash mismatch: {path}")
            paths.append(path)
            continue
        if previous is not None or path.exists():
            raise AFRRunError(f"Incomplete/untracked image collision: {path}")
        image_hash = pipeline._save_png(
            generation.generate(str(row["prompt"]), int(row["evaluation_seed"])),
            path,
        )
        paths.append(path)
        recorded[int(row["case_number"])] = {
            **row,
            "image_path": str(path.resolve()),
            "image_sha256": image_hash,
            "generated_at": protocol.utc_now(),
        }
        manifest["items"] = [recorded[key] for key in sorted(recorded)]
        protocol.write_json_atomic(manifest_path, manifest)
    if len(paths) != len(job["rows"]) or len(recorded) != len(job["rows"]):
        raise AFRRunError(f"Image count mismatch: {job['job_id']}")
    metrics = _evaluate_with_classes(
        evaluator, paths, job["concept"], job["extra_labels"]
    )
    items = [
        {**row, **metric} for row, metric in zip(job["rows"], metrics)
    ]
    correct = sum(bool(item["correct"]) for item in items)
    result = {
        "schema_version": 1,
        "status": "complete",
        "job_id": job["job_id"],
        "job_fingerprint": fingerprint,
        "variant": job["variant"],
        "role": job["role"],
        "concept": job["concept"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "generation_runtime": generation.identity,
        "evaluator": evaluator.identity,
        "correct": correct,
        "total": len(items),
        "accuracy": correct / len(items),
        "mean_expected_probability": _mean(
            float(item["expected_probability"]) for item in items
        ),
        "items": items,
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(result_path, result)
    manifest["status"] = "evaluated"
    manifest["completed_at"] = protocol.utc_now()
    protocol.write_json_atomic(manifest_path, manifest)
    return result


def _build_image_jobs(
    output_root: Path,
    config: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
    anchors: Mapping[str, str],
) -> list[dict[str, Any]]:
    _, by_class = pipeline.load_dataset(config)
    balls = next(group for group in config["groups"] if group["id"] == "balls")
    targets = list(balls["targets"])
    retains = list(balls["similar_non_targets"])
    matched = {target: anchors[target] for target in targets}
    jobs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        checkpoint = checkpoints[variant]
        checkpoint_spec = {"checkpoint_path": checkpoint["checkpoint_path"]}
        for concept in targets + retains:
            rows = [dict(row) for row in by_class[protocol.normalize(concept)][:100]]
            role = "target" if concept in targets else "similar_non_target"
            extras = [matched[concept]] if role == "target" else []
            identity = _image_job_identity(
                variant, role, concept, rows, checkpoint, config, extras
            )
            job_id = f"{variant}__{role}__{protocol.slug(concept)}"
            jobs.append(
                {
                    **identity,
                    "job_id": job_id,
                    "job_fingerprint": protocol.fingerprint(identity),
                    "checkpoint_spec": checkpoint_spec,
                    "rows": rows,
                    "image_dir": str(
                        output_root
                        / "balls_smoke"
                        / "images"
                        / variant
                        / role
                        / protocol.slug(concept)
                    ),
                    "manifest_path": str(
                        output_root / "balls_smoke" / "manifests" / f"{job_id}.json"
                    ),
                    "result_path": str(
                        output_root / "balls_smoke" / "evaluations" / f"{job_id}.json"
                    ),
                    "extra_labels": extras,
                }
            )
        for source_target in targets:
            anchor = matched[source_target]
            source_rows = by_class[protocol.normalize(source_target)][:50]
            rows = [
                {
                    **dict(row),
                    "source_target": source_target,
                    "source_prompt": row["prompt"],
                    "prompt": config["anchor_sanity"]["prompt_template"].format(
                        anchor=anchor
                    ),
                }
                for row in source_rows
            ]
            identity = _image_job_identity(
                variant, "anchor", anchor, rows, checkpoint, config, []
            )
            job_id = f"{variant}__anchor__{protocol.slug(anchor)}"
            jobs.append(
                {
                    **identity,
                    "job_id": job_id,
                    "job_fingerprint": protocol.fingerprint(identity),
                    "checkpoint_spec": checkpoint_spec,
                    "rows": rows,
                    "image_dir": str(
                        output_root
                        / "balls_smoke"
                        / "images"
                        / variant
                        / "anchor"
                        / protocol.slug(anchor)
                    ),
                    "manifest_path": str(
                        output_root / "balls_smoke" / "manifests" / f"{job_id}.json"
                    ),
                    "result_path": str(
                        output_root / "balls_smoke" / "evaluations" / f"{job_id}.json"
                    ),
                    "extra_labels": [],
                }
            )
    if len(jobs) != 21:
        raise AFRRunError(f"Expected 21 C/F/G smoke jobs, found {len(jobs)}")
    counts = {
        variant: sum(len(job["rows"]) for job in jobs if job["variant"] == variant)
        for variant in VARIANTS
    }
    if counts != {variant: 600 for variant in VARIANTS}:
        raise AFRRunError(f"Expected 600 images per new model, got {counts}")
    return jobs


def _anchor_original_references(
    config: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    matched_anchors: Mapping[str, str],
) -> dict[tuple[str, int], Path]:
    primary_root = Path(config["_resolved"]["output_root"])
    pipeline._require_gate(primary_root, "anchor_sanity")
    path = primary_root / "anchor_sanity" / "per_image.json"
    payload = protocol.read_json(path)
    wanted = {anchor: target for target, anchor in matched_anchors.items()}
    references: dict[tuple[str, int], Path] = {}
    for item in payload.get("items", []):
        anchor = str(item.get("anchor", ""))
        if anchor not in wanted or item.get("target") != wanted[anchor]:
            continue
        image_path = Path(str(item.get("image_path", "")))
        if not image_path.is_file():
            raise AFRRunError(f"Anchor-sanity Original PNG is missing: {image_path}")
        if item.get("image_sha256") != protocol.sha256(image_path):
            raise AFRRunError(f"Anchor-sanity Original PNG hash mismatch: {image_path}")
        references[(anchor, int(item["evaluation_seed"]))] = image_path
    if {anchor for anchor, _ in references} != set(wanted):
        raise AFRRunError("Anchor-sanity Original references do not cover both anchors")
    if any(
        sum(anchor == candidate for candidate, _ in references) != 8
        for anchor in wanted
    ):
        raise AFRRunError("Expected exactly 8 retained Original references per anchor")
    # Confirm every reference seed is included in each model's 50-image anchor job.
    for variant in VARIANTS:
        for anchor in wanted:
            job = next(
                row
                for row in jobs
                if row["variant"] == variant
                and row["role"] == "anchor"
                and row["concept"] == anchor
            )
            seeds = {int(row["evaluation_seed"]) for row in job["rows"]}
            missing = [
                seed
                for candidate, seed in references
                if candidate == anchor and seed not in seeds
            ]
            if missing:
                raise AFRRunError(
                    f"Anchor LPIPS reference seeds missing from {variant}: {missing}"
                )
    return references


def _load_lpips(device: str) -> tuple[Any, str]:
    """Preflight the already-declared repository LPIPS dependency."""
    try:
        import importlib.metadata
        import lpips

        installed_version = importlib.metadata.version("lpips")
        if installed_version != "0.1.4":
            raise AFRRunError(
                f"Frozen LPIPS version must be 0.1.4, got {installed_version}"
            )
        model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        raise AFRRunError(
            "The repository-declared LPIPS evaluator could not be initialized; "
            "no smoke image has been generated"
        ) from exc
    return model, installed_version


def _lpips_tensor(path: Path, device: str) -> Any:
    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    with Image.open(path) as image:
        tensor = pil_to_tensor(image.convert("RGB")).to(
            device=device, dtype=torch.float32
        )
    return tensor.unsqueeze(0) / 127.5 - 1.0


def _evaluate_anchor_lpips(
    model: Any,
    references: Mapping[tuple[str, int], Path],
    results: Sequence[Mapping[str, Any]],
    device: str,
) -> list[dict[str, Any]]:
    import torch

    output = []
    lookup = {
        (result["variant"], result["concept"]): result
        for result in results
        if result["role"] == "anchor"
    }
    anchors = tuple(dict.fromkeys(candidate for candidate, _ in references))
    if len(anchors) != 2:
        raise AFRRunError(f"Expected two matched anchor references, got {anchors}")
    with torch.inference_mode():
        for variant in VARIANTS:
            for anchor in anchors:
                result = lookup[(variant, anchor)]
                edited = {
                    int(item["evaluation_seed"]): Path(item["image_path"])
                    for item in result["items"]
                }
                values = []
                for (candidate, seed), original_path in sorted(references.items()):
                    if candidate != anchor:
                        continue
                    edited_path = edited[seed]
                    value = scalar(
                        model(
                            _lpips_tensor(original_path, device),
                            _lpips_tensor(edited_path, device),
                        ).reshape(())
                    )
                    values.append(
                        {
                            "variant": variant,
                            "anchor": anchor,
                            "evaluation_seed": seed,
                            "original_path": str(original_path.resolve()),
                            "edited_path": str(edited_path.resolve()),
                            "lpips": value,
                        }
                    )
                if len(values) != 8:
                    raise AFRRunError(
                        f"Expected 8 LPIPS pairs for {variant}/{anchor}"
                    )
                output.extend(values)
    return output


def _subset_accuracy(result: Mapping[str, Any], case_numbers: set[int]) -> dict[str, Any]:
    items = [
        item for item in result.get("items", []) if int(item["case_number"]) in case_numbers
    ]
    if len(items) != len(case_numbers):
        raise AFRRunError("Context result does not cover the ordered smoke cases")
    correct = sum(bool(item["correct"]) for item in items)
    return {"correct": correct, "total": len(items), "accuracy": correct / len(items)}


def _load_existing_context(
    config: Mapping[str, Any], anchors: Mapping[str, str]
) -> dict[str, Any]:
    """Read, never rerun, Original/Single/released-Joint balls results."""
    primary_root = Path(config["_resolved"]["output_root"])
    pipeline._require_gate(primary_root, "original_canary")
    _, archive_root = pipeline._legacy_reference(config)
    originals = pipeline._load_legacy_original_results(config, archive_root)
    _, by_class = pipeline.load_dataset(config)
    formal_root = primary_root / "baseline_qualified_primary_v1" / "formal"
    completion = protocol.read_json(formal_root / "completion.json")
    if completion.get("status") != "complete":
        raise AFRRunError("Qualified primary completion is missing")
    qualified = prior_audit.qualified_primary
    qualification, qualification_path = qualified.load_and_validate_qualification(
        primary_root, config
    )
    formal_jobs = qualified.build_jobs(
        config,
        anchors,
        primary_root / "baseline_qualified_primary_v1",
        protocol.sha256(qualification_path),
    )
    job_lookup = {job["job_id"]: job for job in formal_jobs}
    balls = next(group for group in config["groups"] if group["id"] == "balls")
    targets = tuple(balls["targets"])
    concepts = [*targets, *balls["similar_non_targets"]]
    context: dict[str, Any] = {"matched_first_100": {}}
    for concept in concepts:
        cases = {
            int(row["case_number"])
            for row in by_class[protocol.normalize(concept)][:100]
        }
        original = originals[("balls", protocol.normalize(concept))]
        joint_id = f"joint__balls__{protocol.slug(concept)}"
        joint = qualified.validate_result(job_lookup[joint_id])
        record: dict[str, Any] = {
            "original": _subset_accuracy(original, cases),
            "released_joint": _subset_accuracy(joint, cases),
        }
        if concept in targets:
            record["matched_single"] = _subset_accuracy(
                qualified.validate_result(
                    job_lookup[
                        f"single__balls__{protocol.slug(concept)}__"
                        f"{protocol.slug(concept)}"
                    ]
                ),
                cases,
            )
        else:
            record["single_context"] = {}
            for target in targets:
                record["single_context"][target] = _subset_accuracy(
                    qualified.validate_result(
                        job_lookup[
                            f"single__balls__{protocol.slug(target)}__"
                            f"{protocol.slug(concept)}"
                        ]
                    ),
                    cases,
                )
        context["matched_first_100"][concept] = record
    context["source"] = "existing completed artifacts; no Original/Single/Joint images rerun"
    return context


def _summarize_images(
    results: Sequence[Mapping[str, Any]],
    lpips_rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    matched_anchors: Mapping[str, str],
    runtime_seconds: float,
) -> dict[str, Any]:
    lookup = {(row["variant"], row["role"], row["concept"]): row for row in results}
    targets = dict(matched_anchors)
    retains = tuple(
        row["concept"]
        for row in results
        if row["variant"] == VARIANT_C and row["role"] == "similar_non_target"
    )
    if len(retains) != 3:
        raise AFRRunError(f"Expected three balls preservation classes, got {retains}")
    anchors = tuple(targets.values())
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        target_rows = []
        for target, anchor in targets.items():
            result = lookup[(variant, "target", target)]
            target_rows.append(
                {
                    "target": target,
                    "matched_anchor": anchor,
                    "accuracy": float(result["accuracy"]),
                    "correct": int(result["correct"]),
                    "mean_target_probability": _mean(
                        float(item["class_probabilities"][target])
                        for item in result["items"]
                    ),
                    "anchor_top1_rate": _mean(
                        1.0 if protocol.normalize(item["predicted_category"])
                        == protocol.normalize(anchor) else 0.0
                        for item in result["items"]
                    ),
                    "mean_anchor_probability": _mean(
                        float(item["class_probabilities"][anchor])
                        for item in result["items"]
                    ),
                }
            )
        similar_rows = [
            {
                "concept": concept,
                "accuracy": float(lookup[(variant, "similar_non_target", concept)]["accuracy"]),
                "correct": int(lookup[(variant, "similar_non_target", concept)]["correct"]),
            }
            for concept in retains
        ]
        anchor_rows = [
            {
                "anchor": anchor,
                "accuracy": float(lookup[(variant, "anchor", anchor)]["accuracy"]),
                "correct": int(lookup[(variant, "anchor", anchor)]["correct"]),
                "mean_anchor_probability": float(
                    lookup[(variant, "anchor", anchor)]["mean_expected_probability"]
                ),
                "original_lpips_mean_8": _mean(
                    float(row["lpips"])
                    for row in lpips_rows
                    if row["variant"] == variant and row["anchor"] == anchor
                ),
            }
            for anchor in anchors
        ]
        variants[variant] = {
            "targets": target_rows,
            "target_macro_accuracy": _mean(row["accuracy"] for row in target_rows),
            "similar_non_targets": similar_rows,
            "similar_macro_accuracy": _mean(row["accuracy"] for row in similar_rows),
            "anchors": anchor_rows,
            "anchor_macro_accuracy": _mean(row["accuracy"] for row in anchor_rows),
            "anchor_original_lpips_mean": _mean(
                row["original_lpips_mean_8"] for row in anchor_rows
            ),
        }

    c, f, g = (variants[key] for key in VARIANTS)
    g_target_directions = all(
        next(row for row in g["targets"] if row["target"] == target)["accuracy"]
        < next(row for row in c["targets"] if row["target"] == target)["accuracy"]
        for target in targets
    )
    g_target_success = (
        c["target_macro_accuracy"] - g["target_macro_accuracy"]
        >= TARGET_SMOKE_IMPROVEMENT
        and g_target_directions
    )
    f_target_directions = all(
        next(row for row in f["targets"] if row["target"] == target)["accuracy"]
        < next(row for row in c["targets"] if row["target"] == target)["accuracy"]
        for target in targets
    )
    f_target_success = (
        c["target_macro_accuracy"] - f["target_macro_accuracy"]
        >= TARGET_SMOKE_IMPROVEMENT
        and f_target_directions
    )
    semantic_direction = all(
        next(row for row in g["targets"] if row["target"] == target)[
            "mean_anchor_probability"
        ]
        > next(row for row in c["targets"] if row["target"] == target)[
            "mean_anchor_probability"
        ]
        for target in targets
    )
    f_semantic_direction = all(
        next(row for row in f["targets"] if row["target"] == target)[
            "mean_anchor_probability"
        ]
        > next(row for row in c["targets"] if row["target"] == target)[
            "mean_anchor_probability"
        ]
        for target in targets
    )
    similar_acceptable = (
        g["similar_macro_accuracy"]
        >= c["similar_macro_accuracy"] - PRESERVATION_COLLAPSE_TOLERANCE
    )
    anchor_acceptable = (
        g["anchor_macro_accuracy"]
        >= c["anchor_macro_accuracy"] - PRESERVATION_COLLAPSE_TOLERANCE
    )
    compensation_benefit_components = {
        "similar_accuracy_g_minus_f": (
            g["similar_macro_accuracy"] - f["similar_macro_accuracy"]
        ),
        "anchor_accuracy_g_minus_f": (
            g["anchor_macro_accuracy"] - f["anchor_macro_accuracy"]
        ),
        "anchor_lpips_f_minus_g": (
            f["anchor_original_lpips_mean"] - g["anchor_original_lpips_mean"]
        ),
    }
    compensation_image_benefit = (
        compensation_benefit_components["similar_accuracy_g_minus_f"]
        >= IMAGE_ABLATION_ACCURACY_BENEFIT
        or compensation_benefit_components["anchor_accuracy_g_minus_f"]
        >= IMAGE_ABLATION_ACCURACY_BENEFIT
        or compensation_benefit_components["anchor_lpips_f_minus_g"]
        >= IMAGE_ABLATION_LPIPS_BENEFIT
    )
    if (
        g_target_success
        and f_target_success
        and semantic_direction
        and f_semantic_direction
        and similar_acceptable
        and anchor_acceptable
        and compensation_image_benefit
    ):
        classification = "AFR-I2"
    elif f_target_success and g_target_success and not compensation_image_benefit:
        classification = "AFR-I1"
    else:
        classification = "AFR-I0"
    return {
        "schema_version": 1,
        "status": "complete",
        "classification": classification,
        "variants": variants,
        "checks": {
            "g_target_success": g_target_success,
            "f_target_success": f_target_success,
            "g_both_targets_improve": g_target_directions,
            "target_to_anchor_probability_moves_for_both": semantic_direction,
            "f_target_to_anchor_probability_moves_for_both": f_semantic_direction,
            "g_similar_non_target_acceptable": similar_acceptable,
            "g_anchor_generation_acceptable": anchor_acceptable,
            "compensation_image_benefit": compensation_image_benefit,
            **compensation_benefit_components,
        },
        "thresholds": {
            "target_macro_improvement": TARGET_SMOKE_IMPROVEMENT,
            "preservation_collapse_tolerance": PRESERVATION_COLLAPSE_TOLERANCE,
            "ablation_accuracy_benefit": IMAGE_ABLATION_ACCURACY_BENEFIT,
            "ablation_lpips_benefit": IMAGE_ABLATION_LPIPS_BENEFIT,
        },
        "new_image_count": sum(int(row["total"]) for row in results),
        "lpips_pair_count": len(lpips_rows),
        "existing_context": context,
        "runtime_seconds": runtime_seconds,
        "completed_at": protocol.utc_now(),
    }


def _image_report_section(summary: Mapping[str, Any]) -> str:
    lines = [
        f"**Final classification: {summary['classification']}.**",
        "",
        f"New images: `{summary['new_image_count']}`. Runtime: `{summary['runtime_seconds']:.1f}` seconds. Original, Single, and released Joint context was read from completed artifacts; none was regenerated.",
        "",
        "### Target erasure and literal target-to-anchor movement",
        "",
        "| Variant | Target | Target accuracy | Target probability | Matched-anchor top-1 | Matched-anchor probability |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for row in summary["variants"][variant]["targets"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant,
                        row["target"],
                        _fmt(float(row["accuracy"])),
                        _fmt(float(row["mean_target_probability"])),
                        _fmt(float(row["anchor_top1_rate"])),
                        _fmt(float(row["mean_anchor_probability"])),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "### Existing first-100 context (read only)",
            "",
            "| Concept | Original | Released Joint | Matched Single / Single soccer | Single volleyball |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for concept, record in summary["existing_context"]["matched_first_100"].items():
        if "matched_single" in record:
            first_single = _fmt(float(record["matched_single"]["accuracy"]))
            second_single = "n/a"
        else:
            single_context = record["single_context"]
            first_single = _fmt(float(single_context["soccer ball"]["accuracy"]))
            second_single = _fmt(float(single_context["volleyball"]["accuracy"]))
        lines.append(
            "| "
            + " | ".join(
                [
                    concept,
                    _fmt(float(record["original"]["accuracy"])),
                    _fmt(float(record["released_joint"]["accuracy"])),
                    first_single,
                    second_single,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Similar non-target preservation",
            "",
            "| Variant | tennis ball | rugby ball | ping-pong ball | Macro |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        record = summary["variants"][variant]
        values = {row["concept"]: row["accuracy"] for row in record["similar_non_targets"]}
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    _fmt(float(values["tennis ball"])),
                    _fmt(float(values["rugby ball"])),
                    _fmt(float(values["ping-pong ball"])),
                    _fmt(float(record["similar_macro_accuracy"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Anchor generation",
            "",
            "| Variant | Anchor | Accuracy | Anchor probability | Original-vs-edited LPIPS (8 fixed seeds) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        for row in summary["variants"][variant]["anchors"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant,
                        row["anchor"],
                        _fmt(float(row["accuracy"])),
                        _fmt(float(row["mean_anchor_probability"])),
                        _fmt(float(row["original_lpips_mean_8"])),
                    ]
                )
                + " |"
            )
    checks = summary["checks"]
    lines.extend(
        [
            "",
            "### Smoke gates and F/G ablation",
            "",
            f"- C-to-F/G target success: `{checks['f_target_success']}` / `{checks['g_target_success']}`; G both targets improve: `{checks['g_both_targets_improve']}`.",
            f"- Target-to-anchor probabilities move for both targets under F/G: `{checks['f_target_to_anchor_probability_moves_for_both']}` / `{checks['target_to_anchor_probability_moves_for_both']}`.",
            f"- G similar preservation acceptable: `{checks['g_similar_non_target_acceptable']}`; G anchor generation acceptable: `{checks['g_anchor_generation_acceptable']}`.",
            f"- G-F similar macro accuracy: `{_fmt(float(checks['similar_accuracy_g_minus_f']))}`; G-F anchor macro accuracy: `{_fmt(float(checks['anchor_accuracy_g_minus_f']))}`; F-G LPIPS: `{_fmt(float(checks['anchor_lpips_f_minus_g']))}`.",
            f"- Compensation image benefit: `{checks['compensation_image_benefit']}`.",
        ]
    )
    if summary["classification"] == "AFR-I1":
        lines.extend(
            [
                "",
                "> The compensated AFR editor empirically reduces to the pure residual projection for this smoke setting.",
            ]
        )
    lines.extend(
        [
            "",
            "### Artifacts and provenance",
            "",
            f"- Command: `{summary['command']}`",
            f"- Git commit: `{summary['git_hash']}`; dirty at completion: `{summary['git_dirty']}`",
            f"- Generation root: `{summary['generation_output_root']}`",
            f"- Evaluator root: `{summary['evaluator_output_root']}`",
            f"- LPIPS output: `{summary['lpips_output']}`",
        ]
    )
    for variant in VARIANTS:
        checkpoint = summary["checkpoints"][variant]
        lines.append(
            f"- {variant} checkpoint: `{checkpoint['path']}`; metadata: "
            f"`{checkpoint['metadata_path']}`"
        )
    return "\n".join(lines)


def run_smoke(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    context: Mapping[str, Any],
    balls_states: Mapping[str, Mapping[str, Any]],
    *,
    skip_existing: bool,
) -> dict[str, Any]:
    if gate["classification"] != "AFR-GO" or gate["status"] != "passed":
        raise AFRRunError("Image smoke is forbidden until the AFR matrix gate passes")
    started = time.monotonic()
    config = context["config"]
    matrix_gate_path = output_root / "matrix" / "gate.json"
    checkpoints = _write_balls_checkpoints(
        output_root,
        balls_states,
        context,
        matrix_gate_path,
        gate,
        skip_existing=skip_existing,
    )
    balls = next(spec for spec in context["specs"] if spec["group_id"] == "balls")
    matched_anchors = {
        target: context["anchors"][target] for target in balls["targets"]
    }
    jobs = _build_image_jobs(
        output_root, config, checkpoints, context["anchors"]
    )

    # All evaluator/dependency/artifact preflights happen before the first new image.
    existing_context = _load_existing_context(config, context["anchors"])
    references = _anchor_original_references(config, jobs, matched_anchors)
    evaluator = pipeline.EvaluationRuntime(config)
    for label in ("soccer ball", "volleyball", "basketball", "baseball"):
        evaluator.class_index(label)
    lpips_model, lpips_version = _load_lpips(str(config["model"]["device"]))
    generation = pipeline.GenerationRuntime(config)

    plan = {
        "schema_version": 1,
        "status": "resolved",
        "matrix_gate_sha256": protocol.sha256(matrix_gate_path),
        "variants": list(VARIANTS),
        "alpha": ALPHA,
        "targets": list(balls["targets"]),
        "matched_anchors": matched_anchors,
        "similar_non_targets": list(balls["retain_concepts"]),
        "jobs": [
            {key: value for key, value in job.items() if key not in {"rows", "checkpoint_spec"}}
            for job in jobs
        ],
        "new_images_per_variant": 600,
        "total_new_images": 1800,
        "new_original_images": 0,
        "lpips_fixed_original_pairs_per_anchor_per_variant": 8,
        "created_at": protocol.utc_now(),
    }
    plan_path = output_root / "balls_smoke" / "resolved_plan.json"
    protocol.write_json_atomic(plan_path, plan)
    results = []
    for index, job in enumerate(jobs, start=1):
        print(
            f"[AFR balls smoke {index}/{len(jobs)}] {job['job_id']}: "
            f"{len(job['rows'])} images",
            flush=True,
        )
        results.append(
            _run_image_job(
                job, generation, evaluator, skip_existing=skip_existing
            )
        )
    if sum(int(result["total"]) for result in results) != 1800:
        raise AFRRunError("Completed balls smoke does not contain exactly 1800 images")
    lpips_rows = _evaluate_anchor_lpips(
        lpips_model, references, results, str(config["model"]["device"])
    )
    lpips_path = output_root / "balls_smoke" / "anchor_lpips.json"
    protocol.write_json_atomic(
        lpips_path,
        {
            "schema_version": 1,
            "status": "complete",
            "implementation": f"lpips=={lpips_version} LPIPS(net='alex')",
            "new_original_images": 0,
            "pair_count": len(lpips_rows),
            "items": lpips_rows,
            "completed_at": protocol.utc_now(),
        },
    )
    summary = _summarize_images(
        results,
        lpips_rows,
        existing_context,
        matched_anchors,
        time.monotonic() - started,
    )
    summary["checkpoints"] = {
        variant: {
            "path": checkpoint["checkpoint_path"],
            "sha256": checkpoint["checkpoint_sha256"],
            "metadata_path": str(_checkpoint_paths(output_root, variant)[1].resolve()),
        }
        for variant, checkpoint in checkpoints.items()
    }
    summary["generation_output_root"] = str(
        (output_root / "balls_smoke" / "images").resolve()
    )
    summary["evaluator_output_root"] = str(
        (output_root / "balls_smoke" / "evaluations").resolve()
    )
    summary["lpips_output"] = str(lpips_path.resolve())
    summary["command"] = (
        "python experiments/confuse5_single_vs_joint/afr/runner.py all "
        "--skip-existing"
    )
    summary.update(protocol.git_state())
    summary_path = output_root / "balls_smoke" / "summary.json"
    protocol.write_json_atomic(summary_path, summary)
    report_path = output_root / "REPORT.md"
    report_path.write_text(
        _render_matrix_report(
            rows,
            context["synthetic"],
            gate,
            context["run_info"],
            _image_report_section(summary),
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "synthetic":
        print(json.dumps(synthetic_unit_tests(), indent=2, sort_keys=True))
        return 0
    output_root = args.output_root.resolve()
    rows, gate, context, balls_states = run_matrix(
        args.config.resolve(), output_root
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
    if args.stage == "matrix" or gate["classification"] != "AFR-GO":
        return 0
    summary = run_smoke(
        output_root,
        rows,
        gate,
        context,
        balls_states,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
