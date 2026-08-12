#!/usr/bin/env python3
"""Float64 anchor-minimum control inside the exact orthogonal feasible family.

This matrix-only audit closes the anchor-drift caveat left by the S-optimal
exact control.  It never edits model weights, writes checkpoints, generates
images, or invokes an image evaluator.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
EXACT_CONTROL_ROOT = HERE.parent
EXACT_CONTROL_RUNNER = EXACT_CONTROL_ROOT / "run.py"
EXACT_CONTROL_MODULE_NAME = "_confuse5_exact_orthogonal_control_runner"


def _load_exact_control_runner() -> Any:
    """Load the parent runner without claiming the ambiguous module name `run`.

    The parent temporarily needs its own bare ``import run`` to resolve to the
    experiment-level checkpoint builder.  Preserve any pre-existing module of
    that name while the uniquely named parent module initializes.
    """
    existing = sys.modules.get(EXACT_CONTROL_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        EXACT_CONTROL_MODULE_NAME, EXACT_CONTROL_RUNNER
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot create an import spec for {EXACT_CONTROL_RUNNER}"
        )
    module = importlib.util.module_from_spec(spec)
    previous_run = sys.modules.pop("run", None)
    sys.modules[EXACT_CONTROL_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(EXACT_CONTROL_MODULE_NAME, None)
        raise
    finally:
        # Do not leave the parent's experiment-level `run` alias in global
        # module state, and do not overwrite a caller's pre-existing alias.
        sys.modules.pop("run", None)
        if previous_run is not None:
            sys.modules["run"] = previous_run
    return module


exact_control = _load_exact_control_runner()


protocol = exact_control.protocol
checkpoint_builder = exact_control.checkpoint_builder
prior_audit = exact_control.prior_audit
ControlError = exact_control.ControlError

VARIANT_C = exact_control.VARIANT_C
VARIANT_D = exact_control.VARIANT_D
VARIANT_E = "E_anchor_optimal_exact_orthogonal"
EXPECTED_GROUPS = exact_control.EXPECTED_GROUPS
EXPECTED_RANK = exact_control.EXPECTED_RANK
EXPECTED_CASES = exact_control.EXPECTED_CASES
MAPPING_ZERO_TOLERANCE = exact_control.MAPPING_ZERO_TOLERANCE
FLOAT64_ORTHOGONALITY_TOLERANCE = exact_control.FLOAT64_ORTHOGONALITY_TOLERANCE

# Reuse the material thresholds preregistered before the preceding D control.
CONSISTENT_LAYER_COUNT = exact_control.CONSISTENT_LAYER_COUNT
SMALL_LAYER_COUNT = exact_control.SMALL_LAYER_COUNT
MATERIAL_PRESERVE_DELTA = exact_control.MATERIAL_PRESERVE_DELTA
MATERIAL_PRESERVE_RATIO = exact_control.MATERIAL_PRESERVE_RATIO
SMALL_PRESERVE_RATIO = exact_control.SMALL_PRESERVE_RATIO
MATERIAL_ANCHOR_DELTA = exact_control.MATERIAL_ANCHOR_DELTA
MATERIAL_ANCHOR_RATIO = exact_control.MATERIAL_ANCHOR_RATIO
SMALL_ANCHOR_DELTA = exact_control.SMALL_ANCHOR_DELTA
SMALL_ANCHOR_RATIO = exact_control.SMALL_ANCHOR_RATIO

PARETO_LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)

# These hashes freeze the exact inputs established by the preceding control.
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

CSV_FIELDS = [
    "group",
    "target_names",
    "anchor_names",
    "layer",
    "variant",
    "target_rank",
    "anchor_rank",
    "true_leakage",
    "projector_leakage_crosscheck",
    "raw_frozen_s_preservation_loss",
    "normalized_frozen_s_preservation_distortion",
    "frozen_s_delta_vs_variant_c",
    "frozen_s_ratio_vs_variant_c",
    "anchor_feature_drift",
    "anchor_delta_vs_variant_c",
    "anchor_ratio_vs_variant_c",
    "anchor_delta_vs_variant_d",
    "orthogonality_residual",
    "exact_mapping_residual",
    "determinant",
]

PARETO_CSV_FIELDS = [
    "group",
    "target_names",
    "anchor_names",
    "layer",
    "lambda",
    "target_rank",
    "anchor_rank",
    "true_leakage",
    "normalized_anchor_feature_drift",
    "anchor_delta_vs_variant_c",
    "anchor_ratio_vs_variant_c",
    "normalized_frozen_s_preservation_distortion",
    "frozen_s_delta_vs_variant_c",
    "frozen_s_ratio_vs_variant_c",
    "orthogonality_residual",
    "exact_mapping_residual",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument(
        "--prior-exact-results",
        type=Path,
        default=EXACT_CONTROL_ROOT / "results_exact_control.csv",
        help="Frozen float64 C/D result rows from the preceding exact control.",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run only the small float64 anchor-minimum tests.",
    )
    return parser.parse_args(argv)


def _random_feasible(
    target: Any,
    anchor: Any,
    target_perp: Any,
    anchor_perp: Any,
    generator: Any,
) -> Any:
    target_rotation = exact_control._random_orthogonal(
        target.shape[1], generator, target.dtype
    )
    complement_rotation = exact_control._random_orthogonal(
        target_perp.shape[1], generator, target.dtype
    )
    return (
        anchor @ target_rotation @ target.T
        + anchor_perp @ complement_rotation @ target_perp.T
    )


def _normalized_anchor_drift(transform: Any, anchor_features: Any) -> float:
    denominator = exact_control._scalar(exact_control._sqnorm(anchor_features))
    if denominator <= 0:
        raise ControlError("Anchor feature drift denominator is zero")
    numerator = exact_control._scalar(
        exact_control._sqnorm(transform @ anchor_features - anchor_features)
    )
    return numerator / denominator


def synthetic_unit_tests() -> dict[str, Any]:
    """Check feasibility, optimality, value, and the G=H zero-drift case."""
    import torch

    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(20260813)
    dimension, rank = 6, 2
    target = exact_control._random_orthogonal(
        dimension, generator, dtype
    )[:, :rank]
    anchor = exact_control._random_orthogonal(
        dimension, generator, dtype
    )[:, :rank]
    coefficients = torch.randn(rank, 5, generator=generator, dtype=dtype)
    anchor_features = anchor @ coefficients
    anchor_covariance = anchor_features @ anchor_features.T

    transform, target_rotation, _, trace_qa = (
        exact_control.exact_orthogonal_mapper(
            target, anchor, anchor_covariance
        )
    )
    identity = torch.eye(dimension, dtype=dtype)
    anchor_projector = anchor @ anchor.T
    target_projector = target @ target.T
    orthogonality = exact_control._scalar(
        torch.linalg.matrix_norm(transform.T @ transform - identity)
    )
    leakage = exact_control._scalar(
        exact_control._sqnorm((identity - anchor_projector) @ transform @ target)
    ) / rank
    projector_residual = exact_control._scalar(
        torch.linalg.matrix_norm(
            transform @ target_projector @ transform.T - anchor_projector
        )
    )
    mapping_residual = exact_control._scalar(
        torch.linalg.matrix_norm(
            transform @ target - anchor @ target_rotation
        )
    )
    feasibility_tolerance = 1e-11
    if max(
        orthogonality, leakage, projector_residual, mapping_residual
    ) > feasibility_tolerance:
        raise ControlError("Synthetic anchor-minimum feasibility test failed")

    optimal_drift = _normalized_anchor_drift(transform, anchor_features)
    target_perp = exact_control._orthogonal_complement(target)
    anchor_perp = exact_control._orthogonal_complement(anchor)
    random_drifts = []
    for _ in range(256):
        candidate = _random_feasible(
            target, anchor, target_perp, anchor_perp, generator
        )
        random_drifts.append(
            _normalized_anchor_drift(candidate, anchor_features)
        )
    optimality_tolerance = 1e-11 * max(1.0, abs(optimal_drift))
    if optimal_drift > min(random_drifts) + optimality_tolerance:
        raise ControlError(
            "Synthetic anchor-minimum mapper was worse than a random feasible map"
        )

    anchor_energy = exact_control._scalar(
        exact_control._sqnorm(anchor_features)
    )
    direct_trace = exact_control._scalar(torch.trace(transform.T @ anchor_covariance))
    loss_from_trace = 2.0 * anchor_energy - 2.0 * direct_trace
    direct_loss = optimal_drift * anchor_energy
    loss_identity_error = abs(direct_loss - loss_from_trace)
    if loss_identity_error > 1e-11 * max(1.0, abs(direct_loss)):
        raise ControlError("Synthetic anchor-loss trace identity failed")

    special_target = anchor
    special_features = anchor @ torch.randn(
        rank, 5, generator=generator, dtype=dtype
    )
    special_covariance = special_features @ special_features.T
    special_transform, special_rotation, _, special_trace_qa = (
        exact_control.exact_orthogonal_mapper(
            special_target, anchor, special_covariance
        )
    )
    special_drift = _normalized_anchor_drift(
        special_transform, special_features
    )
    special_mapping = exact_control._scalar(
        torch.linalg.matrix_norm(
            special_transform @ special_target - anchor @ special_rotation
        )
    )
    if max(special_drift, special_mapping) > 1e-11:
        raise ControlError("Synthetic G=H zero-anchor-drift test failed")

    return {
        "passed": True,
        "dtype": "torch.float64",
        "dimension": dimension,
        "rank": rank,
        "orthogonality_residual": orthogonality,
        "true_leakage": leakage,
        "projector_mapping_residual": projector_residual,
        "exact_mapping_residual": mapping_residual,
        "normalized_anchor_drift": optimal_drift,
        "best_of_256_random_feasible_anchor_drifts": min(random_drifts),
        "closed_form_trace": trace_qa["achieved_trace"],
        "nuclear_norm_sum": trace_qa["theoretical_trace"],
        "closed_form_trace_abs_error": trace_qa["trace_abs_error"],
        "anchor_loss_trace_identity_abs_error": loss_identity_error,
        "special_g_equals_h_normalized_anchor_drift": special_drift,
        "special_g_equals_h_mapping_residual": special_mapping,
        "special_g_equals_h_trace_abs_error": special_trace_qa["trace_abs_error"],
    }


def _sha_guard(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ControlError(
            f"Frozen {label} hash mismatch: expected {expected}, got {actual}"
        )


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ControlError(f"Non-finite numeric value in frozen C/D CSV: {value}")
    return number


def _load_frozen_cd(
    path: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.is_file():
        raise ControlError(f"Frozen exact-control CSV is missing: {path}")
    _sha_guard(
        "exact-control CSV",
        exact_control._sha256(path),
        FROZEN_EXACT_RESULTS_SHA256,
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != EXPECTED_CASES * 2:
        raise ControlError(
            f"Expected 96 frozen C/D rows, found {len(source_rows)}"
        )

    numeric = (
        "true_leakage",
        "projector_leakage_crosscheck",
        "raw_preservation_loss",
        "normalized_preservation_distortion",
        "anchor_feature_drift",
        "orthogonality_residual",
        "exact_mapping_residual",
        "determinant",
    )
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in source_rows:
        variant = source["variant"]
        if variant not in (VARIANT_C, VARIANT_D):
            raise ControlError(f"Unexpected frozen exact variant: {variant}")
        group = source["group"]
        if group not in EXPECTED_GROUPS:
            raise ControlError(f"Unexpected frozen exact group: {group}")
        target_rank = int(source["target_rank"])
        anchor_rank = int(source["anchor_rank"])
        if target_rank != EXPECTED_RANK or anchor_rank != EXPECTED_RANK:
            raise ControlError(
                f"Frozen rank is not 12 at {group} {source['layer']}"
            )
        parsed = {field: _float_or_none(source.get(field)) for field in numeric}
        key = (group, source["layer"], variant)
        if key in lookup:
            raise ControlError(f"Duplicate frozen exact-control row: {key}")
        lookup[key] = {
            "group": group,
            "target_names": source["target_names"],
            "anchor_names": source["anchor_names"],
            "layer": source["layer"],
            "variant": variant,
            "target_rank": target_rank,
            "anchor_rank": anchor_rank,
            **parsed,
        }
    if len(lookup) != EXPECTED_CASES * 2:
        raise ControlError("Frozen exact-control keys are incomplete")
    return lookup


def _case_inputs(
    specs: Sequence[Mapping[str, Any]],
    modules: Sequence[tuple[str, Any]],
    embeddings: Mapping[str, Any],
    k0: Any,
    config: Mapping[str, Any],
    device: str,
) -> Iterable[dict[str, Any]]:
    import torch

    oce = config["oce"]
    normalization_eps = float(oce["normalization_eps"])
    with torch.inference_mode():
        for spec in specs:
            target_prompts = protocol.expanded_prompts(spec["targets"], config)
            anchor_prompts = protocol.expanded_prompts(spec["anchors"], config)
            target_embeddings = [embeddings[value] for value in target_prompts]
            anchor_embeddings = [embeddings[value] for value in anchor_prompts]
            retain_embeddings = [
                embeddings[value] for value in spec["retain_concepts"]
            ]
            for layer_index, (layer, module) in enumerate(modules, start=1):
                weight = module.weight.detach().to(
                    device=device, dtype=torch.float64
                )
                target_columns = exact_control._projected_columns(
                    weight, target_embeddings, normalization_eps
                )
                anchor_columns = exact_control._projected_columns(
                    weight, anchor_embeddings, normalization_eps
                )
                target_basis, target_rank, target_rtol = (
                    exact_control._rank_basis(target_columns)
                )
                anchor_basis, anchor_rank, anchor_rtol = (
                    exact_control._rank_basis(anchor_columns)
                )
                if target_rank != anchor_rank:
                    raise ControlError(
                        f"Unequal rank at {spec['group_id']} {layer}: "
                        f"target={target_rank}, anchor={anchor_rank}"
                    )
                if target_rank != EXPECTED_RANK:
                    raise ControlError(
                        f"Unexpected equal rank at {spec['group_id']} {layer}: "
                        f"expected {EXPECTED_RANK}, got {target_rank}"
                    )
                preservation = exact_control._preservation_matrix(
                    weight, retain_embeddings, k0, oce
                )
                anchor_features = torch.stack(
                    [weight @ embedding for embedding in anchor_embeddings], dim=1
                )
                anchor_covariance = anchor_features @ anchor_features.T
                support_residual = exact_control._scalar(
                    torch.linalg.matrix_norm(
                        (torch.eye(
                            weight.shape[0], device=device, dtype=torch.float64
                        ) - anchor_basis @ anchor_basis.T)
                        @ anchor_features
                    )
                )
                support_scale = max(
                    1.0,
                    exact_control._scalar(
                        torch.linalg.matrix_norm(anchor_features)
                    ),
                )
                support_tolerance = (
                    1024.0
                    * weight.shape[0]
                    * torch.finfo(torch.float64).eps
                    * support_scale
                )
                if support_residual > support_tolerance:
                    raise ControlError(
                        "Anchor features are not numerically contained in H at "
                        f"{spec['group_id']} {layer}: residual={support_residual:.9g}, "
                        f"tolerance={support_tolerance:.9g}"
                    )
                yield {
                    "group": spec["group_id"],
                    "targets": spec["targets"],
                    "anchors": spec["anchors"],
                    "layer": layer,
                    "layer_index": layer_index,
                    "target_basis": target_basis,
                    "anchor_basis": anchor_basis,
                    "target_rank": target_rank,
                    "anchor_rank": anchor_rank,
                    "target_rtol": target_rtol,
                    "anchor_rtol": anchor_rtol,
                    "preservation": preservation,
                    "anchor_features": anchor_features,
                    "anchor_covariance": anchor_covariance,
                    "anchor_support_residual": support_residual,
                }


def _safe_ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 1e-15:
        raise ControlError(f"Unsafe Variant C denominator for {label}")
    return numerator / denominator


def _combined_rows(
    frozen: Mapping[tuple[str, str, str], Mapping[str, Any]],
    e_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(e_rows)
    if len(keys) != EXPECTED_CASES:
        raise ControlError(f"Expected 48 E cases, found {len(keys)}")
    for group, layer in keys:
        c = frozen[(group, layer, VARIANT_C)]
        d = frozen[(group, layer, VARIANT_D)]
        e = e_rows[(group, layer)]
        for variant, source in ((VARIANT_C, c), (VARIANT_D, d), (VARIANT_E, e)):
            s_value = float(source["normalized_preservation_distortion"])
            anchor_value = float(source["anchor_feature_drift"])
            c_s = float(c["normalized_preservation_distortion"])
            c_anchor = float(c["anchor_feature_drift"])
            d_anchor = float(d["anchor_feature_drift"])
            rows.append(
                {
                    "group": group,
                    "target_names": source["target_names"],
                    "anchor_names": source["anchor_names"],
                    "layer": layer,
                    "variant": variant,
                    "target_rank": source["target_rank"],
                    "anchor_rank": source["anchor_rank"],
                    "true_leakage": source["true_leakage"],
                    "projector_leakage_crosscheck": source[
                        "projector_leakage_crosscheck"
                    ],
                    "raw_frozen_s_preservation_loss": source[
                        "raw_preservation_loss"
                    ],
                    "normalized_frozen_s_preservation_distortion": s_value,
                    "frozen_s_delta_vs_variant_c": s_value - c_s,
                    "frozen_s_ratio_vs_variant_c": _safe_ratio(
                        s_value, c_s, "frozen-S preservation"
                    ),
                    "anchor_feature_drift": anchor_value,
                    "anchor_delta_vs_variant_c": anchor_value - c_anchor,
                    "anchor_ratio_vs_variant_c": _safe_ratio(
                        anchor_value, c_anchor, "anchor drift"
                    ),
                    "anchor_delta_vs_variant_d": anchor_value - d_anchor,
                    "orthogonality_residual": source["orthogonality_residual"],
                    "exact_mapping_residual": source["exact_mapping_residual"],
                    "determinant": source["determinant"],
                }
            )
    if len(rows) != EXPECTED_CASES * 3:
        raise ControlError(f"Expected 144 C/D/E rows, found {len(rows)}")
    return rows


def _paired(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["group"], row["layer"], row["variant"]): row for row in rows
    }
    keys = sorted({(row["group"], row["layer"]) for row in rows})
    pairs = []
    for group, layer in keys:
        pairs.append(
            {
                "group": group,
                "layer": layer,
                "c": lookup[(group, layer, VARIANT_C)],
                "d": lookup[(group, layer, VARIANT_D)],
                "e": lookup[(group, layer, VARIANT_E)],
            }
        )
    if len(pairs) != EXPECTED_CASES:
        raise ControlError(f"Expected 48 paired C/D/E cases, got {len(pairs)}")
    return pairs


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ControlError("Cannot summarize an empty vector")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ControlError("Cannot summarize an empty vector")
    return float(statistics.fmean(materialized))


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ControlError("Cannot summarize an empty vector")
    return float(statistics.median(materialized))


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


def _material(delta: float, ratio: float, metric: str) -> bool:
    if metric == "anchor":
        return delta >= MATERIAL_ANCHOR_DELTA and ratio >= MATERIAL_ANCHOR_RATIO
    return delta >= MATERIAL_PRESERVE_DELTA and ratio >= MATERIAL_PRESERVE_RATIO


def _group_material(
    selected: Sequence[Mapping[str, Any]], metric: str
) -> bool:
    if metric == "anchor":
        deltas = [float(pair["e"]["anchor_delta_vs_variant_c"]) for pair in selected]
        ratios = [float(pair["e"]["anchor_ratio_vs_variant_c"]) for pair in selected]
        return (
            _median(deltas) >= MATERIAL_ANCHOR_DELTA
            and _median(ratios) >= MATERIAL_ANCHOR_RATIO
        )
    deltas = [float(pair["e"]["frozen_s_delta_vs_variant_c"]) for pair in selected]
    ratios = [float(pair["e"]["frozen_s_ratio_vs_variant_c"]) for pair in selected]
    return (
        _median(deltas) >= MATERIAL_PRESERVE_DELTA
        and _median(ratios) >= MATERIAL_PRESERVE_RATIO
    )


def _group_small(selected: Sequence[Mapping[str, Any]], metric: str) -> bool:
    if metric == "anchor":
        deltas = [float(pair["e"]["anchor_delta_vs_variant_c"]) for pair in selected]
        ratios = [float(pair["e"]["anchor_ratio_vs_variant_c"]) for pair in selected]
        return (
            _median(deltas) <= SMALL_ANCHOR_DELTA
            and _median(ratios) <= SMALL_ANCHOR_RATIO
        )
    deltas = [float(pair["e"]["frozen_s_delta_vs_variant_c"]) for pair in selected]
    ratios = [float(pair["e"]["frozen_s_ratio_vs_variant_c"]) for pair in selected]
    return (
        _median(deltas) <= MATERIAL_PRESERVE_DELTA
        and _median(ratios) <= SMALL_PRESERVE_RATIO
    )


def _primary_classification(pairs: Sequence[Mapping[str, Any]]) -> str:
    e_leakage = [float(pair["e"]["true_leakage"]) for pair in pairs]
    if max(e_leakage) > MAPPING_ZERO_TOLERANCE:
        raise ControlError(
            "Variant E did not reach numerical-zero leakage; treat this as an "
            "implementation/numerical bug before scientific interpretation"
        )
    anchor_material_count = sum(
        _material(
            float(pair["e"]["anchor_delta_vs_variant_c"]),
            float(pair["e"]["anchor_ratio_vs_variant_c"]),
            "anchor",
        )
        for pair in pairs
    )
    preserve_material_count = sum(
        _material(
            float(pair["e"]["frozen_s_delta_vs_variant_c"]),
            float(pair["e"]["frozen_s_ratio_vs_variant_c"]),
            "preserve",
        )
        for pair in pairs
    )
    groups = {
        group: [pair for pair in pairs if pair["group"] == group]
        for group in EXPECTED_GROUPS
    }
    anchor_all_groups = all(
        _group_material(selected, "anchor") for selected in groups.values()
    )
    preserve_all_groups = all(
        _group_material(selected, "preserve") for selected in groups.values()
    )
    if anchor_material_count >= CONSISTENT_LAYER_COUNT and anchor_all_groups:
        return "E1"
    anchor_clearly_small = (
        anchor_material_count <= SMALL_LAYER_COUNT
        and all(_group_small(selected, "anchor") for selected in groups.values())
    )
    preserve_clearly_small = (
        preserve_material_count <= SMALL_LAYER_COUNT
        and all(_group_small(selected, "preserve") for selected in groups.values())
    )
    if anchor_clearly_small and preserve_clearly_small:
        return "E2"
    if preserve_material_count >= CONSISTENT_LAYER_COUNT and preserve_all_groups:
        return "E3"
    return "ambiguous"


def _pareto_candidate_is_cheap(
    rows: Sequence[Mapping[str, Any]], lambda_value: float
) -> bool:
    selected = [row for row in rows if float(row["lambda"]) == lambda_value]
    if len(selected) != EXPECTED_CASES:
        raise ControlError(f"Incomplete Pareto rows for lambda={lambda_value}")
    layer_cheap = [
        float(row["anchor_delta_vs_variant_c"]) <= SMALL_ANCHOR_DELTA
        and float(row["anchor_ratio_vs_variant_c"]) <= SMALL_ANCHOR_RATIO
        and float(row["frozen_s_delta_vs_variant_c"])
        <= MATERIAL_PRESERVE_DELTA
        and float(row["frozen_s_ratio_vs_variant_c"])
        <= SMALL_PRESERVE_RATIO
        for row in selected
    ]
    if sum(layer_cheap) < CONSISTENT_LAYER_COUNT:
        return False
    for group in EXPECTED_GROUPS:
        group_rows = [row for row in selected if row["group"] == group]
        if not (
            _median(float(row["anchor_delta_vs_variant_c"]) for row in group_rows)
            <= SMALL_ANCHOR_DELTA
            and _median(float(row["anchor_ratio_vs_variant_c"]) for row in group_rows)
            <= SMALL_ANCHOR_RATIO
            and _median(float(row["frozen_s_delta_vs_variant_c"]) for row in group_rows)
            <= MATERIAL_PRESERVE_DELTA
            and _median(float(row["frozen_s_ratio_vs_variant_c"]) for row in group_rows)
            <= SMALL_PRESERVE_RATIO
        ):
            return False
    return True


def _final_classification(
    primary: str, pareto_rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str, bool]:
    if primary == "E1":
        return (
            "Outcome E1 — Exact orthogonal mapping has an unavoidable anchor-feature cost",
            "Even after explicitly optimizing anchor preservation over the entire exact orthogonal feasible family, exact target-subspace alignment still requires substantial movement of the original anchor features. Therefore the previous D2 result is not an artifact of omitting anchors from the frozen preservation covariance. This closes the remaining algebraic caveat and provides a direct motivation for an anchor-fixed non-orthogonal relaxation.",
            False,
        )
    if primary == "E2":
        return (
            "Outcome E2 — Pure orthogonality remains sufficient at matrix level",
            "The large anchor drift observed for Variant D was caused by optimizing a preservation covariance that omitted anchors, not by a fundamental orthogonal trade-off. AFR is not algebraically justified. The appropriate next direction is an exact orthogonal mapper rather than a non-orthogonal relaxation.",
            False,
        )
    sweet_lambdas = [
        value
        for value in PARETO_LAMBDAS
        if _pareto_candidate_is_cheap(pareto_rows, value)
    ]
    if sweet_lambdas:
        return (
            "Ambiguous — A simple exact-orthogonal Pareto candidate remains viable",
            "The anchor-minimum endpoint alone does not settle the matrix-level trade-off, and the fixed Pareto diagnostic contains at least one exact orthogonal candidate that keeps both measured preservation quantities close to Variant C. A non-orthogonal AFR editor is therefore not justified by this audit.",
            True,
        )
    return (
        "Outcome E3 — Exact orthogonal mapping exhibits a preservation trade-off rather than an anchor-only trade-off",
        "Pure orthogonality can choose to preserve the anchors or the frozen retain geometry more strongly, but the fixed closed-form Pareto diagnostic found no simple exact mapper that keeps both quantities close to Variant C. The evidence supports a multi-objective exact-mapping/preservation incompatibility rather than anchor impossibility alone.",
        False,
    )


def _aggregate_table(pairs: Sequence[Mapping[str, Any]], aggregate: Any) -> str:
    lines = [
        "| Group | C anchor | D anchor | E anchor | E-C anchor | E/C anchor | E-D anchor | C frozen-S | D frozen-S | E frozen-S | E-C frozen-S | E/C frozen-S |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in EXPECTED_GROUPS:
        selected = [pair for pair in pairs if pair["group"] == group]
        fields = [
            group,
            _fmt(aggregate(float(p["c"]["anchor_feature_drift"]) for p in selected)),
            _fmt(aggregate(float(p["d"]["anchor_feature_drift"]) for p in selected)),
            _fmt(aggregate(float(p["e"]["anchor_feature_drift"]) for p in selected)),
            _fmt(aggregate(float(p["e"]["anchor_delta_vs_variant_c"]) for p in selected)),
            _fmt(aggregate(float(p["e"]["anchor_ratio_vs_variant_c"]) for p in selected)),
            _fmt(aggregate(float(p["e"]["anchor_delta_vs_variant_d"]) for p in selected)),
            _fmt(aggregate(float(p["c"]["normalized_frozen_s_preservation_distortion"]) for p in selected)),
            _fmt(aggregate(float(p["d"]["normalized_frozen_s_preservation_distortion"]) for p in selected)),
            _fmt(aggregate(float(p["e"]["normalized_frozen_s_preservation_distortion"]) for p in selected)),
            _fmt(aggregate(float(p["e"]["frozen_s_delta_vs_variant_c"]) for p in selected)),
            _fmt(aggregate(float(p["e"]["frozen_s_ratio_vs_variant_c"]) for p in selected)),
        ]
        lines.append("| " + " | ".join(fields) + " |")
    return "\n".join(lines)


def _overall_table(pairs: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Variant | Mean leakage | Median leakage | Mean anchor drift | Median anchor drift | Mean frozen-S distortion | Median frozen-S distortion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("c", VARIANT_C),
        ("d", VARIANT_D),
        ("e", VARIANT_E),
    ):
        leakage = [float(pair[key]["true_leakage"]) for pair in pairs]
        anchor = [float(pair[key]["anchor_feature_drift"]) for pair in pairs]
        preservation = [
            float(pair[key]["normalized_frozen_s_preservation_distortion"])
            for pair in pairs
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt(_mean(leakage)),
                    _fmt(_median(leakage)),
                    _fmt(_mean(anchor)),
                    _fmt(_median(anchor)),
                    _fmt(_mean(preservation)),
                    _fmt(_median(preservation)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _pareto_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "Not triggered because the primary C/D/E result was decisive."
    lines = [
        "| lambda | Median leakage | Median anchor drift | Median frozen-S distortion | Cheap in >=36/48 and all groups? |",
        "|---:|---:|---:|---:|---:|",
    ]
    for value in PARETO_LAMBDAS:
        selected = [row for row in rows if float(row["lambda"]) == value]
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(value),
                    _fmt(_median(float(row["true_leakage"]) for row in selected)),
                    _fmt(_median(float(row["normalized_anchor_feature_drift"]) for row in selected)),
                    _fmt(_median(float(row["normalized_frozen_s_preservation_distortion"]) for row in selected)),
                    "yes" if _pareto_candidate_is_cheap(rows, value) else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render_report(
    rows: Sequence[Mapping[str, Any]],
    pareto_rows: Sequence[Mapping[str, Any]],
    synthetic: Mapping[str, Any],
    run_info: Mapping[str, Any],
) -> str:
    pairs = _paired(rows)
    primary = _primary_classification(pairs)
    outcome, conclusion, pareto_sweet_spot = _final_classification(
        primary, pareto_rows
    )
    e_leaks = [float(pair["e"]["true_leakage"]) for pair in pairs]
    e_anchor = [float(pair["e"]["anchor_feature_drift"]) for pair in pairs]
    e_anchor_delta_c = [float(pair["e"]["anchor_delta_vs_variant_c"]) for pair in pairs]
    e_anchor_ratio_c = [float(pair["e"]["anchor_ratio_vs_variant_c"]) for pair in pairs]
    e_anchor_delta_d = [float(pair["e"]["anchor_delta_vs_variant_d"]) for pair in pairs]
    e_s = [float(pair["e"]["normalized_frozen_s_preservation_distortion"]) for pair in pairs]
    e_s_delta_c = [float(pair["e"]["frozen_s_delta_vs_variant_c"]) for pair in pairs]
    e_s_ratio_c = [float(pair["e"]["frozen_s_ratio_vs_variant_c"]) for pair in pairs]
    anchor_material = sum(
        _material(delta, ratio, "anchor")
        for delta, ratio in zip(e_anchor_delta_c, e_anchor_ratio_c)
    )
    preserve_material = sum(
        _material(delta, ratio, "preserve")
        for delta, ratio in zip(e_s_delta_c, e_s_ratio_c)
    )
    e_orthogonality = [float(pair["e"]["orthogonality_residual"]) for pair in pairs]
    e_mapping = [float(pair["e"]["exact_mapping_residual"]) for pair in pairs]

    group_q2 = []
    group_q3 = []
    for group in EXPECTED_GROUPS:
        selected = [pair for pair in pairs if pair["group"] == group]
        group_q2.append(
            f"`{group}` E drift `{_fmt(_median(float(p['e']['anchor_feature_drift']) for p in selected))}`, "
            f"E-C `{_fmt(_median(float(p['e']['anchor_delta_vs_variant_c']) for p in selected))}`, "
            f"E/C `{_fmt(_median(float(p['e']['anchor_ratio_vs_variant_c']) for p in selected))}`"
        )
        group_q3.append(
            f"`{group}` E frozen-S `{_fmt(_median(float(p['e']['normalized_frozen_s_preservation_distortion']) for p in selected))}`, "
            f"E-C `{_fmt(_median(float(p['e']['frozen_s_delta_vs_variant_c']) for p in selected))}`, "
            f"E/C `{_fmt(_median(float(p['e']['frozen_s_ratio_vs_variant_c']) for p in selected))}`"
        )

    gate = ""
    if primary == "E1":
        gate = (
            "\n\n**Algebraic gate passed. The next step is AFR implementation with a "
            "pure-projection ablation.**"
        )
    elif primary == "E2":
        gate = "\n\n**AFR stops at the algebraic gate; consider an exact orthogonal mapper.**"

    return f"""# Anchor-minimum exact orthogonal control

## Scope and answer

This matrix-only control reuses the frozen float64 C/D rows and the exact-feasible-family implementation from the preceding control. Variant E minimizes anchor feature movement over the full exact orthogonal family by calling the same constrained Procrustes solver with `S_x=YY^T`. The qualified dogs, fruits, and balls Joint settings, including the matched balls anchors, remain unchanged. No image, evaluator, checkpoint, AFR implementation, or production OCE change was created.

**Classification: {outcome}.**

> {conclusion}

Variant E is an oracle-like control, not a repaired OCE method or a proposed editor.

## Derivation and orientation

For `P = H Q G^T + H_perp Q_perp G_perp^T`, substituting into `tr(P^T S_x)` yields `tr(Q^T H^T S_x G) + tr(Q_perp^T H_perp^T S_x G_perp)`. Standard Procrustes therefore gives `Q=U1 V1^T` from `H^T S_x G=U1 Sigma1 V1^T` and `Q_perp=U2 V2^T` from `H_perp^T S_x G_perp=U2 Sigma2 V2^T`. With `S_x=YY^T`, orthogonality makes minimizing `||PY-Y||F^2` equivalent to this trace maximization. Because `Y` lies in `span(H)`, the complement block can be zero or rank-deficient; its SVD completion need not be unique and is not treated as a solver failure.

## Synthetic QA

All four float64 tests passed for `d={synthetic['dimension']}`, `r={synthetic['rank']}`:

- exact feasibility: orthogonality `{_fmt(float(synthetic['orthogonality_residual']))}`, leakage `{_fmt(float(synthetic['true_leakage']))}`, projector residual `{_fmt(float(synthetic['projector_mapping_residual']))}`, mapping residual `{_fmt(float(synthetic['exact_mapping_residual']))}`;
- anchor optimality: closed-form normalized drift `{_fmt(float(synthetic['normalized_anchor_drift']))}` versus best of 256 random feasible maps `{_fmt(float(synthetic['best_of_256_random_feasible_anchor_drifts']))}`;
- closed-form value: direct trace `{_fmt(float(synthetic['closed_form_trace']))}`, nuclear-norm sum `{_fmt(float(synthetic['nuclear_norm_sum']))}`, error `{_fmt(float(synthetic['closed_form_trace_abs_error']))}`;
- known `G=H` case: normalized anchor drift `{_fmt(float(synthetic['special_g_equals_h_normalized_anchor_drift']))}` and mapping residual `{_fmt(float(synthetic['special_g_equals_h_mapping_residual']))}`.

## C/D/E group means

{_aggregate_table(pairs, _mean)}

## C/D/E group medians

{_aggregate_table(pairs, _median)}

## C/D/E overall aggregate

{_overall_table(pairs)}

Every `anchor` quantity means **anchor feature drift at the edited layer** only.

## Layer distributions

| Quantity | Min | Q25 | Median | Q75 | Max | Material layers |
|---|---:|---:|---:|---:|---:|---:|
| E true leakage | {_distribution(e_leaks)} | 48/48 checked |
| E normalized anchor drift | {_distribution(e_anchor)} | n/a |
| E-C anchor drift | {_distribution(e_anchor_delta_c)} | {anchor_material}/48 |
| E/C anchor drift | {_distribution(e_anchor_ratio_c)} | n/a |
| E-D anchor drift | {_distribution(e_anchor_delta_d)} | n/a |
| E normalized frozen-S distortion | {_distribution(e_s)} | n/a |
| E-C frozen-S distortion | {_distribution(e_s_delta_c)} | {preserve_material}/48 |
| E/C frozen-S distortion | {_distribution(e_s_ratio_c)} | n/a |

## Q1 — Does E keep numerical-zero leakage in all 48 layers?

**Yes.** Maximum E leakage is `{_fmt(max(e_leaks))}` under the `{MAPPING_ZERO_TOLERANCE}` fail-closed threshold. Maximum exact mapping residual is `{_fmt(max(e_mapping))}` and maximum `||P^T P-I||F` is `{_fmt(max(e_orthogonality))}`. All 48 cases retain target rank = anchor rank = 12.

## Q2 — What is the minimum achievable anchor drift?

Overall E median is `{_fmt(_median(e_anchor))}`. Overall E-C median is `{_fmt(_median(e_anchor_delta_c))}`, E/C median is `{_fmt(_median(e_anchor_ratio_c))}`, and E-D median is `{_fmt(_median(e_anchor_delta_d))}`. Group medians: {"; ".join(group_q2)}. The complete min/Q25/median/Q75/max distributions are above. E is materially above C in `{anchor_material}/48` layers using the frozen absolute (`{MATERIAL_ANCHOR_DELTA}`) and ratio (`{MATERIAL_ANCHOR_RATIO}`) thresholds.

## Q3 — Does minimizing anchor drift worsen frozen-S preservation?

Overall E normalized frozen-S median is `{_fmt(_median(e_s))}`. Overall E-C median is `{_fmt(_median(e_s_delta_c))}` and E/C median is `{_fmt(_median(e_s_ratio_c))}`. Group medians: {"; ".join(group_q3)}. E is materially above C in `{preserve_material}/48` layers under the frozen preservation thresholds (`{MATERIAL_PRESERVE_DELTA}`, `{MATERIAL_PRESERVE_RATIO}`).

## Q4 — Was D's large anchor drift only an optimizer-choice artifact?

The answer follows from E, not from D: E is the closed-form minimum-anchor-drift member of the entire exact feasible family. Its residual anchor cost relative to C and its separately evaluated frozen-S cost are reported above. Under the preregistered decision rule, the primary result is `{primary}`.

## Q5 — Final classification

**{outcome}.** {conclusion}{gate}

## Optional fixed Pareto diagnostic

{_pareto_summary(pareto_rows)}

Pareto triggered: `{'yes' if pareto_rows else 'no'}`. Pareto sweet spot under the preregistered small-effect rule: `{'yes' if pareto_sweet_spot else 'no'}`.

## Reproducibility and QA

- CSV: `results_anchor_min_control.csv` ({len(rows)} rows = 48 cases x 3 controls)
- Optional Pareto CSV: `{'results_anchor_pareto.csv' if pareto_rows else 'not created'}`
- Computation dtype: float64 after loading frozen production tensors
- Config SHA-256: `{run_info['config_sha256']}`
- Anchors SHA-256: `{run_info['anchors_sha256']}`
- Qualification SHA-256: `{run_info['qualification_sha256']}`
- K0 SHA-256: `{run_info['k0_sha256']}`
- Frozen C/D CSV SHA-256: `{run_info['prior_exact_results_sha256']}`
- Rank relative-tolerance range: `{_fmt(float(run_info['rank_rtol_min']))}` to `{_fmt(float(run_info['rank_rtol_max']))}`
- Maximum anchor-support residual `||(I-HH^T)Y||F`: `{_fmt(float(run_info['max_anchor_support_residual']))}`
- Maximum anchor-objective nuclear-norm trace error: `{_fmt(float(run_info['max_anchor_trace_error']))}`
- Endpoint optimality cross-checks passed: E anchor drift <= D anchor drift and D frozen-S loss <= E frozen-S loss in every layer (tolerance `1e-8`)
- Variant E maximum orthogonality residual: `{_fmt(max(e_orthogonality))}`
- Variant E determinant is metadata only; primary constraint is O(d)
- Runtime: `{run_info['runtime_seconds']:.1f}` seconds
"""


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.17g}"
    return value


def _atomic_write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: _csv_value(row.get(field)) for field in fields}
            for row in rows
        )
    temporary.replace(path)


def _compute_pareto(
    specs: Sequence[Mapping[str, Any]],
    modules: Sequence[tuple[str, Any]],
    embeddings: Mapping[str, Any],
    k0: Any,
    config: Mapping[str, Any],
    device: str,
    combined_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    baseline = {
        (row["group"], row["layer"], row["variant"]): row
        for row in combined_rows
    }
    rows: list[dict[str, Any]] = []
    for case in _case_inputs(specs, modules, embeddings, k0, config, device):
        group = case["group"]
        layer = case["layer"]
        c = baseline[(group, layer, VARIANT_C)]
        d = baseline[(group, layer, VARIANT_D)]
        trace_s = exact_control._scalar(torch.trace(case["preservation"]))
        trace_a = exact_control._scalar(torch.trace(case["anchor_covariance"]))
        if trace_s <= 0 or trace_a <= 0:
            raise ControlError(f"Unsafe Pareto covariance scale at {group} {layer}")
        normalized_s = case["preservation"] / trace_s
        normalized_a = case["anchor_covariance"] / trace_a
        for lambda_value in PARETO_LAMBDAS:
            objective = normalized_s + lambda_value * normalized_a
            transform, q_star, _, _ = exact_control.exact_orthogonal_mapper(
                case["target_basis"], case["anchor_basis"], objective
            )
            mapping_residual = exact_control._scalar(
                torch.linalg.matrix_norm(
                    transform @ case["target_basis"]
                    - case["anchor_basis"] @ q_star
                )
            )
            metrics = exact_control._metrics(
                transform,
                case["target_basis"],
                case["anchor_basis"],
                case["anchor_features"],
                case["preservation"],
                exact_mapping_residual=mapping_residual,
            )
            if (
                float(metrics["true_leakage"]) > MAPPING_ZERO_TOLERANCE
                or mapping_residual > MAPPING_ZERO_TOLERANCE
                or float(metrics["orthogonality_residual"])
                > FLOAT64_ORTHOGONALITY_TOLERANCE
            ):
                raise ControlError(f"Pareto exact-map QA failed at {group} {layer}")
            anchor_value = float(metrics["anchor_feature_drift"])
            s_value = float(metrics["normalized_preservation_distortion"])
            if lambda_value == 0.0:
                d_anchor_error = abs(
                    anchor_value - float(d["anchor_feature_drift"])
                )
                d_s_error = abs(
                    s_value
                    - float(d["normalized_frozen_s_preservation_distortion"])
                )
                if max(d_anchor_error, d_s_error) > 1e-6:
                    raise ControlError(
                        "Pareto lambda=0 failed to reproduce frozen Variant D at "
                        f"{group} {layer}: anchor_error={d_anchor_error:.9g}, "
                        f"frozen_s_error={d_s_error:.9g}"
                    )
            c_anchor = float(c["anchor_feature_drift"])
            c_s = float(c["normalized_frozen_s_preservation_distortion"])
            rows.append(
                {
                    "group": group,
                    "target_names": json.dumps(case["targets"], ensure_ascii=False),
                    "anchor_names": json.dumps(case["anchors"], ensure_ascii=False),
                    "layer": layer,
                    "lambda": lambda_value,
                    "target_rank": case["target_rank"],
                    "anchor_rank": case["anchor_rank"],
                    "true_leakage": metrics["true_leakage"],
                    "normalized_anchor_feature_drift": anchor_value,
                    "anchor_delta_vs_variant_c": anchor_value - c_anchor,
                    "anchor_ratio_vs_variant_c": _safe_ratio(
                        anchor_value, c_anchor, "Pareto anchor drift"
                    ),
                    "normalized_frozen_s_preservation_distortion": s_value,
                    "frozen_s_delta_vs_variant_c": s_value - c_s,
                    "frozen_s_ratio_vs_variant_c": _safe_ratio(
                        s_value, c_s, "Pareto frozen-S preservation"
                    ),
                    "orthogonality_residual": metrics["orthogonality_residual"],
                    "exact_mapping_residual": mapping_residual,
                }
            )
        print(f"[anchor Pareto] {group} {layer}", flush=True)
    if len(rows) != EXPECTED_CASES * len(PARETO_LAMBDAS):
        raise ControlError(f"Expected 288 Pareto rows, found {len(rows)}")
    return rows


def execute(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    if platform.system() == "Darwin":
        raise ControlError(
            "Full anchor-minimum control is forbidden on the local Mac; run it "
            "in the active GPU-server project environment."
        )

    import time

    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        raise ControlError(
            "Full anchor-minimum control requires the configured GPU server"
        )
    started = time.monotonic()
    synthetic = synthetic_unit_tests()
    config_path = args.config.resolve()
    prior_path = args.prior_exact_results.resolve()
    _sha_guard("config", protocol.sha256(config_path), FROZEN_CONFIG_SHA256)
    frozen = _load_frozen_cd(prior_path)
    config, anchors = protocol.load_protocol(config_path)
    if config["oce"].get("anchor_in_local_retain") is not False:
        raise ControlError("Frozen protocol no longer has anchor_in_local_retain=false")
    _sha_guard(
        "anchors",
        protocol.sha256(Path(config["_resolved"]["anchors_path"])),
        FROZEN_ANCHORS_SHA256,
    )
    specs, _, qualification_path = prior_audit._validated_joint_specs(config, anchors)
    if tuple(spec["group_id"] for spec in specs) != EXPECTED_GROUPS:
        raise ControlError("Frozen anchor-minimum groups are not dogs/fruits/balls")
    _sha_guard(
        "qualification",
        protocol.sha256(qualification_path),
        FROZEN_QUALIFICATION_SHA256,
    )
    plan, _, _ = checkpoint_builder.build_plan(config_path)
    k0_matrix, k0_metadata = checkpoint_builder.validate_k0(plan, config)
    _sha_guard("K0", k0_metadata["artifact_sha256"], FROZEN_K0_SHA256)

    device = str(config["model"]["device"])
    pipe = DiffusionPipeline.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)
    modules = checkpoint_builder._projection_modules(pipe.unet)
    if len(modules) != 16:
        raise ControlError(f"Expected 16 edited layers, found {len(modules)}")

    prompt_order: list[str] = []
    for spec in specs:
        prompt_order.extend(protocol.expanded_prompts(spec["targets"], config))
        prompt_order.extend(protocol.expanded_prompts(spec["anchors"], config))
        prompt_order.extend(spec["retain_concepts"])
    prompt_order = list(dict.fromkeys(prompt_order))
    encoded_float32 = {
        prompt_text: checkpoint_builder._encode_prompt(
            pipe, prompt_text, device, torch.float32
        )
        for prompt_text in prompt_order
    }
    embeddings = {
        key: value.to(device=device, dtype=torch.float64)
        for key, value in encoded_float32.items()
    }
    k0 = k0_matrix.to(device=device, dtype=torch.float64)

    e_rows: dict[tuple[str, str], dict[str, Any]] = {}
    rank_rtols: list[float] = []
    support_residuals: list[float] = []
    trace_errors: list[float] = []
    for case in _case_inputs(specs, modules, embeddings, k0, config, device):
        key = (case["group"], case["layer"])
        c = frozen[(case["group"], case["layer"], VARIANT_C)]
        d = frozen[(case["group"], case["layer"], VARIANT_D)]
        target_names = json.dumps(case["targets"], ensure_ascii=False)
        anchor_names = json.dumps(case["anchors"], ensure_ascii=False)
        for prior in (c, d):
            if (
                prior["target_names"] != target_names
                or prior["anchor_names"] != anchor_names
            ):
                raise ControlError(
                    f"Frozen target/anchor registry mismatch at {key}"
                )
        transform, q_star, _, trace_qa = exact_control.exact_orthogonal_mapper(
            case["target_basis"],
            case["anchor_basis"],
            case["anchor_covariance"],
        )
        mapping_residual = exact_control._scalar(
            torch.linalg.matrix_norm(
                transform @ case["target_basis"]
                - case["anchor_basis"] @ q_star
            )
        )
        metrics = exact_control._metrics(
            transform,
            case["target_basis"],
            case["anchor_basis"],
            case["anchor_features"],
            case["preservation"],
            exact_mapping_residual=mapping_residual,
        )
        if float(metrics["true_leakage"]) > MAPPING_ZERO_TOLERANCE:
            raise ControlError(f"E leakage is not numerical zero at {key}")
        if mapping_residual > MAPPING_ZERO_TOLERANCE:
            raise ControlError(f"E exact mapping residual failed at {key}")
        if (
            float(metrics["orthogonality_residual"])
            > FLOAT64_ORTHOGONALITY_TOLERANCE
        ):
            raise ControlError(f"E float64 orthogonality failed at {key}")
        endpoint_tolerance = 1e-8
        if (
            float(metrics["anchor_feature_drift"])
            > float(d["anchor_feature_drift"]) + endpoint_tolerance
        ):
            raise ControlError(
                f"Anchor-optimal E is worse than feasible D at {key}"
            )
        if (
            float(d["normalized_preservation_distortion"])
            > float(metrics["normalized_preservation_distortion"])
            + endpoint_tolerance
        ):
            raise ControlError(
                f"Frozen-S-optimal D is worse than feasible E at {key}"
            )

        anchor_energy = exact_control._scalar(
            exact_control._sqnorm(case["anchor_features"])
        )
        direct_anchor_loss = (
            float(metrics["anchor_feature_drift"]) * anchor_energy
        )
        trace_anchor_loss = 2.0 * anchor_energy - 2.0 * trace_qa["achieved_trace"]
        identity_error = abs(direct_anchor_loss - trace_anchor_loss)
        identity_tolerance = (
            2048.0
            * transform.shape[0]
            * torch.finfo(torch.float64).eps
            * max(1.0, abs(direct_anchor_loss), abs(trace_anchor_loss))
        )
        if identity_error > identity_tolerance:
            raise ControlError(
                f"Anchor objective identity failed at {key}: "
                f"error={identity_error:.9g}, tolerance={identity_tolerance:.9g}"
            )
        e_rows[key] = {
            "group": case["group"],
            "target_names": target_names,
            "anchor_names": anchor_names,
            "layer": case["layer"],
            "variant": VARIANT_E,
            "target_rank": case["target_rank"],
            "anchor_rank": case["anchor_rank"],
            **metrics,
        }
        rank_rtols.extend([case["target_rtol"], case["anchor_rtol"]])
        support_residuals.append(case["anchor_support_residual"])
        trace_errors.append(trace_qa["trace_abs_error"])
        print(
            f"[anchor minimum] {case['group']} layer "
            f"{case['layer_index']}/{len(modules)}",
            flush=True,
        )

    combined = _combined_rows(frozen, e_rows)
    primary = _primary_classification(_paired(combined))
    pareto_rows: list[dict[str, Any]] = []
    if primary in ("E3", "ambiguous"):
        pareto_rows = _compute_pareto(
            specs, modules, embeddings, k0, config, device, combined
        )
    run_info = {
        "config_sha256": protocol.sha256(config_path),
        "anchors_sha256": protocol.sha256(
            Path(config["_resolved"]["anchors_path"])
        ),
        "qualification_sha256": protocol.sha256(qualification_path),
        "k0_sha256": k0_metadata["artifact_sha256"],
        "prior_exact_results_sha256": exact_control._sha256(prior_path),
        "rank_rtol_min": min(rank_rtols),
        "rank_rtol_max": max(rank_rtols),
        "max_anchor_support_residual": max(support_residuals),
        "max_anchor_trace_error": max(trace_errors),
        "runtime_seconds": time.monotonic() - started,
    }
    report = _render_report(combined, pareto_rows, synthetic, run_info)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results_anchor_min_control.csv"
    report_path = output_dir / "REPORT_anchor_min_control.md"
    pareto_path = output_dir / "results_anchor_pareto.csv" if pareto_rows else None
    _atomic_write_csv(csv_path, CSV_FIELDS, combined)
    if pareto_path is not None:
        _atomic_write_csv(pareto_path, PARETO_CSV_FIELDS, pareto_rows)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(report_path)
    return csv_path, report_path, pareto_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.synthetic_only:
        print(json.dumps(synthetic_unit_tests(), indent=2, sort_keys=True))
        return 0
    csv_path, report_path, pareto_path = execute(args)
    print(f"[complete] CSV: {csv_path}")
    print(f"[complete] report: {report_path}")
    if pareto_path is not None:
        print(f"[complete] Pareto CSV: {pareto_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
