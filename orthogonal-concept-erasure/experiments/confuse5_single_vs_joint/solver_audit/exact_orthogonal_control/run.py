#!/usr/bin/env python3
"""Float64 exact-orthogonal mapping control for frozen Confuse5 matrices.

This is an oracle-like matrix control, not an editor or a new OCE variant.  It
never mutates model weights, writes checkpoints, or invokes image generation or
image evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
AUDIT_ROOT = HERE.parent
EXPERIMENT_ROOT = AUDIT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import protocol  # noqa: E402
import run as checkpoint_builder  # noqa: E402
import solver_audit as prior_audit  # noqa: E402


VARIANT_C = "C_objective_faithful"
VARIANT_D = "D_exact_orthogonal"
EXPECTED_GROUPS = ("dogs", "fruits", "balls")
EXPECTED_RANK = 12
EXPECTED_CASES = 48
MAPPING_ZERO_TOLERANCE = 1e-10
FLOAT64_ORTHOGONALITY_TOLERANCE = 1e-10
PRIOR_LEAKAGE_ABS_TOLERANCE = 0.01
PRIOR_PRESERVE_REL_TOLERANCE = 0.01

# Pre-registered interpretation thresholds.  D2 requires consistency across
# at least 75% of layers and a material median in every group.  D1 is reserved
# for a clearly small penalty, not a heterogeneous one; everything between is
# D3.
CONSISTENT_LAYER_COUNT = 36
SMALL_LAYER_COUNT = 12
MATERIAL_PRESERVE_DELTA = 0.01
MATERIAL_PRESERVE_RATIO = 1.25
SMALL_PRESERVE_RATIO = 1.10
MATERIAL_ANCHOR_DELTA = 0.05
MATERIAL_ANCHOR_RATIO = 1.25
SMALL_ANCHOR_DELTA = 0.02
SMALL_ANCHOR_RATIO = 1.10

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
    "raw_preservation_loss",
    "normalized_preservation_distortion",
    "preservation_delta_vs_variant_c",
    "preservation_ratio_vs_variant_c",
    "anchor_feature_drift",
    "orthogonality_residual",
    "exact_mapping_residual",
    "determinant",
]


class ControlError(RuntimeError):
    """Raised when a mathematical, protocol, or numerical invariant fails."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument(
        "--prior-results",
        type=Path,
        default=AUDIT_ROOT / "results.csv",
        help="Frozen float32 solver-audit CSV used only for Variant C reproduction QA.",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run the small float64 constrained-Procrustes tests only.",
    )
    return parser.parse_args(argv)


def _scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _sqnorm(value: Any) -> Any:
    return value.square().sum()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _procrustes(matrix: Any) -> tuple[Any, Any]:
    """Solve max_Q tr(Q^T matrix), Q in O(n), as Q=U@Vh."""
    import torch

    left, singular_values, right_h = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right_h, singular_values


def _rank_basis(columns: Any) -> tuple[Any, int, float]:
    """Float64 rank-revealing SVD basis with a relative machine-epsilon rule."""
    import torch

    if columns.dtype != torch.float64:
        raise ControlError("Rank basis input must be float64")
    left, singular_values, _ = torch.linalg.svd(columns, full_matrices=False)
    relative_tolerance = max(columns.shape) * torch.finfo(columns.dtype).eps
    threshold = relative_tolerance * _scalar(singular_values[0])
    rank = int((singular_values > threshold).sum().item())
    if rank < 1:
        raise ControlError("Rank-revealing SVD returned an empty subspace")
    return left[:, :rank], rank, relative_tolerance


def _orthogonal_complement(basis: Any) -> Any:
    """Complete an orthonormal thin basis without changing its leading block."""
    import torch

    dimension, rank = basis.shape
    complete_q, _ = torch.linalg.qr(basis, mode="complete")
    complement = complete_q[:, rank:]
    identity = torch.eye(dimension, device=basis.device, dtype=basis.dtype)
    residual = torch.linalg.matrix_norm(
        torch.cat([basis, complement], dim=1).T
        @ torch.cat([basis, complement], dim=1)
        - identity
    )
    tolerance = 256.0 * dimension * torch.finfo(basis.dtype).eps
    if _scalar(residual) > tolerance:
        raise ControlError(
            "Failed to construct a float64 orthogonal completion: "
            f"residual={_scalar(residual):.9g}, tolerance={tolerance:.9g}"
        )
    return complement


def exact_orthogonal_mapper(
    target_basis: Any, anchor_basis: Any, preservation_matrix: Any
) -> tuple[Any, Any, Any, dict[str, float]]:
    """Return the preservation-optimal exact target-to-anchor O(d) map.

    With P = H Q G^T + H_perp Q_perp G_perp^T,
    tr(P^T S) separates into tr(Q^T H^T S G) and
    tr(Q_perp^T H_perp^T S G_perp).  Each is a standard Procrustes problem.
    """
    import torch

    if target_basis.shape != anchor_basis.shape:
        raise ControlError("Exact equal-rank mapper received unequal basis shapes")
    if preservation_matrix.dtype != torch.float64:
        raise ControlError("Exact mapper requires float64 matrices")
    target_perp = _orthogonal_complement(target_basis)
    anchor_perp = _orthogonal_complement(anchor_basis)
    target_block = anchor_basis.T @ preservation_matrix @ target_basis
    complement_block = anchor_perp.T @ preservation_matrix @ target_perp
    target_rotation, target_singular_values = _procrustes(target_block)
    complement_rotation, complement_singular_values = _procrustes(complement_block)
    transform = (
        anchor_basis @ target_rotation @ target_basis.T
        + anchor_perp @ complement_rotation @ target_perp.T
    )
    achieved_trace = torch.trace(transform.T @ preservation_matrix)
    theoretical_trace = (
        target_singular_values.sum() + complement_singular_values.sum()
    )
    dimension = transform.shape[0]
    trace_tolerance = (
        512.0
        * dimension
        * torch.finfo(transform.dtype).eps
        * max(1.0, abs(_scalar(theoretical_trace)))
    )
    trace_error = abs(_scalar(achieved_trace - theoretical_trace))
    if trace_error > trace_tolerance:
        raise ControlError(
            "Exact constrained objective missed its nuclear-norm value: "
            f"error={trace_error:.9g}, tolerance={trace_tolerance:.9g}"
        )
    return transform, target_rotation, complement_rotation, {
        "achieved_trace": _scalar(achieved_trace),
        "theoretical_trace": _scalar(theoretical_trace),
        "trace_abs_error": trace_error,
        "trace_tolerance": trace_tolerance,
    }


def _random_orthogonal(dimension: int, generator: Any, dtype: Any) -> Any:
    import torch

    matrix = torch.randn(dimension, dimension, generator=generator, dtype=dtype)
    left, _, right_h = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right_h


def _preservation_loss(transform: Any, preservation_matrix: Any) -> Any:
    import torch

    identity = torch.eye(
        transform.shape[0], device=transform.device, dtype=transform.dtype
    )
    delta = transform - identity
    return torch.sum((delta @ preservation_matrix) * delta)


def synthetic_unit_tests() -> dict[str, Any]:
    """Verify feasibility, feasible-family optimality, and closed-form value."""
    import torch

    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(20260812)
    dimension, rank = 4, 2
    target_full = _random_orthogonal(dimension, generator, dtype)
    anchor_full = _random_orthogonal(dimension, generator, dtype)
    target = target_full[:, :rank]
    anchor = anchor_full[:, :rank]
    raw = torch.randn(dimension, dimension, generator=generator, dtype=dtype)
    preservation = raw @ raw.T + 0.25 * torch.eye(dimension, dtype=dtype)
    exact, target_rotation, _, trace_qa = exact_orthogonal_mapper(
        target, anchor, preservation
    )
    identity = torch.eye(dimension, dtype=dtype)
    anchor_projector = anchor @ anchor.T
    target_projector = target @ target.T
    orthogonality = torch.linalg.matrix_norm(exact.T @ exact - identity)
    leakage = _sqnorm((identity - anchor_projector) @ exact @ target) / rank
    projector_residual = torch.linalg.matrix_norm(
        exact @ target_projector @ exact.T - anchor_projector
    )
    mapping_residual = torch.linalg.matrix_norm(
        exact @ target - anchor @ target_rotation
    )
    feasibility_tolerance = 1e-11
    if max(
        _scalar(orthogonality),
        _scalar(leakage),
        _scalar(projector_residual),
        _scalar(mapping_residual),
    ) > feasibility_tolerance:
        raise ControlError("Synthetic exact-mapping feasibility test failed")

    exact_loss = _scalar(_preservation_loss(exact, preservation))
    target_perp = _orthogonal_complement(target)
    anchor_perp = _orthogonal_complement(anchor)
    random_losses: list[float] = []
    for _ in range(256):
        random_target = _random_orthogonal(rank, generator, dtype)
        random_perp = _random_orthogonal(dimension - rank, generator, dtype)
        candidate = (
            anchor @ random_target @ target.T
            + anchor_perp @ random_perp @ target_perp.T
        )
        random_losses.append(_scalar(_preservation_loss(candidate, preservation)))
    optimality_tolerance = 1e-11 * max(1.0, abs(exact_loss))
    if exact_loss > min(random_losses) + optimality_tolerance:
        raise ControlError(
            "Synthetic exact mapper was worse than a random feasible transform"
        )

    return {
        "passed": True,
        "dtype": "torch.float64",
        "dimension": dimension,
        "rank": rank,
        "orthogonality_residual": _scalar(orthogonality),
        "true_leakage": _scalar(leakage),
        "projector_mapping_residual": _scalar(projector_residual),
        "exact_mapping_residual": _scalar(mapping_residual),
        "closed_form_trace": trace_qa["achieved_trace"],
        "nuclear_norm_sum": trace_qa["theoretical_trace"],
        "closed_form_trace_abs_error": trace_qa["trace_abs_error"],
        "exact_preservation_loss": exact_loss,
        "best_of_256_random_feasible_losses": min(random_losses),
    }


def _projected_columns(
    weight: Any, embeddings: Sequence[Any], normalization_eps: float
) -> Any:
    import torch

    columns = []
    for embedding in embeddings:
        vector = weight @ embedding
        columns.append(vector / (torch.linalg.vector_norm(vector) + normalization_eps))
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
    global_prior = float(oce["lambda_0"]) * (weight @ k0 @ weight.T)
    repo_regularizer = float(oce["lamb_repo_regularizer"]) * (weight @ weight.T)
    preservation = local + global_prior + repo_regularizer
    symmetry_residual = _scalar(
        torch.linalg.matrix_norm(preservation - preservation.T)
    )
    symmetry_tolerance = (
        256.0
        * weight.shape[0]
        * torch.finfo(weight.dtype).eps
        * max(1.0, _scalar(torch.linalg.matrix_norm(preservation)))
    )
    if symmetry_residual > symmetry_tolerance:
        raise ControlError(
            "Preservation matrix is not symmetric within float64 tolerance: "
            f"residual={symmetry_residual:.9g}, tolerance={symmetry_tolerance:.9g}"
        )
    # Remove only antisymmetric roundoff; the mathematical S is unchanged.
    return 0.5 * (preservation + preservation.T)


def _metrics(
    transform: Any,
    target_basis: Any,
    anchor_basis: Any,
    anchor_features: Any,
    preservation_matrix: Any,
    *,
    exact_mapping_residual: float | None,
) -> dict[str, float | None]:
    import torch

    dimension = transform.shape[0]
    rank = target_basis.shape[1]
    identity = torch.eye(dimension, device=transform.device, dtype=transform.dtype)
    target_projector = target_basis @ target_basis.T
    anchor_projector = anchor_basis @ anchor_basis.T
    anchor_complement = identity - anchor_projector
    transformed_target = transform @ target_basis
    true_leakage = _scalar(_sqnorm(anchor_complement @ transformed_target)) / rank
    transformed_projector = transform @ target_projector @ transform.T
    projector_leakage = _scalar(
        _sqnorm(anchor_complement @ transformed_projector)
    ) / rank
    crosscheck_error = abs(true_leakage - projector_leakage)
    crosscheck_tolerance = (
        512.0
        * dimension
        * torch.finfo(transform.dtype).eps
        * max(1.0, abs(true_leakage), abs(projector_leakage))
    )
    if crosscheck_error > crosscheck_tolerance:
        raise ControlError(
            "Float64 leakage cross-check failed: "
            f"basis={true_leakage:.9g}, projector={projector_leakage:.9g}, "
            f"error={crosscheck_error:.9g}, tolerance={crosscheck_tolerance:.9g}"
        )
    raw_preservation = _scalar(_preservation_loss(transform, preservation_matrix))
    preservation_scale = _scalar(torch.trace(preservation_matrix))
    scale_tolerance = (
        256.0
        * dimension
        * torch.finfo(transform.dtype).eps
        * max(1.0, _scalar(torch.linalg.matrix_norm(preservation_matrix)))
    )
    if preservation_scale <= scale_tolerance:
        raise ControlError(
            "trace(S) is not safely positive: "
            f"trace={preservation_scale:.9g}, tolerance={scale_tolerance:.9g}"
        )
    normalized_preservation = raw_preservation / preservation_scale
    anchor_denominator = _scalar(_sqnorm(anchor_features))
    if anchor_denominator <= 0:
        raise ControlError("Anchor feature drift denominator is zero")
    anchor_drift = _scalar(_sqnorm(transform @ anchor_features - anchor_features))
    anchor_drift /= anchor_denominator
    orthogonality = _scalar(
        torch.linalg.matrix_norm(transform.T @ transform - identity)
    )
    metrics = {
        "true_leakage": true_leakage,
        "projector_leakage_crosscheck": projector_leakage,
        "raw_preservation_loss": raw_preservation,
        "normalized_preservation_distortion": normalized_preservation,
        "anchor_feature_drift": anchor_drift,
        "orthogonality_residual": orthogonality,
        "exact_mapping_residual": exact_mapping_residual,
        "determinant": _scalar(torch.linalg.det(transform)),
    }
    numeric_values = [
        float(value) for value in metrics.values() if value is not None
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ControlError("A control metric is non-finite")
    if raw_preservation < -scale_tolerance:
        raise ControlError(
            "PSD preservation matrix produced a materially negative loss: "
            f"loss={raw_preservation:.9g}, tolerance={scale_tolerance:.9g}"
        )
    return metrics


def _load_prior_variant_c(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not path.is_file():
        raise ControlError(f"Prior solver-audit CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows if row.get("solver_variant") == prior_audit.VARIANT_FAITHFUL
    ]
    if len(selected) != EXPECTED_CASES:
        raise ControlError(
            f"Expected {EXPECTED_CASES} prior Variant C rows, found {len(selected)}"
        )
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in selected:
        key = (row["group"], row["layer_name"])
        if key in lookup:
            raise ControlError(f"Duplicate prior Variant C key: {key}")
        lookup[key] = {
            "true_leakage": float(row["true_leakage"]),
            "raw_preservation_loss": float(row["preserve_loss"]),
        }
    return lookup


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


def _paired_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["group"], row["layer"], row["variant"]): row for row in rows
    }
    keys = sorted({(row["group"], row["layer"]) for row in rows})
    pairs = []
    for group, layer in keys:
        c = lookup[(group, layer, VARIANT_C)]
        d = lookup[(group, layer, VARIANT_D)]
        c_preserve = float(c["normalized_preservation_distortion"])
        d_preserve = float(d["normalized_preservation_distortion"])
        c_anchor = float(c["anchor_feature_drift"])
        d_anchor = float(d["anchor_feature_drift"])
        pairs.append(
            {
                "group": group,
                "layer": layer,
                "c": c,
                "d": d,
                "preserve_delta": d_preserve - c_preserve,
                "preserve_ratio": d_preserve / c_preserve if c_preserve > 1e-15 else None,
                "anchor_delta": d_anchor - c_anchor,
                "anchor_ratio": d_anchor / c_anchor if c_anchor > 1e-15 else None,
            }
        )
    if len(pairs) != EXPECTED_CASES:
        raise ControlError(f"Expected {EXPECTED_CASES} paired cases, got {len(pairs)}")
    return pairs


def _material_preservation(pair: Mapping[str, Any]) -> bool:
    ratio = pair["preserve_ratio"]
    return (
        ratio is not None
        and float(pair["preserve_delta"]) >= MATERIAL_PRESERVE_DELTA
        and float(ratio) >= MATERIAL_PRESERVE_RATIO
    )


def _material_anchor(pair: Mapping[str, Any]) -> bool:
    ratio = pair["anchor_ratio"]
    return (
        ratio is not None
        and float(pair["anchor_delta"]) >= MATERIAL_ANCHOR_DELTA
        and float(ratio) >= MATERIAL_ANCHOR_RATIO
    )


def _group_material(
    group_pairs: Sequence[Mapping[str, Any]], metric: str
) -> bool:
    if metric == "preserve":
        deltas = [float(pair["preserve_delta"]) for pair in group_pairs]
        ratios = [
            float(pair["preserve_ratio"])
            for pair in group_pairs
            if pair["preserve_ratio"] is not None
        ]
        return (
            _median(deltas) >= MATERIAL_PRESERVE_DELTA
            and _median(ratios) >= MATERIAL_PRESERVE_RATIO
        )
    deltas = [float(pair["anchor_delta"]) for pair in group_pairs]
    ratios = [
        float(pair["anchor_ratio"])
        for pair in group_pairs
        if pair["anchor_ratio"] is not None
    ]
    return (
        _median(deltas) >= MATERIAL_ANCHOR_DELTA
        and _median(ratios) >= MATERIAL_ANCHOR_RATIO
    )


def _classify(pairs: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    exact_leakage = [float(pair["d"]["true_leakage"]) for pair in pairs]
    if max(exact_leakage) > MAPPING_ZERO_TOLERANCE:
        raise ControlError(
            "Exact mapper did not reach numerical-zero leakage; treat this as an "
            "implementation/numerical bug before scientific interpretation"
        )
    preserve_material_count = sum(_material_preservation(pair) for pair in pairs)
    anchor_material_count = sum(_material_anchor(pair) for pair in pairs)
    groups = {
        group: [pair for pair in pairs if pair["group"] == group]
        for group in EXPECTED_GROUPS
    }
    preserve_all_groups = all(
        _group_material(group_pairs, "preserve") for group_pairs in groups.values()
    )
    anchor_all_groups = all(
        _group_material(group_pairs, "anchor") for group_pairs in groups.values()
    )
    if (
        preserve_material_count >= CONSISTENT_LAYER_COUNT and preserve_all_groups
    ) or (anchor_material_count >= CONSISTENT_LAYER_COUNT and anchor_all_groups):
        return (
            "Outcome D2 — Exact orthogonal mapping has a clear preservation cost",
            "Pure orthogonality can satisfy the geometric target-subspace mapping, "
            "but only at a substantial preservation / anchor-feature cost. This "
            "supports a genuine incompatibility between exact orthogonal target "
            "mapping and preservation, motivating the anchor-fixed non-orthogonal "
            "relaxation.",
        )

    preserve_deltas = [float(pair["preserve_delta"]) for pair in pairs]
    preserve_ratios = [
        float(pair["preserve_ratio"])
        for pair in pairs
        if pair["preserve_ratio"] is not None
    ]
    anchor_deltas = [float(pair["anchor_delta"]) for pair in pairs]
    anchor_ratios = [
        float(pair["anchor_ratio"])
        for pair in pairs
        if pair["anchor_ratio"] is not None
    ]
    clearly_small = (
        preserve_material_count <= SMALL_LAYER_COUNT
        and anchor_material_count <= SMALL_LAYER_COUNT
        and _median(preserve_deltas) <= MATERIAL_PRESERVE_DELTA
        and _median(preserve_ratios) <= SMALL_PRESERVE_RATIO
        and _median(anchor_deltas) <= SMALL_ANCHOR_DELTA
        and _median(anchor_ratios) <= SMALL_ANCHOR_RATIO
    )
    if clearly_small:
        return (
            "Outcome D1 — Surrogate failure dominates",
            "Exact target-subspace alignment is achievable within the orthogonal "
            "family without a clear algebraic preservation penalty. The current "
            "evidence points primarily to OCE's Eq.18 surrogate rather than "
            "orthogonality itself. AFR is not yet justified; the next candidate "
            "should remain orthogonal and directly enforce transformed-subspace "
            "alignment.",
        )
    return (
        "Outcome D3 — Mixed / ambiguous",
        "The matrix-level trade-off is mixed. A minimal image-level comparison of "
        "objective-faithful OCE versus exact orthogonal mapping is needed before "
        "introducing a non-orthogonal editor.",
    )


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


def _render_report(
    rows: Sequence[Mapping[str, Any]],
    synthetic: Mapping[str, Any],
    run_info: Mapping[str, Any],
) -> str:
    pairs = _paired_cases(rows)
    outcome, conclusion = _classify(pairs)
    exact_leaks = [float(pair["d"]["true_leakage"]) for pair in pairs]
    c_leaks = [float(pair["c"]["true_leakage"]) for pair in pairs]
    preserve_deltas = [float(pair["preserve_delta"]) for pair in pairs]
    preserve_ratios = [
        float(pair["preserve_ratio"])
        for pair in pairs
        if pair["preserve_ratio"] is not None
    ]
    anchor_deltas = [float(pair["anchor_delta"]) for pair in pairs]
    d_orthogonality = [
        float(pair["d"]["orthogonality_residual"]) for pair in pairs
    ]
    d_mapping_residual = [
        float(pair["d"]["exact_mapping_residual"]) for pair in pairs
    ]

    mean_table = [
        "| Group | C leakage | D leakage | C norm. preserve | D norm. preserve | D-C preserve | Preserve ratio | C anchor drift | D anchor drift | D-C anchor drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    median_table = list(mean_table)
    for group in EXPECTED_GROUPS:
        selected = [pair for pair in pairs if pair["group"] == group]
        for table, aggregate in ((mean_table, _mean), (median_table, _median)):
            table.append(
                "| "
                + " | ".join(
                    [
                        group,
                        _fmt(aggregate(float(p["c"]["true_leakage"]) for p in selected)),
                        _fmt(aggregate(float(p["d"]["true_leakage"]) for p in selected)),
                        _fmt(aggregate(float(p["c"]["normalized_preservation_distortion"]) for p in selected)),
                        _fmt(aggregate(float(p["d"]["normalized_preservation_distortion"]) for p in selected)),
                        _fmt(aggregate(float(p["preserve_delta"]) for p in selected)),
                        _fmt(aggregate(float(p["preserve_ratio"]) for p in selected if p["preserve_ratio"] is not None)),
                        _fmt(aggregate(float(p["c"]["anchor_feature_drift"]) for p in selected)),
                        _fmt(aggregate(float(p["d"]["anchor_feature_drift"]) for p in selected)),
                        _fmt(aggregate(float(p["anchor_delta"]) for p in selected)),
                    ]
                )
                + " |"
            )

    group_q2 = []
    group_q3 = []
    for group in EXPECTED_GROUPS:
        selected = [pair for pair in pairs if pair["group"] == group]
        group_q2.append(
            f"`{group}` D-C `{_fmt(_median(float(p['preserve_delta']) for p in selected))}`, "
            f"ratio `{_fmt(_median(float(p['preserve_ratio']) for p in selected if p['preserve_ratio'] is not None))}`"
        )
        group_q3.append(
            f"`{group}` D-C `{_fmt(_median(float(p['anchor_delta']) for p in selected))}`"
        )
    preserve_positive = sum(value > 0 for value in preserve_deltas)
    anchor_positive = sum(value > 0 for value in anchor_deltas)
    preserve_material = sum(_material_preservation(pair) for pair in pairs)
    anchor_material = sum(_material_anchor(pair) for pair in pairs)

    return f"""# Exact orthogonal target-subspace mapping control

## Scope and answer

This final matrix-only control uses the frozen qualified Joint settings for `dogs`, `fruits`, and `balls` across the same 16 edited `attn2.to_v` layers. Targets, matched per-target anchors (including `basketball` and `baseball` for balls), prompt expansion, K0, local retain concepts, scales, and layers are unchanged. All audit linear algebra and metrics are float64. No image, image evaluator, editor checkpoint, AFR implementation, or production OCE change was created.

**Classification: {outcome}.**

> {conclusion}

`D_exact_orthogonal` is an oracle-like best feasible orthogonal control, not a repaired OCE method or a proposed contribution.

## Derivation and orientation

For `P = H Q G^T + H_perp Q_perp G_perp^T`, orthonormal completion makes `P` orthogonal and gives `P G = H Q`. Cyclic trace expansion yields

`tr(P^T S) = tr(Q^T H^T S G) + tr(Q_perp^T H_perp^T S G_perp)`.

Thus the constrained maximization separates into two standard Procrustes problems. If `H^T S G = U1 Sigma1 V1^T` and `H_perp^T S G_perp = U2 Sigma2 V2^T`, the correct orientations are `Q*=U1 V1^T` and `Q_perp*=U2 V2^T`. The achieved maximum is the sum of the two nuclear norms. Because `P` is orthogonal and `S` symmetric, minimizing `tr[(P-I)S(P-I)^T]` is equivalent to maximizing `tr(P^T S)`.

## Synthetic tests

All three float64 tests passed for `d={synthetic['dimension']}`, `r={synthetic['rank']}`:

- feasibility: orthogonality residual `{_fmt(float(synthetic['orthogonality_residual']))}`, true leakage `{_fmt(float(synthetic['true_leakage']))}`, projector-mapping residual `{_fmt(float(synthetic['projector_mapping_residual']))}`, and exact mapping residual `{_fmt(float(synthetic['exact_mapping_residual']))}`;
- feasible-family optimality: closed-form preservation loss `{_fmt(float(synthetic['exact_preservation_loss']))}` versus best of 256 random feasible transforms `{_fmt(float(synthetic['best_of_256_random_feasible_losses']))}`;
- closed-form value: direct trace `{_fmt(float(synthetic['closed_form_trace']))}` versus nuclear-norm sum `{_fmt(float(synthetic['nuclear_norm_sum']))}`, absolute error `{_fmt(float(synthetic['closed_form_trace_abs_error']))}`.

## Float64 baseline reproduction

Variant C was rebuilt from frozen inputs as `-lambda_e (I-R*)R + S`, followed by standard O(d) Procrustes. Compared with the prior float32 CSV, maximum absolute leakage difference was `{_fmt(float(run_info['prior_c_max_leakage_abs_difference']))}` and maximum relative raw-preservation difference was `{_fmt(float(run_info['prior_c_max_preservation_relative_difference']))}`. Both passed the fail-closed thresholds (`{PRIOR_LEAKAGE_ABS_TOLERANCE}` and `{PRIOR_PRESERVE_REL_TOLERANCE}`). Float64 Variant C leakage has median `{_fmt(_median(c_leaks))}` and minimum `{_fmt(min(c_leaks))}`; these values are reproduced rather than hard-coded.

## Group-level means

{chr(10).join(mean_table)}

## Group-level medians

{chr(10).join(median_table)}

`anchor drift` in these tables means **anchor feature drift at the edited layer** only.

## Layer distribution

| Quantity (D-C unless noted) | Min | Q25 | Median | Q75 | Max | Positive/material layers |
|---|---:|---:|---:|---:|---:|---:|
| Normalized preservation difference | {_distribution(preserve_deltas)} | {preserve_positive}/48 positive; {preserve_material}/48 material |
| Normalized preservation ratio D/C | {_distribution(preserve_ratios)} | n/a |
| Anchor feature drift difference | {_distribution(anchor_deltas)} | {anchor_positive}/48 positive; {anchor_material}/48 material |
| D true leakage | {_distribution(exact_leaks)} | 48/48 checked |

## Q1 — Can all 48 layers reach numerical-zero leakage?

**Yes.** Maximum Variant D true leakage was `{_fmt(max(exact_leaks))}` under the preregistered `{MAPPING_ZERO_TOLERANCE}` threshold. All 48 layers had target rank = anchor rank = 12. Maximum exact mapping residual was `{_fmt(max(d_mapping_residual))}` and maximum `||P^T P-I||F` was `{_fmt(max(d_orthogonality))}`. The projector leakage cross-check also passed in every layer.

## Q2 — What preservation cost does exact mapping require?

Overall median normalized D-C preservation distortion was `{_fmt(_median(preserve_deltas))}` and median ratio D/C was `{_fmt(_median(preserve_ratios))}`. Group medians were {"; ".join(group_q2)}. The full layer distribution is reported above; {preserve_positive}/48 layers increased and {preserve_material}/48 crossed both the absolute (`{MATERIAL_PRESERVE_DELTA}`) and ratio (`{MATERIAL_PRESERVE_RATIO}`) materiality thresholds.

## Q3 — Does exact mapping increase anchor feature drift?

Overall median D-C anchor feature drift was `{_fmt(_median(anchor_deltas))}`. Group medians were {"; ".join(group_q3)}. Across layers, {anchor_positive}/48 increased and {anchor_material}/48 crossed both the absolute (`{MATERIAL_ANCHOR_DELTA}`) and ratio (`{MATERIAL_ANCHOR_RATIO}`) materiality thresholds.

## Q4 — Interpretation

Statement A is confirmed: equal 12-dimensional ranks make exact orthogonal target-subspace mapping feasible, so the prior high leakage did **not** prove mathematical infeasibility of orthogonal mapping.

Statement B is the actual decision test: whether that exact mapping has a consistent, substantial preservation or anchor-feature cost. The preregistered rule assigns D2 only if a material penalty occurs in at least 36/48 layers and all three group medians; D1 requires clearly small effects, while heterogeneous intermediate evidence is D3. Under that rule, the result is **{outcome}**.

## Reproducibility and QA

- CSV: `results_exact_control.csv` ({len(rows)} rows = 48 cases x 2 controls)
- Computation dtype: float64 after loading frozen production tensors
- Config SHA-256: `{run_info['config_sha256']}`
- Anchors SHA-256: `{run_info['anchors_sha256']}`
- Qualification SHA-256: `{run_info['qualification_sha256']}`
- K0 SHA-256: `{run_info['k0_sha256']}`
- Prior solver-audit CSV SHA-256: `{run_info['prior_results_sha256']}`
- Rank relative-tolerance range: `{_fmt(float(run_info['rank_rtol_min']))}` to `{_fmt(float(run_info['rank_rtol_max']))}`
- Variant C maximum orthogonality residual: `{_fmt(float(run_info['c_max_orthogonality_residual']))}`
- Variant D maximum orthogonality residual: `{_fmt(max(d_orthogonality))}`
- Variant D determinant is recorded only as numerical metadata, not used for interpretation
- Runtime: `{run_info['runtime_seconds']:.1f}` seconds
"""


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.17g}"
    return value


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def execute(args: argparse.Namespace) -> tuple[Path, Path]:
    if platform.system() == "Darwin":
        raise ControlError(
            "Full exact-orthogonal control is forbidden on the local Mac; run it in "
            "the active GPU-server project environment."
        )

    import time

    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        raise ControlError("Full exact-orthogonal control requires the configured GPU server")
    started = time.monotonic()
    synthetic = synthetic_unit_tests()
    config_path = args.config.resolve()
    config, anchors = protocol.load_protocol(config_path)
    specs, _, qualification_path = prior_audit._validated_joint_specs(config, anchors)
    if tuple(spec["group_id"] for spec in specs) != EXPECTED_GROUPS:
        raise ControlError("Frozen exact-control groups are not dogs/fruits/balls")
    prior_results_path = args.prior_results.resolve()
    prior_variant_c = _load_prior_variant_c(prior_results_path)
    plan, _, _ = checkpoint_builder.build_plan(config_path)
    k0_matrix, k0_metadata = checkpoint_builder.validate_k0(plan, config)

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
    # Encoding reproduces the frozen production readout in float32.  Every
    # tensor enters the audit calculation only after explicit float64 casting.
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
    oce = config["oce"]
    normalization_eps = float(oce["normalization_eps"])
    erase_scale = float(oce["lambda_e"])
    rows: list[dict[str, Any]] = []
    rank_rtols: list[float] = []
    prior_leakage_differences: list[float] = []
    prior_preservation_relative_differences: list[float] = []

    with torch.inference_mode():
        for spec in specs:
            target_prompts = protocol.expanded_prompts(spec["targets"], config)
            anchor_prompts = protocol.expanded_prompts(spec["anchors"], config)
            target_embeddings = [embeddings[value] for value in target_prompts]
            anchor_embeddings = [embeddings[value] for value in anchor_prompts]
            retain_embeddings = [embeddings[value] for value in spec["retain_concepts"]]
            for layer_index, (layer, module) in enumerate(modules, start=1):
                weight = module.weight.detach().to(device=device, dtype=torch.float64)
                target_columns = _projected_columns(
                    weight, target_embeddings, normalization_eps
                )
                anchor_columns = _projected_columns(
                    weight, anchor_embeddings, normalization_eps
                )
                target_basis, target_rank, target_rtol = _rank_basis(target_columns)
                anchor_basis, anchor_rank, anchor_rtol = _rank_basis(anchor_columns)
                rank_rtols.extend([target_rtol, anchor_rtol])
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

                preservation = _preservation_matrix(
                    weight, retain_embeddings, k0, oce
                )
                identity = torch.eye(
                    weight.shape[0], device=device, dtype=torch.float64
                )
                target_projector = target_basis @ target_basis.T
                anchor_projector = anchor_basis @ anchor_basis.T
                faithful_matrix = (
                    -erase_scale * (identity - anchor_projector) @ target_projector
                    + preservation
                )
                faithful, _ = _procrustes(faithful_matrix)
                exact, q_star, _, trace_qa = exact_orthogonal_mapper(
                    target_basis, anchor_basis, preservation
                )
                mapping_residual = _scalar(
                    torch.linalg.matrix_norm(
                        exact @ target_basis - anchor_basis @ q_star
                    )
                )
                anchor_features = torch.stack(
                    [weight @ embedding for embedding in anchor_embeddings], dim=1
                )
                c_metrics = _metrics(
                    faithful,
                    target_basis,
                    anchor_basis,
                    anchor_features,
                    preservation,
                    exact_mapping_residual=None,
                )
                d_metrics = _metrics(
                    exact,
                    target_basis,
                    anchor_basis,
                    anchor_features,
                    preservation,
                    exact_mapping_residual=mapping_residual,
                )
                for variant, metrics in ((VARIANT_C, c_metrics), (VARIANT_D, d_metrics)):
                    if (
                        float(metrics["orthogonality_residual"])
                        > FLOAT64_ORTHOGONALITY_TOLERANCE
                    ):
                        raise ControlError(
                            f"{variant} float64 orthogonality failed at "
                            f"{spec['group_id']} {layer}: "
                            f"residual={metrics['orthogonality_residual']:.9g}"
                        )
                if float(d_metrics["true_leakage"]) > MAPPING_ZERO_TOLERANCE:
                    raise ControlError(
                        f"Exact leakage is not numerical zero at {spec['group_id']} "
                        f"{layer}: {d_metrics['true_leakage']:.9g}"
                    )
                if mapping_residual > MAPPING_ZERO_TOLERANCE:
                    raise ControlError(
                        f"Exact mapping residual failed at {spec['group_id']} {layer}: "
                        f"{mapping_residual:.9g}"
                    )
                if trace_qa["trace_abs_error"] > trace_qa["trace_tolerance"]:
                    raise ControlError("Internal exact-objective QA was not satisfied")

                prior = prior_variant_c[(spec["group_id"], layer)]
                leakage_difference = abs(
                    float(c_metrics["true_leakage"]) - prior["true_leakage"]
                )
                preserve_denominator = max(
                    abs(prior["raw_preservation_loss"]), 1.0
                )
                preservation_relative_difference = abs(
                    float(c_metrics["raw_preservation_loss"])
                    - prior["raw_preservation_loss"]
                ) / preserve_denominator
                prior_leakage_differences.append(leakage_difference)
                prior_preservation_relative_differences.append(
                    preservation_relative_difference
                )
                if leakage_difference > PRIOR_LEAKAGE_ABS_TOLERANCE:
                    raise ControlError(
                        "Float64 Variant C differs materially from prior float32 leakage "
                        f"at {spec['group_id']} {layer}: difference={leakage_difference:.9g}"
                    )
                if preservation_relative_difference > PRIOR_PRESERVE_REL_TOLERANCE:
                    raise ControlError(
                        "Float64 Variant C differs materially from prior float32 "
                        f"preservation at {spec['group_id']} {layer}: "
                        f"relative_difference={preservation_relative_difference:.9g}"
                    )

                c_normalized = float(c_metrics["normalized_preservation_distortion"])
                d_normalized = float(d_metrics["normalized_preservation_distortion"])
                if c_normalized <= 1e-15:
                    raise ControlError(
                        f"Variant C normalized preservation denominator is unsafe at {layer}"
                    )
                delta = d_normalized - c_normalized
                ratio = d_normalized / c_normalized
                common = {
                    "group": spec["group_id"],
                    "target_names": json.dumps(spec["targets"], ensure_ascii=False),
                    "anchor_names": json.dumps(spec["anchors"], ensure_ascii=False),
                    "layer": layer,
                    "target_rank": target_rank,
                    "anchor_rank": anchor_rank,
                }
                rows.extend(
                    [
                        {
                            **common,
                            "variant": VARIANT_C,
                            **c_metrics,
                            "preservation_delta_vs_variant_c": 0.0,
                            "preservation_ratio_vs_variant_c": 1.0,
                        },
                        {
                            **common,
                            "variant": VARIANT_D,
                            **d_metrics,
                            "preservation_delta_vs_variant_c": delta,
                            "preservation_ratio_vs_variant_c": ratio,
                        },
                    ]
                )
                print(
                    f"[exact orthogonal control] {spec['group_id']} layer "
                    f"{layer_index}/{len(modules)}",
                    flush=True,
                )

    if len(rows) != EXPECTED_CASES * 2:
        raise ControlError(f"Expected 96 result rows, got {len(rows)}")
    run_info = {
        "config_sha256": protocol.sha256(config_path),
        "anchors_sha256": protocol.sha256(Path(config["_resolved"]["anchors_path"])),
        "qualification_sha256": protocol.sha256(qualification_path),
        "k0_sha256": k0_metadata["artifact_sha256"],
        "prior_results_sha256": _sha256(prior_results_path),
        "prior_c_max_leakage_abs_difference": max(prior_leakage_differences),
        "prior_c_max_preservation_relative_difference": max(
            prior_preservation_relative_differences
        ),
        "rank_rtol_min": min(rank_rtols),
        "rank_rtol_max": max(rank_rtols),
        "c_max_orthogonality_residual": max(
            float(row["orthogonality_residual"])
            for row in rows
            if row["variant"] == VARIANT_C
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    # Render/classify before creating either output so failed scientific QA is
    # fail-closed and cannot leave a partial result artifact.
    report = _render_report(rows, synthetic, run_info)
    csv_rows = [
        {field: _csv_value(row.get(field)) for field in CSV_FIELDS} for row in rows
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results_exact_control.csv"
    report_path = output_dir / "REPORT_exact_control.md"
    _atomic_write_csv(csv_path, csv_rows)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(report_path)
    return csv_path, report_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.synthetic_only:
        print(json.dumps(synthetic_unit_tests(), indent=2, sort_keys=True))
        return 0
    csv_path, report_path = execute(args)
    print(f"[complete] CSV: {csv_path}")
    print(f"[complete] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
