"""Pure float64 linear algebra for AFR and its projection ablation."""

from __future__ import annotations

import math
from typing import Any, Mapping


class AFRError(RuntimeError):
    """Raised when an AFR algebraic or numerical invariant fails."""


def scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def sqnorm(value: Any) -> Any:
    return value.square().sum()


def procrustes(matrix: Any) -> tuple[Any, Any]:
    """Solve max_Q tr(Q^T M), Q in O(n), as Q=U@Vh."""
    import torch

    left, singular_values, right_h = torch.linalg.svd(
        matrix, full_matrices=False
    )
    return left @ right_h, singular_values


def rank_basis(columns: Any) -> tuple[Any, int, float, float]:
    """Return a float64 rank-revealing left-SVD basis."""
    import torch

    if columns.dtype != torch.float64:
        raise AFRError("rank_basis requires float64 input")
    left, singular_values, _ = torch.linalg.svd(columns, full_matrices=False)
    relative_tolerance = max(columns.shape) * torch.finfo(columns.dtype).eps
    sigma_max = scalar(singular_values[0]) if singular_values.numel() else 0.0
    threshold = relative_tolerance * sigma_max
    rank = int((singular_values > threshold).sum().item())
    if rank < 1:
        raise AFRError("Rank-revealing SVD returned an empty subspace")
    return left[:, :rank], rank, relative_tolerance, threshold


def orthogonal_complement(basis: Any) -> Any:
    """Complete a thin orthonormal basis without changing its leading block."""
    import torch

    dimension, rank = basis.shape
    full, _ = torch.linalg.qr(basis, mode="complete")
    complement = full[:, rank:]
    identity = torch.eye(dimension, device=basis.device, dtype=basis.dtype)
    joined = torch.cat([basis, complement], dim=1)
    residual = scalar(torch.linalg.matrix_norm(joined.T @ joined - identity))
    tolerance = 256.0 * dimension * torch.finfo(basis.dtype).eps
    if residual > tolerance:
        raise AFRError(
            "Float64 orthogonal completion failed: "
            f"residual={residual:.9g}, tolerance={tolerance:.9g}"
        )
    return complement


def preservation_loss(transform: Any, preservation: Any) -> Any:
    import torch

    identity = torch.eye(
        transform.shape[0], device=transform.device, dtype=transform.dtype
    )
    delta = transform - identity
    return torch.sum((delta @ preservation) * delta)


def _identity(dimension: int, reference: Any) -> Any:
    import torch

    return torch.eye(
        dimension, device=reference.device, dtype=reference.dtype
    )


def build_afr_transforms(
    target_features: Any,
    anchor_features: Any,
    preservation: Any,
    *,
    alpha: float,
) -> dict[str, Any]:
    """Build D_alpha and the anchor-fixed preservation-optimal compensation.

    The variable part of
      min_P tr[(P D-I) S (P D-I)^T], P^T P=I, P H=H
    is -2 tr(P^T S D).  With
      P=H H^T + H_perp Q H_perp^T,
    standard Procrustes therefore acts on
      M_perp=H_perp^T S D H_perp
    with Q=U V^T.
    """
    import torch

    if not 0.0 <= alpha <= 1.0:
        raise AFRError(f"alpha must lie in [0,1], got {alpha}")
    for name, value in (
        ("target_features", target_features),
        ("anchor_features", anchor_features),
        ("preservation", preservation),
    ):
        if value.dtype != torch.float64:
            raise AFRError(f"{name} must be float64")
    dimension = target_features.shape[0]
    if anchor_features.shape[0] != dimension:
        raise AFRError("Target and anchor feature dimensions differ")
    if preservation.shape != (dimension, dimension):
        raise AFRError("Preservation covariance shape differs from feature dimension")

    anchor_basis, anchor_rank, anchor_rtol, anchor_threshold = rank_basis(
        anchor_features
    )
    identity = _identity(dimension, target_features)
    anchor_projector = anchor_basis @ anchor_basis.T

    # alpha=0 is an explicit no-op.  Do not rely on an arbitrary SVD of a
    # zero residual covariance to happen to return identity.
    if alpha == 0.0:
        return {
            "anchor_basis": anchor_basis,
            "anchor_rank": anchor_rank,
            "anchor_rank_rtol": anchor_rtol,
            "anchor_rank_threshold": anchor_threshold,
            "residual_basis": target_features[:, :0],
            "residual_rank": 0,
            "residual_rank_rtol": None,
            "residual_rank_threshold": None,
            "residual": (identity - anchor_projector) @ target_features,
            "residual_projector": torch.zeros_like(identity),
            "D": identity,
            "P": identity,
            "T_projection": identity,
            "T_afr": identity,
            "Q": identity[:0, :0],
            "compensation_singular_values": identity[:0, 0],
            "compensation_trace": scalar(torch.trace(preservation)),
            "compensation_nuclear_norm": scalar(torch.trace(preservation)),
            "compensation_trace_error": 0.0,
            "alpha": alpha,
            "explicit_noop": True,
        }

    residual = (identity - anchor_projector) @ target_features
    residual_basis, residual_rank, residual_rtol, residual_threshold = rank_basis(
        residual
    )
    residual_projector = residual_basis @ residual_basis.T
    contraction = identity - alpha * residual_projector

    anchor_perp = orthogonal_complement(anchor_basis)
    complement_block = anchor_perp.T @ preservation @ contraction @ anchor_perp
    complement_rotation, singular_values = procrustes(complement_block)
    compensation = (
        anchor_projector
        + anchor_perp @ complement_rotation @ anchor_perp.T
    )
    transform_projection = contraction
    transform_afr = compensation @ contraction

    achieved_trace = scalar(torch.trace(compensation.T @ preservation @ contraction))
    fixed_trace = scalar(torch.trace(anchor_basis.T @ preservation @ contraction @ anchor_basis))
    theoretical_trace = fixed_trace + scalar(singular_values.sum())
    trace_error = abs(achieved_trace - theoretical_trace)
    trace_tolerance = (
        1024.0
        * dimension
        * torch.finfo(torch.float64).eps
        * max(1.0, abs(theoretical_trace))
    )
    if trace_error > trace_tolerance:
        raise AFRError(
            "Anchor-fixed Procrustes missed its closed-form trace optimum: "
            f"error={trace_error:.9g}, tolerance={trace_tolerance:.9g}"
        )

    return {
        "anchor_basis": anchor_basis,
        "anchor_rank": anchor_rank,
        "anchor_rank_rtol": anchor_rtol,
        "anchor_rank_threshold": anchor_threshold,
        "residual_basis": residual_basis,
        "residual_rank": residual_rank,
        "residual_rank_rtol": residual_rtol,
        "residual_rank_threshold": residual_threshold,
        "residual": residual,
        "residual_projector": residual_projector,
        "D": contraction,
        "P": compensation,
        "Q": complement_rotation,
        "T_projection": transform_projection,
        "T_afr": transform_afr,
        "compensation_singular_values": singular_values,
        "compensation_trace": achieved_trace,
        "compensation_nuclear_norm": theoretical_trace,
        "compensation_trace_error": trace_error,
        "compensation_trace_tolerance": trace_tolerance,
        "alpha": alpha,
        "explicit_noop": False,
    }


def transform_metrics(
    transform: Any,
    target_features: Any,
    target_basis: Any,
    anchor_features: Any,
    anchor_basis: Any,
    preservation: Any,
) -> dict[str, float]:
    """Evaluate metrics valid for a general, possibly non-orthogonal T."""
    import torch

    dimension = transform.shape[0]
    identity = _identity(dimension, transform)
    anchor_projector = anchor_basis @ anchor_basis.T
    anchor_complement = identity - anchor_projector
    feature_denominator = scalar(sqnorm(target_features))
    anchor_denominator = scalar(sqnorm(anchor_features))
    if feature_denominator <= 0.0 or anchor_denominator <= 0.0:
        raise AFRError("Target/anchor metric denominator is not positive")
    target_rank = target_basis.shape[1]
    feature_leakage = (
        scalar(sqnorm(anchor_complement @ transform @ target_features))
        / feature_denominator
    )
    basis_leakage = (
        scalar(sqnorm(anchor_complement @ transform @ target_basis))
        / target_rank
    )
    anchor_error = (
        scalar(sqnorm(transform @ anchor_features - anchor_features))
        / anchor_denominator
    )
    raw_preservation = scalar(preservation_loss(transform, preservation))
    preservation_scale = scalar(torch.trace(preservation))
    if preservation_scale <= 0.0:
        raise AFRError("trace(S) is not positive")
    normalized_preservation = raw_preservation / preservation_scale
    values = {
        "target_feature_leakage": feature_leakage,
        "target_basis_leakage": basis_leakage,
        "anchor_feature_error": anchor_error,
        "raw_frozen_s_distortion": raw_preservation,
        "normalized_frozen_s_distortion": normalized_preservation,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise AFRError("A transform metric is non-finite")
    return values


def structural_qa(parts: Mapping[str, Any]) -> dict[str, float]:
    import torch

    compensation = parts["P"]
    contraction = parts["D"]
    transform = parts["T_afr"]
    anchor_basis = parts["anchor_basis"]
    residual_basis = parts["residual_basis"]
    identity = _identity(compensation.shape[0], compensation)
    d_squared = contraction @ contraction
    return {
        "compensation_orthogonality_residual": scalar(
            torch.linalg.matrix_norm(compensation.T @ compensation - identity)
        ),
        "anchor_pointwise_fix_residual": scalar(
            torch.linalg.matrix_norm(compensation @ anchor_basis - anchor_basis)
        ),
        "anchor_residual_basis_orthogonality": scalar(
            torch.linalg.matrix_norm(anchor_basis.T @ residual_basis)
        ),
        "gram_residual": scalar(
            torch.linalg.matrix_norm(transform.T @ transform - d_squared)
        ),
        "projection_gram_match_residual": scalar(
            torch.linalg.matrix_norm(
                transform.T @ transform - contraction.T @ contraction
            )
        ),
        "compensation_magnitude": scalar(
            torch.linalg.matrix_norm(compensation - identity)
        ) / math.sqrt(compensation.shape[0]),
    }


def random_orthogonal(dimension: int, generator: Any, dtype: Any) -> Any:
    import torch

    raw = torch.randn(dimension, dimension, generator=generator, dtype=dtype)
    left, _, right_h = torch.linalg.svd(raw, full_matrices=False)
    return left @ right_h


def synthetic_unit_tests() -> dict[str, Any]:
    """Verify alpha endpoints, guarantees, orientation, optimum, and Gram QA."""
    import torch

    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(20260814)
    dimension, anchor_rank, target_count = 7, 2, 3
    full = random_orthogonal(dimension, generator, dtype)
    anchor_basis = full[:, :anchor_rank]
    residual_seed = full[:, anchor_rank : anchor_rank + target_count]
    anchor_features = anchor_basis @ torch.randn(
        anchor_rank, 4, generator=generator, dtype=dtype
    )
    target_features = (
        anchor_basis
        @ torch.randn(anchor_rank, target_count, generator=generator, dtype=dtype)
        + residual_seed
        @ torch.diag(torch.tensor([1.2, 0.8, 0.4], dtype=dtype))
    )
    raw = torch.randn(dimension, dimension, generator=generator, dtype=dtype)
    preservation = raw @ raw.T + 0.2 * torch.eye(dimension, dtype=dtype)

    alpha_zero = build_afr_transforms(
        target_features, anchor_features, preservation, alpha=0.0
    )
    identity = torch.eye(dimension, dtype=dtype)
    alpha_zero_error = scalar(
        torch.linalg.matrix_norm(alpha_zero["T_afr"] - identity)
    )
    if not alpha_zero["explicit_noop"] or alpha_zero_error != 0.0:
        raise AFRError("Synthetic alpha=0 explicit no-op failed")

    parts = build_afr_transforms(
        target_features, anchor_features, preservation, alpha=1.0
    )
    target_basis, _, _, _ = rank_basis(target_features)
    projection_metrics = transform_metrics(
        parts["T_projection"],
        target_features,
        target_basis,
        anchor_features,
        parts["anchor_basis"],
        preservation,
    )
    afr_metrics = transform_metrics(
        parts["T_afr"],
        target_features,
        target_basis,
        anchor_features,
        parts["anchor_basis"],
        preservation,
    )
    qa = structural_qa(parts)
    tolerance = 1e-11
    required_zero = [
        projection_metrics["target_feature_leakage"],
        afr_metrics["target_feature_leakage"],
        projection_metrics["anchor_feature_error"],
        afr_metrics["anchor_feature_error"],
        qa["compensation_orthogonality_residual"],
        qa["anchor_pointwise_fix_residual"],
        qa["anchor_residual_basis_orthogonality"],
        qa["gram_residual"],
    ]
    if max(required_zero) > tolerance:
        raise AFRError("Synthetic alpha=1 structural guarantee failed")
    if (
        afr_metrics["normalized_frozen_s_distortion"]
        > projection_metrics["normalized_frozen_s_distortion"] + tolerance
    ):
        raise AFRError("Synthetic AFR preservation is worse than projection")

    anchor_perp = orthogonal_complement(parts["anchor_basis"])
    random_losses = []
    for _ in range(256):
        rotation = random_orthogonal(
            anchor_perp.shape[1], generator, dtype
        )
        candidate_p = (
            parts["anchor_basis"] @ parts["anchor_basis"].T
            + anchor_perp @ rotation @ anchor_perp.T
        )
        random_losses.append(
            scalar(preservation_loss(candidate_p @ parts["D"], preservation))
        )
    optimal_loss = afr_metrics["raw_frozen_s_distortion"]
    if optimal_loss > min(random_losses) + 1e-10 * max(1.0, optimal_loss):
        raise AFRError("Synthetic AFR is worse than a random feasible compensation")

    return {
        "passed": True,
        "dtype": "torch.float64",
        "dimension": dimension,
        "anchor_rank": anchor_rank,
        "residual_rank": parts["residual_rank"],
        "alpha_zero_explicit_noop": alpha_zero["explicit_noop"],
        "alpha_zero_transform_error": alpha_zero_error,
        "alpha_one_projection_target_leakage": projection_metrics[
            "target_feature_leakage"
        ],
        "alpha_one_afr_target_leakage": afr_metrics["target_feature_leakage"],
        "alpha_one_projection_anchor_error": projection_metrics[
            "anchor_feature_error"
        ],
        "alpha_one_afr_anchor_error": afr_metrics["anchor_feature_error"],
        "projection_normalized_frozen_s_distortion": projection_metrics[
            "normalized_frozen_s_distortion"
        ],
        "afr_normalized_frozen_s_distortion": afr_metrics[
            "normalized_frozen_s_distortion"
        ],
        "preservation_improvement": (
            projection_metrics["normalized_frozen_s_distortion"]
            - afr_metrics["normalized_frozen_s_distortion"]
        ),
        "best_of_256_random_feasible_losses": min(random_losses),
        "closed_form_optimal_loss": optimal_loss,
        "compensation_trace_error": parts["compensation_trace_error"],
        **qa,
    }
