#!/usr/bin/env python3
"""Matrix-only audit of the three baseline-qualified Confuse5 joint OCE edits.

This script never mutates a model, writes a checkpoint, or generates/evaluates
images.  It reconstructs the matrices used by the existing checkpoint builder,
checks the released variant against the already-completed joint checkpoints,
and only then writes the requested CSV and Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_OUTPUT = HERE / "solver_audit"
PAPER_URL = "https://arxiv.org/abs/2605.28902"
VARIANT_RELEASED = "A_released_oce"
VARIANT_RANK = "B_rank_corrected_released_oce"
VARIANT_FAITHFUL = "C_objective_faithful_oce"
CSV_FIELDS = [
    "group",
    "target_names",
    "anchor_names",
    "layer_name",
    "solver_variant",
    "target_expanded_column_count",
    "target_numerical_rank",
    "anchor_expanded_column_count",
    "anchor_numerical_rank",
    "paper_erase_loss",
    "preserve_loss",
    "total_paper_objective",
    "true_leakage",
    "anchor_feature_drift",
    "orthogonality_error",
    "determinant",
]

sys.path.insert(0, str(HERE))

import pipeline  # noqa: E402
import protocol  # noqa: E402
import qualified_primary  # noqa: E402
import run as checkpoint_builder  # noqa: E402


class AuditError(RuntimeError):
    """Raised when an audit invariant is not satisfied."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run only the tiny 2D/3D solver tests and do not load SD or artifacts.",
    )
    parser.add_argument(
        "--checkpoint-match-atol",
        type=float,
        default=2e-5,
        help="Absolute tolerance for Variant A versus the production checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-match-rtol",
        type=float,
        default=2e-5,
        help="Relative tolerance for Variant A versus the production checkpoint.",
    )
    return parser.parse_args(argv)


def _scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _matrix_sqnorm(value: Any) -> Any:
    return value.square().sum()


def _orthogonal_solver(matrix: Any, *, released_det_correction: bool) -> Any:
    """Return U @ Vh, optionally reproducing oce.py's post-product column flip."""
    import torch

    left, _, right_h = torch.linalg.svd(matrix, full_matrices=False)
    transform = left @ right_h
    if released_det_correction and bool(torch.linalg.det(transform).item() < 0):
        # Exact released behavior: flip a column of the already-computed P.
        transform[:, -1] *= -1
    return transform


def synthetic_unit_tests() -> dict[str, float | bool]:
    """Catch SVD, objective-orientation, and projector-orientation mistakes."""
    import torch

    dtype = torch.float64
    matrix = torch.tensor(
        [[1.2, -0.7, 0.3], [0.4, 2.1, -1.3], [-0.8, 0.5, 0.9]],
        dtype=dtype,
    )
    transform = _orthogonal_solver(matrix, released_det_correction=False)
    trace_value = torch.trace(transform.T @ matrix)
    nuclear_norm = torch.linalg.svdvals(matrix).sum()
    if not torch.allclose(trace_value, nuclear_norm, atol=1e-11, rtol=1e-11):
        raise AuditError("Synthetic Procrustes test failed: U@Vh missed the nuclear norm")

    target = torch.tensor([[1.0], [1.0], [0.2]], dtype=dtype)
    target = target / torch.linalg.vector_norm(target)
    anchor = torch.tensor([[0.2], [-0.4], [1.0]], dtype=dtype)
    anchor = anchor / torch.linalg.vector_norm(anchor)
    target_projector = target @ target.T
    anchor_projector = anchor @ anchor.T
    identity = torch.eye(3, dtype=dtype)
    anchor_complement = identity - anchor_projector
    preserve_matrix = torch.diag(torch.tensor([0.35, 0.7, 1.1], dtype=dtype))
    faithful_matrix = -anchor_complement @ target_projector + preserve_matrix
    transposed_erasure_matrix = -target_projector @ anchor_complement + preserve_matrix
    faithful = _orthogonal_solver(faithful_matrix, released_det_correction=False)
    wrong = _orthogonal_solver(transposed_erasure_matrix, released_det_correction=False)

    def eq18(candidate: Any) -> Any:
        erase = -_matrix_sqnorm(candidate @ target_projector - anchor_complement)
        preserve = _matrix_sqnorm(
            (candidate - identity) @ torch.linalg.cholesky(preserve_matrix)
        )
        return erase + preserve

    faithful_objective = eq18(faithful)
    wrong_objective = eq18(wrong)
    tolerance = 1e-11 * max(1.0, abs(_scalar(faithful_objective)))
    if _scalar(faithful_objective) > _scalar(wrong_objective) + tolerance:
        raise AuditError("Synthetic Eq.18 test failed: faithful orientation was worse")

    target_2d = torch.tensor([[1.0], [0.0]], dtype=dtype)
    anchor_2d = torch.tensor([[0.0], [1.0]], dtype=dtype)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)
    projector = target_2d @ target_2d.T
    transformed_projector = rotation @ projector @ rotation.T
    expected_projector = anchor_2d @ anchor_2d.T
    one_sided = rotation @ projector
    if torch.allclose(one_sided, expected_projector, atol=1e-12, rtol=0):
        raise AuditError("Synthetic projector test is non-discriminating")
    if not torch.allclose(
        transformed_projector, expected_projector, atol=1e-12, rtol=0
    ):
        raise AuditError("Synthetic projector test failed: PRP^T is incorrect")

    return {
        "passed": True,
        "procrustes_trace": _scalar(trace_value),
        "procrustes_nuclear_norm": _scalar(nuclear_norm),
        "faithful_eq18_objective": _scalar(faithful_objective),
        "transposed_orientation_eq18_objective": _scalar(wrong_objective),
        "projector_prpt_error": _scalar(
            torch.linalg.matrix_norm(transformed_projector - expected_projector)
        ),
    }


def _normalized_projected_columns(
    weight: Any, embeddings: Sequence[Any], eps: float
) -> Any:
    import torch

    columns = []
    for embedding in embeddings:
        vector = weight @ embedding
        columns.append(vector / (torch.linalg.vector_norm(vector) + eps))
    if not columns:
        raise AuditError("A target/anchor subspace cannot be empty")
    return torch.stack(columns, dim=1)


def _released_qr_basis(columns: Any) -> Any:
    import torch

    return torch.linalg.qr(columns, mode="reduced")[0]


def _rank_revealing_basis(columns: Any) -> tuple[Any, int, float]:
    """SVD basis with threshold sigma_i > max(m,n)*eps(dtype)*sigma_max."""
    import torch

    left, singular_values, _ = torch.linalg.svd(columns, full_matrices=False)
    relative_tolerance = max(columns.shape) * torch.finfo(columns.dtype).eps
    threshold = relative_tolerance * _scalar(singular_values[0])
    rank = int((singular_values > threshold).sum().item())
    if rank < 1:
        raise AuditError("Rank-revealing SVD produced an empty subspace")
    return left[:, :rank], rank, relative_tolerance


def _preservation_components(
    weight: Any,
    retain_embeddings: Sequence[Any],
    k0: Any,
    oce: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    import torch

    local = torch.zeros(
        weight.shape[0], weight.shape[0], device=weight.device, dtype=weight.dtype
    )
    for embedding in retain_embeddings:
        vector = weight @ embedding
        local.add_(float(oce["lambda_r"]) * torch.outer(vector, vector))
    global_prior = float(oce["lambda_0"]) * (weight @ k0 @ weight.T)
    repo_regularizer = float(oce["lamb_repo_regularizer"]) * (weight @ weight.T)
    return local, global_prior, repo_regularizer


def _assemble_matrix(
    erasure: Any, local: Any, global_prior: Any, repo_regularizer: Any
) -> Any:
    """Match the Confuse5 checkpoint builder's float32 addition grouping."""
    paper_matrix = erasure + local + global_prior
    return paper_matrix + repo_regularizer


def _paper_metrics(
    transform: Any,
    target_basis: Any,
    anchor_basis: Any,
    anchor_features: Any,
    local: Any,
    global_prior: Any,
    repo_regularizer: Any,
    erase_scale: float,
) -> dict[str, float]:
    import torch

    dimension = transform.shape[0]
    identity = torch.eye(dimension, device=transform.device, dtype=transform.dtype)
    target_projector = target_basis @ target_basis.T
    anchor_projector = anchor_basis @ anchor_basis.T
    anchor_complement = identity - anchor_projector
    delta = transform - identity

    paper_erase = -erase_scale * _matrix_sqnorm(
        transform @ target_projector - anchor_complement
    )
    preserve = (
        torch.trace(delta @ local @ delta.T)
        + torch.trace(delta @ global_prior @ delta.T)
        + torch.trace(delta @ repo_regularizer @ delta.T)
    )
    transformed_target = transform @ target_basis
    leakage_numerator = _matrix_sqnorm(anchor_complement @ transformed_target)
    leakage_projector = _matrix_sqnorm(
        anchor_complement @ transform @ target_projector @ transform.T
    )
    leakage_tolerance = 2e-5 * max(1.0, abs(_scalar(leakage_numerator)))
    if abs(_scalar(leakage_numerator - leakage_projector)) > leakage_tolerance:
        raise AuditError(
            "Leakage cross-check failed: ||A P G||_F^2 != ||A P R P^T||_F^2"
        )
    target_rank = int(target_basis.shape[1])
    anchor_denominator = _matrix_sqnorm(anchor_features)
    if not bool(anchor_denominator.item() > 0):
        raise AuditError("Anchor feature drift denominator is zero")
    anchor_drift = _matrix_sqnorm(
        transform @ anchor_features - anchor_features
    ) / anchor_denominator
    orthogonality = torch.linalg.matrix_norm(transform.T @ transform - identity)
    return {
        "paper_erase_loss": _scalar(paper_erase),
        "preserve_loss": _scalar(preserve),
        "total_paper_objective": _scalar(paper_erase + preserve),
        "true_leakage": _scalar(leakage_numerator) / target_rank,
        "anchor_feature_drift": _scalar(anchor_drift),
        "orthogonality_error": _scalar(orthogonality),
        "determinant": _scalar(torch.linalg.det(transform)),
    }


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _fmt(value: float) -> str:
    return f"{value:.8g}"


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise AuditError("Cannot aggregate an empty metric")
    return float(statistics.median(materialized))


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise AuditError("Cannot aggregate an empty metric")
    return float(statistics.fmean(materialized))


def _variant_rows(rows: Sequence[Mapping[str, Any]], variant: str) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if row["solver_variant"] == variant]
    if not selected:
        raise AuditError(f"No rows found for {variant}")
    return selected


def _paired_deltas(
    rows: Sequence[Mapping[str, Any]], left_variant: str, right_variant: str, key: str
) -> list[float]:
    lookup = {
        (row["group"], row["layer_name"], row["solver_variant"]): float(row[key])
        for row in rows
    }
    pairs = sorted({(row["group"], row["layer_name"]) for row in rows})
    return [
        lookup[(group, layer, left_variant)] - lookup[(group, layer, right_variant)]
        for group, layer in pairs
    ]


def _normalized_objective_improvements(
    rows: Sequence[Mapping[str, Any]], baseline: str, candidate: str
) -> list[float]:
    lookup = {
        (row["group"], row["layer_name"], row["solver_variant"]): float(
            row["total_paper_objective"]
        )
        for row in rows
    }
    pairs = sorted({(row["group"], row["layer_name"]) for row in rows})
    values = []
    for group, layer in pairs:
        before = lookup[(group, layer, baseline)]
        after = lookup[(group, layer, candidate)]
        values.append((before - after) / max(abs(before), abs(after), 1.0))
    return values


def _classify_outcome(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    faithful = _variant_rows(rows, VARIANT_FAITHFUL)
    faithful_leakage = [float(row["true_leakage"]) for row in faithful]
    all_leakage = [float(row["true_leakage"]) for row in rows]
    pairwise_spread = []
    for group, layer in sorted({(row["group"], row["layer_name"]) for row in rows}):
        values = [
            float(row["true_leakage"])
            for row in rows
            if row["group"] == group and row["layer_name"] == layer
        ]
        pairwise_spread.append(max(values) - min(values))

    near_zero = max(all_leakage) <= 1e-4 and max(pairwise_spread) <= 1e-4
    substantial_fraction = sum(value >= 0.01 for value in faithful_leakage) / len(
        faithful_leakage
    )
    substantial_residual = (
        _median(faithful_leakage) >= 0.01 or substantial_fraction >= 0.25
    )
    released_to_faithful = _paired_deltas(
        rows, VARIANT_RELEASED, VARIANT_FAITHFUL, "true_leakage"
    )
    objective_improvement = _normalized_objective_improvements(
        rows, VARIANT_RELEASED, VARIANT_FAITHFUL
    )
    implementation_dominates = (
        _median(released_to_faithful) >= 0.05
        and _median(objective_improvement) >= 1e-3
        and not substantial_residual
    )

    if near_zero:
        return (
            "Outcome C — No meaningful real-matrix defect",
            "Although the PR-versus-PRP^T distinction and orthogonal feasibility issue "
            "exist mathematically, they do not appear to create a meaningful defect in "
            "the current Confuse5 matrices. AFR should not be prioritized without "
            "another behavioral mechanism.",
        )
    if implementation_dominates:
        return (
            "Outcome A — Implementation issue dominates",
            "Current evidence suggests that the released rank / solver convention is a "
            "major confound. AFR should NOT be tested yet. The next step should be a "
            "small image-level comparison between released OCE and objective-faithful OCE.",
        )
    return (
        "Outcome B — Orthogonality gap remains",
        "The leakage cannot be explained away by QR rank inflation or Procrustes "
        "orientation. This provides a valid algebraic motivation for testing the "
        "proposed anchor-fixed non-orthogonal correction.",
    )


def _render_report(
    rows: Sequence[Mapping[str, Any]],
    run_info: Mapping[str, Any],
    synthetic: Mapping[str, Any],
) -> str:
    variants = [VARIANT_RELEASED, VARIANT_RANK, VARIANT_FAITHFUL]
    summary_lines = [
        "| Group | Variant | Layers | Mean paper objective | Median leakage | Max leakage | Mean anchor drift |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in qualified_primary.ELIGIBLE_SCOPE:
        for variant in variants:
            selected = [
                row
                for row in rows
                if row["group"] == group and row["solver_variant"] == variant
            ]
            summary_lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        variant,
                        str(len(selected)),
                        _fmt(_mean(float(row["total_paper_objective"]) for row in selected)),
                        _fmt(_median(float(row["true_leakage"]) for row in selected)),
                        _fmt(max(float(row["true_leakage"]) for row in selected)),
                        _fmt(_mean(float(row["anchor_feature_drift"]) for row in selected)),
                    ]
                )
                + " |"
            )

    unique_layers = {
        (row["group"], row["layer_name"]): row
        for row in rows
        if row["solver_variant"] == VARIANT_RELEASED
    }
    target_inflated = sum(
        int(row["target_expanded_column_count"])
        > int(row["target_numerical_rank"])
        for row in unique_layers.values()
    )
    anchor_inflated = sum(
        int(row["anchor_expanded_column_count"])
        > int(row["anchor_numerical_rank"])
        for row in unique_layers.values()
    )
    target_max_extra = max(
        int(row["target_expanded_column_count"])
        - int(row["target_numerical_rank"])
        for row in unique_layers.values()
    )
    anchor_max_extra = max(
        int(row["anchor_expanded_column_count"])
        - int(row["anchor_numerical_rank"])
        for row in unique_layers.values()
    )

    rank_leak_delta = _paired_deltas(
        rows, VARIANT_RELEASED, VARIANT_RANK, "true_leakage"
    )
    rank_objective_delta = _paired_deltas(
        rows, VARIANT_RELEASED, VARIANT_RANK, "total_paper_objective"
    )
    orientation_leak_delta = _paired_deltas(
        rows, VARIANT_RANK, VARIANT_FAITHFUL, "true_leakage"
    )
    orientation_objective_delta = _paired_deltas(
        rows, VARIANT_RANK, VARIANT_FAITHFUL, "total_paper_objective"
    )
    faithful = _variant_rows(rows, VARIANT_FAITHFUL)
    faithful_leaks = [float(row["true_leakage"]) for row in faithful]
    faithful_at_one_percent = sum(value >= 0.01 for value in faithful_leaks)
    outcome, conclusion = _classify_outcome(rows)

    q1_word = "yes" if target_inflated or anchor_inflated else "no"
    q2_word = "material" if abs(_median(rank_leak_delta)) >= 0.01 else "limited"
    q3_word = "material" if _median(orientation_leak_delta) >= 0.01 else "limited"
    q4_word = (
        "substantial across multiple layers"
        if faithful_at_one_percent >= math.ceil(0.25 * len(faithful_leaks))
        else "not substantial across multiple layers"
    )

    return f"""# OCE Confuse5 matrix-level solver audit

## Scope and result

This audit uses only the frozen, baseline-qualified Joint groups `dogs`, `fruits`, and `balls`, with targets, matched per-target anchors, prompt expansion, local retain concepts, K0, scales, and the 16 edited `attn2.to_v` layers resolved from the existing Confuse5 protocol. It generated no images and changed no production weights.

**Classification: {outcome}.**

> {conclusion}

## Code and paper convention audit

- Paper main text Eq. 17 constructs `G = orth(W C1)`, `G* = orth(W C*)`, `R = G G^T`, and `R* = G* G*^T`. Eq. 18 states `min -||P R - (I-R*)||F^2 + preserve(P)`. Eq. 19 writes `max tr(P M_total)`, while Eq. 20 gives `M_total = -R(I-R*) + S`; the following sentence applies `P=U V^T` to the SVD of that matrix.
- Appendix A.2 instead uses the standard convention `max tr(P^T M_e)`, derives `M_e = -(I-R*)R`, adds the symmetric preservation matrix, and uses `P=U V^T`.
- Released `oce.py` and the frozen Confuse5 checkpoint path use reduced QR, `-R(I-R*) + S`, `U V^T`, and then flip the last column of the already-computed transform when its determinant is negative.
- Direct expansion gives `-lambda_e ||P R-A||F^2 + preserve(P) = const - 2 tr(P^T[-lambda_e A R+S])`, where `A=I-R*`. Thus Appendix A.2 is objective-faithful under standard Procrustes. Main-text `tr(P M)` with `M=-R A+S` is equivalent only when solved in that convention (its maximizer is `V U^T`); pairing that matrix with `U V^T` changes the solved objective.

The rank-revealing basis applies SVD to the same L2-normalized projected columns used by released QR and retains singular values satisfying `sigma_i > max(m,n) * eps(float32) * sigma_max` (zero absolute tolerance). All three variants are evaluated against the same rank-revealed `R`, `R*`, and exact Eq. 18 loss. Preservation includes the frozen weighted local-retain, K0, and repository regularizer quadratic terms.

## Synthetic checks

All checks passed. `tr(P^T M)` at `P=U V^T` was `{_fmt(float(synthetic['procrustes_trace']))}`, equal to the nuclear norm `{_fmt(float(synthetic['procrustes_nuclear_norm']))}`. The objective-faithful synthetic Eq. 18 value was `{_fmt(float(synthetic['faithful_eq18_objective']))}` versus `{_fmt(float(synthetic['transposed_orientation_eq18_objective']))}` for the transposed-erasure orientation. The 2D projector test verified `P R P^T`, not `P R` (error `{_fmt(float(synthetic['projector_prpt_error']))}`).

## Aggregate diagnostics

{chr(10).join(summary_lines)}

Leakage is `||(I-R*) P G||F^2 / r_t`. The cross-check `||(I-R*) P R P^T||F^2 / r_t` agrees within the configured numerical tolerance. Anchor drift means only **anchor feature drift at the edited layer**.

## Answers

### Q1 — Released QR rank inflation

**{q1_word.capitalize()}.** Target QR included extra dependent directions in {target_inflated}/{len(unique_layers)} group-layer matrices (maximum inflation {target_max_extra}); anchor QR did so in {anchor_inflated}/{len(unique_layers)} (maximum inflation {anchor_max_extra}).

### Q2 — Effect of rank correction alone

The effect was **{q2_word}** by the pre-registered one-percentage-point leakage criterion. Across matched layers, median `A leakage - B leakage` was `{_fmt(_median(rank_leak_delta))}` and median `A objective - B objective` was `{_fmt(_median(rank_objective_delta))}`; positive values favor Variant B.

### Q3 — Effect of objective-faithful orientation

The leakage change was **{q3_word}** by the same criterion. Holding the SVD bases fixed, median `B leakage - C leakage` was `{_fmt(_median(orientation_leak_delta))}`, and median `B objective - C objective` was `{_fmt(_median(orientation_objective_delta))}`; positive values favor objective-faithful Variant C. Variant C was also checked layer-by-layer not to have a worse Eq. 18 objective than Variant B beyond floating-point tolerance.

### Q4 — Residual leakage after both corrections

Residual objective-faithful leakage was **{q4_word}**: median `{_fmt(_median(faithful_leaks))}`, maximum `{_fmt(max(faithful_leaks))}`, with {faithful_at_one_percent}/{len(faithful_leaks)} group-layer matrices at or above 0.01.

## Reproducibility

- CSV: `results.csv` ({len(rows)} rows = 3 groups x 16 layers x 3 variants)
- Config SHA-256: `{run_info['config_sha256']}`
- Anchors SHA-256: `{run_info['anchors_sha256']}`
- Qualification SHA-256: `{run_info['qualification_sha256']}`
- K0 SHA-256: `{run_info['k0_sha256']}`
- Variant A checkpoint agreement: all layers passed `atol={run_info['checkpoint_match_atol']}` and `rtol={run_info['checkpoint_match_rtol']}`; maximum absolute edited-weight error `{_fmt(float(run_info['checkpoint_max_abs_error']))}`
- Numerical-rank relative tolerance range: `{_fmt(float(run_info['rank_rtol_min']))}` to `{_fmt(float(run_info['rank_rtol_max']))}`
- Paper: {PAPER_URL}
- Runtime estimate for one cached SD 1.4 GPU run: about 5–15 minutes; no image generation or image evaluator is involved.
"""


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.17g}"
    return value


def _validated_joint_specs(
    config: Mapping[str, Any], anchors: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    output_root = Path(config["_resolved"]["output_root"])
    qualification, qualification_path = qualified_primary.load_and_validate_qualification(
        output_root, config
    )
    expected_groups = list(qualified_primary.ELIGIBLE_SCOPE)
    if qualification.get("eligible_groups") != expected_groups:
        raise AuditError("Qualification registry no longer matches the frozen audit scope")
    specs = [
        spec
        for spec in protocol.checkpoint_specs(config, anchors)
        if spec["mode"] == "joint" and spec["group_id"] in expected_groups
    ]
    if [spec["group_id"] for spec in specs] != expected_groups:
        raise AuditError("Joint audit specs do not match dogs/fruits/balls in frozen order")
    for spec in specs:
        expected_targets = qualified_primary.ELIGIBLE_SCOPE[spec["group_id"]]
        if list(spec["targets"]) != expected_targets:
            raise AuditError(f"Frozen targets changed for {spec['group_id']}")
        expected_anchors = [anchors[target] for target in expected_targets]
        if list(spec["anchors"]) != expected_anchors:
            raise AuditError(f"Matched anchors changed for {spec['group_id']}")
        pipeline._validate_checkpoint(spec, config)
    balls = next(spec for spec in specs if spec["group_id"] == "balls")
    if list(balls["anchors"]) != ["basketball", "baseball"]:
        raise AuditError("Balls must retain matched anchors basketball/baseball")
    return specs, qualification, qualification_path


def execute(args: argparse.Namespace) -> tuple[Path, Path]:
    if platform.system() == "Darwin":
        raise AuditError(
            "Full solver audit is forbidden on the local Mac; run it in the active "
            "GPU-server project environment."
        )

    import torch
    from diffusers import DiffusionPipeline
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise AuditError("Full solver audit requires the configured GPU server")

    synthetic = synthetic_unit_tests()
    config_path = args.config.resolve()
    config, anchors = protocol.load_protocol(config_path)
    specs, _, qualification_path = _validated_joint_specs(config, anchors)
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
    expected_layers = int(config["evaluation"]["expected_checkpoint_keys"])
    if len(modules) != expected_layers:
        raise AuditError(f"Expected {expected_layers} edited layers, found {len(modules)}")

    prompt_order: list[str] = []
    for spec in specs:
        prompt_order.extend(protocol.expanded_prompts(spec["targets"], config))
        prompt_order.extend(protocol.expanded_prompts(spec["anchors"], config))
        prompt_order.extend(spec["retain_concepts"])
    prompt_order = list(dict.fromkeys(prompt_order))
    embeddings = {
        prompt_text: checkpoint_builder._encode_prompt(
            pipe, prompt_text, device, torch.float32
        )
        for prompt_text in prompt_order
    }
    k0 = k0_matrix.to(device=device, dtype=torch.float32)
    oce = config["oce"]
    eps = float(oce["normalization_eps"])
    erase_scale = float(oce["lambda_e"])
    raw_rows: list[dict[str, Any]] = []
    checkpoint_max_abs_error = 0.0
    rank_rtols: list[float] = []

    with torch.inference_mode():
        for spec in specs:
            target_prompts = protocol.expanded_prompts(spec["targets"], config)
            anchor_prompts = protocol.expanded_prompts(spec["anchors"], config)
            target_embeddings = [embeddings[value] for value in target_prompts]
            anchor_embeddings = [embeddings[value] for value in anchor_prompts]
            retain_embeddings = [embeddings[value] for value in spec["retain_concepts"]]
            checkpoint_state = load_file(spec["checkpoint_path"], device="cpu")
            expected_keys = {f"{name}.weight" for name, _ in modules}
            if set(checkpoint_state) != expected_keys:
                raise AuditError(f"Checkpoint tensor keys differ for {spec['group_id']}")

            for layer_index, (name, module) in enumerate(modules, start=1):
                weight = module.weight.detach().to(device=device, dtype=torch.float32)
                target_columns = _normalized_projected_columns(
                    weight, target_embeddings, eps
                )
                anchor_columns = _normalized_projected_columns(
                    weight, anchor_embeddings, eps
                )
                target_qr = _released_qr_basis(target_columns)
                anchor_qr = _released_qr_basis(anchor_columns)
                target_svd, target_rank, target_rtol = _rank_revealing_basis(
                    target_columns
                )
                anchor_svd, anchor_rank, anchor_rtol = _rank_revealing_basis(
                    anchor_columns
                )
                rank_rtols.extend([target_rtol, anchor_rtol])

                identity = torch.eye(
                    weight.shape[0], device=weight.device, dtype=weight.dtype
                )
                target_qr_projector = target_qr @ target_qr.T
                anchor_qr_projector = anchor_qr @ anchor_qr.T
                target_projector = target_svd @ target_svd.T
                anchor_projector = anchor_svd @ anchor_svd.T
                anchor_complement = identity - anchor_projector
                local, global_prior, repo_regularizer = _preservation_components(
                    weight, retain_embeddings, k0, oce
                )

                released_erasure = -erase_scale * target_qr_projector @ (
                    identity - anchor_qr_projector
                )
                rank_erasure = -erase_scale * target_projector @ anchor_complement
                faithful_erasure = -erase_scale * anchor_complement @ target_projector
                released_matrix = _assemble_matrix(
                    released_erasure, local, global_prior, repo_regularizer
                )
                rank_matrix = _assemble_matrix(
                    rank_erasure, local, global_prior, repo_regularizer
                )
                faithful_matrix = _assemble_matrix(
                    faithful_erasure, local, global_prior, repo_regularizer
                )
                transforms = {
                    VARIANT_RELEASED: _orthogonal_solver(
                        released_matrix, released_det_correction=True
                    ),
                    VARIANT_RANK: _orthogonal_solver(
                        rank_matrix, released_det_correction=True
                    ),
                    VARIANT_FAITHFUL: _orthogonal_solver(
                        faithful_matrix, released_det_correction=False
                    ),
                }

                checkpoint_key = f"{name}.weight"
                expected_weight = checkpoint_state[checkpoint_key].to(
                    device=device, dtype=torch.float32
                )
                recomputed_weight = transforms[VARIANT_RELEASED] @ weight
                max_abs_error = _scalar((recomputed_weight - expected_weight).abs().max())
                checkpoint_max_abs_error = max(checkpoint_max_abs_error, max_abs_error)
                if not torch.allclose(
                    recomputed_weight,
                    expected_weight,
                    atol=args.checkpoint_match_atol,
                    rtol=args.checkpoint_match_rtol,
                ):
                    raise AuditError(
                        "Variant A failed production checkpoint agreement at "
                        f"{spec['group_id']} layer {layer_index} {name}: "
                        f"max_abs_error={max_abs_error:.8g}. No CSV/report was written."
                    )

                anchor_features = torch.stack(
                    [weight @ embedding for embedding in anchor_embeddings], dim=1
                )
                layer_metrics: dict[str, dict[str, float]] = {}
                for variant, transform in transforms.items():
                    metrics = _paper_metrics(
                        transform,
                        target_svd,
                        anchor_svd,
                        anchor_features,
                        local,
                        global_prior,
                        repo_regularizer,
                        erase_scale,
                    )
                    layer_metrics[variant] = metrics
                    raw_rows.append(
                        {
                            "group": spec["group_id"],
                            "target_names": json.dumps(
                                spec["targets"], ensure_ascii=False
                            ),
                            "anchor_names": json.dumps(
                                spec["anchors"], ensure_ascii=False
                            ),
                            "layer_name": name,
                            "solver_variant": variant,
                            "target_expanded_column_count": len(target_prompts),
                            "target_numerical_rank": target_rank,
                            "anchor_expanded_column_count": len(anchor_prompts),
                            "anchor_numerical_rank": anchor_rank,
                            **metrics,
                        }
                    )

                faithful_value = layer_metrics[VARIANT_FAITHFUL][
                    "total_paper_objective"
                ]
                rank_value = layer_metrics[VARIANT_RANK]["total_paper_objective"]
                objective_tolerance = 2e-5 * max(
                    1.0, abs(faithful_value), abs(rank_value)
                )
                if faithful_value > rank_value + objective_tolerance:
                    raise AuditError(
                        "Objective-faithful solver was worse than the rank-matched "
                        f"released orientation at {spec['group_id']} {name}: "
                        f"C={faithful_value:.8g}, B={rank_value:.8g}"
                    )
                print(
                    f"[solver audit] {spec['group_id']} layer "
                    f"{layer_index}/{len(modules)}",
                    flush=True,
                )

    expected_rows = len(specs) * len(modules) * 3
    if len(raw_rows) != expected_rows:
        raise AuditError(f"Expected {expected_rows} result rows, got {len(raw_rows)}")
    run_info = {
        "config_sha256": protocol.sha256(config_path),
        "anchors_sha256": protocol.sha256(Path(config["_resolved"]["anchors_path"])),
        "qualification_sha256": protocol.sha256(qualification_path),
        "k0_sha256": k0_metadata["artifact_sha256"],
        "checkpoint_match_atol": args.checkpoint_match_atol,
        "checkpoint_match_rtol": args.checkpoint_match_rtol,
        "checkpoint_max_abs_error": checkpoint_max_abs_error,
        "rank_rtol_min": min(rank_rtols),
        "rank_rtol_max": max(rank_rtols),
    }
    csv_rows = [{key: _csv_value(row[key]) for key in CSV_FIELDS} for row in raw_rows]
    report = _render_report(raw_rows, run_info, synthetic)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    report_path = output_dir / "REPORT.md"
    _atomic_write_csv(csv_path, csv_rows)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(report_path)
    return csv_path, report_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint_match_atol < 0 or args.checkpoint_match_rtol < 0:
        raise AuditError("Checkpoint tolerances must be non-negative")
    if args.synthetic_only:
        print(json.dumps(synthetic_unit_tests(), indent=2, sort_keys=True))
        return 0
    csv_path, report_path = execute(args)
    print(f"[complete] CSV: {csv_path}")
    print(f"[complete] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
