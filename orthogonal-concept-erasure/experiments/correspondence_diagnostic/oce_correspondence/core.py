from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PairSpec:
    target: str
    anchor: str
    prompt: str

    @property
    def slug(self) -> str:
        def clean(value: str) -> str:
            return "_".join(value.casefold().replace("-", " ").split())

        return f"{clean(self.target)}_to_{clean(self.anchor)}"


def validate_experiment_sets(
    pairs: Sequence[PairSpec], controls: Sequence[str]
) -> dict[str, object]:
    if not pairs:
        raise ValueError("At least one target-anchor pair is required")
    targets = [pair.target.casefold().strip() for pair in pairs]
    anchors = [pair.anchor.casefold().strip() for pair in pairs]
    controls_normalized = [value.casefold().strip() for value in controls]
    if len(targets) != len(set(targets)):
        raise ValueError(f"Duplicate targets are not allowed: {targets}")
    if len(anchors) != len(set(anchors)):
        raise ValueError(f"Duplicate anchors are not allowed: {anchors}")
    target_anchor_overlap = sorted(set(targets).intersection(anchors))
    if target_anchor_overlap:
        raise ValueError(
            f"Target and anchor sets must be disjoint: {target_anchor_overlap}"
        )
    control_overlap = sorted(
        set(controls_normalized).intersection(set(targets).union(anchors))
    )
    if control_overlap:
        raise ValueError(
            "Control concepts overlap the actual target/anchor sets: "
            f"{control_overlap}"
        )
    if len(controls_normalized) != len(set(controls_normalized)):
        raise ValueError(f"Duplicate controls are not allowed: {controls_normalized}")
    return {
        "targets": [pair.target for pair in pairs],
        "anchors": [pair.anchor for pair in pairs],
        "controls": list(controls),
        "target_anchor_overlap": [],
        "control_overlap": [],
        "valid": True,
    }


def expand_object_pairs(pairs: Sequence[PairSpec]) -> list[PairSpec]:
    """Reproduce the object prompt expansion in upstream ``oce.py``."""
    target_templates = [
        "image of {concept}",
        "photo of {concept}",
        "portrait of {concept}",
        "picture of {concept}",
        "painting of {concept}",
    ]
    expanded = list(pairs)
    for pair in pairs:
        for template in target_templates:
            expanded.append(
                PairSpec(
                    target=template.format(concept=pair.target),
                    anchor=template.format(concept=pair.anchor),
                    prompt=pair.prompt,
                )
            )
    return expanded


def collect_projection_modules(
    unet: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in unet.named_modules()
        if "attn2" in name and name.endswith("to_v")
    ]


def _orthonormal_feature_basis(
    weight: torch.Tensor, embeddings: Sequence[torch.Tensor], eps: float = 1e-8
) -> torch.Tensor:
    if not embeddings:
        raise ValueError("Cannot build a subspace from an empty embedding list")
    columns = []
    for embedding in embeddings:
        vector = weight @ embedding
        columns.append(vector / (torch.linalg.vector_norm(vector) + eps))
    basis, _ = torch.linalg.qr(torch.stack(columns, dim=1), mode="reduced")
    return basis


def _upstream_procrustes(
    objective: torch.Tensor, reflection_correction: str
) -> tuple[torch.Tensor, bool]:
    u, _, vh = torch.linalg.svd(objective, full_matrices=False)
    rotation = u @ vh
    reflected = bool(torch.linalg.det(rotation).item() < 0)
    if reflected:
        if reflection_correction == "upstream":
            # This intentionally reproduces oce.py exactly.
            rotation[:, -1] *= -1
        elif reflection_correction == "proper":
            u[:, -1] *= -1
            rotation = u @ vh
        elif reflection_correction != "none":
            raise ValueError(
                "reflection_correction must be upstream, proper, or none"
            )
    return rotation, reflected


@torch.inference_mode()
def build_vector_objective(
    weight: torch.Tensor,
    target_embeddings: Sequence[torch.Tensor],
    anchor_embeddings: Sequence[torch.Tensor],
    preserve_embeddings: Sequence[torch.Tensor],
    global_second_moment: torch.Tensor,
    erase_scale: float,
    preserve_global_scale: float,
    preserve_concept_scale: float,
    lamb: float,
) -> torch.Tensor:
    """Paper Eq. 16 with the repository's explicit weighting convention."""
    if len(target_embeddings) != len(anchor_embeddings):
        raise ValueError("Vector-wise targets and anchors must be paired one-to-one")
    objective = torch.zeros(
        weight.shape[0], weight.shape[0], device=weight.device, dtype=weight.dtype
    )
    for target, anchor in zip(target_embeddings, anchor_embeddings):
        objective += erase_scale * torch.outer(weight @ anchor, weight @ target)
    for retain in preserve_embeddings:
        projected = weight @ retain
        objective += preserve_concept_scale * torch.outer(projected, projected)
    objective += preserve_global_scale * (
        weight @ global_second_moment @ weight.T
    )
    objective += lamb * (weight @ weight.T)
    return objective


@torch.inference_mode()
def build_subspace_objective(
    weight: torch.Tensor,
    target_embeddings: Sequence[torch.Tensor],
    anchor_embeddings: Sequence[torch.Tensor],
    preserve_embeddings: Sequence[torch.Tensor],
    global_second_moment: torch.Tensor,
    erase_scale: float,
    preserve_global_scale: float,
    preserve_concept_scale: float,
    lamb: float,
) -> torch.Tensor:
    """Reproduce the current upstream ``oce.py`` subspace objective."""
    target_basis = _orthonormal_feature_basis(weight, target_embeddings)
    anchor_basis = _orthonormal_feature_basis(weight, anchor_embeddings)
    target_projector = target_basis @ target_basis.T
    anchor_projector = anchor_basis @ anchor_basis.T
    identity = torch.eye(
        weight.shape[0], device=weight.device, dtype=weight.dtype
    )
    objective = -erase_scale * target_projector @ (
        identity - anchor_projector
    )
    for retain in preserve_embeddings:
        projected = weight @ retain
        objective += preserve_concept_scale * torch.outer(projected, projected)
    objective += preserve_global_scale * (
        weight @ global_second_moment @ weight.T
    )
    objective += lamb * (weight @ weight.T)
    return objective


@torch.inference_mode()
def edit_projection_weights(
    unet: torch.nn.Module,
    embeddings: Mapping[str, torch.Tensor],
    pairs: Sequence[PairSpec],
    preserve_concepts: Sequence[str],
    global_second_moment: torch.Tensor,
    objective: str,
    erase_scale: float,
    preserve_global_scale: float,
    preserve_concept_scale: float,
    lamb: float,
    reflection_correction: str = "upstream",
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    if objective not in {"vector", "subspace"}:
        raise ValueError("objective must be vector or subspace")
    target_embeddings = [embeddings[pair.target] for pair in pairs]
    anchor_embeddings = [embeddings[pair.anchor] for pair in pairs]
    preserve_embeddings = [embeddings[value] for value in preserve_concepts]
    builder = (
        build_vector_objective
        if objective == "vector"
        else build_subspace_objective
    )
    edited: dict[str, torch.Tensor] = {}
    audit_rows: list[dict[str, object]] = []
    for layer_index, (name, module) in enumerate(
        collect_projection_modules(unet), start=1
    ):
        weight = module.weight.detach().float()
        layer_objective = builder(
            weight=weight,
            target_embeddings=target_embeddings,
            anchor_embeddings=anchor_embeddings,
            preserve_embeddings=preserve_embeddings,
            global_second_moment=global_second_moment,
            erase_scale=erase_scale,
            preserve_global_scale=preserve_global_scale,
            preserve_concept_scale=preserve_concept_scale,
            lamb=lamb,
        )
        rotation, reflected = _upstream_procrustes(
            layer_objective, reflection_correction
        )
        new_weight = rotation @ weight
        edited[f"{name}.weight"] = new_weight.cpu()
        orthogonality_error = torch.linalg.matrix_norm(
            rotation.T @ rotation
            - torch.eye(
                rotation.shape[0],
                device=rotation.device,
                dtype=rotation.dtype,
            )
        ) / max(rotation.shape[0], 1)
        gram_error = torch.linalg.matrix_norm(
            new_weight.T @ new_weight - weight.T @ weight
        ) / torch.linalg.matrix_norm(weight.T @ weight).clamp_min(1e-12)
        audit_rows.append(
            {
                "layer_index": layer_index,
                "layer": name,
                "out_dim": int(weight.shape[0]),
                "in_dim": int(weight.shape[1]),
                "objective": objective,
                "reflection_detected": reflected,
                "reflection_correction": reflection_correction,
                "orthogonality_error": float(orthogonality_error.item()),
                "weight_gram_relative_error": float(gram_error.item()),
            }
        )
    return edited, audit_rows


@torch.inference_mode()
def edit_projection_weights_with_rotations(
    unet: torch.nn.Module,
    embeddings: Mapping[str, torch.Tensor],
    pairs: Sequence[PairSpec],
    preserve_concepts: Sequence[str],
    global_second_moment: torch.Tensor,
    objective: str,
    erase_scale: float,
    preserve_global_scale: float,
    preserve_concept_scale: float,
    lamb: float,
    reflection_correction: str = "upstream",
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[dict[str, object]],
]:
    """Run the unchanged edit and additionally expose each computed rotation.

    The saved rotations are diagnostic artifacts only.  This function uses the
    same objective builders and the same upstream Procrustes implementation as
    :func:`edit_projection_weights`.
    """
    if objective not in {"vector", "subspace"}:
        raise ValueError("objective must be vector or subspace")
    target_embeddings = [embeddings[pair.target] for pair in pairs]
    anchor_embeddings = [embeddings[pair.anchor] for pair in pairs]
    preserve_embeddings = [embeddings[value] for value in preserve_concepts]
    builder = (
        build_vector_objective
        if objective == "vector"
        else build_subspace_objective
    )
    edited: dict[str, torch.Tensor] = {}
    rotations: dict[str, torch.Tensor] = {}
    audit_rows: list[dict[str, object]] = []
    for layer_index, (name, module) in enumerate(
        collect_projection_modules(unet), start=1
    ):
        weight = module.weight.detach().float()
        layer_objective = builder(
            weight=weight,
            target_embeddings=target_embeddings,
            anchor_embeddings=anchor_embeddings,
            preserve_embeddings=preserve_embeddings,
            global_second_moment=global_second_moment,
            erase_scale=erase_scale,
            preserve_global_scale=preserve_global_scale,
            preserve_concept_scale=preserve_concept_scale,
            lamb=lamb,
        )
        rotation, reflected = _upstream_procrustes(
            layer_objective, reflection_correction
        )
        new_weight = rotation @ weight
        edited[f"{name}.weight"] = new_weight.cpu()
        rotations[f"{name}.rotation"] = rotation.cpu()
        orthogonality_error = torch.linalg.matrix_norm(
            rotation.T @ rotation
            - torch.eye(
                rotation.shape[0],
                device=rotation.device,
                dtype=rotation.dtype,
            )
        ) / max(rotation.shape[0], 1)
        gram_error = torch.linalg.matrix_norm(
            new_weight.T @ new_weight - weight.T @ weight
        ) / torch.linalg.matrix_norm(weight.T @ weight).clamp_min(1e-12)
        audit_rows.append(
            {
                "layer_index": layer_index,
                "layer": name,
                "out_dim": int(weight.shape[0]),
                "in_dim": int(weight.shape[1]),
                "objective": objective,
                "reflection_detected": reflected,
                "reflection_correction": reflection_correction,
                "orthogonality_error": float(orthogonality_error.item()),
                "weight_gram_relative_error": float(gram_error.item()),
            }
        )
    return edited, rotations, audit_rows


def cosine_matrix(
    row_vectors: torch.Tensor, column_vectors: torch.Tensor
) -> torch.Tensor:
    row_vectors = torch.nn.functional.normalize(row_vectors.float(), dim=-1)
    column_vectors = torch.nn.functional.normalize(column_vectors.float(), dim=-1)
    return row_vectors @ column_vectors.T


def compute_correspondence_metrics(
    similarities: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Summarize an image/feature by anchor similarity matrix.

    ``similarities`` is shaped ``[n_targets, n_samples, n_anchors]``. The
    target and own-anchor indices must use the same ordering.
    """
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("similarities must have shape [target, sample, anchor]")
    n_targets, n_samples, n_anchors = values.shape
    if n_targets != n_anchors:
        raise ValueError("Correspondence requires equal target and anchor counts")
    rows: list[dict[str, object]] = []
    margins: list[float] = []
    positives: list[bool] = []
    top1s: list[bool] = []
    for target_index in range(n_targets):
        own = values[target_index, :, target_index]
        predictions = values[target_index].argmax(axis=1)
        own_top1 = predictions == target_index
        if n_anchors > 1:
            other_mask = np.arange(n_anchors) != target_index
            best_other = values[target_index][:, other_mask].max(axis=1)
            margin = own - best_other
            margins.extend(margin.tolist())
            positives.extend((margin > 0).tolist())
        else:
            best_other = np.full(n_samples, np.nan)
            margin = np.full(n_samples, np.nan)
        top1s.extend(own_top1.tolist())
        for sample_index in range(n_samples):
            rows.append(
                {
                    "target_index": target_index,
                    "sample_index": sample_index,
                    "own_anchor_similarity": float(own[sample_index]),
                    "best_other_anchor_similarity": (
                        None
                        if np.isnan(best_other[sample_index])
                        else float(best_other[sample_index])
                    ),
                    "correspondence_margin": (
                        None
                        if np.isnan(margin[sample_index])
                        else float(margin[sample_index])
                    ),
                    "predicted_anchor_index": int(predictions[sample_index]),
                    "own_anchor_top1": bool(own_top1[sample_index]),
                }
            )
    aggregate: dict[str, object] = {
        "own_anchor_top1_rate": float(np.mean(top1s)),
        "mean_margin": float(np.mean(margins)) if margins else None,
        "minimum_margin": float(np.min(margins)) if margins else None,
        "positive_margin_fraction": float(np.mean(positives)) if positives else None,
    }
    return rows, aggregate


def apply_weight_state(
    unet: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    missing, unexpected = unet.load_state_dict(dict(state), strict=False)
    expected_missing = [
        key
        for key in missing
        if not ("attn2" in key and key.endswith("to_v.weight"))
    ]
    if unexpected or not state:
        raise RuntimeError(
            f"Invalid edited weight state: unexpected={unexpected}, keys={len(state)}"
        )
    # Missing keys are expected because an OCE checkpoint only contains to_v.
    _ = expected_missing


def clone_projection_state(unet: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{name}.weight": copy.deepcopy(module.weight.detach()).cpu()
        for name, module in collect_projection_modules(unet)
    }
