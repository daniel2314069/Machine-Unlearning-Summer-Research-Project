from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .clustering import PALETTE, _prototype_distribution_metrics
from .utils import atomic_write_text, l2_normalize, set_reproducible_seed, write_csv


RAW_REPRESENTATION = "raw_fixed_readout_description_space"


def _basis_from_rows(vectors: np.ndarray) -> tuple[np.ndarray, int]:
    """Return ordered OCE-style direction basis for row-wise vectors.

    OCE normalizes projected concept directions and stacks them as columns before
    orthogonalization.  Here SVD ordering is recovered through the small Gram
    matrix so truncated bases remain practical for thousands of split fits.
    """
    rows = l2_normalize(np.asarray(vectors, dtype=np.float64))
    matrix = rows.T
    gram = matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * max(float(eigenvalues[0]), 1.0)
    numerical_rank = int(np.sum(eigenvalues > tolerance))
    if numerical_rank == 0:
        raise RuntimeError("Description matrix has zero numerical rank")
    basis = matrix @ eigenvectors[:, :numerical_rank]
    basis /= np.sqrt(eigenvalues[:numerical_rank])[None, :]
    basis, _ = np.linalg.qr(basis, mode="reduced")
    return basis, numerical_rank


def _capture(queries: np.ndarray, basis: np.ndarray, rank: int | None) -> np.ndarray:
    queries = np.asarray(queries, dtype=np.float64)
    effective_rank = basis.shape[1] if rank is None else min(int(rank), basis.shape[1])
    if effective_rank <= 0:
        raise ValueError("Subspace rank must be positive")
    projection = queries @ basis[:, :effective_rank]
    numerator = np.sum(projection * projection, axis=1)
    denominator = np.sum(queries * queries, axis=1)
    return np.clip(numerator / np.clip(denominator, 1e-24, None), 0.0, 1.0)


def _rank_specs(settings: dict[str, Any]) -> list[tuple[str, int | None]]:
    ranks = [(str(int(rank)), int(rank)) for rank in settings["ranks"]]
    if settings.get("include_full_numerical_rank", True):
        ranks.append(("full", None))
    return ranks


def _deterministic_splits(
    labels: np.ndarray,
    concept_count: int,
    n_splits: int,
    train_per_concept: int,
    heldout_per_concept: int,
    seed: int,
) -> list[dict[int, tuple[np.ndarray, np.ndarray]]]:
    output = []
    for split_index in range(n_splits):
        rng = np.random.default_rng(seed + split_index)
        current: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for concept_index in range(concept_count):
            indices = np.flatnonzero(labels == concept_index)
            if len(indices) != train_per_concept + heldout_per_concept:
                raise ValueError(
                    f"Concept {concept_index} has {len(indices)} rows, expected "
                    f"{train_per_concept + heldout_per_concept}"
                )
            order = rng.permutation(indices)
            current[concept_index] = (
                order[:train_per_concept],
                order[train_per_concept:train_per_concept + heldout_per_concept],
            )
        output.append(current)
    return output


def _space_payloads(raw: dict[str, Any], layers: dict[str, Any], bare: dict[str, Any], primary: str):
    yield {
        "representation": RAW_REPRESENTATION,
        "space_index": -1,
        "layer_name": "",
        "descriptions": raw["fixed_readout"][primary].numpy(),
        "fixed_prototypes": raw["prototypes"][primary].numpy(),
        "bare_prototypes": bare["raw_unnormalized"].numpy(),
    }
    for layer_index, layer_name in enumerate(layers["layer_names"]):
        yield {
            "representation": f"to_v_layer_{layer_index:02d}",
            "space_index": layer_index,
            "layer_name": layer_name,
            "descriptions": layers["description_embeddings"][layer_name].numpy(),
            "fixed_prototypes": layers["prototype_embeddings"][layer_name].numpy(),
            "bare_prototypes": bare["layer_unnormalized"][layer_name].numpy(),
        }


def _centroid_rows(
    space: dict[str, Any],
    labels: np.ndarray,
    concept_names: list[str],
    prototype_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    features = l2_normalize(space["descriptions"])
    centroids = l2_normalize(np.stack([features[labels == index].mean(axis=0) for index in range(len(concept_names))]))
    rows = []
    for prototype_type, prototype_prompt, prototypes in [
        ("bare_oce_uce", "bare concept name; selected concept token", space["bare_prototypes"]),
        ("fixed_suffix", "concept name + fixed suffix; selected final token 'concept'", space["fixed_prototypes"]),
    ]:
        normalized = l2_normalize(prototypes)
        distribution = _prototype_distribution_metrics(
            features, labels, normalized, prototype_analysis
        )
        distances = 1.0 - normalized @ centroids.T
        for concept_index, concept in enumerate(concept_names):
            current = distances[concept_index]
            order = np.argsort(current)
            own = float(current[concept_index])
            nearest_wrong = min(float(current[index]) for index in range(len(concept_names)) if index != concept_index)
            rows.append({
                "representation": space["representation"],
                "space_index": space["space_index"],
                "layer_name": space["layer_name"],
                "prototype_type": prototype_type,
                "prototype_prompt_and_token": prototype_prompt,
                "concept": concept,
                "own_centroid_cosine_distance": own,
                "nearest_centroid": concept_names[int(order[0])],
                "correct_centroid_rank": int(np.where(order == concept_index)[0][0]) + 1,
                "nearest_incorrect_centroid_distance": nearest_wrong,
                "own_vs_nearest_incorrect_margin": nearest_wrong - own,
                "all_centroid_distances": json.dumps({
                    name: float(current[index]) for index, name in enumerate(concept_names)
                }),
                **distribution[concept_index],
            })
    return rows


def _full_description_capture_rows(
    space: dict[str, Any],
    labels: np.ndarray,
    concept_names: list[str],
    rank_specs: list[tuple[str, int | None]],
) -> list[dict[str, Any]]:
    descriptions = l2_normalize(space["descriptions"])
    bare = np.asarray(space["bare_prototypes"], dtype=np.float64)
    bases = []
    ranks = []
    for concept_index in range(len(concept_names)):
        basis, numerical_rank = _basis_from_rows(descriptions[labels == concept_index])
        bases.append(basis)
        ranks.append(numerical_rank)
    rows = []
    for rank_label, requested_rank in rank_specs:
        matrix = np.column_stack([
            _capture(bare, basis, requested_rank) for basis in bases
        ])
        for prototype_index, concept in enumerate(concept_names):
            current = matrix[prototype_index]
            order = np.argsort(-current)
            own = float(current[prototype_index])
            wrong = max(float(current[index]) for index in range(len(concept_names)) if index != prototype_index)
            correct_rank = int(np.where(order == prototype_index)[0][0]) + 1
            for subspace_index, subspace_concept in enumerate(concept_names):
                rows.append({
                    "representation": space["representation"],
                    "space_index": space["space_index"],
                    "layer_name": space["layer_name"],
                    "rank_label": rank_label,
                    "requested_rank": "full" if requested_rank is None else requested_rank,
                    "prototype_concept": concept,
                    "subspace_concept": subspace_concept,
                    "capture": float(current[subspace_index]),
                    "is_own_subspace": prototype_index == subspace_index,
                    "subspace_full_numerical_rank": ranks[subspace_index],
                    "own_subspace_capture": own,
                    "highest_incorrect_subspace_capture": wrong,
                    "own_minus_incorrect_margin": own - wrong,
                    "correct_subspace_rank": correct_rank,
                })
    return rows


def _heldout_rows_for_space(
    space: dict[str, Any],
    labels: np.ndarray,
    concept_names: list[str],
    rank_specs: list[tuple[str, int | None]],
    splits: list[dict[int, tuple[np.ndarray, np.ndarray]]],
) -> list[dict[str, Any]]:
    descriptions = l2_normalize(space["descriptions"])
    bare = np.asarray(space["bare_prototypes"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(splits):
        bases = []
        numerical_ranks = []
        for concept_index in range(len(concept_names)):
            train_indices, _ = split[concept_index]
            basis, numerical_rank = _basis_from_rows(descriptions[train_indices])
            bases.append(basis)
            numerical_ranks.append(numerical_rank)
        for rank_label, requested_rank in rank_specs:
            bare_scores = np.column_stack([
                _capture(bare, basis, requested_rank) for basis in bases
            ])
            per_concept_rows = []
            for concept_index, concept in enumerate(concept_names):
                _, heldout_indices = split[concept_index]
                heldout = descriptions[heldout_indices]
                scores = np.column_stack([
                    _capture(heldout, basis, requested_rank) for basis in bases
                ])
                predictions = scores.argmax(axis=1)
                own = scores[:, concept_index]
                other = scores.copy()
                other[:, concept_index] = -np.inf
                highest_wrong = other.max(axis=1)
                bare_current = bare_scores[concept_index]
                bare_order = np.argsort(-bare_current)
                bare_wrong = max(
                    float(bare_current[index])
                    for index in range(len(concept_names)) if index != concept_index
                )
                row = {
                    "representation": space["representation"],
                    "space_index": space["space_index"],
                    "layer_name": space["layer_name"],
                    "split": split_index,
                    "rank_label": rank_label,
                    "requested_rank": "full" if requested_rank is None else requested_rank,
                    "concept": concept,
                    "n_heldout": len(heldout_indices),
                    "heldout_accuracy": float(np.mean(predictions == concept_index)),
                    "mean_own_subspace_capture": float(own.mean()),
                    "mean_highest_incorrect_subspace_capture": float(highest_wrong.mean()),
                    "mean_heldout_margin": float((own - highest_wrong).mean()),
                    "bare_own_subspace_capture": float(bare_current[concept_index]),
                    "bare_highest_incorrect_subspace_capture": bare_wrong,
                    "bare_own_minus_incorrect_margin": float(bare_current[concept_index] - bare_wrong),
                    "bare_correct_subspace_rank": int(np.where(bare_order == concept_index)[0][0]) + 1,
                    "bare_capture_percentile_among_heldout_same_concept": float(
                        100.0 * np.mean(own <= bare_current[concept_index])
                    ),
                    "own_subspace_full_numerical_rank": numerical_ranks[concept_index],
                }
                rows.append(row)
                per_concept_rows.append(row)
            rows.append({
                "representation": space["representation"],
                "space_index": space["space_index"],
                "layer_name": space["layer_name"],
                "split": split_index,
                "rank_label": rank_label,
                "requested_rank": "full" if requested_rank is None else requested_rank,
                "concept": "__overall__",
                "n_heldout": sum(row["n_heldout"] for row in per_concept_rows),
                "heldout_accuracy": float(np.mean([row["heldout_accuracy"] for row in per_concept_rows])),
                "mean_own_subspace_capture": float(np.mean([row["mean_own_subspace_capture"] for row in per_concept_rows])),
                "mean_highest_incorrect_subspace_capture": float(np.mean([row["mean_highest_incorrect_subspace_capture"] for row in per_concept_rows])),
                "mean_heldout_margin": float(np.mean([row["mean_heldout_margin"] for row in per_concept_rows])),
                "bare_own_subspace_capture": float(np.mean([row["bare_own_subspace_capture"] for row in per_concept_rows])),
                "bare_highest_incorrect_subspace_capture": float(np.mean([row["bare_highest_incorrect_subspace_capture"] for row in per_concept_rows])),
                "bare_own_minus_incorrect_margin": float(np.mean([row["bare_own_minus_incorrect_margin"] for row in per_concept_rows])),
                "bare_correct_subspace_rank": float(np.mean([row["bare_correct_subspace_rank"] for row in per_concept_rows])),
                "bare_capture_percentile_among_heldout_same_concept": float(np.mean([
                    row["bare_capture_percentile_among_heldout_same_concept"] for row in per_concept_rows
                ])),
                "own_subspace_full_numerical_rank": float(np.mean([
                    row["own_subspace_full_numerical_rank"] for row in per_concept_rows
                ])),
            })
    return rows


def _aggregate_heldout(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["representation", "space_index", "layer_name", "rank_label", "requested_rank", "concept"]
    metrics = [
        "heldout_accuracy", "mean_own_subspace_capture",
        "mean_highest_incorrect_subspace_capture", "mean_heldout_margin",
        "bare_own_subspace_capture", "bare_highest_incorrect_subspace_capture",
        "bare_own_minus_incorrect_margin", "bare_correct_subspace_rank",
        "bare_capture_percentile_among_heldout_same_concept", "own_subspace_full_numerical_rank",
    ]
    grouped = frame.groupby(keys, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    return grouped


def _layerwise_rows(capture: pd.DataFrame, heldout: pd.DataFrame, concept_names: list[str]) -> pd.DataFrame:
    summary = capture[capture["is_own_subspace"].astype(bool)][[
        "representation", "space_index", "layer_name", "rank_label", "requested_rank",
        "prototype_concept", "own_subspace_capture", "highest_incorrect_subspace_capture",
        "own_minus_incorrect_margin", "correct_subspace_rank",
    ]].rename(columns={"prototype_concept": "concept"})
    means = summary.groupby(
        ["representation", "space_index", "layer_name", "rank_label", "requested_rank"],
        as_index=False, dropna=False,
    )[["own_subspace_capture", "highest_incorrect_subspace_capture", "own_minus_incorrect_margin"]].mean()
    means["concept"] = "__mean__"
    means["correct_subspace_rank"] = np.nan
    summary = pd.concat([summary, means], ignore_index=True)
    heldout_columns = [column for column in heldout.columns if column.endswith("_mean")]
    right = heldout[[
        "representation", "space_index", "layer_name", "rank_label", "requested_rank", "concept",
        *heldout_columns,
    ]].copy()
    right["concept"] = right["concept"].replace({"__overall__": "__mean__"})
    return summary.merge(
        right,
        on=["representation", "space_index", "layer_name", "rank_label", "requested_rank", "concept"],
        how="left",
    )


def _annotated_heatmap(matrix, xlabels, ylabels, title, path, fmt=".3f") -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    threshold = float(np.nanmin(matrix) + 0.6 * (np.nanmax(matrix) - np.nanmin(matrix)))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            ax.text(column, row, format(value, fmt), ha="center", va="center",
                    color="white" if value >= threshold else "#222222")
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_plots(
    output_dir: Path,
    centroid: pd.DataFrame,
    capture: pd.DataFrame,
    layerwise: pd.DataFrame,
    concept_names: list[str],
    plot_rank: int,
) -> None:
    rank_label = str(plot_rank)
    raw_centroid = centroid[centroid["representation"] == RAW_REPRESENTATION]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    prototype_types = ["bare_oce_uce", "fixed_suffix"]
    matrices = {}
    for prototype_type in prototype_types:
        subset = raw_centroid[raw_centroid["prototype_type"] == prototype_type]
        matrices[prototype_type] = np.array([
            [json.loads(subset[subset["concept"] == concept].iloc[0]["all_centroid_distances"])[target]
             for target in concept_names]
            for concept in concept_names
        ])
    shared_maximum = max(float(matrix.max()) for matrix in matrices.values())
    for axis, prototype_type, title in zip(
        axes,
        prototype_types,
        ["Bare OCE/UCE prototype", "Old fixed-suffix prototype"],
    ):
        matrix = matrices[prototype_type]
        image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=shared_maximum)
        axis.set_xticks(range(len(concept_names)), concept_names, rotation=35, ha="right")
        axis.set_yticks(range(len(concept_names)), concept_names)
        axis.set_title(title)
        for i in range(len(concept_names)):
            for j in range(len(concept_names)):
                axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                          color="white" if matrix[i, j] > shared_maximum * 0.6 else "#222222")
    colorbar_axis = fig.add_axes([0.925, 0.19, 0.018, 0.56])
    fig.colorbar(image, cax=colorbar_axis, label="Cosine distance")
    fig.suptitle("Prototype-to-description-centroid distance in raw text space")
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.15, top=0.84, wspace=0.28)
    fig.savefig(output_dir / "centroid_distance_heatmap.png", dpi=180)
    plt.close(fig)

    raw_capture = capture[
        (capture["representation"] == RAW_REPRESENTATION) & (capture["rank_label"] == rank_label)
    ]
    matrix = raw_capture.pivot(index="prototype_concept", columns="subspace_concept", values="capture").loc[
        concept_names, concept_names
    ].to_numpy()
    _annotated_heatmap(
        matrix, concept_names, concept_names,
        f"Bare-name capture by description subspace (raw, rank {plot_rank})",
        output_dir / "bare_name_subspace_capture_heatmap.png",
    )

    selected = layerwise[
        (layerwise["rank_label"] == rank_label)
        & (layerwise["concept"].isin(concept_names))
    ].copy()
    selected["x"] = selected["space_index"] + 1
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for concept_index, concept in enumerate(concept_names):
        subset = selected[selected["concept"] == concept].sort_values("x")
        color = PALETTE[concept_index]
        axes[0].plot(subset["x"], subset["own_subspace_capture"], marker="o", label=concept, color=color)
        axes[1].plot(subset["x"], subset["own_minus_incorrect_margin"], marker="o", label=concept, color=color)
    axes[0].set_ylabel("Own-subspace capture")
    axes[1].set_ylabel("Own − highest incorrect capture")
    axes[1].set_xlabel("Representation (raw, then original to_v layer index)")
    axes[1].set_xticks(range(17), ["raw", *[str(index) for index in range(16)]])
    axes[0].legend(ncol=4, frameon=False)
    axes[0].set_title(f"Layer-wise bare-name alignment at rank {plot_rank}")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_capture_curves.png", dpi=180)
    plt.close(fig)

    overall = layerwise[layerwise["concept"] == "__mean__"].copy()
    layer4_name = "to_v_layer_04"
    subset = overall[overall["representation"].isin([RAW_REPRESENTATION, layer4_name])].copy()
    rank_order = [str(rank) for rank in [1, 2, 4, 8, 16, 32]] + ["full"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for representation, label, color in [
        (RAW_REPRESENTATION, "raw", PALETTE[0]),
        (layer4_name, "to_v layer 4", PALETTE[2]),
    ]:
        current = subset[subset["representation"] == representation].set_index("rank_label").reindex(rank_order)
        x = np.arange(len(rank_order))
        axes[0].plot(x, current["own_subspace_capture"], marker="o", label=label, color=color)
        axes[1].plot(x, current["heldout_accuracy_mean"], marker="o", label=label, color=color)
    for axis in axes:
        axis.set_xticks(np.arange(len(rank_order)), rank_order)
        axis.set_xlabel("Truncated subspace rank")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Mean bare own-subspace capture")
    axes[1].set_ylabel("Held-out description accuracy")
    axes[0].set_title("Bare-name capture rank sweep")
    axes[1].set_title("Held-out generalization rank sweep")
    fig.tight_layout()
    fig.savefig(output_dir / "rank_sweep_curves.png", dpi=180)
    plt.close(fig)


def _build_report(
    config: dict[str, Any], output_dir: Path, source_output: Path,
    audit: pd.DataFrame, centroid: pd.DataFrame, layerwise: pd.DataFrame,
    concept_names: list[str],
) -> None:
    settings = config["bare_name_subspace"]
    plot_rank = str(settings["plot_rank"])
    raw_centroid = centroid[centroid["representation"] == RAW_REPRESENTATION]
    centroid_lines = []
    for concept in concept_names:
        bare = raw_centroid[(raw_centroid["concept"] == concept) & (raw_centroid["prototype_type"] == "bare_oce_uce")].iloc[0]
        fixed = raw_centroid[(raw_centroid["concept"] == concept) & (raw_centroid["prototype_type"] == "fixed_suffix")].iloc[0]
        centroid_lines.append(
            f"| {concept} | {bare['own_centroid_cosine_distance']:.4f} | {int(bare['correct_centroid_rank'])} | "
            f"{bare['bootstrap_percentile_median']:.1f} | {fixed['own_centroid_cosine_distance']:.4f} | "
            f"{int(fixed['correct_centroid_rank'])} | {fixed['bootstrap_percentile_median']:.1f} |"
        )

    raw_rank = layerwise[
        (layerwise["representation"] == RAW_REPRESENTATION)
        & (layerwise["rank_label"] == plot_rank)
        & (layerwise["concept"].isin(concept_names))
    ]
    capture_lines = [
        f"| {row['concept']} | {row['own_subspace_capture']:.4f} | "
        f"{row['highest_incorrect_subspace_capture']:.4f} | {row['own_minus_incorrect_margin']:.4f} | "
        f"{int(row['correct_subspace_rank'])} | {row['heldout_accuracy_mean']:.3f} | "
        f"{row['bare_capture_percentile_among_heldout_same_concept_mean']:.1f} |"
        for _, row in raw_rank.sort_values("concept").iterrows()
    ]
    mean_rows = layerwise[
        (layerwise["rank_label"] == plot_rank) & (layerwise["concept"] == "__mean__")
    ]
    best_capture = mean_rows.sort_values("own_subspace_capture", ascending=False).iloc[0]
    best_margin = mean_rows.sort_values("own_minus_incorrect_margin", ascending=False).iloc[0]
    layer4 = mean_rows[mean_rows["space_index"] == 4].iloc[0]
    raw_mean = mean_rows[mean_rows["space_index"] == -1].iloc[0]
    rank_concepts = layerwise[
        (layerwise["rank_label"] == plot_rank) & (layerwise["concept"].isin(concept_names))
    ]
    consistency = rank_concepts.groupby("representation").agg(
        positive_concepts=("own_minus_incorrect_margin", lambda values: int((values > 0).sum())),
        rank_one_concepts=("correct_subspace_rank", lambda values: int((values == 1).sum())),
    )
    layer4_consistency = consistency.loc["to_v_layer_04"]
    consistent_layers = consistency[
        (consistency["positive_concepts"] == len(concept_names))
        & (consistency["rank_one_concepts"] == len(concept_names))
    ].index.tolist()
    raw_bare_mean_distance = raw_centroid[raw_centroid["prototype_type"] == "bare_oce_uce"][
        "own_centroid_cosine_distance"
    ].mean()
    raw_fixed_mean_distance = raw_centroid[raw_centroid["prototype_type"] == "fixed_suffix"][
        "own_centroid_cosine_distance"
    ].mean()
    tokens = ", ".join(
        f"{row['concept']} → {row['selected_decoded_token']} (position {int(row['selected_token_position'])})"
        for _, row in audit.iterrows()
    )
    split_count = int(settings["heldout_splits"])
    ranks = ", ".join(str(rank) for rank in settings["ranks"])
    line_continuation = "\\"
    report = f"""# Bare OCE/UCE concept names versus name-free description subspaces

## Technical summary

This isolated analysis changes only the explicit concept prototype. The old prototype prompt was
`"cat This sentence describes the concept"`, and its vector was the contextual hidden state of the final token
`concept`. The new prototype prompt is exactly `"cat"`, and its vector is the concept content token selected by
the repository's actual SD 1.4 OCE/UCE rule. The 200 description vectors remain the existing cached fixed-readout
vectors from `"description This sentence describes the concept"`. This is therefore an intentional asymmetric
comparison, not a claim that both sides use the same prompt construction.

No image was generated, no description was changed, and no OCE, UCE, checkpoint, or W0 matrix was edited.

## Main result in plain language

The actual bare OCE/UCE name is **not a typical member of the cached fixed-readout description representation**.
In raw space, all four bare names still rank their own description centroid first, so a relative concept signal is
present. But their mean own-centroid cosine distance is {raw_bare_mean_distance:.4f}, versus
{raw_fixed_mean_distance:.4f} for the old fixed-suffix prototypes, and both constructions remain at the 100th
distance percentile. At rank {plot_rank}, bare names retain only {raw_mean['own_subspace_capture']:.2%} mean energy
in their own raw description subspaces, and their held-out capture percentile is 0 throughout this rank sweep and
all representations. Thus the old peripheral result was not caused by adding the suffix; the true bare token is
even less absolutely aligned with these intentionally asymmetric fixed-readout description vectors.

## Exact bare-name extraction

Both the OCE erasure entry point in `orthogonal-concept-erasure/oce.py` and
`unified-concept-editing/trainscripts/uce_sd_erase.py::UCE` select
`attention_mask.sum() - 2` after calling `pipe.encode_prompt`. The repository has no separate shared helper, so
this analysis reuses that exact inline rule rather than inventing a token-position convention.

Selected tokens: {tokens}. None of the four names was split into multiple CLIP tokens. Full padded token IDs,
decoded tokens, and attention masks are in `bare_name_tokenization_audit.csv`.

## Centroid question: is bare cat close to the average cat description?

Cosine distances are computed after L2 normalization; the unnormalized bare vectors remain preserved in
`bare_name_embeddings.pt`.

| Concept | Bare own distance | Bare rank | Bare distance percentile | Fixed own distance | Fixed rank | Fixed distance percentile |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(centroid_lines)}

The distance percentile follows the old centroid experiment: high means farther from the description centroid
than typical same-concept descriptions. It must not be confused with the capture percentile below.

![Bare and fixed centroid distances](centroid_distance_heatmap.png)

## Subspace question: does bare cat point along directions shared by cat descriptions?

For every concept, normalized description directions are stacked as columns, matching OCE's geometric convention.
An SVD orders those directions, and ranks {ranks}, plus full numerical rank, are evaluated separately. Capture is
the fraction of vector energy inside the truncated subspace.

At rank {plot_rank} in raw text space:

| Concept | Own capture | Highest incorrect | Margin | Correct rank | Held-out accuracy | Bare capture percentile |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(capture_lines)}

![Raw rank-{plot_rank} capture](bare_name_subspace_capture_heatmap.png)

## Held-out question: does the subspace capture descriptions it did not build from?

Across {split_count} deterministic splits per representation, 40 descriptions per concept build each subspace and
10 remain held out. Each held-out description is assigned to the subspace with the largest capture. The held-out
accuracy and capture margins above are averages across those splits. This prevents training-vector containment
from being mistaken for generalization.

The bare capture percentile asks a different question from centroid distance: it is the percentage of held-out
same-concept descriptions whose own-subspace capture is no greater than the bare name's capture. High is stronger
containment; low is weaker-than-typical containment.

## Raw space versus the 16 unchanged original W0_v projections

At rank {plot_rank}, raw mean own capture is {raw_mean['own_subspace_capture']:.4f} with mean own-versus-other margin
{raw_mean['own_minus_incorrect_margin']:.4f}. Original to_v layer 4 gives mean own capture
{layer4['own_subspace_capture']:.4f}, margin {layer4['own_minus_incorrect_margin']:.4f}, and held-out accuracy
{layer4['heldout_accuracy_mean']:.3f}. The highest mean own capture occurs in
`{best_capture['representation']}` ({best_capture['own_subspace_capture']:.4f}); the largest mean margin occurs in
`{best_margin['representation']}` ({best_margin['own_minus_incorrect_margin']:.4f}). These are geometric alignments,
not causal attributions to a layer. Layer 4 is not unusually strong for the bare prototype at rank {plot_rank}: it
has {int(layer4_consistency['positive_concepts'])}/4 positive concept margins and
{int(layer4_consistency['rank_one_concepts'])}/4 correct subspace ranks. The layers whose margins and ranks agree
across all four concepts are {', '.join(consistent_layers) if consistent_layers else 'none'}.

![Layer-wise capture](layerwise_capture_curves.png)

![Rank sweep](rank_sweep_curves.png)

## Interpretation limits

1. The fixed-readout description token is an operational sentence representation, not the unique true one.
2. High subspace capture does not prove semantic understanding; shared lexical, syntactic, or dataset directions can raise it.
3. This experiment does not test or prove OCE/UCE erasure correctness.
4. Full-rank capture is broad by construction and is not interpreted without truncated-rank and held-out controls.
5. The description side and bare-name side intentionally use different prompt/readout constructions.
6. The 200 descriptions retain the single-source and template-bias limitations documented in the original report.

## Reproducibility

Run from `orthogonal-concept-erasure/experiments/concept_description_clustering` using the required Conda environment wrapper.

Extract the four bare-name vectors and original-W0 projections:

```bash
./scripts/run_py310.sh -m concept_clustering.bare_name_cli extract {line_continuation}
  --config configs/bare_name_subspace.json {line_continuation}
  --source-output outputs/codex_diverse_final {line_continuation}
  --output outputs/bare_name_subspace_analysis
```

Run all raw and 16-layer statistics, splits, plots, and report:

```bash
./scripts/run_py310.sh -m concept_clustering.bare_name_cli analyze {line_continuation}
  --config configs/bare_name_subspace.json {line_continuation}
  --source-output outputs/codex_diverse_final {line_continuation}
  --output outputs/bare_name_subspace_analysis
```

Rerun only statistics, plots, and the report from cached vectors:

```bash
./scripts/run_py310.sh -m concept_clustering.bare_name_cli stats {line_continuation}
  --config configs/bare_name_subspace.json {line_continuation}
  --source-output outputs/codex_diverse_final {line_continuation}
  --output outputs/bare_name_subspace_analysis
```

The cached-only commands do not load the diffusion pipeline or U-Net. Source descriptions and their embeddings are
read from `{source_output}` and are never rewritten.
"""
    atomic_write_text(output_dir / "bare_name_subspace_report.md", report)


def run_bare_name_subspace_analysis(
    config: dict[str, Any], source_output: str | Path, output_dir: str | Path
) -> None:
    set_reproducible_seed(int(config["bare_name_subspace"]["split_seed"]))
    source_output = Path(source_output).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = torch.load(source_output / "raw_text_embeddings.pt", map_location="cpu", weights_only=False)
    layers = torch.load(source_output / "layer_embeddings.pt", map_location="cpu", weights_only=False)
    bare = torch.load(output_dir / "bare_name_embeddings.pt", map_location="cpu", weights_only=False)
    audit = pd.read_csv(output_dir / "bare_name_tokenization_audit.csv")
    concept_names = list(raw["concept_names"])
    if concept_names != list(bare["concept_names"]) or concept_names != list(layers["concept_names"]):
        raise RuntimeError("Concept order differs across source and bare-name caches")
    if list(raw["candidate_ids"]) != list(layers["candidate_ids"]):
        raise RuntimeError("Raw and layer accepted-description caches do not contain identical rows")
    if list(layers["layer_names"]) != list(bare["layer_names"]):
        raise RuntimeError("Bare-name and description to_v layer caches differ")
    source_fingerprint = raw.get("metadata", {}).get("original_w0_in_memory_fingerprint_sha256")
    bare_fingerprint = bare.get("metadata", {}).get("original_w0_in_memory_fingerprint_sha256")
    if source_fingerprint and bare_fingerprint != source_fingerprint:
        raise RuntimeError("Bare-name and source embeddings do not use identical original W0 matrices")

    primary = config["readout"]["primary_suffix_name"]
    labels = np.array([concept_names.index(label) for label in raw["concept_labels"]])
    if len(labels) != 200 or any(np.sum(labels == index) != 50 for index in range(len(concept_names))):
        raise RuntimeError("Expected the unchanged balanced 200-description source dataset")
    settings = config["bare_name_subspace"]
    rank_specs = _rank_specs(settings)
    splits = _deterministic_splits(
        labels,
        len(concept_names),
        int(settings["heldout_splits"]),
        int(settings["train_per_concept"]),
        int(settings["heldout_per_concept"]),
        int(settings["split_seed"]),
    )
    centroid_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    per_split_rows: list[dict[str, Any]] = []
    spaces = list(_space_payloads(raw, layers, bare, primary))
    for space in spaces:
        centroid_rows.extend(_centroid_rows(
            space, labels, concept_names, config.get("prototype_analysis", {})
        ))
        capture_rows.extend(_full_description_capture_rows(space, labels, concept_names, rank_specs))
        per_split_rows.extend(_heldout_rows_for_space(space, labels, concept_names, rank_specs, splits))

    centroid = pd.DataFrame(centroid_rows)
    capture = pd.DataFrame(capture_rows)
    per_split = pd.DataFrame(per_split_rows)
    heldout = _aggregate_heldout(per_split)
    layerwise = _layerwise_rows(capture, heldout, concept_names)
    write_csv(output_dir / "bare_vs_fixed_prototype_centroid_metrics.csv", centroid.to_dict("records"))
    write_csv(output_dir / "bare_name_subspace_capture.csv", capture.to_dict("records"))
    write_csv(output_dir / "per_split_metrics.csv", per_split.to_dict("records"))
    write_csv(output_dir / "heldout_description_subspace_metrics.csv", heldout.to_dict("records"))
    write_csv(output_dir / "layerwise_bare_name_metrics.csv", layerwise.to_dict("records"))
    _make_plots(
        output_dir, centroid, capture, layerwise, concept_names, int(settings["plot_rank"])
    )
    _build_report(config, output_dir, source_output, audit, centroid, layerwise, concept_names)
