from __future__ import annotations

import argparse
import inspect
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)

from scripts.eot_spherical_clustering import fit_spherical_kmeans, normalize_rows

from .balanced_paired import FIXED_SUFFIX
from .balanced_paired_w0_geometry import (
    CONDITIONS,
    CONDITION_LABELS,
    CONCEPTS,
    REPRESENTATIONS,
    _audit_dataset,
    _inspect_oce_source,
    _resolve_dataset,
    _runtime_config,
    _tensor_sha256,
)
from .modeling import load_original_pipeline, model_metadata, original_projection_modules
from .utils import atomic_write_text, package_versions, read_jsonl


METHODS = ["spherical_normalized", "euclidean_raw", "euclidean_normalized"]
METHOD_LABELS = {
    "spherical_normalized": "Spherical normalized",
    "euclidean_raw": "Euclidean raw",
    "euclidean_normalized": "Euclidean normalized",
}
NAME_REPRESENTATIONS = ["last", "eot", "fixed"]
BLUE = "#2458A6"
ORANGE = "#D65F30"
GOLD = "#D28E00"
INK = "#222222"
GRID = "#D9D9D9"
MARGIN_CMAP = LinearSegmentedColormap.from_list(
    "orange_white_blue", [ORANGE, "#F7F7F7", BLUE]
)
REQUIRED_OUTPUTS = [
    "dataset_audit.json",
    "experiment_config.json",
    "name_tokenization_audit.csv",
    "layer_inventory.csv",
    "w0_immutability.json",
    "vector_norm_summary.csv",
    "clustering_metrics_all_methods.csv",
    "clustering_per_class_recall.csv",
    "prototype_distance_by_animal.csv",
    "prototype_layer_summary.csv",
    "readout_margin_comparison.csv",
    "analysis_checks.json",
    "chart_map.json",
    "report.md",
]


def _space_specs(
    modules: list[tuple[str, torch.nn.Module]],
) -> list[tuple[str, str, int, str, torch.Tensor | None]]:
    specs: list[tuple[str, str, int, str, torch.Tensor | None]] = [
        ("text", "Text", -1, "text_space", None)
    ]
    specs.extend(
        (f"layer_{index:02d}", f"L{index}", index, f"unet.{name}", module.weight)
        for index, (name, module) in enumerate(modules)
    )
    return specs


def _project_raw(raw: np.ndarray, weight: torch.Tensor | None, device: str) -> np.ndarray:
    matrix = np.asarray(raw, dtype=np.float32)
    if weight is None:
        return matrix.copy()
    with torch.inference_mode():
        source = torch.from_numpy(matrix).to(device=device, dtype=weight.dtype)
        projected = (source @ weight.detach().T).float().cpu().numpy()
    return projected.astype(np.float32, copy=False)


def _load_cached_embeddings(
    cache_dir: Path,
    source_dir: Path,
    dataset_audit: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    cache_dir = cache_dir.expanduser().resolve()
    required = {
        "description_eot": cache_dir / "description_embeddings_eot.npy",
        "description_fixed": cache_dir / "description_embeddings_fixed.npy",
        "name_last": cache_dir / "name_embeddings_last.npy",
        "name_eot": cache_dir / "name_embeddings_eot.npy",
        "name_fixed": cache_dir / "name_embeddings_fixed.npy",
        "token_audit": cache_dir / "name_tokenization_audit.csv",
        "dataset_audit": cache_dir / "dataset_audit.json",
        "layer_inventory": cache_dir / "layer_inventory.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Embedding cache is incomplete: {missing}")
    cached_audit = json.loads(required["dataset_audit"].read_text())
    if cached_audit.get("dataset_sha256") != dataset_audit.get("dataset_sha256"):
        raise RuntimeError("Cached embeddings were extracted from a different accepted dataset")

    descriptions = {
        "eot": np.load(required["description_eot"]).astype(np.float32, copy=False),
        "fixed_suffix": np.load(required["description_fixed"]).astype(np.float32, copy=False),
    }
    names = {
        "last": np.load(required["name_last"]).astype(np.float32, copy=False),
        "eot": np.load(required["name_eot"]).astype(np.float32, copy=False),
        "fixed": np.load(required["name_fixed"]).astype(np.float32, copy=False),
    }
    if any(matrix.shape != (400, 768) for matrix in descriptions.values()):
        raise RuntimeError(f"Unexpected cached description shapes: {[x.shape for x in descriptions.values()]}")
    if any(matrix.shape != (8, 768) for matrix in names.values()):
        raise RuntimeError(f"Unexpected cached name shapes: {[x.shape for x in names.values()]}")

    balanced_eot = np.load(source_dir / "eot_embeddings.npy")
    balanced_fixed = np.load(source_dir / "fixed_suffix_embeddings.npy")
    eot_match = bool(np.allclose(descriptions["eot"], balanced_eot, atol=2e-5, rtol=2e-5))
    fixed_match = bool(
        np.allclose(normalize_rows(descriptions["fixed_suffix"]), balanced_fixed, atol=2e-5, rtol=2e-5)
    )
    if not eot_match or not fixed_match:
        raise RuntimeError(
            f"Cached W0-analysis embeddings disagree with balanced-paired caches: "
            f"eot={eot_match}, fixed={fixed_match}"
        )
    shutil.copy2(required["token_audit"], output_dir / "name_tokenization_audit.csv")
    return descriptions, names, {
        "embedding_cache_directory": str(cache_dir),
        "cached_dataset_hash_matches": True,
        "cached_eot_matches_balanced_cache": eot_match,
        "cached_fixed_matches_balanced_cache_after_normalization": fixed_match,
        "cached_layer_inventory_path": str(required["layer_inventory"]),
        "text_space_description_dimensions": {
            key: list(value.shape) for key, value in descriptions.items()
        },
        "text_space_name_dimensions": {key: list(value.shape) for key, value in names.items()},
    }


def _build_vector_spaces_without_labels(
    description_text_raw: dict[str, np.ndarray],
    name_text_raw: dict[str, np.ndarray],
    modules: list[tuple[str, torch.nn.Module]],
    device: str,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
    """Projection boundary: no labels or concept identities are accepted."""
    description_raw: dict[tuple[str, str], np.ndarray] = {}
    description_normalized: dict[tuple[str, str], np.ndarray] = {}
    name_raw: dict[tuple[str, str], np.ndarray] = {}
    name_normalized: dict[tuple[str, str], np.ndarray] = {}
    for layer_id, _, _, _, weight in _space_specs(modules):
        for representation in REPRESENTATIONS:
            key = (layer_id, representation)
            projected = _project_raw(description_text_raw[representation], weight, device)
            description_raw[key] = projected
            description_normalized[key] = normalize_rows(projected).astype(np.float32)
        for representation in NAME_REPRESENTATIONS:
            key = (layer_id, representation)
            projected = _project_raw(name_text_raw[representation], weight, device)
            name_raw[key] = projected
            name_normalized[key] = normalize_rows(projected).astype(np.float32)
    return description_raw, description_normalized, name_raw, name_normalized


def _save_vector_archives(
    output_dir: Path,
    description_raw: dict[tuple[str, str], np.ndarray],
    description_normalized: dict[tuple[str, str], np.ndarray],
    name_raw: dict[tuple[str, str], np.ndarray],
    name_normalized: dict[tuple[str, str], np.ndarray],
) -> None:
    def flattened(values: dict[tuple[str, str], np.ndarray]) -> dict[str, np.ndarray]:
        return {f"{space}__{representation}": matrix for (space, representation), matrix in values.items()}

    np.savez_compressed(output_dir / "description_vectors_raw.npz", **flattened(description_raw))
    np.savez_compressed(
        output_dir / "description_vectors_normalized.npz", **flattened(description_normalized)
    )
    np.savez_compressed(output_dir / "name_vectors_raw.npz", **flattened(name_raw))
    np.savez_compressed(output_dir / "name_vectors_normalized.npz", **flattened(name_normalized))


def _fit_all_clustering_without_labels(
    description_raw: dict[tuple[str, str], np.ndarray],
    description_normalized: dict[tuple[str, str], np.ndarray],
    settings: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Complete unsupervised fit boundary; true labels cannot enter."""
    fits: dict[tuple[str, str, str], dict[str, Any]] = {}
    spherical_kwargs = {
        "k": int(settings["k"]),
        "n_init": int(settings["n_init"]),
        "max_iter": int(settings["max_iter"]),
        "tolerance": float(settings["tolerance"]),
        "random_seed": int(settings["random_seed"]),
    }
    euclidean_kwargs = {
        "n_clusters": int(settings["k"]),
        "init": "k-means++",
        "n_init": int(settings["n_init"]),
        "max_iter": int(settings["max_iter"]),
        "tol": float(settings["tolerance"]),
        "random_state": int(settings["random_seed"]),
        "algorithm": "lloyd",
    }
    for space_representation, raw_features in description_raw.items():
        normalized_features = description_normalized[space_representation]
        spherical = fit_spherical_kmeans(normalized_features, **spherical_kwargs)
        fits[(*space_representation, "spherical_normalized")] = {
            "labels": spherical.labels,
            "centers": spherical.centers,
            "objective": float(spherical.objective),
            "inertia": np.nan,
            "n_iter": int(spherical.n_iter),
            "converged": bool(spherical.converged),
            "best_initialization": int(spherical.best_initialization),
            "input_normalized": True,
            "centers_normalized_during_fit": True,
        }
        for method, features in [
            ("euclidean_raw", raw_features),
            ("euclidean_normalized", normalized_features),
        ]:
            estimator = KMeans(**euclidean_kwargs).fit(features)
            fits[(*space_representation, method)] = {
                "labels": estimator.labels_.astype(np.int64),
                "centers": estimator.cluster_centers_.astype(np.float64),
                "objective": np.nan,
                "inertia": float(estimator.inertia_),
                "n_iter": int(estimator.n_iter_),
                "converged": int(estimator.n_iter_) < int(settings["max_iter"]),
                "best_initialization": np.nan,
                "input_normalized": method == "euclidean_normalized",
                "centers_normalized_during_fit": False,
            }
    return fits


def _hungarian_evaluation(
    features: np.ndarray,
    fit: dict[str, Any],
    true_ids: np.ndarray,
    silhouette_metric: str,
) -> tuple[dict[str, Any], np.ndarray]:
    labels = np.asarray(fit["labels"], dtype=np.int64)
    raw_confusion = confusion_matrix(true_ids, labels, labels=np.arange(len(CONCEPTS)))
    true_rows, cluster_columns = linear_sum_assignment(-raw_confusion)
    mapping = {
        int(cluster): int(true) for true, cluster in zip(true_rows.tolist(), cluster_columns.tolist())
    }
    predicted = np.asarray([mapping[int(cluster)] for cluster in labels], dtype=np.int64)
    matched_confusion = confusion_matrix(true_ids, predicted, labels=np.arange(len(CONCEPTS)))
    counts = np.bincount(true_ids, minlength=len(CONCEPTS))
    recalls = np.diag(matched_confusion) / counts
    input_norms = np.linalg.norm(features, axis=1)
    center_norms = np.linalg.norm(np.asarray(fit["centers"]), axis=1)
    metrics = {
        "ari": float(adjusted_rand_score(true_ids, labels)),
        "nmi": float(normalized_mutual_info_score(true_ids, labels)),
        "matched_accuracy": float(accuracy_score(true_ids, predicted)),
        "silhouette": float(silhouette_score(features, labels, metric=silhouette_metric)),
        "cluster_sizes": np.bincount(labels, minlength=len(CONCEPTS)).astype(int).tolist(),
        "per_class_recall": {
            concept: float(recalls[index]) for index, concept in enumerate(CONCEPTS)
        },
        "input_norm_min": float(input_norms.min()),
        "input_norm_max": float(input_norms.max()),
        "center_norm_min": float(center_norms.min()),
        "center_norm_max": float(center_norms.max()),
    }
    return metrics, matched_confusion


def _evaluate_all_clustering(
    rows: list[dict[str, Any]],
    modules: list[tuple[str, torch.nn.Module]],
    description_raw: dict[tuple[str, str], np.ndarray],
    description_normalized: dict[tuple[str, str], np.ndarray],
    fits: dict[tuple[str, str, str], dict[str, Any]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # True labels are constructed only after all unsupervised fits have completed.
    true_ids = np.asarray([CONCEPTS.index(row["concept"]) for row in rows], dtype=np.int64)
    metrics_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for layer_id, _, layer_index, layer_name, weight in _space_specs(modules):
        for representation in REPRESENTATIONS:
            for method in METHODS:
                fit = fits[(layer_id, representation, method)]
                normalized = method != "euclidean_raw"
                features = (
                    description_normalized[(layer_id, representation)]
                    if normalized
                    else description_raw[(layer_id, representation)]
                )
                metric_name = "cosine" if method == "spherical_normalized" else "euclidean"
                metrics, matched_confusion = _hungarian_evaluation(
                    features, fit, true_ids, metric_name
                )
                row = {
                    "representation": representation,
                    "space_type": "text" if weight is None else "W0",
                    "layer_id": layer_id,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "clustering_method": method,
                    "normalization": "row_l2" if normalized else "none",
                    "ari": metrics["ari"],
                    "nmi": metrics["nmi"],
                    "matched_accuracy": metrics["matched_accuracy"],
                    "silhouette": metrics["silhouette"],
                    "silhouette_metric": metric_name,
                    "cluster_sizes": json.dumps(metrics["cluster_sizes"]),
                    "objective": fit["objective"],
                    "inertia": fit["inertia"],
                    "iterations": fit["n_iter"],
                    "converged": fit["converged"],
                    "best_initialization": fit["best_initialization"],
                    "input_norm_min": metrics["input_norm_min"],
                    "input_norm_max": metrics["input_norm_max"],
                    "center_norm_min": metrics["center_norm_min"],
                    "center_norm_max": metrics["center_norm_max"],
                    "centers_normalized_during_fit": fit["centers_normalized_during_fit"],
                    "labels_available_to_fit": False,
                }
                metrics_rows.append(row)
                recall_rows.extend(
                    {
                        "representation": representation,
                        "space_type": row["space_type"],
                        "layer_id": layer_id,
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "clustering_method": method,
                        "concept": concept,
                        "recall": metrics["per_class_recall"][concept],
                    }
                    for concept in CONCEPTS
                )
                frame = pd.DataFrame(matched_confusion, index=CONCEPTS, columns=CONCEPTS)
                frame.rename_axis("true_concept").to_csv(
                    confusion_dir / f"confusion_{layer_id}_{representation}_{method}.csv"
                )
    metrics_frame = pd.DataFrame(metrics_rows)
    recall_frame = pd.DataFrame(recall_rows)
    metrics_frame.to_csv(output_dir / "clustering_metrics_all_methods.csv", index=False)
    recall_frame.to_csv(output_dir / "clustering_per_class_recall.csv", index=False)
    return metrics_frame, recall_frame


def _vector_norm_summary(
    rows: list[dict[str, Any]],
    modules: list[tuple[str, torch.nn.Module]],
    description_raw: dict[tuple[str, str], np.ndarray],
    name_raw: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    true_ids = np.asarray([CONCEPTS.index(row["concept"]) for row in rows], dtype=np.int64)
    output: list[dict[str, Any]] = []
    for layer_id, _, layer_index, layer_name, weight in _space_specs(modules):
        common = {
            "space_type": "text" if weight is None else "W0",
            "layer_id": layer_id,
            "layer_index": layer_index,
            "layer_name": layer_name,
            "normalization_stage": "raw_before_row_l2",
        }
        for representation in REPRESENTATIONS:
            matrix = description_raw[(layer_id, representation)]
            for concept_index, concept in enumerate(CONCEPTS):
                values = np.linalg.norm(matrix[true_ids == concept_index], axis=1)
                output.append({
                    **common,
                    "representation": representation,
                    "concept": concept,
                    "vector_type": "description",
                    "count": int(len(values)),
                    "vector_dimension": int(matrix.shape[1]),
                    "mean_norm": float(values.mean()),
                    "std_norm": float(values.std(ddof=0)),
                    "min_norm": float(values.min()),
                    "max_norm": float(values.max()),
                })
        for representation in NAME_REPRESENTATIONS:
            matrix = name_raw[(layer_id, representation)]
            for concept_index, concept in enumerate(CONCEPTS):
                value = float(np.linalg.norm(matrix[concept_index]))
                output.append({
                    **common,
                    "representation": f"name_{representation}",
                    "concept": concept,
                    "vector_type": "name",
                    "count": 1,
                    "vector_dimension": int(matrix.shape[1]),
                    "mean_norm": value,
                    "std_norm": 0.0,
                    "min_norm": value,
                    "max_norm": value,
                })
    return pd.DataFrame(output)


def _prototype_analysis(
    rows: list[dict[str, Any]],
    modules: list[tuple[str, torch.nn.Module]],
    description_normalized: dict[tuple[str, str], np.ndarray],
    name_normalized: dict[tuple[str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    true_ids = np.asarray([CONCEPTS.index(row["concept"]) for row in rows], dtype=np.int64)
    output: list[dict[str, Any]] = []
    centroid_norms: list[float] = []
    for layer_id, display, layer_index, layer_name, weight in _space_specs(modules):
        centroids: dict[str, np.ndarray] = {}
        for representation in REPRESENTATIONS:
            features = description_normalized[(layer_id, representation)]
            means = np.asarray(
                [features[true_ids == index].mean(axis=0) for index in range(len(CONCEPTS))]
            )
            centroids[representation] = normalize_rows(means).astype(np.float32)
            centroid_norms.extend(np.linalg.norm(centroids[representation], axis=1).tolist())
        condition_specs = {
            "matched_eot": (
                name_normalized[(layer_id, "eot")], centroids["eot"]
            ),
            "matched_fixed": (
                name_normalized[(layer_id, "fixed")], centroids["fixed_suffix"]
            ),
            "oce_last_to_eot": (
                name_normalized[(layer_id, "last")], centroids["eot"]
            ),
        }
        for condition, (prototypes, condition_centroids) in condition_specs.items():
            distances = 1.0 - prototypes @ condition_centroids.T
            for concept_index, concept in enumerate(CONCEPTS):
                row_distances = distances[concept_index]
                other_indices = [index for index in range(len(CONCEPTS)) if index != concept_index]
                nearest_other_index = min(other_indices, key=lambda index: row_distances[index])
                order = np.argsort(row_distances, kind="stable")
                own_distance = float(row_distances[concept_index])
                other_distance = float(row_distances[nearest_other_index])
                own_rank = int(np.flatnonzero(order == concept_index)[0]) + 1
                output.append({
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "cross_readout": condition == "oce_last_to_eot",
                    "space_type": "text" if weight is None else "W0",
                    "space_label": display,
                    "layer_id": layer_id,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "animal": concept,
                    "own_centroid_cosine_distance": own_distance,
                    "nearest_other_concept": CONCEPTS[nearest_other_index],
                    "nearest_other_cosine_distance": other_distance,
                    "margin": other_distance - own_distance,
                    "own_centroid_rank": own_rank,
                    "rank1_own_centroid": own_rank == 1,
                })
    distances = pd.DataFrame(output)
    summaries = []
    for keys, group in distances.groupby(
        ["condition", "condition_label", "cross_readout", "space_type", "space_label", "layer_id", "layer_index", "layer_name"],
        sort=False,
    ):
        minimum_index = group["margin"].idxmin()
        minimum = group.loc[minimum_index]
        summaries.append({
            **dict(zip(
                ["condition", "condition_label", "cross_readout", "space_type", "space_label", "layer_id", "layer_index", "layer_name"],
                keys,
            )),
            "rank1_count_out_of_8": int(group["rank1_own_centroid"].sum()),
            "mean_margin": float(group["margin"].mean()),
            "median_margin": float(group["margin"].median()),
            "minimum_margin": float(minimum["margin"]),
            "animal_with_minimum_margin": str(minimum["animal"]),
            "mean_own_centroid_distance": float(group["own_centroid_cosine_distance"].mean()),
        })
    summary = pd.DataFrame(summaries)
    comparison_rows = []
    index_columns = ["space_type", "space_label", "layer_id", "layer_index", "layer_name", "animal"]
    for keys, group in distances.groupby(index_columns, sort=False):
        by_condition = group.set_index("condition")
        comparison_rows.append({
            **dict(zip(index_columns, keys)),
            "matched_eot_margin": float(by_condition.loc["matched_eot", "margin"]),
            "matched_fixed_margin": float(by_condition.loc["matched_fixed", "margin"]),
            "oce_last_to_eot_margin": float(by_condition.loc["oce_last_to_eot", "margin"]),
            "oce_minus_matched_eot_margin": float(
                by_condition.loc["oce_last_to_eot", "margin"]
                - by_condition.loc["matched_eot", "margin"]
            ),
            "matched_eot_rank": int(by_condition.loc["matched_eot", "own_centroid_rank"]),
            "matched_fixed_rank": int(by_condition.loc["matched_fixed", "own_centroid_rank"]),
            "oce_last_to_eot_rank": int(by_condition.loc["oce_last_to_eot", "own_centroid_rank"]),
        })
    comparison = pd.DataFrame(comparison_rows)
    checks = {
        "prototype_input_norm_min": float(
            min(np.linalg.norm(matrix, axis=1).min() for matrix in name_normalized.values())
        ),
        "prototype_input_norm_max": float(
            max(np.linalg.norm(matrix, axis=1).max() for matrix in name_normalized.values())
        ),
        "centroid_norm_min": float(min(centroid_norms)),
        "centroid_norm_max": float(max(centroid_norms)),
    }
    return distances, summary, comparison, checks


def _annotated_heatmap(
    values: np.ndarray,
    space_labels: list[str],
    title: str,
    subtitle: str,
    path: Path,
    *,
    margin: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(14.2, 6.0))
    if margin:
        maximum = max(float(np.abs(values).max()), 1e-6)
        image = axis.imshow(
            values,
            aspect="auto",
            cmap=MARGIN_CMAP,
            norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        )
    else:
        image = axis.imshow(values, aspect="auto", cmap="Blues_r", vmin=1, vmax=8)
    axis.set_xticks(np.arange(len(space_labels)), space_labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(CONCEPTS)), CONCEPTS)
    axis.set_title(f"{title}\n{subtitle}", loc="left", fontsize=13, color=INK, pad=12)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            if margin:
                text = f"{value:+.3f}"
                color = "white" if abs(value) > 0.58 * maximum else INK
            else:
                text = str(int(round(value)))
                color = "white" if value <= 2 else INK
            axis.text(column, row, text, ha="center", va="center", fontsize=8, color=color)
    colorbar = figure.colorbar(image, ax=axis, pad=0.012)
    colorbar.set_label("Own-vs-nearest-other margin" if margin else "Own-centroid rank")
    axis.spines[:].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_clustering_metric(
    metrics: pd.DataFrame,
    representation: str,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11.8, 6.2))
    colors = {
        "spherical_normalized": BLUE,
        "euclidean_raw": ORANGE,
        "euclidean_normalized": GOLD,
    }
    styles = {
        "spherical_normalized": "-",
        "euclidean_raw": "--",
        "euclidean_normalized": "-.",
    }
    subset = metrics[metrics["representation"] == representation]
    ordered_indices = sorted(subset["layer_index"].unique())
    space_labels = ["Text" if index == -1 else f"L{index}" for index in ordered_indices]
    for method in METHODS:
        rows = subset[subset["clustering_method"] == method].set_index("layer_index").loc[ordered_indices]
        axis.plot(
            np.arange(len(ordered_indices)),
            rows[metric].to_numpy(),
            label=METHOD_LABELS[method],
            color=colors[method],
            linestyle=styles[method],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
        )
    label = "Unsuffixed EOT" if representation == "eot" else "Fixed suffix"
    axis.set_title(
        f"{label}: {ylabel} across clustering controls\n"
        f"Text space and {len(ordered_indices) - 1} unchanged original SD 1.4 attn2.to_v projections",
        loc="left",
        fontsize=13,
        color=INK,
        pad=12,
    )
    axis.set_xticks(np.arange(len(space_labels)), space_labels, rotation=45, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Representation space")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_prototype_summary(summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(14.0, 12.0), sharex=True)
    ordered_indices = sorted(summary["layer_index"].unique())
    space_labels = ["Text" if index == -1 else f"L{index}" for index in ordered_indices]
    for row_index, condition in enumerate(CONDITIONS):
        rows = summary[summary["condition"] == condition].set_index("layer_index").loc[ordered_indices]
        x = np.arange(len(ordered_indices))
        axes[row_index, 0].plot(x, rows["rank1_count_out_of_8"], color=BLUE, marker="o", linewidth=2)
        axes[row_index, 0].set_ylim(-0.3, 8.3)
        axes[row_index, 0].set_yticks(range(0, 9, 2))
        axes[row_index, 0].set_ylabel("Rank-1 names / 8")
        axes[row_index, 0].set_title(CONDITION_LABELS[condition], loc="left", fontsize=11, color=INK)
        axes[row_index, 1].plot(x, rows["mean_margin"], color=ORANGE, marker="o", linewidth=2)
        axes[row_index, 1].axhline(0.0, color=INK, linewidth=0.9)
        axes[row_index, 1].set_ylabel("Mean cosine margin")
        axes[row_index, 1].set_title("Mean across eight animals", loc="left", fontsize=11, color=INK)
        for axis in axes[row_index]:
            axis.grid(axis="y", color=GRID, linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1]:
        axis.set_xticks(np.arange(len(space_labels)), space_labels, rotation=45, ha="right")
        axis.set_xlabel("Representation space")
    figure.suptitle(
        "Prototype-to-description summary across every space\n"
        "Rank-1 count must be read alongside mean margin; individual animals remain in the heatmaps",
        x=0.06,
        ha="left",
        fontsize=14,
        color=INK,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _create_plots(
    metrics: pd.DataFrame,
    distances: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ordered_indices = sorted(distances["layer_index"].unique())
    space_labels = ["Text" if index == -1 else f"L{index}" for index in ordered_indices]
    for representation, suffix in [("eot", "eot"), ("fixed_suffix", "fixed")]:
        _plot_clustering_metric(
            metrics,
            representation,
            "ari",
            "Adjusted Rand Index",
            plot_dir / f"clustering_ari_{suffix}_all_methods.png",
        )
        _plot_clustering_metric(
            metrics,
            representation,
            "matched_accuracy",
            "Hungarian-matched accuracy",
            plot_dir / f"clustering_accuracy_{suffix}_all_methods.png",
        )
    for condition, suffix in [
        ("matched_eot", "matched_eot"),
        ("matched_fixed", "matched_fixed"),
        ("oce_last_to_eot", "oce_to_eot"),
    ]:
        rows = distances[distances["condition"] == condition]
        margin_pivot = rows.pivot(index="animal", columns="space_label", values="margin").reindex(
            index=CONCEPTS, columns=space_labels
        )
        rank_pivot = rows.pivot(
            index="animal", columns="space_label", values="own_centroid_rank"
        ).reindex(index=CONCEPTS, columns=space_labels)
        cross_note = "Intentional cross-readout condition" if condition == "oce_last_to_eot" else "Matched readout condition"
        _annotated_heatmap(
            margin_pivot.to_numpy(),
            space_labels,
            f"Per-animal cosine margin: {CONDITION_LABELS[condition]}",
            f"Positive cells favor the animal's own centroid; {cross_note.lower()}",
            plot_dir / f"margin_heatmap_{suffix}.png",
            margin=True,
        )
        _annotated_heatmap(
            rank_pivot.to_numpy(),
            space_labels,
            f"Per-animal own-centroid rank: {CONDITION_LABELS[condition]}",
            f"Rank 1 is nearest among eight centroids; {cross_note.lower()}",
            plot_dir / f"rank_heatmap_{suffix}.png",
            margin=False,
        )
    _plot_prototype_summary(summary, plot_dir / "prototype_layer_summary.png")
    difference = comparison.pivot(
        index="animal", columns="space_label", values="oce_minus_matched_eot_margin"
    ).reindex(index=CONCEPTS, columns=space_labels)
    _annotated_heatmap(
        difference.to_numpy(),
        space_labels,
        "OCE last-token readout minus matched-EOT margin",
        "Positive means the cross-readout improves own-vs-other separation relative to matched EOT",
        plot_dir / "readout_margin_difference.png",
        margin=True,
    )
    chart_map = [
        {
            "report_section": "Description Clustering Across Text Space and W0 Layers",
            "question": f"How does {representation} clustering vary by method and space?",
            "family": "ordered comparison",
            "chart_type": "multi-series line",
            "fields": ["space_label", "clustering_method", metric],
            "path": f"plots/clustering_{metric_name}_{suffix}_all_methods.png",
        }
        for representation, suffix in [("eot", "eot"), ("fixed_suffix", "fixed")]
        for metric, metric_name in [("ari", "ari"), ("matched_accuracy", "accuracy")]
    ]
    chart_map.extend({
        "report_section": "Per-Animal Distances Across All W0 Layers",
        "question": f"How stable are individual animals under {CONDITION_LABELS[condition]}?",
        "family": "matrix",
        "chart_type": kind,
        "fields": ["animal", "space_label", "margin" if "margin" in kind else "own_centroid_rank"],
        "path": f"plots/{kind}_{suffix}.png",
    } for condition, suffix in [
        ("matched_eot", "matched_eot"),
        ("matched_fixed", "matched_fixed"),
        ("oce_last_to_eot", "oce_to_eot"),
    ] for kind in ["margin_heatmap", "rank_heatmap"])
    chart_map.extend([
        {
            "report_section": "Per-Animal Distances Across All W0 Layers",
            "question": "How do rank-1 counts and mean margins vary over all spaces?",
            "family": "small multiples",
            "chart_type": "line panels",
            "fields": ["space_label", "condition", "rank1_count_out_of_8", "mean_margin"],
            "path": "plots/prototype_layer_summary.png",
        },
        {
            "report_section": "Readout Comparison",
            "question": "Where does OCE last-token change margin relative to matched EOT?",
            "family": "matrix",
            "chart_type": "diverging heatmap",
            "fields": ["animal", "space_label", "oce_minus_matched_eot_margin"],
            "path": "plots/readout_margin_difference.png",
        },
    ])
    atomic_write_text(output_dir / "chart_map.json", json.dumps(chart_map, indent=2) + "\n")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def _text_prototype_table(distances: pd.DataFrame, condition: str) -> str:
    rows = distances[(distances["layer_index"] == -1) & (distances["condition"] == condition)]
    rows = rows.set_index("animal").loc[CONCEPTS].reset_index()
    formatted = [[
        row.animal,
        f"{row.own_centroid_cosine_distance:.4f}",
        row.nearest_other_concept,
        f"{row.nearest_other_cosine_distance:.4f}",
        f"{row.margin:+.4f}",
        str(int(row.own_centroid_rank)),
    ] for row in rows.itertuples()]
    return _markdown_table(
        ["Animal", "Own distance", "Nearest other", "Other distance", "Margin", "Own rank"],
        formatted,
    )


def _build_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    norms: pd.DataFrame,
    distances: pd.DataFrame,
    summary: pd.DataFrame,
    inventory: pd.DataFrame,
) -> None:
    available_layer_indices = sorted(
        int(value) for value in metrics.loc[metrics["layer_index"] >= 0, "layer_index"].unique()
    )
    space_count = len(available_layer_indices) + 1
    comparison_counts = {}
    raw_normalized_ari_gap = {}
    for representation in REPRESENTATIONS:
        subset = metrics[metrics["representation"] == representation]
        pivot = subset.pivot(index="layer_index", columns="clustering_method", values="ari")
        comparison_counts[representation] = {
            "raw_beats_spherical": int((pivot["euclidean_raw"] > pivot["spherical_normalized"]).sum()),
            "normalized_beats_spherical": int((pivot["euclidean_normalized"] > pivot["spherical_normalized"]).sum()),
        }
        raw_normalized_ari_gap[representation] = float(
            (pivot["euclidean_raw"] - pivot["euclidean_normalized"]).abs().mean()
        )
    best_worst: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}
    for representation in REPRESENTATIONS:
        for method in METHODS:
            subset = metrics[
                (metrics["representation"] == representation)
                & (metrics["clustering_method"] == method)
                & (metrics["layer_index"] >= 0)
            ]
            best_worst[(representation, method)] = (
                subset.sort_values(["ari", "layer_index"], ascending=[False, True]).iloc[0],
                subset.sort_values(["ari", "layer_index"], ascending=[True, True]).iloc[0],
            )
    eot_wins = {}
    for method in METHODS:
        method_rows = metrics[metrics["clustering_method"] == method]
        pivot = method_rows.pivot(index="layer_index", columns="representation", values="ari")
        eot_wins[method] = int((pivot["eot"] > pivot["fixed_suffix"]).sum())

    clustering_table = []
    for representation, representation_label in [("eot", "Unsuffixed EOT"), ("fixed_suffix", "Fixed suffix")]:
        for method in METHODS:
            text_row = metrics[
                (metrics["representation"] == representation)
                & (metrics["clustering_method"] == method)
                & (metrics["layer_index"] == -1)
            ].iloc[0]
            best, worst = best_worst[(representation, method)]
            clustering_table.append([
                representation_label,
                METHOD_LABELS[method],
                f"{text_row.ari:.4f}",
                f"L{int(best.layer_index)} ({best.ari:.4f})",
                f"L{int(worst.layer_index)} ({worst.ari:.4f})",
            ])

    summary_table = []
    for layer_index, label in [
        (-1, "Text"), *[(index, f"L{index}") for index in available_layer_indices]
    ]:
        current = summary[summary["layer_index"] == layer_index].set_index("condition")
        summary_table.append([
            label,
            *[
                f"{int(current.loc[condition, 'rank1_count_out_of_8'])}/8; "
                f"{current.loc[condition, 'mean_margin']:+.3f}"
                for condition in CONDITIONS
            ],
        ])

    description_norms = norms[norms["vector_type"] == "description"]
    norm_range = (
        float(description_norms["min_norm"].min()),
        float(description_norms["max_norm"].max()),
    )
    shape_counts = inventory.groupby(["w0_output_dim", "w0_input_dim"]).size().to_dict()
    shape_text = ", ".join(
        f"{count}×({output_dim}×{input_dim})"
        for (output_dim, input_dim), count in shape_counts.items()
    )
    best_layers_by_method = {
        representation: [int(best_worst[(representation, method)][0].layer_index) for method in METHODS]
        for representation in REPRESENTATIONS
    }
    worst_layers_by_method = {
        representation: [int(best_worst[(representation, method)][1].layer_index) for method in METHODS]
        for representation in REPRESENTATIONS
    }
    best_text = "; ".join(
        f"{representation}: {layers}" for representation, layers in best_layers_by_method.items()
    )
    worst_text = "; ".join(
        f"{representation}: {layers}" for representation, layers in worst_layers_by_method.items()
    )

    report = f"""# Revised Original-W0 Geometry of Balanced Animal Descriptions

## 1. Research Questions

This revision tests whether unchanged original SD 1.4 cross-attention projections preserve description clustering under spherical and ordinary Euclidean objectives, and reports explicit concept-name proximity for all eight animals in text space and all {len(available_layer_indices)} selected `W0` spaces. The analysis uses the same final balanced 8×50 dataset and does not generate images, load an edited checkpoint, or modify model weights.

## 2. Dataset and Representations

The dataset contains exactly 400 name-free descriptions: 50 each for cat, dog, fox, bear, wolf, rabbit, deer, and horse. The dataset and cached-embedding audits pass, and the ordered original `attn2.to_v` inventory contains {shape_text} matrices.

Unsuffixed EOT is the final hidden state at `attention_mask.sum(dim=1) - 1` with no added text. Fixed suffix appends exactly `{FIXED_SUFFIX}` and reads the contextual hidden state of the final content token `concept`. The OCE-faithful bare-name representation reads the final content token immediately before EOT at `attention_mask.sum() - 2`; the matched name controls retain bare-name EOT and fixed-suffix `concept` readouts.

## 3. Why Spherical and Euclidean K-Means Are Both Tested

Spherical k-means emphasizes vector direction: it receives row-normalized inputs, assigns by cosine similarity, and normalizes updated centroids. Raw Euclidean k-means is not assumed to be worse; it receives unnormalized vectors and therefore uses both direction and magnitude. Normalized Euclidean k-means starts from the same unit vectors as spherical k-means but uses ordinary arithmetic centroid updates without manually renormalizing centroids, isolating that algorithmic difference.

Across the {space_count} spaces, raw Euclidean ARI exceeds spherical ARI in {comparison_counts['eot']['raw_beats_spherical']}/{space_count} EOT cases and {comparison_counts['fixed_suffix']['raw_beats_spherical']}/{space_count} fixed-suffix cases. Normalized Euclidean exceeds spherical in {comparison_counts['eot']['normalized_beats_spherical']}/{space_count} and {comparison_counts['fixed_suffix']['normalized_beats_spherical']}/{space_count} cases, respectively. Ordinary Euclidean k-means therefore does not necessarily perform worse.

## 4. Normalization Protocol

For every text vector `h`, the normalized control is `h / ||h||₂`. For every layer, raw output is `W0_l h`, followed by row normalization for spherical clustering and all cosine prototype comparisons. Raw description norms range from {norm_range[0]:.3f} to {norm_range[1]:.3f} across concepts, representations, and spaces; full description/name summaries are in `vector_norm_summary.csv`.

Raw and normalized vectors are saved separately. Spherical inputs and centers are unit-normalized. Euclidean-raw inputs are not normalized. Euclidean-normalized inputs are unit-normalized, but their fitted centroids are not manually renormalized.

The mean absolute ARI difference between raw and normalized Euclidean k-means is {raw_normalized_ari_gap['eot']:.4f} for EOT and {raw_normalized_ari_gap['fixed_suffix']:.4f} for fixed suffix across Text and all selected layers. This confirms that retaining vector magnitude changes results, but it does not by itself identify magnitude as the causal source of any individual-layer change.

## 5. Description Clustering Across Text Space and W0 Layers

{_markdown_table(['Representation', 'Method', 'Text ARI', 'Best W0 ARI', 'Worst W0 ARI'], clustering_table)}

In method order spherical, raw Euclidean, and normalized Euclidean, the post-hoc best W0 layer indices are {best_text}, while the worst indices are {worst_text}. Thus the best layer is not stable across methods. The worst layer is more stable for fixed suffix, but EOT shifts from L12 under spherical to L13 under both Euclidean controls. EOT has higher ARI than fixed suffix in {eot_wins['spherical_normalized']}/{space_count} spherical spaces, {eot_wins['euclidean_raw']}/{space_count} raw-Euclidean spaces, and {eot_wins['euclidean_normalized']}/{space_count} normalized-Euclidean spaces. EOT therefore usually, but not universally, outperforms fixed suffix.

The next four figures show all spaces and all three clustering methods. Their silhouettes use cosine only for spherical normalized and Euclidean distance for both ordinary k-means controls.

![EOT ARI across methods](plots/clustering_ari_eot_all_methods.png)

![Fixed-suffix ARI across methods](plots/clustering_ari_fixed_all_methods.png)

![EOT matched accuracy across methods](plots/clustering_accuracy_eot_all_methods.png)

![Fixed-suffix matched accuracy across methods](plots/clustering_accuracy_fixed_all_methods.png)

## 6. Per-Animal Name-to-Description Distances in Text Space

All prototype values below use cosine distance after normalizing every description and prototype, averaging the 50 normalized descriptions for a concept, and normalizing that mean. Positive margin means the name is closer to its own centroid than to every other animal centroid.

### Matched EOT

{_text_prototype_table(distances, 'matched_eot')}

### Matched fixed suffix

{_text_prototype_table(distances, 'matched_fixed')}

### OCE-last-token -> EOT descriptions

{_text_prototype_table(distances, 'oce_last_to_eot')}

The third table is intentionally cross-readout. Its absolute distances are not pooled with either matched condition.

## 7. Per-Animal Distances Across All W0 Layers

Each heatmap retains all eight animals and Text plus every selected W0 layer. Margin heatmaps use a diverging scale centered at zero; rank heatmaps show the own-centroid position among eight centroids. In the full analysis, no single illustrative layer, including L8, replaces the all-layer results.

![Matched EOT margins](plots/margin_heatmap_matched_eot.png)

![Matched fixed-suffix margins](plots/margin_heatmap_matched_fixed.png)

![OCE last-token to EOT margins](plots/margin_heatmap_oce_to_eot.png)

![Matched EOT ranks](plots/rank_heatmap_matched_eot.png)

![Matched fixed-suffix ranks](plots/rank_heatmap_matched_fixed.png)

![OCE last-token to EOT ranks](plots/rank_heatmap_oce_to_eot.png)

The compact all-space summary below gives `rank-1 count / 8; mean margin`. Mean margin is only a summary: a positive mean does not imply that every animal succeeded, so the individual rows in `prototype_distance_by_animal.csv` remain primary.

{_markdown_table(['Space', 'Matched EOT', 'Matched fixed', 'OCE-last -> EOT'], summary_table)}

![Prototype layer summary](plots/prototype_layer_summary.png)

## 8. Readout Comparison

The difference heatmap subtracts matched-EOT margin from OCE-last-token-to-EOT margin for every animal and space. It isolates where changing the name readout changes own-versus-other separation while preserving the cross-readout label.

![Readout margin difference](plots/readout_margin_difference.png)

Text-space `c` and layer-wise `W0c` are both tested, but `c` is never directly compared with `W0c`: different W0 layers can have different output dimensions and coordinate systems.

## 9. Main Findings

- Ordinary Euclidean k-means is a genuine robustness control and is not uniformly worse than spherical k-means; the exact win counts are reported above.
- Raw Euclidean differs from both normalized methods because the pre-normalization norms vary and remain part of its distance objective. The norm audit should be consulted before attributing raw-Euclidean changes only to direction.
- The best W0 layer changes across all three clustering methods for both representations. The fixed-suffix worst layer stays at L12, whereas the EOT worst layer is L12 for spherical and L13 for both Euclidean controls.
- EOT outperforms fixed suffix in most, but not all, spaces: {eot_wins['spherical_normalized']}/{space_count}, {eot_wins['euclidean_raw']}/{space_count}, and {eot_wins['euclidean_normalized']}/{space_count} for spherical, raw Euclidean, and normalized Euclidean, respectively. This conclusion is not inferred from L8 alone.
- Name-to-description conclusions are animal-, layer-, and readout-specific. Rank-1 count and mean margin summarize the eight rows but never replace them.

## 10. Limitations

This is a descriptive geometry analysis, not a test of whether OCE can erase a concept distribution. Prototype centroids use true labels only after clustering and do not establish semantic identity or causal representation. Best/worst layer labels are post-hoc. Raw Euclidean can be sensitive to magnitude for reasons unrelated to concept identity. Cross-readout OCE-last-token distances intentionally mix name and description readout rules and cannot be interpreted alone as pure semantic distance.
"""
    atomic_write_text(output_dir / "report.md", report)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
    previous_output = Path(args.embedding_cache).expanduser().resolve()
    if output_dir == previous_output:
        raise ValueError("Revised output directory must differ from the previous embedding-cache directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path, validation_path, source_config_path = _resolve_dataset(args.dataset)
    source_dir = dataset_path.parent
    rows = read_jsonl(dataset_path)
    source_config = json.loads(source_config_path.read_text())
    dataset_audit = _audit_dataset(
        rows, dataset_path, validation_path, source_config_path, source_config
    )
    oce_audit = _inspect_oce_source(Path(args.oce_repo).expanduser().resolve())
    description_text_raw, name_text_raw, cache_audit = _load_cached_embeddings(
        previous_output, source_dir, dataset_audit, output_dir
    )
    dataset_audit.update(cache_audit)
    cached_dataset_audit = json.loads((previous_output / "dataset_audit.json").read_text())
    dataset_audit["checks"].update({
        "cached_no_unsuffixed_prompt_truncated": bool(
            cached_dataset_audit["checks"]["no_unsuffixed_prompt_truncated"]
        ),
        "cached_no_fixed_suffix_prompt_truncated": bool(
            cached_dataset_audit["checks"]["no_fixed_suffix_prompt_truncated"]
        ),
    })
    dataset_audit["status"] = (
        "passed" if all(dataset_audit["checks"].values()) else "failed"
    )
    atomic_write_text(output_dir / "dataset_audit.json", json.dumps(dataset_audit, indent=2) + "\n")

    config = _runtime_config(args.model_id, args.device, args.batch_size, args.random_seed)
    torch.manual_seed(int(args.random_seed))
    np.random.seed(int(args.random_seed))
    pipe = load_original_pipeline(config, purpose="embedding", include_vae=False)
    if int(pipe.text_encoder.config.hidden_size) != 768 or int(pipe.tokenizer.model_max_length) != 77:
        raise RuntimeError("Loaded model is not the expected SD 1.4 CLIP text configuration")
    modules = original_projection_modules(pipe, "to_v")
    if args.layer_limit is not None:
        modules = modules[: int(args.layer_limit)]
    if not modules:
        raise RuntimeError("No original W0 layers selected")
    if args.layer_limit is None and len(modules) != 16:
        raise RuntimeError(f"Expected all 16 original attn2.to_v layers, found {len(modules)}")

    cached_inventory = pd.read_csv(previous_output / "layer_inventory.csv")
    before_hashes = {name: _tensor_sha256(module.weight) for name, module in modules}
    inventory_rows = []
    for index, (name, module) in enumerate(modules):
        cached = cached_inventory[cached_inventory["layer_index"] == index]
        expected_name = f"unet.{name}"
        matches_cache = bool(
            len(cached) == 1
            and str(cached.iloc[0]["full_module_name"]) == expected_name
            and int(cached.iloc[0]["w0_input_dim"]) == int(module.weight.shape[1])
            and int(cached.iloc[0]["w0_output_dim"]) == int(module.weight.shape[0])
            and str(cached.iloc[0]["w0_sha256_before"]) == before_hashes[name]
        )
        inventory_rows.append({
            "layer_index": index,
            "full_module_name": expected_name,
            "pipeline_relative_module_name": name,
            "matrix_type": "to_v",
            "w0_input_dim": int(module.weight.shape[1]),
            "w0_output_dim": int(module.weight.shape[0]),
            "weight_dtype": str(module.weight.dtype),
            "requires_grad": bool(module.weight.requires_grad),
            "w0_sha256_before": before_hashes[name],
            "matches_existing_layer_inventory": matches_cache,
        })
    if not all(row["matches_existing_layer_inventory"] for row in inventory_rows):
        raise RuntimeError("Current original W0 inventory differs from the existing audited inventory")

    description_raw, description_normalized, name_raw, name_normalized = (
        _build_vector_spaces_without_labels(
            description_text_raw, name_text_raw, modules, args.device
        )
    )
    _save_vector_archives(
        output_dir, description_raw, description_normalized, name_raw, name_normalized
    )
    fits = _fit_all_clustering_without_labels(
        description_raw, description_normalized, config["spherical_kmeans"]
    )
    metrics, recall = _evaluate_all_clustering(
        rows, modules, description_raw, description_normalized, fits, output_dir
    )
    norms = _vector_norm_summary(rows, modules, description_raw, name_raw)
    norms.to_csv(output_dir / "vector_norm_summary.csv", index=False)
    distances, prototype_summary, margin_comparison, prototype_checks = _prototype_analysis(
        rows, modules, description_normalized, name_normalized
    )
    distances.to_csv(output_dir / "prototype_distance_by_animal.csv", index=False)
    prototype_summary.to_csv(output_dir / "prototype_layer_summary.csv", index=False)
    margin_comparison.to_csv(output_dir / "readout_margin_comparison.csv", index=False)

    after_hashes = {name: _tensor_sha256(module.weight) for name, module in modules}
    unchanged = {name: before_hashes[name] == after_hashes[name] for name, _ in modules}
    if not all(unchanged.values()):
        raise RuntimeError(f"At least one original W0 matrix changed: {unchanged}")
    for row, (name, _) in zip(inventory_rows, modules):
        row["w0_sha256_after"] = after_hashes[name]
        row["unchanged"] = unchanged[name]
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(output_dir / "layer_inventory.csv", index=False)
    integrity = {
        "status": "passed",
        "all_selected_w0_matrices_unchanged": all(unchanged.values()),
        "layer_count": len(modules),
        "matrix_type": "to_v",
        "edited_checkpoint_loaded": False,
        "model_edit_function_called": False,
        "pw0_computed": False,
        "image_generation_performed": False,
        "before": before_hashes,
        "after": after_hashes,
    }
    atomic_write_text(output_dir / "w0_immutability.json", json.dumps(integrity, indent=2) + "\n")

    _create_plots(metrics, distances, prototype_summary, margin_comparison, output_dir)
    _build_report(output_dir, metrics, norms, distances, prototype_summary, inventory)

    model_info = model_metadata(pipe, config, projection="to_v")
    experiment = {
        "experiment_name": "balanced_paired_w0_geometry_revised",
        "dataset": dataset_audit,
        "model": model_info,
        "oce_repository_audit": oce_audit,
        "concepts": CONCEPTS,
        "readouts": {
            "description_eot": "unsuffixed; attention_mask.sum(dim=1)-1",
            "description_fixed": f"description plus {FIXED_SUFFIX!r}; contextual concept token",
            "name_last": "bare name via OCE encode_prompt rule; attention_mask.sum()-2",
            "name_eot": "bare name; attention_mask.sum(dim=1)-1",
            "name_fixed": f"bare name plus {FIXED_SUFFIX!r}; contextual concept token",
        },
        "clustering_methods": {
            "spherical_normalized": {
                **config["spherical_kmeans"],
                "normalization": "row_l2",
                "silhouette_metric": "cosine",
            },
            "euclidean_raw": {
                "implementation": "sklearn.cluster.KMeans",
                "k": 8,
                "n_init": 50,
                "max_iter": 300,
                "tolerance": 1e-6,
                "random_seed": int(args.random_seed),
                "normalization": "none",
                "silhouette_metric": "euclidean",
                "manual_centroid_renormalization": False,
            },
            "euclidean_normalized": {
                "implementation": "sklearn.cluster.KMeans",
                "k": 8,
                "n_init": 50,
                "max_iter": 300,
                "tolerance": 1e-6,
                "random_seed": int(args.random_seed),
                "normalization": "row_l2",
                "silhouette_metric": "euclidean",
                "manual_centroid_renormalization": False,
            },
        },
        "prototype_geometry": (
            "row-normalize descriptions; mean 50 normalized rows per concept; normalize mean; "
            "row-normalize every name vector; cosine distance within each space only"
        ),
        "labels_available_to_clustering_fit": False,
        "fit_boundary_signature": str(inspect.signature(_fit_all_clustering_without_labels)),
        "layer_limit": args.layer_limit,
        "package_versions": package_versions(),
        "cli": {
            "dataset": args.dataset,
            "embedding_cache": str(previous_output),
            "model_id_or_path": args.model_id,
            "oce_repository": str(Path(args.oce_repo).expanduser().resolve()),
            "output_directory": str(output_dir),
            "device": args.device,
            "batch_size": int(args.batch_size),
            "random_seed": int(args.random_seed),
        },
    }
    atomic_write_text(output_dir / "experiment_config.json", json.dumps(experiment, indent=2) + "\n")

    token_audit = pd.read_csv(output_dir / "name_tokenization_audit.csv")
    expected_spaces = len(modules) + 1
    input_norm_tolerance = 1e-5
    spherical_metrics = metrics[metrics["clustering_method"] == "spherical_normalized"]
    raw_metrics = metrics[metrics["clustering_method"] == "euclidean_raw"]
    normalized_metrics = metrics[metrics["clustering_method"] == "euclidean_normalized"]
    expected_confusions = expected_spaces * len(REPRESENTATIONS) * len(METHODS)
    checks = {
        "status": "passed",
        "dataset_is_final_balanced_8x50": dataset_audit["status"] == "passed" and len(rows) == 400,
        "all_16_w0_spaces_included": args.layer_limit is not None or len(modules) == 16,
        "text_and_all_selected_w0_spaces_in_metrics": metrics["layer_id"].nunique() == expected_spaces,
        "all_selected_w0_unchanged": all(unchanged.values()),
        "all_layers_match_existing_inventory": all(
            row["matches_existing_layer_inventory"] for row in inventory_rows
        ),
        "no_edited_checkpoint_loaded": not oce_audit["edited_checkpoint_loaded"],
        "no_image_generation_performed": True,
        "fit_boundary_has_no_label_or_concept_parameter": not any(
            "label" in name or "concept" in name
            for name in inspect.signature(_fit_all_clustering_without_labels).parameters
        ),
        "all_metrics_record_labels_unavailable_to_fit": bool(
            (~metrics["labels_available_to_fit"].astype(bool)).all()
        ),
        "spherical_inputs_are_unit_normalized": bool(
            np.allclose(spherical_metrics["input_norm_min"], 1.0, atol=input_norm_tolerance)
            and np.allclose(spherical_metrics["input_norm_max"], 1.0, atol=input_norm_tolerance)
        ),
        "spherical_centers_are_unit_normalized": bool(
            np.allclose(spherical_metrics["center_norm_min"], 1.0, atol=input_norm_tolerance)
            and np.allclose(spherical_metrics["center_norm_max"], 1.0, atol=input_norm_tolerance)
        ),
        "raw_euclidean_inputs_are_not_normalized": bool(
            ((raw_metrics["input_norm_min"] - 1.0).abs() > 1e-3).any()
            or ((raw_metrics["input_norm_max"] - 1.0).abs() > 1e-3).any()
        ),
        "normalized_euclidean_inputs_are_unit_normalized": bool(
            np.allclose(normalized_metrics["input_norm_min"], 1.0, atol=input_norm_tolerance)
            and np.allclose(normalized_metrics["input_norm_max"], 1.0, atol=input_norm_tolerance)
        ),
        "normalized_euclidean_centers_not_manually_renormalized": bool(
            (~normalized_metrics["centers_normalized_during_fit"].astype(bool)).all()
            and ((normalized_metrics["center_norm_min"] - 1.0).abs() > 1e-6).any()
        ),
        "all_eight_animals_individual_in_every_prototype_space_condition": bool(
            distances.groupby(["condition", "layer_id"])["animal"].nunique().eq(8).all()
        ),
        "three_readout_conditions_remain_separate": set(distances["condition"]) == set(CONDITIONS)
            and distances.groupby("condition").size().eq(expected_spaces * 8).all(),
        "prototype_vectors_and_centroids_are_unit_normalized": bool(
            abs(prototype_checks["prototype_input_norm_min"] - 1.0) <= input_norm_tolerance
            and abs(prototype_checks["prototype_input_norm_max"] - 1.0) <= input_norm_tolerance
            and abs(prototype_checks["centroid_norm_min"] - 1.0) <= input_norm_tolerance
            and abs(prototype_checks["centroid_norm_max"] - 1.0) <= input_norm_tolerance
        ),
        "all_confusion_matrices_present": len(list((output_dir / "confusion_matrices").glob("*.csv")))
            == expected_confusions,
        "all_name_token_indices_preserved": bool(
            (token_audit["oce_last_content_index"] == token_audit["effective_token_length"] - 2).all()
            and (token_audit["eot_index"] == token_audit["effective_token_length"] - 1).all()
        ),
    }
    checks = {
        key: bool(value) if isinstance(value, np.bool_) else value
        for key, value in checks.items()
    }
    checks["status"] = "passed" if all(
        value for key, value in checks.items() if key != "status"
    ) else "failed"
    atomic_write_text(output_dir / "analysis_checks.json", json.dumps(checks, indent=2) + "\n")
    if checks["status"] != "passed":
        raise RuntimeError(f"Final revised-analysis checks failed: {checks}")

    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    empty = [
        name for name in REQUIRED_OUTPUTS
        if (output_dir / name).exists() and (output_dir / name).stat().st_size == 0
    ]
    if missing or empty:
        raise RuntimeError(f"Output validation failed: missing={missing}, empty={empty}")
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revised balanced-paired clustering and original-W0 prototype geometry"
    )
    parser.add_argument("--dataset", required=True, help="Balanced 8x50 output directory or JSONL")
    parser.add_argument(
        "--embedding-cache", required=True, help="Existing balanced_paired_w0_geometry output directory"
    )
    parser.add_argument("--model-id", required=True, help="Original SD 1.4 identifier or local path")
    parser.add_argument("--oce-repo", required=True, help="OCE repository root containing oce.py")
    parser.add_argument("--output", required=True, help="New revised output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=314159)
    parser.add_argument("--layer-limit", type=int, default=None, help="Smoke test: first N W0 layers")
    parser.add_argument("--force", action="store_true", help="Replace files only in the selected revised output")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
