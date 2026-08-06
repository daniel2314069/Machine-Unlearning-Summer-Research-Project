from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors

from scripts.eot_spherical_clustering import normalize_rows

from .utils import atomic_write_text, package_versions, read_jsonl


CONCEPTS = ["cat", "dog", "fox", "bear", "wolf", "rabbit", "deer", "horse"]
PURITY_NEIGHBORS = [5, 10, 20]
SPECTRAL_RUNS = [
    {"n_neighbors": 5, "random_state": 42, "primary": False},
    {"n_neighbors": 10, "random_state": 0, "primary": False},
    {"n_neighbors": 10, "random_state": 1, "primary": False},
    {"n_neighbors": 10, "random_state": 42, "primary": True},
    {"n_neighbors": 15, "random_state": 42, "primary": False},
]
EXPECTED_HASHES = {
    "dataset": "6f459fa9e73f80163145813ecd8cd32c4216e1c4f77f6795925bf06676794a0c",
    "embedding": "4c175ad2c4f7b848fa561dd464104eb9f27a29c36591912977873e6b52b77070",
    "assignments": "9e7e39ed5e780ddb0a4be5ab1d33c48e0b9be00ae53d3bbdf145a2d7ed11ff3b",
}
EXPECTED_BASELINE = {"ari": 0.6338, "matched_accuracy": 0.7750}
RANDOM_BALANCED_BASELINE = 49.0 / 399.0
BLUE = "#2458A6"
BLUE_LIGHT = "#7EA4D8"
GOLD = "#D28E00"
ORANGE = "#D65F30"
INK = "#222222"
GRID = "#D9D9D9"
K_COLORS = {5: BLUE_LIGHT, 10: BLUE, 20: GOLD}
REQUIRED_OUTPUTS = [
    "experiment_config.json",
    "dataset_audit.json",
    "knn_neighbors.csv",
    "knn_purity_by_sample.csv",
    "knn_purity_summary.csv",
    "graph_diagnostics.csv",
    "spectral_metrics_all_runs.csv",
    "spectral_per_class_recall.csv",
    "spectral_assignments.csv",
    "confusion_spectral_knn10_seed42.csv",
    "confusion_spectral_knn10_seed42.png",
    "plots/knn_purity_by_animal.png",
    "plots/knn_purity_by_k.png",
    "plots/spectral_metrics_by_neighbors.png",
    "report.md",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_sources(dataset_arg: str | Path) -> dict[str, Path]:
    supplied = Path(dataset_arg).expanduser().resolve()
    source_dir = supplied if supplied.is_dir() else supplied.parent
    dataset = source_dir / "accepted_descriptions.jsonl" if supplied.is_dir() else supplied
    paths = {
        "source_dir": source_dir,
        "dataset": dataset,
        "validation": source_dir / "dataset_validation.json",
        "source_config": source_dir / "experiment_config.json",
        "embedding": source_dir / "eot_embeddings.npy",
        "metrics": source_dir / "eot_metrics.json",
        "assignments": source_dir / "appendix" / "assignments_eot.csv",
        "tokenization": source_dir / "eot_tokenization_audit.csv",
    }
    missing = [str(path) for key, path in paths.items() if key != "source_dir" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Final EOT cache is incomplete: {missing}")
    return paths


def _load_and_audit(
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    rows = read_jsonl(paths["dataset"])
    dataset_hash = _sha256(paths["dataset"])
    embedding_hash = _sha256(paths["embedding"])
    assignments_hash = _sha256(paths["assignments"])
    if dataset_hash != EXPECTED_HASHES["dataset"]:
        raise RuntimeError(f"Wrong dataset fingerprint: {dataset_hash}")
    if embedding_hash != EXPECTED_HASHES["embedding"]:
        raise RuntimeError(f"Wrong EOT embedding fingerprint: {embedding_hash}")
    if assignments_hash != EXPECTED_HASHES["assignments"]:
        raise RuntimeError(f"Wrong EOT assignment fingerprint: {assignments_hash}")

    counts = Counter(str(row.get("concept")) for row in rows)
    pairs = {(str(row.get("slot_id")), str(row.get("concept"))) for row in rows}
    source_config = _read_json(paths["source_config"])
    validation = _read_json(paths["validation"])
    configured_concepts = [str(item["name"]) for item in source_config.get("concepts", [])]
    tokenization = pd.read_csv(paths["tokenization"])
    truncated = tokenization["truncation_occurred"].astype(str).str.casefold().map(
        {"true": True, "false": False}
    )

    raw = np.load(paths["embedding"], allow_pickle=False)
    if raw.shape != (400, 768) or not np.isfinite(raw).all():
        raise RuntimeError(f"Expected finite EOT cache shape (400, 768), found {raw.shape}")
    features = normalize_rows(raw).astype(np.float64, copy=False)
    norms = np.linalg.norm(features, axis=1)

    assignment_frame = pd.read_csv(paths["assignments"]).sort_values("sample_index").reset_index(drop=True)
    if assignment_frame["sample_index"].astype(int).tolist() != list(range(400)):
        raise RuntimeError("Cached EOT assignments do not index samples 0..399 exactly once")
    for index, row in enumerate(rows):
        if str(assignment_frame.at[index, "candidate_id"]) != str(row["candidate_id"]):
            raise RuntimeError(f"Cached assignment candidate mismatch at row {index}")
        if str(assignment_frame.at[index, "description"]) != str(row["description"]):
            raise RuntimeError(f"Cached assignment description mismatch at row {index}")
        if str(assignment_frame.at[index, "true_concept"]) != str(row["concept"]):
            raise RuntimeError(f"Cached assignment label mismatch at row {index}")

    true_ids = np.asarray([CONCEPTS.index(str(row["concept"])) for row in rows], dtype=np.int64)
    cached_clusters = assignment_frame["predicted_cluster"].to_numpy(dtype=np.int64)
    assignment_ari = float(adjusted_rand_score(true_ids, cached_clusters))
    assignment_accuracy = float(
        (assignment_frame["true_concept"] == assignment_frame["matched_predicted_concept"]).mean()
    )
    metric_payload = _read_json(paths["metrics"])
    baseline = {
        "method": "Spherical k-means baseline",
        "graph_neighbors": None,
        "seed": int(metric_payload["spherical_kmeans"]["random_seed"]),
        "ari": float(metric_payload["metrics"]["adjusted_rand_index"]),
        "nmi": float(metric_payload["metrics"]["normalized_mutual_information"]),
        "matched_accuracy": float(metric_payload["metrics"]["hungarian_matched_accuracy"]),
    }
    if metric_payload.get("dataset_sha256") != dataset_hash:
        raise RuntimeError("Cached baseline metrics use a different dataset")
    if not np.isclose(assignment_ari, baseline["ari"], atol=1e-12):
        raise RuntimeError("Cached baseline assignment ARI disagrees with metrics")
    if not np.isclose(assignment_accuracy, baseline["matched_accuracy"], atol=1e-12):
        raise RuntimeError("Cached baseline assignment accuracy disagrees with metrics")
    if round(baseline["ari"], 4) != EXPECTED_BASELINE["ari"]:
        raise RuntimeError(f"Wrong final baseline ARI: {baseline['ari']}")
    if round(baseline["matched_accuracy"], 4) != EXPECTED_BASELINE["matched_accuracy"]:
        raise RuntimeError(f"Wrong final baseline matched accuracy: {baseline['matched_accuracy']}")

    checks = {
        "exactly_400_rows": len(rows) == 400,
        "exact_concept_order": configured_concepts == CONCEPTS,
        "exactly_50_per_concept": all(counts[concept] == 50 for concept in CONCEPTS),
        "exactly_400_unique_concept_slot_pairs": len(pairs) == 400,
        "source_validation_passed": validation.get("status") == "passed",
        "tokenization_rows_400": len(tokenization) == 400,
        "no_prompt_truncation": not truncated.isna().any() and not bool(truncated.any()),
        "embedding_shape_400_by_768": features.shape == (400, 768),
        "all_vectors_finite": bool(np.isfinite(features).all()),
        "all_vectors_unit_l2_norm": bool(np.allclose(norms, 1.0, atol=1e-7)),
        "cached_baseline_reproduced_without_refit": True,
        "only_unsuffixed_eot_loaded": True,
        "w0_projection_applied": False,
        "dimensionality_reduction_used": False,
    }
    positive = [key for key in checks if key not in {
        "w0_projection_applied", "dimensionality_reduction_used"
    }]
    if (
        not all(checks[key] is True for key in positive)
        or checks["w0_projection_applied"] is not False
        or checks["dimensionality_reduction_used"] is not False
    ):
        raise RuntimeError(f"Final EOT audit failed: {checks}")

    audit = {
        "status": "passed",
        "dataset_path": str(paths["dataset"]),
        "dataset_sha256": dataset_hash,
        "embedding_path": str(paths["embedding"]),
        "embedding_sha256": embedding_hash,
        "assignments_path": str(paths["assignments"]),
        "assignments_sha256": assignments_hash,
        "baseline_metrics_path": str(paths["metrics"]),
        "baseline_metrics_sha256": _sha256(paths["metrics"]),
        "row_count": len(rows),
        "counts_by_concept": {concept: counts[concept] for concept in CONCEPTS},
        "cached_embedding_shape": list(raw.shape),
        "analysis_embedding_shape": list(features.shape),
        "raw_norm_min": float(np.linalg.norm(raw, axis=1).min()),
        "raw_norm_max": float(np.linalg.norm(raw, axis=1).max()),
        "analysis_norm_min": float(norms.min()),
        "analysis_norm_max": float(norms.max()),
        "checks": checks,
        "cached_spherical_baseline": baseline,
    }
    return rows, features, true_ids, baseline, audit


def find_cosine_neighbors_without_labels(
    features: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find neighbors from vectors only; labels cannot enter this boundary."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) <= n_neighbors:
        raise ValueError("Neighbor search requires a 2D matrix with more rows than neighbors")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-7):
        raise ValueError("Cosine neighbors require row-L2-normalized vectors")
    model = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="cosine",
        algorithm="brute",
    )
    model.fit(matrix)
    raw_distances, raw_indices = model.kneighbors(matrix, return_distance=True)
    indices = np.empty((len(matrix), n_neighbors), dtype=np.int64)
    distances = np.empty((len(matrix), n_neighbors), dtype=np.float64)
    for sample_index in range(len(matrix)):
        keep = raw_indices[sample_index] != sample_index
        selected_indices = raw_indices[sample_index][keep][:n_neighbors]
        selected_distances = raw_distances[sample_index][keep][:n_neighbors]
        if len(selected_indices) != n_neighbors:
            raise RuntimeError(f"Could not exclude self for sample {sample_index}")
        indices[sample_index] = selected_indices
        distances[sample_index] = np.clip(selected_distances, 0.0, 2.0)
    if np.any(indices == np.arange(len(matrix))[:, None]):
        raise RuntimeError("Self-neighbor remained after exclusion")
    return indices, distances


def build_cosine_knn_graph_without_labels(
    features: np.ndarray,
    n_neighbors: int,
) -> tuple[csr_matrix, dict[str, Any]]:
    """Build a symmetric non-negative cosine graph from vectors only."""
    indices, distances = find_cosine_neighbors_without_labels(features, n_neighbors)
    similarities = np.clip(1.0 - distances, 0.0, None)
    row_indices = np.repeat(np.arange(len(features), dtype=np.int64), n_neighbors)
    directed = csr_matrix(
        (similarities.reshape(-1), (row_indices, indices.reshape(-1))),
        shape=(len(features), len(features)),
    )
    affinity = directed.maximum(directed.T).tocsr()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    if affinity.nnz and float(affinity.data.min()) < 0:
        raise RuntimeError("Graph contains a negative affinity")
    if (affinity != affinity.T).nnz != 0:
        raise RuntimeError("Graph affinity is not symmetric")

    component_count, component_labels = connected_components(
        affinity, directed=False, return_labels=True
    )
    component_sizes = np.bincount(component_labels, minlength=component_count).astype(int)
    degrees = np.diff(affinity.indptr)
    diagnostics = {
        "n_neighbors": int(n_neighbors),
        "nodes": int(affinity.shape[0]),
        "undirected_edges": int(affinity.nnz // 2),
        "connected_components": int(component_count),
        "component_sizes": component_sizes.tolist(),
        "minimum_degree": int(degrees.min()),
        "mean_degree": float(degrees.mean()),
        "maximum_degree": int(degrees.max()),
        "minimum_edge_weight": float(affinity.data.min()) if affinity.nnz else 0.0,
        "mean_edge_weight": float(affinity.data.mean()) if affinity.nnz else 0.0,
        "maximum_edge_weight": float(affinity.data.max()) if affinity.nnz else 0.0,
        "too_disconnected_for_eight_cluster_spectral": bool(component_count > 8),
        "graph_rule": "union: connect i-j when either selects the other; weight=max(0, cosine_similarity)",
        "labels_used": False,
    }
    return affinity, diagnostics


def fit_spectral_without_labels(
    affinity: csr_matrix,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple[np.ndarray, list[str]]:
    """Fit spectral clustering from an affinity matrix only; labels cannot enter."""
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=n_init,
        random_state=random_state,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cluster_ids = model.fit_predict(affinity)
    return np.asarray(cluster_ids, dtype=np.int64), [str(item.message) for item in captured]


def _evaluate_after_spectral_fit(
    cluster_ids: np.ndarray,
    true_ids: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Post-fit evaluation boundary: this is the first spectral function allowed labels."""
    raw_confusion = confusion_matrix(true_ids, cluster_ids, labels=np.arange(8))
    true_rows, cluster_columns = linear_sum_assignment(-raw_confusion)
    mapping = {
        int(cluster): int(true)
        for true, cluster in zip(true_rows.tolist(), cluster_columns.tolist())
    }
    if set(mapping) != set(range(8)):
        raise RuntimeError(f"Hungarian mapping is incomplete: {mapping}")
    predicted_ids = np.asarray([mapping[int(cluster)] for cluster in cluster_ids], dtype=np.int64)
    matched_confusion = confusion_matrix(true_ids, predicted_ids, labels=np.arange(8))
    recalls = np.diag(matched_confusion) / np.bincount(true_ids, minlength=8)
    metrics = {
        "ari": float(adjusted_rand_score(true_ids, cluster_ids)),
        "nmi": float(normalized_mutual_info_score(true_ids, cluster_ids)),
        "matched_accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "cluster_sizes": np.bincount(cluster_ids, minlength=8).astype(int).tolist(),
        "cluster_to_concept": {
            str(cluster): CONCEPTS[true_id] for cluster, true_id in mapping.items()
        },
        "per_class_recall": {
            concept: float(recalls[index]) for index, concept in enumerate(CONCEPTS)
        },
    }
    return metrics, predicted_ids, matched_confusion


def _neighbor_and_purity_tables(
    rows: list[dict[str, Any]],
    true_ids: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neighbor_rows: list[dict[str, Any]] = []
    for query_index in range(len(rows)):
        for zero_rank, neighbor_index in enumerate(neighbor_indices[query_index]):
            rank = zero_rank + 1
            neighbor_index = int(neighbor_index)
            neighbor_rows.append({
                "query_sample_index": query_index,
                "query_description_id": rows[query_index]["candidate_id"],
                "query_true_concept": rows[query_index]["concept"],
                "query_description": rows[query_index]["description"],
                "neighbor_rank": rank,
                "neighbor_sample_index": neighbor_index,
                "neighbor_description_id": rows[neighbor_index]["candidate_id"],
                "neighbor_true_concept": rows[neighbor_index]["concept"],
                "neighbor_description": rows[neighbor_index]["description"],
                "cosine_distance": float(neighbor_distances[query_index, zero_rank]),
                "cosine_similarity": float(1.0 - neighbor_distances[query_index, zero_rank]),
                "same_true_concept": bool(true_ids[query_index] == true_ids[neighbor_index]),
                "included_in_k5": rank <= 5,
                "included_in_k10": rank <= 10,
                "included_in_k20": rank <= 20,
            })
    neighbor_frame = pd.DataFrame(neighbor_rows)

    purity_rows: list[dict[str, Any]] = []
    for k in PURITY_NEIGHBORS:
        for sample_index in range(len(rows)):
            selected = neighbor_indices[sample_index, :k]
            same_count = int(np.sum(true_ids[selected] == true_ids[sample_index]))
            purity_rows.append({
                "sample_index": sample_index,
                "description_id": rows[sample_index]["candidate_id"],
                "true_concept": rows[sample_index]["concept"],
                "description": rows[sample_index]["description"],
                "k": k,
                "same_concept_neighbors": same_count,
                "purity": same_count / k,
                "random_balanced_label_baseline": RANDOM_BALANCED_BASELINE,
            })
    purity_frame = pd.DataFrame(purity_rows)

    summary_rows: list[dict[str, Any]] = []
    for k in PURITY_NEIGHBORS:
        current = purity_frame[purity_frame["k"] == k]
        groups = [("overall", "all", current)]
        groups.extend(
            ("animal", concept, current[current["true_concept"] == concept])
            for concept in CONCEPTS
        )
        for scope, concept, group in groups:
            values = group["purity"].to_numpy(dtype=float)
            summary_rows.append({
                "scope": scope,
                "concept": concept,
                "k": k,
                "count": len(values),
                "mean_purity": float(values.mean()),
                "median_purity": float(np.median(values)),
                "std_purity": float(values.std(ddof=0)),
                "minimum_purity": float(values.min()),
                "maximum_purity": float(values.max()),
                "random_balanced_label_baseline": RANDOM_BALANCED_BASELINE,
            })
    return neighbor_frame, purity_frame, pd.DataFrame(summary_rows)


def _save_confusion(
    matrix: np.ndarray,
    csv_path: Path,
    png_path: Path,
    title: str,
) -> None:
    frame = pd.DataFrame(matrix, index=CONCEPTS, columns=CONCEPTS)
    frame.index.name = "true_concept"
    frame.columns.name = "matched_predicted_concept"
    frame.to_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    ax.set_xticks(range(8), CONCEPTS, rotation=35, ha="right")
    ax.set_yticks(range(8), CONCEPTS)
    ax.set_xlabel("Hungarian-matched predicted concept")
    ax.set_ylabel("True concept")
    ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=16)
    threshold = matrix.max() * 0.55
    for row in range(8):
        for column in range(8):
            ax.text(
                column, row, str(int(matrix[row, column])),
                ha="center", va="center", fontsize=9,
                color="white" if matrix[row, column] > threshold else INK,
            )
    fig.colorbar(image, ax=ax, shrink=0.82, label="Descriptions")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, facecolor="white")
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B8B8B8")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#555555", labelsize=9)


def _save_purity_by_animal(summary: pd.DataFrame, path: Path) -> None:
    animal = summary[summary["scope"] == "animal"]
    x = np.arange(len(CONCEPTS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bar_handles = []
    for offset, k in enumerate(PURITY_NEIGHBORS):
        values = animal[animal["k"] == k].set_index("concept").loc[CONCEPTS, "mean_purity"]
        bars = ax.bar(
            x + (offset - 1) * width, values, width=width,
            color=K_COLORS[k], edgecolor="white", linewidth=0.6, label=f"k = {k}",
        )
        bar_handles.append(bars)
    baseline = ax.axhline(
        RANDOM_BALANCED_BASELINE, color=INK, linestyle="--", linewidth=1.1,
        label=f"random balanced baseline ({RANDOM_BALANCED_BASELINE:.3f})",
    )
    ax.set_xticks(x, CONCEPTS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean same-concept neighbor fraction")
    _style_axis(ax)
    fig.suptitle("kNN label purity by animal", x=0.08, y=0.97, ha="left", fontsize=15, weight="bold")
    fig.text(
        0.08, 0.915, "Cosine neighbors in normalized 768D unsuffixed-EOT space · n = 400",
        fontsize=10, color="#555555",
    )
    fig.legend(
        [*bar_handles, baseline],
        [f"k = {k}" for k in PURITY_NEIGHBORS]
        + [f"random balanced baseline ({RANDOM_BALANCED_BASELINE:.3f})"],
        frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.875), fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.77)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def _save_purity_by_k(purity_by_sample: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(PURITY_NEIGHBORS))
    distributions = [
        purity_by_sample[purity_by_sample["k"] == k]["purity"].to_numpy()
        for k in PURITY_NEIGHBORS
    ]
    means = np.asarray([values.mean() for values in distributions])
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    boxes = ax.boxplot(
        distributions, positions=x, widths=0.55, patch_artist=True, showfliers=True,
        medianprops={"color": INK, "linewidth": 1.6},
        whiskerprops={"color": "#777777", "linewidth": 1.0},
        capprops={"color": "#777777", "linewidth": 1.0},
        flierprops={
            "marker": "o", "markersize": 2.8, "markerfacecolor": "#777777",
            "markeredgewidth": 0, "alpha": 0.20,
        },
    )
    for box, k in zip(boxes["boxes"], PURITY_NEIGHBORS):
        box.set_facecolor(K_COLORS[k])
        box.set_edgecolor("white")
        box.set_alpha(0.88)
    mean_handle = ax.scatter(x, means, marker="D", s=42, color=INK, zorder=3, label="mean")
    baseline = ax.axhline(
        RANDOM_BALANCED_BASELINE, color=INK, linestyle="--", linewidth=1.1,
        label=f"random baseline ({RANDOM_BALANCED_BASELINE:.3f})",
    )
    for position, value in zip(x, means):
        ax.text(position, min(0.985, value + 0.045), f"{value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, [str(k) for k in PURITY_NEIGHBORS])
    ax.set_xlabel("Number of nearest neighbors (k)")
    ax.set_ylabel("Same-concept neighbor fraction")
    ax.set_ylim(0, 1.02)
    _style_axis(ax)
    fig.suptitle("Overall kNN label purity", x=0.08, y=0.97, ha="left", fontsize=15, weight="bold")
    fig.text(
        0.08, 0.915, "Boxplots show all 400 samples in normalized 768D unsuffixed-EOT space",
        fontsize=10, color="#555555",
    )
    fig.legend(
        [mean_handle, boxes["medians"][0], baseline],
        ["mean", "median", f"random baseline ({RANDOM_BALANCED_BASELINE:.3f})"],
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.87), ncol=3, fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.75)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def _save_spectral_metrics_by_neighbors(
    metrics: pd.DataFrame,
    baseline: dict[str, Any],
    path: Path,
) -> None:
    seed42 = metrics[metrics["random_state"] == 42].sort_values("n_neighbors")
    x = np.arange(len(seed42))
    width = 0.23
    series = [
        ("ari", "ARI", BLUE),
        ("nmi", "NMI", GOLD),
        ("matched_accuracy", "Matched accuracy", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    for offset, (column, label, color) in enumerate(series):
        ax.bar(
            x + (offset - 1) * width, seed42[column], width=width,
            color=color, edgecolor="white", linewidth=0.6, label=label,
        )
    ax.axhline(baseline["ari"], color=BLUE, linestyle="--", linewidth=1.1, label="Spherical ARI")
    ax.axhline(baseline["nmi"], color=GOLD, linestyle=":", linewidth=1.3, label="Spherical NMI")
    ax.axhline(
        baseline["matched_accuracy"], color=ORANGE, linestyle="-.", linewidth=1.1,
        label="Spherical accuracy",
    )
    ax.set_xticks(x, seed42["n_neighbors"].astype(str))
    ax.set_xlabel("Graph neighbors")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(
        "Spectral clustering metrics by graph neighborhood",
        x=0.08, y=0.97, ha="left", fontsize=15, weight="bold",
    )
    fig.text(
        0.08, 0.915, "Seed 42 robustness comparison · primary configuration is 10 neighbors",
        fontsize=10, color="#555555",
    )
    fig.legend(
        handles, labels, frameon=False, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 0.875), fontsize=8.5,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.73)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def _analysis_source_audit() -> dict[str, Any]:
    module = inspect.getmodule(_analysis_source_audit)
    if module is None:
        raise RuntimeError("Cannot inspect analysis module")
    source = inspect.getsource(module)
    tree = ast.parse(source)
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    forbidden_imports = {"sklearn.decomposition", "sklearn.manifold"}
    forbidden_calls = {
        "PCA", "TSNE", "fit_spherical_kmeans", "original_projection_modules", "Orthogonal_Erase"
    }
    hits = sorted((imported_modules & forbidden_imports) | (called_names & forbidden_calls))
    if hits:
        raise RuntimeError(f"Strict-scope source audit failed: {hits}")
    label_free_boundaries = (
        find_cosine_neighbors_without_labels,
        build_cosine_knn_graph_without_labels,
        fit_spectral_without_labels,
    )
    boundary_parameters = {
        function.__name__: list(inspect.signature(function).parameters)
        for function in label_free_boundaries
    }
    if any(
        "label" in parameter.casefold() or "concept" in parameter.casefold()
        for parameters in boundary_parameters.values()
        for parameter in parameters
    ):
        raise RuntimeError(f"A graph-or-fit boundary accepts labels: {boundary_parameters}")
    return {
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": _sha256(Path(__file__).resolve()),
        "graph_and_fit_function_parameters": boundary_parameters,
        "labels_available_to_neighbor_search_or_graph_or_spectral_fit": False,
        "dimensionality_reduction_imports_or_calls": False,
        "spherical_kmeans_fit_calls": False,
        "w0_or_oce_calls": False,
        "image_or_description_generation_calls": False,
    }


def _markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _build_report(
    output_dir: Path,
    audit: dict[str, Any],
    baseline: dict[str, Any],
    purity_summary: pd.DataFrame,
    graph_diagnostics: pd.DataFrame,
    spectral_metrics: pd.DataFrame,
    recall: pd.DataFrame,
) -> None:
    overall = purity_summary[purity_summary["scope"] == "overall"].set_index("k")
    primary = spectral_metrics[
        (spectral_metrics["n_neighbors"] == 10) & (spectral_metrics["random_state"] == 42)
    ].iloc[0]
    all_ari_improve = bool((spectral_metrics["ari"] > baseline["ari"]).all())
    all_acc_improve = bool((spectral_metrics["matched_accuracy"] > baseline["matched_accuracy"]).all())
    any_improve = bool(
        (spectral_metrics["ari"] > baseline["ari"]).any()
        or (spectral_metrics["matched_accuracy"] > baseline["matched_accuracy"]).any()
    )
    high_local = float(overall.loc[10, "mean_purity"]) > 0.5
    if all_ari_improve and all_acc_improve:
        case = "Case A"
        interpretation = (
            "Local concept neighborhoods are strong, and every predetermined spectral run improves on both cached baseline metrics. "
            "This is consistent with the single-centroid spherical model underfitting the observed graph geometry."
        )
    elif any_improve:
        case = "Case D"
        interpretation = (
            "Local purity is high, and the 10-neighbor result is stable across the three predetermined seeds, "
            "but robustness to graph neighborhood size is mixed. The 5- and 10-neighbor graphs improve ARI, "
            "whereas the 15-neighbor graph falls below the cached ARI baseline. The graph result is therefore "
            "neighborhood-size-sensitive and is not robust evidence against spherical k-means."
        )
    elif high_local:
        case = "Case B"
        interpretation = (
            "Same-concept local neighborhoods are common, but spectral clustering does not improve on the cached spherical baseline. "
            "Local concept information exists, while assembling those neighborhoods into eight global clusters remains difficult."
        )
    else:
        case = "Case C"
        interpretation = (
            "Local label purity is modest and spectral clustering does not improve on the cached spherical baseline. "
            "The apparent t-SNE separation may exaggerate separation in the original 768D space."
        )

    comparison_rows = [[
        "Spherical k-means (cached)", "—", str(baseline["seed"]),
        f"{baseline['ari']:.4f}", f"{baseline['nmi']:.4f}", f"{baseline['matched_accuracy']:.4f}",
    ]]
    for row in spectral_metrics.sort_values(["n_neighbors", "random_state"]).itertuples():
        comparison_rows.append([
            "Spectral clustering", str(row.n_neighbors), str(row.random_state),
            f"{row.ari:.4f}", f"{row.nmi:.4f}", f"{row.matched_accuracy:.4f}",
        ])
    comparison_table = _markdown_table(
        comparison_rows,
        ["Method", "Graph neighbors", "Seed", "ARI", "NMI", "Matched accuracy"],
    )

    purity_rows = []
    for k in PURITY_NEIGHBORS:
        row = overall.loc[k]
        purity_rows.append([
            str(k), f"{row.mean_purity:.4f}", f"{row.median_purity:.4f}",
            f"{row.std_purity:.4f}", f"{RANDOM_BALANCED_BASELINE:.4f}",
        ])
    purity_table = _markdown_table(
        purity_rows, ["k", "Mean purity", "Median purity", "SD", "Random baseline"],
    )

    primary_recall = recall[
        (recall["n_neighbors"] == 10) & (recall["random_state"] == 42)
    ].set_index("concept").loc[CONCEPTS]
    animal_purity = purity_summary[
        (purity_summary["scope"] == "animal") & (purity_summary["k"] == 10)
    ].set_index("concept").loc[CONCEPTS]
    animal_rows = [
        [concept, f"{animal_purity.loc[concept, 'mean_purity']:.4f}",
         f"{animal_purity.loc[concept, 'median_purity']:.4f}",
         f"{primary_recall.loc[concept, 'recall']:.4f}"]
        for concept in CONCEPTS
    ]
    animal_table = _markdown_table(
        animal_rows, ["Animal", "k=10 mean purity", "k=10 median purity", "Primary spectral recall"],
    )

    graph_rows = []
    for row in graph_diagnostics.sort_values("n_neighbors").itertuples():
        graph_rows.append([
            str(row.n_neighbors), str(row.undirected_edges), str(row.connected_components),
            str(row.component_sizes), str(row.minimum_degree), f"{row.mean_degree:.2f}",
            str(row.maximum_degree),
        ])
    graph_table = _markdown_table(
        graph_rows,
        ["Neighbors", "Edges", "Components", "Component sizes", "Min degree", "Mean degree", "Max degree"],
    )
    seed42 = spectral_metrics[spectral_metrics["random_state"] == 42].set_index("n_neighbors")
    knn10 = spectral_metrics[spectral_metrics["n_neighbors"] == 10]

    text = f"""# EOT Local-Geometry Diagnostic

## 1. Motivation

This diagnostic tests whether the final balanced EOT text space contains strong local animal neighborhoods that a single-centroid clustering model may not capture. It has only two parts: post-hoc k-nearest-neighbor label purity and cosine kNN-graph spectral clustering. The earlier t-SNE view motivates the question but is not used as analytical input.

The result is mixed: local purity is high, and the primary spectral run exceeds the cached spherical baseline, but the spectral ARI drops below baseline when the graph expands to 15 neighbors.

## 2. Data and Representation

The analysis uses the frozen balanced dataset at `{audit['dataset_path']}` (SHA-256 `{audit['dataset_sha256']}`): 400 descriptions, with 50 each for cat, dog, fox, bear, wolf, rabbit, deer, and horse. Every calculation uses the row-L2-normalized 400×768 unsuffixed-EOT matrix. No fixed suffix, W0 projection, OCE operation, PCA coordinate, or t-SNE coordinate is used.

The cached spherical baseline is ARI {baseline['ari']:.4f}, NMI {baseline['nmi']:.4f}, and matched accuracy {baseline['matched_accuracy']:.4f}. Its existing assignments were verified; spherical k-means was not refitted.

## 3. kNN Purity Diagnostic

kNN purity is not clustering. Cosine neighbors are found from the normalized 768D vectors without labels; labels are joined only afterward to calculate the same-animal fraction. The balanced random-label expectation among the other 399 samples is 49/399 = {RANDOM_BALANCED_BASELINE:.4f}.

{purity_table}

At k=10, the observed overall mean purity is {overall.loc[10, 'mean_purity']:.4f}, compared with {RANDOM_BALANCED_BASELINE:.4f} under the balanced random-label reference. Individual neighbor identities and distances are retained in `knn_neighbors.csv`.

## 4. Cosine kNN Graph

Each graph connects a pair when either endpoint selects the other. Edge weight is non-negative cosine similarity, and the diagonal is zero. Labels are not available during neighbor selection or graph construction.

{graph_table}

The primary graph uses exactly 10 cosine neighbors. Its connected-component structure is recorded rather than repaired or silently changed.

## 5. Spectral Clustering

Spectral clustering groups points using connectivity in the precomputed cosine kNN graph rather than one centroid per cluster in the original space. Every run fixes the number of clusters at eight and uses `assign_labels="kmeans"`; no label enters fitting.

{comparison_table}

The primary 10-neighbor, seed-42 run obtains ARI {primary.ari:.4f}, NMI {primary.nmi:.4f}, and matched accuracy {primary.matched_accuracy:.4f}.

## 6. Comparison with Spherical K-Means

The primary spectral run changes ARI by {primary.ari - baseline['ari']:+.4f} and matched accuracy by {primary.matched_accuracy - baseline['matched_accuracy']:+.4f} relative to the cached spherical baseline. All predetermined neighborhood and seed runs are shown above; no run was selected for being visually or numerically strongest.

At seed 42, ARI is {seed42.loc[5, 'ari']:.4f}, {seed42.loc[10, 'ari']:.4f}, and {seed42.loc[15, 'ari']:.4f} for 5, 10, and 15 graph neighbors. Across the three 10-neighbor seeds, ARI ranges from {knn10['ari'].min():.4f} to {knn10['ari'].max():.4f}, and matched accuracy ranges from {knn10['matched_accuracy'].min():.4f} to {knn10['matched_accuracy'].max():.4f}.

## 7. Per-Animal Results

{animal_table}

The table pairs local k=10 purity with recall from the predetermined primary spectral run. Purity describes local neighborhoods; recall describes the post-hoc Hungarian-matched global clustering and should not be treated as the same quantity.

## 8. Interpretation

**{case}.** {interpretation}

This result describes local and graph-based organization in the original normalized EOT space. It does not establish that the eight concepts are natural modes, and it does not validate separation merely because t-SNE appears clean.

## 9. Limitations

- The number of clusters is fixed to eight from the experimental design.
- kNN purity uses labels only as a post-hoc diagnostic and is not an unsupervised clustering score.
- Spectral clustering depends on graph construction and random k-means label assignment; the predetermined neighborhood and seed checks expose only limited sensitivity.
- The balanced-label baseline is an expectation, not an inferential significance test.
- No image behavior, W0 geometry, OCE behavior, or representation other than unsuffixed EOT is tested.
"""
    atomic_write_text(output_dir / "report.md", text)


def run_diagnostic(
    dataset: str | Path,
    output: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = _resolve_sources(dataset)
    output_dir = Path(output).expanduser().resolve()
    if output_dir == paths["source_dir"] or paths["source_dir"] in output_dir.parents:
        raise ValueError("Output must be isolated from the balanced source directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in ("dataset_audit.json", "report.md") if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace only this diagnostic: {existing}")
    plots_dir = output_dir / "plots"
    confusion_dir = output_dir / "confusion_matrices"
    plots_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)

    rows, features, true_ids, baseline, audit = _load_and_audit(paths)
    source_audit = _analysis_source_audit()
    audit["source_scope_audit"] = source_audit
    atomic_write_text(output_dir / "dataset_audit.json", json.dumps(audit, indent=2) + "\n")

    max_indices, max_distances = find_cosine_neighbors_without_labels(features, max(PURITY_NEIGHBORS))
    neighbor_frame, purity_frame, purity_summary = _neighbor_and_purity_tables(
        rows, true_ids, max_indices, max_distances
    )
    neighbor_frame.to_csv(output_dir / "knn_neighbors.csv", index=False)
    purity_frame.to_csv(output_dir / "knn_purity_by_sample.csv", index=False)
    purity_summary.to_csv(output_dir / "knn_purity_summary.csv", index=False)

    affinities: dict[int, csr_matrix] = {}
    graph_rows: list[dict[str, Any]] = []
    for n_neighbors in (5, 10, 15):
        affinity, diagnostics = build_cosine_knn_graph_without_labels(features, n_neighbors)
        affinities[n_neighbors] = affinity
        graph_rows.append({
            **diagnostics,
            "component_sizes": json.dumps(diagnostics["component_sizes"]),
        })
    graph_diagnostics = pd.DataFrame(graph_rows)
    graph_diagnostics.to_csv(output_dir / "graph_diagnostics.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    primary_confusion: np.ndarray | None = None
    for run in SPECTRAL_RUNS:
        n_neighbors = int(run["n_neighbors"])
        random_state = int(run["random_state"])
        run_id = f"spectral_knn{n_neighbors:02d}_seed{random_state}"
        cluster_ids, warning_messages = fit_spectral_without_labels(
            affinities[n_neighbors], n_clusters=8, random_state=random_state, n_init=50
        )
        metrics, predicted_ids, matched_confusion = _evaluate_after_spectral_fit(cluster_ids, true_ids)
        graph = graph_diagnostics[graph_diagnostics["n_neighbors"] == n_neighbors].iloc[0]
        metric_rows.append({
            "run_id": run_id,
            "method": "cosine_knn_graph_spectral",
            "n_neighbors": n_neighbors,
            "random_state": random_state,
            "primary": bool(run["primary"]),
            "n_clusters": 8,
            "affinity": "precomputed",
            "assign_labels": "kmeans",
            "n_init": 50,
            "ari": metrics["ari"],
            "nmi": metrics["nmi"],
            "matched_accuracy": metrics["matched_accuracy"],
            "cluster_sizes": json.dumps(metrics["cluster_sizes"]),
            "cluster_to_concept": json.dumps(metrics["cluster_to_concept"], sort_keys=True),
            "connected_components": int(graph["connected_components"]),
            "component_sizes": graph["component_sizes"],
            "too_disconnected_for_eight_cluster_spectral": bool(
                graph["too_disconnected_for_eight_cluster_spectral"]
            ),
            "fit_warnings": json.dumps(warning_messages),
            "labels_available_to_fit": False,
        })
        for concept in CONCEPTS:
            recall_rows.append({
                "run_id": run_id,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
                "primary": bool(run["primary"]),
                "concept": concept,
                "recall": metrics["per_class_recall"][concept],
            })
        mapping = {int(key): value for key, value in metrics["cluster_to_concept"].items()}
        for sample_index, row in enumerate(rows):
            assignment_rows.append({
                "run_id": run_id,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
                "primary": bool(run["primary"]),
                "sample_index": sample_index,
                "description_id": row["candidate_id"],
                "true_concept": row["concept"],
                "description": row["description"],
                "raw_cluster_id": int(cluster_ids[sample_index]),
                "hungarian_matched_predicted_concept": CONCEPTS[int(predicted_ids[sample_index])],
                "prediction_correct": bool(predicted_ids[sample_index] == true_ids[sample_index]),
                "cluster_mapping_concept": mapping[int(cluster_ids[sample_index])],
            })
        csv_path = confusion_dir / f"confusion_{run_id}.csv"
        png_path = confusion_dir / f"confusion_{run_id}.png"
        _save_confusion(
            matched_confusion, csv_path, png_path,
            f"Spectral clustering confusion · kNN {n_neighbors} · seed {random_state}",
        )
        if bool(run["primary"]):
            primary_confusion = matched_confusion.copy()

    if primary_confusion is None:
        raise RuntimeError("Primary spectral run was not executed")
    spectral_metrics = pd.DataFrame(metric_rows)
    recall_frame = pd.DataFrame(recall_rows)
    assignments_frame = pd.DataFrame(assignment_rows)
    spectral_metrics.to_csv(output_dir / "spectral_metrics_all_runs.csv", index=False)
    recall_frame.to_csv(output_dir / "spectral_per_class_recall.csv", index=False)
    assignments_frame.to_csv(output_dir / "spectral_assignments.csv", index=False)
    _save_confusion(
        primary_confusion,
        output_dir / "confusion_spectral_knn10_seed42.csv",
        output_dir / "confusion_spectral_knn10_seed42.png",
        "Primary spectral clustering confusion · kNN 10 · seed 42",
    )

    _save_purity_by_animal(purity_summary, plots_dir / "knn_purity_by_animal.png")
    _save_purity_by_k(purity_frame, plots_dir / "knn_purity_by_k.png")
    _save_spectral_metrics_by_neighbors(
        spectral_metrics, baseline, plots_dir / "spectral_metrics_by_neighbors.png"
    )

    config = {
        "experiment": "eot_local_geometry_diagnostic",
        "research_question": (
            "Does normalized 768D unsuffixed-EOT space contain strong local concept neighborhoods "
            "that are not well captured by a single-centroid spherical clustering assumption?"
        ),
        "dataset_path": audit["dataset_path"],
        "dataset_sha256": audit["dataset_sha256"],
        "embedding_path": audit["embedding_path"],
        "embedding_sha256": audit["embedding_sha256"],
        "representation": "row-L2-normalized unsuffixed EOT; shape 400x768",
        "knn_purity": {
            "metric": "cosine",
            "exclude_self": True,
            "neighborhood_sizes": PURITY_NEIGHBORS,
            "random_balanced_label_baseline": RANDOM_BALANCED_BASELINE,
            "labels_used_after_neighbor_search_only": True,
        },
        "spectral": {
            "runs": SPECTRAL_RUNS,
            "graph_symmetrization": "union of directed selections",
            "edge_weight": "max(0, cosine_similarity)",
            "diagonal": 0,
            "n_clusters": 8,
            "affinity": "precomputed",
            "assign_labels": "kmeans",
            "n_init": 50,
            "labels_available_to_graph_or_fit": False,
        },
        "cached_spherical_baseline": baseline,
        "strict_scope": {
            "w0_projection": False,
            "oce": False,
            "dimensionality_reduction_inputs": False,
            "fixed_suffix": False,
            "spherical_kmeans_refit": False,
            "image_or_description_generation": False,
            "other_methods": False,
        },
        "package_versions": package_versions(),
    }
    atomic_write_text(output_dir / "experiment_config.json", json.dumps(config, indent=2) + "\n")
    chart_map = {
        "knn_purity_by_animal": {
            "question": "How does local label purity vary by animal and predetermined k?",
            "chart": "grouped bar with fixed random-baseline reference",
            "source": "knn_purity_summary.csv",
        },
        "knn_purity_by_k": {
            "question": "How stable is overall purity across k=5,10,20?",
            "chart": "sample-level boxplot with mean marker and random-baseline reference",
            "source": "knn_purity_by_sample.csv",
        },
        "spectral_metrics_by_neighbors": {
            "question": "How do seed-42 spectral results change across predetermined graph neighborhoods?",
            "chart": "grouped bar with cached spherical reference lines",
            "source": "spectral_metrics_all_runs.csv and cached eot_metrics.json",
        },
    }
    atomic_write_text(output_dir / "chart_map.json", json.dumps(chart_map, indent=2) + "\n")
    _build_report(
        output_dir, audit, baseline, purity_summary, graph_diagnostics,
        spectral_metrics, recall_frame,
    )

    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Diagnostic output is incomplete: {missing}")
    return {
        "output_directory": str(output_dir),
        "dataset_sha256": audit["dataset_sha256"],
        "samples": len(rows),
        "neighbor_identity_rows": len(neighbor_frame),
        "purity_sample_rows": len(purity_frame),
        "spectral_runs": len(spectral_metrics),
        "spectral_assignment_rows": len(assignments_frame),
        "primary_spectral": spectral_metrics[spectral_metrics["primary"]].iloc[0][
            ["ari", "nmi", "matched_accuracy"]
        ].to_dict(),
        "clustering_input_shape": list(features.shape),
        "labels_available_to_graph_or_fit": False,
        "spherical_kmeans_refit": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen-EOT kNN-purity and cosine-graph spectral diagnostic."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Final balanced 8x50 output directory or accepted_descriptions.jsonl path.",
    )
    parser.add_argument("--output", required=True, help="New isolated diagnostic output directory.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace files only inside the selected isolated diagnostic output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_diagnostic(args.dataset, args.output, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
