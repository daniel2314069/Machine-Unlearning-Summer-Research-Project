from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import l2_normalize, set_reproducible_seed, write_csv


PALETTE = ["#2458A6", "#D28E00", "#D65F30", "#708238", "#C04F83", "#5B6770", "#6F4E7C", "#2A7F9E", "#A65D40", "#526D3F"]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _hungarian_mapping(true_ids: np.ndarray, cluster_ids: np.ndarray, k: int):
    raw = confusion_matrix(true_ids, cluster_ids, labels=np.arange(k))
    rows, cols = linear_sum_assignment(-raw)
    cluster_to_true = {int(cluster): int(true) for true, cluster in zip(rows, cols)}
    predicted = np.array([cluster_to_true[int(cluster)] for cluster in cluster_ids])
    return cluster_to_true, predicted


def _true_centroids(features: np.ndarray, true_ids: np.ndarray, k: int) -> np.ndarray:
    centroids = np.stack([features[true_ids == index].mean(axis=0) for index in range(k)])
    return l2_normalize(centroids)


def _prototype_distribution_metrics(
    features: np.ndarray,
    true_ids: np.ndarray,
    prototypes: np.ndarray,
    config: dict[str, Any] | None,
) -> list[dict[str, float]]:
    """Place held-out name prototypes on the within-concept distance scale.

    The bootstrap uses disjoint reference/evaluation subsets.  This prevents an
    evaluated description from pulling its own comparison centroid inward.
    """
    settings = config or {}
    n_bootstrap = int(settings.get("n_bootstrap", 1000))
    reference_fraction = float(settings.get("reference_fraction", 0.8))
    seed = int(settings.get("seed", 2027))
    rng = np.random.default_rng(seed)
    output = []
    for concept_id in range(len(prototypes)):
        concept_features = features[true_ids == concept_id]
        n = len(concept_features)
        if n < 3:
            raise ValueError("Prototype percentile analysis requires at least three descriptions per concept")
        centroid = l2_normalize(concept_features.mean(axis=0, keepdims=True))[0]
        prototype_distance = float(1 - prototypes[concept_id] @ centroid)
        description_distances = 1 - concept_features @ centroid

        leave_one_out_centroids = l2_normalize(
            (concept_features.sum(axis=0, keepdims=True) - concept_features) / (n - 1)
        )
        leave_one_out_distances = 1 - np.sum(concept_features * leave_one_out_centroids, axis=1)
        median = float(np.median(leave_one_out_distances))
        mad = float(np.median(np.abs(leave_one_out_distances - median)))
        robust_scale = max(1.4826 * mad, 1e-12)

        reference_n = min(n - 1, max(2, int(round(reference_fraction * n))))
        bootstrap_percentiles = []
        bootstrap_distances = []
        for _ in range(n_bootstrap):
            order = rng.permutation(n)
            reference = concept_features[order[:reference_n]]
            evaluation = concept_features[order[reference_n:]]
            reference_centroid = l2_normalize(reference.mean(axis=0, keepdims=True))[0]
            current_prototype_distance = float(1 - prototypes[concept_id] @ reference_centroid)
            evaluation_distances = 1 - evaluation @ reference_centroid
            bootstrap_percentiles.append(100.0 * float(np.mean(evaluation_distances <= current_prototype_distance)))
            bootstrap_distances.append(current_prototype_distance)

        output.append({
            "description_distance_percentile": 100.0 * float(np.mean(description_distances <= prototype_distance)),
            "leave_one_out_distance_percentile": 100.0 * float(np.mean(leave_one_out_distances <= prototype_distance)),
            "description_distance_median": median,
            "description_distance_p95": float(np.quantile(leave_one_out_distances, 0.95)),
            "prototype_distance_robust_z": (prototype_distance - median) / robust_scale,
            "bootstrap_percentile_median": float(np.median(bootstrap_percentiles)),
            "bootstrap_percentile_ci_low": float(np.quantile(bootstrap_percentiles, 0.025)),
            "bootstrap_percentile_ci_high": float(np.quantile(bootstrap_percentiles, 0.975)),
            "bootstrap_prototype_distance_median": float(np.median(bootstrap_distances)),
            "bootstrap_prototype_distance_ci_low": float(np.quantile(bootstrap_distances, 0.025)),
            "bootstrap_prototype_distance_ci_high": float(np.quantile(bootstrap_distances, 0.975)),
            "bootstrap_iterations": n_bootstrap,
            "bootstrap_reference_fraction": reference_fraction,
        })
    return output


def _fit_runs(
    features: np.ndarray,
    true_ids: np.ndarray,
    facet_ids: np.ndarray,
    config: dict[str, Any],
    representation: str,
    layer_name: str = "",
    prototype_features: np.ndarray | None = None,
):
    clustering = config["clustering"]
    k = int(clustering["k"])
    true_centroids = _true_centroids(features, true_ids, k)
    mean_proto_own = np.nan
    if prototype_features is not None:
        mean_proto_own = float(np.mean(1.0 - np.sum(prototype_features * true_centroids, axis=1)))
    metrics, fitted = [], {}
    for seed in clustering["random_seeds"][: int(clustering["n_runs"])]:
        model = KMeans(k, n_init=int(clustering["n_init"]), random_state=int(seed), algorithm="lloyd")
        cluster_ids = model.fit_predict(features)
        mapping, predicted = _hungarian_mapping(true_ids, cluster_ids, k)
        row = {
            "representation": representation,
            "layer_name": layer_name,
            "seed": int(seed),
            "ari_concept": adjusted_rand_score(true_ids, cluster_ids),
            "nmi_concept": normalized_mutual_info_score(true_ids, cluster_ids),
            "hungarian_accuracy": accuracy_score(true_ids, predicted),
            "silhouette": silhouette_score(features, cluster_ids, metric="euclidean"),
            "ari_facet": adjusted_rand_score(facet_ids, cluster_ids),
            "nmi_facet": normalized_mutual_info_score(facet_ids, cluster_ids),
            "inertia": float(model.inertia_),
            "cluster_sizes": json.dumps(np.bincount(cluster_ids, minlength=k).tolist()),
            "mean_prototype_own_centroid_distance": mean_proto_own,
        }
        metrics.append(row)
        fitted[int(seed)] = (model, cluster_ids, mapping, predicted, true_centroids)
    return metrics, fitted


def _canonical_outputs(
    features: np.ndarray,
    prototypes: np.ndarray | None,
    candidate_ids: list[str],
    descriptions: list[str],
    concept_labels: list[str],
    facet_labels: list[str],
    concept_names: list[str],
    fitted,
    canonical_seed: int,
    representation: str,
    layer_name: str = "",
    prototype_analysis: dict[str, Any] | None = None,
):
    model, cluster_ids, mapping, predicted, true_centroids = fitted[canonical_seed]
    normalized_kmeans_centers = l2_normalize(model.cluster_centers_)
    true_ids = np.array([concept_names.index(label) for label in concept_labels])
    assignment_rows = []
    for index, candidate_id in enumerate(candidate_ids):
        assigned_cluster = int(cluster_ids[index])
        true_id = int(true_ids[index])
        own_distance = float(1 - features[index] @ true_centroids[true_id])
        other_distances = 1 - features[index] @ true_centroids.T
        other_distances[true_id] = np.inf
        nearest_wrong_id = int(other_distances.argmin())
        wrong_distance = float(other_distances[nearest_wrong_id])
        assignment_rows.append({
            "representation": representation,
            "layer_name": layer_name,
            "candidate_id": candidate_id,
            "concept": concept_labels[index],
            "facet_id": facet_labels[index],
            "description": descriptions[index],
            "cluster": assigned_cluster,
            "matched_cluster_concept": concept_names[mapping[assigned_cluster]],
            "correct_after_hungarian": bool(predicted[index] == true_id),
            "distance_to_assigned_centroid": float(1 - features[index] @ normalized_kmeans_centers[assigned_cluster]),
            "distance_to_true_concept_centroid": own_distance,
            "nearest_incorrect_concept_centroid": concept_names[nearest_wrong_id],
            "nearest_incorrect_centroid_distance": wrong_distance,
            "classification_margin": wrong_distance - own_distance,
        })
    concept_confusion = confusion_matrix(true_ids, predicted, labels=np.arange(len(concept_names)))

    prototype_rows = []
    prototype_distance_matrix = None
    if prototypes is not None:
        distribution_metrics = _prototype_distribution_metrics(
            features, true_ids, prototypes, prototype_analysis
        )
        prototype_distance_matrix = 1 - prototypes @ true_centroids.T
        kmeans_distances = 1 - prototypes @ normalized_kmeans_centers.T
        for concept_id, concept in enumerate(concept_names):
            nearest_cluster = int(kmeans_distances[concept_id].argmin())
            distances = prototype_distance_matrix[concept_id]
            order = np.argsort(distances)
            nearest_wrong = min(float(distances[j]) for j in range(len(concept_names)) if j != concept_id)
            own = float(distances[concept_id])
            prototype_rows.append({
                "representation": representation,
                "layer_name": layer_name,
                "concept": concept,
                "nearest_cluster": nearest_cluster,
                "nearest_cluster_matched_concept": concept_names[mapping[nearest_cluster]],
                "own_description_centroid_distance": own,
                "all_concept_centroid_distances": json.dumps({name: float(distances[j]) for j, name in enumerate(concept_names)}),
                "correct_centroid_rank": int(np.where(order == concept_id)[0][0]) + 1,
                "nearest_incorrect_centroid_distance": nearest_wrong,
                "prototype_margin": nearest_wrong - own,
                **distribution_metrics[concept_id],
            })
    return assignment_rows, prototype_rows, concept_confusion, true_centroids, prototype_distance_matrix


def _save_heatmap(matrix, xlabels, ylabels, title, path, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(max(6, len(xlabels) * 0.7), max(5, len(ylabels) * 0.6)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = format(value, fmt) if fmt else str(value)
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if value > np.nanmax(matrix) * 0.55 else "#222222")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_pca(features, labels, names, title, path):
    points = PCA(n_components=2, random_state=0).fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 6))
    for index, name in enumerate(names):
        mask = labels == index
        ax.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.65, label=name, color=PALETTE[index % len(PALETTE)])
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_description_length_control(output_dir: str | Path) -> None:
    """Quantify how much concept/facet information sentence length alone carries."""
    output_dir = Path(output_dir)
    lengths = pd.read_csv(output_dir / "description_lengths.csv")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    scores = cross_val_score(estimator, lengths[["word_count"]], lengths["concept"], cv=cv)
    write_csv(output_dir / "description_length_control.csv", [{
        "n_descriptions": len(lengths),
        "chance_accuracy": 1.0 / lengths["concept"].nunique(),
        "exact_length_ari_concept": adjusted_rand_score(lengths["concept"], lengths["word_count"]),
        "exact_length_nmi_concept": normalized_mutual_info_score(lengths["concept"], lengths["word_count"]),
        "exact_length_ari_facet": adjusted_rand_score(lengths["facet_id"], lengths["word_count"]),
        "exact_length_nmi_facet": normalized_mutual_info_score(lengths["facet_id"], lengths["word_count"]),
        "length_only_logistic_cv_accuracy_mean": float(scores.mean()),
        "length_only_logistic_cv_accuracy_std": float(scores.std(ddof=1)),
        "cv_folds": 5,
        "cv_seed": 0,
    }])


def run_clustering(config: dict[str, Any], output_dir: str | Path) -> None:
    set_reproducible_seed(0)
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    confusion_dir = output_dir / "confusion_matrices"
    plots_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    raw = torch.load(output_dir / "raw_text_embeddings.pt", map_location="cpu", weights_only=False)
    layers = torch.load(output_dir / "layer_embeddings.pt", map_location="cpu", weights_only=False)
    concept_names = list(raw["concept_names"])
    facet_names = [item["id"] for item in config["facets"]]
    concept_ids = np.array([concept_names.index(label) for label in raw["concept_labels"]])
    facet_ids = np.array([facet_names.index(label) for label in raw["facet_labels"]])
    canonical_seed = int(config["clustering"]["canonical_seed"])

    raw_metrics, facet_metrics, assignment_rows, prototype_rows = [], [], [], []
    group_metrics, error_rows = [], []
    primary = config["readout"]["primary_suffix_name"]
    raw_conditions = {f"fixed:{name}": tensor for name, tensor in raw["fixed_readout"].items()}
    raw_conditions["natural_last_token"] = raw["natural_last_token"]
    if raw.get("shuffled_fixed_readout") is not None:
        raw_conditions[f"shuffled_words:fixed:{primary}"] = raw["shuffled_fixed_readout"]

    primary_centroids = None
    primary_proto_distances = None
    for representation, tensor in raw_conditions.items():
        features = l2_normalize(tensor.numpy())
        if representation.startswith("fixed:"):
            prototypes = raw["prototypes"].get(representation.split(":", 1)[1])
        elif representation == f"shuffled_words:fixed:{primary}":
            prototypes = raw["prototypes"].get(primary)
        else:
            prototypes = None
        prototype_array = l2_normalize(prototypes.numpy()) if prototypes is not None else None
        metrics, fitted = _fit_runs(features, concept_ids, facet_ids, config, representation, prototype_features=prototype_array)
        raw_metrics.extend(metrics)
        facet_metrics.extend({key: row[key] for key in ["representation", "layer_name", "seed", "ari_concept", "nmi_concept", "ari_facet", "nmi_facet"]} for row in metrics)
        assignments, prototypes_out, confusion, true_centroids, proto_distances = _canonical_outputs(
            features, prototype_array, raw["candidate_ids"], raw["descriptions"], raw["concept_labels"],
            raw["facet_labels"], concept_names, fitted, canonical_seed, representation,
            prototype_analysis=config.get("prototype_analysis"),
        )
        assignment_rows.extend(assignments)
        prototype_rows.extend(prototypes_out)
        pd.DataFrame(confusion, index=concept_names, columns=concept_names).to_csv(confusion_dir / f"{_slug(representation)}.csv")
        _save_heatmap(confusion, concept_names, concept_names, f"Concept confusion: {representation}", plots_dir / f"confusion_{_slug(representation)}.png", fmt="d")
        if representation == f"fixed:{primary}":
            primary_centroids = true_centroids
            primary_proto_distances = proto_distances
            centroid_distances = 1 - true_centroids @ true_centroids.T
            _save_heatmap(centroid_distances, concept_names, concept_names, "Concept-centroid cosine distance", plots_dir / "concept_centroid_distances.png")
            if proto_distances is not None:
                _save_heatmap(proto_distances, concept_names, concept_names, "Prototype-to-concept-centroid cosine distance", plots_dir / "prototype_to_centroid_distances.png")
            _save_pca(features, concept_ids, concept_names, "Fixed-readout PCA by concept", plots_dir / "pca_by_concept.png")
            _save_pca(features, facet_ids, facet_names, "Fixed-readout PCA by facet", plots_dir / "pca_by_facet.png")

        frame = pd.DataFrame(assignments)
        for group_type, group_column in [("concept", "concept"), ("facet", "facet_id")]:
            for group, subset in frame.groupby(group_column):
                group_metrics.append({
                    "representation": representation,
                    "group_type": group_type,
                    "group": group,
                    "count": len(subset),
                    "accuracy": float(subset["correct_after_hungarian"].mean()),
                })
        if representation == f"fixed:{primary}":
            for (concept, facet), subset in frame.groupby(["concept", "facet_id"]):
                errors = int((~subset["correct_after_hungarian"].astype(bool)).sum())
                error_rows.append({"concept": concept, "facet_id": facet, "count": len(subset), "errors": errors, "accuracy": 1 - errors / len(subset)})

    layer_metrics, layer_assignment_rows = [], []
    for layer_index, layer_name in enumerate(layers["layer_names"]):
        representation = f"layer:{layers['projection']}:{layer_index:02d}"
        features = l2_normalize(layers["description_embeddings"][layer_name].numpy())
        prototypes = l2_normalize(layers["prototype_embeddings"][layer_name].numpy())
        metrics, fitted = _fit_runs(features, concept_ids, facet_ids, config, representation, layer_name, prototypes)
        for row in metrics:
            row["layer_index"] = layer_index
        layer_metrics.extend(metrics)
        assignments, prototypes_out, confusion, _, _ = _canonical_outputs(
            features, prototypes, raw["candidate_ids"], raw["descriptions"], raw["concept_labels"],
            raw["facet_labels"], concept_names, fitted, canonical_seed, representation, layer_name,
            prototype_analysis=config.get("prototype_analysis"),
        )
        layer_assignment_rows.extend(assignments)
        prototype_rows.extend(prototypes_out)
        pd.DataFrame(confusion, index=concept_names, columns=concept_names).to_csv(confusion_dir / f"layer_{layer_index:02d}_{_slug(layer_name)}.csv")

    write_csv(output_dir / "clustering_metrics.csv", raw_metrics)
    write_csv(output_dir / "facet_confounding_metrics.csv", facet_metrics)
    write_csv(output_dir / "layer_metrics.csv", layer_metrics)
    write_csv(output_dir / "clustering_assignments.csv", assignment_rows + layer_assignment_rows)
    write_csv(output_dir / "prototype_metrics.csv", prototype_rows)
    write_csv(output_dir / "per_group_metrics.csv", group_metrics)
    write_csv(output_dir / "concept_by_facet_error_table.csv", error_rows)

    lengths = pd.DataFrame({
        "concept": raw["concept_labels"],
        "facet_id": raw["facet_labels"],
        "word_count": [len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text)) for text in raw["descriptions"]],
    })
    lengths.to_csv(output_dir / "description_lengths.csv", index=False)
    save_description_length_control(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [lengths.loc[lengths["concept"] == concept, "word_count"] for concept in concept_names]
    ax.boxplot(data, tick_labels=concept_names, showfliers=True)
    ax.set_title("Description length by concept")
    ax.set_ylabel("Words per description")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plots_dir / "description_length_by_concept.png", dpi=180)
    plt.close(fig)

    layer_frame = pd.DataFrame(layer_metrics)
    summaries = layer_frame.groupby("layer_index").agg({
        "ari_concept": ["mean", "std"],
        "nmi_concept": ["mean", "std"],
        "hungarian_accuracy": ["mean", "std"],
        "silhouette": ["mean", "std"],
        "ari_facet": ["mean", "std"],
        "nmi_facet": ["mean", "std"],
    })
    summaries.to_csv(output_dir / "layer_metrics_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, label, color in [
        ("ari_concept", "ARI", PALETTE[0]),
        ("nmi_concept", "NMI", PALETTE[1]),
        ("hungarian_accuracy", "Matched accuracy", PALETTE[2]),
        ("silhouette", "Silhouette", PALETTE[3]),
    ]:
        mean = summaries[(metric, "mean")]
        std = summaries[(metric, "std")].fillna(0)
        x = mean.index.to_numpy()
        ax.plot(x, mean, marker="o", label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
    ax.set_title(f"Layer-wise clustering metrics ({layers['projection']})")
    ax.set_xlabel("Original projection layer index")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(plots_dir / "layer_clustering_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, label, style in [("ari_concept", "Concept ARI", "-"), ("ari_facet", "Facet ARI", "--"), ("nmi_concept", "Concept NMI", "-"), ("nmi_facet", "Facet NMI", "--")]:
        mean = summaries[(metric, "mean")]
        ax.plot(mean.index, mean, linestyle=style, marker="o", label=label)
    ax.set_title(f"Concept-versus-facet clustering ({layers['projection']})")
    ax.set_xlabel("Original projection layer index")
    ax.set_ylabel("Agreement score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "layer_concept_vs_facet.png", dpi=180)
    plt.close(fig)

    if bool(config.get("lexical_baselines", {}).get("enabled", False)):
        from .lexical_baselines import run_lexical_baselines

        run_lexical_baselines(config, output_dir)
