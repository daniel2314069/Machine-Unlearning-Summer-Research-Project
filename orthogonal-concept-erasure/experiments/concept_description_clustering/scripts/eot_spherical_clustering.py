#!/usr/bin/env python
"""Cluster unsuffixed SD 1.4 CLIP EOT states with true spherical k-means.

This is intentionally isolated from the repository's fixed-readout, OCE/UCE,
projection, generation, and probing experiments.  The clustering boundary
accepts feature matrices only; labels are constructed only after both fits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)

# Make direct execution (``python scripts/eot_spherical_clustering.py``) use
# the adjacent repository package just like ``python -m scripts...`` does.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from concept_clustering.config import config_concept_names, load_config
from concept_clustering.modeling import load_original_pipeline
from concept_clustering.utils import (
    atomic_write_text,
    package_versions,
    read_jsonl,
    set_reproducible_seed,
    write_csv,
)


PALETTE = {
    "cat": "#2458A6",
    "dog": "#D28E00",
    "fox": "#D65F30",
    "bear": "#708238",
}
REQUIRED_OUTPUTS = [
    "eot_embeddings.npy",
    "eot_raw_normalized.npy",
    "eot_centered_normalized.npy",
    "metadata.csv",
    "metrics.json",
    "metrics_summary.txt",
    "confusion_raw.csv",
    "confusion_centered.csv",
    "confusion_raw.png",
    "confusion_centered.png",
    "pca_raw_true_labels.png",
    "pca_raw_predicted_clusters.png",
    "pca_centered_true_labels.png",
    "pca_centered_predicted_clusters.png",
    "misclustered_raw.csv",
    "misclustered_centered.csv",
    "fox_errors_raw.txt",
    "fox_errors_centered.txt",
]


@dataclass(frozen=True)
class SphericalKMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    best_initialization: int
    all_objectives: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("accepted_descriptions", "descriptions", "records", "rows", "items", "data"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        if rows is None:
            raise ValueError(
                f"{path} is JSON but not a description dataset. Expected a row list or a supported row-list key; "
                "the syntax_independent_4x50.json file in this repository is a config, not the 200-row corpus."
            )
    else:
        raise ValueError(f"Unsupported JSON dataset root in {path}: {type(payload).__name__}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Every dataset row must be a JSON object")
    return list(rows)


def load_description_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load the inspected repository row schema without guessing field aliases."""
    path = Path(path)
    if path.suffix.casefold() == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix.casefold() == ".json":
        rows = _load_json_dataset(path)
    else:
        raise ValueError(f"Dataset must be .jsonl or .json, got {path}")
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    required = {"description", "concept"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Dataset row {index} is missing fields {sorted(missing)}")
        if not isinstance(row["description"], str) or not row["description"].strip():
            raise ValueError(f"Dataset row {index} has an empty/non-string description")
        if not isinstance(row["concept"], str) or not row["concept"].strip():
            raise ValueError(f"Dataset row {index} has an empty/non-string concept")
    descriptions = [row["description"] for row in rows]
    duplicates = [text for text, count in Counter(descriptions).items() if count > 1]
    if duplicates:
        raise ValueError(f"Dataset contains {len(duplicates)} duplicated descriptions")
    return rows


def validate_dataset(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[str], dict[str, int]]:
    concept_names = config_concept_names(config)
    counts = Counter(str(row["concept"]) for row in rows)
    if set(counts) != set(concept_names):
        raise ValueError(
            f"Dataset concepts {sorted(counts)} do not exactly match config concepts {sorted(concept_names)}"
        )
    expected_per_class = (
        len(config["facets"])
        * int(config["candidate_validation"]["accepted_per_concept_facet"])
    )
    wrong = {name: counts[name] for name in concept_names if counts[name] != expected_per_class}
    if wrong:
        raise ValueError(
            f"Expected {expected_per_class} descriptions per class from the config, observed {dict(counts)}"
        )
    return concept_names, {name: int(counts[name]) for name in concept_names}


def _validate_eot_batch(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if last_hidden_state.ndim != 3:
        raise RuntimeError(f"Expected [batch, sequence, hidden], got {tuple(last_hidden_state.shape)}")
    effective_lengths = attention_mask.sum(dim=1).to(torch.long)
    eot_indices = effective_lengths - 1
    sequence_length = last_hidden_state.shape[1]
    if torch.any(eot_indices < 0) or torch.any(eot_indices >= sequence_length):
        raise RuntimeError(f"EOT index outside [0, {sequence_length - 1}]")
    row_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    eot_ids = input_ids[row_indices, eot_indices]
    if torch.any(eot_ids != int(eos_token_id)):
        raise RuntimeError(
            f"Attention-mask EOT selection did not select eos_token_id={eos_token_id}: "
            f"{eot_ids.detach().cpu().tolist()}"
        )
    vectors = last_hidden_state[row_indices, eot_indices, :]
    if not torch.isfinite(vectors).all():
        raise RuntimeError("EOT embeddings contain NaN or Inf")
    return vectors, eot_indices


def _untruncated_token_length(tokenizer, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def extract_eot_embeddings(
    tokenizer,
    text_encoder,
    descriptions: list[str],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Extract final-layer EOT states from the exact, unsuffixed prompts."""
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    vectors: list[torch.Tensor] = []
    audits: list[dict[str, Any]] = []
    maximum_length = int(tokenizer.model_max_length)
    eos_token_id = int(tokenizer.eos_token_id)
    with torch.no_grad():
        for start in range(0, len(descriptions), batch_size):
            prompts = descriptions[start:start + batch_size]
            tokenized = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=maximum_length,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids_cpu = tokenized["input_ids"].clone()
            attention_mask_cpu = tokenized["attention_mask"].clone()
            model_inputs = {
                "input_ids": input_ids_cpu.to(device),
                "attention_mask": attention_mask_cpu.to(device),
            }
            output = text_encoder(**model_inputs, return_dict=True)
            hidden = output.last_hidden_state
            if hidden.shape[1] != maximum_length:
                raise RuntimeError(
                    f"Expected sequence length {maximum_length}, received {hidden.shape[1]}"
                )
            batch_vectors, eot_indices = _validate_eot_batch(
                hidden,
                model_inputs["attention_mask"],
                model_inputs["input_ids"],
                eos_token_id,
            )
            vectors.append(batch_vectors.detach().to(torch.float32).cpu())
            for offset, prompt in enumerate(prompts):
                eot_index = int(eot_indices[offset].item())
                effective_length = int(attention_mask_cpu[offset].sum().item())
                untruncated_length = _untruncated_token_length(tokenizer, prompt)
                token_ids = input_ids_cpu[offset].tolist()
                audits.append({
                    "sample_index": start + offset,
                    "prompt": prompt,
                    "eot_index": eot_index,
                    "effective_token_length": effective_length,
                    "untruncated_token_length": untruncated_length,
                    "truncation_occurred": untruncated_length > maximum_length,
                    "eot_token_id": int(token_ids[eot_index]),
                    "eot_decoded_token": tokenizer.convert_ids_to_tokens([token_ids[eot_index]])[0],
                    "input_ids": json.dumps(token_ids),
                    "attention_mask": json.dumps(attention_mask_cpu[offset].tolist()),
                })
    embeddings = torch.cat(vectors, dim=0)
    if embeddings.shape != (len(descriptions), 768):
        raise RuntimeError(f"Expected EOT embedding shape ({len(descriptions)}, 768), got {tuple(embeddings.shape)}")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Final EOT embedding matrix contains NaN or Inf")
    return embeddings.numpy(), audits


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("Cannot L2-normalize non-finite or zero-length vectors")
    result = matrix / norms
    if not np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-7):
        raise RuntimeError("Row normalization did not produce unit vectors")
    return result


def build_eot_representations(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = normalize_rows(embeddings)
    global_mean = np.asarray(embeddings, dtype=np.float64).mean(axis=0)
    centered = normalize_rows(np.asarray(embeddings, dtype=np.float64) - global_mean)
    return raw, centered, global_mean


def _cosine_kmeans_plus_plus(
    features: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    n_samples = features.shape[0]
    selected = [int(rng.integers(n_samples))]
    closest_similarity = features @ features[selected[0]]
    while len(selected) < k:
        cosine_distance = np.clip(1.0 - closest_similarity, 0.0, 2.0)
        weights = cosine_distance**2
        weights[np.asarray(selected, dtype=int)] = 0.0
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 1e-15:
            available = np.setdiff1d(np.arange(n_samples), np.asarray(selected), assume_unique=False)
            next_index = int(rng.choice(available))
        else:
            next_index = int(rng.choice(n_samples, p=weights / total))
            if next_index in selected:
                available = np.setdiff1d(np.arange(n_samples), np.asarray(selected), assume_unique=False)
                next_index = int(rng.choice(available))
        selected.append(next_index)
        closest_similarity = np.maximum(closest_similarity, features @ features[next_index])
    return features[np.asarray(selected)].copy()


def _updated_spherical_centers(
    features: np.ndarray,
    labels: np.ndarray,
    similarities: np.ndarray,
    k: int,
) -> np.ndarray:
    centers = np.empty((k, features.shape[1]), dtype=np.float64)
    represented = similarities.max(axis=1).copy()
    reserved: set[int] = set()
    for cluster in range(k):
        members = features[labels == cluster]
        if len(members):
            center = members.mean(axis=0)
            norm = float(np.linalg.norm(center))
            if norm > 1e-12 and math.isfinite(norm):
                centers[cluster] = center / norm
                continue
        # Empty or numerically cancelled cluster: use the least represented
        # sample, never reusing the same rescue sample for another empty center.
        order = np.argsort(represented)
        sample_index = next(int(index) for index in order if int(index) not in reserved)
        reserved.add(sample_index)
        represented[sample_index] = np.inf
        centers[cluster] = features[sample_index]
    return centers


def fit_spherical_kmeans(
    features: np.ndarray,
    k: int = 4,
    n_init: int = 50,
    max_iter: int = 300,
    tolerance: float = 1e-6,
    random_seed: int = 0,
) -> SphericalKMeansResult:
    """Fit spherical k-means.  This API deliberately has no label argument."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < k:
        raise ValueError(f"Need a 2D feature matrix with at least k={k} samples")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or Inf")
    if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-7):
        raise ValueError("Spherical k-means requires unit-normalized input rows")
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive")

    master = np.random.default_rng(random_seed)
    initialization_seeds = master.integers(0, np.iinfo(np.int64).max, size=n_init, dtype=np.int64)
    best: SphericalKMeansResult | None = None
    all_objectives: list[float] = []
    for initialization, initialization_seed in enumerate(initialization_seeds):
        rng = np.random.default_rng(int(initialization_seed))
        centers = _cosine_kmeans_plus_plus(features, k, rng)
        previous_objective: float | None = None
        previous_labels: np.ndarray | None = None
        converged = False
        labels = np.zeros(features.shape[0], dtype=np.int64)
        objective = -np.inf
        iteration = 0
        for iteration in range(1, max_iter + 1):
            similarities = features @ centers.T
            labels = similarities.argmax(axis=1).astype(np.int64)
            objective = float(similarities[np.arange(len(features)), labels].sum())
            assignments_stable = previous_labels is not None and np.array_equal(labels, previous_labels)
            objective_stable = (
                previous_objective is not None
                and abs(objective - previous_objective)
                <= tolerance * max(1.0, abs(previous_objective))
            )
            if assignments_stable or objective_stable:
                converged = True
                break
            centers = _updated_spherical_centers(features, labels, similarities, k)
            previous_labels = labels.copy()
            previous_objective = objective

        # Make the returned labels/objective correspond exactly to returned centers.
        similarities = features @ centers.T
        labels = similarities.argmax(axis=1).astype(np.int64)
        objective = float(similarities[np.arange(len(features)), labels].sum())
        all_objectives.append(objective)
        candidate = SphericalKMeansResult(
            labels=labels,
            centers=centers,
            objective=objective,
            n_iter=iteration,
            converged=converged,
            best_initialization=initialization,
            all_objectives=(),
        )
        if best is None or candidate.objective > best.objective:
            best = candidate
    assert best is not None
    return SphericalKMeansResult(
        labels=best.labels,
        centers=best.centers,
        objective=best.objective,
        n_iter=best.n_iter,
        converged=best.converged,
        best_initialization=best.best_initialization,
        all_objectives=tuple(all_objectives),
    )


def cluster_representations_without_labels(
    raw_features: np.ndarray,
    centered_features: np.ndarray,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    tolerance: float,
    random_seed: int,
) -> dict[str, SphericalKMeansResult]:
    """The complete unsupervised fitting boundary; true labels cannot enter."""
    settings = dict(
        k=k,
        n_init=n_init,
        max_iter=max_iter,
        tolerance=tolerance,
        random_seed=random_seed,
    )
    return {
        "raw": fit_spherical_kmeans(raw_features, **settings),
        "centered": fit_spherical_kmeans(centered_features, **settings),
    }


def _hungarian_match(
    true_ids: np.ndarray, cluster_ids: np.ndarray, k: int
) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    true_by_cluster = confusion_matrix(true_ids, cluster_ids, labels=np.arange(k))
    true_rows, cluster_columns = linear_sum_assignment(-true_by_cluster)
    mapping = {
        int(cluster): int(true)
        for true, cluster in zip(true_rows.tolist(), cluster_columns.tolist())
    }
    predicted_ids = np.asarray([mapping[int(cluster)] for cluster in cluster_ids], dtype=np.int64)
    matched_confusion = confusion_matrix(true_ids, predicted_ids, labels=np.arange(k))
    return mapping, predicted_ids, matched_confusion


def evaluate_after_clustering(
    features: np.ndarray,
    result: SphericalKMeansResult,
    true_ids: np.ndarray,
    concept_names: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """The first function in the pipeline allowed to consume true labels."""
    k = len(concept_names)
    mapping, predicted_ids, matched_confusion = _hungarian_match(true_ids, result.labels, k)
    class_counts = np.bincount(true_ids, minlength=k)
    recalls = np.diag(matched_confusion) / class_counts
    counts_by_direction = {
        concept_names[true_id]: {
            concept_names[predicted_id]: int(matched_confusion[true_id, predicted_id])
            for predicted_id in range(k)
        }
        for true_id in range(k)
    }
    off_diagonal = matched_confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    largest_index = np.unravel_index(int(off_diagonal.argmax()), off_diagonal.shape)
    largest_count = int(off_diagonal[largest_index])
    largest_pair = None if largest_count == 0 else {
        "true_concept": concept_names[int(largest_index[0])],
        "predicted_concept": concept_names[int(largest_index[1])],
        "count": largest_count,
    }
    metrics = {
        "adjusted_rand_index": float(adjusted_rand_score(true_ids, result.labels)),
        "normalized_mutual_information": float(normalized_mutual_info_score(true_ids, result.labels)),
        "hungarian_matched_accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "cosine_silhouette_score": float(silhouette_score(features, result.labels, metric="cosine")),
        "spherical_objective": float(result.objective),
        "objective_per_sample": float(result.objective / len(features)),
        "iterations": int(result.n_iter),
        "converged": bool(result.converged),
        "best_initialization": int(result.best_initialization),
        "cluster_sizes": np.bincount(result.labels, minlength=k).astype(int).tolist(),
        "cluster_to_concept": {
            str(cluster): concept_names[concept_id] for cluster, concept_id in mapping.items()
        },
        "per_class_recall": {
            concept_names[index]: float(recalls[index]) for index in range(k)
        },
        "assignment_counts_by_true_class": counts_by_direction,
        "largest_confusion_pair": largest_pair,
    }
    return metrics, predicted_ids, matched_confusion


def _save_confusion(
    matrix: np.ndarray,
    concept_names: list[str],
    title: str,
    csv_path: Path,
    image_path: Path,
) -> None:
    frame = pd.DataFrame(matrix, index=concept_names, columns=concept_names)
    frame.index.name = "true_concept"
    frame.columns.name = "matched_predicted_concept"
    frame.to_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    ax.set_xticks(range(len(concept_names)), concept_names)
    ax.set_yticks(range(len(concept_names)), concept_names)
    ax.set_xlabel("Hungarian-matched predicted concept")
    ax.set_ylabel("True concept")
    ax.set_title(title)
    threshold = float(matrix.max()) * 0.58
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            ax.text(column, row, str(value), ha="center", va="center",
                    color="white" if value > threshold else "#222222")
    fig.colorbar(image, ax=ax, shrink=0.82, label="Description count")
    fig.tight_layout()
    fig.savefig(image_path, dpi=180)
    plt.close(fig)


def _pca_coordinates(features: np.ndarray, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = PCA(n_components=2, random_state=random_seed)
    points = model.fit_transform(features)
    return points, model.explained_variance_ratio_


def _save_pca_plot(
    points: np.ndarray,
    variance_ratio: np.ndarray,
    labels: np.ndarray,
    concept_names: list[str],
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for concept_id, concept in enumerate(concept_names):
        mask = labels == concept_id
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=30,
            alpha=0.76,
            label=f"{concept} (n={int(mask.sum())})",
            color=PALETTE.get(concept, f"C{concept_id}"),
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({variance_ratio[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance_ratio[1] * 100:.1f}% variance)")
    ax.axhline(0, color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _true_class_means(features: np.ndarray, true_ids: np.ndarray, k: int) -> np.ndarray:
    means = np.stack([features[true_ids == concept_id].mean(axis=0) for concept_id in range(k)])
    return normalize_rows(means)


def _save_error_analysis(
    rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    features: np.ndarray,
    true_ids: np.ndarray,
    predicted_ids: np.ndarray,
    concept_names: list[str],
    representation: str,
    output_dir: Path,
) -> None:
    class_means = _true_class_means(features, true_ids, len(concept_names))
    error_rows: list[dict[str, Any]] = []
    for index, (true_id, predicted_id) in enumerate(zip(true_ids, predicted_ids)):
        if int(true_id) == int(predicted_id):
            continue
        true_similarity = float(features[index] @ class_means[int(true_id)])
        predicted_similarity = float(features[index] @ class_means[int(predicted_id)])
        error_rows.append({
            "sample_index": index,
            "candidate_id": rows[index].get("candidate_id", f"row_{index:04d}"),
            "description": rows[index]["description"],
            "true_concept": concept_names[int(true_id)],
            "matched_predicted_concept": concept_names[int(predicted_id)],
            "eot_token_index": audits[index]["eot_index"],
            "effective_token_length": audits[index]["effective_token_length"],
            "cosine_similarity_to_true_class_mean": true_similarity,
            "cosine_similarity_to_predicted_class_mean": predicted_similarity,
            "true_minus_predicted_similarity": true_similarity - predicted_similarity,
        })
    error_rows.sort(key=lambda row: (row["true_concept"], row["true_minus_predicted_similarity"]))
    fields = [
        "sample_index", "candidate_id", "description", "true_concept",
        "matched_predicted_concept", "eot_token_index", "effective_token_length",
        "cosine_similarity_to_true_class_mean", "cosine_similarity_to_predicted_class_mean",
        "true_minus_predicted_similarity",
    ]
    write_csv(output_dir / f"misclustered_{representation}.csv", error_rows, fields)

    lines = [
        f"Misclustered fox descriptions — {representation} EOT",
        "Descriptions are unchanged. Groups use Hungarian-matched predicted concepts.",
        "",
    ]
    fox_errors = [row for row in error_rows if row["true_concept"] == "fox"]
    for predicted_concept in [name for name in concept_names if name != "fox"]:
        group = [row for row in fox_errors if row["matched_predicted_concept"] == predicted_concept]
        lines.append(f"Assigned to {predicted_concept} ({len(group)})")
        lines.append("-" * (14 + len(predicted_concept)))
        if not group:
            lines.append("(none)")
        else:
            for row in group:
                lines.append(
                    f"- {row['description']} [true−predicted similarity="
                    f"{row['true_minus_predicted_similarity']:.6f}]"
                )
        lines.append("")
    atomic_write_text(output_dir / f"fox_errors_{representation}.txt", "\n".join(lines) + "\n")


def _summary_text(
    metrics: dict[str, dict[str, Any]], concept_names: list[str]
) -> str:
    header = (
        "Representation | ARI | NMI | Matched Accuracy | Cosine Silhouette | Spherical Objective\n"
        "--- | ---: | ---: | ---: | ---: | ---:\n"
    )
    rows = []
    for key, name in [("raw", "Raw EOT"), ("centered", "Centered EOT")]:
        item = metrics[key]
        rows.append(
            f"{name} | {item['adjusted_rand_index']:.6f} | "
            f"{item['normalized_mutual_information']:.6f} | "
            f"{item['hungarian_matched_accuracy']:.6f} | "
            f"{item['cosine_silhouette_score']:.6f} | "
            f"{item['spherical_objective']:.6f}"
        )
    lines = [header + "\n".join(rows), ""]
    for key, name in [("raw", "Raw EOT"), ("centered", "Centered EOT")]:
        item = metrics[key]
        recalls = ", ".join(
            f"{concept}={item['per_class_recall'][concept]:.3f}" for concept in concept_names
        )
        lines.append(f"{name} per-class recall: {recalls}")
        lines.append(f"{name} assignment counts (true → matched prediction):")
        for true_concept in concept_names:
            counts = item["assignment_counts_by_true_class"][true_concept]
            lines.append(
                "  " + true_concept + ": "
                + ", ".join(f"{predicted}={counts[predicted]}" for predicted in concept_names)
            )
        pair = item["largest_confusion_pair"]
        lines.append(
            f"{name} largest confusion: none"
            if pair is None
            else f"{name} largest confusion: {pair['true_concept']} → "
                 f"{pair['predicted_concept']} ({pair['count']})"
        )
        lines.append("")
    raw = metrics["raw"]
    centered = metrics["centered"]
    lines.extend([
        f"Centered ARI minus raw ARI: "
        f"{centered['adjusted_rand_index'] - raw['adjusted_rand_index']:+.6f}",
        f"Centered fox recall minus raw fox recall: "
        f"{centered['per_class_recall']['fox'] - raw['per_class_recall']['fox']:+.6f}",
    ])
    all_directions: list[tuple[int, str, str, str]] = []
    for representation in ("raw", "centered"):
        counts = metrics[representation]["assignment_counts_by_true_class"]
        for true_concept in concept_names:
            for predicted in concept_names:
                if true_concept != predicted:
                    all_directions.append((counts[true_concept][predicted], representation, true_concept, predicted))
    count, representation, true_concept, predicted = max(all_directions)
    lines.append(
        f"Most frequent misclassification direction across the two fits: "
        f"{representation} {true_concept} → {predicted} ({count})"
    )
    lines.append("")
    lines.append(
        "Numerical comparison only: PCA is qualitative and all clustering/evaluation used 768 dimensions."
    )
    return "\n".join(lines) + "\n"


def _write_metadata(
    path: Path,
    rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
) -> None:
    metadata = []
    for index, (row, audit) in enumerate(zip(rows, audits)):
        metadata.append({
            "sample_index": index,
            "candidate_id": row.get("candidate_id", f"row_{index:04d}"),
            "concept": row["concept"],
            "facet_id": row.get("facet_id", ""),
            "source": row.get("source", ""),
            "description": row["description"],
            **{key: value for key, value in audit.items() if key not in {"sample_index", "prompt"}},
        })
    write_csv(path, metadata)


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in REQUIRED_OUTPUTS if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing result files in {output_dir}; "
            "choose a new --output or pass --overwrite"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_reproducible_seed(args.seed)
    dataset_path = args.dataset.resolve()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()
    config = load_config(config_path)
    config = copy.deepcopy(config)
    config["model"]["device"] = args.device
    if args.model:
        config["model"]["model_id"] = args.model
    _prepare_output(output_dir, args.overwrite)

    rows = load_description_dataset(dataset_path)
    concept_names, class_counts = validate_dataset(rows, config)
    if len(concept_names) != 4:
        raise ValueError(f"This experiment requires exactly four concepts, got {concept_names}")
    descriptions = [row["description"] for row in rows]
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(rows)}; per class: {class_counts}")

    pipe = load_original_pipeline(config, purpose="embedding", include_vae=False)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    embeddings, audits = extract_eot_embeddings(
        tokenizer,
        text_encoder,
        descriptions,
        args.device,
        args.batch_size,
    )
    print(f"EOT embedding tensor shape: {embeddings.shape}")
    print("Manual EOT audit (first five prompts):")
    for audit in audits[:5]:
        print(
            f"  row={audit['sample_index']} index={audit['eot_index']} "
            f"length={audit['effective_token_length']} token={audit['eot_decoded_token']} "
            f"prompt={audit['prompt']!r}"
        )
    if any(bool(row["truncation_occurred"]) for row in audits):
        raise RuntimeError("At least one description was truncated; refusing to cluster altered prompts")

    raw_features, centered_features, global_mean = build_eot_representations(embeddings)
    np.save(output_dir / "eot_embeddings.npy", embeddings.astype(np.float32, copy=False))
    np.save(output_dir / "eot_raw_normalized.npy", raw_features.astype(np.float32))
    np.save(output_dir / "eot_centered_normalized.npy", centered_features.astype(np.float32))
    _write_metadata(output_dir / "metadata.csv", rows, audits)

    # Critical leakage boundary: neither this call nor either fit has access to labels.
    fits = cluster_representations_without_labels(
        raw_features,
        centered_features,
        k=4,
        n_init=args.n_init,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        random_seed=args.seed,
    )

    # True labels are intentionally constructed only after both clustering fits.
    true_ids = np.asarray([concept_names.index(str(row["concept"])) for row in rows], dtype=np.int64)
    representation_features = {"raw": raw_features, "centered": centered_features}
    representation_metrics: dict[str, dict[str, Any]] = {}
    for representation in ("raw", "centered"):
        features = representation_features[representation]
        metrics, predicted_ids, matched_confusion = evaluate_after_clustering(
            features,
            fits[representation],
            true_ids,
            concept_names,
        )
        representation_metrics[representation] = metrics
        _save_confusion(
            matched_confusion,
            concept_names,
            f"{representation.capitalize()} EOT — matched confusion (n={len(rows)})",
            output_dir / f"confusion_{representation}.csv",
            output_dir / f"confusion_{representation}.png",
        )
        points, variance = _pca_coordinates(features, args.seed)
        _save_pca_plot(
            points,
            variance,
            true_ids,
            concept_names,
            f"{representation.capitalize()} EOT PCA — true concepts (visualization only)",
            output_dir / f"pca_{representation}_true_labels.png",
        )
        _save_pca_plot(
            points,
            variance,
            predicted_ids,
            concept_names,
            f"{representation.capitalize()} EOT PCA — matched clusters (visualization only)",
            output_dir / f"pca_{representation}_predicted_clusters.png",
        )
        _save_error_analysis(
            rows,
            audits,
            features,
            true_ids,
            predicted_ids,
            concept_names,
            representation,
            output_dir,
        )

    metrics_payload = {
        "experiment": "unsuffixed_sd14_eot_true_spherical_kmeans",
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "schema_fields": sorted(rows[0]),
            "sample_count": len(rows),
            "concept_order": concept_names,
            "class_counts": class_counts,
        },
        "model": {
            "model_id_or_path": config["model"]["model_id"],
            "pipeline_class": type(pipe).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_model_max_length": int(tokenizer.model_max_length),
            "text_encoder_class": type(text_encoder).__name__,
            "text_hidden_size": int(text_encoder.config.hidden_size),
            "text_encoder_dtype": str(next(text_encoder.parameters()).dtype),
            "device": args.device,
        },
        "extraction": {
            "prompt_transform": "none; each original description is encoded exactly as stored",
            "eot_rule": "attention_mask.sum(dim=1) - 1",
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype_saved": str(embeddings.dtype),
            "truncated_prompt_count": int(sum(bool(row["truncation_occurred"]) for row in audits)),
            "effective_token_length_min": int(min(row["effective_token_length"] for row in audits)),
            "effective_token_length_max": int(max(row["effective_token_length"] for row in audits)),
        },
        "preprocessing": {
            "raw": "one L2 normalization per unmodified EOT vector",
            "centered": "one label-free global mean over all EOT vectors, subtract, then row L2 normalize",
            "global_mean_norm": float(np.linalg.norm(global_mean)),
            "labels_used": False,
        },
        "spherical_kmeans": {
            "implementation": "cosine assignment, normalized mean update, cosine k-means++ initialization",
            "k": 4,
            "n_init": args.n_init,
            "max_iter": args.max_iter,
            "tolerance": args.tolerance,
            "random_seed": args.seed,
            "labels_available_to_fit": False,
        },
        "representations": representation_metrics,
        "package_versions": package_versions(),
    }
    atomic_write_text(output_dir / "metrics.json", json.dumps(metrics_payload, indent=2) + "\n")
    atomic_write_text(
        output_dir / "metrics_summary.txt",
        _summary_text(representation_metrics, concept_names),
    )

    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).is_file()]
    empty = [name for name in REQUIRED_OUTPUTS if (output_dir / name).is_file() and (output_dir / name).stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(f"Output validation failed; missing={missing}, empty={empty}")
    print(f"Produced all {len(REQUIRED_OUTPUTS)} requested files in {output_dir}")
    print((output_dir / "metrics_summary.txt").read_text())
    return metrics_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster exact unsuffixed SD 1.4 CLIP EOT states using true spherical k-means"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Accepted description .jsonl/.json")
    parser.add_argument("--config", type=Path, required=True, help="Repository experiment config")
    parser.add_argument("--model", default=None, help="Override SD 1.4 model id/path from config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
