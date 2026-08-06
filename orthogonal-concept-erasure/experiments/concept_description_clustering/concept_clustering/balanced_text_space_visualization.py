from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from scripts.eot_spherical_clustering import normalize_rows

from .utils import atomic_write_text, package_versions, read_jsonl


CONCEPTS = ["cat", "dog", "fox", "bear", "wolf", "rabbit", "deer", "horse"]
REPRESENTATIONS = ["eot", "fixed_suffix"]
REPRESENTATION_LABELS = {
    "eot": "Unsuffixed EOT",
    "fixed_suffix": "Fixed suffix",
}
EXPECTED_METRICS = {
    "eot": {"ari": 0.6338, "matched_accuracy": 0.7750},
    "fixed_suffix": {"ari": 0.4697, "matched_accuracy": 0.7025},
}
EXPECTED_SOURCE_HASHES = {
    "dataset": "6f459fa9e73f80163145813ecd8cd32c4216e1c4f77f6795925bf06676794a0c",
    "eot_embedding": "4c175ad2c4f7b848fa561dd464104eb9f27a29c36591912977873e6b52b77070",
    "fixed_suffix_embedding": "033aac5e6158387f40d3fb4fcdb8be68990879bc47ce4b5a48aeec5c882ea03f",
    "eot_assignments": "9e7e39ed5e780ddb0a4be5ab1d33c48e0b9be00ae53d3bbdf145a2d7ed11ff3b",
    "fixed_suffix_assignments": "d6da64a9335c6cc685624f182e34812c1345e84dcd8a85dbb61d99f49e23f9b3",
}
ANIMAL_COLORS = {
    "cat": "#0072B2",
    "dog": "#E69F00",
    "fox": "#D55E00",
    "bear": "#8C564B",
    "wolf": "#7A7A7A",
    "rabbit": "#CC79A7",
    "deer": "#009E73",
    "horse": "#56B4E9",
}
ANIMAL_MARKERS = {
    "cat": "o",
    "dog": "s",
    "fox": "^",
    "bear": "D",
    "wolf": "v",
    "rabbit": "P",
    "deer": "X",
    "horse": "<",
}
TSNE_CONFIG = {
    "n_components": 2,
    "perplexity": 30,
    "init": "pca",
    "learning_rate": "auto",
    "max_iter": 2000,
    "metric": "cosine",
}
REQUIRED_PNGS = [
    "pca_eot_true_labels.png",
    "pca_eot_predicted_clusters.png",
    "tsne_eot_true_labels.png",
    "tsne_eot_predicted_clusters.png",
    "pca_fixed_true_labels.png",
    "pca_fixed_predicted_clusters.png",
    "tsne_fixed_true_labels.png",
    "tsne_fixed_predicted_clusters.png",
    "tsne_eot_seed_robustness.png",
    "pca_eot_3d_true_labels.png",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_sources(dataset_arg: str | Path) -> dict[str, Path]:
    supplied = Path(dataset_arg).expanduser().resolve()
    source_dir = supplied if supplied.is_dir() else supplied.parent
    dataset_path = source_dir / "accepted_descriptions.jsonl" if supplied.is_dir() else supplied
    paths = {
        "source_dir": source_dir,
        "dataset": dataset_path,
        "validation": source_dir / "dataset_validation.json",
        "source_config": source_dir / "experiment_config.json",
        "eot_embedding": source_dir / "eot_embeddings.npy",
        "fixed_suffix_embedding": source_dir / "fixed_suffix_embeddings.npy",
        "eot_metrics": source_dir / "eot_metrics.json",
        "fixed_suffix_metrics": source_dir / "fixed_suffix_metrics.json",
        "eot_assignments": source_dir / "appendix" / "assignments_eot.csv",
        "fixed_suffix_assignments": source_dir / "appendix" / "assignments_fixed_suffix.csv",
        "eot_tokenization": source_dir / "eot_tokenization_audit.csv",
        "fixed_suffix_tokenization": source_dir / "fixed_suffix_tokenization_audit.csv",
    }
    missing = [str(path) for key, path in paths.items() if key != "source_dir" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Final text-space cache is incomplete: {missing}")
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().map({"true": True, "false": False})


def _audit_dataset_and_cache(
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, pd.DataFrame], dict[str, Any]]:
    rows = read_jsonl(paths["dataset"])
    counts = Counter(str(row.get("concept")) for row in rows)
    pairs = {(str(row.get("slot_id")), str(row.get("concept"))) for row in rows}
    validation = _read_json(paths["validation"])
    source_config = _read_json(paths["source_config"])
    source_concepts = [str(item["name"]) for item in source_config.get("concepts", [])]
    dataset_hash = _sha256(paths["dataset"])
    if dataset_hash != EXPECTED_SOURCE_HASHES["dataset"]:
        raise RuntimeError(
            f"Dataset is not the frozen final balanced 8x50 cache: {dataset_hash}"
        )

    embeddings_raw = {
        "eot": np.load(paths["eot_embedding"], allow_pickle=False),
        "fixed_suffix": np.load(paths["fixed_suffix_embedding"], allow_pickle=False),
    }
    embeddings: dict[str, np.ndarray] = {}
    embedding_audit: dict[str, Any] = {}
    for representation, raw in embeddings_raw.items():
        embedding_hash = _sha256(paths[f"{representation}_embedding"])
        if embedding_hash != EXPECTED_SOURCE_HASHES[f"{representation}_embedding"]:
            raise RuntimeError(f"{representation} embedding fingerprint does not match the final experiment")
        if raw.shape != (400, 768):
            raise RuntimeError(f"Expected {representation} cache shape (400, 768), found {raw.shape}")
        if not np.isfinite(raw).all():
            raise RuntimeError(f"{representation} cache contains NaN or Inf")
        normalized = normalize_rows(raw).astype(np.float32)
        norms = np.linalg.norm(normalized, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise RuntimeError(f"{representation} visualization inputs are not row-normalized")
        embeddings[representation] = normalized
        embedding_audit[representation] = {
            "path": str(paths[f"{representation}_embedding"]),
            "sha256": embedding_hash,
            "cached_shape": list(raw.shape),
            "visualization_input_shape": list(normalized.shape),
            "cached_norm_min": float(np.linalg.norm(raw, axis=1).min()),
            "cached_norm_max": float(np.linalg.norm(raw, axis=1).max()),
            "normalized_norm_min": float(norms.min()),
            "normalized_norm_max": float(norms.max()),
            "all_768_dimensions_used": True,
        }

    tokenization_audit: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        frame = pd.read_csv(paths[f"{representation}_tokenization"])
        if len(frame) != 400:
            raise RuntimeError(f"Expected 400 {representation} tokenization rows, found {len(frame)}")
        truncated = _as_bool(frame["truncation_occurred"])
        if truncated.isna().any() or bool(truncated.any()):
            raise RuntimeError(f"{representation} tokenization audit contains truncation")
        tokenization_audit[representation] = {
            "path": str(paths[f"{representation}_tokenization"]),
            "sha256": _sha256(paths[f"{representation}_tokenization"]),
            "row_count": len(frame),
            "truncated_count": int(truncated.sum()),
        }

    assignments: dict[str, pd.DataFrame] = {}
    metric_audit: dict[str, Any] = {}
    expected_ids = list(range(400))
    for representation in REPRESENTATIONS:
        assignment_hash = _sha256(paths[f"{representation}_assignments"])
        if assignment_hash != EXPECTED_SOURCE_HASHES[f"{representation}_assignments"]:
            raise RuntimeError(f"{representation} assignment fingerprint does not match the final experiment")
        frame = pd.read_csv(paths[f"{representation}_assignments"]).sort_values("sample_index").reset_index(drop=True)
        if frame["sample_index"].astype(int).tolist() != expected_ids:
            raise RuntimeError(f"{representation} assignments do not contain sample_index 0..399 exactly once")
        for index, row in enumerate(rows):
            if str(frame.at[index, "candidate_id"]) != str(row["candidate_id"]):
                raise RuntimeError(f"{representation} candidate identity mismatch at sample {index}")
            if str(frame.at[index, "description"]) != str(row["description"]):
                raise RuntimeError(f"{representation} description mismatch at sample {index}")
            if str(frame.at[index, "true_concept"]) != str(row["concept"]):
                raise RuntimeError(f"{representation} concept mismatch at sample {index}")
        frame["predicted_cluster"] = frame["predicted_cluster"].astype(int)
        frame["prediction_correct"] = frame["true_concept"] == frame["matched_predicted_concept"]

        metric_payload = _read_json(paths[f"{representation}_metrics"])
        if metric_payload.get("dataset_sha256") != dataset_hash:
            raise RuntimeError(f"{representation} metrics were computed on a different dataset")
        mapping = {int(key): str(value) for key, value in metric_payload["metrics"]["cluster_to_concept"].items()}
        mapped = frame["predicted_cluster"].map(mapping)
        if mapped.isna().any() or not mapped.equals(frame["matched_predicted_concept"]):
            raise RuntimeError(f"{representation} Hungarian mapping disagrees with cached assignments")
        true_ids = frame["true_concept"].map({name: index for index, name in enumerate(CONCEPTS)}).to_numpy()
        ari = float(adjusted_rand_score(true_ids, frame["predicted_cluster"].to_numpy()))
        accuracy = float(frame["prediction_correct"].mean())
        reported_ari = float(metric_payload["metrics"]["adjusted_rand_index"])
        reported_accuracy = float(metric_payload["metrics"]["hungarian_matched_accuracy"])
        expected = EXPECTED_METRICS[representation]
        if not np.isclose(ari, reported_ari, atol=1e-12):
            raise RuntimeError(f"{representation} assignment ARI does not reproduce its saved metric")
        if not np.isclose(accuracy, reported_accuracy, atol=1e-12):
            raise RuntimeError(f"{representation} assignment accuracy does not reproduce its saved metric")
        if round(ari, 4) != expected["ari"] or round(accuracy, 4) != expected["matched_accuracy"]:
            raise RuntimeError(
                f"{representation} is not the final experiment: ARI={ari:.8f}, accuracy={accuracy:.8f}"
            )
        assignments[representation] = frame
        metric_audit[representation] = {
            "metrics_path": str(paths[f"{representation}_metrics"]),
            "metrics_sha256": _sha256(paths[f"{representation}_metrics"]),
            "assignments_path": str(paths[f"{representation}_assignments"]),
            "assignments_sha256": assignment_hash,
            "saved_ari": reported_ari,
            "assignment_recomputed_ari": ari,
            "required_rounded_ari": expected["ari"],
            "saved_matched_accuracy": reported_accuracy,
            "assignment_recomputed_matched_accuracy": accuracy,
            "required_rounded_matched_accuracy": expected["matched_accuracy"],
            "hungarian_mapping": {str(key): value for key, value in mapping.items()},
            "clustering_refit": False,
        }

    if not assignments["eot"][["candidate_id", "true_concept", "description"]].equals(
        assignments["fixed_suffix"][["candidate_id", "true_concept", "description"]]
    ):
        raise RuntimeError("EOT and fixed-suffix assignments do not index the same 400 descriptions")

    checks = {
        "exactly_400_rows": len(rows) == 400,
        "exact_concept_order": source_concepts == CONCEPTS,
        "exactly_50_rows_per_concept": all(counts[name] == 50 for name in CONCEPTS),
        "exactly_400_unique_concept_slot_pairs": len(pairs) == 400,
        "source_validation_passed": validation.get("status") == "passed",
        "source_validation_no_truncation": validation.get("checks", {}).get("no_truncation") is True,
        "both_embedding_shapes_are_400_by_768": all(value.shape == (400, 768) for value in embeddings.values()),
        "both_visualization_inputs_row_l2_normalized": all(
            np.allclose(np.linalg.norm(value, axis=1), 1.0, atol=1e-6) for value in embeddings.values()
        ),
        "both_assignments_match_dataset_row_order": True,
        "both_assignments_match_saved_hungarian_mappings": True,
        "reported_metrics_reproduced_without_refitting": True,
        "frozen_dataset_embedding_and_assignment_hashes_match": True,
        "only_text_space_embeddings_used": True,
        "w0_projection_applied": False,
        "clustering_refit": False,
    }
    positive_checks = [key for key in checks if key not in {"w0_projection_applied", "clustering_refit"}]
    if (
        not all(checks[key] is True for key in positive_checks)
        or checks["w0_projection_applied"] is not False
        or checks["clustering_refit"] is not False
    ):
        raise RuntimeError(f"Final balanced text-space cache audit failed: {checks}")
    audit = {
        "status": "passed",
        "dataset_path": str(paths["dataset"]),
        "dataset_sha256": dataset_hash,
        "dataset_validation_path": str(paths["validation"]),
        "dataset_validation_sha256": _sha256(paths["validation"]),
        "source_experiment_config_path": str(paths["source_config"]),
        "source_experiment_config_sha256": _sha256(paths["source_config"]),
        "row_count": len(rows),
        "counts_by_concept": {concept: counts[concept] for concept in CONCEPTS},
        "checks": checks,
        "embeddings": embedding_audit,
        "tokenization": tokenization_audit,
        "metrics_and_assignments": metric_audit,
    }
    return rows, embeddings, assignments, audit


def _padded_limits(coordinates: np.ndarray, fraction: float = 0.055) -> dict[str, list[float]]:
    x_min, y_min = np.min(coordinates, axis=0)
    x_max, y_max = np.max(coordinates, axis=0)
    x_pad = max(float(x_max - x_min) * fraction, 1e-6)
    y_pad = max(float(y_max - y_min) * fraction, 1e-6)
    return {
        "x": [float(x_min - x_pad), float(x_max + x_pad)],
        "y": [float(y_min - y_pad), float(y_max + y_pad)],
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B8B8B8")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.grid(color="#E8E8E8", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)


def _legend_handles(include_misclustered: bool) -> list[Line2D]:
    handles = [
        Line2D(
            [0], [0], marker=ANIMAL_MARKERS[concept], linestyle="none",
            markerfacecolor=ANIMAL_COLORS[concept], markeredgecolor="white",
            markeredgewidth=0.55, markersize=7.5, label=concept,
        )
        for concept in CONCEPTS
    ]
    if include_misclustered:
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.2, markersize=7.5,
            label="misclustered outline",
        ))
    return handles


def _save_scatter(
    coordinates: np.ndarray,
    frame: pd.DataFrame,
    representation: str,
    method: str,
    label_column: str,
    axis_limits: dict[str, list[float]],
    path: Path,
    explained_variance: np.ndarray | None = None,
) -> None:
    by_prediction = label_column == "matched_predicted_concept"
    label_phrase = "matched predicted cluster" if by_prediction else "true animal concept"
    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    for concept in CONCEPTS:
        concept_mask = frame[label_column].to_numpy() == concept
        if not np.any(concept_mask):
            continue
        if by_prediction:
            correct = frame["prediction_correct"].to_numpy(dtype=bool)
            for correctness, edge, width in [(True, "white", 0.4), (False, "black", 0.85)]:
                mask = concept_mask & (correct == correctness)
                if np.any(mask):
                    ax.scatter(
                        coordinates[mask, 0], coordinates[mask, 1],
                        s=28, alpha=0.68, c=ANIMAL_COLORS[concept],
                        marker=ANIMAL_MARKERS[concept], edgecolors=edge,
                        linewidths=width, rasterized=False,
                    )
        else:
            ax.scatter(
                coordinates[concept_mask, 0], coordinates[concept_mask, 1],
                s=28, alpha=0.68, c=ANIMAL_COLORS[concept],
                marker=ANIMAL_MARKERS[concept], edgecolors="white",
                linewidths=0.4, rasterized=False,
            )

    display = REPRESENTATION_LABELS[representation]
    fig.suptitle(
        f"{display}: {method} by {label_phrase}",
        x=0.10, y=0.975, ha="left", fontsize=16, weight="bold",
    )
    subtitle = "Balanced 8×50 descriptions · normalized 768D input · n = 400"
    if by_prediction:
        subtitle = "Existing Hungarian mapping · " + subtitle
    if method == "t-SNE":
        subtitle += " · perplexity 30 · seed 42"
    fig.text(0.10, 0.925, subtitle, fontsize=10, color="#555555", va="bottom")
    if method == "PCA":
        if explained_variance is None:
            raise ValueError("PCA plot requires explained variance")
        ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.2f}%)")
        ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.2f}%)")
        ax.axhline(0, color="#CFCFCF", linewidth=0.75, zorder=0)
        ax.axvline(0, color="#CFCFCF", linewidth=0.75, zorder=0)
    else:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    ax.set_xlim(axis_limits["x"])
    ax.set_ylim(axis_limits["y"])
    _style_axis(ax)
    fig.legend(
        handles=_legend_handles(by_prediction), loc="upper center",
        bbox_to_anchor=(0.5, 0.895), ncol=5,
        frameon=False, fontsize=9, handletextpad=0.45, columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.76)
    fig.savefig(path, dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _save_tsne_robustness(
    coordinate_by_seed: dict[int, np.ndarray],
    assignments: pd.DataFrame,
    path: Path,
) -> dict[str, list[float]]:
    stacked = np.concatenate([coordinate_by_seed[seed] for seed in (0, 1, 42)], axis=0)
    limits = _padded_limits(stacked)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), sharex=True, sharey=True)
    labels = assignments["true_concept"].to_numpy()
    for ax, seed in zip(axes, (0, 1, 42)):
        coordinates = coordinate_by_seed[seed]
        for concept in CONCEPTS:
            mask = labels == concept
            ax.scatter(
                coordinates[mask, 0], coordinates[mask, 1], s=18, alpha=0.65,
                c=ANIMAL_COLORS[concept], marker=ANIMAL_MARKERS[concept],
                edgecolors="white", linewidths=0.3,
            )
        ax.set_title(f"random_state = {seed}", fontsize=12, weight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_xlim(limits["x"])
        ax.set_ylim(limits["y"])
        _style_axis(ax)
    axes[0].set_ylabel("t-SNE 2")
    fig.suptitle("Unsuffixed EOT t-SNE seed robustness", x=0.055, ha="left", fontsize=16, weight="bold")
    fig.text(
        0.055, 0.91,
        "Predetermined seeds 0, 1, and 42 · all other settings fixed · true animal concepts",
        fontsize=10, color="#555555",
    )
    fig.legend(
        handles=_legend_handles(False), loc="upper center", bbox_to_anchor=(0.5, 0.88),
        ncol=8, frameon=False, fontsize=9, handletextpad=0.4, columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.11, top=0.76, wspace=0.12)
    fig.savefig(path, dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return limits


def _save_3d_pca(
    coordinates: np.ndarray,
    assignments: pd.DataFrame,
    explained_variance: np.ndarray,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(10.2, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    labels = assignments["true_concept"].to_numpy()
    for concept in CONCEPTS:
        mask = labels == concept
        ax.scatter(
            coordinates[mask, 0], coordinates[mask, 1], coordinates[mask, 2],
            s=26, alpha=0.68, c=ANIMAL_COLORS[concept],
            marker=ANIMAL_MARKERS[concept], edgecolors="white", linewidths=0.35,
            depthshade=False,
        )
    fig.suptitle("Text Space: 3D PCA", x=0.08, y=0.975, ha="left", fontsize=16, weight="bold")
    fig.text(
        0.08, 0.925,
        "Unsuffixed EOT · Balanced 8×50 descriptions · normalized 768D input · n = 400",
        fontsize=10, color="#555555",
    )
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.2f}%)", labelpad=10)
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.2f}%)", labelpad=10)
    ax.set_zlabel(f"PC3 ({explained_variance[2] * 100:.2f}%)", labelpad=8)
    ax.view_init(elev=22, azim=-55)
    ax.grid(True, color="#E8E8E8", linewidth=0.65, alpha=0.75)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor("#D8D8D8")
    fig.legend(
        handles=_legend_handles(False), loc="upper center", bbox_to_anchor=(0.52, 0.89),
        ncol=4, frameon=False, fontsize=9, handletextpad=0.45, columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.04, top=0.80)
    fig.savefig(path, dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _coordinate_rows(
    representation: str,
    method: str,
    random_state: str,
    coordinates: np.ndarray,
    assignments: pd.DataFrame,
    explained_variance: np.ndarray | None,
) -> list[dict[str, Any]]:
    output = []
    for index, row in assignments.iterrows():
        output.append({
            "representation": representation,
            "method": method,
            "random_state": random_state,
            "sample_index": int(row["sample_index"]),
            "description_id": str(row["candidate_id"]),
            "description": str(row["description"]),
            "true_concept": str(row["true_concept"]),
            "raw_cluster_id": int(row["predicted_cluster"]),
            "hungarian_matched_predicted_concept": str(row["matched_predicted_concept"]),
            "prediction_correct": bool(row["prediction_correct"]),
            "x": float(coordinates[index, 0]),
            "y": float(coordinates[index, 1]),
            "explained_variance_ratio_pc1": (
                float(explained_variance[0]) if explained_variance is not None else np.nan
            ),
            "explained_variance_ratio_pc2": (
                float(explained_variance[1]) if explained_variance is not None else np.nan
            ),
        })
    return output


def _analysis_source_audit() -> dict[str, Any]:
    source = inspect.getsource(inspect.getmodule(_analysis_source_audit))
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    forbidden_imports = {"sklearn.cluster"}
    forbidden_calls = {"KMeans", "fit_spherical_kmeans", "original_projection_modules", "Orthogonal_Erase"}
    hits = sorted((imported_modules & forbidden_imports) | (called_names & forbidden_calls))
    if hits:
        raise RuntimeError(f"Visualization source contains forbidden operations: {hits}")
    return {
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": _sha256(Path(__file__).resolve()),
        "clustering_fit_calls_present": False,
        "w0_projection_calls_present": False,
        "oce_calls_present": False,
        "image_or_description_generation_calls_present": False,
    }


def _write_summary(
    output_dir: Path,
    audit: dict[str, Any],
    pca_info: dict[str, Any],
    pca_3d_info: dict[str, Any],
) -> None:
    eot_total = 100.0 * float(pca_info["eot"]["explained_variance_ratio_total"])
    fixed_total = 100.0 * float(pca_info["fixed_suffix"]["explained_variance_ratio_total"])
    text = f"""# Balanced Text-Space Visualization Summary

## PCA explained variance

| Representation | PC1 | PC2 | PC1 + PC2 |
|---|---:|---:|---:|
| Unsuffixed EOT | {100 * pca_info['eot']['explained_variance_ratio'][0]:.2f}% | {100 * pca_info['eot']['explained_variance_ratio'][1]:.2f}% | {eot_total:.2f}% |
| Fixed suffix | {100 * pca_info['fixed_suffix']['explained_variance_ratio'][0]:.2f}% | {100 * pca_info['fixed_suffix']['explained_variance_ratio'][1]:.2f}% | {fixed_total:.2f}% |

The EOT PC3 explains {100 * pca_3d_info['explained_variance_ratio'][2]:.2f}%. PC1–PC3 together explain {100 * pca_3d_info['explained_variance_ratio_total']:.2f}%.

## Visible structure

The plots use the same 400 balanced descriptions and the existing spherical-k-means assignments. The EOT views show broad animal-associated structure with substantial local overlap; fox is the least isolated class and visibly intersects neighboring small-animal regions. The fixed-suffix views retain animal-associated structure but show broader mixing, consistent with their lower unchanged 768-dimensional ARI and matched accuracy.

Across the EOT t-SNE runs with seeds 0, 1, and 42, the broad class neighborhoods and recurring overlap patterns remain visible, while orientation, local arrangement, apparent gaps, and exact boundary shapes change. Seed 42 remains the predetermined main result; no seed was selected for visual quality.

## Metric and source audit

- Dataset: `{audit['dataset_path']}`
- Dataset SHA-256: `{audit['dataset_sha256']}`
- Unsuffixed EOT: ARI {audit['metrics_and_assignments']['eot']['saved_ari']:.4f}; matched accuracy {audit['metrics_and_assignments']['eot']['saved_matched_accuracy']:.4f}.
- Fixed suffix: ARI {audit['metrics_and_assignments']['fixed_suffix']['saved_ari']:.4f}; matched accuracy {audit['metrics_and_assignments']['fixed_suffix']['saved_matched_accuracy']:.4f}.
- The existing raw cluster IDs and Hungarian mappings were reused. No clustering was refitted.
- PCA and t-SNE both received all 768 dimensions of the row-L2-normalized text-space vectors. No W0 projection or OCE operation was applied.

## Interpretation warning

PCA and t-SNE are low-dimensional visualizations only: the main PCA/t-SNE views use two dimensions, and the supplementary PCA view uses three. The clustering assignments and ARI/NMI were computed in the original 768-dimensional space. A clean-looking t-SNE does not prove that natural clusters exist, and t-SNE global distances, apparent cluster sizes, and empty gaps must not be interpreted literally.

## Presentation recommendation

Use `pca_eot_true_labels.png`, `tsne_eot_true_labels.png`, and `pca_eot_3d_true_labels.png` in the main presentation. Treat predicted-cluster views, fixed-suffix views, and the seed-robustness panel as comparison or appendix material.
"""
    atomic_write_text(output_dir / "visualization_summary.md", text)


def run_visualization(dataset: str | Path, output: str | Path, overwrite: bool = False) -> dict[str, Any]:
    paths = _resolve_sources(dataset)
    output_dir = Path(output).expanduser().resolve()
    if output_dir == paths["source_dir"] or paths["source_dir"] in output_dir.parents:
        raise ValueError("Output must be isolated from the balanced-paired source directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in [output_dir / "dataset_audit.json", output_dir / "visualization_summary.md"] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace only this derived output: {existing}")

    rows, embeddings, assignments, audit = _audit_dataset_and_cache(paths)
    atomic_write_text(output_dir / "dataset_audit.json", json.dumps(audit, indent=2) + "\n")

    coordinate_records: list[dict[str, Any]] = []
    pca_info: dict[str, Any] = {}
    pca_3d_info: dict[str, Any] = {}
    axis_limits: dict[str, Any] = {}
    pca_coordinates: dict[str, np.ndarray] = {}
    tsne_coordinates: dict[tuple[str, int], np.ndarray] = {}

    for representation in REPRESENTATIONS:
        matrix = embeddings[representation]
        if matrix.shape[1] != 768:
            raise RuntimeError("Dimensionality reduction must receive all 768 text dimensions")
        pca = PCA(n_components=2, svd_solver="full")
        coordinates = pca.fit_transform(matrix)
        if coordinates.shape != (400, 2) or not np.isfinite(coordinates).all():
            raise RuntimeError(f"Invalid {representation} PCA coordinates")
        pca_coordinates[representation] = coordinates
        variance = pca.explained_variance_ratio_
        pca_info[representation] = {
            "explained_variance_ratio": [float(value) for value in variance],
            "explained_variance_ratio_total": float(variance.sum()),
            "input_shape": list(matrix.shape),
        }
        limits = _padded_limits(coordinates)
        axis_limits[f"pca_{representation}"] = limits
        suffix = "eot" if representation == "eot" else "fixed"
        _save_scatter(
            coordinates, assignments[representation], representation, "PCA", "true_concept",
            limits, output_dir / f"pca_{suffix}_true_labels.png", variance,
        )
        _save_scatter(
            coordinates, assignments[representation], representation, "PCA", "matched_predicted_concept",
            limits, output_dir / f"pca_{suffix}_predicted_clusters.png", variance,
        )
        coordinate_records.extend(_coordinate_rows(
            representation, "pca", "not_applicable", coordinates,
            assignments[representation], variance,
        ))

    pca_3d = PCA(n_components=3, svd_solver="full")
    eot_coordinates_3d = pca_3d.fit_transform(embeddings["eot"])
    if eot_coordinates_3d.shape != (400, 3) or not np.isfinite(eot_coordinates_3d).all():
        raise RuntimeError("Invalid EOT 3D PCA coordinates")
    pca_3d_variance = pca_3d.explained_variance_ratio_
    pca_3d_info = {
        "explained_variance_ratio": [float(value) for value in pca_3d_variance],
        "explained_variance_ratio_total": float(pca_3d_variance.sum()),
        "input_shape": list(embeddings["eot"].shape),
        "view_elevation_degrees": 22,
        "view_azimuth_degrees": -55,
    }
    _save_3d_pca(
        eot_coordinates_3d, assignments["eot"], pca_3d_variance,
        output_dir / "pca_eot_3d_true_labels.png",
    )
    pca_3d_frame = assignments["eot"][[
        "sample_index", "candidate_id", "description", "true_concept",
        "predicted_cluster", "matched_predicted_concept", "prediction_correct",
    ]].copy()
    pca_3d_frame = pca_3d_frame.rename(columns={
        "candidate_id": "description_id",
        "predicted_cluster": "raw_cluster_id",
        "matched_predicted_concept": "hungarian_matched_predicted_concept",
    })
    pca_3d_frame["pc1"] = eot_coordinates_3d[:, 0]
    pca_3d_frame["pc2"] = eot_coordinates_3d[:, 1]
    pca_3d_frame["pc3"] = eot_coordinates_3d[:, 2]
    pca_3d_frame.to_csv(output_dir / "pca_eot_3d_coordinates.csv", index=False)

    for representation in REPRESENTATIONS:
        seeds = (0, 1, 42) if representation == "eot" else (42,)
        for seed in seeds:
            model = TSNE(random_state=seed, **TSNE_CONFIG)
            coordinates = model.fit_transform(embeddings[representation])
            if coordinates.shape != (400, 2) or not np.isfinite(coordinates).all():
                raise RuntimeError(f"Invalid {representation} t-SNE coordinates at seed {seed}")
            tsne_coordinates[(representation, seed)] = coordinates
            coordinate_records.extend(_coordinate_rows(
                representation, "tsne", str(seed), coordinates,
                assignments[representation], None,
            ))

        main = tsne_coordinates[(representation, 42)]
        limits = _padded_limits(main)
        axis_limits[f"tsne_{representation}_seed_42"] = limits
        suffix = "eot" if representation == "eot" else "fixed"
        _save_scatter(
            main, assignments[representation], representation, "t-SNE", "true_concept",
            limits, output_dir / f"tsne_{suffix}_true_labels.png",
        )
        _save_scatter(
            main, assignments[representation], representation, "t-SNE", "matched_predicted_concept",
            limits, output_dir / f"tsne_{suffix}_predicted_clusters.png",
        )

    robustness_limits = _save_tsne_robustness(
        {seed: tsne_coordinates[("eot", seed)] for seed in (0, 1, 42)},
        assignments["eot"], output_dir / "tsne_eot_seed_robustness.png",
    )
    axis_limits["tsne_eot_seed_robustness"] = robustness_limits

    coordinates_frame = pd.DataFrame(coordinate_records)
    coordinates_frame.to_csv(output_dir / "text_space_visualization_coordinates.csv", index=False)
    expected_coordinate_rows = 400 * (2 + 3 + 1)
    if len(coordinates_frame) != expected_coordinate_rows:
        raise RuntimeError(f"Expected {expected_coordinate_rows} coordinate rows, found {len(coordinates_frame)}")

    source_audit = _analysis_source_audit()
    config = {
        "experiment": "balanced_text_space_visualization",
        "dataset_path": audit["dataset_path"],
        "dataset_sha256": audit["dataset_sha256"],
        "concept_order": CONCEPTS,
        "animal_colors": ANIMAL_COLORS,
        "animal_markers": ANIMAL_MARKERS,
        "representations": {
            "eot": "Unsuffixed EOT; original description; actual EOT hidden state",
            "fixed_suffix": "Exact suffix ' This sentence describes the concept'; contextual concept-token hidden state",
        },
        "preprocessing": "row L2 normalization; no centering; all 768 input dimensions retained",
        "pca": {"n_components": 2, "svd_solver": "full", **pca_info},
        "pca_3d_eot": {"n_components": 3, "svd_solver": "full", **pca_3d_info},
        "tsne": {**TSNE_CONFIG, "main_random_state": 42, "eot_robustness_random_states": [0, 1, 42]},
        "axis_limits": axis_limits,
        "existing_assignments_reused": True,
        "existing_hungarian_mapping_reused": True,
        "clustering_refit": False,
        "w0_projection_applied": False,
        "oce_run": False,
        "source_audit": source_audit,
        "package_versions": package_versions(),
    }
    atomic_write_text(output_dir / "visualization_config.json", json.dumps(config, indent=2) + "\n")
    _write_summary(output_dir, audit, pca_info, pca_3d_info)

    missing = [name for name in REQUIRED_PNGS if not (output_dir / name).exists()]
    missing.extend(
        name for name in [
            "dataset_audit.json", "visualization_config.json",
            "text_space_visualization_coordinates.csv", "pca_eot_3d_coordinates.csv",
            "visualization_summary.md",
        ] if not (output_dir / name).exists()
    )
    if missing:
        raise RuntimeError(f"Visualization output is incomplete: {missing}")
    return {
        "output_directory": str(output_dir),
        "dataset_sha256": audit["dataset_sha256"],
        "coordinate_rows": len(coordinates_frame),
        "pca_explained_variance": pca_info,
        "png_count": len(REQUIRED_PNGS),
        "pdf_count": sum((output_dir / name).with_suffix(".pdf").exists() for name in REQUIRED_PNGS),
        "clustering_refit": False,
        "w0_projection_applied": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize cached final balanced 8x50 text-space embeddings without refitting clustering."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Final balanced 8x50 output directory or its accepted_descriptions.jsonl path.",
    )
    parser.add_argument("--output", required=True, help="New isolated visualization output directory.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace files only inside the selected derived visualization output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_visualization(args.dataset, args.output, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
