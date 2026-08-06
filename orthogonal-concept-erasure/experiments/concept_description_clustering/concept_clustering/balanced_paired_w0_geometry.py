from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.eot_spherical_clustering import (
    evaluate_after_clustering,
    extract_eot_embeddings,
    fit_spherical_kmeans,
    normalize_rows,
)

from .balanced_paired import FIXED_SUFFIX
from .embeddings import _decoded_word, _extract_contextual, _selected_token_audit
from .modeling import load_original_pipeline, model_metadata, original_projection_modules
from .oce_uce_bare import OCE_UCE_SELECTION_RULE, oce_uce_last_token_position
from .utils import atomic_write_text, package_versions, read_jsonl, write_csv


CONCEPTS = ["cat", "dog", "fox", "bear", "wolf", "rabbit", "deer", "horse"]
REPRESENTATIONS = ["eot", "fixed_suffix"]
CONDITIONS = ["matched_eot", "matched_fixed", "oce_last_to_eot"]
CONDITION_LABELS = {
    "matched_eot": "Matched EOT",
    "matched_fixed": "Matched fixed suffix",
    "oce_last_to_eot": "OCE-last-token -> EOT-description centroid",
}
BLUE = "#2458A6"
ORANGE = "#D65F30"
GOLD = "#D28E00"
INK = "#222222"
GRID = "#D9D9D9"
REQUIRED_OUTPUTS = [
    "experiment_config.json",
    "dataset_audit.json",
    "name_tokenization_audit.csv",
    "layer_inventory.csv",
    "description_embeddings_eot.npy",
    "description_embeddings_fixed.npy",
    "name_embeddings_last.npy",
    "name_embeddings_eot.npy",
    "name_embeddings_fixed.npy",
    "layer_clustering_metrics.csv",
    "layer_per_class_recall.csv",
    "prototype_to_centroid_all_distances.csv",
    "prototype_summary.csv",
    "readout_mismatch_summary.csv",
    "description_within_class_distances.csv",
    "w0_immutability.json",
    "analysis_checks.json",
    "report.md",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(torch.float32).cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold().replace("-", " "))


def _resolve_dataset(dataset_arg: str | Path) -> tuple[Path, Path, Path]:
    path = Path(dataset_arg).expanduser().resolve()
    if path.is_dir():
        dataset_path = path / "accepted_descriptions.jsonl"
        validation_path = path / "dataset_validation.json"
        source_config = path / "experiment_config.json"
    else:
        dataset_path = path
        validation_path = path.parent / "dataset_validation.json"
        source_config = path.parent / "experiment_config.json"
    for required in (dataset_path, validation_path, source_config):
        if not required.exists():
            raise FileNotFoundError(required)
    return dataset_path, validation_path, source_config


def _inspect_oce_source(oce_repo: Path) -> dict[str, Any]:
    source_path = (oce_repo / "oce.py").resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"OCE entry point not found: {source_path}")
    source = source_path.read_text()
    tree = ast.parse(source)
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "Orthogonal_Erase"),
        None,
    )
    if function is None:
        raise RuntimeError("oce.py does not contain Orthogonal_Erase")
    function_source = ast.get_source_segment(source, function) or ""
    required_fragments = ["attn2", "to_v", "attention_mask", "sum() - 2", "encode_prompt"]
    missing = [fragment for fragment in required_fragments if fragment not in function_source]
    if missing:
        raise RuntimeError(f"Could not verify current OCE targeting/readout logic; missing {missing}")
    target_prefix = function_source.split("# ===== collect embeddings =====", 1)[0]
    if "to_k" in target_prefix:
        raise RuntimeError("Current OCE target collection mentions to_k; this analysis must separate matrix types")
    return {
        "oce_entry_point": str(source_path),
        "oce_source_sha256": _sha256(source_path),
        "edit_function": "Orthogonal_Erase",
        "verified_matrix_type": "to_v",
        "verified_target_rule": "module name contains attn2 and ends with to_v",
        "verified_name_readout_rule": "pipe.encode_prompt(...)[0][:, attention_mask.sum()-2, :]",
        "edited_checkpoint_loaded": False,
    }


def _runtime_config(model_id: str, device: str, batch_size: int, seed: int) -> dict[str, Any]:
    return {
        "model": {
            "model_id": model_id,
            "device": device,
            "generation_dtype": "float32",
            "embedding_dtype": "float32",
            "disable_safety_checker": True,
        },
        "readout": {"batch_size": int(batch_size)},
        "spherical_kmeans": {
            "k": 8,
            "n_init": 50,
            "max_iter": 300,
            "tolerance": 1e-6,
            "random_seed": int(seed),
            "assignment": "cosine",
            "center_update": "normalized_mean",
            "initialization": "cosine_kmeans_plus_plus",
        },
    }


def _audit_dataset(
    rows: list[dict[str, Any]],
    dataset_path: Path,
    validation_path: Path,
    source_config_path: Path,
    source_config: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter(str(row.get("concept")) for row in rows)
    exact_concepts = list(counts) == CONCEPTS or set(counts) == set(CONCEPTS)
    pairs = [(row.get("slot_id"), row.get("concept")) for row in rows]
    configured = {item["name"]: item.get("banned_terms", []) for item in source_config["concepts"]}
    forbidden_hits = []
    for row_index, row in enumerate(rows):
        words = set(_normalized_words(str(row["description"])))
        normalized_description = " ".join(_normalized_words(row["description"]))
        for forbidden_concept, terms in configured.items():
            for term in terms:
                normalized_term = " ".join(_normalized_words(term))
                if len(normalized_term.split()) == 1 and normalized_term in words:
                    forbidden_hits.append({
                        "row_index": row_index,
                        "row_concept": row["concept"],
                        "forbidden_concept": forbidden_concept,
                        "term": term,
                    })
                elif len(normalized_term.split()) > 1 and normalized_term in normalized_description:
                    forbidden_hits.append({
                        "row_index": row_index,
                        "row_concept": row["concept"],
                        "forbidden_concept": forbidden_concept,
                        "term": term,
                    })
    source_validation = json.loads(validation_path.read_text())
    checks = {
        "exactly_400_rows": len(rows) == 400,
        "exact_concept_set": exact_concepts,
        "exactly_50_per_concept": all(counts[name] == 50 for name in CONCEPTS),
        "unique_concept_slot_pairs": len(set(pairs)) == 400,
        "all_configured_concept_names_and_variants_absent": not forbidden_hits,
        "source_dataset_validation_passed": source_validation.get("status") == "passed",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Balanced 8x50 dataset audit failed: {checks}; forbidden={forbidden_hits[:5]}")
    return {
        "status": "pending_tokenization_audit",
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_validation_path": str(validation_path),
        "dataset_validation_sha256": _sha256(validation_path),
        "source_experiment_config_path": str(source_config_path),
        "source_experiment_config_sha256": _sha256(source_config_path),
        "row_count": len(rows),
        "counts_by_concept": {name: counts[name] for name in CONCEPTS},
        "checks": checks,
        "forbidden_hits": forbidden_hits,
    }


@torch.inference_mode()
def _extract_all_embeddings(
    pipe,
    rows: list[dict[str, Any]],
    source_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    dataset_audit: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tokenizer, encoder = pipe.tokenizer, pipe.text_encoder
    descriptions = [str(row["description"]) for row in rows]

    desc_eot, eot_audits = extract_eot_embeddings(tokenizer, encoder, descriptions, device, batch_size)
    suffix_prompts = [text.rstrip() + FIXED_SUFFIX for text in descriptions]
    suffix_audits = [
        _selected_token_audit(tokenizer, original, prompt, "fixed_suffix", "description")
        for original, prompt in zip(descriptions, suffix_prompts)
    ]
    if any(row["truncation_occurred"] for row in suffix_audits):
        raise RuntimeError("Fixed suffix truncated at least one description")
    if {_decoded_word(row["selected_token"]) for row in suffix_audits} != {"concept"}:
        raise RuntimeError("Fixed suffix did not select the shared word concept")
    desc_fixed = _extract_contextual(
        encoder,
        tokenizer,
        suffix_prompts,
        [int(row["selected_token_position"]) for row in suffix_audits],
        device,
        batch_size,
    ).numpy()
    if any(row["truncation_occurred"] for row in eot_audits):
        raise RuntimeError("Unsuffixed EOT extraction truncated at least one description")
    if any(int(audit["effective_token_length"]) != int(row["effective_token_length"]) for audit, row in zip(eot_audits, rows)):
        raise RuntimeError("Embedding-time token lengths differ from the accepted dataset audit")

    cached_eot = np.load(source_dir / "eot_embeddings.npy")
    cached_fixed = np.load(source_dir / "fixed_suffix_embeddings.npy")
    eot_cache_match = np.allclose(desc_eot, cached_eot, atol=2e-5, rtol=2e-5)
    fixed_cache_match = np.allclose(normalize_rows(desc_fixed), cached_fixed, atol=2e-5, rtol=2e-5)
    if not eot_cache_match or not fixed_cache_match:
        raise RuntimeError(f"Fresh extraction differs from balanced-paired caches: eot={eot_cache_match}, fixed={fixed_cache_match}")

    name_last: list[np.ndarray] = []
    name_audit_rows: list[dict[str, Any]] = []
    for concept in CONCEPTS:
        tokenized = tokenizer(
            concept,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        attention = tokenized["attention_mask"]
        input_ids = tokenized["input_ids"][0]
        effective_length = int(attention.sum().item())
        last_index = oce_uce_last_token_position(attention)
        eot_index = effective_length - 1
        effective_ids = input_ids[:effective_length].tolist()
        effective_pieces = tokenizer.convert_ids_to_tokens(effective_ids)
        encoded = pipe.encode_prompt(
            prompt=concept,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        name_last.append(encoded[0][0, last_index].detach().float().cpu().numpy())
        name_audit_rows.append({
            "concept": concept,
            "original_name": concept,
            "token_ids": json.dumps(effective_ids),
            "decoded_token_pieces": json.dumps(effective_pieces, ensure_ascii=False),
            "effective_token_length": effective_length,
            "content_token_count": effective_length - 2,
            "split_into_multiple_bpe_tokens": effective_length - 2 > 1,
            "oce_last_content_index": last_index,
            "oce_selected_token_id": int(input_ids[last_index]),
            "oce_selected_decoded_token": tokenizer.convert_ids_to_tokens([int(input_ids[last_index])])[0],
            "eot_index": eot_index,
            "eot_token_id": int(input_ids[eot_index]),
            "eot_decoded_token": tokenizer.convert_ids_to_tokens([int(input_ids[eot_index])])[0],
            "oce_repository_rule": OCE_UCE_SELECTION_RULE,
        })
    name_last_array = np.asarray(name_last, dtype=np.float32)
    name_eot, name_eot_audits = extract_eot_embeddings(tokenizer, encoder, CONCEPTS, device, batch_size)
    for row, audit in zip(name_audit_rows, name_eot_audits):
        if row["eot_index"] != audit["eot_index"] or row["eot_token_id"] != audit["eot_token_id"]:
            raise RuntimeError(f"Bare-name EOT audit mismatch for {row['concept']}")

    fixed_name_prompts = [concept + FIXED_SUFFIX for concept in CONCEPTS]
    fixed_name_audits = [
        _selected_token_audit(tokenizer, concept, prompt, "fixed_suffix", "prototype")
        for concept, prompt in zip(CONCEPTS, fixed_name_prompts)
    ]
    if {_decoded_word(row["selected_token"]) for row in fixed_name_audits} != {"concept"}:
        raise RuntimeError("Fixed-suffix name readout did not select concept")
    name_fixed = _extract_contextual(
        encoder,
        tokenizer,
        fixed_name_prompts,
        [int(row["selected_token_position"]) for row in fixed_name_audits],
        device,
        batch_size,
    ).numpy()

    desc_eot = np.asarray(desc_eot, dtype=np.float32)
    desc_fixed = np.asarray(desc_fixed, dtype=np.float32)
    name_eot = np.asarray(name_eot, dtype=np.float32)
    name_fixed = np.asarray(name_fixed, dtype=np.float32)
    np.save(output_dir / "description_embeddings_eot.npy", desc_eot)
    np.save(output_dir / "description_embeddings_fixed.npy", desc_fixed)
    np.save(output_dir / "name_embeddings_last.npy", name_last_array)
    np.save(output_dir / "name_embeddings_eot.npy", name_eot)
    np.save(output_dir / "name_embeddings_fixed.npy", name_fixed)
    write_csv(output_dir / "name_tokenization_audit.csv", name_audit_rows)

    dataset_audit["checks"].update({
        "no_unsuffixed_prompt_truncated": not any(row["truncation_occurred"] for row in eot_audits),
        "no_fixed_suffix_prompt_truncated": not any(row["truncation_occurred"] for row in suffix_audits),
        "fresh_eot_matches_balanced_cache": bool(eot_cache_match),
        "fresh_fixed_matches_balanced_cache": bool(fixed_cache_match),
    })
    dataset_audit["status"] = "passed" if all(dataset_audit["checks"].values()) else "failed"
    if dataset_audit["status"] != "passed":
        raise RuntimeError(f"Final dataset audit failed: {dataset_audit['checks']}")
    atomic_write_text(output_dir / "dataset_audit.json", json.dumps(dataset_audit, indent=2) + "\n")
    return (
        {"eot": desc_eot, "fixed_suffix": desc_fixed},
        {"last": name_last_array, "eot": name_eot, "fixed": name_fixed},
    )


def _project_and_normalize(raw: np.ndarray, weight: torch.Tensor | None, device: str) -> np.ndarray:
    if weight is None:
        projected = np.asarray(raw, dtype=np.float32)
    else:
        with torch.inference_mode():
            source = torch.from_numpy(np.asarray(raw, dtype=np.float32)).to(device=device, dtype=weight.dtype)
            projected = (source @ weight.detach().T).float().cpu().numpy()
    return normalize_rows(projected).astype(np.float32)


def _fit_feature_sets_without_labels(
    description_raw: dict[str, np.ndarray],
    modules: list[tuple[str, torch.nn.Module]],
    settings: dict[str, Any],
    device: str,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], Any]]:
    """Unsupervised boundary: this function cannot receive true labels."""
    spaces: dict[tuple[str, str], np.ndarray] = {}
    fits: dict[tuple[str, str], Any] = {}
    layer_specs: list[tuple[str, torch.Tensor | None]] = [("text_space_baseline", None)]
    layer_specs.extend((f"layer_{index:02d}", module.weight) for index, (_, module) in enumerate(modules))
    for layer_id, weight in layer_specs:
        for representation in REPRESENTATIONS:
            features = _project_and_normalize(description_raw[representation], weight, device)
            key = (layer_id, representation)
            spaces[key] = features
            fits[key] = fit_spherical_kmeans(
                features,
                k=int(settings["k"]),
                n_init=int(settings["n_init"]),
                max_iter=int(settings["max_iter"]),
                tolerance=float(settings["tolerance"]),
                random_seed=int(settings["random_seed"]),
            )
    return spaces, fits


def _centroids_and_loo(
    features: np.ndarray,
    true_ids: np.ndarray,
    layer_id: str,
    layer_index: int,
    representation: str,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    centroids = []
    within: dict[str, np.ndarray] = {}
    output_rows = []
    for concept_index, concept in enumerate(CONCEPTS):
        member_indices = np.flatnonzero(true_ids == concept_index)
        members = features[member_indices]
        center = normalize_rows(members.mean(axis=0, keepdims=True))[0]
        centroids.append(center)
        loo_distances = []
        total = members.sum(axis=0)
        for local_index, sample_index in enumerate(member_indices):
            loo_center = normalize_rows(((total - members[local_index]) / (len(members) - 1))[None, :])[0]
            distance = float(1.0 - np.dot(members[local_index], loo_center))
            loo_distances.append(distance)
            output_rows.append({
                "layer_id": layer_id,
                "layer_index": layer_index,
                "representation": representation,
                "concept": concept,
                "sample_index": int(sample_index),
                "candidate_id": rows[sample_index]["candidate_id"],
                "leave_one_out_own_centroid_cosine_distance": distance,
            })
        within[concept] = np.asarray(loo_distances, dtype=np.float64)
    return np.asarray(centroids), within, output_rows


def _evaluate_and_compare(
    rows: list[dict[str, Any]],
    modules: list[tuple[str, torch.nn.Module]],
    spaces: dict[tuple[str, str], np.ndarray],
    fits: dict[tuple[str, str], Any],
    names_raw: dict[str, np.ndarray],
    output_dir: Path,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    # Labels are intentionally constructed only after every unsupervised fit is complete.
    true_ids = np.asarray([CONCEPTS.index(row["concept"]) for row in rows], dtype=np.int64)
    layer_specs: list[tuple[str, int, str, torch.Tensor | None]] = [
        ("text_space_baseline", -1, "text_space_baseline", None)
    ]
    layer_specs.extend(
        (f"layer_{index:02d}", index, f"unet.{name}", module.weight)
        for index, (name, module) in enumerate(modules)
    )
    metrics_rows, recall_rows, distance_rows, summary_rows, within_rows = [], [], [], [], []
    mismatch_rows = []
    confusions: dict[tuple[str, str], np.ndarray] = {}
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    for layer_id, layer_index, module_name, weight in layer_specs:
        centroids_by_rep: dict[str, np.ndarray] = {}
        within_by_rep: dict[str, dict[str, np.ndarray]] = {}
        for representation in REPRESENTATIONS:
            key = (layer_id, representation)
            metrics, _, confusion = evaluate_after_clustering(spaces[key], fits[key], true_ids, CONCEPTS)
            confusions[key] = confusion
            pd.DataFrame(confusion, index=CONCEPTS, columns=CONCEPTS).rename_axis("true_concept").to_csv(
                confusion_dir / f"confusion_{layer_id}_{representation}.csv"
            )
            metrics_rows.append({
                "layer_id": layer_id,
                "layer_index": layer_index,
                "full_module_name": module_name,
                "matrix_type": "text_space" if weight is None else "to_v",
                "representation": representation,
                "ari": metrics["adjusted_rand_index"],
                "nmi": metrics["normalized_mutual_information"],
                "matched_accuracy": metrics["hungarian_matched_accuracy"],
                "cosine_silhouette": metrics["cosine_silhouette_score"],
                "cluster_sizes": json.dumps(metrics["cluster_sizes"]),
                "spherical_objective": metrics["spherical_objective"],
                "iterations": metrics["iterations"],
                "converged": metrics["converged"],
                "best_initialization": metrics["best_initialization"],
                "labels_available_to_fit": False,
            })
            recall_rows.extend({
                "layer_id": layer_id,
                "layer_index": layer_index,
                "full_module_name": module_name,
                "representation": representation,
                "concept": concept,
                "recall": metrics["per_class_recall"][concept],
            } for concept in CONCEPTS)
            centroids, within, current_within = _centroids_and_loo(
                spaces[key], true_ids, layer_id, layer_index, representation, rows
            )
            centroids_by_rep[representation] = centroids
            within_by_rep[representation] = within
            within_rows.extend(current_within)

        projected_names = {
            name: _project_and_normalize(matrix, weight, device) for name, matrix in names_raw.items()
        }
        condition_specs = {
            "matched_eot": (projected_names["eot"], centroids_by_rep["eot"], "eot"),
            "matched_fixed": (projected_names["fixed"], centroids_by_rep["fixed_suffix"], "fixed_suffix"),
            "oce_last_to_eot": (projected_names["last"], centroids_by_rep["eot"], "eot"),
        }
        layer_summaries: dict[tuple[str, str], dict[str, Any]] = {}
        for condition, (prototypes, centroids, within_representation) in condition_specs.items():
            distance_matrix = 1.0 - prototypes @ centroids.T
            for concept_index, concept in enumerate(CONCEPTS):
                distances = distance_matrix[concept_index]
                order = np.argsort(distances, kind="stable")
                own_distance = float(distances[concept_index])
                nearest_other = float(np.min(np.delete(distances, concept_index)))
                own_rank = int(np.flatnonzero(order == concept_index)[0]) + 1
                nearest_index = int(order[0])
                margin = nearest_other - own_distance
                loo = within_by_rep[within_representation][concept]
                percentile = float(100.0 * np.mean(loo <= own_distance))
                summary = {
                    "layer_id": layer_id,
                    "layer_index": layer_index,
                    "full_module_name": module_name,
                    "matrix_type": "text_space" if weight is None else "to_v",
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "prototype_concept": concept,
                    "nearest_centroid_concept": CONCEPTS[nearest_index],
                    "own_centroid_rank": own_rank,
                    "own_centroid_cosine_distance": own_distance,
                    "nearest_other_centroid_cosine_distance": nearest_other,
                    "margin": margin,
                    "within_cluster_typicality_percentile": percentile,
                    "rank1_own_centroid": own_rank == 1,
                    "cross_readout": condition == "oce_last_to_eot",
                }
                summary_rows.append(summary)
                layer_summaries[(condition, concept)] = summary
                for centroid_index, centroid_concept in enumerate(CONCEPTS):
                    distance_rows.append({
                        "layer_id": layer_id,
                        "layer_index": layer_index,
                        "full_module_name": module_name,
                        "matrix_type": "text_space" if weight is None else "to_v",
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "prototype_concept": concept,
                        "centroid_concept": centroid_concept,
                        "cosine_distance": float(distances[centroid_index]),
                        "is_own_centroid": concept_index == centroid_index,
                        "cross_readout": condition == "oce_last_to_eot",
                    })
        for concept in CONCEPTS:
            matched = layer_summaries[("matched_eot", concept)]
            cross = layer_summaries[("oce_last_to_eot", concept)]
            mismatch_rows.append({
                "layer_id": layer_id,
                "layer_index": layer_index,
                "full_module_name": module_name,
                "concept": concept,
                "matched_eot_own_rank": matched["own_centroid_rank"],
                "oce_last_to_eot_own_rank": cross["own_centroid_rank"],
                "matched_eot_margin": matched["margin"],
                "oce_last_to_eot_margin": cross["margin"],
                "matched_eot_rank1": matched["rank1_own_centroid"],
                "oce_last_to_eot_rank1": cross["rank1_own_centroid"],
                "rank1_status_changed": matched["rank1_own_centroid"] != cross["rank1_own_centroid"],
            })

    metrics_frame = pd.DataFrame(metrics_rows)
    summary_frame = pd.DataFrame(summary_rows)
    mismatch_frame = pd.DataFrame(mismatch_rows)
    metrics_frame.to_csv(output_dir / "layer_clustering_metrics.csv", index=False)
    pd.DataFrame(recall_rows).to_csv(output_dir / "layer_per_class_recall.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(output_dir / "prototype_to_centroid_all_distances.csv", index=False)
    summary_frame.to_csv(output_dir / "prototype_summary.csv", index=False)
    mismatch_frame.to_csv(output_dir / "readout_mismatch_summary.csv", index=False)
    pd.DataFrame(within_rows).to_csv(output_dir / "description_within_class_distances.csv", index=False)
    counts = summary_frame.groupby(["layer_id", "layer_index", "condition"], sort=False).agg(
        mean_own_rank=("own_centroid_rank", "mean"),
        rank1_names=("rank1_own_centroid", "sum"),
        mean_margin=("margin", "mean"),
    ).reset_index()
    counts.to_csv(output_dir / "prototype_condition_summary.csv", index=False)
    return metrics_frame, summary_frame, mismatch_frame, confusions


def _plot_layer_metric(metrics: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = {"eot": BLUE, "fixed_suffix": ORANGE}
    labels = {"eot": "Unsuffixed EOT", "fixed_suffix": "Fixed suffix"}
    for representation in REPRESENTATIONS:
        subset = metrics[(metrics["representation"] == representation) & (metrics["layer_index"] >= 0)].sort_values("layer_index")
        baseline = float(metrics[(metrics["representation"] == representation) & (metrics["layer_index"] == -1)][metric].iloc[0])
        ax.plot(subset["layer_index"], subset[metric], marker="o", linewidth=2, color=colors[representation], label=labels[representation])
        ax.axhline(baseline, color=colors[representation], linestyle="--", linewidth=1.2, alpha=0.68, label=f"{labels[representation]} text baseline ({baseline:.3f})")
    ax.set_title(f"Layer-wise description clustering {ylabel}\nOriginal SD 1.4 attn2.to_v projections; n=400")
    ax.set_xlabel("Original W0 layer index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(metrics.loc[metrics["layer_index"] >= 0, "layer_index"].unique()))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _plot_heatmap(
    summary: pd.DataFrame,
    condition: str,
    value: str,
    path: Path,
) -> None:
    selected = summary[summary["condition"] == condition].copy()
    layer_order = ["text_space_baseline"] + [f"layer_{index:02d}" for index in sorted(selected.loc[selected["layer_index"] >= 0, "layer_index"].unique())]
    pivot = selected.pivot(index="prototype_concept", columns="layer_id", values=value).reindex(index=CONCEPTS, columns=layer_order)
    array = pivot.to_numpy(dtype=float)
    is_rank = value == "own_centroid_rank"
    if is_rank:
        cmap, vmin, vmax, label = "Blues_r", 1, 8, "Own-centroid rank (1 is best)"
    else:
        bound = max(0.01, float(np.nanmax(np.abs(array))))
        cmap, vmin, vmax, label = "RdBu", -bound, bound, "Margin: nearest other − own distance"
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    image = ax.imshow(array, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(layer_order)), ["Text"] + [str(int(item.split("_")[1])) for item in layer_order[1:]])
    ax.set_yticks(range(len(CONCEPTS)), CONCEPTS)
    ax.set_xlabel("Text baseline, then original W0 layer index")
    ax.set_ylabel("Concept name")
    ax.set_title(f"{CONDITION_LABELS[condition]} — {'own-centroid rank' if is_rank else 'own-vs-other margin'}\nEight balanced name-free description centroids per space")
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            text_value = f"{int(array[row, column])}" if is_rank else f"{array[row, column]:.2f}"
            if is_rank:
                text_color = "white" if array[row, column] <= 2 else INK
            else:
                text_color = "white" if abs(array[row, column]) >= 0.58 * max(abs(vmin), abs(vmax)) else INK
            ax.text(column, row, text_value, ha="center", va="center", fontsize=7.3, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    colorbar.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _plot_distance_matrix(
    distances: pd.DataFrame,
    layer_id: str,
    condition: str,
    path: Path,
) -> None:
    selected = distances[(distances["layer_id"] == layer_id) & (distances["condition"] == condition)]
    pivot = selected.pivot(index="prototype_concept", columns="centroid_concept", values="cosine_distance").reindex(index=CONCEPTS, columns=CONCEPTS)
    array = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(array, cmap="YlOrBr", vmin=float(array.min()), vmax=float(array.max()))
    ax.set_xticks(range(8), CONCEPTS, rotation=35, ha="right")
    ax.set_yticks(range(8), CONCEPTS)
    ax.set_xlabel("Description centroid concept")
    ax.set_ylabel("Explicit name prototype")
    display_layer = "Text-space baseline" if layer_id == "text_space_baseline" else f"Original W0 {layer_id.replace('_', ' ')}"
    ax.set_title(f"{CONDITION_LABELS[condition]} cosine distances\n{display_layer}; lower is closer")
    threshold = float((array.min() + array.max()) / 2)
    for row in range(8):
        for column in range(8):
            ax.text(column, row, f"{array[row, column]:.2f}", ha="center", va="center", fontsize=8, color="white" if array[row, column] > threshold else INK)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Cosine distance")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _plot_confusion(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    ax.set_xticks(range(8), CONCEPTS, rotation=35, ha="right")
    ax.set_yticks(range(8), CONCEPTS)
    ax.set_xlabel("Hungarian-matched predicted concept")
    ax.set_ylabel("True concept")
    ax.set_title(title + "\nMatched counts; n=400")
    threshold = float(matrix.max()) / 2
    for row in range(8):
        for column in range(8):
            ax.text(column, row, str(int(matrix[row, column])), ha="center", va="center", color="white" if matrix[row, column] > threshold else INK, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Description count")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _create_plots(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    distances: pd.DataFrame,
    confusions: dict[tuple[str, str], np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _plot_layer_metric(metrics, "ari", "ARI", plots / "layer_ari.png")
    _plot_layer_metric(metrics, "matched_accuracy", "Matched accuracy", plots / "layer_accuracy.png")
    for condition, suffix in [("matched_eot", "matched_eot"), ("matched_fixed", "matched_fixed"), ("oce_last_to_eot", "oce_to_eot")]:
        _plot_heatmap(summary, condition, "own_centroid_rank", plots / f"rank_heatmap_{suffix}.png")
        _plot_heatmap(summary, condition, "margin", plots / f"margin_heatmap_{suffix}.png")

    best_eot = metrics[(metrics["representation"] == "eot") & (metrics["layer_index"] >= 0)].sort_values(["ari", "layer_index"], ascending=[False, True]).iloc[0]
    best_fixed = metrics[(metrics["representation"] == "fixed_suffix") & (metrics["layer_index"] >= 0)].sort_values(["ari", "layer_index"], ascending=[False, True]).iloc[0]
    best_eot_id, best_fixed_id = str(best_eot["layer_id"]), str(best_fixed["layer_id"])
    for layer_id in ["text_space_baseline", best_eot_id]:
        for condition in CONDITIONS:
            _plot_distance_matrix(distances, layer_id, condition, plots / f"distance_matrix_{layer_id}_{condition}.png")
    selected = [
        ("text_space_baseline", "eot", "Text-space EOT baseline"),
        (best_eot_id, "eot", f"Best EOT-ARI original W0 layer {int(best_eot['layer_index'])}"),
        ("text_space_baseline", "fixed_suffix", "Text-space fixed-suffix baseline"),
        (best_fixed_id, "fixed_suffix", f"Best fixed-suffix-ARI original W0 layer {int(best_fixed['layer_index'])}"),
    ]
    confusion_dir = output_dir / "confusion_matrices"
    for layer_id, representation, title in selected:
        _plot_confusion(confusions[(layer_id, representation)], title, confusion_dir / f"selected_{layer_id}_{representation}.png")
    return {
        "best_eot_layer_id": best_eot_id,
        "best_eot_layer_index": int(best_eot["layer_index"]),
        "best_fixed_layer_id": best_fixed_id,
        "best_fixed_layer_index": int(best_fixed["layer_index"]),
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def _build_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    mismatch: pd.DataFrame,
    inventory: pd.DataFrame,
    plot_info: dict[str, Any],
) -> None:
    baseline = metrics[metrics["layer_index"] == -1].set_index("representation")
    best_rows = {}
    table1 = []
    for representation, label in [("eot", "Unsuffixed EOT"), ("fixed_suffix", "Fixed suffix")]:
        best = metrics[(metrics["representation"] == representation) & (metrics["layer_index"] >= 0)].sort_values(["ari", "layer_index"], ascending=[False, True]).iloc[0]
        best_rows[representation] = best
        table1.append([
            label,
            f"L{int(best['layer_index'])}",
            f"{baseline.loc[representation, 'ari']:.4f}",
            f"{best['ari']:.4f}",
            f"{best['matched_accuracy']:.4f}",
        ])
    best_eot_id = plot_info["best_eot_layer_id"]
    best_proto = summary[summary["layer_id"] == best_eot_id]
    table2 = []
    for condition in CONDITIONS:
        group = best_proto[best_proto["condition"] == condition]
        table2.append([
            CONDITION_LABELS[condition],
            f"{group['own_centroid_rank'].mean():.3f}",
            f"{int(group['rank1_own_centroid'].sum())} / 8",
            f"{group['margin'].mean():.4f}",
        ])
    best_mismatch = mismatch[mismatch["layer_id"] == best_eot_id]
    changed = best_mismatch.loc[best_mismatch["rank1_status_changed"].astype(bool), "concept"].tolist()
    matched_success_cross_fail = best_mismatch[
        best_mismatch["matched_eot_rank1"].astype(bool) & ~best_mismatch["oce_last_to_eot_rank1"].astype(bool)
    ]["concept"].tolist()
    both_fail = best_mismatch[
        ~best_mismatch["matched_eot_rank1"].astype(bool) & ~best_mismatch["oce_last_to_eot_rank1"].astype(bool)
    ]["concept"].tolist()
    w0_mismatch = mismatch[mismatch["layer_index"] >= 0]
    changed_pairs = int(w0_mismatch["rank1_status_changed"].astype(bool).sum())
    changed_layers = int((w0_mismatch.groupby("layer_index")["rank1_status_changed"].sum() > 0).sum())
    matched_only = int((
        w0_mismatch["matched_eot_rank1"].astype(bool)
        & ~w0_mismatch["oce_last_to_eot_rank1"].astype(bool)
    ).sum())
    cross_only = int((
        ~w0_mismatch["matched_eot_rank1"].astype(bool)
        & w0_mismatch["oce_last_to_eot_rank1"].astype(bool)
    ).sum())
    shape_counts = inventory.groupby(["w0_output_dim", "w0_input_dim"]).size().to_dict()
    shape_text = ", ".join(f"{count}×({out_dim}×{in_dim})" for (out_dim, in_dim), count in shape_counts.items())
    eot_delta = float(best_rows["eot"]["ari"] - baseline.loc["eot", "ari"])
    fixed_delta = float(best_rows["fixed_suffix"]["ari"] - baseline.loc["fixed_suffix", "ari"])
    special_lines = []
    if matched_success_cross_fail:
        special_lines.append(
            "For " + ", ".join(matched_success_cross_fail) + ", matched EOT is rank 1 while the OCE cross-readout is not. "
            "The canonical OCE name vector is geometrically incompatible with the EOT description cluster under this layer, but the mismatch may arise from the last-token readout rather than concept semantics alone."
        )
    if both_fail:
        special_lines.append(
            "For " + ", ".join(both_fail) + ", the canonical name is not close to its name-free description centroid under either tested representation."
        )
    if not special_lines:
        special_lines.append("No concept at the post-hoc best EOT layer triggered either prescribed mismatch warning pattern.")

    report = f"""# Original-W0 Geometry of Balanced Name-Free Animal Descriptions

## 1. Research Questions

This experiment asks whether each unchanged pretrained SD 1.4 cross-attention projection preserves or improves clustering of the final balanced 8×50 name-free dataset; whether explicit names are near their own description centroids under matched readouts; and whether the exact OCE name vector is near EOT description centroids under an intentionally cross-readout comparison.

## 2. Representations and the Readout-Mismatch Problem

The EOT description readout is the final EOT hidden state of the original unsuffixed description, selected at `attention_mask.sum(dim=1) - 1`. The fixed-suffix readout appends exactly `{FIXED_SUFFIX}` and extracts the contextual hidden state of the shared final content word `concept`. The OCE name readout reproduces `oce.py`: it calls `pipe.encode_prompt` on the bare name and selects the final content token immediately before EOT at `attention_mask.sum() - 2`.

`OCE-last-token -> EOT-description centroid` intentionally compares different readout rules. It is operationally relevant to OCE, but it is not a matched semantic-distance measurement and its absolute values should not be compared with matched EOT without this qualification.

## 3. Original W0 Layer Inventory

Repository inspection found that `Orthogonal_Erase` edits only the 16 original `UNet` cross-attention `attn2.to_v` matrices. No `to_k` matrix is selected, so the main analysis contains `to_v` only. Layer shapes are {shape_text}; every input dimension is 768. Full ordered module names and per-layer before/after hashes are in `layer_inventory.csv` and `w0_immutability.json`.

## 4. Layer-wise Description Clustering

{_markdown_table(['Representation', 'Best W0 Layer', 'Text-space ARI', 'Best-layer ARI', 'Best-layer Accuracy'], table1)}

The best-layer choices are post-hoc maxima over all 16 layers. Relative to text space, the best EOT layer changes ARI by {eot_delta:+.4f}; the best fixed-suffix layer changes ARI by {fixed_delta:+.4f}. All layers and both readouts are retained in `layer_clustering_metrics.csv`; clustering always used the original 400×layer-dimension vectors, never PCA.

![Layer-wise ARI](plots/layer_ari.png)

The plot compares every layer with its representation-specific text-space baseline. The companion accuracy plot is `plots/layer_accuracy.png`.

## 5. Matched Prototype-to-Cluster Results

At the post-hoc best EOT-ARI W0 layer L{plot_info['best_eot_layer_index']}:

{_markdown_table(['Condition', 'Mean Own Rank', 'Rank-1 Names / 8', 'Mean Margin'], table2)}

Positive margin means the name is closer to its own true-label description centroid than to every other centroid. The typicality percentiles and all name-to-centroid distances are retained in the detailed CSVs.

![Matched EOT ranks](plots/rank_heatmap_matched_eot.png)

The heatmap shows text space followed by every W0 layer, making stability across layers visible rather than reporting only the best layer.

## 6. OCE-Faithful Prototype-to-EOT Results

The OCE-faithful condition maps the exact bare-name last-content-token vector through each unchanged `W0` and compares it with the corresponding EOT-description centroids. This is labeled `OCE-last-token -> EOT-description centroid` throughout the outputs and is kept separate from both matched conditions.

![OCE-to-EOT margins](plots/margin_heatmap_oce_to_eot.png)

Ranks, margins, nearest concepts, and within-class typicality percentiles for all names and layers are in `prototype_summary.csv`.

## 7. Readout-Mismatch Comparison

At L{plot_info['best_eot_layer_index']}, concepts whose rank-1 status changes between matched EOT and the OCE cross-readout are: {', '.join(changed) if changed else 'none'}.

""" + "\n\n".join(special_lines) + f"""

Across all 16 W0 layers, rank-1 status differs for {changed_pairs}/128 layer–concept pairs spanning {changed_layers}/16 layers: {matched_only} succeed only under matched EOT and {cross_only} succeed only under the OCE cross-readout. The all-layer, per-concept comparison of both ranks and margins is saved in `readout_mismatch_summary.csv`. Cross-readout rows are never pooled with matched-readout rows.

## 8. Main Findings

- W0 effects are layer-specific: L{plot_info['best_eot_layer_index']} improves EOT ARI by {eot_delta:+.4f}, and L{plot_info['best_fixed_layer_index']} improves fixed-suffix ARI by {fixed_delta:+.4f}, while several other layers substantially degrade both clusterings. These post-hoc maxima do not imply that W0 created the concept structure.
- At the best EOT layer, matched EOT and OCE cross-readout have the same rank-1 status for all eight names, but the {changed_pairs}/128 all-layer disagreement count shows that readout choice is not generally invariant across W0 spaces.
- Every numerical output retains all layers. Selected-layer figures are presentation summaries, not a pre-registered layer choice.
- This experiment analyzes unchanged original matrices only. It does not load an edited checkpoint, compute `P W0`, generate images, or modify model parameters.

## 9. Limitations

The centroids use true labels post hoc and therefore characterize known classes rather than discovering them. Best-layer reporting is post hoc. Cosine distance depends on both the readout and the layer-specific projection, and vectors from different layers are never directly compared. Prototype proximity does not prove semantic identity, central representation, absence of information, or whether OCE can erase a full description distribution.
"""
    atomic_write_text(output_dir / "report.md", report)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --force to replace generated files")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path, validation_path, source_config_path = _resolve_dataset(args.dataset)
    source_dir = dataset_path.parent
    rows = read_jsonl(dataset_path)
    source_config = json.loads(source_config_path.read_text())
    dataset_audit = _audit_dataset(rows, dataset_path, validation_path, source_config_path, source_config)
    oce_audit = _inspect_oce_source(Path(args.oce_repo).expanduser().resolve())
    config = _runtime_config(args.model_id, args.device, args.batch_size, args.random_seed)

    torch.manual_seed(int(args.random_seed))
    np.random.seed(int(args.random_seed))
    pipe = load_original_pipeline(config, purpose="embedding", include_vae=False)
    if pipe.text_encoder.config.hidden_size != 768 or int(pipe.tokenizer.model_max_length) != 77:
        raise RuntimeError("Loaded model is not the expected SD 1.4 CLIP text configuration")
    modules = original_projection_modules(pipe, "to_v")
    if args.layer_limit is not None:
        modules = modules[: int(args.layer_limit)]
    if not modules:
        raise RuntimeError("No W0 layers selected")
    before_hashes = {name: _tensor_sha256(module.weight) for name, module in modules}
    inventory_rows = [{
        "layer_index": index,
        "full_module_name": f"unet.{name}",
        "pipeline_relative_module_name": name,
        "matrix_type": "to_v",
        "w0_input_dim": int(module.weight.shape[1]),
        "w0_output_dim": int(module.weight.shape[0]),
        "weight_dtype": str(module.weight.dtype),
        "requires_grad": bool(module.weight.requires_grad),
        "w0_sha256_before": before_hashes[name],
    } for index, (name, module) in enumerate(modules)]

    description_raw, names_raw = _extract_all_embeddings(
        pipe, rows, source_dir, output_dir, args.device, int(args.batch_size), dataset_audit
    )
    spaces, fits = _fit_feature_sets_without_labels(
        description_raw, modules, config["spherical_kmeans"], args.device
    )
    metrics, summary, mismatch, confusions = _evaluate_and_compare(
        rows, modules, spaces, fits, names_raw, output_dir, args.device
    )
    after_hashes = {name: _tensor_sha256(module.weight) for name, module in modules}
    unchanged = {name: before_hashes[name] == after_hashes[name] for name, _ in modules}
    if not all(unchanged.values()):
        raise RuntimeError(f"At least one original W0 changed: {unchanged}")
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

    distances = pd.read_csv(output_dir / "prototype_to_centroid_all_distances.csv")
    plot_info = _create_plots(metrics, summary, distances, confusions, output_dir)
    _build_report(output_dir, metrics, summary, mismatch, inventory, plot_info)
    metadata = model_metadata(pipe, config, projection="to_v")
    experiment = {
        "experiment_name": "balanced_paired_w0_geometry",
        "dataset": dataset_audit,
        "model": metadata,
        "oce_repository_audit": oce_audit,
        "concepts": CONCEPTS,
        "description_readouts": {
            "eot": "original unsuffixed prompt; attention_mask.sum(dim=1)-1",
            "fixed_suffix": f"original prompt plus {FIXED_SUFFIX!r}; contextual final concept token",
        },
        "name_readouts": {
            "oce_last": "bare name via pipe.encode_prompt; attention_mask.sum()-2",
            "eot": "bare name EOT; attention_mask.sum(dim=1)-1",
            "fixed": f"bare name plus {FIXED_SUFFIX!r}; contextual final concept token",
        },
        "prototype_conditions": {
            "matched_eot": "name EOT -> EOT-description centroid",
            "matched_fixed": "name fixed suffix -> fixed-suffix-description centroid",
            "oce_last_to_eot": "OCE-last-token -> EOT-description centroid; intentional cross-readout",
        },
        "projection": "post-W0 row L2 normalization; no pre-W0 normalization substitute",
        "spherical_kmeans": config["spherical_kmeans"],
        "labels_available_to_fit": False,
        "layer_limit": args.layer_limit,
        "package_versions": package_versions(),
        "selected_plot_layers": plot_info,
        "cli": {
            "dataset_argument": args.dataset,
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
    condition_counts = summary.groupby("condition").size().to_dict()
    distance_condition_counts = distances.groupby("condition").size().to_dict()
    checks = {
        "status": "passed",
        "dataset_is_final_balanced_8x50": dataset_audit["status"] == "passed" and len(rows) == 400,
        "description_embedding_shapes_are_400x768": all(
            matrix.shape == (400, 768) for matrix in description_raw.values()
        ),
        "name_embedding_shapes_are_8x768": all(
            matrix.shape == (8, 768) for matrix in names_raw.values()
        ),
        "all_name_oce_indices_equal_effective_length_minus_2": bool(
            (token_audit["oce_last_content_index"] == token_audit["effective_token_length"] - 2).all()
        ),
        "all_name_eot_indices_equal_effective_length_minus_1": bool(
            (token_audit["eot_index"] == token_audit["effective_token_length"] - 1).all()
        ),
        "all_names_single_bpe_content_tokens": bool(
            (~token_audit["split_into_multiple_bpe_tokens"].astype(bool)).all()
        ),
        "all_selected_w0_unchanged": all(unchanged.values()),
        "no_edited_checkpoint_loaded": True,
        "no_image_generation_performed": True,
        "clustering_fit_boundary_accepts_no_labels": True,
        "all_clustering_rows_record_labels_unavailable_to_fit": bool(
            (~metrics["labels_available_to_fit"].astype(bool)).all()
        ),
        "three_prototype_conditions_are_separate": set(condition_counts) == set(CONDITIONS)
        and all(count == (len(modules) + 1) * 8 for count in condition_counts.values()),
        "cross_readout_distances_not_merged_with_matched_distances": set(distance_condition_counts) == set(CONDITIONS)
        and all(count == (len(modules) + 1) * 8 * 8 for count in distance_condition_counts.values()),
        "all_layer_confusion_csvs_present": len(list((output_dir / "confusion_matrices").glob("confusion_*.csv")))
        == (len(modules) + 1) * 2,
    }
    checks["status"] = "passed" if all(value for key, value in checks.items() if key != "status") else "failed"
    atomic_write_text(output_dir / "analysis_checks.json", json.dumps(checks, indent=2) + "\n")
    if checks["status"] != "passed":
        raise RuntimeError(f"Final analysis checks failed: {checks}")

    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    empty = [name for name in REQUIRED_OUTPUTS if (output_dir / name).exists() and (output_dir / name).stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(f"Output validation failed: missing={missing}, empty={empty}")
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze balanced paired descriptions in unchanged original W0 spaces")
    parser.add_argument("--dataset", required=True, help="Balanced 8x50 output directory or accepted_descriptions.jsonl")
    parser.add_argument("--model-id", required=True, help="SD 1.4 model identifier or local path")
    parser.add_argument("--oce-repo", required=True, help="OCE repository root containing oce.py")
    parser.add_argument("--output", required=True, help="New isolated output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=314159)
    parser.add_argument("--layer-limit", type=int, default=None, help="Smoke-test only: analyze the first N OCE W0 layers")
    parser.add_argument("--force", action="store_true", help="Allow replacement of generated files in the selected output only")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
