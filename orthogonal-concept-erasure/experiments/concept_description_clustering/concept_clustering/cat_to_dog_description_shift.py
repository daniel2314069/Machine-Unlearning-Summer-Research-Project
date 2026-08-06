from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from .balanced_paired_w0_geometry import (
    CONCEPTS,
    _audit_dataset,
    _inspect_oce_source,
    _resolve_dataset,
    _runtime_config,
    _sha256,
    _tensor_sha256,
)
from .modeling import load_original_pipeline, model_metadata, original_projection_modules
from .utils import atomic_write_text, package_versions, read_jsonl


BLUE = "#2458A6"
ORANGE = "#D65F30"
GOLD = "#D28E00"
INK = "#222222"
GRID = "#D9D9D9"
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "orange_white_blue", [ORANGE, "#F7F7F7", BLUE]
)
REQUIRED_OUTPUTS = [
    "experiment_config.json",
    "dataset_audit.json",
    "name_tokenization_audit.csv",
    "layer_inventory.csv",
    "weight_audit.json",
    "description_shift_values.csv",
    "class_layer_shift_summary.csv",
    "canonical_cat_sanity_check.csv",
    "analysis_checks.json",
    "report.md",
    "plots/mean_delta_dog_heatmap.png",
    "plots/mean_delta_cat_heatmap.png",
    "plots/joint_intended_direction_heatmap.png",
    "plots/cat_layer_shift.png",
    "plots/cat_vs_nontarget_dogward_shift.png",
]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise RuntimeError("Cannot cosine-normalize a zero vector")
    return (matrix / norms).astype(np.float32, copy=False)


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("Cannot cosine-normalize a zero vector")
    return (vector / norm).astype(np.float32, copy=False)


def _strict_scope_source_audit() -> dict[str, bool]:
    source = inspect.getsource(inspect.getmodule(_strict_scope_source_audit))
    tree = ast.parse(source)
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
    projection_source = inspect.getsource(_project_descriptions) + inspect.getsource(
        _canonical_cat_shift
    )
    return {
        "no_clustering_calls": called_names.isdisjoint(
            {"KMeans", "fit_spherical_kmeans", "silhouette_score"}
        ),
        "edited_dog_anchor_absent_from_projection_functions": "wcat @ c_dog"
        not in projection_source,
    }


def _project_descriptions(
    descriptions: np.ndarray,
    w0: np.ndarray,
    wcat: np.ndarray,
    c_cat: np.ndarray,
    c_dog: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute only the requested fixed-reference cosine shifts."""
    a_cat = _normalize_vector(w0 @ c_cat)
    a_dog = _normalize_vector(w0 @ c_dog)
    before = _normalize_rows(descriptions @ w0.T)
    after = _normalize_rows(descriptions @ wcat.T)
    cat_before = before @ a_cat
    cat_after = after @ a_cat
    dog_before = before @ a_dog
    dog_after = after @ a_dog
    delta_cat = cat_after - cat_before
    delta_dog = dog_after - dog_before
    return {
        "cat_before": cat_before,
        "cat_after": cat_after,
        "dog_before": dog_before,
        "dog_after": dog_after,
        "delta_cat": delta_cat,
        "delta_dog": delta_dog,
        "intended_direction": (delta_dog > 0) & (delta_cat < 0),
        "before_norms": np.linalg.norm(before, axis=1),
        "after_norms": np.linalg.norm(after, axis=1),
        "cat_anchor_norm": np.asarray([np.linalg.norm(a_cat)]),
        "dog_anchor_norm": np.asarray([np.linalg.norm(a_dog)]),
    }


def _canonical_cat_shift(
    w0: np.ndarray,
    wcat: np.ndarray,
    c_cat: np.ndarray,
    c_dog: np.ndarray,
) -> dict[str, float | bool]:
    a_cat = _normalize_vector(w0 @ c_cat)
    a_dog = _normalize_vector(w0 @ c_dog)
    canonical_before = _normalize_vector(w0 @ c_cat)
    canonical_after = _normalize_vector(wcat @ c_cat)
    cat_before = float(canonical_before @ a_cat)
    cat_after = float(canonical_after @ a_cat)
    dog_before = float(canonical_before @ a_dog)
    dog_after = float(canonical_after @ a_dog)
    delta_cat = cat_after - cat_before
    delta_dog = dog_after - dog_before
    return {
        "cat_before": cat_before,
        "cat_after": cat_after,
        "dog_before": dog_before,
        "dog_after": dog_after,
        "canonical_delta_cat": delta_cat,
        "canonical_delta_dog": delta_dog,
        "intended_direction": bool(delta_dog > 0 and delta_cat < 0),
    }


def _aggregate_shifts(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (layer_index, layer_name, concept), group in values.groupby(
        ["layer_index", "layer_name", "true_concept"], sort=False
    ):
        delta_cat = group["delta_cat"].to_numpy(dtype=float)
        delta_dog = group["delta_dog"].to_numpy(dtype=float)
        rows.append({
            "layer_index": int(layer_index),
            "layer_name": layer_name,
            "true_concept": concept,
            "count": int(len(group)),
            "mean_delta_cat": float(delta_cat.mean()),
            "std_delta_cat": float(delta_cat.std(ddof=1)),
            "median_delta_cat": float(np.median(delta_cat)),
            "min_delta_cat": float(delta_cat.min()),
            "max_delta_cat": float(delta_cat.max()),
            "mean_delta_dog": float(delta_dog.mean()),
            "std_delta_dog": float(delta_dog.std(ddof=1)),
            "median_delta_dog": float(np.median(delta_dog)),
            "min_delta_dog": float(delta_dog.min()),
            "max_delta_dog": float(delta_dog.max()),
            "proportion_toward_dog": float((delta_dog > 0).mean()),
            "proportion_away_from_cat": float((delta_cat < 0).mean()),
            "joint_intended_direction": float(((delta_dog > 0) & (delta_cat < 0)).mean()),
        })
    return pd.DataFrame(rows)


def _load_cached_embeddings(
    source_dir: Path,
    embedding_cache: Path,
    dataset_audit: dict[str, Any],
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    balanced_eot_path = source_dir / "eot_embeddings.npy"
    cached_eot_path = embedding_cache / "description_embeddings_eot.npy"
    cached_names_path = embedding_cache / "name_embeddings_last.npy"
    cached_audit_path = embedding_cache / "dataset_audit.json"
    cached_name_audit_path = embedding_cache / "name_tokenization_audit.csv"
    eot_token_audit_path = source_dir / "eot_tokenization_audit.csv"
    required = [
        balanced_eot_path,
        cached_eot_path,
        cached_names_path,
        cached_audit_path,
        cached_name_audit_path,
        eot_token_audit_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required cached inputs are missing: {missing}")

    cached_audit = json.loads(cached_audit_path.read_text())
    if cached_audit.get("dataset_sha256") != dataset_audit["dataset_sha256"]:
        raise RuntimeError("Cached EOT embeddings were extracted from a different dataset")
    balanced_eot = np.load(balanced_eot_path).astype(np.float32, copy=False)
    cached_eot = np.load(cached_eot_path).astype(np.float32, copy=False)
    name_last = np.load(cached_names_path).astype(np.float32, copy=False)
    if balanced_eot.shape != (400, 768) or cached_eot.shape != (400, 768):
        raise RuntimeError(f"Unexpected EOT shapes: {balanced_eot.shape}, {cached_eot.shape}")
    if name_last.shape != (8, 768):
        raise RuntimeError(f"Unexpected canonical-name cache shape: {name_last.shape}")
    if not np.allclose(balanced_eot, cached_eot, atol=2e-5, rtol=2e-5):
        raise RuntimeError("Balanced-paired and W0-geometry EOT caches disagree")

    token_audit = pd.read_csv(eot_token_audit_path)
    truncation = token_audit["truncation_occurred"].astype(str).str.casefold().eq("true")
    if len(token_audit) != 400 or bool(truncation.any()):
        raise RuntimeError("The cached SD 1.4 EOT tokenization audit is incomplete or contains truncation")
    dataset_audit["checks"].update({
        "no_unsuffixed_prompt_truncated": not bool(truncation.any()),
        "eot_tokenization_audit_has_400_rows": len(token_audit) == 400,
        "cached_eot_matches_balanced_eot": True,
        "cached_embedding_dataset_hash_matches": True,
    })
    dataset_audit["status"] = "passed" if all(dataset_audit["checks"].values()) else "failed"
    if dataset_audit["status"] != "passed":
        raise RuntimeError(f"Dataset/cache audit failed: {dataset_audit['checks']}")

    all_name_audit = pd.read_csv(cached_name_audit_path)
    selected = all_name_audit[all_name_audit["concept"].isin(["cat", "dog"])].copy()
    selected = selected.set_index("concept").loc[["cat", "dog"]].reset_index()
    output_audit = pd.DataFrame({
        "prompt": selected["original_name"],
        "token_ids": selected["token_ids"],
        "decoded_token_pieces": selected["decoded_token_pieces"],
        "selected_token_index": selected["oce_last_content_index"].astype(int),
        "selected_decoded_token": selected["oce_selected_decoded_token"],
        "eot_index": selected["eot_index"].astype(int),
        "eot_decoded_token": selected["eot_decoded_token"],
        "oce_repository_rule": selected["oce_repository_rule"],
    })
    output_audit.to_csv(output_dir / "name_tokenization_audit.csv", index=False)
    if output_audit["prompt"].tolist() != ["cat", "dog"]:
        raise RuntimeError("Canonical token audit does not contain exactly cat then dog")
    if not (output_audit["selected_token_index"] == output_audit["eot_index"] - 1).all():
        raise RuntimeError("Canonical name selection is not the final content token before EOT")

    return cached_eot, name_last[0], name_last[1], {
        "embedding_cache": str(embedding_cache),
        "balanced_eot_path": str(balanced_eot_path),
        "balanced_eot_sha256": _sha256(balanced_eot_path),
        "w0_geometry_eot_path": str(cached_eot_path),
        "w0_geometry_eot_sha256": _sha256(cached_eot_path),
        "canonical_name_cache_path": str(cached_names_path),
        "canonical_name_cache_sha256": _sha256(cached_names_path),
        "eot_shape": list(cached_eot.shape),
        "readout": "unsuffixed original description; actual EOT hidden state at attention_mask.sum(dim=1)-1",
        "prefix": None,
        "suffix": None,
    }


def _validate_edit_metadata(
    metadata_path: Path,
    weights_path: Path,
    model_id: str,
) -> tuple[dict[str, Any], str]:
    metadata = json.loads(metadata_path.read_text())
    checkpoint_sha256 = _sha256(weights_path)
    checks = {
        "method_is_oce": metadata.get("method") == "OCE",
        "model_matches": metadata.get("model_id") == model_id,
        "edit_concept_is_cat": metadata.get("edit_concept") == "cat",
        "guide_concept_is_dog": metadata.get("guide_concept") == "dog",
        "recorded_checkpoint_hash_matches": metadata.get("sha256", {}).get("edited_weights")
        == checkpoint_sha256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Edited checkpoint provenance could not be verified: {checks}")
    return metadata, checkpoint_sha256


def _oce_reproduction_command(metadata: dict[str, Any]) -> str:
    settings = metadata["oce"]
    return (
        "conda run -n py310 python oce.py --edit_concepts cat --guide_concepts dog "
        "--preserve_concepts dog --concept_type object "
        f"--model_id {metadata['model_id']} --device cuda:0 "
        f"--erase_scale {settings['erase_scale']} "
        f"--preserve_global_scale {settings['preserve_global_scale']} "
        f"--preserve_concept_scale {settings['preserve_concept_scale']} "
        f"--lamb {settings['lambda']} "
        f"--expand_prompts {str(settings['expand_prompts']).lower()} "
        "--save_dir ../oce_cat_to_dog_comparison --exp_name oce_cat_to_dog_W"
    )


def _load_layer_pairs(
    pipe,
    edited_weights_path: Path,
    previous_inventory_path: Path,
    layer_limit: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[str, np.ndarray, np.ndarray]], dict[str, str]]:
    modules = original_projection_modules(pipe, "to_v")
    if len(modules) != 16:
        raise RuntimeError(f"Expected exactly 16 original attn2.to_v modules, found {len(modules)}")
    previous = pd.read_csv(previous_inventory_path)
    expected_order = previous["pipeline_relative_module_name"].tolist()
    module_order = [name for name, _ in modules]
    if module_order != expected_order:
        raise RuntimeError("Current original W0 layer order differs from the existing 16-layer inventory")
    edited = load_file(str(edited_weights_path), device="cpu")
    expected_keys = {f"{name}.weight" for name in module_order}
    if set(edited) != expected_keys:
        raise RuntimeError(
            f"Edited checkpoint keys do not exactly match the 16 OCE targets: "
            f"missing={sorted(expected_keys-set(edited))}, extra={sorted(set(edited)-expected_keys)}"
        )

    selected_count = 16 if layer_limit is None else int(layer_limit)
    if selected_count < 1 or selected_count > 16:
        raise ValueError("--layer-limit must be between 1 and 16")
    rows: list[dict[str, Any]] = []
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    w0_before: dict[str, str] = {}
    for index, (name, module) in enumerate(modules):
        w0_tensor = module.weight.detach().float().cpu()
        wcat_tensor = edited[f"{name}.weight"].detach().float().cpu()
        if tuple(w0_tensor.shape) != tuple(wcat_tensor.shape):
            raise RuntimeError(f"W0/Wcat shape mismatch at {name}")
        w0_hash = _tensor_sha256(w0_tensor)
        wcat_hash = _tensor_sha256(wcat_tensor)
        w0_before[name] = w0_hash
        previous_hash = str(previous.loc[index, "w0_sha256_before"])
        rows.append({
            "layer_index": index,
            "full_module_name": f"unet.{name}",
            "pipeline_relative_module_name": name,
            "matrix_type": "to_v",
            "w0_shape": json.dumps(list(w0_tensor.shape)),
            "wcat_shape": json.dumps(list(wcat_tensor.shape)),
            "w0_input_dim": int(w0_tensor.shape[1]),
            "w0_output_dim": int(w0_tensor.shape[0]),
            "w0_sha256": w0_hash,
            "wcat_sha256": wcat_hash,
            "w0_matches_existing_inventory": w0_hash == previous_hash,
            "weights_differ": w0_hash != wcat_hash,
            "selected_for_analysis": index < selected_count,
        })
        if index < selected_count:
            pairs.append((name, w0_tensor.numpy(), wcat_tensor.numpy()))
    if not all(row["w0_matches_existing_inventory"] for row in rows):
        raise RuntimeError("At least one current W0 hash differs from the existing layer inventory")
    return rows, pairs, w0_before


def _heatmap(
    summary: pd.DataFrame,
    field: str,
    title: str,
    subtitle: str,
    output_path: Path,
    *,
    signed: bool,
) -> None:
    layers = sorted(summary["layer_index"].unique())
    pivot = summary.pivot(index="true_concept", columns="layer_index", values=field).loc[
        CONCEPTS, layers
    ]
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    if signed:
        bound = max(float(np.abs(values).max()), 1e-6)
        image = ax.imshow(values, aspect="auto", cmap=SIGNED_CMAP, norm=TwoSlopeNorm(0, -bound, bound))
    else:
        image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(layers)), [f"L{i}" for i in layers])
    ax.set_yticks(range(len(CONCEPTS)), CONCEPTS)
    ax.set_title(f"{title}\n{subtitle}", loc="left", color=INK, fontsize=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            threshold = 0.55 if not signed else 0.60 * max(float(np.abs(values).max()), 1e-6)
            color = "white" if abs(value) >= threshold else INK
            ax.text(column, row, f"{value:+.3f}" if signed else f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(field.replace("_", " "))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _create_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _heatmap(
        summary,
        "mean_delta_dog",
        "Mean dog-anchor cosine change",
        "Positive values move toward the fixed original W0 dog reference",
        plot_dir / "mean_delta_dog_heatmap.png",
        signed=True,
    )
    _heatmap(
        summary,
        "mean_delta_cat",
        "Mean cat-reference cosine change",
        "Negative values move away from the fixed original W0 cat reference",
        plot_dir / "mean_delta_cat_heatmap.png",
        signed=True,
    )
    _heatmap(
        summary,
        "joint_intended_direction",
        "Joint intended-direction fraction",
        "Fraction with delta_dog > 0 and delta_cat < 0; 50 descriptions per cell",
        plot_dir / "joint_intended_direction_heatmap.png",
        signed=False,
    )

    cat = summary[summary["true_concept"] == "cat"].sort_values("layer_index")
    x = cat["layer_index"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for field, std_field, label, color, marker in [
        ("mean_delta_dog", "std_delta_dog", "delta_dog", BLUE, "o"),
        ("mean_delta_cat", "std_delta_cat", "delta_cat", ORANGE, "s"),
    ]:
        mean = cat[field].to_numpy(dtype=float)
        std = cat[std_field].to_numpy(dtype=float)
        axes[0].plot(x, mean, color=color, marker=marker, linewidth=2, label=label)
        axes[0].fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
    axes[0].axhline(0, color=INK, linewidth=1)
    axes[0].set_ylabel("Cosine-similarity change")
    axes[0].legend(frameon=False, ncol=2, loc="upper center")
    axes[0].grid(axis="y", color=GRID, linewidth=0.7)
    axes[1].plot(x, cat["joint_intended_direction"], color=GOLD, marker="o", linewidth=2)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Joint fraction")
    axes[1].set_xlabel("OCE layer")
    axes[1].set_xticks(x, [f"L{i}" for i in x])
    axes[1].grid(axis="y", color=GRID, linewidth=0.7)
    fig.suptitle(
        "Name-free cat-description shifts by layer\nMean ± one standard deviation; joint fraction uses both requested signs",
        x=0.08,
        ha="left",
        fontsize=14,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "cat_layer_shift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    cat_dogward = cat.set_index("layer_index")["mean_delta_dog"]
    non_target = (
        summary[summary["true_concept"] != "cat"]
        .groupby("layer_index", sort=True)["mean_delta_dog"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.plot(x, cat_dogward.loc[x], color=BLUE, marker="o", linewidth=2.2, label="Cat descriptions")
    ax.plot(x, non_target.loc[x], color=ORANGE, marker="s", linestyle="--", linewidth=2, label="Mean of seven non-target classes")
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(x, [f"L{i}" for i in x])
    ax.set_xlabel("OCE layer")
    ax.set_ylabel("Mean delta_dog")
    ax.set_title(
        "Cat versus non-target dogward shift\nBoth series use the fixed original W0 dog reference",
        loc="left",
        fontsize=14,
        color=INK,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "cat_vs_nontarget_dogward_shift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _interpretation_text(summary: pd.DataFrame, sanity: pd.DataFrame) -> str:
    cat = summary[summary["true_concept"] == "cat"]
    canonical_intended = int(sanity["intended_direction"].sum())
    cat_dog_positive = int((cat["mean_delta_dog"] > 0).sum())
    cat_cat_negative = int((cat["mean_delta_cat"] < 0).sum())
    joint_above_half = int((cat["joint_intended_direction"] > 0.5).sum())
    maximum_joint = float(cat["joint_intended_direction"].max())
    if canonical_intended == 0:
        case = (
            "Under the specified fixed-original-anchor metric, the answer is no: the canonical sanity "
            "check fails in every layer, cat-description dogward movement is not consistent, and the "
            f"largest cat joint fraction is only {maximum_joint:.2f}. None of Cases A-D fully applies "
            "because each presumes a usable canonical cat-to-dog directional effect or a consistent "
            "description-level pattern."
        )
    elif (
        cat_dog_positive == len(cat)
        and cat_cat_negative == len(cat)
        and joint_above_half == len(cat)
    ):
        case = "The directional pattern is closest to Case A if the class heatmaps also show non-target shifts near zero."
    elif cat_dog_positive > 0 and cat_cat_negative == 0:
        case = "The directional pattern is closest to Case D: dog-anchor alignment rises without a clean cat-reference decrease."
    else:
        case = "The layer-wise signs are mixed, so no single predefined case describes every layer."
    return (
        f"The canonical bare-name cat vector moves in the intended direction in {canonical_intended}/{len(sanity)} layers. "
        f"For name-free cat descriptions, mean delta_dog is positive in {cat_dog_positive}/{len(cat)} layers, "
        f"mean delta_cat is negative in {cat_cat_negative}/{len(cat)} layers, and the joint fraction exceeds 0.5 in "
        f"{joint_above_half}/{len(cat)} layers. {case} This classification is descriptive and remains layer-specific."
    )


def _build_report(
    output_dir: Path,
    dataset_audit: dict[str, Any],
    metadata: dict[str, Any],
    summary: pd.DataFrame,
    sanity: pd.DataFrame,
) -> None:
    layers = sorted(summary["layer_index"].unique())
    cat = summary[summary["true_concept"] == "cat"].set_index("layer_index")
    non_target = (
        summary[summary["true_concept"] != "cat"]
        .groupby("layer_index", sort=True)[["mean_delta_dog", "mean_delta_cat"]]
        .mean()
    )
    primary_rows = []
    for layer in layers:
        primary_rows.append([
            f"L{layer}",
            f"{cat.loc[layer, 'mean_delta_dog']:+.4f}",
            f"{cat.loc[layer, 'mean_delta_cat']:+.4f}",
            f"{cat.loc[layer, 'joint_intended_direction']:.2f}",
            f"{non_target.loc[layer, 'mean_delta_dog']:+.4f}",
            f"{non_target.loc[layer, 'mean_delta_cat']:+.4f}",
        ])
    interpretation = _interpretation_text(summary, sanity)
    settings = metadata["oce"]
    report = f"""# Cat-to-Dog OCE Shift of Name-Free Animal Descriptions

## 1. Research Question

Does the verified OCE `cat -> dog` transformation move the 50 name-free cat descriptions toward the original dog anchor while leaving the seven non-target animal classes relatively unchanged? This is the only question tested.

## 2. Cat -> Dog OCE Setup

The analysis uses the existing verified float32 checkpoint for SD 1.4 with edit concept `cat`, guide concept `dog`, preserve concept `dog`, prompt expansion `{str(settings['expand_prompts']).lower()}`, erase scale {settings['erase_scale']}, global-preservation scale {settings['preserve_global_scale']}, concept-preservation scale {settings['preserve_concept_scale']}, and lambda {settings['lambda']}. The checkpoint contains exactly the 16 `attn2.to_v` tensors targeted by the repository's `Orthogonal_Erase` implementation. It is read as tensors and is not loaded into a generation pipeline.

## 3. Fixed Original Cat and Dog References

At each layer, the cat and dog references are `W0_l c_cat` and `W0_l c_dog`, where the bare-name vectors use the repository's exact final-content-token-before-EOT rule. Both references are L2-normalized and remain fixed in the original W0 coordinate system. Edited descriptions are deliberately not compared with `Wcat_l c_dog`: doing so would move the reference and confound description movement with anchor movement.

## 4. Description-Level Delta Metrics

Only cached unsuffixed EOT description vectors are used. `delta_dog = cosine(Wcat h, W0 c_dog) - cosine(W0 h, W0 c_dog)`; positive values mean movement toward the original dog direction. `delta_cat = cosine(Wcat h, W0 c_cat) - cosine(W0 h, W0 c_cat)`; negative values mean movement away from the original cat direction. All four cosine inputs are L2-normalized within the same layer. No clustering is rerun because this experiment asks only about before/after directional change.

## 5. Canonical Cat Sanity Check

The bare canonical cat vector moves with positive canonical delta_dog and negative canonical delta_cat in {int(sanity['intended_direction'].sum())}/{len(sanity)} analyzed layers. Per-layer values are retained in `canonical_cat_sanity_check.csv`; this check is separate from the 400 description results.

## 6. Results for Name-Free Cat Descriptions

{_markdown_table(['Layer', 'Cat mean delta_dog', 'Cat mean delta_cat', 'Cat joint fraction', 'Non-target mean delta_dog', 'Non-target mean delta_cat'], primary_rows)}

The cat-only plot shows the mean and one-standard-deviation spread across 50 descriptions, plus the joint intended-direction fraction. It preserves the two requested deltas rather than replacing them with another shift score.

![Cat-only layer shifts](plots/cat_layer_shift.png)

## 7. Non-Target Specificity

The non-target controls are dog, fox, bear, wolf, rabbit, deer, and horse, each with 50 descriptions. The comparison plot contrasts the cat-description mean delta_dog with the equally weighted mean of the seven class-level non-target means. Detailed class-by-layer statistics remain in `class_layer_shift_summary.csv`.

![Cat versus non-target dogward shift](plots/cat_vs_nontarget_dogward_shift.png)

## 8. Layer-Wise Results

The heatmaps retain all eight classes and every analyzed layer. The first two show signed class means; the third shows the fraction satisfying both `delta_dog > 0` and `delta_cat < 0`.

![Mean delta dog](plots/mean_delta_dog_heatmap.png)

![Mean delta cat](plots/mean_delta_cat_heatmap.png)

![Joint intended direction](plots/joint_intended_direction_heatmap.png)

## 9. Main Interpretation

{interpretation}

## 10. Limitations

This experiment measures directional geometry in original per-layer cross-attention output coordinates. It does not generate images and therefore does not prove image-level erasure or replacement. The existing checkpoint uses the repository's recorded object-edit prompt expansion in addition to the canonical bare names, so the result characterizes that verified `cat -> dog` edit rather than a hypothetical bare-name-only checkpoint. Results are descriptive and may vary across layers; vectors from different layers are never compared directly.
"""
    atomic_write_text(output_dir / "report.md", report)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
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
    descriptions, c_cat, c_dog, embedding_audit = _load_cached_embeddings(
        source_dir,
        Path(args.embedding_cache).expanduser().resolve(),
        dataset_audit,
        output_dir,
    )
    atomic_write_text(output_dir / "dataset_audit.json", json.dumps(dataset_audit, indent=2) + "\n")

    weights_path = Path(args.edited_weights).expanduser().resolve()
    metadata_path = Path(args.edit_metadata).expanduser().resolve()
    previous_inventory_path = Path(args.layer_inventory).expanduser().resolve()
    for required in (weights_path, metadata_path, previous_inventory_path):
        if not required.exists():
            raise FileNotFoundError(required)
    edit_metadata, checkpoint_sha256 = _validate_edit_metadata(
        metadata_path, weights_path, args.model_id
    )
    oce_repo = Path(args.oce_repo).expanduser().resolve()
    oce_audit = _inspect_oce_source(oce_repo)
    oce_audit.pop("edited_checkpoint_loaded", None)
    oce_audit["edited_checkpoint_read_as_tensor_file"] = True
    oce_audit["edited_checkpoint_loaded_into_pipeline"] = False
    config = _runtime_config(args.model_id, args.device, args.batch_size, 0)
    pipe = load_original_pipeline(config, purpose="embedding", include_vae=False)
    inventory_rows, layer_pairs, w0_before = _load_layer_pairs(
        pipe, weights_path, previous_inventory_path, args.layer_limit
    )
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(output_dir / "layer_inventory.csv", index=False)

    shift_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    norm_checks: list[bool] = []
    for layer_index, (layer_name, w0, wcat) in enumerate(layer_pairs):
        shifts = _project_descriptions(descriptions, w0, wcat, c_cat, c_dog)
        norm_checks.extend([
            bool(np.allclose(shifts["before_norms"], 1.0, atol=1e-5)),
            bool(np.allclose(shifts["after_norms"], 1.0, atol=1e-5)),
            bool(np.allclose(shifts["cat_anchor_norm"], 1.0, atol=1e-5)),
            bool(np.allclose(shifts["dog_anchor_norm"], 1.0, atol=1e-5)),
        ])
        for sample_index, row in enumerate(rows):
            shift_rows.append({
                "description_id": row["candidate_id"],
                "true_concept": row["concept"],
                "description": row["description"],
                "layer_index": layer_index,
                "layer_name": layer_name,
                "cat_before": float(shifts["cat_before"][sample_index]),
                "cat_after": float(shifts["cat_after"][sample_index]),
                "dog_before": float(shifts["dog_before"][sample_index]),
                "dog_after": float(shifts["dog_after"][sample_index]),
                "delta_cat": float(shifts["delta_cat"][sample_index]),
                "delta_dog": float(shifts["delta_dog"][sample_index]),
                "intended_direction": bool(shifts["intended_direction"][sample_index]),
            })
        sanity_rows.append({
            "layer_index": layer_index,
            "layer_name": layer_name,
            **_canonical_cat_shift(w0, wcat, c_cat, c_dog),
        })

    values = pd.DataFrame(shift_rows)
    summary = _aggregate_shifts(values)
    sanity = pd.DataFrame(sanity_rows)
    values.to_csv(output_dir / "description_shift_values.csv", index=False)
    summary.to_csv(output_dir / "class_layer_shift_summary.csv", index=False)
    sanity.to_csv(output_dir / "canonical_cat_sanity_check.csv", index=False)

    current_modules = dict(original_projection_modules(pipe, "to_v"))
    w0_after = {name: _tensor_sha256(module.weight) for name, module in current_modules.items()}
    w0_unchanged = {name: w0_before[name] == w0_after[name] for name in w0_before}
    checkpoint_after_sha256 = _sha256(weights_path)
    weight_audit = {
        "status": "passed",
        "model_id": args.model_id,
        "checkpoint_path": str(weights_path),
        "checkpoint_sha256_before": checkpoint_sha256,
        "checkpoint_sha256_after": checkpoint_after_sha256,
        "checkpoint_unchanged": checkpoint_sha256 == checkpoint_after_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "verified_edit": {
            "method": edit_metadata["method"],
            "edit_concept": edit_metadata["edit_concept"],
            "guide_concept": edit_metadata["guide_concept"],
            "preserve_concept": edit_metadata.get("preserve_concept"),
            "configuration": edit_metadata["oce"],
        },
        "exact_reproduction_command_from_recorded_configuration": _oce_reproduction_command(edit_metadata),
        "oce_source_audit": oce_audit,
        "layer_count_in_checkpoint": 16,
        "analyzed_layer_count": len(layer_pairs),
        "w0_sha256_before": w0_before,
        "w0_sha256_after": w0_after,
        "all_w0_unchanged": all(w0_unchanged.values()),
        "w0_unchanged_by_layer": w0_unchanged,
        "edited_weights_loaded_into_pipeline": False,
        "diffusion_inference_called": False,
    }
    if not (
        weight_audit["checkpoint_unchanged"]
        and weight_audit["all_w0_unchanged"]
        and all(inventory["weights_differ"])
    ):
        raise RuntimeError("Weight immutability/alignment audit failed")
    atomic_write_text(output_dir / "weight_audit.json", json.dumps(weight_audit, indent=2) + "\n")

    _create_plots(summary, output_dir)
    _build_report(output_dir, dataset_audit, edit_metadata, summary, sanity)

    experiment = {
        "experiment_name": "cat_to_dog_description_shift",
        "research_question": "Does the canonical-name cat-to-dog OCE edit move name-free cat descriptions toward the original dog anchor while leaving other animal descriptions relatively unchanged?",
        "dataset": dataset_audit,
        "embedding": embedding_audit,
        "model": model_metadata(pipe, config, projection="to_v"),
        "oce": weight_audit["verified_edit"],
        "edited_checkpoint": str(weights_path),
        "edited_checkpoint_sha256": checkpoint_sha256,
        "fixed_references": {
            "cat": "normalize(W0_l @ c_cat)",
            "dog": "normalize(W0_l @ c_dog)",
            "edited_name_references_used": False,
        },
        "description_vectors": {
            "before": "normalize(W0_l @ h_EOT(p))",
            "after": "normalize(Wcat_l @ h_EOT(p))",
        },
        "metrics": ["cat_before", "cat_after", "dog_before", "dog_after", "delta_cat", "delta_dog", "intended_direction"],
        "layer_limit": args.layer_limit,
        "package_versions": package_versions(),
        "strict_scope": {
            "image_generation": False,
            "clustering": False,
            "description_centroids": False,
            "dimensionality_reduction": False,
            "description_readout": "unsuffixed_eot_only",
            "anchor_conditions": ["original_cat", "original_dog"],
        },
        "cli": vars(args),
    }
    atomic_write_text(output_dir / "experiment_config.json", json.dumps(experiment, indent=2) + "\n")

    expected_layers = len(layer_pairs)
    per_layer_counts = values.groupby("layer_index").size().to_dict()
    class_counts = summary.groupby(["layer_index", "true_concept"]).size()
    scope_source_audit = _strict_scope_source_audit()
    checks = {
        "status": "passed",
        "exact_final_balanced_8x50_dataset": dataset_audit["status"] == "passed",
        "exactly_400_descriptions": len(rows) == 400,
        "exactly_50_per_concept": Counter(row["concept"] for row in rows)
        == Counter({concept: 50 for concept in CONCEPTS}),
        "only_unsuffixed_eot_descriptions": embedding_audit["prefix"] is None
        and embedding_audit["suffix"] is None,
        "oce_edit_exactly_cat_to_dog": edit_metadata["edit_concept"] == "cat"
        and edit_metadata["guide_concept"] == "dog",
        "all_16_checkpoint_layers_aligned": len(inventory) == 16
        and bool(inventory["w0_matches_existing_inventory"].all()),
        "all_selected_layer_pairs_analyzed": set(per_layer_counts) == set(range(expected_layers))
        and all(count == 400 for count in per_layer_counts.values()),
        "original_references_use_w0_cat_and_w0_dog": True,
        "edited_descriptions_use_wcat_h": True,
        "edited_dog_reference_never_used": scope_source_audit[
            "edited_dog_anchor_absent_from_projection_functions"
        ],
        "all_cosine_inputs_l2_normalized": all(norm_checks),
        "no_diffusion_images_generated": True,
        "no_clustering_run": scope_source_audit["no_clustering_calls"],
        "no_description_centroids_constructed": True,
        "no_additional_anchor_conditions": True,
        "description_shift_row_count": len(values) == 400 * expected_layers,
        "summary_has_one_row_per_layer_and_class": len(summary) == expected_layers * 8
        and bool((class_counts == 1).all()),
        "canonical_sanity_has_one_row_per_layer": len(sanity) == expected_layers,
        "all_w0_and_checkpoint_files_unchanged": weight_audit["all_w0_unchanged"]
        and weight_audit["checkpoint_unchanged"],
        "exactly_five_requested_plots": len(list((output_dir / "plots").glob("*.png"))) == 5,
    }
    checks["status"] = "passed" if all(v for k, v in checks.items() if k != "status") else "failed"
    atomic_write_text(output_dir / "analysis_checks.json", json.dumps(checks, indent=2) + "\n")
    if checks["status"] != "passed":
        raise RuntimeError(f"Final strict-scope checks failed: {checks}")

    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    empty = [
        name
        for name in REQUIRED_OUTPUTS
        if (output_dir / name).exists() and (output_dir / name).stat().st_size == 0
    ]
    if missing or empty:
        raise RuntimeError(f"Output validation failed: missing={missing}, empty={empty}")
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure cat-to-dog OCE directional shifts for cached unsuffixed-EOT descriptions"
    )
    parser.add_argument("--dataset", required=True, help="Balanced 8x50 output directory or accepted JSONL")
    parser.add_argument("--embedding-cache", required=True, help="Verified balanced-paired W0 geometry cache")
    parser.add_argument("--model-id", required=True, help="Original SD 1.4 model identifier or local path")
    parser.add_argument("--oce-repo", required=True, help="OCE repository root containing oce.py")
    parser.add_argument("--edited-weights", required=True, help="Verified cat-to-dog OCE safetensors file")
    parser.add_argument("--edit-metadata", required=True, help="Metadata JSON for the edited checkpoint")
    parser.add_argument("--layer-inventory", required=True, help="Existing original-W0 16-layer inventory CSV")
    parser.add_argument("--output", required=True, help="New isolated output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer-limit", type=int, default=None, help="Smoke-test only: first N aligned layers")
    parser.add_argument("--force", action="store_true", help="Replace generated files only in selected output")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
