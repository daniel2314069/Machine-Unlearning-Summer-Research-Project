from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import CLIPTextModel, CLIPTokenizer


HERE = Path(__file__).resolve().parent
SMOKE_ROOT = HERE.parent


MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_CHECKPOINT = SMOKE_ROOT / "joint_official_subspace.safetensors"
TARGETS = ["cat", "truck", "church", "Van Gogh", "Adam Driver"]
ANCHORS = ["dog", "car", "castle", "art", "celebrity"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def cosine_matrix(
    row_vectors: torch.Tensor, column_vectors: torch.Tensor
) -> torch.Tensor:
    if row_vectors.ndim != 2 or column_vectors.ndim != 2:
        raise ValueError("Cosine inputs must both be rank-2 matrices")
    if row_vectors.shape[1] != column_vectors.shape[1]:
        raise ValueError(
            "Feature shapes differ before cosine: "
            f"{tuple(row_vectors.shape)} vs {tuple(column_vectors.shape)}"
        )
    rows = torch.nn.functional.normalize(row_vectors.double(), dim=1)
    columns = torch.nn.functional.normalize(column_vectors.double(), dim=1)
    matrix = rows @ columns.T
    if matrix.shape != (len(TARGETS), len(ANCHORS)):
        raise RuntimeError(f"Unexpected matrix shape: {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise RuntimeError("Similarity matrix contains non-finite values")
    return matrix


def matrix_to_list(matrix: torch.Tensor) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in matrix.detach().cpu().tolist()
    ]


def markdown_matrix(matrix: Sequence[Sequence[float]]) -> list[str]:
    lines = [
        "| target \\ anchor | "
        + " | ".join(ANCHORS)
        + " |",
        "|---|" + "|".join(["---:"] * len(ANCHORS)) + "|",
    ]
    for target, row in zip(TARGETS, matrix):
        lines.append(
            f"| {target} | "
            + " | ".join(f"{float(value):.6f}" for value in row)
            + " |"
        )
    return lines


def build_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Attention Layer Similarity Matrices",
        "",
        "Rows: `cat`, `truck`, `church`, `Van Gogh`, `Adam Driver`  ",
        "Columns: `dog`, `car`, `castle`, `art`, `celebrity`",
        "",
        "Before uses `cos(W₀cᵢ, W₀cⱼ*)`; after uses "
        "`cos(Wedited cᵢ, W₀cⱼ*)`, where `Wedited = PℓW₀`; "
        "difference is after minus before.",
        "",
    ]
    for layer in results["layers"]:
        layer_id = layer["layer_id"]
        layer_name = layer["layer_name"]
        lines.extend(
            [
                f"## Layer {layer_id:02d}: `{layer_name}`",
                "",
                "### Before",
                "",
                *markdown_matrix(
                    results["before"]["per_layer"][str(layer_id)]["matrix"]
                ),
                "",
                "### After",
                "",
                *markdown_matrix(
                    results["after"]["per_layer"][str(layer_id)]["matrix"]
                ),
                "",
                "### Difference (after − before)",
                "",
                *markdown_matrix(
                    results["difference"]["per_layer"][str(layer_id)][
                        "matrix"
                    ]
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-layer average: before",
            "",
            *markdown_matrix(results["before"]["cross_layer_average"]),
            "",
            "## Cross-layer average: after",
            "",
            *markdown_matrix(results["after"]["cross_layer_average"]),
            "",
            "## Cross-layer average: difference (after − before)",
            "",
            *markdown_matrix(results["difference"]["cross_layer_average"]),
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def run(checkpoint: Path, output: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            local_files_only=True,
        )
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        snapshot / "tokenizer",
        local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        snapshot / "text_encoder",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    original_unet_path = (
        snapshot / "unet" / "diffusion_pytorch_model.safetensors"
    )
    if not original_unet_path.is_file():
        raise FileNotFoundError(original_unet_path)
    with safe_open(checkpoint, framework="pt", device="cpu") as edited_file:
        actual_keys = list(edited_file.keys())
        edited_state = {
            key: edited_file.get_tensor(key) for key in actual_keys
        }
    if len(actual_keys) != 16:
        raise RuntimeError(
            f"Expected exactly 16 edited checkpoint keys, found "
            f"{len(actual_keys)}"
        )
    invalid_keys = [
        key
        for key in actual_keys
        if not ("attn2" in key and key.endswith("to_v.weight"))
    ]
    if invalid_keys:
        raise RuntimeError(f"Checkpoint contains non-edited keys: {invalid_keys}")
    with safe_open(
        original_unet_path, framework="pt", device="cpu"
    ) as original_file:
        missing = [key for key in actual_keys if key not in original_file.keys()]
        if missing:
            raise RuntimeError(
                f"Edited keys missing from original UNet: {missing}"
            )
        original_state = {
            key: original_file.get_tensor(key) for key in actual_keys
        }

    concepts = TARGETS + ANCHORS
    embeddings: dict[str, torch.Tensor] = {}
    tokenization: list[dict[str, Any]] = []
    for concept in concepts:
        tokenized = tokenizer(
            concept,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden = text_encoder(tokenized.input_ids).last_hidden_state[0]
        selected_index = int(tokenized.attention_mask.sum().item()) - 2
        embeddings[concept] = hidden[selected_index].float()
        bare_ids = tokenizer(
            concept, add_special_tokens=False
        )["input_ids"]
        tokenization.append(
            {
                "concept": concept,
                "token_ids": [int(value) for value in bare_ids],
                "token_strings": tokenizer.convert_ids_to_tokens(bare_ids),
                "selected_hidden_state_index": selected_index,
                "selected_representation": (
                    "last content token hidden state, matching official OCE"
                ),
            }
        )

    before_by_layer: dict[str, dict[str, Any]] = {}
    after_by_layer: dict[str, dict[str, Any]] = {}
    difference_by_layer: dict[str, dict[str, Any]] = {}
    layers: list[dict[str, Any]] = []
    before_stack: list[torch.Tensor] = []
    after_stack: list[torch.Tensor] = []
    difference_stack: list[torch.Tensor] = []
    cell_rows: list[dict[str, Any]] = []

    for layer_id, key in enumerate(actual_keys, start=1):
        layer_name = key.removesuffix(".weight")
        before_weight = original_state[key].detach().double()
        after_weight = edited_state[key].detach().double()
        if before_weight.shape != after_weight.shape:
            raise RuntimeError(
                f"Weight shape mismatch at {layer_name}: "
                f"{tuple(before_weight.shape)} vs "
                f"{tuple(after_weight.shape)}"
            )
        target_embeddings = torch.stack(
            [embeddings[target].double() for target in TARGETS]
        )
        anchor_embeddings = torch.stack(
            [embeddings[anchor].double() for anchor in ANCHORS]
        )
        if target_embeddings.shape[1] != before_weight.shape[1]:
            raise RuntimeError(
                f"Target embedding/weight mismatch at {layer_name}: "
                f"{tuple(target_embeddings.shape)} vs "
                f"{tuple(before_weight.shape)}"
            )
        if anchor_embeddings.shape[1] != before_weight.shape[1]:
            raise RuntimeError(
                f"Anchor embedding/weight mismatch at {layer_name}: "
                f"{tuple(anchor_embeddings.shape)} vs "
                f"{tuple(before_weight.shape)}"
            )

        before_target_features = target_embeddings @ before_weight.T
        after_target_features = target_embeddings @ after_weight.T
        anchor_features = anchor_embeddings @ before_weight.T
        if (
            before_target_features.shape != anchor_features.shape
            or after_target_features.shape != anchor_features.shape
        ):
            raise RuntimeError(
                f"Feature shape mismatch at {layer_name}: "
                f"before={tuple(before_target_features.shape)}, "
                f"after={tuple(after_target_features.shape)}, "
                f"anchor={tuple(anchor_features.shape)}"
            )

        before = cosine_matrix(before_target_features, anchor_features)
        after = cosine_matrix(after_target_features, anchor_features)
        difference = after - before
        before_stack.append(before)
        after_stack.append(after)
        difference_stack.append(difference)
        layer_key = str(layer_id)
        common = {
            "layer_id": layer_id,
            "layer_name": layer_name,
            "weight_shape": [int(value) for value in before_weight.shape],
            "feature_shape": [
                int(value) for value in before_target_features.shape
            ],
        }
        layers.append(common)
        before_by_layer[layer_key] = {
            **common,
            "matrix": matrix_to_list(before),
        }
        after_by_layer[layer_key] = {
            **common,
            "matrix": matrix_to_list(after),
        }
        difference_by_layer[layer_key] = {
            **common,
            "matrix": matrix_to_list(difference),
        }
        for row_index, target in enumerate(TARGETS):
            for column_index, anchor in enumerate(ANCHORS):
                cell_rows.append(
                    {
                        "layer_id": layer_id,
                        "layer_name": layer_name,
                        "target_index": row_index,
                        "target": target,
                        "anchor_index": column_index,
                        "anchor": anchor,
                        "before": float(before[row_index, column_index]),
                        "after": float(after[row_index, column_index]),
                        "difference": float(
                            difference[row_index, column_index]
                        ),
                    }
                )

    before_average = torch.stack(before_stack).mean(dim=0)
    after_average = torch.stack(after_stack).mean(dim=0)
    difference_average = torch.stack(difference_stack).mean(dim=0)
    direct_average_difference = after_average - before_average
    if not torch.allclose(
        difference_average,
        direct_average_difference,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Cross-layer difference does not equal after average minus "
            "before average"
        )
    if len(layers) != len(actual_keys):
        raise RuntimeError("Not every checkpoint layer was analyzed exactly once")

    average_rows: list[dict[str, Any]] = []
    for row_index, target in enumerate(TARGETS):
        for column_index, anchor in enumerate(ANCHORS):
            average_rows.append(
                {
                    "target_index": row_index,
                    "target": target,
                    "anchor_index": column_index,
                    "anchor": anchor,
                    "before": float(before_average[row_index, column_index]),
                    "after": float(after_average[row_index, column_index]),
                    "difference": float(
                        difference_average[row_index, column_index]
                    ),
                }
            )

    results = {
        "metadata": {
            "model_id": MODEL_ID,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "original_unet_weights": str(original_unet_path.resolve()),
            "model_snapshot": str(snapshot.resolve()),
            "objective": "official OCE subspace",
            "row_order": TARGETS,
            "column_order": ANCHORS,
            "edited_layer_count": len(layers),
            "edited_layer_selector": (
                "module name contains attn2 and ends with to_v"
            ),
            "before_definition": "cos(W0 c_i, W0 c_j_star)",
            "after_definition": (
                "cos(Wedited c_i, W0 c_j_star), Wedited = P_l W0"
            ),
            "difference_definition": "after - before",
            "cosine_compute_dtype": "float64",
            "tokenization": tokenization,
        },
        "layers": layers,
        "before": {
            "per_layer": before_by_layer,
            "cross_layer_average": matrix_to_list(before_average),
        },
        "after": {
            "per_layer": after_by_layer,
            "cross_layer_average": matrix_to_list(after_average),
        },
        "difference": {
            "per_layer": difference_by_layer,
            "cross_layer_average": matrix_to_list(difference_average),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", results)
    write_csv(output / "per_layer_cells.csv", cell_rows)
    write_csv(output / "cross_layer_average.csv", average_rows)
    (output / "matrices.md").write_text(
        build_markdown(results), encoding="utf-8"
    )

    validation = {
        "status": "passed",
        "checkpoint_layer_count": len(actual_keys),
        "analyzed_layer_count": len(layers),
        "only_checkpoint_layers_analyzed": len(layers) == len(actual_keys),
        "all_matrix_shapes": [len(TARGETS), len(ANCHORS)],
        "all_values_finite": all(
            math.isfinite(row[column])
            for matrix in (
                matrix_to_list(value)
                for value in before_stack + after_stack + difference_stack
            )
            for row in matrix
            for column in range(len(row))
        ),
        "difference_identity_atol": 1e-12,
        "difference_identity_passed": True,
        "row_order": TARGETS,
        "column_order": ANCHORS,
    }
    write_json(output / "validation.json", validation)
    print(f"[complete] {output / 'results.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-edited-layer 5x5 before/after OCE feature cosine "
            "matrices and their elementwise differences."
        )
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=HERE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.checkpoint.resolve(), args.output.resolve())
