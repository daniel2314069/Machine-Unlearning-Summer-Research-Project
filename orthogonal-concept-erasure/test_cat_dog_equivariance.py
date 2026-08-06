"""Test OCE equivariance and sequential composition on cat/dog.

This is an operator-level diagnostic: it does not generate images.  It uses the
actual SD 1.4 text encoder and cross-attention ``attn2.to_v`` weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


@torch.inference_mode()
def encode_last_token(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result = {}
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        hidden = text_encoder(input_ids).last_hidden_state[0]
        last_idx = int(inputs.attention_mask.sum().item()) - 2
        result[prompt] = hidden[last_idx].float()
    return result


@torch.inference_mode()
def empirical_cg(
    csv_path: Path,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
    max_tokens: int,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    """Compute a small fixed global second moment for this diagnostic."""
    dim = text_encoder.config.hidden_size
    total = torch.zeros(dim, dim, device=device, dtype=torch.float32)
    count = 0
    batch: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            batch.append(row["prompt"])
            if len(batch) < batch_size:
                continue
            count = _accumulate_batch(
                batch, tokenizer, text_encoder, device, total, count, max_tokens
            )
            batch = []
            if count >= max_tokens:
                break

    if batch and count < max_tokens:
        count = _accumulate_batch(
            batch, tokenizer, text_encoder, device, total, count, max_tokens
        )
    if count == 0:
        raise RuntimeError("No tokens were collected for C_g")
    return total / count, count


@torch.inference_mode()
def _accumulate_batch(
    texts: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
    total: torch.Tensor,
    count: int,
    max_tokens: int,
) -> int:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=tokenizer.model_max_length,
    ).to(device)
    hidden = text_encoder(**inputs).last_hidden_state.float()
    flat = hidden.reshape(-1, hidden.shape[-1])
    flat = flat[inputs.attention_mask.reshape(-1).bool()]
    flat = flat[: max_tokens - count]
    total.addmm_(flat.T, flat)
    return count + flat.shape[0]


def projector(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vector = vector / (torch.linalg.vector_norm(vector) + eps)
    return vector[:, None] @ vector[None, :]


@torch.inference_mode()
def oce_operator(
    weight: torch.Tensor,
    erase_embedding: torch.Tensor,
    guide_embedding: torch.Tensor,
    cg: torch.Tensor,
    erase_scale: float,
    global_scale: float,
    lamb: float,
    correction: str,
) -> tuple[torch.Tensor, bool]:
    erase = projector(weight @ erase_embedding)
    guide = projector(weight @ guide_embedding)
    identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
    objective = -erase_scale * erase @ (identity - guide)
    objective += global_scale * (weight @ cg @ weight.T)
    objective += lamb * (weight @ weight.T)

    u, _, vh = torch.linalg.svd(objective, full_matrices=False)
    raw = u @ vh
    reflected = bool(torch.linalg.det(raw).item() < 0)
    if reflected:
        if correction == "upstream":
            # This exactly matches oce.py:121-122.
            raw[:, -1] *= -1
        elif correction == "proper":
            # Standard constrained orthogonal Procrustes correction.
            u[:, -1] *= -1
            raw = u @ vh
        elif correction != "none":
            raise ValueError(correction)
    return raw, reflected


def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.matrix_norm(expected).clamp_min(1e-12)
    return float((torch.linalg.matrix_norm(actual - expected) / denominator).item())


@torch.inference_mode()
def test_layer(
    name: str,
    weight: torch.Tensor,
    cat: torch.Tensor,
    dog: torch.Tensor,
    guide: torch.Tensor,
    cg: torch.Tensor,
    args: argparse.Namespace,
    correction: str,
) -> dict[str, object]:
    kwargs = dict(
        guide_embedding=guide,
        cg=cg,
        erase_scale=args.erase_scale,
        global_scale=args.global_scale,
        lamb=args.lamb,
        correction=correction,
    )
    r_cat, refl_cat = oce_operator(weight, erase_embedding=cat, **kwargs)
    r_dog, refl_dog = oce_operator(weight, erase_embedding=dog, **kwargs)

    w_cat = r_cat @ weight
    w_dog = r_dog @ weight
    r_dog_after_cat, refl_dog_after = oce_operator(
        w_cat, erase_embedding=dog, **kwargs
    )
    r_cat_after_dog, refl_cat_after = oce_operator(
        w_dog, erase_embedding=cat, **kwargs
    )

    predicted_dog_after_cat = r_cat @ r_dog @ r_cat.T
    predicted_cat_after_dog = r_dog @ r_cat @ r_dog.T
    w_cat_dog = r_dog_after_cat @ w_cat
    w_dog_cat = r_cat_after_dog @ w_dog
    predicted_cat_dog = r_cat @ w_dog
    predicted_dog_cat = r_dog @ w_cat
    actual_difference = w_cat_dog - w_dog_cat
    commutator_on_weight = predicted_cat_dog - predicted_dog_cat

    return {
        "layer": name,
        "out_dim": weight.shape[0],
        "correction": correction,
        "reflection_flags": {
            "cat": refl_cat,
            "dog": refl_dog,
            "dog_after_cat": refl_dog_after,
            "cat_after_dog": refl_cat_after,
        },
        "dog_after_cat_conjugation_relerr": rel_error(
            r_dog_after_cat, predicted_dog_after_cat
        ),
        "cat_after_dog_conjugation_relerr": rel_error(
            r_cat_after_dog, predicted_cat_after_dog
        ),
        "cat_then_dog_weight_relerr": rel_error(w_cat_dog, predicted_cat_dog),
        "dog_then_cat_weight_relerr": rel_error(w_dog_cat, predicted_dog_cat),
        "commutator_prediction_relerr": rel_error(
            actual_difference, commutator_on_weight
        ),
        "sequential_order_effect_rel": float(
            (
                torch.linalg.matrix_norm(actual_difference)
                / torch.linalg.matrix_norm(weight).clamp_min(1e-12)
            ).item()
        ),
        "commutator_effect_rel": float(
            (
                torch.linalg.matrix_norm(commutator_on_weight)
                / torch.linalg.matrix_norm(weight).clamp_min(1e-12)
            ).item()
        ),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "dog_after_cat_conjugation_relerr",
        "cat_after_dog_conjugation_relerr",
        "cat_then_dog_weight_relerr",
        "dog_then_cat_weight_relerr",
        "commutator_prediction_relerr",
        "sequential_order_effect_rel",
        "commutator_effect_rel",
    ]
    return {
        metric: {
            "mean": sum(float(row[metric]) for row in rows) / len(rows),
            "max": max(float(row[metric]) for row in rows),
        }
        for metric in metrics
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cg-path", type=Path)
    parser.add_argument("--coco-csv", type=Path, default=Path("data/coco_30k.csv"))
    parser.add_argument("--cg-tokens", type=int, default=8192)
    parser.add_argument("--cg-batch-size", type=int, default=32)
    parser.add_argument("--erase-scale", type=float, default=2000.0)
    parser.add_argument("--global-scale", type=float, default=10.0)
    parser.add_argument("--lamb", type=float, default=10.0)
    parser.add_argument(
        "--corrections", nargs="+", choices=["upstream", "proper", "none"],
        default=["upstream", "proper"]
    )
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--output", type=Path, default=Path("cat_dog_equivariance.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder", local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    embeddings = encode_last_token(
        ["cat", "dog", " "], tokenizer, text_encoder, device
    )

    if args.cg_path:
        stats = torch.load(args.cg_path, map_location=device)
        cg = stats["C"].to(device=device, dtype=torch.float32)
        cg_source = {"path": str(args.cg_path), "count": stats.get("count")}
    else:
        cg, count = empirical_cg(
            args.coco_csv, tokenizer, text_encoder, device,
            args.cg_tokens, args.cg_batch_size,
        )
        cg_source = {"path": str(args.coco_csv), "count": count}
    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet", local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    modules = [
        (name, module)
        for name, module in unet.named_modules()
        if "attn2" in name and name.endswith("to_v")
    ]
    if args.max_layers is not None:
        modules = modules[: args.max_layers]

    rows = []
    for correction in args.corrections:
        for index, (name, module) in enumerate(modules, start=1):
            print(f"[{correction}] {index}/{len(modules)} {name}", flush=True)
            rows.append(
                test_layer(
                    name, module.weight.detach().float(), embeddings["cat"],
                    embeddings["dog"], embeddings[" "], cg, args, correction,
                )
            )

    result = {
        "configuration": {
            "erase_concepts": ["cat", "dog"],
            "guide": " ",
            "explicit_retain": [],
            "prompt_expansion": False,
            "erase_scale": args.erase_scale,
            "global_scale": args.global_scale,
            "lambda": args.lamb,
            "cg_source": cg_source,
            "layers": len(modules),
        },
        "summary": {
            correction: summarize(
                [row for row in rows if row["correction"] == correction]
            )
            for correction in args.corrections
        },
        "layers": rows,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
