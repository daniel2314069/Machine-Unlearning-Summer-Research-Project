from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from .modeling import load_original_pipeline, model_metadata, original_projection_modules
from .utils import set_reproducible_seed, write_csv


OCE_UCE_SELECTION_RULE = (
    "orthogonal-concept-erasure/oce.py OCE erasure entry point and "
    "unified-concept-editing/trainscripts/uce_sd_erase.py::UCE; "
    "last_token_idx = attention_mask.sum() - 2; "
    "vector = pipe.encode_prompt(...)[0][:, last_token_idx, :]"
)


def oce_uce_last_token_position(attention_mask: torch.Tensor) -> int:
    """Exact inline token-position rule used by the repository's SD OCE and UCE."""
    if attention_mask.ndim == 2:
        if attention_mask.shape[0] != 1:
            raise ValueError("OCE/UCE bare-name extraction expects one prompt at a time")
        attention_mask = attention_mask[0]
    return int(attention_mask.sum().item()) - 2


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".pt") as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def extract_bare_name_embeddings(
    config: dict[str, Any],
    source_output: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Extract bare-name vectors and their unchanged-original-W0 projections.

    No image pipeline call is made.  ``pipe.encode_prompt`` and the exact
    attention-mask rule are intentionally used because those are the operations
    in both the repository's OCE and SD UCE implementations.
    """
    source_output = Path(source_output).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "bare_name_embeddings.pt"
    audit_path = output_dir / "bare_name_tokenization_audit.csv"
    if (cache_path.exists() or audit_path.exists()) and not force:
        raise FileExistsError(f"Bare-name cache already exists in {output_dir}; pass --force to replace only it")

    raw_cache = torch.load(source_output / "raw_text_embeddings.pt", map_location="cpu", weights_only=False)
    layer_cache = torch.load(source_output / "layer_embeddings.pt", map_location="cpu", weights_only=False)
    concepts = [item["name"] for item in config["concepts"]]
    if concepts != list(raw_cache["concept_names"]):
        raise RuntimeError("Configured concepts do not match the cached accepted-description embeddings")
    if layer_cache.get("projection") != "to_v":
        raise RuntimeError("The source layer cache is not the required original to_v analysis")

    set_reproducible_seed(0)
    pipe = load_original_pipeline(config, purpose="embedding", include_vae=False)
    tokenizer = pipe.tokenizer
    device = config["model"]["device"]
    audit_rows: list[dict[str, Any]] = []
    raw_vectors: list[torch.Tensor] = []

    pipe.text_encoder.eval()
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.eval()
    pipe.unet.requires_grad_(False)
    for concept in concepts:
        tokenized = tokenizer(
            concept,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        selected_position = oce_uce_last_token_position(tokenized["attention_mask"])
        input_ids = tokenized["input_ids"][0].tolist()
        attention_mask = tokenized["attention_mask"][0].tolist()
        decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        content_positions = list(range(1, selected_position + 1))
        encoded = pipe.encode_prompt(
            prompt=concept,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        vector = encoded[0][:, selected_position, :].squeeze(0).detach().float().cpu()
        raw_vectors.append(vector)
        audit_rows.append({
            "concept": concept,
            "prompt": concept,
            "token_ids": json.dumps(input_ids),
            "decoded_tokens": json.dumps(decoded_tokens, ensure_ascii=False),
            "attention_mask": json.dumps(attention_mask),
            "selected_token_position": selected_position,
            "selected_token_id": int(input_ids[selected_position]),
            "selected_decoded_token": decoded_tokens[selected_position],
            "concept_token_count": len(content_positions),
            "concept_split_into_multiple_tokens": len(content_positions) > 1,
            "concept_token_positions": json.dumps(content_positions),
            "exact_repository_function_or_rule": OCE_UCE_SELECTION_RULE,
        })

    raw_unnormalized = torch.stack(raw_vectors)
    raw_normalized = _normalize_rows(raw_unnormalized)
    layer_unnormalized: dict[str, torch.Tensor] = {}
    layer_normalized: dict[str, torch.Tensor] = {}
    layer_shapes: dict[str, list[int]] = {}
    modules = original_projection_modules(pipe, "to_v")
    source_layer_names = list(layer_cache["layer_names"])
    if [name for name, _ in modules] != source_layer_names:
        raise RuntimeError("Current original to_v layer order differs from the accepted-description cache")
    for name, module in modules:
        weight = module.weight.detach()
        projected = raw_unnormalized.to(device=device, dtype=weight.dtype) @ weight.T
        projected = projected.detach().float().cpu()
        layer_unnormalized[name] = projected
        layer_normalized[name] = _normalize_rows(projected)
        layer_shapes[name] = list(weight.shape)

    metadata = model_metadata(pipe, config, projection="to_v")
    cached_metadata = raw_cache.get("metadata", {})
    expected_fingerprint = cached_metadata.get("original_w0_in_memory_fingerprint_sha256")
    if expected_fingerprint and metadata["original_w0_in_memory_fingerprint_sha256"] != expected_fingerprint:
        raise RuntimeError("Bare-name extraction resolved different original W0 matrices than the source cache")
    metadata.update({
        "concepts": concepts,
        "prompt_rule": "exact bare concept string; no suffix, punctuation, or additional words",
        "token_selection_rule": OCE_UCE_SELECTION_RULE,
        "source_output": str(source_output),
        "source_candidate_ids": list(raw_cache["candidate_ids"]),
        "source_fixed_readout_name": config["readout"]["primary_suffix_name"],
        "global_seed": 0,
        "image_generation_performed": False,
        "immutable_source_sha256": {
            name: _file_sha256(source_output / name)
            for name in [
                "accepted_descriptions.jsonl",
                "raw_text_embeddings.pt",
                "layer_embeddings.pt",
            ]
        },
    })
    payload = {
        "concept_names": concepts,
        "prompts": concepts,
        "raw_unnormalized": raw_unnormalized,
        "raw_normalized": raw_normalized,
        "layer_names": source_layer_names,
        "layer_shapes": layer_shapes,
        "layer_unnormalized": layer_unnormalized,
        "layer_normalized": layer_normalized,
        "metadata": metadata,
    }
    _atomic_torch_save(payload, cache_path)
    write_csv(audit_path, audit_rows)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache_path
