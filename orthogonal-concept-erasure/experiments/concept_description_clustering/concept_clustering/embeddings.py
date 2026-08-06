from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

import torch

from .modeling import load_original_pipeline, model_metadata, original_projection_modules
from .utils import atomic_write_text, read_jsonl, set_reproducible_seed, write_csv


def _selected_token_audit(
    tokenizer,
    original: str,
    prompt: str,
    condition: str,
    record_type: str,
    select_last_content_token: bool = False,
):
    encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    max_length = int(tokenizer.model_max_length)
    truncated = len(token_ids) > max_length
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    selected_position = len(token_ids) - 2
    if select_last_content_token:
        while selected_position > 0 and not _decoded_word(tokens[selected_position]):
            selected_position -= 1
        if selected_position <= 0 or not _decoded_word(tokens[selected_position]):
            raise RuntimeError(f"No natural content token found for prompt: {prompt!r}")
    selected_id = int(token_ids[selected_position])
    return {
        "condition": condition,
        "record_type": record_type,
        "original_description": original,
        "embedding_prompt": prompt,
        "token_ids": json.dumps(token_ids),
        "decoded_tokens": json.dumps(tokens, ensure_ascii=False),
        "selected_token_id": selected_id,
        "selected_token": tokens[selected_position],
        "selected_token_position": selected_position,
        "token_count_with_special_tokens": len(token_ids),
        "model_max_length": max_length,
        "truncation_occurred": truncated,
    }


def _decoded_word(token: str) -> str:
    token = token.replace("</w>", "")
    return re.sub(r"[^A-Za-z]", "", token).casefold()


@torch.inference_mode()
def _extract_contextual(text_encoder, tokenizer, prompts: list[str], positions: list[int], device: str, batch_size: int):
    vectors = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_positions = positions[start:start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=False,
            return_tensors="pt",
        ).to(device)
        hidden = text_encoder(**inputs).last_hidden_state
        row_indices = torch.arange(len(batch_prompts), device=device)
        position_tensor = torch.tensor(batch_positions, device=device)
        vectors.append(hidden[row_indices, position_tensor].float().cpu())
    return torch.cat(vectors, dim=0)


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)


def shuffle_description_words(description: str, candidate_id: str, seed: int) -> str:
    """Deterministically destroy word order while preserving the case-folded word bag."""
    tokens = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", description)
    if len(tokens) < 2:
        return description
    digest = hashlib.sha256(f"{seed}:{candidate_id}".encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
    order = torch.randperm(len(tokens), generator=generator).tolist()
    if order == list(range(len(tokens))):
        order = order[1:] + order[:1]
    terminal = description.rstrip()[-1] if description.rstrip()[-1] in ".!?" else "."
    return " ".join(tokens[index] for index in order) + terminal


def extract_embeddings(
    config: dict[str, Any],
    output_dir: str | Path,
    projection: str = "to_v",
    suffix_names: list[str] | None = None,
    force: bool = False,
) -> None:
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw_text_embeddings.pt"
    layer_path = output_dir / "layer_embeddings.pt"
    if raw_path.exists() and layer_path.exists() and not force:
        raise FileExistsError("Cached embeddings already exist; use --reuse-embeddings or pass --force")
    accepted = read_jsonl(output_dir / "accepted_descriptions.jsonl")
    expected = (
        len(config["concepts"])
        * len(config["facets"])
        * int(config["candidate_validation"]["accepted_per_concept_facet"])
    )
    if len(accepted) != expected:
        raise ValueError(f"Expected {expected} accepted descriptions, found {len(accepted)}")

    set_reproducible_seed(0)
    pipe = load_original_pipeline(config, purpose="embedding", include_vae=False)
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
    device = config["model"]["device"]
    batch_size = int(config["readout"]["batch_size"])
    primary = config["readout"]["primary_suffix_name"]
    suffix_names = suffix_names or [primary]
    unknown = set(suffix_names) - set(config["readout"]["suffixes"])
    if unknown:
        raise ValueError(f"Unknown suffix names: {sorted(unknown)}")

    descriptions = [row["description"] for row in accepted]
    concepts = [item["name"] for item in config["concepts"]]
    audit_rows = []
    fixed_vectors = {}
    prototype_vectors = {}
    shuffled_vectors = None
    shuffled_descriptions: list[str] = []

    for suffix_name in suffix_names:
        suffix = config["readout"]["suffixes"][suffix_name]
        expected_final_word = suffix.strip().split()[-1].casefold()
        prompts = [description.rstrip() + suffix for description in descriptions]
        prototype_prompts = [concept + suffix for concept in concepts]
        local_audits = [
            _selected_token_audit(tokenizer, original, prompt, suffix_name, "description")
            for original, prompt in zip(descriptions, prompts)
        ] + [
            _selected_token_audit(tokenizer, concept, prompt, suffix_name, "prototype")
            for concept, prompt in zip(concepts, prototype_prompts)
        ]
        if any(row["truncation_occurred"] for row in local_audits):
            bad = [row["original_description"] for row in local_audits if row["truncation_occurred"]][:5]
            raise RuntimeError(f"Suffix condition {suffix_name} truncates prompts: {bad}")
        selected_ids = {row["selected_token_id"] for row in local_audits}
        if len(selected_ids) != 1:
            raise RuntimeError(f"Final selected token differs within {suffix_name}: {selected_ids}")
        selected_words = {_decoded_word(row["selected_token"]) for row in local_audits}
        if selected_words != {expected_final_word}:
            raise RuntimeError(
                f"Final token for {suffix_name} is {selected_words}, expected {expected_final_word!r}"
            )
        if suffix_name == primary and expected_final_word != "concept":
            raise RuntimeError("Primary suffix must end in the exact word 'concept'")
        audit_rows.extend(local_audits)
        desc_positions = [row["selected_token_position"] for row in local_audits[:len(descriptions)]]
        proto_positions = [row["selected_token_position"] for row in local_audits[len(descriptions):]]
        fixed_vectors[suffix_name] = _normalize_tensor(
            _extract_contextual(text_encoder, tokenizer, prompts, desc_positions, device, batch_size)
        )
        prototype_vectors[suffix_name] = _normalize_tensor(
            _extract_contextual(text_encoder, tokenizer, prototype_prompts, proto_positions, device, batch_size)
        )

    shuffle_config = config.get("controls", {}).get("word_shuffle", {})
    if bool(shuffle_config.get("enabled", False)):
        shuffle_seed = int(shuffle_config.get("seed", 1729))
        candidate_ids = [row["candidate_id"] for row in accepted]
        shuffled_descriptions = [
            shuffle_description_words(description, candidate_id, shuffle_seed)
            for description, candidate_id in zip(descriptions, candidate_ids)
        ]
        suffix = config["readout"]["suffixes"][primary]
        shuffled_prompts = [description.rstrip() + suffix for description in shuffled_descriptions]
        shuffled_audits = [
            _selected_token_audit(
                tokenizer,
                original,
                prompt,
                f"shuffled_words_{primary}",
                "description",
            )
            for original, prompt in zip(shuffled_descriptions, shuffled_prompts)
        ]
        if any(row["truncation_occurred"] for row in shuffled_audits):
            raise RuntimeError("At least one shuffled description exceeds the tokenizer limit")
        expected_id = next(
            row["selected_token_id"]
            for row in audit_rows
            if row["condition"] == primary and row["record_type"] == "description"
        )
        if {row["selected_token_id"] for row in shuffled_audits} != {expected_id}:
            raise RuntimeError("Shuffled descriptions do not share the primary fixed-readout token")
        audit_rows.extend(shuffled_audits)
        shuffled_vectors = _normalize_tensor(
            _extract_contextual(
                text_encoder,
                tokenizer,
                shuffled_prompts,
                [row["selected_token_position"] for row in shuffled_audits],
                device,
                batch_size,
            )
        )
        original_bags = [sorted(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text.casefold())) for text in descriptions]
        shuffled_bags = [sorted(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text.casefold())) for text in shuffled_descriptions]
        shuffle_rows = [
            {
                "candidate_id": candidate_id,
                "shuffle_seed": shuffle_seed,
                "original_description": original,
                "shuffled_description": shuffled,
                "word_bag_identical": original_bag == shuffled_bag,
            }
            for candidate_id, original, shuffled, original_bag, shuffled_bag in zip(
                candidate_ids, descriptions, shuffled_descriptions, original_bags, shuffled_bags
            )
        ]
        if not all(row["word_bag_identical"] for row in shuffle_rows):
            raise RuntimeError("Word shuffling changed at least one word bag")
        write_csv(output_dir / "word_shuffle_audit.csv", shuffle_rows)

    natural_audits = [
        _selected_token_audit(
            tokenizer,
            description,
            description,
            "natural_last_token",
            "description",
            select_last_content_token=True,
        )
        for description in descriptions
    ]
    if any(row["truncation_occurred"] for row in natural_audits):
        raise RuntimeError("At least one unsuffixed accepted description exceeds the tokenizer limit")
    audit_rows.extend(natural_audits)
    natural_vectors = _normalize_tensor(
        _extract_contextual(
            text_encoder,
            tokenizer,
            descriptions,
            [row["selected_token_position"] for row in natural_audits],
            device,
            batch_size,
        )
    )
    write_csv(output_dir / "tokenization_audit.csv", audit_rows)

    metadata = model_metadata(pipe, config, projection=projection)
    metadata["primary_suffix_name"] = primary
    metadata["extracted_suffix_names"] = suffix_names
    metadata["global_seed"] = 0
    metadata["word_shuffle"] = shuffle_config
    if (output_dir / "run_metadata.json").exists():
        generation_metadata = json.loads((output_dir / "run_metadata.json").read_text())
        comparable = ["model_id", "checkpoint_revision"]
        # Generation always audits to_v because it is the primary W0 analysis.
        # Its layer structure is directly comparable only when extraction also uses to_v.
        if projection == "to_v":
            comparable.append("projection_layers")
        mismatches = [key for key in comparable if generation_metadata.get(key) != metadata.get(key)]
        if mismatches:
            raise RuntimeError(
                f"Generation and embedding stages did not resolve the same original checkpoint structure: {mismatches}"
            )
        metadata["generation_checkpoint_verified"] = True
        metadata["generation_w0_structure_verified"] = projection == "to_v"

    raw_payload = {
        "candidate_ids": [row["candidate_id"] for row in accepted],
        "concept_labels": [row["concept"] for row in accepted],
        "facet_labels": [row["facet_id"] for row in accepted],
        "descriptions": descriptions,
        "concept_names": concepts,
        "fixed_readout": fixed_vectors,
        "natural_last_token": natural_vectors,
        "shuffled_descriptions": shuffled_descriptions,
        "shuffled_fixed_readout": shuffled_vectors,
        "prototypes": prototype_vectors,
        "metadata": metadata,
    }
    torch.save(raw_payload, raw_path)

    fixed_primary = fixed_vectors[primary]
    proto_primary = prototype_vectors[primary]
    layer_descriptions = {}
    layer_prototypes = {}
    layer_shapes = {}
    for layer_index, (name, module) in enumerate(original_projection_modules(pipe, projection)):
        weight = module.weight.detach()
        projected = fixed_primary.to(device=device, dtype=weight.dtype) @ weight.T
        projected_proto = proto_primary.to(device=device, dtype=weight.dtype) @ weight.T
        layer_descriptions[name] = _normalize_tensor(projected.float().cpu())
        layer_prototypes[name] = _normalize_tensor(projected_proto.float().cpu())
        layer_shapes[name] = list(weight.shape)
    torch.save({
        "projection": projection,
        "layer_names": list(layer_descriptions),
        "layer_shapes": layer_shapes,
        "candidate_ids": raw_payload["candidate_ids"],
        "concept_names": concepts,
        "description_embeddings": layer_descriptions,
        "prototype_embeddings": layer_prototypes,
        "metadata": metadata,
    }, layer_path)
    atomic_write_text(output_dir / "embedding_metadata.json", json.dumps(metadata, indent=2, default=str) + "\n")
