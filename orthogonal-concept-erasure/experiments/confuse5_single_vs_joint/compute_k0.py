#!/usr/bin/env python3
"""Compute a clean, non-resumable COCO-30k OCE global prior on the GPU server."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import protocol


def read_prompts(path: Path, column: str) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise protocol.ProtocolError(f"K0 dataset must contain column {column!r}")
            prompts = [str(row[column]).strip() for row in reader]
    except OSError as exc:
        raise protocol.ProtocolError(f"Cannot read K0 dataset {path}: {exc}") from exc
    if not prompts or any(not value for value in prompts):
        raise protocol.ProtocolError("K0 dataset contains no rows or an empty prompt")
    return prompts


def component_identity(component: Any) -> dict[str, Any]:
    config = getattr(component, "config", None)
    init_kwargs = getattr(component, "init_kwargs", {})
    return {
        "class": f"{type(component).__module__}.{type(component).__name__}",
        "name_or_path": getattr(component, "name_or_path", None)
        or getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None)
        or (init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None),
    }


def validate_existing(
    artifact: Path, metadata_path: Path, expected_fingerprint: str
) -> bool:
    if not artifact.is_file() or not metadata_path.is_file():
        return False
    metadata = protocol.read_json(metadata_path)
    return (
        metadata.get("status") == "complete"
        and metadata.get("request_fingerprint") == expected_fingerprint
        and metadata.get("artifact_sha256") == protocol.sha256(artifact)
        and metadata.get("finite") is True
    )


def compute(config_path: Path, *, skip_existing: bool) -> dict[str, Any]:
    config, _ = protocol.load_protocol(config_path)
    k0 = config["k0"]
    output_root = Path(config["_resolved"]["output_root"])
    artifact = output_root / "artifacts" / k0["output_filename"]
    metadata_path = output_root / "artifacts" / k0["metadata_filename"]
    incomplete = artifact.with_suffix(artifact.suffix + ".incomplete")
    dataset = Path(config["_resolved"]["k0_dataset"])
    dataset_hash = protocol.sha256(dataset)
    request = {
        "definition": k0["definition"],
        "dataset_path": str(dataset),
        "dataset_sha256": dataset_hash,
        "dataset_column": k0["dataset_column"],
        "base_model": config["model"]["base_model"],
        "batch_size": k0["batch_size"],
        "accumulation_dtype": k0["accumulation_dtype"],
        "output_dtype": k0["output_dtype"],
        "process_all_rows": k0["process_all_rows"],
        "flush_final_partial_batch": k0["flush_final_partial_batch"],
        "resume_allowed": k0["resume_allowed"],
        "compute_script_sha256": protocol.sha256(Path(__file__)),
    }
    request_fingerprint = protocol.fingerprint(request)
    if skip_existing and validate_existing(artifact, metadata_path, request_fingerprint):
        print(f"[skip complete K0] {artifact}")
        return protocol.read_json(metadata_path)
    if artifact.exists() or metadata_path.exists() or incomplete.exists():
        raise FileExistsError(
            "Refusing to resume or overwrite K0 artifacts. Use a fresh primary namespace: "
            f"{artifact.parent}"
        )

    prompts = read_prompts(dataset, k0["dataset_column"])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": protocol.utc_now(),
        "request": request,
        "request_fingerprint": request_fingerprint,
        "processed_row_count": 0,
        "token_count": 0,
        "runtime": protocol.runtime_provenance(),
    }
    protocol.write_json_atomic(metadata_path, metadata)

    import torch
    from diffusers import DiffusionPipeline

    if k0["accumulation_dtype"] != "float32" or k0["output_dtype"] != "float32":
        raise protocol.ProtocolError("Primary K0 dtype must remain float32")
    device = config["model"]["device"]
    pipe = DiffusionPipeline.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
    ).to(device)
    text_encoder = pipe.text_encoder.eval()
    tokenizer = pipe.tokenizer
    hidden_dim = int(text_encoder.config.hidden_size)
    accumulator = torch.zeros(
        hidden_dim, hidden_dim, device=device, dtype=torch.float32
    )
    token_count = 0
    batch_size = int(k0["batch_size"])
    processed_rows = 0
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=tokenizer.model_max_length,
        ).to(device)
        with torch.inference_mode():
            hidden = text_encoder(**inputs, output_hidden_states=True).last_hidden_state
        flattened = hidden.reshape(-1, hidden.shape[-1]).to(dtype=torch.float32)
        mask = inputs["attention_mask"].reshape(-1).bool()
        tokens = flattened[mask]
        accumulator.add_(tokens.T @ tokens)
        token_count += int(tokens.shape[0])
        processed_rows += len(batch)
        print(f"[K0] rows={processed_rows}/{len(prompts)} tokens={token_count}", flush=True)

    if processed_rows != len(prompts) or token_count <= 0:
        raise RuntimeError("K0 did not process every ordered row or found no tokens")
    matrix = accumulator / token_count
    finite = bool(torch.isfinite(matrix).all().item())
    if not finite:
        raise RuntimeError("K0 contains a non-finite value")
    payload = {"C": matrix.detach().cpu(), "count": token_count}
    torch.save(payload, incomplete)
    incomplete.replace(artifact)

    metadata.update({
        "status": "complete",
        "finished_at": protocol.utc_now(),
        "dataset_identity": {
            "path": str(dataset),
            "sha256": dataset_hash,
            "row_count": len(prompts),
            "ordered_prompts_sha256": protocol.fingerprint({"prompts": prompts}),
        },
        "model_identity": component_identity(pipe),
        "text_encoder_identity": component_identity(text_encoder),
        "tokenizer_identity": component_identity(tokenizer),
        "processed_row_count": processed_rows,
        "token_count": token_count,
        "tensor_shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "finite": finite,
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": protocol.sha256(artifact),
        "source_hashes": protocol.source_hashes([Path(__file__), dataset]),
    })
    protocol.write_json_atomic(metadata_path, metadata)
    print(f"[K0 complete] {artifact} sha256={metadata['artifact_sha256']}")
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    compute(args.config.resolve(), skip_existing=args.skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
