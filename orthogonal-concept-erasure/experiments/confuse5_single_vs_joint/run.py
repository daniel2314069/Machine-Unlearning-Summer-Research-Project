#!/usr/bin/env python3
"""Build the 10 Single and 5 Joint primary Confuse5 OCE checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import protocol


HERE = Path(__file__).resolve().parent


def build_plan(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config, anchors = protocol.load_protocol(config_path)
    specs = protocol.checkpoint_specs(config, anchors)
    output_root = Path(config["_resolved"]["output_root"])
    k0_path = output_root / "artifacts" / config["k0"]["output_filename"]
    k0_metadata = output_root / "artifacts" / config["k0"]["metadata_filename"]
    plan = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "created_at": protocol.utc_now(),
        "config_path": str(config_path.resolve()),
        "config_sha256": protocol.sha256(config_path),
        "anchors_path": config["_resolved"]["anchors_path"],
        "anchors_sha256": protocol.sha256(Path(config["_resolved"]["anchors_path"])),
        "output_root": str(output_root),
        "k0_path": str(k0_path),
        "k0_metadata_path": str(k0_metadata),
        "checkpoint_count": len(specs),
        "single_count": sum(item["mode"] == "single" for item in specs),
        "joint_count": sum(item["mode"] == "joint" for item in specs),
        "source_hashes": protocol.source_hashes([Path(__file__)]),
        "resolved_config": config,
        "full_anchor_mapping": anchors,
        "checkpoints": specs,
    }
    plan["plan_fingerprint"] = protocol.fingerprint({
        "experiment_id": plan["experiment_id"],
        "config_sha256": plan["config_sha256"],
        "anchors_sha256": plan["anchors_sha256"],
        "source_hashes": plan["source_hashes"],
        "checkpoints": specs,
    })
    return plan, config, anchors


def validate_k0(plan: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch

    path = Path(plan["k0_path"])
    metadata_path = Path(plan["k0_metadata_path"])
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Primary K0 artifact/metadata is missing; run compute_k0.py first")
    metadata = protocol.read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise protocol.ProtocolError("K0 metadata is not complete")
    if metadata.get("artifact_sha256") != protocol.sha256(path):
        raise protocol.ProtocolError("K0 artifact hash differs from metadata")
    if metadata.get("dataset_identity", {}).get("sha256") != protocol.sha256(
        Path(config["_resolved"]["k0_dataset"])
    ):
        raise protocol.ProtocolError("K0 dataset hash differs from the resolved protocol")
    if metadata.get("processed_row_count") != metadata.get("dataset_identity", {}).get("row_count"):
        raise protocol.ProtocolError("K0 did not process every dataset row")
    if metadata.get("finite") is not True or metadata.get("dtype") != "torch.float32":
        raise protocol.ProtocolError("K0 must be finite float32")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "C" not in payload or "count" not in payload:
        raise protocol.ProtocolError("K0 artifact must contain C and count")
    matrix = payload["C"]
    if list(matrix.shape) != metadata.get("tensor_shape"):
        raise protocol.ProtocolError("K0 tensor shape differs from metadata")
    if int(payload["count"]) != int(metadata.get("token_count", -1)):
        raise protocol.ProtocolError("K0 token count differs from metadata")
    return matrix, metadata


def _projection_modules(unet: Any) -> list[tuple[str, Any]]:
    modules = [
        (name, module)
        for name, module in unet.named_modules()
        if "attn2" in name and name.endswith("to_v")
    ]
    return modules


def _encode_prompt(pipe: Any, prompt_text: str, device: str, dtype: Any) -> Any:
    encoded = pipe.encode_prompt(
        prompt=prompt_text,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )[0]
    attention = pipe.tokenizer(
        prompt_text,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )["attention_mask"]
    last_index = int(attention.sum().item()) - 2
    return encoded[:, last_index, :].squeeze(0).to(device=device, dtype=dtype)


def _subspace(weight: Any, embeddings: Sequence[Any], eps: float) -> Any:
    import torch

    vectors = []
    for embedding in embeddings:
        vector = weight @ embedding
        vectors.append(vector / (vector.norm() + eps))
    basis, _ = torch.linalg.qr(torch.stack(vectors, dim=1), mode="reduced")
    return basis


def _solve(matrix: Any, *, determinant_correction: bool) -> tuple[Any, float, float, bool]:
    import torch

    left, _, right_h = torch.linalg.svd(matrix, full_matrices=False)
    transform = left @ right_h
    determinant_before = float(torch.linalg.det(transform).item())
    corrected = determinant_correction and determinant_before < 0
    if corrected:
        transform[:, -1] *= -1
    determinant_after = float(torch.linalg.det(transform).item())
    return transform, determinant_before, determinant_after, corrected


def _scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _component_identity(component: Any) -> dict[str, Any]:
    config = getattr(component, "config", None)
    return {
        "class": f"{type(component).__module__}.{type(component).__name__}",
        "name_or_path": getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None),
    }


def build_checkpoint(
    pipe: Any,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    full_anchors: Mapping[str, str],
    k0_matrix: Any,
    k0_metadata: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    skip_existing: bool,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    checkpoint_path = Path(spec["checkpoint_path"])
    metadata_path = Path(spec["metadata_path"])
    if checkpoint_path.is_file() and metadata_path.is_file() and skip_existing:
        existing = protocol.read_json(metadata_path)
        if (
            existing.get("status") == "complete"
            and existing.get("plan_fingerprint") == plan["plan_fingerprint"]
            and existing.get("checkpoint_sha256") == protocol.sha256(checkpoint_path)
        ):
            print(f"[skip complete checkpoint] {checkpoint_path}")
            return existing
    if checkpoint_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Primary checkpoint collision: {checkpoint_path.parent}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    oce = config["oce"]
    device = config["model"]["device"]
    dtype = torch.float32
    target_prompts = protocol.expanded_prompts(spec["targets"], config)
    anchor_prompts = protocol.expanded_prompts(spec["anchors"], config)
    retain_prompts = list(spec["retain_concepts"])
    if len(target_prompts) != len(anchor_prompts):
        raise protocol.ProtocolError("Expanded target and anchor prompt counts differ")
    prompt_order = list(dict.fromkeys(target_prompts + anchor_prompts + retain_prompts))
    embeddings = {
        prompt_text: _encode_prompt(pipe, prompt_text, device, dtype)
        for prompt_text in prompt_order
    }
    target_embeddings = [embeddings[value] for value in target_prompts]
    anchor_embeddings = [embeddings[value] for value in anchor_prompts]
    retain_embeddings = [embeddings[value] for value in retain_prompts]

    modules = _projection_modules(pipe.unet)
    expected = int(config["evaluation"]["expected_checkpoint_keys"])
    if len(modules) != expected:
        raise RuntimeError(f"Resolved {len(modules)} projection modules; expected {expected}")
    updated = copy.deepcopy([module for _, module in modules])
    k0 = k0_matrix.to(device=device, dtype=dtype)
    layer_diagnostics: list[dict[str, Any]] = []
    identity_cache: dict[int, Any] = {}

    metadata: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_at": protocol.utc_now(),
        "experiment_id": config["experiment_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "mode": spec["mode"],
        "group_id": spec["group_id"],
        "targets": list(spec["targets"]),
        "anchors": list(spec["anchors"]),
        "full_anchor_mapping": dict(full_anchors),
        "retain_concepts": list(spec["retain_concepts"]),
        "expanded_target_prompts": target_prompts,
        "expanded_anchor_prompts": anchor_prompts,
        "resolved_config": config,
        "k0_path": plan["k0_path"],
        "k0_hash": k0_metadata["artifact_sha256"],
        "k0_sha256": k0_metadata["artifact_sha256"],
        "source_hashes": plan["source_hashes"],
        "runtime": protocol.runtime_provenance(),
        **protocol.git_state(),
        "requested_base_model": config["model"]["base_model"],
        "base_model_identity": {
            "pipeline": _component_identity(pipe),
            "unet": _component_identity(pipe.unet),
            "text_encoder": _component_identity(pipe.text_encoder),
            "tokenizer": _component_identity(pipe.tokenizer),
        },
    }
    protocol.write_json_atomic(metadata_path, metadata)
    try:
        for index, ((name, module), updated_module) in enumerate(zip(modules, updated), start=1):
            weight = module.weight.detach().to(device=device, dtype=dtype)
            target_basis = _subspace(weight, target_embeddings, float(oce["normalization_eps"]))
            anchor_basis = _subspace(weight, anchor_embeddings, float(oce["normalization_eps"]))
            target_projector = target_basis @ target_basis.T
            anchor_projector = anchor_basis @ anchor_basis.T
            dimension = int(weight.shape[0])
            if dimension not in identity_cache:
                identity_cache[dimension] = torch.eye(
                    dimension, device=device, dtype=dtype
                )
            identity = identity_cache[dimension]
            erasure = -float(oce["lambda_e"]) * target_projector @ (
                identity - anchor_projector
            )
            local = torch.zeros_like(erasure)
            for embedding in retain_embeddings:
                vector = weight @ embedding
                local.add_(float(oce["lambda_r"]) * torch.outer(vector, vector))
            global_prior = float(oce["lambda_0"]) * (weight @ k0 @ weight.T)
            repo_regularizer = float(oce["lamb_repo_regularizer"]) * (
                weight @ weight.T
            )
            paper_matrix = erasure + local + global_prior
            repo_matrix = paper_matrix + repo_regularizer
            repo_transform, det_before, det_after, corrected = _solve(
                repo_matrix, determinant_correction=True
            )
            paper_transform, paper_det, paper_det_after, _ = _solve(
                paper_matrix, determinant_correction=False
            )
            updated_module.weight = torch.nn.Parameter(repo_transform @ weight)

            retain_error_repo = torch.zeros((), device=device, dtype=dtype)
            retain_error_paper = torch.zeros((), device=device, dtype=dtype)
            for embedding in retain_embeddings:
                vector = weight @ embedding
                retain_error_repo += ((repo_transform @ vector) - vector).square().sum()
                retain_error_paper += ((paper_transform @ vector) - vector).square().sum()
            layer_diagnostics.append({
                "layer_index": index,
                "layer": name,
                "weight_shape": list(weight.shape),
                "target_subspace_rank": int(target_basis.shape[1]),
                "anchor_subspace_rank": int(anchor_basis.shape[1]),
                "official_repo": {
                    "frobenius_P_minus_I": _scalar(torch.linalg.matrix_norm(repo_transform - identity)),
                    "weight_delta_frobenius": _scalar(
                        torch.linalg.matrix_norm((repo_transform - identity) @ weight)
                    ),
                    "target_subspace_displacement_frobenius": _scalar(
                        torch.linalg.matrix_norm((repo_transform - identity) @ target_basis)
                    ),
                    "anchor_subspace_displacement_frobenius": _scalar(
                        torch.linalg.matrix_norm((repo_transform - identity) @ anchor_basis)
                    ),
                    "determinant_before_correction": det_before,
                    "determinant_after_correction": det_after,
                    "correction_triggered": corrected,
                    "repo_objective_trace": _scalar(torch.trace(repo_transform.T @ repo_matrix)),
                    "paper_objective_trace": _scalar(torch.trace(repo_transform.T @ paper_matrix)),
                    "objective_component_traces": {
                        "erasure": _scalar(torch.trace(repo_transform.T @ erasure)),
                        "global_prior": _scalar(torch.trace(repo_transform.T @ global_prior)),
                        "local_retain": _scalar(torch.trace(repo_transform.T @ local)),
                        "repo_regularizer": _scalar(torch.trace(repo_transform.T @ repo_regularizer)),
                    },
                    "local_retain_squared_error": _scalar(retain_error_repo),
                },
                "paper_literal": {
                    "frobenius_P_minus_I": _scalar(torch.linalg.matrix_norm(paper_transform - identity)),
                    "weight_delta_frobenius": _scalar(
                        torch.linalg.matrix_norm((paper_transform - identity) @ weight)
                    ),
                    "target_subspace_displacement_frobenius": _scalar(
                        torch.linalg.matrix_norm((paper_transform - identity) @ target_basis)
                    ),
                    "anchor_subspace_displacement_frobenius": _scalar(
                        torch.linalg.matrix_norm((paper_transform - identity) @ anchor_basis)
                    ),
                    "determinant": paper_det,
                    "determinant_after_no_correction": paper_det_after,
                    "repo_objective_trace": _scalar(torch.trace(paper_transform.T @ repo_matrix)),
                    "paper_objective_trace": _scalar(torch.trace(paper_transform.T @ paper_matrix)),
                    "objective_component_traces": {
                        "erasure": _scalar(torch.trace(paper_transform.T @ erasure)),
                        "global_prior": _scalar(torch.trace(paper_transform.T @ global_prior)),
                        "local_retain": _scalar(torch.trace(paper_transform.T @ local)),
                        "repo_regularizer": _scalar(torch.trace(paper_transform.T @ repo_regularizer)),
                    },
                    "local_retain_squared_error": _scalar(retain_error_paper),
                },
                "repo_vs_paper_literal_P_frobenius": _scalar(
                    torch.linalg.matrix_norm(repo_transform - paper_transform)
                ),
                "matrix_component_frobenius": {
                    "erasure": _scalar(torch.linalg.matrix_norm(erasure)),
                    "global_prior": _scalar(torch.linalg.matrix_norm(global_prior)),
                    "local_retain": _scalar(torch.linalg.matrix_norm(local)),
                    "repo_regularizer": _scalar(torch.linalg.matrix_norm(repo_regularizer)),
                    "paper_total": _scalar(torch.linalg.matrix_norm(paper_matrix)),
                    "repo_total": _scalar(torch.linalg.matrix_norm(repo_matrix)),
                },
            })
            print(f"[checkpoint] {spec['group_id']} {spec['mode']} layer {index}/{len(modules)}", flush=True)
    except Exception as exc:
        metadata.update(status="failed", finished_at=protocol.utc_now(), error=repr(exc))
        protocol.write_json_atomic(metadata_path, metadata)
        raise

    state = {
        f"{name}.weight": edited.weight.detach().cpu()
        for (name, _), edited in zip(modules, updated)
    }
    save_file(state, str(checkpoint_path))
    correction_count = sum(
        bool(row["official_repo"]["correction_triggered"])
        for row in layer_diagnostics
    )
    metadata.update({
        "status": "complete",
        "finished_at": protocol.utc_now(),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_hash": protocol.sha256(checkpoint_path),
        "checkpoint_sha256": protocol.sha256(checkpoint_path),
        "module_names": [f"{name}.weight" for name, _ in modules],
        "module_count": len(modules),
        "layer_diagnostics": layer_diagnostics,
        "determinant_correction_summary": {
            "trigger_count": correction_count,
            "layer_count": len(layer_diagnostics),
            "trigger_ratio": correction_count / max(len(layer_diagnostics), 1),
        },
        "discrepancy_labels": {
            "lamb_repo_regularizer": "repo implementation term; absent from final-paper Appendix C objective",
            "determinant_correction": "official-repo determinant correction; not explicitly specified in paper closed-form equation",
        },
    })
    missing_metadata = [
        key for key in config["checkpointing"]["required_metadata"]
        if key not in metadata
    ]
    if missing_metadata:
        raise protocol.ProtocolError(
            f"Checkpoint metadata contract is incomplete: {missing_metadata}"
        )
    protocol.write_json_atomic(metadata_path, metadata)
    print(f"[checkpoint complete] {checkpoint_path}")
    return metadata


def execute(config_path: Path, *, skip_existing: bool) -> dict[str, Any]:
    import torch
    from diffusers import DiffusionPipeline

    plan, config, anchors = build_plan(config_path)
    output_root = Path(config["_resolved"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(output_root / "resolved_checkpoint_plan.json", plan)
    k0_matrix, k0_metadata = validate_k0(plan, config)
    pipe = DiffusionPipeline.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
    ).to(config["model"]["device"])
    results = [
        build_checkpoint(
            pipe, spec, config, anchors, k0_matrix, k0_metadata, plan,
            skip_existing=skip_existing,
        )
        for spec in plan["checkpoints"]
    ]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": config["experiment_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "checkpoint_count": len(results),
        "single_count": sum(row["mode"] == "single" for row in results),
        "joint_count": sum(row["mode"] == "joint" for row in results),
        "determinant_correction": {
            "trigger_count": sum(row["determinant_correction_summary"]["trigger_count"] for row in results),
            "layer_count": sum(row["determinant_correction_summary"]["layer_count"] for row in results),
        },
        "checkpoints": [
            {
                "mode": row["mode"],
                "group_id": row["group_id"],
                "targets": row["targets"],
                "anchors": row["anchors"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "determinant_correction_summary": row["determinant_correction_summary"],
            }
            for row in results
        ],
        "completed_at": protocol.utc_now(),
    }
    total = summary["determinant_correction"]
    total["trigger_ratio"] = total["trigger_count"] / max(total["layer_count"], 1)
    protocol.write_json_atomic(output_root / "checkpoint_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan, _, _ = build_plan(args.config.resolve())
    if args.plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    execute(args.config.resolve(), skip_existing=args.skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
