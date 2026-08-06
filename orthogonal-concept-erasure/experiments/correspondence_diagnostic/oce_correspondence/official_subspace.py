from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .core import (
    PairSpec,
    _orthonormal_feature_basis,
    apply_weight_state,
    build_subspace_objective,
    clone_projection_state,
    collect_projection_modules,
    compute_correspondence_metrics,
    edit_projection_weights_with_rotations,
    expand_object_pairs,
    validate_experiment_sets,
)
from .io_utils import (
    image_rgb_std,
    make_seed_grid,
    plot_heatmap,
    read_csv,
    read_json,
    sha256,
    write_csv,
    write_json,
)
from .runner import ClipEvaluator, _dtype, _encode_last_content_tokens


HERE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = HERE / "config_official_subspace.json"
DEFAULT_OUTPUT = HERE / "outputs" / "official_subspace"


def pairs(config: Mapping[str, object]) -> list[PairSpec]:
    return [PairSpec(**row) for row in config["pairs"]]  # type: ignore[arg-type]


def evaluation_text(config: Mapping[str, object], concept: str) -> str:
    return str(config["evaluation"]["text_template"]).format(concept=concept)


def checkpoint(output: Path, method: str) -> Path:
    return output / "checkpoints" / f"{method}.safetensors"


def rotation_checkpoint(output: Path, method: str) -> Path:
    return output / "transformations" / f"{method}.safetensors"


def single_method(pair: PairSpec) -> str:
    return f"single_subspace_{pair.slug}"


def _load_config(config_path: Path, output: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    set_validation = validate_experiment_sets(
        pairs(config), list(config["control_concepts"])
    )
    expected_prompts = {
        pair.target: f"a photo of a {pair.target}" for pair in pairs(config)
    }
    actual_prompts = {pair.target: pair.prompt for pair in pairs(config)}
    if actual_prompts != expected_prompts:
        raise ValueError(
            "Every target prompt must exactly use 'a photo of a {concept}'"
        )
    for phase in ("feasibility", "screening", "joint"):
        values = list(config["seeds"][phase])
        if not 10 <= len(values) <= 20 or len(values) != len(set(values)):
            raise ValueError(f"{phase} requires 10-20 unique fixed seeds")
    cg_path = (config_path.parent / str(config["cg_path"])).resolve()
    payload = torch.load(cg_path, map_location="cpu")
    if tuple(payload["C"].shape) != (768, 768):
        raise ValueError(f"Unexpected Cg tensor at {cg_path}")
    resolved = dict(config)
    resolved.update(
        {
            "config_path": str(config_path.resolve()),
            "output_dir": str(output.resolve()),
            "cg_path": str(cg_path),
            "cg_sha256": sha256(cg_path),
            "cg_count": payload.get("count"),
            "set_validation": set_validation,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
            },
        }
    )
    return resolved


def preflight(config_path: Path, output: Path) -> dict[str, object]:
    resolved = _load_config(config_path, output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", resolved)
    write_csv(
        output / "inputs" / "target_anchor_pairs.csv",
        [
            {
                "pair_index": index,
                "target": pair.target,
                "anchor": pair.anchor,
                "prompt": pair.prompt,
            }
            for index, pair in enumerate(pairs(resolved), 1)
        ],
    )
    concepts = [pair.target for pair in pairs(resolved)] + [
        pair.anchor for pair in pairs(resolved)
    ]
    write_csv(
        output / "inputs" / "prompts.csv",
        [
            {
                "kind": "target" if index < 5 else "anchor",
                "concept": concept,
                "prompt": evaluation_text(resolved, concept),
            }
            for index, concept in enumerate(concepts)
        ]
        + [
            {
                "kind": "control",
                "concept": concept,
                "prompt": evaluation_text(resolved, concept),
            }
            for concept in resolved["control_concepts"]
        ],
    )
    write_csv(
        output / "inputs" / "seeds.csv",
        [
            {"phase": phase, "sample_index": index, "seed": seed}
            for phase, values in resolved["seeds"].items()
            for index, seed in enumerate(values)
        ],
    )
    state = {
        "preflight": "complete",
        "tokenizer": "pending",
        "original_feasibility": "pending",
        "single_pair_subspace": "pending",
        "n2_selection": "pending",
        "n2_feature_evaluation": "pending",
        "n2_image_evaluation": "pending",
        "n2_permutation_check": "pending",
        "control_set": "not_run_current_stage",
        "n5": "not_run_current_stage",
    }
    write_json(output / "run_state.json", state)
    return resolved


def resolved(config_path: Path, output: Path) -> dict[str, object]:
    path = output / "resolved_config.json"
    if path.exists():
        return read_json(path)  # type: ignore[return-value]
    return preflight(config_path, output)


def tokenizer_audit(config: Mapping[str, object], output: Path) -> None:
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        str(config["model_id"]), subfolder="tokenizer", local_files_only=True
    )
    concepts = [pair.target for pair in pairs(config)] + [
        pair.anchor for pair in pairs(config)
    ]
    rows = []
    for concept in concepts:
        ids = tokenizer(concept, add_special_tokens=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        rows.append(
            {
                "concept": concept,
                "token_ids": json.dumps(ids),
                "token_strings": json.dumps(tokens, ensure_ascii=False),
                "token_count": len(ids),
                "split_into_multiple_tokens": len(ids) > 1,
            }
        )
    write_csv(output / "metrics" / "tokenization.csv", rows)
    write_json(output / "metrics" / "tokenization.json", {"concepts": rows})
    state = read_json(output / "run_state.json")
    state["tokenizer"] = "complete"
    write_json(output / "run_state.json", state)


def feasibility_image(output: Path, concept: str, seed: int) -> Path:
    return output / "images" / "feasibility" / concept / f"seed_{seed}.png"


@torch.inference_mode()
def generate_feasibility(config: Mapping[str, object], output: Path) -> None:
    from diffusers import DiffusionPipeline

    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=_dtype(str(config["generation_dtype"])),
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    generation = config["generation"]
    seeds = list(config["seeds"]["feasibility"])
    concepts = [pair.target for pair in pairs(config)] + [
        pair.anchor for pair in pairs(config)
    ]
    for concept in concepts:
        for index, seed in enumerate(seeds, 1):
            path = feasibility_image(output, concept, seed)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            image = pipe(
                prompt=evaluation_text(config, concept),
                num_inference_steps=int(generation["num_inference_steps"]),
                guidance_scale=float(generation["guidance_scale"]),
                height=int(generation["height"]),
                width=int(generation["width"]),
                generator=torch.Generator(device=device).manual_seed(int(seed)),
            ).images[0]
            image.save(path)
            print(f"[feasibility] {concept} {index}/{len(seeds)} seed={seed}", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_feasibility(config: Mapping[str, object], output: Path) -> None:
    seeds = list(config["seeds"]["feasibility"])
    concepts = [pair.target for pair in pairs(config)] + [
        pair.anchor for pair in pairs(config)
    ]
    evaluator = ClipEvaluator(str(config["clip_model_id"]), str(config["device"]))
    per_image = []
    summary = []
    for concept in concepts:
        paths = [feasibility_image(output, concept, seed) for seed in seeds]
        values = evaluator.similarities(
            paths, [evaluation_text(config, concept)]
        )[:, 0]
        rgb = np.array([image_rgb_std(path) for path in paths])
        for seed, path, similarity, std in zip(seeds, paths, values, rgb):
            per_image.append(
                {
                    "concept": concept,
                    "seed": seed,
                    "prompt": evaluation_text(config, concept),
                    "clip_prompt_alignment": float(similarity),
                    "rgb_std": float(std),
                    "image_path": str(path.resolve()),
                }
            )
        summary.append(
            {
                "concept": concept,
                "kind": "target" if concept in [p.target for p in pairs(config)] else "anchor",
                "mean_clip_prompt_alignment": float(values.mean()),
                "std_clip_prompt_alignment": float(values.std(ddof=1)),
                "minimum_clip_prompt_alignment": float(values.min()),
                "mean_rgb_std": float(rgb.mean()),
            }
        )
        make_seed_grid(
            output / "grids" / "feasibility" / f"{concept}.png",
            {"Original SD": paths},
            seeds,
            f"Original SD feasibility | {evaluation_text(config, concept)}",
        )
    write_csv(output / "metrics" / "feasibility_per_image.csv", per_image)
    write_csv(output / "metrics" / "feasibility_summary.csv", summary)
    write_json(output / "metrics" / "feasibility_summary.json", {"concepts": summary})
    state = read_json(output / "run_state.json")
    state["original_feasibility"] = "complete"
    write_json(output / "run_state.json", state)
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()


def _expanded(pair_list: Sequence[PairSpec], config: Mapping[str, object]) -> list[PairSpec]:
    return expand_object_pairs(pair_list) if config["oce"]["expand_prompts"] else list(pair_list)


@torch.inference_mode()
def _prepare_methods(
    config: Mapping[str, object],
    output: Path,
    specifications: Sequence[tuple[str, str, Sequence[PairSpec]]],
) -> None:
    from diffusers import DiffusionPipeline

    pending = [
        specification
        for specification in specifications
        if not checkpoint(output, specification[0]).exists()
        or not rotation_checkpoint(output, specification[0]).exists()
    ]
    if not pending:
        return
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=_dtype(str(config["edit_dtype"])),
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)
    if len(collect_projection_modules(pipe.unet)) != 16:
        raise RuntimeError("Official OCE configuration expects 16 attn2.to_v layers")
    all_text = []
    for _, _, pair_list in pending:
        for pair in _expanded(pair_list, config):
            all_text.extend([pair.target, pair.anchor])
    all_text.extend(anchor for _, _, pair_list in pending for anchor in [p.anchor for p in pair_list])
    embeddings = _encode_last_content_tokens(pipe, all_text, device)
    cg = torch.load(str(config["cg_path"]), map_location=device)["C"].float()
    oce = config["oce"]
    audit_rows = []
    for method, objective, base_pairs in pending:
        method_pairs = _expanded(base_pairs, config)
        preserve = (
            [pair.anchor for pair in base_pairs]
            if oce["preserve_anchors"]
            else []
        )
        weights, rotations, audit = edit_projection_weights_with_rotations(
            unet=pipe.unet,
            embeddings=embeddings,
            pairs=method_pairs,
            preserve_concepts=preserve,
            global_second_moment=cg,
            objective=objective,
            erase_scale=float(oce["erase_scale"]),
            preserve_global_scale=float(oce["preserve_global_scale"]),
            preserve_concept_scale=float(oce["preserve_concept_scale"]),
            lamb=float(oce["lambda"]),
            reflection_correction=str(oce["reflection_correction"]),
        )
        checkpoint(output, method).parent.mkdir(parents=True, exist_ok=True)
        rotation_checkpoint(output, method).parent.mkdir(parents=True, exist_ok=True)
        save_file(weights, str(checkpoint(output, method)))
        save_file(rotations, str(rotation_checkpoint(output, method)))
        for row in audit:
            row.update(
                {
                    "method": method,
                    "role": (
                        "official_oce_baseline"
                        if objective == "subspace"
                        else "eq6_ablation"
                    ),
                    "checkpoint_sha256": sha256(checkpoint(output, method)),
                    "rotation_sha256": sha256(rotation_checkpoint(output, method)),
                }
            )
            audit_rows.append(row)
        print(f"[checkpoint] {method}", flush=True)
    audit_path = output / "metrics" / "weight_audit.csv"
    previous = read_csv(audit_path) if audit_path.exists() else []
    replaced = {row["method"] for row in audit_rows}
    write_csv(audit_path, [row for row in previous if row["method"] not in replaced] + audit_rows)
    del pipe, cg
    gc.collect()
    torch.cuda.empty_cache()


def prepare_single_weights(config: Mapping[str, object], output: Path) -> None:
    _prepare_methods(
        config,
        output,
        [(single_method(pair), "subspace", [pair]) for pair in pairs(config)],
    )


def single_image(output: Path, pair: PairSpec, method: str, seed: int) -> Path:
    return output / "images" / "single_pair" / pair.slug / method / f"seed_{seed}.png"


@torch.inference_mode()
def generate_single(config: Mapping[str, object], output: Path) -> None:
    from diffusers import DiffusionPipeline

    prepare_single_weights(config, output)
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=_dtype(str(config["generation_dtype"])),
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    generation = config["generation"]
    seeds = list(config["seeds"]["screening"])
    for pair in pairs(config):
        method = single_method(pair)
        apply_weight_state(pipe.unet, load_file(str(checkpoint(output, method))))
        for index, seed in enumerate(seeds, 1):
            path = single_image(output, pair, method, seed)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            image = pipe(
                prompt=pair.prompt,
                num_inference_steps=int(generation["num_inference_steps"]),
                guidance_scale=float(generation["guidance_scale"]),
                height=int(generation["height"]),
                width=int(generation["width"]),
                generator=torch.Generator(device=device).manual_seed(int(seed)),
            ).images[0]
            image.save(path)
            print(f"[single] {pair.slug} {index}/{len(seeds)} seed={seed}", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_single(config: Mapping[str, object], output: Path) -> None:
    pair_list = pairs(config)
    seeds = list(config["seeds"]["screening"])
    anchors = [pair.anchor for pair in pair_list]
    anchor_texts = [evaluation_text(config, anchor) for anchor in anchors]
    evaluator = ClipEvaluator(str(config["clip_model_id"]), str(config["device"]))
    per_image = []
    summary = []
    rule = config["evaluation"]["screening_rule"]
    for pair_index, pair in enumerate(pair_list):
        method = single_method(pair)
        original_paths = [feasibility_image(output, pair.target, seed) for seed in seeds]
        edited_paths = [single_image(output, pair, method, seed) for seed in seeds]
        target_text = [evaluation_text(config, pair.target)]
        original_target = evaluator.similarities(original_paths, target_text)[:, 0]
        edited_target = evaluator.similarities(edited_paths, target_text)[:, 0]
        original_anchors = evaluator.similarities(original_paths, anchor_texts)
        edited_anchors = evaluator.similarities(edited_paths, anchor_texts)
        for label, paths, targets, anchor_values in (
            ("original", original_paths, original_target, original_anchors),
            ("single_subspace", edited_paths, edited_target, edited_anchors),
        ):
            prediction = anchor_values.argmax(axis=1)
            others = np.delete(anchor_values, pair_index, axis=1)
            margins = anchor_values[:, pair_index] - others.max(axis=1)
            for sample_index, (seed, path) in enumerate(zip(seeds, paths)):
                row = {
                    "pair_index": pair_index,
                    "pair": pair.slug,
                    "target": pair.target,
                    "anchor": pair.anchor,
                    "method": label,
                    "seed": seed,
                    "target_similarity": float(targets[sample_index]),
                    "own_anchor_similarity": float(anchor_values[sample_index, pair_index]),
                    "best_other_anchor_similarity": float(others[sample_index].max()),
                    "correspondence_margin": float(margins[sample_index]),
                    "predicted_anchor": anchors[int(prediction[sample_index])],
                    "own_anchor_top1": bool(prediction[sample_index] == pair_index),
                    "rgb_std": image_rgb_std(path),
                    "image_path": str(path.resolve()),
                }
                for anchor_index, anchor in enumerate(anchors):
                    row[f"similarity_anchor_{anchor}"] = float(
                        anchor_values[sample_index, anchor_index]
                    )
                per_image.append(row)
        edited_other = np.delete(edited_anchors, pair_index, axis=1)
        edited_margin = edited_anchors[:, pair_index] - edited_other.max(axis=1)
        target_delta = float(edited_target.mean() - original_target.mean())
        anchor_delta = float(
            edited_anchors[:, pair_index].mean()
            - original_anchors[:, pair_index].mean()
        )
        low_variance = float(
            np.mean(
                [
                    image_rgb_std(path)
                    < float(rule["low_variance_rgb_std_threshold"])
                    for path in edited_paths
                ]
            )
        )
        screening_pass = (
            target_delta < float(rule["target_similarity_delta_must_be_below"])
            and anchor_delta > float(rule["anchor_similarity_delta_must_be_above"])
            and low_variance <= float(rule["max_low_variance_fraction"])
        )
        summary.append(
            {
                "pair_index": pair_index,
                "pair": pair.slug,
                "target": pair.target,
                "anchor": pair.anchor,
                "original_mean_target_similarity": float(original_target.mean()),
                "edited_mean_target_similarity": float(edited_target.mean()),
                "target_similarity_delta": target_delta,
                "original_mean_own_anchor_similarity": float(
                    original_anchors[:, pair_index].mean()
                ),
                "edited_mean_own_anchor_similarity": float(
                    edited_anchors[:, pair_index].mean()
                ),
                "own_anchor_similarity_delta": anchor_delta,
                "edited_mean_best_other_similarity": float(edited_other.max(axis=1).mean()),
                "edited_mean_margin": float(edited_margin.mean()),
                "edited_minimum_margin": float(edited_margin.min()),
                "edited_positive_margin_fraction": float(np.mean(edited_margin > 0)),
                "edited_own_anchor_top1_rate": float(
                    np.mean(edited_anchors.argmax(axis=1) == pair_index)
                ),
                "low_variance_fraction": low_variance,
                "directional_screening_pass": screening_pass,
            }
        )
        make_seed_grid(
            output / "grids" / "single_pair" / f"{pair.slug}.png",
            {"Original SD": original_paths, "Single Subspace": edited_paths},
            seeds,
            f"Official OCE single-pair subspace | {pair.target} -> {pair.anchor}",
        )
    write_csv(output / "metrics" / "single_pair_per_image.csv", per_image)
    write_csv(output / "metrics" / "single_pair_summary.csv", summary)
    write_json(
        output / "metrics" / "single_pair_summary.json",
        {
            "screening_rule": rule,
            "pairs": summary,
            "note": "Labels are descriptive; no pair is automatically removed.",
        },
    )
    state = read_json(output / "run_state.json")
    state["single_pair_subspace"] = "complete"
    write_json(output / "run_state.json", state)
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()


def select_n2(config: Mapping[str, object], output: Path, slugs: Sequence[str] | None) -> list[PairSpec]:
    pair_by_slug = {pair.slug: pair for pair in pairs(config)}
    rows = read_csv(output / "metrics" / "single_pair_summary.csv")
    if slugs:
        if len(slugs) != 2 or any(slug not in pair_by_slug for slug in slugs):
            raise ValueError("--n2-pairs requires exactly two valid pair slugs")
        selected = [pair_by_slug[slug] for slug in slugs]
        basis = "manual selection after numeric and grid review"
    else:
        # Transparent fallback: directionality first, then top-1, margin,
        # target reduction, and own-anchor gain. Semantic separation is reviewed
        # before accepting this output for the joint run.
        ranked = sorted(
            rows,
            key=lambda row: (
                str(row["directional_screening_pass"]).casefold() == "true",
                float(row["edited_own_anchor_top1_rate"]),
                float(row["edited_mean_margin"]),
                -float(row["target_similarity_delta"]),
                float(row["own_anchor_similarity_delta"]),
            ),
            reverse=True,
        )
        selected = [pair_by_slug[row["pair"]] for row in ranked[:2]]
        basis = "deterministic screening ranking; semantic separation requires review"
    payload = {
        "pairs": [
            {"target": pair.target, "anchor": pair.anchor, "slug": pair.slug}
            for pair in selected
        ],
        "selection_basis": basis,
        "visual_review": (
            read_json(output / "metrics" / "single_pair_visual_review.json")
            if (output / "metrics" / "single_pair_visual_review.json").exists()
            else {"status": "not_recorded"}
        ),
        "screening_evidence": [
            row for row in rows if row["pair"] in {pair.slug for pair in selected}
        ],
    }
    write_json(output / "inputs" / "n2_selection.json", payload)
    state = read_json(output / "run_state.json")
    state["n2_selection"] = "complete"
    write_json(output / "run_state.json", state)
    return selected


def selected_n2(output: Path) -> list[PairSpec]:
    payload = read_json(output / "inputs" / "n2_selection.json")
    return [
        PairSpec(row["target"], row["anchor"], f"a photo of a {row['target']}")
        for row in payload["pairs"]
    ]


def prepare_n2(config: Mapping[str, object], output: Path) -> None:
    selected = selected_n2(output)
    swapped = [
        PairSpec(selected[0].target, selected[1].anchor, selected[0].prompt),
        PairSpec(selected[1].target, selected[0].anchor, selected[1].prompt),
    ]
    _prepare_methods(
        config,
        output,
        [
            ("joint_subspace_n2", "subspace", selected),
            ("joint_vector_eq6_n2", "vector", selected),
            ("joint_subspace_n2_anchor_permuted", "subspace", swapped),
        ],
    )


def n2_image(output: Path, pair: PairSpec, method: str, seed: int) -> Path:
    return output / "images" / "joint_n2" / pair.slug / method / f"seed_{seed}.png"


def n2_paths(
    config: Mapping[str, object], output: Path, pair: PairSpec, method: str
) -> list[Path]:
    seeds = list(config["seeds"]["joint"])
    if method == "original":
        return [feasibility_image(output, pair.target, seed) for seed in seeds]
    if method == "single_subspace":
        return [single_image(output, pair, single_method(pair), seed) for seed in seeds]
    return [n2_image(output, pair, method, seed) for seed in seeds]


@torch.inference_mode()
def evaluate_n2_features(config: Mapping[str, object], output: Path) -> None:
    from diffusers import DiffusionPipeline

    selected = selected_n2(output)
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)
    embeddings = _encode_last_content_tokens(
        pipe,
        [value for pair in selected for value in (pair.target, pair.anchor)],
        device,
    )
    joint_states = {
        method: load_file(str(checkpoint(output, method)), device=str(device))
        for method in ("joint_subspace_n2", "joint_vector_eq6_n2")
    }
    single_states = {
        pair.slug: load_file(
            str(checkpoint(output, single_method(pair))), device=str(device)
        )
        for pair in selected
    }
    cells = []
    summaries = []
    for layer_index, (layer, module) in enumerate(collect_projection_modules(pipe.unet), 1):
        base_weight = module.weight.detach().float()
        anchor_features = torch.stack(
            [base_weight @ embeddings[pair.anchor] for pair in selected]
        )
        for method in (
            "original",
            "single_subspace",
            "joint_subspace_n2",
            "joint_vector_eq6_n2",
        ):
            target_features = []
            for pair in selected:
                if method == "original":
                    weight = base_weight
                elif method == "single_subspace":
                    weight = single_states[pair.slug][f"{layer}.weight"]
                else:
                    weight = joint_states[method][f"{layer}.weight"]
                target_features.append(weight @ embeddings[pair.target])
            target_matrix = torch.nn.functional.normalize(
                torch.stack(target_features).float(), dim=1
            )
            anchor_matrix = torch.nn.functional.normalize(anchor_features.float(), dim=1)
            matrix = (target_matrix @ anchor_matrix.T).cpu().numpy()
            _, aggregate = compute_correspondence_metrics(matrix[:, None, :])
            summaries.append(
                {
                    "layer_index": layer_index,
                    "layer": layer,
                    "method": method,
                    **aggregate,
                }
            )
            for target_index, pair in enumerate(selected):
                for anchor_index, anchor_pair in enumerate(selected):
                    cells.append(
                        {
                            "layer_index": layer_index,
                            "layer": layer,
                            "method": method,
                            "target_index": target_index,
                            "target": pair.target,
                            "anchor_index": anchor_index,
                            "anchor": anchor_pair.anchor,
                            "similarity": float(matrix[target_index, anchor_index]),
                            "is_own_anchor": target_index == anchor_index,
                        }
                    )
            plot_heatmap(
                output / "heatmaps" / "joint_n2" / "feature" / f"layer_{layer_index:02d}_{method}.png",
                matrix,
                [pair.target for pair in selected],
                [pair.anchor for pair in selected],
                f"N=2 feature correspondence | L{layer_index} | {method}",
            )
    write_csv(output / "metrics" / "n2_feature_cells.csv", cells)
    write_csv(output / "metrics" / "n2_feature_summary.csv", summaries)
    write_json(output / "metrics" / "n2_feature_summary.json", {"layers": summaries})
    state = read_json(output / "run_state.json")
    state["n2_feature_evaluation"] = "complete_before_image_generation"
    write_json(output / "run_state.json", state)
    del pipe, joint_states, single_states
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_n2(config: Mapping[str, object], output: Path) -> None:
    from diffusers import DiffusionPipeline

    selected = selected_n2(output)
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=_dtype(str(config["generation_dtype"])),
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    generation = config["generation"]
    seeds = list(config["seeds"]["joint"])
    for method in ("joint_subspace_n2", "joint_vector_eq6_n2"):
        apply_weight_state(pipe.unet, load_file(str(checkpoint(output, method))))
        for pair in selected:
            for index, seed in enumerate(seeds, 1):
                path = n2_image(output, pair, method, seed)
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                image = pipe(
                    prompt=pair.prompt,
                    num_inference_steps=int(generation["num_inference_steps"]),
                    guidance_scale=float(generation["guidance_scale"]),
                    height=int(generation["height"]),
                    width=int(generation["width"]),
                    generator=torch.Generator(device=device).manual_seed(int(seed)),
                ).images[0]
                image.save(path)
                print(f"[N2] {method} {pair.slug} {index}/{len(seeds)} seed={seed}", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_n2_images(config: Mapping[str, object], output: Path) -> None:
    selected = selected_n2(output)
    seeds = list(config["seeds"]["joint"])
    methods = (
        "original",
        "single_subspace",
        "joint_subspace_n2",
        "joint_vector_eq6_n2",
    )
    anchor_texts = [evaluation_text(config, pair.anchor) for pair in selected]
    evaluator = ClipEvaluator(str(config["clip_model_id"]), str(config["device"]))
    per_image = []
    cells = []
    confusion_rows = []
    per_concept = []
    aggregate_rows = []
    for method in methods:
        cube = np.empty((2, len(seeds), 2), dtype=np.float32)
        target_scores = np.empty((2, len(seeds)), dtype=np.float32)
        paths_by_pair = {}
        for target_index, pair in enumerate(selected):
            paths = n2_paths(config, output, pair, method)
            paths_by_pair[pair.slug] = paths
            cube[target_index] = evaluator.similarities(paths, anchor_texts)
            target_scores[target_index] = evaluator.similarities(
                paths, [evaluation_text(config, pair.target)]
            )[:, 0]
        rows, aggregate = compute_correspondence_metrics(cube)
        aggregate_rows.append({"method": method, **aggregate})
        predictions = cube.argmax(axis=2)
        for row in rows:
            ti, si = int(row["target_index"]), int(row["sample_index"])
            pair = selected[ti]
            per_image.append(
                {
                    "method": method,
                    "target": pair.target,
                    "anchor": pair.anchor,
                    "seed": seeds[si],
                    "target_similarity": float(target_scores[ti, si]),
                    **{key: value for key, value in row.items() if key not in {"target_index", "sample_index"}},
                    "predicted_anchor": selected[int(row["predicted_anchor_index"])].anchor,
                    "image_path": str(paths_by_pair[pair.slug][si].resolve()),
                }
            )
        matrix = cube.mean(axis=1)
        plot_heatmap(
            output / "heatmaps" / "joint_n2" / f"image_similarity_{method}.png",
            matrix,
            [pair.target for pair in selected],
            [pair.anchor for pair in selected],
            f"N=2 image-to-anchor similarity | {method}",
        )
        confusion = np.zeros((2, 2), dtype=np.int64)
        for ti in range(2):
            for prediction in predictions[ti]:
                confusion[ti, prediction] += 1
        plot_heatmap(
            output / "heatmaps" / "joint_n2" / f"image_confusion_{method}.png",
            confusion,
            [pair.target for pair in selected],
            [pair.anchor for pair in selected],
            f"N=2 anchor top-1 counts | {method}",
            value_format=".0f",
        )
        for ti, pair in enumerate(selected):
            own = cube[ti, :, ti]
            other = cube[ti, :, 1 - ti]
            margin = own - other
            per_concept.append(
                {
                    "method": method,
                    "target": pair.target,
                    "anchor": pair.anchor,
                    "mean_target_similarity": float(target_scores[ti].mean()),
                    "mean_own_anchor_similarity": float(own.mean()),
                    "mean_best_other_anchor_similarity": float(other.mean()),
                    "mean_margin": float(margin.mean()),
                    "minimum_margin": float(margin.min()),
                    "positive_margin_fraction": float(np.mean(margin > 0)),
                    "own_anchor_top1_rate": float(np.mean(predictions[ti] == ti)),
                }
            )
            for si, seed in enumerate(seeds):
                for ai, anchor_pair in enumerate(selected):
                    cells.append(
                        {
                            "method": method,
                            "target": pair.target,
                            "seed": seed,
                            "anchor": anchor_pair.anchor,
                            "similarity": float(cube[ti, si, ai]),
                            "is_own_anchor": ti == ai,
                        }
                    )
            for ai, anchor_pair in enumerate(selected):
                confusion_rows.append(
                    {
                        "method": method,
                        "target": pair.target,
                        "predicted_anchor": anchor_pair.anchor,
                        "count": int(confusion[ti, ai]),
                    }
                )
    for pair in selected:
        make_seed_grid(
            output / "grids" / "joint_n2" / f"{pair.slug}.png",
            {
                "Original SD": n2_paths(config, output, pair, "original"),
                "Single Subspace": n2_paths(config, output, pair, "single_subspace"),
                "Joint Subspace": n2_paths(config, output, pair, "joint_subspace_n2"),
                "Eq. 6 Ablation": n2_paths(config, output, pair, "joint_vector_eq6_n2"),
            },
            seeds,
            f"N=2 correspondence | {pair.target} -> {pair.anchor}",
        )
    write_csv(output / "metrics" / "n2_per_image.csv", per_image)
    write_csv(output / "metrics" / "n2_image_similarity_cells.csv", cells)
    write_csv(output / "metrics" / "n2_image_confusion.csv", confusion_rows)
    write_csv(output / "metrics" / "n2_per_concept.csv", per_concept)
    write_csv(output / "metrics" / "n2_aggregate.csv", aggregate_rows)
    write_json(
        output / "metrics" / "n2_image_summary.json",
        {"aggregate": aggregate_rows, "per_concept": per_concept},
    )
    state = read_json(output / "run_state.json")
    state["n2_image_evaluation"] = "complete"
    write_json(output / "run_state.json", state)
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()


def permutation_check(output: Path) -> None:
    reference_method = "joint_subspace_n2"
    permuted_method = "joint_subspace_n2_anchor_permuted"
    reference_weights = load_file(str(checkpoint(output, reference_method)))
    permuted_weights = load_file(str(checkpoint(output, permuted_method)))
    reference_rotations = load_file(str(rotation_checkpoint(output, reference_method)))
    permuted_rotations = load_file(str(rotation_checkpoint(output, permuted_method)))
    rows = []
    for weight_key in sorted(reference_weights):
        layer = weight_key.removesuffix(".weight")
        rotation_key = f"{layer}.rotation"
        weight_a = reference_weights[weight_key].float()
        weight_b = permuted_weights[weight_key].float()
        rotation_a = reference_rotations[rotation_key].float()
        rotation_b = permuted_rotations[rotation_key].float()
        weight_diff = weight_a - weight_b
        rotation_diff = rotation_a - rotation_b
        rows.append(
            {
                "layer": layer,
                "weight_max_abs_difference": float(weight_diff.abs().max()),
                "weight_relative_frobenius_difference": float(
                    torch.linalg.matrix_norm(weight_diff)
                    / torch.linalg.matrix_norm(weight_a).clamp_min(1e-12)
                ),
                "rotation_max_abs_difference": float(rotation_diff.abs().max()),
                "rotation_relative_frobenius_difference": float(
                    torch.linalg.matrix_norm(rotation_diff)
                    / torch.linalg.matrix_norm(rotation_a).clamp_min(1e-12)
                ),
                "weight_exact_equal": bool(torch.equal(weight_a, weight_b)),
                "rotation_exact_equal": bool(torch.equal(rotation_a, rotation_b)),
                "weight_allclose_rtol1e-5_atol1e-6": bool(
                    torch.allclose(weight_a, weight_b, rtol=1e-5, atol=1e-6)
                ),
                "rotation_allclose_rtol1e-5_atol1e-6": bool(
                    torch.allclose(rotation_a, rotation_b, rtol=1e-5, atol=1e-6)
                ),
            }
        )
    summary = {
        "reference_checkpoint_sha256": sha256(checkpoint(output, reference_method)),
        "permuted_checkpoint_sha256": sha256(checkpoint(output, permuted_method)),
        "reference_rotation_sha256": sha256(rotation_checkpoint(output, reference_method)),
        "permuted_rotation_sha256": sha256(rotation_checkpoint(output, permuted_method)),
        "checkpoint_hash_equal": sha256(checkpoint(output, reference_method))
        == sha256(checkpoint(output, permuted_method)),
        "rotation_hash_equal": sha256(rotation_checkpoint(output, reference_method))
        == sha256(rotation_checkpoint(output, permuted_method)),
        "all_weights_exact_equal": all(row["weight_exact_equal"] for row in rows),
        "all_rotations_exact_equal": all(row["rotation_exact_equal"] for row in rows),
        "all_weights_numerically_close": all(
            row["weight_allclose_rtol1e-5_atol1e-6"] for row in rows
        ),
        "all_rotations_numerically_close": all(
            row["rotation_allclose_rtol1e-5_atol1e-6"] for row in rows
        ),
        "maximum_weight_abs_difference": max(
            row["weight_max_abs_difference"] for row in rows
        ),
        "maximum_rotation_abs_difference": max(
            row["rotation_max_abs_difference"] for row in rows
        ),
        "interpretation_rule": (
            "Permutation insensitivity is supported when all layer weights and "
            "rotations are numerically close; byte hashes may differ because "
            "QR/SVD computations are floating-point order-sensitive."
        ),
    }
    write_csv(output / "metrics" / "n2_permutation_layer_comparison.csv", rows)
    write_json(
        output / "metrics" / "n2_permutation_check.json",
        {"summary": summary, "layers": rows},
    )
    state = read_json(output / "run_state.json")
    state["n2_permutation_check"] = "complete"
    write_json(output / "run_state.json", state)


@torch.inference_mode()
def permutation_objective_check(
    config: Mapping[str, object], output: Path
) -> None:
    """Locate whether permutation differences arise before or after Procrustes."""
    from diffusers import DiffusionPipeline

    selected = selected_n2(output)
    swapped = [
        PairSpec(selected[0].target, selected[1].anchor, selected[0].prompt),
        PairSpec(selected[1].target, selected[0].anchor, selected[1].prompt),
    ]
    reference_pairs = _expanded(selected, config)
    permuted_pairs = _expanded(swapped, config)
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)
    all_text = [
        value
        for pair_list in (reference_pairs, permuted_pairs)
        for pair in pair_list
        for value in (pair.target, pair.anchor)
    ]
    embeddings = _encode_last_content_tokens(pipe, all_text, device)
    cg = torch.load(str(config["cg_path"]), map_location=device)["C"].float()
    oce = config["oce"]
    rows = []
    for layer_index, (layer, module) in enumerate(
        collect_projection_modules(pipe.unet), 1
    ):
        weight = module.weight.detach().float()
        targets = [embeddings[pair.target] for pair in reference_pairs]
        reference_anchors = [embeddings[pair.anchor] for pair in reference_pairs]
        permuted_anchors = [embeddings[pair.anchor] for pair in permuted_pairs]
        reference_basis = _orthonormal_feature_basis(
            weight, reference_anchors
        )
        permuted_basis = _orthonormal_feature_basis(
            weight, permuted_anchors
        )
        reference_projector = reference_basis @ reference_basis.T
        permuted_projector = permuted_basis @ permuted_basis.T
        reference_objective = build_subspace_objective(
            weight,
            targets,
            reference_anchors,
            [embeddings[pair.anchor] for pair in selected],
            cg,
            erase_scale=float(oce["erase_scale"]),
            preserve_global_scale=float(oce["preserve_global_scale"]),
            preserve_concept_scale=float(oce["preserve_concept_scale"]),
            lamb=float(oce["lambda"]),
        )
        permuted_objective = build_subspace_objective(
            weight,
            targets,
            permuted_anchors,
            [embeddings[pair.anchor] for pair in swapped],
            cg,
            erase_scale=float(oce["erase_scale"]),
            preserve_global_scale=float(oce["preserve_global_scale"]),
            preserve_concept_scale=float(oce["preserve_concept_scale"]),
            lamb=float(oce["lambda"]),
        )
        feature_columns = []
        for embedding in reference_anchors:
            feature = weight @ embedding
            feature_columns.append(
                feature / torch.linalg.vector_norm(feature).clamp_min(1e-8)
            )
        feature_matrix = torch.stack(feature_columns, dim=1)
        singular_values = torch.linalg.svdvals(feature_matrix)
        projector_diff = reference_projector - permuted_projector
        objective_diff = reference_objective - permuted_objective
        rows.append(
            {
                "layer_index": layer_index,
                "layer": layer,
                "anchor_column_count": len(reference_anchors),
                "anchor_numeric_rank_default": int(
                    torch.linalg.matrix_rank(feature_matrix).item()
                ),
                "anchor_largest_singular_value": float(singular_values.max()),
                "anchor_smallest_singular_value": float(singular_values.min()),
                "anchor_condition_number": float(
                    singular_values.max()
                    / singular_values.min().clamp_min(1e-30)
                ),
                "anchor_projector_max_abs_difference": float(
                    projector_diff.abs().max()
                ),
                "anchor_projector_relative_frobenius_difference": float(
                    torch.linalg.matrix_norm(projector_diff)
                    / torch.linalg.matrix_norm(reference_projector).clamp_min(1e-12)
                ),
                "objective_max_abs_difference": float(
                    objective_diff.abs().max()
                ),
                "objective_relative_frobenius_difference": float(
                    torch.linalg.matrix_norm(objective_diff)
                    / torch.linalg.matrix_norm(reference_objective).clamp_min(1e-12)
                ),
            }
        )
    write_csv(
        output / "metrics" / "n2_permutation_objective_comparison.csv", rows
    )
    path = output / "metrics" / "n2_permutation_check.json"
    payload = read_json(path)
    payload["objective_diagnostic"] = {
        "maximum_anchor_projector_abs_difference": max(
            row["anchor_projector_max_abs_difference"] for row in rows
        ),
        "maximum_objective_abs_difference": max(
            row["objective_max_abs_difference"] for row in rows
        ),
        "maximum_objective_relative_frobenius_difference": max(
            row["objective_relative_frobenius_difference"] for row in rows
        ),
        "minimum_anchor_numeric_rank": min(
            row["anchor_numeric_rank_default"] for row in rows
        ),
        "maximum_anchor_condition_number": max(
            row["anchor_condition_number"] for row in rows
        ),
        "layers": rows,
    }
    write_json(path, payload)
    del pipe, cg
    gc.collect()
    torch.cuda.empty_cache()


def build_report(config: Mapping[str, object], output: Path) -> Path:
    token_rows = read_csv(output / "metrics" / "tokenization.csv")
    feasibility = read_csv(output / "metrics" / "feasibility_summary.csv")
    screening = read_csv(output / "metrics" / "single_pair_summary.csv")
    visual_review = read_json(output / "metrics" / "single_pair_visual_review.json")
    n2_visual_review = read_json(output / "metrics" / "n2_visual_review.json")
    selection = read_json(output / "inputs" / "n2_selection.json")
    n2 = read_csv(output / "metrics" / "n2_per_concept.csv")
    n2_aggregate = read_csv(output / "metrics" / "n2_aggregate.csv")
    feature = read_csv(output / "metrics" / "n2_feature_summary.csv")
    permutation = read_json(output / "metrics" / "n2_permutation_check.json")
    token_table = "\n".join(
        f"| {row['concept']} | `{row['token_ids']}` | `{row['token_strings']}` | {row['token_count']} |"
        for row in token_rows
    )
    feasibility_table = "\n".join(
        f"| {row['concept']} | {float(row['mean_clip_prompt_alignment']):.4f} | "
        f"{float(row['std_clip_prompt_alignment']):.4f} | "
        f"{float(row['minimum_clip_prompt_alignment']):.4f} |"
        for row in feasibility
    )
    screening_table = "\n".join(
        "| {target} → {anchor} | {original:.4f} | {edited:.4f} | {td:+.4f} | "
        "{own:.4f} | {ad:+.4f} | {other:.4f} | {margin:+.4f} | {minimum:+.4f} | "
        "{positive:.0%} | {top1:.0%} | {status} |".format(
            target=row["target"],
            anchor=row["anchor"],
            original=float(row["original_mean_target_similarity"]),
            edited=float(row["edited_mean_target_similarity"]),
            td=float(row["target_similarity_delta"]),
            own=float(row["edited_mean_own_anchor_similarity"]),
            ad=float(row["own_anchor_similarity_delta"]),
            other=float(row["edited_mean_best_other_similarity"]),
            margin=float(row["edited_mean_margin"]),
            minimum=float(row["edited_minimum_margin"]),
            positive=float(row["edited_positive_margin_fraction"]),
            top1=float(row["edited_own_anchor_top1_rate"]),
            status="passes directional screen"
            if str(row["directional_screening_pass"]).casefold() == "true"
            else "weaker / does not pass",
        )
        for row in screening
    )
    selected_labels = ", ".join(
        f"{row['target']} → {row['anchor']}" for row in selection["pairs"]
    )
    def n2_table(method: str) -> str:
        return "\n".join(
            "| {target} → {anchor} | {target_sim:.4f} | {own:.4f} | {other:.4f} | "
            "{margin:+.4f} | {minimum:+.4f} | {positive:.0%} | {top1:.0%} |".format(
                target=row["target"],
                anchor=row["anchor"],
                target_sim=float(row["mean_target_similarity"]),
                own=float(row["mean_own_anchor_similarity"]),
                other=float(row["mean_best_other_anchor_similarity"]),
                margin=float(row["mean_margin"]),
                minimum=float(row["minimum_margin"]),
                positive=float(row["positive_margin_fraction"]),
                top1=float(row["own_anchor_top1_rate"]),
            )
            for row in n2 if row["method"] == method
        )
    feature_summary = []
    for method in (
        "original",
        "single_subspace",
        "joint_subspace_n2",
        "joint_vector_eq6_n2",
    ):
        rows = [row for row in feature if row["method"] == method]
        feature_summary.append(
            {
                "method": method,
                "mean_top1": np.mean([float(row["own_anchor_top1_rate"]) for row in rows]),
                "mean_margin": np.mean([float(row["mean_margin"]) for row in rows]),
                "minimum_margin": np.min([float(row["minimum_margin"]) for row in rows]),
            }
        )
    feature_table = "\n".join(
        f"| {row['method']} | {row['mean_top1']:.1%} | {row['mean_margin']:+.4f} | {row['minimum_margin']:+.4f} |"
        for row in feature_summary
    )
    p = permutation["summary"]
    objective_diagnostic = permutation["objective_diagnostic"]
    selected_pairs = selected_n2(output)
    report = f"""# Official OCE Subspace Correspondence Diagnostic

This report records observations rather than presuming H1–H4. The official
baseline is the repository's unchanged **subspace objective**. The vector-wise
paired objective is labeled **Eq. 6 ablation** throughout.

## Observed answers at the current stop gate

- All ten configured words are visually recognizable in Original SD across
  the ten fixed seeds. `car` has the largest CLIP spread but no obvious visual
  feasibility failure.
- `cat → dog` is the only single-pair subspace mapping with clear, stable
  own-anchor images. The other four erase their targets but mostly generate
  unrelated content; `guitar → piano` is only the numeric second-best.
- N=2 joint subspace retains positive dog/piano CLIP margins, but the
  `guitar` grid does not show stable pianos. The CLIP candidate-set result
  therefore does not establish visual pairwise correspondence.
- The Eq. 6 ablation has a stronger feature diagonal and visibly produces
  pianos for most guitar seeds, while its guitar target similarity is higher
  than joint subspace (weaker erasure by that measure).
- Swapping the anchor assignment changes the official objective only at about
  float32 numerical precision, confirming that pair identity is not encoded
  in that objective. The resulting rotations/checkpoints are nevertheless
  not all numerically close, consistent with an unstable/non-unique
  Procrustes solution near this objective.
- Non-target preservation is not yet evaluated, so no preservation conclusion
  is made. N=5 should not be started at this gate: four single-pair mappings
  have obvious visual own-anchor failures and the control phase is pending.

## 1. Tokenizer check

Tokenization below uses the exact Stable Diffusion 1.4 CLIP tokenizer with
special BOS/EOS tokens excluded.

| Concept | Token ids | Token strings | Content-token count |
|---|---|---|---:|
{token_table}

Raw files: [CSV](metrics/tokenization.csv), [JSON](metrics/tokenization.json).

## 2. Original SD feasibility

Each concept uses `{evaluation_text(config, "{concept}")}`, seeds
`{config["seeds"]["feasibility"][0]}–{config["seeds"]["feasibility"][-1]}`,
50 steps, guidance 7.5, and 512×512 images. CLIP values are raw cosine
similarities, not probabilities. Visual stability must be read together with
the saved grids.

| Concept | Mean alignment | SD | Minimum |
|---|---:|---:|---:|
{feasibility_table}

Individual grids are under [grids/feasibility](grids/feasibility/).

## 3. Single-pair subspace screening

Every edited image is scored against all five anchors, not only its own
anchor. The directional label requires target similarity to decrease,
own-anchor similarity to increase, and the mechanical low-variance alarm to
remain within its configured bound; it does not automatically remove a pair.

| Mapping | Orig target | Edited target | Target Δ | Own anchor | Anchor Δ | Best other | Mean margin | Min margin | Positive | Own top-1 | Label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{screening_table}

Seed-aligned grids are under [grids/single_pair](grids/single_pair/). Per-image
scores, including all five anchor columns, are in
[single_pair_per_image.csv](metrics/single_pair_per_image.csv).

Manual grid review: {visual_review["summary"]}

"""
    for row in visual_review["pairs"]:
        report += f"- **{row['mapping']}:** {row['observation']}\n"
    report += f"""

### N=2 selection

Selected: **{selected_labels}**.

Selection record and exact screening evidence:
[n2_selection.json](inputs/n2_selection.json). Basis:
`{selection["selection_basis"]}`.

## 4. N = 2 joint subspace

| Mapping | Target sim | Own anchor | Other anchor | Mean margin | Min margin | Positive | Own top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{n2_table("joint_subspace_n2")}

![Joint subspace image similarity](heatmaps/joint_n2/image_similarity_joint_subspace_n2.png)

![Joint subspace image confusion](heatmaps/joint_n2/image_confusion_joint_subspace_n2.png)

## 5. N = 2 joint vector-wise Eq. 6 ablation

This is an ablation and is not described as the official OCE baseline.

| Mapping | Target sim | Own anchor | Other anchor | Mean margin | Min margin | Positive | Own top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{n2_table("joint_vector_eq6_n2")}

![Eq. 6 image similarity](heatmaps/joint_n2/image_similarity_joint_vector_eq6_n2.png)

![Eq. 6 image confusion](heatmaps/joint_n2/image_confusion_joint_vector_eq6_n2.png)

### Feature-level correspondence

Feature matrices were computed before N=2 image generation. Anchor features
use original `W a_j`; target features use original or edited `W c_i` as
appropriate.

| Method | Layer-mean own top-1 | Layer-mean margin | Minimum layer/concept margin |
|---|---:|---:|---:|
{feature_table}

All 16 per-layer heatmaps are under
[heatmaps/joint_n2/feature](heatmaps/joint_n2/feature/); exact cells are in
[n2_feature_cells.csv](metrics/n2_feature_cells.csv).

Manual N=2 grid review: {n2_visual_review["summary"]}

"""
    for row in n2_visual_review["mappings"]:
        report += f"- **{row['mapping']}:** {row['observation']}\n"
    report += f"""

### Seed-aligned N=2 grids

"""
    for pair in selected_pairs:
        report += (
            f"![{pair.target} correspondence grid]"
            f"(grids/joint_n2/{pair.slug}.png)\n\n"
        )
    report += f"""## 6. N = 2 permutation check

Only the two anchor assignments were swapped; the target list and anchor set
were unchanged.

- Checkpoint byte hashes equal: `{str(p["checkpoint_hash_equal"]).lower()}`
- Rotation byte hashes equal: `{str(p["rotation_hash_equal"]).lower()}`
- All 16 edited weights numerically close (`rtol=1e-5`, `atol=1e-6`):
  `{str(p["all_weights_numerically_close"]).lower()}`
- All 16 rotations numerically close: `{str(p["all_rotations_numerically_close"]).lower()}`
- Maximum absolute weight difference: `{p["maximum_weight_abs_difference"]:.8g}`
- Maximum absolute rotation difference: `{p["maximum_rotation_abs_difference"]:.8g}`
- Maximum anchor-projector difference before Procrustes:
  `{objective_diagnostic["maximum_anchor_projector_abs_difference"]:.8g}`
- Maximum objective difference before Procrustes:
  `{objective_diagnostic["maximum_objective_abs_difference"]:.8g}`
- Minimum numerical anchor-feature rank across layers:
  `{objective_diagnostic["minimum_anchor_numeric_rank"]}` of 12 columns

Exact hashes and layer-level differences:
[n2_permutation_check.json](metrics/n2_permutation_check.json).

## 7. Control-set preservation

Not executed in this stage. Preflight verified that `frog`, `horse`, `ship`,
`deer`, and `boat` do not overlap any configured target or anchor. LPIPS
results are therefore pending, not missing due to a failed run.

## 8. Whether to proceed to N = 5

Not executed automatically. The current stop gate is
`n2_permutation_check`. **Recommendation at this gate: do not enter N=5 yet.**
Four of five single-pair subspace mappings erase their targets without
visually stable own-anchor generation, and the required control evaluation
has not yet been run.

## Resolved methods and settings

- Model: `{config["model_id"]}`
- Edited layers: all 16 modules whose name contains `attn2` and ends in `to_v`
- Official subspace parameters: erase `{config["oce"]["erase_scale"]}`,
  global retain `{config["oce"]["preserve_global_scale"]}`, local retain
  `{config["oce"]["preserve_concept_scale"]}`, lambda `{config["oce"]["lambda"]}`
- Object expansion: bare phrase plus image/photo/portrait/picture/painting
  paired forms
- Generation: 50 steps, guidance 7.5, 512×512, bfloat16
- Cg SHA-256: `{config["cg_sha256"]}`; valid-token count `{config["cg_count"]}`
- Full configuration: [resolved_config.json](resolved_config.json)
- Artifact QA: [artifact_validation.json](metrics/artifact_validation.json)
- Archival compatibility check:
  [archival_compatibility.json](metrics/archival_compatibility.json)

## Limitations

- Ten fixed seeds make this a diagnostic, not a population estimate.
- CLIP does not establish object identity, visual quality, hybrids, or
  generation collapse; grids require human inspection.
- A top-1 result can be positive while absolute similarities remain weak, so
  raw similarities and margins are retained.
- No Y→Y_tilde, new loss, clustering, sequential editing, or replacement
  mapping was introduced.

"""
    path = output / "report.md"
    path.write_text(report, encoding="utf-8")
    return path


def validate_artifacts(output: Path) -> dict[str, object]:
    expected_csv_rows = {
        "metrics/tokenization.csv": 10,
        "metrics/feasibility_per_image.csv": 100,
        "metrics/feasibility_summary.csv": 10,
        "metrics/single_pair_per_image.csv": 100,
        "metrics/single_pair_summary.csv": 5,
        "metrics/n2_feature_cells.csv": 256,
        "metrics/n2_feature_summary.csv": 64,
        "metrics/n2_per_image.csv": 80,
        "metrics/n2_image_similarity_cells.csv": 160,
        "metrics/n2_image_confusion.csv": 16,
        "metrics/n2_per_concept.csv": 8,
        "metrics/n2_aggregate.csv": 4,
        "metrics/n2_permutation_layer_comparison.csv": 16,
        "metrics/n2_permutation_objective_comparison.csv": 16,
        "metrics/weight_audit.csv": 128,
    }
    csv_checks = []
    for relative, expected in expected_csv_rows.items():
        path = output / relative
        actual = len(read_csv(path)) if path.exists() else None
        csv_checks.append(
            {
                "path": relative,
                "expected_rows": expected,
                "actual_rows": actual,
                "pass": actual == expected,
            }
        )
    count_checks = [
        {
            "artifact": "feasibility_images",
            "expected": 100,
            "actual": len(list((output / "images" / "feasibility").rglob("*.png"))),
        },
        {
            "artifact": "single_pair_images",
            "expected": 50,
            "actual": len(list((output / "images" / "single_pair").rglob("*.png"))),
        },
        {
            "artifact": "joint_n2_images",
            "expected": 40,
            "actual": len(list((output / "images" / "joint_n2").rglob("*.png"))),
        },
        {
            "artifact": "checkpoints",
            "expected": 8,
            "actual": len(list((output / "checkpoints").glob("*.safetensors"))),
        },
        {
            "artifact": "saved_rotations",
            "expected": 8,
            "actual": len(list((output / "transformations").glob("*.safetensors"))),
        },
        {
            "artifact": "grids",
            "expected": 17,
            "actual": len(list((output / "grids").rglob("*.png"))),
        },
        {
            "artifact": "heatmaps",
            "expected": 72,
            "actual": len(list((output / "heatmaps").rglob("*.png"))),
        },
    ]
    for row in count_checks:
        row["pass"] = row["actual"] == row["expected"]
    required_headings = [
        "## 1. Tokenizer check",
        "## 2. Original SD feasibility",
        "## 3. Single-pair subspace screening",
        "## 4. N = 2 joint subspace",
        "## 5. N = 2 joint vector-wise Eq. 6 ablation",
        "## 6. N = 2 permutation check",
        "## 7. Control-set preservation",
        "## 8. Whether to proceed to N = 5",
    ]
    report_path = output / "report.md"
    report = report_path.read_text(encoding="utf-8")
    heading_checks = [
        {"heading": heading, "pass": heading in report}
        for heading in required_headings
    ]
    link_targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", report)
    link_checks = [
        {
            "target": target,
            "pass": (
                target == "metrics/artifact_validation.json"
                or (output / target).exists()
            ),
        }
        for target in link_targets
        if not target.startswith(("http://", "https://"))
    ]
    state = read_json(output / "run_state.json")
    state_pass = (
        state["n2_permutation_check"] == "complete"
        and state["control_set"] == "not_run_current_stage"
        and state["n5"] == "not_run_current_stage"
    )
    passed = (
        all(row["pass"] for row in csv_checks)
        and all(row["pass"] for row in count_checks)
        and all(row["pass"] for row in heading_checks)
        and all(row["pass"] for row in link_checks)
        and state_pass
    )
    payload = {
        "status": "pass" if passed else "fail",
        "csv_checks": csv_checks,
        "count_checks": count_checks,
        "report_heading_checks": heading_checks,
        "report_link_checks": link_checks,
        "run_state_check": {"pass": state_pass, "state": state},
        "manual_visual_review_files": [
            "metrics/single_pair_visual_review.json",
            "metrics/n2_visual_review.json",
        ],
        "scope_note": "This validation does not claim control or N=5 completion.",
    }
    write_json(output / "metrics" / "artifact_validation.json", payload)
    if not passed:
        raise RuntimeError("Artifact validation failed; inspect artifact_validation.json")
    return payload


def run_through_permutation(
    config_path: Path, output: Path, n2_pair_slugs: Sequence[str] | None
) -> None:
    config = preflight(config_path, output)
    tokenizer_audit(config, output)
    generate_feasibility(config, output)
    evaluate_feasibility(config, output)
    generate_single(config, output)
    evaluate_single(config, output)
    select_n2(config, output, n2_pair_slugs)
    prepare_n2(config, output)
    evaluate_n2_features(config, output)
    generate_n2(config, output)
    evaluate_n2_images(config, output)
    permutation_check(output)
    permutation_objective_check(config, output)
    print(f"Report: {build_report(config, output)}", flush=True)
    validate_artifacts(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official OCE subspace correspondence diagnostic"
    )
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "tokenizer",
            "feasibility",
            "single",
            "select-n2",
            "prepare-n2",
            "features-n2",
            "images-n2",
            "permutation",
            "report",
            "validate",
            "through-permutation",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n2-pairs", nargs=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "through-permutation":
        run_through_permutation(args.config, args.output, args.n2_pairs)
        return
    config = (
        preflight(args.config, args.output)
        if args.command == "preflight"
        else resolved(args.config, args.output)
    )
    if args.command == "preflight":
        return
    if args.command == "tokenizer":
        tokenizer_audit(config, args.output)
    elif args.command == "feasibility":
        generate_feasibility(config, args.output)
        evaluate_feasibility(config, args.output)
    elif args.command == "single":
        generate_single(config, args.output)
        evaluate_single(config, args.output)
    elif args.command == "select-n2":
        select_n2(config, args.output, args.n2_pairs)
    elif args.command == "prepare-n2":
        prepare_n2(config, args.output)
    elif args.command == "features-n2":
        evaluate_n2_features(config, args.output)
    elif args.command == "images-n2":
        generate_n2(config, args.output)
        evaluate_n2_images(config, args.output)
    elif args.command == "permutation":
        permutation_check(args.output)
        permutation_objective_check(config, args.output)
    elif args.command == "report":
        print(build_report(config, args.output))
    elif args.command == "validate":
        print(json.dumps(validate_artifacts(args.output), indent=2))


if __name__ == "__main__":
    main()
