from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .core import (
    PairSpec,
    apply_weight_state,
    clone_projection_state,
    collect_projection_modules,
    compute_correspondence_metrics,
    edit_projection_weights,
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


HERE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT = HERE / "outputs" / "initial_screening"


def _dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def _pairs(config: Mapping[str, object]) -> list[PairSpec]:
    return [PairSpec(**row) for row in config["pairs"]]  # type: ignore[arg-type]


def _resolve_config(config_path: Path, output_dir: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pairs = _pairs(config)
    set_audit = validate_experiment_sets(pairs, config["control_concepts"])
    cg_path = (config_path.parent / str(config["cg_path"])).resolve()
    if not cg_path.is_file():
        raise FileNotFoundError(f"Missing Cg file: {cg_path}")
    cg_payload = torch.load(cg_path, map_location="cpu")
    if "C" not in cg_payload or tuple(cg_payload["C"].shape) != (768, 768):
        raise ValueError(f"Unexpected Cg payload at {cg_path}")
    screening_seeds = list(config["seeds"]["screening"])
    smoke_seeds = list(config["seeds"]["smoke"])
    if len(smoke_seeds) != 20 or len(set(smoke_seeds)) != 20:
        raise ValueError("Smoke test must use exactly 20 unique fixed seeds")
    if not 10 <= len(screening_seeds) <= 20:
        raise ValueError("Screening must use 10 to 20 fixed seeds")
    resolved = dict(config)
    resolved["config_path"] = str(config_path.resolve())
    resolved["output_dir"] = str(output_dir.resolve())
    resolved["cg_path"] = str(cg_path)
    resolved["cg_sha256"] = sha256(cg_path)
    resolved["cg_count"] = cg_payload.get("count")
    resolved["set_validation"] = set_audit
    resolved["resolved_at"] = datetime.now(timezone.utc).isoformat()
    resolved["software"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    return resolved


def preflight(config_path: Path, output_dir: Path) -> dict[str, object]:
    resolved = _resolve_config(config_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", resolved)
    pair_rows = [
        {
            "pair_index": index,
            "target": pair.target,
            "anchor": pair.anchor,
            "prompt": pair.prompt,
        }
        for index, pair in enumerate(_pairs(resolved), start=1)
    ]
    write_csv(output_dir / "inputs" / "target_anchor_pairs.csv", pair_rows)
    seed_rows = []
    for phase, values in resolved["seeds"].items():
        for index, seed in enumerate(values):
            seed_rows.append({"phase": phase, "index": index, "seed": seed})
    write_csv(output_dir / "inputs" / "seeds.csv", seed_rows)
    prompt_rows = [
        {
            "kind": "target",
            "concept": pair.target,
            "prompt": pair.prompt,
            "evaluation_text": resolved["evaluation"]["text_template"].format(
                concept=pair.target
            ),
        }
        for pair in _pairs(resolved)
    ]
    prompt_rows.extend(
        {
            "kind": "control",
            "concept": concept,
            "prompt": resolved["evaluation"]["text_template"].format(concept=concept),
            "evaluation_text": resolved["evaluation"]["text_template"].format(
                concept=concept
            ),
        }
        for concept in resolved["control_concepts"]
    )
    write_csv(output_dir / "inputs" / "prompts.csv", prompt_rows)
    write_json(
        output_dir / "run_state.json",
        {
            "preflight": "complete",
            "weights": "pending",
            "smoke": "pending",
            "screening": "pending",
            "joint_n2": "pending",
            "joint_n5": "gated",
            "controls": "pending",
        },
    )
    return resolved


@torch.inference_mode()
def _encode_last_content_tokens(
    pipe: object, prompts: Sequence[str], device: torch.device
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for prompt in dict.fromkeys(prompts):
        tokenized = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden = pipe.text_encoder(
            tokenized.input_ids.to(device)
        ).last_hidden_state[0]
        last_index = int(tokenized.attention_mask.sum().item()) - 2
        result[prompt] = hidden[last_index].float()
    return result


def _method_name(objective: str, pairs: Sequence[PairSpec], joint: bool) -> str:
    if joint:
        return f"joint_{objective}_n{len(pairs)}"
    if len(pairs) != 1:
        raise ValueError("Single method names require exactly one pair")
    return f"single_{objective}_{pairs[0].slug}"


def prepare_initial_weights(resolved: Mapping[str, object], output_dir: Path) -> None:
    from diffusers import DiffusionPipeline

    device = torch.device(str(resolved["device"]))
    dtype = _dtype(str(resolved["edit_dtype"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(resolved["model_id"]),
        torch_dtype=dtype,
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)
    modules = collect_projection_modules(pipe.unet)
    if len(modules) != 16:
        raise RuntimeError(f"Expected 16 attn2.to_v modules, found {len(modules)}")
    base_pairs = _pairs(resolved)
    expanded_pairs = (
        expand_object_pairs(base_pairs)
        if resolved["oce"]["expand_prompts"]
        else base_pairs
    )
    all_text = []
    for pair in expanded_pairs:
        all_text.extend([pair.target, pair.anchor])
    all_text.extend(pair.anchor for pair in base_pairs)
    embeddings = _encode_last_content_tokens(pipe, all_text, device)
    cg_payload = torch.load(str(resolved["cg_path"]), map_location=device)
    cg = cg_payload["C"].to(device=device, dtype=torch.float32)
    oce = resolved["oce"]
    methods: list[tuple[str, str, list[PairSpec], list[str]]] = []
    for pair in base_pairs:
        pair_expanded = (
            expand_object_pairs([pair]) if oce["expand_prompts"] else [pair]
        )
        methods.append(
            (
                _method_name("vector", [pair], False),
                "vector",
                pair_expanded,
                [pair.anchor] if oce["preserve_anchors"] else [],
            )
        )
    cat_pair = [base_pairs[0]]
    cat_expanded = (
        expand_object_pairs(cat_pair) if oce["expand_prompts"] else cat_pair
    )
    methods.append(
        (
            _method_name("subspace", cat_pair, False),
            "subspace",
            cat_expanded,
            [cat_pair[0].anchor] if oce["preserve_anchors"] else [],
        )
    )
    audit_rows: list[dict[str, object]] = []
    for method, objective, method_pairs, preserve in methods:
        checkpoint = output_dir / "weights" / f"{method}.safetensors"
        if checkpoint.exists():
            print(f"[weights] reuse {checkpoint}", flush=True)
            continue
        print(f"[weights] {method}", flush=True)
        edited, layer_audit = edit_projection_weights(
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
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_file(edited, str(checkpoint))
        for row in layer_audit:
            row.update(
                {
                    "method": method,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
            audit_rows.append(row)
    existing_audit = output_dir / "metrics" / "weight_audit.csv"
    if audit_rows:
        if existing_audit.exists():
            prior = read_csv(existing_audit)
            completed = {row["method"] for row in audit_rows}
            audit_rows = [
                row for row in prior if row["method"] not in completed
            ] + audit_rows
        write_csv(existing_audit, audit_rows)
    state = read_json(output_dir / "run_state.json")
    state["weights"] = "initial_complete"
    write_json(output_dir / "run_state.json", state)
    del pipe, cg, cg_payload
    gc.collect()
    torch.cuda.empty_cache()


def _image_path(
    output_dir: Path, phase: str, pair: PairSpec, method: str, seed: int
) -> Path:
    return (
        output_dir
        / "images"
        / phase
        / pair.slug
        / method
        / f"seed_{seed}.png"
    )


@torch.inference_mode()
def generate_initial_images(
    resolved: Mapping[str, object], output_dir: Path
) -> None:
    from diffusers import DiffusionPipeline

    device = torch.device(str(resolved["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(resolved["model_id"]),
        torch_dtype=_dtype(str(resolved["generation_dtype"])),
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    original_state = clone_projection_state(pipe.unet)
    pairs = _pairs(resolved)
    generation = resolved["generation"]
    methods_for_pair: dict[str, list[str]] = {}
    for index, pair in enumerate(pairs):
        methods = ["original", _method_name("vector", [pair], False)]
        if index == 0:
            methods.append(_method_name("subspace", [pair], False))
        methods_for_pair[pair.slug] = methods
    for pair in pairs:
        phase = "smoke" if pair == pairs[0] else "screening"
        # Cat images are shared between smoke and screening through the same seeds.
        if pair == pairs[0]:
            phase = "smoke"
        seeds = list(resolved["seeds"]["screening"])
        for method in methods_for_pair[pair.slug]:
            if method == "original":
                apply_weight_state(pipe.unet, original_state)
            else:
                checkpoint = output_dir / "weights" / f"{method}.safetensors"
                apply_weight_state(pipe.unet, load_file(str(checkpoint)))
            for sample_index, seed in enumerate(seeds, start=1):
                destination = _image_path(output_dir, phase, pair, method, seed)
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                generator = torch.Generator(device=device).manual_seed(int(seed))
                image = pipe(
                    prompt=pair.prompt,
                    num_inference_steps=int(generation["num_inference_steps"]),
                    guidance_scale=float(generation["guidance_scale"]),
                    height=int(generation["height"]),
                    width=int(generation["width"]),
                    generator=generator,
                ).images[0]
                image.save(destination)
                print(
                    f"[generate] {pair.slug} {method} "
                    f"{sample_index}/{len(seeds)} seed={seed}",
                    flush=True,
                )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


class ClipEvaluator:
    def __init__(self, model_id: str, device: str):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(
            model_id, local_files_only=True
        ).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            model_id, local_files_only=True
        )

    @torch.inference_mode()
    def similarities(
        self, paths: Sequence[Path], texts: Sequence[str], batch_size: int = 8
    ) -> np.ndarray:
        text_inputs = self.processor(
            text=list(texts), return_tensors="pt", padding=True
        )
        text_inputs = {
            key: value.to(self.device) for key, value in text_inputs.items()
        }
        text_features = self.model.get_text_features(**text_inputs)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
        result = []
        for start in range(0, len(paths), batch_size):
            images = [
                Image.open(path).convert("RGB")
                for path in paths[start : start + batch_size]
            ]
            image_inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(self.device)
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            image_features = torch.nn.functional.normalize(
                image_features.float(), dim=-1
            )
            result.append((image_features @ text_features.T).cpu().numpy())
        return np.concatenate(result, axis=0)


def _evaluation_text(resolved: Mapping[str, object], concept: str) -> str:
    return str(resolved["evaluation"]["text_template"]).format(concept=concept)


def evaluate_initial(
    resolved: Mapping[str, object], output_dir: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pairs = _pairs(resolved)
    seeds = list(resolved["seeds"]["screening"])
    evaluator = ClipEvaluator(
        str(resolved["clip_model_id"]), str(resolved["device"])
    )
    per_image: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    rule = resolved["evaluation"]["screening_rule"]
    for pair_index, pair in enumerate(pairs):
        phase = "smoke" if pair_index == 0 else "screening"
        methods = ["original", _method_name("vector", [pair], False)]
        if pair_index == 0:
            methods.append(_method_name("subspace", [pair], False))
        texts = [
            _evaluation_text(resolved, pair.target),
            _evaluation_text(resolved, pair.anchor),
        ]
        method_values: dict[str, np.ndarray] = {}
        method_paths: dict[str, list[Path]] = {}
        for method in methods:
            paths = [
                _image_path(output_dir, phase, pair, method, seed) for seed in seeds
            ]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing generated images: {missing[:3]}")
            values = evaluator.similarities(paths, texts)
            method_values[method] = values
            method_paths[method] = paths
            predictions = values.argmax(axis=1)
            for index, (seed, path) in enumerate(zip(seeds, paths)):
                per_image.append(
                    {
                        "phase": phase,
                        "pair_index": pair_index + 1,
                        "pair": pair.slug,
                        "target": pair.target,
                        "anchor": pair.anchor,
                        "prompt": pair.prompt,
                        "method": method,
                        "seed": seed,
                        "target_similarity": float(values[index, 0]),
                        "anchor_similarity": float(values[index, 1]),
                        "anchor_minus_target": float(
                            values[index, 1] - values[index, 0]
                        ),
                        "predicted_label": (
                            pair.target if predictions[index] == 0 else pair.anchor
                        ),
                        "own_anchor_top1": bool(predictions[index] == 1),
                        "rgb_std": image_rgb_std(path),
                        "image_path": str(path.resolve()),
                    }
                )
        original = method_values["original"]
        vector_name = _method_name("vector", [pair], False)
        vector = method_values[vector_name]
        vector_rows = [
            row
            for row in per_image
            if row["pair"] == pair.slug and row["method"] == vector_name
        ]
        target_delta = float(vector[:, 0].mean() - original[:, 0].mean())
        anchor_delta = float(vector[:, 1].mean() - original[:, 1].mean())
        low_variance_fraction = float(
            np.mean(
                [
                    float(row["rgb_std"])
                    < float(rule["low_variance_rgb_std_threshold"])
                    for row in vector_rows
                ]
            )
        )
        passed = (
            target_delta < float(rule["target_similarity_delta_must_be_below"])
            and anchor_delta > float(rule["anchor_similarity_delta_must_be_above"])
            and low_variance_fraction <= float(rule["max_low_variance_fraction"])
        )
        summaries.append(
            {
                "pair_index": pair_index + 1,
                "pair": pair.slug,
                "target": pair.target,
                "anchor": pair.anchor,
                "n_seeds": len(seeds),
                "original_mean_target_similarity": float(original[:, 0].mean()),
                "original_mean_anchor_similarity": float(original[:, 1].mean()),
                "edited_mean_target_similarity": float(vector[:, 0].mean()),
                "edited_mean_anchor_similarity": float(vector[:, 1].mean()),
                "target_similarity_delta": target_delta,
                "anchor_similarity_delta": anchor_delta,
                "own_anchor_top1_rate": float(np.mean(vector.argmax(axis=1) == 1)),
                "low_variance_fraction": low_variance_fraction,
                "screening_rule": str(rule["id"]),
                "screening_pass": passed,
                "screening_pass_is_automatic_elimination": False,
            }
        )
        grid_methods = {
            "Original": method_paths["original"],
            "Single Vector": method_paths[vector_name],
        }
        if pair_index == 0:
            subspace_name = _method_name("subspace", [pair], False)
            grid_methods["Single Subspace"] = method_paths[subspace_name]
        make_seed_grid(
            output_dir / "grids" / f"{phase}_{pair.slug}.png",
            grid_methods,
            seeds,
            f"{pair.target} -> {pair.anchor} | {pair.prompt}",
        )
    write_csv(output_dir / "metrics" / "initial_per_image_metrics.csv", per_image)
    write_csv(output_dir / "metrics" / "single_pair_screening_summary.csv", summaries)
    write_json(
        output_dir / "metrics" / "single_pair_screening_summary.json",
        {
            "rule": rule,
            "pairs": summaries,
            "passed_pairs": [
                row["pair"] for row in summaries if row["screening_pass"]
            ],
            "note": (
                "No pair is automatically removed. screening_pass is a transparent "
                "directional label for user review."
            ),
        },
    )
    state = read_json(output_dir / "run_state.json")
    state["smoke"] = "complete"
    state["screening"] = "complete"
    state["joint_n5"] = "awaiting_user_pair_selection"
    write_json(output_dir / "run_state.json", state)
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    return per_image, summaries


def build_initial_report(
    resolved: Mapping[str, object],
    output_dir: Path,
    per_image: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
) -> Path:
    def truth(value: object) -> bool:
        if isinstance(value, str):
            return value.casefold() in {"true", "1", "yes"}
        return bool(value)

    cat = _pairs(resolved)[0]
    vector_name = _method_name("vector", [cat], False)
    subspace_name = _method_name("subspace", [cat], False)
    cat_rows = [
        row
        for row in per_image
        if row["pair"] == cat.slug
    ]
    by_method = {}
    for method in ["original", vector_name, subspace_name]:
        rows = [row for row in cat_rows if row["method"] == method]
        by_method[method] = {
            "cat": float(np.mean([float(row["target_similarity"]) for row in rows])),
            "dog": float(np.mean([float(row["anchor_similarity"]) for row in rows])),
            "dog_top1": float(
                np.mean([truth(row["own_anchor_top1"]) for row in rows])
            ),
        }
    screening_table = "\n".join(
        "| {target} → {anchor} | {ot:.4f} | {et:.4f} | {ea:.4f} | "
        "{td:+.4f} | {ad:+.4f} | {top:.1%} | {status} |".format(
            target=row["target"],
            anchor=row["anchor"],
            ot=float(row["original_mean_target_similarity"]),
            et=float(row["edited_mean_target_similarity"]),
            ea=float(row["edited_mean_anchor_similarity"]),
            td=float(row["target_similarity_delta"]),
            ad=float(row["anchor_similarity_delta"]),
            top=float(row["own_anchor_top1_rate"]),
            status="pass" if truth(row["screening_pass"]) else "does not pass",
        )
        for row in summaries
    )
    seed_table_rows = []
    seeds = list(resolved["seeds"]["smoke"])
    for seed in seeds:
        cells = []
        for method in ["original", vector_name, subspace_name]:
            row = next(
                row
                for row in cat_rows
                if int(row["seed"]) == int(seed) and row["method"] == method
            )
            cells.append(
                "{:.4f}/{:.4f}/{}".format(
                    float(row["target_similarity"]),
                    float(row["anchor_similarity"]),
                    row["predicted_label"],
                )
            )
        seed_table_rows.append(
            f"| {seed} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    passed = [row["pair"] for row in summaries if truth(row["screening_pass"])]
    failed = [row["pair"] for row in summaries if not truth(row["screening_pass"])]
    visual_review_path = output_dir / "metrics" / "manual_visual_review.json"
    visual_review = read_json(visual_review_path) if visual_review_path.exists() else None
    visual_review_text = ""
    if visual_review:
        visual_review_text = "\n### Manual grid review\n\n"
        for row in visual_review["pairs"]:
            pair_label = str(row["pair"]).replace("_to_", " → ").replace("_", " ")
            visual_review_text += (
                f"- **{pair_label}:** {row['observation']}\n"
            )
        visual_review_text += (
            "\nStructured visual-review notes: "
            "[manual_visual_review.json](metrics/manual_visual_review.json)\n"
        )
    n5_gate_path = output_dir / "metrics" / "joint_n5_gate.json"
    n5_gate = read_json(n5_gate_path) if n5_gate_path.exists() else None
    if n5_gate:
        n5_status = (
            f"The recorded directional rule yields "
            f"{n5_gate['eligible_count']}/{n5_gate['required_count']} eligible "
            f"candidates (`{', '.join(n5_gate['eligible_pairs'])}`). "
            "This is fewer than five, so no N=5 checkpoint or image was "
            "created and no replacement mapping was added."
        )
    else:
        n5_status = (
            "Candidate mappings are not automatically removed; the final "
            "mapping set must be selected after reviewing the screening labels "
            "and grids. No N=5 generation was run."
        )
    report = f"""# OCE Target–Anchor Correspondence Diagnostic

## Technical summary

- Phase A and Phase B use Stable Diffusion 1.4, the same 20 seeds ({seeds[0]}–{seeds[-1]}), 50 denoising steps, guidance 7.5, 512×512 output, and all 16 `attn2.to_v` layers.
- The vector-wise checkpoint uses the paper's paired objective; the subspace checkpoint reproduces the current upstream `oce.py` objective. No `Y_tilde`, new loss, clustering, sequential editing, or alternate erasure method is present.
- The automatic screening label uses only directional evidence: edited target similarity must decrease, edited own-anchor similarity must increase, and low-variance image incidence must stay at or below 20%. It does not remove mappings automatically.
- Directional screening labels: passed `{", ".join(passed) if passed else "none"}`; did not pass `{", ".join(failed) if failed else "none"}`. These labels require visual/user review before an N=5 set is selected.

## 1. Single-pair screening

CLIP similarities are cosine similarities from `{resolved["clip_model_id"]}` using the text template `a photo of a {{concept}}`. “Own-anchor top-1” is evaluated against the pair's target and anchor labels. A pass is descriptive, not a positive conclusion or an automatic inclusion decision.

| Mapping | Original target | Edited target | Edited own anchor | Target Δ | Anchor Δ | Own-anchor top-1 | Directional label |
|---|---:|---:|---:|---:|---:|---:|---|
{screening_table}

Per-image results: [initial_per_image_metrics.csv](metrics/initial_per_image_metrics.csv)  
Machine-readable summary: [single_pair_screening_summary.json](metrics/single_pair_screening_summary.json)
{visual_review_text}

### Cat → dog smoke test

| Method | Mean cat similarity | Mean dog similarity | Dog top-1 rate |
|---|---:|---:|---:|
| Original SD | {by_method["original"]["cat"]:.4f} | {by_method["original"]["dog"]:.4f} | {by_method["original"]["dog_top1"]:.1%} |
| Single vector-wise | {by_method[vector_name]["cat"]:.4f} | {by_method[vector_name]["dog"]:.4f} | {by_method[vector_name]["dog_top1"]:.1%} |
| Single subspace | {by_method[subspace_name]["cat"]:.4f} | {by_method[subspace_name]["dog"]:.4f} | {by_method[subspace_name]["dog_top1"]:.1%} |

![Cat to dog smoke grid](grids/smoke_cat_to_dog.png)

Per-seed cells are `cat similarity / dog similarity / top-1 label`.

| Seed | Original | Single vector-wise | Single subspace |
|---:|---|---|---|
{chr(10).join(seed_table_rows)}

### Screening image grids

"""
    for pair in _pairs(resolved)[1:]:
        report += (
            f"#### {pair.target} → {pair.anchor}\n\n"
            f"![{pair.target} to {pair.anchor} screening grid]"
            f"(grids/screening_{pair.slug}.png)\n\n"
        )
    report += f"""## 2. N = 2 joint vector-wise

Pending. This phase has not been executed in the initial smoke/screening run.

## 3. N = 2 joint subspace

Pending. This phase has not been executed in the initial smoke/screening run.

## 4. N = 5 joint vector-wise

Gated. {n5_status}

## 5. N = 5 joint subspace

Gated by the same candidate-count check as the vector-wise N=5 condition. See [joint_n5_gate.json](metrics/joint_n5_gate.json). No N=5 generation was run.

## 6. Control-set preservation

Pending until the joint checkpoints are evaluated. Preflight verified that `horse`, `ship`, `truck`, `frog`, and `deer` do not overlap the configured target or anchor sets.

## Scope, definitions, and resolved settings

- Target similarity: cosine similarity between a generated image and its target evaluation text.
- Own-anchor similarity: cosine similarity between a generated image and its paired anchor evaluation text.
- Screening candidate set: exactly the target and own anchor for that single-pair mapping.
- Cg: `{resolved["cg_path"]}`, SHA-256 `{resolved["cg_sha256"]}`, `{resolved["cg_count"]}` valid tokens.
- OCE scales: erase `{resolved["oce"]["erase_scale"]}`, global retain `{resolved["oce"]["preserve_global_scale"]}`, local retain `{resolved["oce"]["preserve_concept_scale"]}`, lambda `{resolved["oce"]["lambda"]}`.
- Object prompt expansion: `{str(resolved["oce"]["expand_prompts"]).lower()}`; each bare pair plus `image/photo/portrait/picture/painting of` paired forms.
- Full inputs and parameters: [resolved_config.json](resolved_config.json), [target_anchor_pairs.csv](inputs/target_anchor_pairs.csv), [prompts.csv](inputs/prompts.csv), [seeds.csv](inputs/seeds.csv).
- Artifact QA: [artifact_validation.json](metrics/artifact_validation.json).

## Methodology and validation notes

The image comparisons are paired by prompt and seed. Raw cosine similarities—not two-label softmax probabilities—are stored. Low RGB variance is only a mechanical collapse alarm; semantic collapse and cat–dog hybridization require inspection of the saved grids. The current sample has no confidence interval and is intended as a diagnostic screen, not a population estimate.

## Limitations, uncertainty, and robustness checks

- A two-label top-1 result can improve even when both absolute similarities are poor; raw similarities are therefore reported beside top-1 rates.
- Single-pair screening does not measure cross-anchor confusion because there is only one anchor. N=2/N=5 matrices are required for correspondence claims.
- CLIP similarity does not establish object identity, visual quality, or absence of hybrids.
- The automatic directional label uses zero-crossing deltas and a low-variance alarm. It is intentionally not an effect-size threshold or an automatic exclusion rule.

## Recommended next steps

1. Review all five screening grids and retain the numeric screening decisions as annotations rather than automatic exclusions.
2. Run N=2 joint vector-wise and joint subspace with `cat → dog` and `airplane → sky`.
3. Select the N=5 mapping list explicitly; do not add replacement mappings if fewer than five candidates are accepted.
4. Run the control-set comparison only after the joint checkpoints exist.

## Further questions

- Do visually acceptable images agree with the directional CLIP label for every candidate?
- Does N=2 reduce own-anchor correspondence relative to the corresponding single-pair models?
- Does feature-level diagonal structure predict the image-level confusion matrix?
"""
    destination = output_dir / "report.md"
    destination.write_text(report, encoding="utf-8")
    return destination


def run_initial(config_path: Path, output_dir: Path) -> None:
    resolved = preflight(config_path, output_dir)
    prepare_initial_weights(resolved, output_dir)
    generate_initial_images(resolved, output_dir)
    per_image, summaries = evaluate_initial(resolved, output_dir)
    report = build_initial_report(resolved, output_dir, per_image, summaries)
    print(f"Initial report: {report}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCE target-anchor correspondence diagnostic"
    )
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "prepare-initial",
            "generate-initial",
            "evaluate-initial",
            "initial",
            "check-joint5",
            "prepare-joint2",
            "prepare-joint5",
            "joint2",
            "joint5",
            "controls",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        preflight(args.config, args.output)
        return
    if args.command == "initial":
        run_initial(args.config, args.output)
        return
    resolved_path = args.output / "resolved_config.json"
    resolved = (
        read_json(resolved_path)
        if resolved_path.exists()
        else preflight(args.config, args.output)
    )
    if args.command in {
        "check-joint5",
        "prepare-joint2",
        "prepare-joint5",
        "joint2",
        "joint5",
        "controls",
    }:
        from .joint import (
            check_joint5_gate,
            prepare_joint_weights,
            run_controls,
            run_joint,
        )

        if args.command == "check-joint5":
            print(json.dumps(check_joint5_gate(resolved, args.output), indent=2))
        elif args.command in {"prepare-joint2", "prepare-joint5"}:
            prepare_joint_weights(
                resolved, args.output, 2 if args.command.endswith("2") else 5
            )
        elif args.command in {"joint2", "joint5"}:
            run_joint(
                resolved, args.output, 2 if args.command.endswith("2") else 5
            )
        else:
            run_controls(resolved, args.output)
        return
    if args.command == "prepare-initial":
        prepare_initial_weights(resolved, args.output)
    elif args.command == "generate-initial":
        generate_initial_images(resolved, args.output)
    elif args.command == "evaluate-initial":
        per_image, summaries = evaluate_initial(resolved, args.output)
        build_initial_report(resolved, args.output, per_image, summaries)


if __name__ == "__main__":
    main()
