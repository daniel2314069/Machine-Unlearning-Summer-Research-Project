from __future__ import annotations

import gc
from pathlib import Path
from typing import Mapping, Sequence

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
)
from .io_utils import (
    make_seed_grid,
    plot_heatmap,
    read_csv,
    read_json,
    sha256,
    write_csv,
    write_json,
)
from .runner import (
    ClipEvaluator,
    _dtype,
    _encode_last_content_tokens,
    _evaluation_text,
    _image_path,
    _method_name,
    _pairs,
)


def _joint_checkpoint(output_dir: Path, objective: str, n: int) -> Path:
    return output_dir / "weights" / f"joint_{objective}_n{n}.safetensors"


def check_joint5_gate(
    resolved: Mapping[str, object], output_dir: Path
) -> dict[str, object]:
    summary_path = output_dir / "metrics" / "single_pair_screening_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run single-pair screening before checking N=5")
    summary = read_json(summary_path)
    passed = list(summary["passed_pairs"])
    candidates = [pair.slug for pair in _pairs(resolved)]
    eligible = [slug for slug in candidates if slug in passed]
    result = {
        "candidate_pairs": candidates,
        "screening_rule": summary["rule"],
        "eligible_pairs": eligible,
        "eligible_count": len(eligible),
        "required_count": 5,
        "can_run_n5": len(eligible) == 5,
        "new_pairs_added": [],
        "action": (
            "ready_for_explicit_user_confirmation"
            if len(eligible) == 5
            else "report_insufficient_candidates_and_do_not_run"
        ),
    }
    write_json(output_dir / "metrics" / "joint_n5_gate.json", result)
    state = read_json(output_dir / "run_state.json")
    state["joint_n5"] = (
        "awaiting_user_confirmation"
        if result["can_run_n5"]
        else "blocked_insufficient_screened_candidates"
    )
    write_json(output_dir / "run_state.json", state)
    return result


def prepare_joint_weights(
    resolved: Mapping[str, object], output_dir: Path, n: int
) -> bool:
    from diffusers import DiffusionPipeline

    if n not in {2, 5}:
        raise ValueError("Only N=2 and N=5 are defined")
    if n == 5 and not check_joint5_gate(resolved, output_dir)["can_run_n5"]:
        return False
    pairs = _pairs(resolved)[:n]
    expanded = (
        expand_object_pairs(pairs)
        if resolved["oce"]["expand_prompts"]
        else pairs
    )
    device = torch.device(str(resolved["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(resolved["model_id"]),
        torch_dtype=_dtype(str(resolved["edit_dtype"])),
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)
    if len(collect_projection_modules(pipe.unet)) != 16:
        raise RuntimeError("Joint edit did not resolve the expected 16 layers")
    texts = []
    for pair in expanded:
        texts.extend([pair.target, pair.anchor])
    texts.extend(pair.anchor for pair in pairs)
    embeddings = _encode_last_content_tokens(pipe, texts, device)
    cg = torch.load(str(resolved["cg_path"]), map_location=device)["C"].float()
    oce = resolved["oce"]
    new_audit: list[dict[str, object]] = []
    for objective in ["vector", "subspace"]:
        destination = _joint_checkpoint(output_dir, objective, n)
        if destination.exists():
            continue
        edited, audit = edit_projection_weights(
            unet=pipe.unet,
            embeddings=embeddings,
            pairs=expanded,
            preserve_concepts=(
                [pair.anchor for pair in pairs]
                if oce["preserve_anchors"]
                else []
            ),
            global_second_moment=cg,
            objective=objective,
            erase_scale=float(oce["erase_scale"]),
            preserve_global_scale=float(oce["preserve_global_scale"]),
            preserve_concept_scale=float(oce["preserve_concept_scale"]),
            lamb=float(oce["lambda"]),
            reflection_correction=str(oce["reflection_correction"]),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_file(edited, str(destination))
        for row in audit:
            row.update(
                {
                    "method": f"joint_{objective}_n{n}",
                    "checkpoint": str(destination.resolve()),
                    "checkpoint_sha256": sha256(destination),
                }
            )
            new_audit.append(row)
    if new_audit:
        audit_path = output_dir / "metrics" / "weight_audit.csv"
        existing = read_csv(audit_path) if audit_path.exists() else []
        new_methods = {str(row["method"]) for row in new_audit}
        existing = [row for row in existing if row["method"] not in new_methods]
        write_csv(audit_path, existing + new_audit)
    del pipe, cg
    gc.collect()
    torch.cuda.empty_cache()
    return True


def _joint_image_path(
    output_dir: Path, n: int, pair: PairSpec, method: str, seed: int
) -> Path:
    return (
        output_dir
        / "images"
        / f"joint_n{n}"
        / pair.slug
        / method
        / f"seed_{seed}.png"
    )


def _initial_reference_paths(
    resolved: Mapping[str, object],
    output_dir: Path,
    pair: PairSpec,
    method: str,
) -> list[Path]:
    pair_index = _pairs(resolved).index(pair)
    phase = "smoke" if pair_index == 0 else "screening"
    seeds = list(resolved["seeds"]["joint"])
    if method == "single_pair":
        method = _method_name("vector", [pair], False)
    return [
        _image_path(output_dir, phase, pair, method, seed) for seed in seeds
    ]


@torch.inference_mode()
def generate_joint_images(
    resolved: Mapping[str, object], output_dir: Path, n: int
) -> None:
    from diffusers import DiffusionPipeline

    if not prepare_joint_weights(resolved, output_dir, n):
        return
    pairs = _pairs(resolved)[:n]
    device = torch.device(str(resolved["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(resolved["model_id"]),
        torch_dtype=_dtype(str(resolved["generation_dtype"])),
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    generation = resolved["generation"]
    seeds = list(resolved["seeds"]["joint"])
    for objective in ["vector", "subspace"]:
        method = f"joint_{objective}"
        apply_weight_state(
            pipe.unet, load_file(str(_joint_checkpoint(output_dir, objective, n)))
        )
        for pair in pairs:
            for index, seed in enumerate(seeds, start=1):
                destination = _joint_image_path(
                    output_dir, n, pair, method, seed
                )
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                image = pipe(
                    prompt=pair.prompt,
                    num_inference_steps=int(generation["num_inference_steps"]),
                    guidance_scale=float(generation["guidance_scale"]),
                    height=int(generation["height"]),
                    width=int(generation["width"]),
                    generator=torch.Generator(device=device).manual_seed(int(seed)),
                ).images[0]
                image.save(destination)
                print(
                    f"[joint N={n}] {method} {pair.slug} "
                    f"{index}/{len(seeds)} seed={seed}",
                    flush=True,
                )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def _method_paths(
    resolved: Mapping[str, object],
    output_dir: Path,
    n: int,
    pair: PairSpec,
    method: str,
) -> list[Path]:
    seeds = list(resolved["seeds"]["joint"])
    if method in {"original", "single_pair"}:
        return _initial_reference_paths(resolved, output_dir, pair, method)
    return [
        _joint_image_path(output_dir, n, pair, method, seed) for seed in seeds
    ]


def evaluate_joint_images(
    resolved: Mapping[str, object], output_dir: Path, n: int
) -> None:
    pairs = _pairs(resolved)[:n]
    if n == 5 and not check_joint5_gate(resolved, output_dir)["can_run_n5"]:
        return
    seeds = list(resolved["seeds"]["joint"])
    methods = ["original", "single_pair", "joint_vector", "joint_subspace"]
    anchor_texts = [_evaluation_text(resolved, pair.anchor) for pair in pairs]
    target_texts = [_evaluation_text(resolved, pair.target) for pair in pairs]
    evaluator = ClipEvaluator(
        str(resolved["clip_model_id"]), str(resolved["device"])
    )
    per_image: list[dict[str, object]] = []
    similarity_cells: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    per_concept: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []
    for method in methods:
        cube = np.empty((n, len(seeds), n), dtype=np.float32)
        target_scores = np.empty((n, len(seeds)), dtype=np.float32)
        method_paths: dict[str, list[Path]] = {}
        for target_index, pair in enumerate(pairs):
            paths = _method_paths(resolved, output_dir, n, pair, method)
            missing = [path for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing N={n} {method} images: {missing[:3]}"
                )
            method_paths[pair.slug] = paths
            cube[target_index] = evaluator.similarities(paths, anchor_texts)
            target_scores[target_index] = evaluator.similarities(
                paths, [target_texts[target_index]]
            )[:, 0]
        correspondence_rows, correspondence_aggregate = (
            compute_correspondence_metrics(cube)
        )
        for target_index, pair in enumerate(pairs):
            for sample_index, seed in enumerate(seeds):
                for anchor_index, anchor_pair in enumerate(pairs):
                    similarity_cells.append(
                        {
                            "n": n,
                            "method": method,
                            "target_index": target_index,
                            "target": pair.target,
                            "seed": seed,
                            "anchor_index": anchor_index,
                            "anchor": anchor_pair.anchor,
                            "similarity": float(
                                cube[target_index, sample_index, anchor_index]
                            ),
                            "is_own_anchor": target_index == anchor_index,
                        }
                    )
        for row in correspondence_rows:
            target_index = int(row["target_index"])
            sample_index = int(row["sample_index"])
            pair = pairs[target_index]
            per_image.append(
                {
                    "n": n,
                    "method": method,
                    "target_index": target_index,
                    "target": pair.target,
                    "anchor": pair.anchor,
                    "seed": seeds[sample_index],
                    "target_similarity": float(
                        target_scores[target_index, sample_index]
                    ),
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"target_index", "sample_index"}
                    },
                    "predicted_anchor": pairs[
                        int(row["predicted_anchor_index"])
                    ].anchor,
                    "image_path": str(
                        method_paths[pair.slug][sample_index].resolve()
                    ),
                }
            )
        mean_matrix = cube.mean(axis=1)
        plot_heatmap(
            output_dir
            / "heatmaps"
            / f"joint_n{n}"
            / f"image_similarity_{method}.png",
            mean_matrix,
            [pair.target for pair in pairs],
            [pair.anchor for pair in pairs],
            f"N={n} image-to-anchor cosine similarity | {method}",
        )
        confusion = np.zeros((n, n), dtype=np.int64)
        predictions = cube.argmax(axis=2)
        for target_index in range(n):
            for prediction in predictions[target_index]:
                confusion[target_index, prediction] += 1
        for target_index, pair in enumerate(pairs):
            for anchor_index, anchor_pair in enumerate(pairs):
                confusion_rows.append(
                    {
                        "n": n,
                        "method": method,
                        "target_index": target_index,
                        "target": pair.target,
                        "predicted_anchor_index": anchor_index,
                        "predicted_anchor": anchor_pair.anchor,
                        "count": int(confusion[target_index, anchor_index]),
                    }
                )
        plot_heatmap(
            output_dir
            / "heatmaps"
            / f"joint_n{n}"
            / f"image_confusion_{method}.png",
            confusion,
            [pair.target for pair in pairs],
            [pair.anchor for pair in pairs],
            f"N={n} image anchor top-1 counts | {method}",
            value_format=".0f",
        )
        for target_index, pair in enumerate(pairs):
            other = np.delete(cube[target_index], target_index, axis=1)
            own = cube[target_index, :, target_index]
            margin = own - other.max(axis=1)
            per_concept.append(
                {
                    "n": n,
                    "method": method,
                    "target": pair.target,
                    "anchor": pair.anchor,
                    "mean_target_similarity": float(
                        target_scores[target_index].mean()
                    ),
                    "mean_own_anchor_similarity": float(own.mean()),
                    "mean_best_other_anchor_similarity": float(
                        other.max(axis=1).mean()
                    ),
                    "mean_margin": float(margin.mean()),
                    "minimum_margin": float(margin.min()),
                    "positive_margin_fraction": float(np.mean(margin > 0)),
                    "own_anchor_top1_rate": float(
                        np.mean(predictions[target_index] == target_index)
                    ),
                }
            )
        aggregate.append(
            {"n": n, "method": method, **correspondence_aggregate}
        )
        for pair in pairs:
            grid_paths = {
                "Original": _method_paths(
                    resolved, output_dir, n, pair, "original"
                ),
                "Single Pair": _method_paths(
                    resolved, output_dir, n, pair, "single_pair"
                ),
                "Joint Vector": _method_paths(
                    resolved, output_dir, n, pair, "joint_vector"
                ),
                "Joint Subspace": _method_paths(
                    resolved, output_dir, n, pair, "joint_subspace"
                ),
            }
            grid_path = (
                output_dir / "grids" / f"joint_n{n}_{pair.slug}.png"
            )
            if not grid_path.exists():
                make_seed_grid(
                    grid_path,
                    grid_paths,
                    seeds,
                    f"N={n} | {pair.target} -> {pair.anchor}",
                )
    write_csv(
        output_dir / "metrics" / f"joint_n{n}_per_image.csv", per_image
    )
    write_csv(
        output_dir / "metrics" / f"joint_n{n}_image_similarity_cells.csv",
        similarity_cells,
    )
    write_csv(
        output_dir / "metrics" / f"joint_n{n}_image_confusion.csv",
        confusion_rows,
    )
    write_csv(
        output_dir / "metrics" / f"joint_n{n}_per_concept.csv", per_concept
    )
    write_csv(
        output_dir / "metrics" / f"joint_n{n}_aggregate.csv", aggregate
    )
    write_json(
        output_dir / "metrics" / f"joint_n{n}_summary.json",
        {
            "n": n,
            "pairs": [
                {"target": pair.target, "anchor": pair.anchor} for pair in pairs
            ],
            "aggregate": aggregate,
            "per_concept": per_concept,
            "similarity_cells_csv": f"joint_n{n}_image_similarity_cells.csv",
            "confusion_csv": f"joint_n{n}_image_confusion.csv",
        },
    )
    state = read_json(output_dir / "run_state.json")
    state[f"joint_n{n}"] = (
        "complete"
        if state.get(f"joint_n{n}") == "feature_evaluation_complete"
        else "image_evaluation_complete"
    )
    write_json(output_dir / "run_state.json", state)
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate_joint_features(
    resolved: Mapping[str, object], output_dir: Path, n: int
) -> None:
    from diffusers import DiffusionPipeline

    pairs = _pairs(resolved)[:n]
    if n == 5 and not check_joint5_gate(resolved, output_dir)["can_run_n5"]:
        return
    device = torch.device(str(resolved["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(resolved["model_id"]),
        torch_dtype=torch.float32,
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)
    embeddings = _encode_last_content_tokens(
        pipe,
        [value for pair in pairs for value in (pair.target, pair.anchor)],
        device,
    )
    base_layers = {
        name: module.weight.detach().float()
        for name, module in collect_projection_modules(pipe.unet)
    }
    joint_states = {
        "joint_vector": load_file(
            str(_joint_checkpoint(output_dir, "vector", n)), device=str(device)
        ),
        "joint_subspace": load_file(
            str(_joint_checkpoint(output_dir, "subspace", n)), device=str(device)
        ),
    }
    single_states = {
        pair.slug: load_file(
            str(
                output_dir
                / "weights"
                / f"{_method_name('vector', [pair], False)}.safetensors"
            ),
            device=str(device),
        )
        for pair in pairs
    }
    cell_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for layer_index, (layer, base_weight) in enumerate(
        base_layers.items(), start=1
    ):
        anchor_features = torch.stack(
            [base_weight @ embeddings[pair.anchor] for pair in pairs]
        )
        for method in [
            "original",
            "single_pair",
            "joint_vector",
            "joint_subspace",
        ]:
            target_features = []
            for pair in pairs:
                if method == "original":
                    edited_weight = base_weight
                elif method == "single_pair":
                    edited_weight = single_states[pair.slug][f"{layer}.weight"]
                else:
                    edited_weight = joint_states[method][f"{layer}.weight"]
                target_features.append(edited_weight @ embeddings[pair.target])
            target_matrix = torch.nn.functional.normalize(
                torch.stack(target_features).float(), dim=1
            )
            anchor_matrix = torch.nn.functional.normalize(
                anchor_features.float(), dim=1
            )
            matrix = (target_matrix @ anchor_matrix.T).cpu().numpy()
            rows, aggregate = compute_correspondence_metrics(
                matrix[:, None, :]
            )
            summary_rows.append(
                {
                    "n": n,
                    "layer_index": layer_index,
                    "layer": layer,
                    "method": method,
                    **aggregate,
                }
            )
            for target_index, pair in enumerate(pairs):
                for anchor_index, anchor_pair in enumerate(pairs):
                    cell_rows.append(
                        {
                            "n": n,
                            "layer_index": layer_index,
                            "layer": layer,
                            "method": method,
                            "target_index": target_index,
                            "target": pair.target,
                            "anchor_index": anchor_index,
                            "anchor": anchor_pair.anchor,
                            "similarity": float(
                                matrix[target_index, anchor_index]
                            ),
                            "is_own_anchor": target_index == anchor_index,
                        }
                    )
            plot_heatmap(
                output_dir
                / "heatmaps"
                / f"joint_n{n}"
                / "feature"
                / f"layer_{layer_index:02d}_{method}.png",
                matrix,
                [pair.target for pair in pairs],
                [pair.anchor for pair in pairs],
                f"N={n} feature correspondence | L{layer_index} | {method}",
            )
    write_csv(
        output_dir / "metrics" / f"joint_n{n}_feature_cells.csv", cell_rows
    )
    write_csv(
        output_dir / "metrics" / f"joint_n{n}_feature_summary.csv",
        summary_rows,
    )
    write_json(
        output_dir / "metrics" / f"joint_n{n}_feature_summary.json",
        {"n": n, "layers": summary_rows},
    )
    state = read_json(output_dir / "run_state.json")
    state[f"joint_n{n}"] = "feature_evaluation_complete"
    write_json(output_dir / "run_state.json", state)
    del pipe, joint_states, single_states
    gc.collect()
    torch.cuda.empty_cache()


def run_joint(
    resolved: Mapping[str, object], output_dir: Path, n: int
) -> None:
    if not prepare_joint_weights(resolved, output_dir, n):
        return
    # The feature-level diagnostic is intentionally computed before generation.
    evaluate_joint_features(resolved, output_dir, n)
    generate_joint_images(resolved, output_dir, n)
    evaluate_joint_images(resolved, output_dir, n)
    update_joint_report(resolved, output_dir, n)


def _replace_report_section(
    report: str, heading: str, next_heading: str, body: str
) -> str:
    start = report.index(heading)
    end = report.index(next_heading, start)
    return report[:start] + heading + "\n\n" + body.rstrip() + "\n\n" + report[end:]


def update_joint_report(
    resolved: Mapping[str, object], output_dir: Path, n: int
) -> None:
    report_path = output_dir / "report.md"
    if not report_path.exists():
        raise FileNotFoundError("Initial Markdown report is required")
    per_concept = read_csv(
        output_dir / "metrics" / f"joint_n{n}_per_concept.csv"
    )
    aggregate = read_csv(
        output_dir / "metrics" / f"joint_n{n}_aggregate.csv"
    )
    features = read_csv(
        output_dir / "metrics" / f"joint_n{n}_feature_summary.csv"
    )
    pairs = _pairs(resolved)[:n]
    report = report_path.read_text(encoding="utf-8")
    heading_map = {
        (2, "joint_vector"): (
            "## 2. N = 2 joint vector-wise",
            "## 3. N = 2 joint subspace",
        ),
        (2, "joint_subspace"): (
            "## 3. N = 2 joint subspace",
            "## 4. N = 5 joint vector-wise",
        ),
        (5, "joint_vector"): (
            "## 4. N = 5 joint vector-wise",
            "## 5. N = 5 joint subspace",
        ),
        (5, "joint_subspace"): (
            "## 5. N = 5 joint subspace",
            "## 6. Control-set preservation",
        ),
    }
    for method in ["joint_vector", "joint_subspace"]:
        rows = [row for row in per_concept if row["method"] == method]
        baseline = {
            row["target"]: row
            for row in per_concept
            if row["method"] == "single_pair"
        }
        table_rows = []
        for row in rows:
            base = baseline[row["target"]]
            table_rows.append(
                "| {target} → {anchor} | {target_sim:.4f} | {own:.4f} | "
                "{other:.4f} | {margin:+.4f} | {minimum:+.4f} | "
                "{positive:.1%} | {top1:.1%} | {delta:+.1%} |".format(
                    target=row["target"],
                    anchor=row["anchor"],
                    target_sim=float(row["mean_target_similarity"]),
                    own=float(row["mean_own_anchor_similarity"]),
                    other=float(row["mean_best_other_anchor_similarity"]),
                    margin=float(row["mean_margin"]),
                    minimum=float(row["minimum_margin"]),
                    positive=float(row["positive_margin_fraction"]),
                    top1=float(row["own_anchor_top1_rate"]),
                    delta=(
                        float(row["own_anchor_top1_rate"])
                        - float(base["own_anchor_top1_rate"])
                    ),
                )
            )
        agg = next(row for row in aggregate if row["method"] == method)
        feature_rows = [row for row in features if row["method"] == method]
        feature_mean_top1 = float(
            np.mean([float(row["own_anchor_top1_rate"]) for row in feature_rows])
        )
        feature_mean_margin = float(
            np.mean([float(row["mean_margin"]) for row in feature_rows])
        )
        feature_min_margin = float(
            np.min([float(row["minimum_margin"]) for row in feature_rows])
        )
        objective_label = (
            "vector-wise" if method == "joint_vector" else "subspace"
        )
        body = f"""The N={n} {objective_label} checkpoint uses exactly the same target/anchor pairs, retain set, scales, layers, generation settings, prompts, and seeds as its paired comparator. The table reports descriptive correspondence results; it does not assume the joint method improved or degraded.

| Mapping | Target sim | Own anchor | Best other | Mean margin | Min margin | Positive margin | Own-anchor top-1 | Top-1 Δ vs single |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

Across all images, own-anchor top-1 is {float(agg["own_anchor_top1_rate"]):.1%}, mean correspondence margin is {float(agg["mean_margin"]):+.4f}, minimum margin is {float(agg["minimum_margin"]):+.4f}, and {float(agg["positive_margin_fraction"]):.1%} of images have positive margin.

The pre-generation feature diagnostic spans all 16 edited layers. Its layer-mean own-anchor top-1 is {feature_mean_top1:.1%}, layer-mean correspondence margin is {feature_mean_margin:+.4f}, and the minimum layer/concept margin is {feature_min_margin:+.4f}. Exact layer rows and every matrix cell are saved in `metrics/joint_n{n}_feature_summary.csv` and `metrics/joint_n{n}_feature_cells.csv`.

![N={n} {objective_label} image similarity](heatmaps/joint_n{n}/image_similarity_{method}.png)

![N={n} {objective_label} image confusion](heatmaps/joint_n{n}/image_confusion_{method}.png)

"""
        for pair in pairs:
            body += (
                f"![N={n} {pair.target} comparison]"
                f"(grids/joint_n{n}_{pair.slug}.png)\n\n"
            )
        heading, next_heading = heading_map[(n, method)]
        report = _replace_report_section(report, heading, next_heading, body)
    report_path.write_text(report, encoding="utf-8")


def _control_image_path(
    output_dir: Path, concept: str, method: str, seed: int
) -> Path:
    slug = "_".join(concept.casefold().split())
    return (
        output_dir
        / "images"
        / "controls"
        / slug
        / method
        / f"seed_{seed}.png"
    )


@torch.inference_mode()
def run_controls(
    resolved: Mapping[str, object], output_dir: Path
) -> None:
    from diffusers import DiffusionPipeline
    import lpips

    source_n = 5
    if not all(
        _joint_checkpoint(output_dir, objective, source_n).exists()
        for objective in ["vector", "subspace"]
    ):
        source_n = 2
    for objective in ["vector", "subspace"]:
        if not _joint_checkpoint(output_dir, objective, source_n).exists():
            raise FileNotFoundError(
                "Run a joint experiment before the control evaluation"
            )
    controls = list(resolved["control_concepts"])
    targets_and_anchors = {
        value.casefold()
        for pair in _pairs(resolved)
        for value in (pair.target, pair.anchor)
    }
    overlap = sorted(
        value for value in controls if value.casefold() in targets_and_anchors
    )
    if overlap:
        raise ValueError(f"Control overlap detected at execution time: {overlap}")
    device = torch.device(str(resolved["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(resolved["model_id"]),
        torch_dtype=_dtype(str(resolved["generation_dtype"])),
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    original_state = clone_projection_state(pipe.unet)
    states = {
        "original": original_state,
        "joint_vector": load_file(
            str(_joint_checkpoint(output_dir, "vector", source_n))
        ),
        "joint_subspace": load_file(
            str(_joint_checkpoint(output_dir, "subspace", source_n))
        ),
    }
    seeds = list(resolved["seeds"]["control"])
    generation = resolved["generation"]
    for method, state in states.items():
        apply_weight_state(pipe.unet, state)
        for concept in controls:
            prompt = _evaluation_text(resolved, concept)
            for seed in seeds:
                destination = _control_image_path(
                    output_dir, concept, method, seed
                )
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=int(generation["num_inference_steps"]),
                    guidance_scale=float(generation["guidance_scale"]),
                    height=int(generation["height"]),
                    width=int(generation["width"]),
                    generator=torch.Generator(device=device).manual_seed(int(seed)),
                ).images[0]
                image.save(destination)
    del pipe, states, original_state
    gc.collect()
    torch.cuda.empty_cache()
    evaluator = ClipEvaluator(
        str(resolved["clip_model_id"]), str(resolved["device"])
    )
    lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for concept in controls:
        text = [_evaluation_text(resolved, concept)]
        original_paths = [
            _control_image_path(output_dir, concept, "original", seed)
            for seed in seeds
        ]
        original_clip = evaluator.similarities(original_paths, text)[:, 0]
        for method in ["original", "joint_vector", "joint_subspace"]:
            paths = [
                _control_image_path(output_dir, concept, method, seed)
                for seed in seeds
            ]
            clip_values = evaluator.similarities(paths, text)[:, 0]
            lpips_values = []
            for original_path, edited_path in zip(original_paths, paths):
                if method == "original":
                    lpips_values.append(0.0)
                    continue
                tensors = []
                for path in [original_path, edited_path]:
                    array = np.asarray(
                        Image.open(path).convert("RGB"), dtype=np.float32
                    )
                    tensor = (
                        torch.from_numpy(array)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device)
                        / 127.5
                        - 1.0
                    )
                    tensors.append(tensor)
                lpips_values.append(
                    float(lpips_model(tensors[0], tensors[1]).item())
                )
            for index, seed in enumerate(seeds):
                rows.append(
                    {
                        "concept": concept,
                        "source_joint_n": source_n,
                        "method": method,
                        "seed": seed,
                        "prompt_image_clip": float(clip_values[index]),
                        "clip_delta_vs_original_mean": float(
                            clip_values[index] - original_clip[index]
                        ),
                        "lpips_vs_same_seed_original": lpips_values[index],
                        "image_path": str(paths[index].resolve()),
                    }
                )
            summaries.append(
                {
                    "concept": concept,
                    "source_joint_n": source_n,
                    "method": method,
                    "mean_prompt_image_clip": float(clip_values.mean()),
                    "mean_clip_delta_vs_original": float(
                        (clip_values - original_clip).mean()
                    ),
                    "mean_lpips_vs_same_seed_original": float(
                        np.mean(lpips_values)
                    ),
                }
            )
        make_seed_grid(
            output_dir / "grids" / f"control_{concept.replace(' ', '_')}.png",
            {
                "Original": original_paths,
                "Joint Vector": [
                    _control_image_path(
                        output_dir, concept, "joint_vector", seed
                    )
                    for seed in seeds
                ],
                "Joint Subspace": [
                    _control_image_path(
                        output_dir, concept, "joint_subspace", seed
                    )
                    for seed in seeds
                ],
            },
            seeds,
            f"Control concept: {concept}",
        )
    write_csv(output_dir / "metrics" / "control_per_image.csv", rows)
    write_csv(output_dir / "metrics" / "control_summary.csv", summaries)
    write_json(
        output_dir / "metrics" / "control_summary.json",
        {"controls": controls, "source_joint_n": source_n, "summary": summaries},
    )
    state = read_json(output_dir / "run_state.json")
    state["controls"] = "complete"
    write_json(output_dir / "run_state.json", state)
    update_control_report(output_dir, summaries, controls, source_n)


def update_control_report(
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    controls: Sequence[str],
    source_n: int,
) -> None:
    report_path = output_dir / "report.md"
    report = report_path.read_text(encoding="utf-8")
    rows = "\n".join(
        "| {concept} | {method} | {clip:.4f} | {delta:+.4f} | {lpips:.4f} |".format(
            concept=row["concept"],
            method=row["method"],
            clip=float(row["mean_prompt_image_clip"]),
            delta=float(row["mean_clip_delta_vs_original"]),
            lpips=float(row["mean_lpips_vs_same_seed_original"]),
        )
        for row in summaries
    )
    body = f"""The control set was revalidated against the actual configured target and anchor strings immediately before execution. These results use the N={source_n} joint checkpoints and the fixed control seeds.

| Control | Method | Prompt-image CLIP | CLIP Δ vs original | LPIPS vs same-seed original |
|---|---|---:|---:|---:|
{rows}

LPIPS is paired at identical prompt and seed. It measures perceptual change, not semantic damage by itself. CLIP alignment and the grids must be read together; no automatic preservation conclusion is applied.

"""
    for concept in controls:
        body += (
            f"![Control {concept} grid]"
            f"(grids/control_{concept.replace(' ', '_')}.png)\n\n"
        )
    report = _replace_report_section(
        report,
        "## 6. Control-set preservation",
        "## Scope, definitions, and resolved settings",
        body,
    )
    report_path.write_text(report, encoding="utf-8")
