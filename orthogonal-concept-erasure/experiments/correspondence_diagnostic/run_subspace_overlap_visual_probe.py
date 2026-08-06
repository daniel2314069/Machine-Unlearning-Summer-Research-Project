from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from oce_correspondence.core import (
    PairSpec,
    apply_weight_state,
    clone_projection_state,
    edit_projection_weights,
    expand_object_pairs,
)
from oce_correspondence.io_utils import make_seed_grid, sha256, write_json
from oce_correspondence.runner import _dtype, _encode_last_content_tokens


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config_official_subspace.json"
DEFAULT_OUTPUT = HERE / "outputs" / "subspace_overlap_visual_probe"
OFFICIAL_OUTPUT = HERE / "outputs" / "official_subspace"


CASES = {
    "reciprocal_cat_dog": [
        PairSpec("cat", "dog", "a photo of a cat"),
        PairSpec("dog", "cat", "a photo of a dog"),
    ],
    "chain_cat_dog_bird_cat": [
        PairSpec("cat", "dog", "a photo of a cat"),
        PairSpec("bird", "cat", "a photo of a bird"),
    ],
}


def single_method(pair: PairSpec) -> str:
    return f"single_subspace_{pair.slug}"


def joint_method(case: str) -> str:
    return f"joint_subspace_{case}"


def checkpoint(output: Path, method: str) -> Path:
    return output / "checkpoints" / f"{method}.safetensors"


def image_path(
    output: Path, case: str, pair: PairSpec, method: str, seed: int
) -> Path:
    return (
        output
        / "images"
        / case
        / pair.slug
        / method
        / f"seed_{seed}.png"
    )


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    cg_path = (path.parent / str(config["cg_path"])).resolve()
    config["cg_path"] = str(cg_path)
    # This probe intentionally violates the disjoint-set rule. That is the
    # experimental variable, not a validation omission.
    config["overlap_validation_override"] = {
        "intentional": True,
        "reason": (
            "Visual probe of reciprocal and chained target-anchor overlap; "
            "the official subspace objective itself is unchanged."
        ),
    }
    return config


@torch.inference_mode()
def prepare_checkpoints(config: Mapping[str, object], output: Path) -> None:
    from diffusers import DiffusionPipeline

    methods: list[tuple[str, list[PairSpec]]] = []
    unique_singles: dict[str, PairSpec] = {}
    for case, pair_list in CASES.items():
        methods.append((joint_method(case), pair_list))
        for pair in pair_list:
            unique_singles[pair.slug] = pair
    for pair in unique_singles.values():
        if pair.slug != "cat_to_dog":
            methods.append((single_method(pair), [pair]))
    pending = [
        (method, pair_list)
        for method, pair_list in methods
        if not checkpoint(output, method).exists()
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
    all_text: list[str] = []
    expanded_by_method: dict[str, list[PairSpec]] = {}
    for method, pair_list in pending:
        expanded = (
            expand_object_pairs(pair_list)
            if config["oce"]["expand_prompts"]
            else pair_list
        )
        expanded_by_method[method] = expanded
        for pair in expanded:
            all_text.extend([pair.target, pair.anchor])
        all_text.extend(pair.anchor for pair in pair_list)
    embeddings = _encode_last_content_tokens(pipe, all_text, device)
    cg = torch.load(str(config["cg_path"]), map_location=device)["C"].float()
    oce = config["oce"]
    audit = []
    for method, pair_list in pending:
        edited, rows = edit_projection_weights(
            unet=pipe.unet,
            embeddings=embeddings,
            pairs=expanded_by_method[method],
            preserve_concepts=(
                [pair.anchor for pair in pair_list]
                if oce["preserve_anchors"]
                else []
            ),
            global_second_moment=cg,
            objective="subspace",
            erase_scale=float(oce["erase_scale"]),
            preserve_global_scale=float(oce["preserve_global_scale"]),
            preserve_concept_scale=float(oce["preserve_concept_scale"]),
            lamb=float(oce["lambda"]),
            reflection_correction=str(oce["reflection_correction"]),
        )
        checkpoint(output, method).parent.mkdir(parents=True, exist_ok=True)
        save_file(edited, str(checkpoint(output, method)))
        audit.append(
            {
                "method": method,
                "pairs": [
                    {"target": pair.target, "anchor": pair.anchor}
                    for pair in pair_list
                ],
                "checkpoint": str(checkpoint(output, method).resolve()),
                "sha256": sha256(checkpoint(output, method)),
                "layer_count": len(rows),
                "objective": "official OCE subspace",
            }
        )
        print(f"[checkpoint] {method}", flush=True)
    write_json(output / "checkpoint_manifest.json", {"checkpoints": audit})
    del pipe, cg
    gc.collect()
    torch.cuda.empty_cache()


def _copy_reusable_images(
    config: Mapping[str, object], output: Path
) -> dict[str, str]:
    seeds = list(config["seeds"]["joint"])
    sources = {
        "original_cat": OFFICIAL_OUTPUT / "images" / "feasibility" / "cat",
        "original_dog": OFFICIAL_OUTPUT / "images" / "feasibility" / "dog",
        "single_cat_to_dog": (
            OFFICIAL_OUTPUT
            / "images"
            / "single_pair"
            / "cat_to_dog"
            / "single_subspace_cat_to_dog"
        ),
    }
    copied: dict[str, str] = {}
    for case, pair_list in CASES.items():
        for pair in pair_list:
            if pair.target in {"cat", "dog"}:
                source_dir = sources[f"original_{pair.target}"]
                for seed in seeds:
                    destination = image_path(
                        output, case, pair, "original", seed
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        shutil.copy2(source_dir / f"seed_{seed}.png", destination)
                copied[f"{case}:{pair.target}:original"] = str(
                    source_dir.resolve()
                )
            if pair.slug == "cat_to_dog":
                source_dir = sources["single_cat_to_dog"]
                for seed in seeds:
                    destination = image_path(
                        output, case, pair, "single_subspace", seed
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        shutil.copy2(source_dir / f"seed_{seed}.png", destination)
                copied[f"{case}:{pair.slug}:single_subspace"] = str(
                    source_dir.resolve()
                )
    return copied


@torch.inference_mode()
def generate_images(config: Mapping[str, object], output: Path) -> None:
    from diffusers import DiffusionPipeline

    reused = _copy_reusable_images(config, output)
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=_dtype(str(config["generation_dtype"])),
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    original_state = clone_projection_state(pipe.unet)
    generation = config["generation"]
    seeds = list(config["seeds"]["joint"])
    for case, pair_list in CASES.items():
        for pair in pair_list:
            methods = ["original", "single_subspace", "joint_subspace"]
            for method in methods:
                destinations = [
                    image_path(output, case, pair, method, seed)
                    for seed in seeds
                ]
                if all(path.exists() for path in destinations):
                    continue
                if method == "original":
                    apply_weight_state(pipe.unet, original_state)
                elif method == "single_subspace":
                    state_path = (
                        OFFICIAL_OUTPUT
                        / "checkpoints"
                        / "single_subspace_cat_to_dog.safetensors"
                        if pair.slug == "cat_to_dog"
                        else checkpoint(output, single_method(pair))
                    )
                    apply_weight_state(pipe.unet, load_file(str(state_path)))
                else:
                    apply_weight_state(
                        pipe.unet,
                        load_file(str(checkpoint(output, joint_method(case)))),
                    )
                for index, (seed, destination) in enumerate(
                    zip(seeds, destinations), 1
                ):
                    if destination.exists():
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image = pipe(
                        prompt=pair.prompt,
                        num_inference_steps=int(
                            generation["num_inference_steps"]
                        ),
                        guidance_scale=float(generation["guidance_scale"]),
                        height=int(generation["height"]),
                        width=int(generation["width"]),
                        generator=torch.Generator(device=device).manual_seed(
                            int(seed)
                        ),
                    ).images[0]
                    image.save(destination)
                    print(
                        f"[image] {case} {pair.slug} {method} "
                        f"{index}/{len(seeds)} seed={seed}",
                        flush=True,
                    )
    write_json(output / "reuse_manifest.json", reused)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def build_grids(config: Mapping[str, object], output: Path) -> None:
    seeds = list(config["seeds"]["joint"])
    grid_manifest = []
    for case, pair_list in CASES.items():
        for pair in pair_list:
            destination = output / "grids" / f"{case}_{pair.slug}.png"
            make_seed_grid(
                destination,
                {
                    "Original SD": [
                        image_path(output, case, pair, "original", seed)
                        for seed in seeds
                    ],
                    "Single Subspace": [
                        image_path(
                            output, case, pair, "single_subspace", seed
                        )
                        for seed in seeds
                    ],
                    "Joint Subspace": [
                        image_path(
                            output, case, pair, "joint_subspace", seed
                        )
                        for seed in seeds
                    ],
                },
                seeds,
                f"{case} | {pair.target} -> {pair.anchor}",
            )
            grid_manifest.append(
                {
                    "case": case,
                    "target": pair.target,
                    "anchor": pair.anchor,
                    "grid": str(destination.resolve()),
                }
            )
    write_json(output / "grid_manifest.json", {"grids": grid_manifest})


def run(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "resolved_visual_probe.json",
        {
            "cases": {
                case: [
                    {
                        "target": pair.target,
                        "anchor": pair.anchor,
                        "prompt": pair.prompt,
                    }
                    for pair in pair_list
                ]
                for case, pair_list in CASES.items()
            },
            "seeds": list(config["seeds"]["joint"]),
            "model_id": config["model_id"],
            "oce": config["oce"],
            "generation": config["generation"],
            "overlap_validation_override": config[
                "overlap_validation_override"
            ],
            "scope": "images and seed-aligned grids only; no evaluation metrics",
        },
    )
    prepare_checkpoints(config, output)
    generate_images(config, output)
    build_grids(config, output)
    print(f"Grids: {output / 'grids'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual-only official-subspace overlap probe"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
