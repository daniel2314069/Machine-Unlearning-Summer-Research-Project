from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


HERE = Path(__file__).resolve().parent
CORRESPONDENCE_DIR = HERE.parent / "correspondence_diagnostic"
sys.path.insert(0, str(CORRESPONDENCE_DIR))

from oce_correspondence.core import (  # noqa: E402
    PairSpec,
    apply_weight_state,
    clone_projection_state,
    edit_projection_weights,
    expand_object_pairs,
)
from oce_correspondence.io_utils import make_seed_grid, sha256, write_json  # noqa: E402
from oce_correspondence.runner import _dtype, _encode_last_content_tokens  # noqa: E402


DEFAULT_CONFIG = CORRESPONDENCE_DIR / "config_official_subspace.json"
OFFICIAL_OUTPUT = CORRESPONDENCE_DIR / "outputs" / "official_subspace"
OVERLAP_OUTPUT = (
    CORRESPONDENCE_DIR / "outputs" / "subspace_overlap_visual_probe"
)

SINGLE_PAIRS = [
    PairSpec("cat", "dog", "a photo of a cat"),
    PairSpec("dog", "cat", "a photo of a dog"),
    PairSpec("horse", "deer", "a photo of a horse"),
    PairSpec("dog", "wolf", "a photo of a dog"),
    PairSpec("wolf", "cat", "a photo of a wolf"),
]

JOINT_CASES = {
    "cycle2": [
        PairSpec("cat", "dog", "a photo of a cat"),
        PairSpec("dog", "cat", "a photo of a dog"),
    ],
    "cycle3": [
        PairSpec("cat", "dog", "a photo of a cat"),
        PairSpec("dog", "wolf", "a photo of a dog"),
        PairSpec("wolf", "cat", "a photo of a wolf"),
    ],
    "no_overlap": [
        PairSpec("cat", "dog", "a photo of a cat"),
        PairSpec("horse", "deer", "a photo of a horse"),
    ],
}


def single_method(pair: PairSpec) -> str:
    return f"single_subspace_{pair.slug}"


def joint_method(case: str) -> str:
    return f"joint_subspace_{case}"


def checkpoint(method: str) -> Path:
    return HERE / "checkpoints" / f"{method}.safetensors"


def single_image(pair: PairSpec, method: str, seed: int) -> Path:
    return (
        HERE
        / "single_pairs"
        / pair.slug
        / "images"
        / method
        / f"seed_{seed}.png"
    )


def joint_image(
    case: str, pair: PairSpec, method: str, seed: int
) -> Path:
    return (
        HERE
        / case
        / "images"
        / pair.target
        / method
        / f"seed_{seed}.png"
    )


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["cg_path"] = str(
        (path.parent / str(config["cg_path"])).resolve()
    )
    config["seeds"] = {
        **config["seeds"],
        "visual": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    }
    return config


def reusable_checkpoints() -> dict[str, Path]:
    return {
        "single_subspace_cat_to_dog": (
            OFFICIAL_OUTPUT
            / "checkpoints"
            / "single_subspace_cat_to_dog.safetensors"
        ),
        "single_subspace_dog_to_cat": (
            OVERLAP_OUTPUT
            / "checkpoints"
            / "single_subspace_dog_to_cat.safetensors"
        ),
        "joint_subspace_cycle2": (
            OVERLAP_OUTPUT
            / "checkpoints"
            / "joint_subspace_reciprocal_cat_dog.safetensors"
        ),
    }


def copy_reusable_checkpoints() -> None:
    for method, source in reusable_checkpoints().items():
        destination = checkpoint(method)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)


@torch.inference_mode()
def prepare_checkpoints(config: Mapping[str, object]) -> None:
    from diffusers import DiffusionPipeline

    copy_reusable_checkpoints()
    specifications: list[tuple[str, list[PairSpec]]] = []
    for pair in SINGLE_PAIRS:
        specifications.append((single_method(pair), [pair]))
    for case, pairs in JOINT_CASES.items():
        specifications.append((joint_method(case), pairs))
    pending = [
        (method, pairs)
        for method, pairs in specifications
        if not checkpoint(method).exists()
    ]
    if pending:
        device = torch.device(str(config["device"]))
        pipe = DiffusionPipeline.from_pretrained(
            str(config["model_id"]),
            torch_dtype=_dtype(str(config["edit_dtype"])),
            safety_checker=None,
            vae=None,
            local_files_only=True,
        ).to(device)
        expanded: dict[str, list[PairSpec]] = {}
        all_text: list[str] = []
        for method, pairs in pending:
            method_pairs = (
                expand_object_pairs(pairs)
                if config["oce"]["expand_prompts"]
                else pairs
            )
            expanded[method] = method_pairs
            for pair in method_pairs:
                all_text.extend([pair.target, pair.anchor])
            all_text.extend(pair.anchor for pair in pairs)
        embeddings = _encode_last_content_tokens(pipe, all_text, device)
        cg = torch.load(
            str(config["cg_path"]), map_location=device
        )["C"].float()
        oce = config["oce"]
        for method, pairs in pending:
            edited, audit = edit_projection_weights(
                unet=pipe.unet,
                embeddings=embeddings,
                pairs=expanded[method],
                preserve_concepts=(
                    [pair.anchor for pair in pairs]
                    if oce["preserve_anchors"]
                    else []
                ),
                global_second_moment=cg,
                objective="subspace",
                erase_scale=float(oce["erase_scale"]),
                preserve_global_scale=float(
                    oce["preserve_global_scale"]
                ),
                preserve_concept_scale=float(
                    oce["preserve_concept_scale"]
                ),
                lamb=float(oce["lambda"]),
                reflection_correction=str(
                    oce["reflection_correction"]
                ),
            )
            checkpoint(method).parent.mkdir(parents=True, exist_ok=True)
            save_file(edited, str(checkpoint(method)))
            if len(audit) != 16:
                raise RuntimeError(
                    f"{method}: expected 16 edited layers, got {len(audit)}"
                )
            print(f"[checkpoint] {method}", flush=True)
        del pipe, cg
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        HERE / "checkpoints" / "manifest.json",
        {
            "objective": "official OCE subspace",
            "checkpoints": [
                {
                    "method": method,
                    "pairs": [
                        {"target": pair.target, "anchor": pair.anchor}
                        for pair in pairs
                    ],
                    "path": str(checkpoint(method).resolve()),
                    "sha256": sha256(checkpoint(method)),
                }
                for method, pairs in specifications
            ],
        },
    )


def reusable_single_sources(
    pair: PairSpec,
) -> tuple[Path | None, Path | None]:
    original = {
        "cat": OFFICIAL_OUTPUT / "images" / "feasibility" / "cat",
        "dog": OFFICIAL_OUTPUT / "images" / "feasibility" / "dog",
    }.get(pair.target)
    edited = {
        "cat_to_dog": (
            OFFICIAL_OUTPUT
            / "images"
            / "single_pair"
            / "cat_to_dog"
            / "single_subspace_cat_to_dog"
        ),
        "dog_to_cat": (
            OVERLAP_OUTPUT
            / "images"
            / "reciprocal_cat_dog"
            / "dog_to_cat"
            / "single_subspace"
        ),
    }.get(pair.slug)
    return original, edited


def copy_directory_seeds(
    source: Path, destinations: Sequence[Path], seeds: Sequence[int]
) -> None:
    for seed, destination in zip(seeds, destinations):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source / f"seed_{seed}.png", destination)


def seed_single_reuse(config: Mapping[str, object]) -> None:
    seeds = list(config["seeds"]["visual"])
    for pair in SINGLE_PAIRS:
        original_source, edited_source = reusable_single_sources(pair)
        if original_source is not None:
            copy_directory_seeds(
                original_source,
                [single_image(pair, "original", seed) for seed in seeds],
                seeds,
            )
        if edited_source is not None:
            copy_directory_seeds(
                edited_source,
                [
                    single_image(pair, "single_subspace", seed)
                    for seed in seeds
                ],
                seeds,
            )


@torch.inference_mode()
def generate_single_images(config: Mapping[str, object]) -> None:
    from diffusers import DiffusionPipeline

    seed_single_reuse(config)
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
    seeds = list(config["seeds"]["visual"])
    for pair in SINGLE_PAIRS:
        for method in ("original", "single_subspace"):
            destinations = [
                single_image(pair, method, seed) for seed in seeds
            ]
            if all(path.exists() for path in destinations):
                continue
            state = (
                original_state
                if method == "original"
                else load_file(str(checkpoint(single_method(pair))))
            )
            apply_weight_state(pipe.unet, state)
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
                    f"[single] {pair.slug} {method} "
                    f"{index}/{len(seeds)} seed={seed}",
                    flush=True,
                )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def build_single_grids(config: Mapping[str, object]) -> None:
    seeds = list(config["seeds"]["visual"])
    for pair in SINGLE_PAIRS:
        make_seed_grid(
            HERE / "single_pairs" / pair.slug / f"{pair.slug}_single.png",
            {
                "Original SD": [
                    single_image(pair, "original", seed) for seed in seeds
                ],
                "Single Subspace": [
                    single_image(pair, "single_subspace", seed)
                    for seed in seeds
                ],
            },
            seeds,
            f"Single official subspace | {pair.target} -> {pair.anchor}",
        )


def seed_cycle2_reuse(config: Mapping[str, object]) -> None:
    seeds = list(config["seeds"]["visual"])
    old_case = "reciprocal_cat_dog"
    for pair in JOINT_CASES["cycle2"]:
        for old_method, new_method in (
            ("original", "original"),
            ("single_subspace", "single_subspace"),
            ("joint_subspace", "joint_subspace"),
        ):
            source = (
                OVERLAP_OUTPUT
                / "images"
                / old_case
                / pair.slug
                / old_method
            )
            copy_directory_seeds(
                source,
                [
                    joint_image(
                        "cycle2", pair, new_method, seed
                    )
                    for seed in seeds
                ],
                seeds,
            )


def seed_joint_references(config: Mapping[str, object]) -> None:
    seeds = list(config["seeds"]["visual"])
    seed_cycle2_reuse(config)
    single_by_slug = {pair.slug: pair for pair in SINGLE_PAIRS}
    for case, pairs in JOINT_CASES.items():
        if case == "cycle2":
            continue
        for pair in pairs:
            source_pair = single_by_slug[pair.slug]
            for method in ("original", "single_subspace"):
                for seed in seeds:
                    source = single_image(source_pair, method, seed)
                    destination = joint_image(
                        case, pair, method, seed
                    )
                    destination.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    if not destination.exists():
                        shutil.copy2(source, destination)


@torch.inference_mode()
def generate_joint_images(config: Mapping[str, object]) -> None:
    from diffusers import DiffusionPipeline

    seed_joint_references(config)
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=_dtype(str(config["generation_dtype"])),
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    generation = config["generation"]
    seeds = list(config["seeds"]["visual"])
    for case, pairs in JOINT_CASES.items():
        destinations = [
            joint_image(case, pair, "joint_subspace", seed)
            for pair in pairs
            for seed in seeds
        ]
        if all(path.exists() for path in destinations):
            continue
        apply_weight_state(
            pipe.unet, load_file(str(checkpoint(joint_method(case))))
        )
        for pair in pairs:
            for index, seed in enumerate(seeds, 1):
                destination = joint_image(
                    case, pair, "joint_subspace", seed
                )
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
                    f"[joint] {case} {pair.slug} "
                    f"{index}/{len(seeds)} seed={seed}",
                    flush=True,
                )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def build_joint_grids(config: Mapping[str, object]) -> None:
    seeds = list(config["seeds"]["visual"])
    for case, pairs in JOINT_CASES.items():
        for pair in pairs:
            make_seed_grid(
                HERE / case / f"{case}_{pair.target}_prompt.png",
                {
                    "Original SD": [
                        joint_image(case, pair, "original", seed)
                        for seed in seeds
                    ],
                    "Single Subspace": [
                        joint_image(
                            case, pair, "single_subspace", seed
                        )
                        for seed in seeds
                    ],
                    "Joint Subspace": [
                        joint_image(
                            case, pair, "joint_subspace", seed
                        )
                        for seed in seeds
                    ],
                },
                seeds,
                f"{case} | {pair.target} -> {pair.anchor}",
            )


def write_manifest(config: Mapping[str, object]) -> None:
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        str(config["model_id"]),
        subfolder="tokenizer",
        local_files_only=True,
    )
    ids = tokenizer("wolf", add_special_tokens=False)["input_ids"]
    write_json(
        HERE / "resolved_experiment.json",
        {
            "scope": "images and seed-aligned grids only",
            "all_gates_overridden_by_user": True,
            "single_pairs": [
                {
                    "target": pair.target,
                    "anchor": pair.anchor,
                    "prompt": pair.prompt,
                }
                for pair in SINGLE_PAIRS
            ],
            "joint_cases": {
                case: [
                    {
                        "target": pair.target,
                        "anchor": pair.anchor,
                        "prompt": pair.prompt,
                    }
                    for pair in pairs
                ]
                for case, pairs in JOINT_CASES.items()
            },
            "wolf_tokenization": {
                "token_ids": ids,
                "token_strings": tokenizer.convert_ids_to_tokens(ids),
                "token_count": len(ids),
                "problem_detected": False,
            },
            "seeds": list(config["seeds"]["visual"]),
            "generation": config["generation"],
            "oce": config["oce"],
            "model_id": config["model_id"],
            "executed": {
                "clip_metrics": False,
                "lpips": False,
                "feature_matrices": False,
                "permutation": False,
                "confusion": False,
                "heatmaps": False,
                "vector_ablation": False,
            },
        },
    )


def write_grid_index() -> None:
    rows = []
    for pair in SINGLE_PAIRS:
        rows.append(
            {
                "stage": "single_pair",
                "mapping": f"{pair.target} -> {pair.anchor}",
                "path": str(
                    (
                        HERE
                        / "single_pairs"
                        / pair.slug
                        / f"{pair.slug}_single.png"
                    ).resolve()
                ),
            }
        )
    for case, pairs in JOINT_CASES.items():
        for pair in pairs:
            rows.append(
                {
                    "stage": case,
                    "mapping": f"{pair.target} -> {pair.anchor}",
                    "path": str(
                        (
                            HERE
                            / case
                            / f"{case}_{pair.target}_prompt.png"
                        ).resolve()
                    ),
                }
            )
    write_json(HERE / "grid_index.json", {"grids": rows})


def run(config_path: Path) -> None:
    config = load_config(config_path)
    write_manifest(config)
    prepare_checkpoints(config)
    # Strict stage ordering: finish every single pair before any joint image.
    generate_single_images(config)
    build_single_grids(config)
    generate_joint_images(config)
    build_joint_grids(config)
    write_grid_index()
    print(f"Experiment output: {HERE}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual-only official OCE overlap/cycle experiment"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
