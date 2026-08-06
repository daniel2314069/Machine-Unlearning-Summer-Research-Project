from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
CORRESPONDENCE_DIR = EXPERIMENT_ROOT.parent / "correspondence_diagnostic"
sys.path.insert(0, str(CORRESPONDENCE_DIR))

from oce_correspondence.core import (  # noqa: E402
    PairSpec,
    apply_weight_state,
    edit_projection_weights,
)
from oce_correspondence.io_utils import make_seed_grid  # noqa: E402
from oce_correspondence.runner import (  # noqa: E402
    _dtype,
    _encode_last_content_tokens,
)
from run_mixed_joint_smoke import (  # noqa: E402
    BASE_PAIRS,
    DEFAULT_CONFIG,
    OFFICIAL_OUTPUT,
    art_expansion,
    image_path,
    load_config,
    object_expansion,
    slug,
)


SEEDS = list(range(42, 52))


def single_image_path(pair: PairSpec, seed: int) -> Path:
    if pair in BASE_PAIRS[:3]:
        return (
            OFFICIAL_OUTPUT
            / "images"
            / "single_pair"
            / pair.slug
            / f"single_subspace_{pair.slug}"
            / f"seed_{seed}.png"
        )
    return (
        HERE
        / "images"
        / slug(pair.target)
        / "single_subspace"
        / f"seed_{seed}.png"
    )


def comparison_grid_path(pair: PairSpec) -> Path:
    return HERE / "single_vs_joint" / f"{slug(pair.target)}.png"


def expanded_single_pair(pair: PairSpec) -> list[PairSpec]:
    if pair.target in {"cat", "truck", "church"}:
        return object_expansion(pair)
    if pair.target == "Van Gogh":
        return art_expansion(pair)
    return [pair]


@torch.inference_mode()
def build_missing_single_states(
    config: Mapping[str, object],
    pairs: Sequence[PairSpec],
) -> dict[str, dict[str, torch.Tensor]]:
    from diffusers import DiffusionPipeline

    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=_dtype(str(config["edit_dtype"])),
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)

    expanded = {
        pair.target: expanded_single_pair(pair)
        for pair in pairs
    }
    all_text = [
        value
        for pair in pairs
        for edit_pair in expanded[pair.target]
        for value in (edit_pair.target, edit_pair.anchor)
    ]
    all_text.extend(pair.anchor for pair in pairs)
    embeddings = _encode_last_content_tokens(pipe, all_text, device)
    cg = torch.load(
        str(config["cg_path"]), map_location=device
    )["C"].float()
    oce = config["oce"]

    states: dict[str, dict[str, torch.Tensor]] = {}
    for pair in pairs:
        state, audit = edit_projection_weights(
            unet=pipe.unet,
            embeddings=embeddings,
            pairs=expanded[pair.target],
            preserve_concepts=[pair.anchor],
            global_second_moment=cg,
            objective="subspace",
            erase_scale=float(oce["erase_scale"]),
            preserve_global_scale=float(oce["preserve_global_scale"]),
            preserve_concept_scale=float(oce["preserve_concept_scale"]),
            lamb=float(oce["lambda"]),
            reflection_correction=str(oce["reflection_correction"]),
        )
        if len(audit) != 16:
            raise RuntimeError(
                f"Expected 16 edited layers for {pair.target}, got {len(audit)}"
            )
        states[pair.target] = state

    del pipe, embeddings, cg
    gc.collect()
    torch.cuda.empty_cache()
    return states


@torch.inference_mode()
def generate_missing_single_images(
    config: Mapping[str, object],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    pairs: Sequence[PairSpec],
) -> None:
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

    for pair in pairs:
        apply_weight_state(pipe.unet, states[pair.target])
        for seed in SEEDS:
            destination = single_image_path(pair, seed)
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            image = pipe(
                prompt=pair.prompt,
                num_inference_steps=int(generation["num_inference_steps"]),
                guidance_scale=float(generation["guidance_scale"]),
                height=int(generation["height"]),
                width=int(generation["width"]),
                generator=torch.Generator(device=device).manual_seed(seed),
            ).images[0]
            image.save(destination)
            print(
                f"[image] {pair.target} single_subspace seed={seed}",
                flush=True,
            )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def build_comparison_grids() -> None:
    for pair in BASE_PAIRS:
        make_seed_grid(
            comparison_grid_path(pair),
            {
                "Original SD": [
                    image_path(pair, "original", seed)
                    for seed in SEEDS
                ],
                "Single Subspace": [
                    single_image_path(pair, seed)
                    for seed in SEEDS
                ],
                "Five-concept Joint Subspace": [
                    image_path(pair, "joint_subspace", seed)
                    for seed in SEEDS
                ],
            },
            SEEDS,
            f"Single vs joint subspace | {pair.target} -> {pair.anchor}",
        )


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    pairs_with_missing_images = [
        pair
        for pair in BASE_PAIRS
        if any(
            not single_image_path(pair, seed).exists()
            for seed in SEEDS
        )
    ]
    if pairs_with_missing_images:
        states = build_missing_single_states(
            config, pairs_with_missing_images
        )
        generate_missing_single_images(
            config, states, pairs_with_missing_images
        )
    build_comparison_grids()
    print(f"Output: {HERE / 'single_vs_joint'}", flush=True)


if __name__ == "__main__":
    main()
