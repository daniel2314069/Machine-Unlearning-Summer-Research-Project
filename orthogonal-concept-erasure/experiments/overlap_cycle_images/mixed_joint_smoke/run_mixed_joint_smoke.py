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
EXPERIMENT_ROOT = HERE.parent
CORRESPONDENCE_DIR = EXPERIMENT_ROOT.parent / "correspondence_diagnostic"
sys.path.insert(0, str(CORRESPONDENCE_DIR))

from oce_correspondence.core import (  # noqa: E402
    PairSpec,
    apply_weight_state,
    clone_projection_state,
    edit_projection_weights,
)
from oce_correspondence.io_utils import make_seed_grid, sha256, write_json  # noqa: E402
from oce_correspondence.runner import _dtype, _encode_last_content_tokens  # noqa: E402


DEFAULT_CONFIG = CORRESPONDENCE_DIR / "config_official_subspace.json"
OFFICIAL_OUTPUT = CORRESPONDENCE_DIR / "outputs" / "official_subspace"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

BASE_PAIRS = [
    PairSpec("cat", "dog", "a photo of a cat"),
    PairSpec("truck", "car", "a photo of a truck"),
    PairSpec("church", "castle", "a photo of a church"),
    PairSpec(
        "Van Gogh",
        "art",
        "Van Gogh style painting of the night sky with bold strokes.",
    ),
    PairSpec("Adam Driver", "celebrity", "a portrait of Adam Driver"),
]


def slug(value: str) -> str:
    return "_".join(value.casefold().replace("-", " ").split())


def checkpoint_path() -> Path:
    return HERE / "joint_official_subspace.safetensors"


def image_path(pair: PairSpec, method: str, seed: int) -> Path:
    return (
        HERE
        / "images"
        / slug(pair.target)
        / method
        / f"seed_{seed}.png"
    )


def grid_path(pair: PairSpec) -> Path:
    return HERE / f"{slug(pair.target)}_mixed_joint_smoke.png"


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["cg_path"] = str(
        (path.parent / str(config["cg_path"])).resolve()
    )
    return config


def object_expansion(pair: PairSpec) -> list[PairSpec]:
    templates = [
        "image of {concept}",
        "photo of {concept}",
        "portrait of {concept}",
        "picture of {concept}",
        "painting of {concept}",
    ]
    return [pair] + [
        PairSpec(
            template.format(concept=pair.target),
            template.format(concept=pair.anchor),
            pair.prompt,
        )
        for template in templates
    ]


def art_expansion(pair: PairSpec) -> list[PairSpec]:
    # Exact concept_type=art expansion used by repository oce.py.
    templates = [
        "painting by {concept}",
        "art by {concept}",
        "artwork by {concept}",
        "picture by {concept}",
        "style of {concept}",
    ]
    return [pair] + [
        PairSpec(
            template.format(concept=pair.target),
            template.format(concept=pair.anchor),
            pair.prompt,
        )
        for template in templates
    ]


def expanded_edit_pairs() -> list[PairSpec]:
    result: list[PairSpec] = []
    for pair in BASE_PAIRS:
        if pair.target in {"cat", "truck", "church"}:
            result.extend(object_expansion(pair))
        elif pair.target == "Van Gogh":
            result.extend(art_expansion(pair))
        else:
            # Celebrity benchmark training scripts leave expand_prompts off.
            result.append(pair)
    return result


def validate_sets() -> dict[str, object]:
    targets = {pair.target.casefold() for pair in BASE_PAIRS}
    anchors = {pair.anchor.casefold() for pair in BASE_PAIRS}
    overlap = sorted(targets.intersection(anchors))
    if overlap:
        raise ValueError(f"Target/anchor overlap is forbidden: {overlap}")
    return {
        "targets": [pair.target for pair in BASE_PAIRS],
        "anchors": [pair.anchor for pair in BASE_PAIRS],
        "intersection": [],
        "disjoint": True,
    }


@torch.inference_mode()
def prepare_checkpoint(config: Mapping[str, object]) -> None:
    from diffusers import DiffusionPipeline

    if checkpoint_path().exists():
        return
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=_dtype(str(config["edit_dtype"])),
        safety_checker=None,
        vae=None,
        local_files_only=True,
    ).to(device)
    edit_pairs = expanded_edit_pairs()
    all_text = [
        value
        for pair in edit_pairs
        for value in (pair.target, pair.anchor)
    ]
    all_text.extend(pair.anchor for pair in BASE_PAIRS)
    embeddings = _encode_last_content_tokens(pipe, all_text, device)
    cg = torch.load(
        str(config["cg_path"]), map_location=device
    )["C"].float()
    oce = config["oce"]
    edited, audit = edit_projection_weights(
        unet=pipe.unet,
        embeddings=embeddings,
        pairs=edit_pairs,
        preserve_concepts=[pair.anchor for pair in BASE_PAIRS],
        global_second_moment=cg,
        objective="subspace",
        erase_scale=float(oce["erase_scale"]),
        preserve_global_scale=float(oce["preserve_global_scale"]),
        preserve_concept_scale=float(oce["preserve_concept_scale"]),
        lamb=float(oce["lambda"]),
        reflection_correction=str(oce["reflection_correction"]),
    )
    if len(audit) != 16:
        raise RuntimeError(f"Expected 16 edited layers, got {len(audit)}")
    save_file(edited, str(checkpoint_path()))
    print(f"[checkpoint] {checkpoint_path()}", flush=True)
    del pipe, cg
    gc.collect()
    torch.cuda.empty_cache()


def copy_reusable_originals() -> None:
    for pair in BASE_PAIRS[:3]:
        source = (
            OFFICIAL_OUTPUT
            / "images"
            / "feasibility"
            / pair.target
        )
        for seed in SEEDS:
            destination = image_path(pair, "original", seed)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source / f"seed_{seed}.png", destination)


@torch.inference_mode()
def generate_images(config: Mapping[str, object]) -> None:
    from diffusers import DiffusionPipeline

    copy_reusable_originals()
    device = torch.device(str(config["device"]))
    pipe = DiffusionPipeline.from_pretrained(
        str(config["model_id"]),
        torch_dtype=_dtype(str(config["generation_dtype"])),
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    original_state = clone_projection_state(pipe.unet)
    edited_state = load_file(str(checkpoint_path()))
    generation = config["generation"]
    for method, state in (
        ("original", original_state),
        ("joint_subspace", edited_state),
    ):
        apply_weight_state(pipe.unet, state)
        for pair in BASE_PAIRS:
            for index, seed in enumerate(SEEDS, 1):
                destination = image_path(pair, method, seed)
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
                    f"[image] {pair.target} {method} "
                    f"{index}/{len(SEEDS)} seed={seed}",
                    flush=True,
                )
    del pipe, edited_state
    gc.collect()
    torch.cuda.empty_cache()


def build_grids() -> None:
    for pair in BASE_PAIRS:
        make_seed_grid(
            grid_path(pair),
            {
                "Original SD": [
                    image_path(pair, "original", seed)
                    for seed in SEEDS
                ],
                "Joint Subspace": [
                    image_path(pair, "joint_subspace", seed)
                    for seed in SEEDS
                ],
            },
            SEEDS,
            f"Mixed heterogeneous joint smoke | {pair.target} -> {pair.anchor}",
        )


def write_manifest(config: Mapping[str, object]) -> None:
    set_check = validate_sets()
    edit_pairs = expanded_edit_pairs()
    write_json(
        HERE / "resolved_mixed_joint_smoke.json",
        {
            "pairs": [
                {
                    "target": pair.target,
                    "anchor": pair.anchor,
                    "generation_prompt": pair.prompt,
                }
                for pair in BASE_PAIRS
            ],
            "set_validation": set_check,
            "celebrity_source": {
                "target": "Adam Driver",
                "reason": (
                    "First target in repository E10/E50/E100 celebrity "
                    "benchmark lists."
                ),
                "prompt_template_source": "generate_celeb.py: a portrait of {}",
            },
            "artist_source": {
                "target": "Van Gogh",
                "standard_string_source": "trainscripts/style.sh",
                "generation_prompt_source": (
                    "first template in generate_style.py"
                ),
            },
            "edit_prompt_rows": [
                {"target": pair.target, "anchor": pair.anchor}
                for pair in edit_pairs
            ],
            "seeds": SEEDS,
            "model_id": config["model_id"],
            "generation": config["generation"],
            "oce": config["oce"],
            "edited_layers": (
                "all 16 module names containing attn2 and ending in to_v"
            ),
            "scope": (
                "Original SD vs joint official OCE subspace images only; "
                "no numerical evaluation"
            ),
        },
    )


def write_grid_index() -> None:
    write_json(
        HERE / "grid_index.json",
        {
            "grids": [
                {
                    "target": pair.target,
                    "anchor": pair.anchor,
                    "prompt": pair.prompt,
                    "path": str(grid_path(pair).resolve()),
                }
                for pair in BASE_PAIRS
            ]
        },
    )


def run(config_path: Path) -> None:
    config = load_config(config_path)
    write_manifest(config)
    prepare_checkpoint(config)
    generate_images(config)
    build_grids()
    write_grid_index()
    write_json(
        HERE / "checkpoint_manifest.json",
        {
            "path": str(checkpoint_path().resolve()),
            "sha256": sha256(checkpoint_path()),
            "objective": "official OCE subspace",
        },
    )
    print(f"Output: {HERE}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed heterogeneous joint official-subspace smoke"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
