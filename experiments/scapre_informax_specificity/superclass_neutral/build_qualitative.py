#!/usr/bin/env python
"""Build the fixed paired qualitative set without rerunning baseline scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
from torchvision.models import ResNet50_Weights, resnet50


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO = HERE.parents[2]
SCAPRE = REPO / "scapre"
EDITOR = SCAPRE / "edit" / "erase_scale.py"
ROBUSTNESS_CONFIG = PARENT / "seed_robustness" / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def editor_args(
    base: dict,
    assets: dict,
    targets: list[str],
    mode: str,
    output: Path,
    diagnostics: Path,
    matched_config: Path,
) -> list[str]:
    edit = base["edit"]
    args = [
        "--concepts", ", ".join(targets), "--concept_type", edit["concept_type"],
        "--device", "0", "--base", edit["base"],
        "--model-id-or-path", assets["snapshot_path"], "--use_mi_softmask",
        "--erase_scale", str(edit["erase_scale"]), "--p", str(edit["p"]),
        "--bures_iters", str(edit["bures_iters"]), "--enable_ased",
        "--entropy_samples", str(edit["entropy_samples"]),
        "--entropy_bins", str(edit["entropy_bins"]),
        "--noise_sigma", str(edit["noise_sigma"]),
        "--T_sigma", str(edit["T_sigma"]), "--p_sigma", str(edit["p_sigma"]),
        "--informax-negative-mode", mode,
        "--informax-diagnostics-path", str(diagnostics.resolve()),
        "--edit-seed", "20260820", "--output_model", str(output.resolve()),
    ]
    if mode == "matched-retain":
        args.extend(["--informax-matched-retain-config", str(matched_config.resolve())])
    return args


def locate_or_recreate_checkpoints(
    run_dir: Path,
    prior_run: Path,
    base: dict,
    assets: dict,
    targets: list[str],
) -> tuple[dict[str, Path], dict[str, object]]:
    robustness = json.loads(ROBUSTNESS_CONFIG.read_text())
    original_run = PARENT / "runs" / robustness["prior_seed"]["run_id"]
    candidates = [
        original_run / "checkpoints",
        prior_run / "seeds" / "20260820" / "checkpoints",
    ]
    checkpoint_paths: dict[str, Path] = {}
    sources: dict[str, object] = {}
    for variant in ("official", "matched_retain"):
        for directory in candidates:
            candidate = directory / f"{variant}.pt"
            if candidate.is_file():
                checkpoint_paths[variant] = candidate
                sources[variant] = {
                    "source": "reused existing checkpoint",
                    "path": str(candidate.resolve()),
                    "sha256": sha256(candidate),
                }
                break

    missing = [variant for variant in ("official", "matched_retain") if variant not in checkpoint_paths]
    if not missing:
        return checkpoint_paths, sources

    fallback = run_dir / "qualitative" / "recreated_checkpoints"
    fallback.mkdir(parents=True, exist_ok=True)
    matched_config = fallback / "matched_retain_config.json"
    matched_map = {
        target: list(group["retains"])
        for group in base["groups"]
        for target in group["targets"]
    }
    matched_config.write_text(json.dumps({"matched_retain_by_target": matched_map}, indent=2) + "\n")
    for variant in missing:
        output = fallback / f"{variant}.pt"
        diagnostics = fallback / f"{variant}_diagnostics.pt"
        audit = fallback / f"{variant}_rng.json"
        command_manifest = fallback / f"{variant}_command.json"
        mode = "official" if variant == "official" else "matched-retain"
        command = [
            sys.executable, str(EDITOR),
            *editor_args(base, assets, targets, mode, output, diagnostics, matched_config),
        ]
        payload = {"argv": command}
        if command_manifest.exists() and json.loads(command_manifest.read_text()) != payload:
            raise RuntimeError(f"qualitative fallback command changed for {variant}")
        command_manifest.write_text(json.dumps(payload, indent=2) + "\n")
        if not output.is_file():
            run(command, SCAPRE)
        audit_payload = {
            "completed": True,
            "informax_seed": 20260820,
            "mode": "legacy global RNG, matching the original seed-20260820 checkpoint",
            "intercepted_randn_calls": 0,
        }
        audit.write_text(json.dumps(audit_payload, indent=2) + "\n")
        checkpoint_paths[variant] = output
        sources[variant] = {
            "source": "deterministically recreated only for qualitative images",
            "path": str(output.resolve()),
            "sha256": sha256(output),
            "baseline_scores_rerun": False,
            "command_manifest": str(command_manifest.resolve()),
        }
    return checkpoint_paths, sources


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads(args.config.read_text())
    base = json.loads(args.base_config.read_text())
    assets = json.loads(args.assets.read_text())
    qualitative = config["qualitative"]
    output = run_dir / "qualitative"
    completed = output / "COMPLETED"
    if completed.exists():
        print("[resume] qualitative set", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)

    protocol_rows = read_csv(run_dir / "protocol.csv")
    concepts = {
        concept
        for group_concepts in qualitative["concepts_by_group"].values()
        for concept in group_concepts
    }
    indices = {int(value) for value in qualitative["sample_indices"]}
    selected = [
        row for row in protocol_rows
        if row["concept"] in concepts and int(row["sample_index"]) in indices
    ]
    expected_selected = qualitative["expected_images_per_variant"]
    if len(selected) != expected_selected:
        raise RuntimeError(f"qualitative selection has {len(selected)} rows, expected {expected_selected}")
    group_order = list(qualitative["concepts_by_group"])
    concept_order = [concept for group in group_order for concept in qualitative["concepts_by_group"][group]]
    order = {(concept, sample): (concept_order.index(concept), sample) for concept in concepts for sample in indices}
    selected.sort(key=lambda row: order[(row["concept"], int(row["sample_index"]))])

    seed = str(qualitative["edit_seed"])
    score_sources = {
        "official": run_dir / "baselines" / seed / "official" / "scores.csv",
        "matched_retain": run_dir / "baselines" / seed / "matched_retain" / "scores.csv",
        "superclass_neutral": run_dir / "seeds" / seed / "evaluation" / "superclass_neutral" / "scores.csv",
    }
    score_lookup: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    for variant, path in score_sources.items():
        score_lookup[variant] = {
            (row["concept"], int(row["sample_index"])): row for row in read_csv(path)
        }

    targets = [target for group in base["groups"] for target in group["targets"]]
    checkpoints, checkpoint_sources = locate_or_recreate_checkpoints(
        run_dir, args.prior_run.resolve(), base, assets, targets
    )
    checkpoints["superclass_neutral"] = (
        run_dir / "seeds" / seed / "checkpoints" / "superclass_neutral.pt"
    )
    checkpoint_sources["superclass_neutral"] = {
        "source": "new superclass-neutral formal edit",
        "path": str(checkpoints["superclass_neutral"].resolve()),
        "sha256": sha256(checkpoints["superclass_neutral"]),
    }

    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights).to(args.device).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    manifest_rows: list[dict[str, object]] = []
    images_by_key: dict[tuple[str, int], dict[str, Path]] = {}
    for variant in qualitative["variants"]:
        if variant == "superclass_neutral":
            for row in selected:
                key = (row["concept"], int(row["sample_index"]))
                source = Path(score_lookup[variant][key]["image_path"])
                destination = output / "images" / variant / row["group"] / slug(row["concept"]) / f"{key[1]:04d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                images_by_key.setdefault(key, {})[variant] = destination
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                assets["snapshot_path"], torch_dtype=torch.float16, local_files_only=True
            )
            state_dict = torch.load(checkpoints[variant], map_location="cpu")
            incompatible = pipe.unet.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"qualitative checkpoint incompatible for {variant}")
            pipe = pipe.to(args.device)
            pipe.set_progress_bar_config(disable=True)
            for index, row in enumerate(selected, 1):
                key = (row["concept"], int(row["sample_index"]))
                destination = output / "images" / variant / row["group"] / slug(row["concept"]) / f"{key[1]:04d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    generation_seed = int(row["seed"])
                    torch.manual_seed(generation_seed)
                    np.random.seed(generation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(generation_seed)
                    image = pipe(
                        row["prompt"],
                        num_inference_steps=base["evaluation"]["num_inference_steps"],
                        guidance_scale=base["evaluation"]["guidance_scale"],
                        height=base["evaluation"]["height"],
                        width=base["evaluation"]["width"],
                    ).images[0]
                    image.save(destination)
                images_by_key.setdefault(key, {})[variant] = destination
                print(f"[qualitative {variant} {index}/{len(selected)}] {row['concept']}", flush=True)
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for row in selected:
            key = (row["concept"], int(row["sample_index"]))
            score = score_lookup[variant][key]
            image_path = images_by_key[key][variant]
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                prediction_index = classifier(
                    preprocess(image).unsqueeze(0).to(args.device)
                ).argmax(dim=-1).item()
            prediction = categories[prediction_index].lower()
            label = row["concept"].lower()
            correct = str(int(label in prediction or prediction in label))
            if prediction != score["prediction"] or correct != score["correct"]:
                raise RuntimeError(
                    f"qualitative regeneration drift for {variant}/{key}: "
                    f"{prediction}/{correct} != {score['prediction']}/{score['correct']}"
                )
            manifest_rows.append({
                "variant": variant, "group": row["group"], "role": row["role"],
                "concept": row["concept"], "sample_index": row["sample_index"],
                "prompt": row["prompt"], "seed": row["seed"],
                "prediction": score["prediction"], "correct": score["correct"],
                "image_path": str(image_path.relative_to(run_dir)),
                "image_sha256": sha256(image_path),
                "selection_basis": "predeclared concept and ordered sample index",
            })

    comparison_dir = output / "comparisons"
    comparison_dir.mkdir(exist_ok=True)
    variants = qualitative["variants"]
    for row in selected:
        key = (row["concept"], int(row["sample_index"]))
        panels = []
        for variant in variants:
            image = Image.open(images_by_key[key][variant]).convert("RGB")
            canvas = Image.new("RGB", (image.width, image.height + 42), "white")
            canvas.paste(image, (0, 42))
            ImageDraw.Draw(canvas).text((8, 12), variant, fill="black")
            panels.append(canvas)
        comparison = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
        x = 0
        for panel in panels:
            comparison.paste(panel, (x, 0))
            x += panel.width
        destination = comparison_dir / row["group"] / slug(row["concept"]) / f"{key[1]:04d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(destination)

    fields = list(manifest_rows[0])
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    if len(manifest_rows) != qualitative["expected_total_images"]:
        raise RuntimeError("qualitative image manifest count changed")
    if len(list((output / "images").rglob("*.png"))) != qualitative["expected_total_images"]:
        raise RuntimeError("qualitative image file count changed")
    if len(list(comparison_dir.rglob("*.png"))) != expected_selected:
        raise RuntimeError("qualitative comparison panel count changed")

    (output / "provenance.json").write_text(json.dumps({
        "status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": qualitative, "checkpoint_sources": checkpoint_sources,
        "base_model": assets["base_model"], "resolved_revision": assets["resolved_revision"],
        "generation": {
            "protocol_sha256": sha256(run_dir / "protocol.csv"),
            "steps": base["evaluation"]["num_inference_steps"],
            "guidance_scale": base["evaluation"]["guidance_scale"],
            "height": base["evaluation"]["height"], "width": base["evaluation"]["width"],
            "scheduler": base["evaluation"]["sampler"], "dtype": base["evaluation"]["dtype"],
            "safety_checker": "unchanged StableDiffusionPipeline default",
        },
        "baseline_score_evaluation_rerun": False,
        "qualitative_classifier_rechecks": qualitative["expected_total_images"],
        "official_images_regenerated": expected_selected,
        "matched_retain_images_regenerated": expected_selected,
        "superclass_images_copied_from_formal_evaluation": expected_selected,
        "comparison_panels": expected_selected,
    }, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Fixed qualitative comparison set\n\n"
        "This set was selected before viewing outcomes: both targets and one retain per "
        "group, protocol sample indices 0 and 1. Every row uses the identical prompt, "
        "generation seed, sampler, steps, CFG, resolution, base model, and safety checker "
        "across variants. `comparisons/` is ordered left-to-right as official, "
        "matched_retain, superclass_neutral. Predictions are inherited from the unchanged "
        "formal classifier score rows and are listed in `manifest.csv`.\n"
    )
    completed.write_text("ok\n")


if __name__ == "__main__":
    main()
