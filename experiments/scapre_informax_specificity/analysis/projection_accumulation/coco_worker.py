#!/usr/bin/env python3
"""Project-defined SD1.5 COCO general-generation safeguard (never auto-run)."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_fidelity
from diffusers import StableDiffusionPipeline
from PIL import Image
from torch_fidelity.datasets import ImagesPathDataset
from transformers import CLIPModel, CLIPProcessor

from worker import (
    BASE_CONFIG, CONFIG, EDITOR, REPO_ROOT, VARIANTS, cleanup_checkpoint, edit_command,
    git, git_status, run_edit, sha256, validate_configuration,
    validate_edit_isolation, validate_sources, write_json,
)


HERE = Path(__file__).resolve().parent
REFERENCE_INFRA = REPO_ROOT / "orthogonal-concept-erasure" / "experiments" / "evaluation_references"
sys.path.insert(0, str(REFERENCE_INFRA))
from reference_registry import resolve_reference, upsert_reference  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reference_id(count: int) -> str:
    return f"sd15_mscoco30k_first{count}_pndm50_cfg7p5_512_fp16"


def identity(config: dict[str, Any], assets: dict[str, Any], count: int) -> dict[str, Any]:
    coco = config["coco"]
    return {
        "model_id": assets["base_model"],
        "resolved_revision": assets["resolved_revision"],
        "prompt_source_sha256": coco["prompt_source_sha256"],
        "prompt_subset": f"first {count} rows in source order",
        "prompt_count": count,
        "seed_column": "evaluation_seed",
        "num_inference_steps": coco["num_inference_steps"],
        "guidance_scale": coco["guidance_scale"],
        "height": coco["height"], "width": coco["width"],
        "dtype": coco["dtype"], "scheduler": coco["scheduler"],
        "clip_model_id": coco["clip_model_id"],
        "clip_implementation": coco["clip_implementation"],
        "fid_implementation": coco["fid_implementation"],
        "fid_feature_extractor": coco["fid_feature_extractor"],
        "fid_feature_layer": coco["fid_feature_layer"],
        "protocol_label": coco["label"],
    }


def reference_paths(count: int) -> dict[str, Path]:
    root = REFERENCE_INFRA / "references" / reference_id(count)
    return {
        "root": root,
        "clip": root / "clip_baseline.json",
        "prompts": root / f"prompts_first{count}.csv",
        "protocol": root / "protocol.json",
        "cache": root / "torch_fidelity_cache",
    }


def artifacts(count: int) -> dict[str, str]:
    paths = reference_paths(count)
    return {
        "clip_baseline": str(paths["clip"]),
        "fid_statistics_glob": str(paths["cache"] / f"{reference_id(count)}-*-stat-fid-2048.pt"),
        "prompt_manifest": str(paths["prompts"]),
        "protocol_manifest": str(paths["protocol"]),
    }


def image_path(run_dir: Path, method: str, case_number: int) -> Path:
    return run_dir / "images" / method / f"{case_number:06d}.png"


def load_pipe(assets: dict[str, Any]) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        assets["snapshot_path"], torch_dtype=torch.float16,
        safety_checker=None, local_files_only=True,
    ).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    if pipe.scheduler.__class__.__name__ != "PNDMScheduler":
        raise RuntimeError("COCO scheduler is not PNDMScheduler")
    return pipe


def apply_checkpoint(pipe: StableDiffusionPipeline, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    incompatible = pipe.unet.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint incompatibility: {incompatible}")


@torch.inference_mode()
def generate(
    pipe: StableDiffusionPipeline, frame: pd.DataFrame, run_dir: Path,
    method: str, config: dict[str, Any], batch_size: int,
) -> None:
    coco = config["coco"]
    destination = run_dir / "images" / method
    destination.mkdir(parents=True, exist_ok=True)
    pending = frame[[
        not image_path(run_dir, method, int(case)).is_file()
        for case in frame["case_number"]
    ]]
    for offset in range(0, len(pending), batch_size):
        batch = pending.iloc[offset:offset + batch_size]
        images = pipe(
            prompt=batch["prompt"].astype(str).tolist(),
            num_inference_steps=coco["num_inference_steps"],
            guidance_scale=coco["guidance_scale"],
            height=coco["height"], width=coco["width"],
            generator=[torch.Generator(device="cuda:0").manual_seed(int(seed)) for seed in batch["evaluation_seed"]],
        ).images
        for case, image in zip(batch["case_number"], images):
            image.save(image_path(run_dir, method, int(case)))
        write_json(run_dir / "status.json", {
            "stage": f"generate_{method}", "mode": len(frame),
            "completed_images": {
                name: len(list((run_dir / "images" / name).glob("*.png")))
                for name in ("original_sd15", "official", "projection_accumulation")
            },
            "updated_at_utc": utc_now(),
        })


@torch.inference_mode()
def clip_score(frame: pd.DataFrame, run_dir: Path, method: str, model_id: str, batch_size: int) -> dict[str, float]:
    model = CLIPModel.from_pretrained(model_id, local_files_only=True).eval().to("cuda:0")
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    values: list[float] = []
    for offset in range(0, len(frame), batch_size):
        batch = frame.iloc[offset:offset + batch_size]
        images = []
        for case in batch["case_number"]:
            with Image.open(image_path(run_dir, method, int(case))) as image:
                images.append(image.convert("RGB"))
        inputs = processor(
            text=batch["prompt"].astype(str).tolist(), images=images,
            return_tensors="pt", padding=True, truncation=True,
        )
        outputs = model(**{key: value.to("cuda:0") for key, value in inputs.items()})
        values.extend(float(value) for value in outputs.logits_per_image.diagonal().cpu())
    array = np.asarray(values, dtype=np.float64)
    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    return {"mean": float(array.mean()), "std": float(array.std()), "count": int(array.size)}


def fid_to_original(frame: pd.DataFrame, run_dir: Path, method: str, count: int) -> float:
    edited = ImagesPathDataset([str(image_path(run_dir, method, int(case))) for case in frame["case_number"]])
    original = ImagesPathDataset([str(image_path(run_dir, "original_sd15", int(case))) for case in frame["case_number"]])
    paths = reference_paths(count)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    result = torch_fidelity.calculate_metrics(
        input1=edited, input2=original, input2_cache_name=reference_id(count),
        cache_root=str(paths["cache"]), cuda=True, fid=True, verbose=False,
    )
    return float(result["frechet_inception_distance"])


def validate_images(
    frame: pd.DataFrame, run_dir: Path, methods: list[str], output: Path
) -> dict[str, Any]:
    expected = [int(value) for value in frame["case_number"]]
    expected_set = set(expected)
    seed_by_case = {
        int(row.case_number): int(row.evaluation_seed)
        for row in frame.itertuples(index=False)
    }
    if len(expected) != len(expected_set):
        raise RuntimeError("COCO protocol contains duplicate case numbers")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    per_method: dict[str, Any] = {}
    for method in methods:
        directory = run_dir / "images" / method
        observed_paths = sorted(directory.glob("*.png"))
        observed: dict[int, Path] = {}
        for path in observed_paths:
            try:
                case_number = int(path.stem)
            except ValueError as error:
                raise RuntimeError(f"unexpected COCO image filename: {path}") from error
            if case_number in observed:
                raise RuntimeError(f"duplicate COCO image key: {method}/{case_number}")
            observed[case_number] = path
        if set(observed) != expected_set:
            missing = sorted(expected_set - set(observed))
            extra = sorted(set(observed) - expected_set)
            raise RuntimeError(
                f"COCO image keys mismatch for {method}: missing={missing[:10]}, extra={extra[:10]}"
            )
        total_bytes = 0
        for case_number in expected:
            path = observed[case_number]
            with Image.open(path) as image:
                image.verify()
            size = path.stat().st_size
            if size <= 0:
                raise RuntimeError(f"empty COCO image: {path}")
            total_bytes += size
            rows.append({
                "method": method, "case_number": case_number,
                "evaluation_seed": seed_by_case[case_number],
                "image_path": str(path.resolve()), "size_bytes": size,
                "sha256": sha256(path),
            })
        per_method[method] = {
            "expected": len(expected), "observed": len(observed),
            "missing": 0, "extra": 0, "duplicate_keys": 0,
            "total_bytes": total_bytes,
        }
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "case_number", "evaluation_seed", "image_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "passed", "methods": per_method, "manifest": str(output)}


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    official = metrics["methods"]["official"]
    projection = metrics["methods"]["projection_accumulation"]
    delta = metrics["projection_minus_official"]
    fid_note = (
        "First-1k FID is descriptive screening only and is not a strong final-quality claim."
        if metrics["prompt_count"] == 1000 else
        "First-10k is the manually launched, more formal general-generation safeguard."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "# Projection accumulation — COCO safeguard", "",
        "Protocol: project-defined secondary general-generation safeguard; not a ScaPre paper COCO reproduction.",
        f"Mode: {metrics['mode']} ({metrics['prompt_count']} ordered prompts).", "",
        "| Method | CLIP mean ↑ | FID to Original SD1.5 ↓ |",
        "| --- | ---: | ---: |",
        f"| official | {official['clip']['mean']:.6f} | {official['fid_to_original_sd15']:.6f} |",
        f"| projection_accumulation | {projection['clip']['mean']:.6f} | {projection['fid_to_original_sd15']:.6f} |",
        f"| projection - official | {delta['clip_mean']:+.6f} | {delta['fid_to_original_sd15']:+.6f} |",
        "", fid_note, "",
    ]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["first-1k", "first-10k"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "MU" or not torch.cuda.is_available():
        raise RuntimeError("COCO worker requires active MU and CUDA")
    if git_status():
        raise RuntimeError(f"COCO launch requires a clean worktree: {git_status()}")
    run_dir = args.run_dir.resolve()
    config = json.loads(CONFIG.read_text())
    base = json.loads(BASE_CONFIG.read_text())
    assets = json.loads(args.assets.read_text())
    validate_configuration(config, base); validate_sources(config)
    production_hash_start = sha256(EDITOR)
    count = 1000 if args.mode == "first-1k" else 10000
    prompt_source = REPO_ROOT / config["coco"]["prompt_source"]
    if sha256(prompt_source) != config["coco"]["prompt_source_sha256"]:
        raise RuntimeError("COCO prompt source hash mismatch")
    frame = pd.read_csv(prompt_source).iloc[:count].copy()
    if len(frame) != count or frame["case_number"].duplicated().any():
        raise RuntimeError("COCO ordered subset is incomplete or duplicated")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "reproducibility" / "protocol.json"
    write_json(protocol_path, {
        "label": config["coco"]["label"], "paper_reproduction": False,
        "identity": identity(config, assets, count),
        "prompt_source": str(prompt_source), "prompt_source_sha256": sha256(prompt_source),
    })
    frame.to_csv(run_dir / "reproducibility" / f"prompts_first{count}.csv", index=False)
    manifest_path = run_dir / "reproducibility" / "run_manifest.json"
    manifest = {
        "started_at_utc": utc_now(), "mode": args.mode,
        "git_commit": git("rev-parse", "HEAD"), "git_status_start": [],
        "python_executable": sys.executable, "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "assets": assets, "assets_sha256": sha256(args.assets),
        "config_sha256": sha256(CONFIG), "protocol_sha256": sha256(protocol_path),
        "prompt_manifest_sha256": sha256(run_dir / "reproducibility" / f"prompts_first{count}.csv"),
        "production_editor_sha256_start": production_hash_start,
    }
    write_json(manifest_path, manifest)

    seed_dir = run_dir / "edit_seed_20260820"
    for name in ("checkpoints", "diagnostics", "audits", "stages", "evaluation"):
        (seed_dir / name).mkdir(parents=True, exist_ok=True)
    commands = {variant: edit_command(config, base, assets, 20260820, seed_dir, variant) for variant in VARIANTS}
    for variant in VARIANTS:
        run_edit(commands[variant], seed_dir, variant)
    isolation = validate_edit_isolation(seed_dir, commands)
    write_json(run_dir / "reproducibility" / "edit_isolation.json", isolation)

    expected_identity = identity(config, assets, count)
    reference = resolve_reference(reference_id(count), expected_identity, require_complete=True)
    build_reference = reference is None
    if build_reference:
        paths = reference_paths(count)
        paths["root"].mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths["prompts"], index=False)
        write_json(paths["protocol"], {
            "reference_id": reference_id(count), "identity": expected_identity,
            "source_prompt_csv": str(prompt_source), "source_prompt_csv_sha256": sha256(prompt_source),
        })
        upsert_reference(reference_id(count), expected_identity, "building", artifacts(count), str(Path(__file__).resolve()))

    methods = ["original_sd15", "official", "projection_accumulation"] if build_reference else ["official", "projection_accumulation"]
    checkpoints = {
        "official": seed_dir / "checkpoints" / "official.pt",
        "projection_accumulation": seed_dir / "checkpoints" / "projection_accumulation.pt",
    }
    for method in methods:
        pipe = load_pipe(assets)
        if method != "original_sd15":
            apply_checkpoint(pipe, checkpoints[method])
        generate(pipe, frame, run_dir, method, config, args.batch_size)
        pipe.to("cpu"); del pipe; gc.collect(); torch.cuda.empty_cache()

    image_validation = validate_images(
        frame, run_dir, methods,
        run_dir / "reproducibility" / "generated_image_manifest.csv",
    )

    metrics: dict[str, Any] = {
        "protocol_label": config["coco"]["label"], "paper_reproduction": False,
        "mode": args.mode, "prompt_count": count,
        "first_1k_fid_interpretation": config["coco"]["first_1k_fid_interpretation"] if count == 1000 else None,
        "methods": {},
    }
    if build_reference:
        original_clip = clip_score(frame, run_dir, "original_sd15", config["coco"]["clip_model_id"], args.batch_size)
        write_json(reference_paths(count)["clip"], {
            "reference_id": reference_id(count), "reference_identity": expected_identity,
            "clip_score": original_clip, "created_at_utc": utc_now(),
        })
        metrics["methods"]["original_sd15"] = {"clip": original_clip, "fid_to_original_sd15": 0.0}
    else:
        original_clip_payload = json.loads(Path(reference["artifacts"]["clip_baseline"]).read_text())
        metrics["methods"]["original_sd15"] = {"clip": original_clip_payload["clip_score"], "fid_to_original_sd15": 0.0}

    for method in ("official", "projection_accumulation"):
        method_clip = clip_score(frame, run_dir, method, config["coco"]["clip_model_id"], args.batch_size)
        method_fid = fid_to_original(frame, run_dir, method, count)
        metrics["methods"][method] = {"clip": method_clip, "fid_to_original_sd15": method_fid}
    if build_reference:
        stats = list(reference_paths(count)["cache"].glob(f"{reference_id(count)}-*-stat-fid-2048.pt"))
        if len(stats) != 1:
            raise RuntimeError("SD1.5 reference FID statistics were not created exactly once")
        upsert_reference(reference_id(count), expected_identity, "complete", artifacts(count), str(Path(__file__).resolve()))
    metrics["projection_minus_official"] = {
        "clip_mean": metrics["methods"]["projection_accumulation"]["clip"]["mean"] - metrics["methods"]["official"]["clip"]["mean"],
        "fid_to_original_sd15": metrics["methods"]["projection_accumulation"]["fid_to_original_sd15"] - metrics["methods"]["official"]["fid_to_original_sd15"],
    }
    write_json(run_dir / "results" / "metrics.json", metrics)
    write_summary(run_dir / "results" / "summary.md", metrics)
    for variant in VARIANTS:
        cleanup_checkpoint(seed_dir, variant)
    if sha256(EDITOR) != production_hash_start:
        raise RuntimeError("production editor changed during COCO safeguard")
    resolved = resolve_reference(reference_id(count), expected_identity, require_complete=True)
    if resolved is None:
        raise RuntimeError("SD1.5 reference did not resolve after registration")
    write_json(run_dir / "results" / "integrity_report.json", {
        "status": "passed", "protocol_label": config["coco"]["label"],
        "paper_reproduction": False, "mode": args.mode, "prompt_count": count,
        "same_prompts_and_generation_seeds": True, "edit_seed": 20260820,
        "production_editor_byte_unchanged": True,
        "production_editor_sha256": production_hash_start,
        "image_validation": image_validation, "edit_isolation": isolation,
        "reference_id": reference_id(count), "reference_was_built": build_reference,
        "reference_fingerprint": resolved["fingerprint"],
        "reference_registry_entry": resolved,
        "sd14_reference_reused": False,
        "first_1k_fid_is_descriptive_only": count == 1000,
    })
    manifest["production_editor_sha256_end"] = sha256(EDITOR)
    manifest["finished_at_utc"] = utc_now()
    manifest["registry_status_at_completion"] = resolved["status"]
    write_json(manifest_path, manifest)
    write_json(run_dir / "worker_complete.json", {"status": "passed", "completed_at_utc": utc_now()})


if __name__ == "__main__":
    main()
