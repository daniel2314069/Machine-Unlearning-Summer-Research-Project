#!/usr/bin/env python3
"""
Run a small ScaPre ASED ablation for religious-building erasure.

The experiment compares:
  - original SD 1.5
  - ScaPre without ASED R
  - ScaPre with ASED R = U diag(tilde_sigma) U^T

It is designed for low-storage runs: each model/group image batch is scored,
sampled into a small kept set, then deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_MODEL = "runwayml/stable-diffusion-v1-5"
EDIT_CONCEPTS = [
    "church",
    "cathedral",
    "chapel",
    "basilica",
    "mosque",
    "synagogue",
    "temple",
    "monastery",
    "shrine",
    "pagoda",
]

PROMPT_GROUPS = {
    "target_religious_buildings": [
        "church",
        "cathedral",
        "chapel",
        "basilica",
        "mosque",
        "synagogue",
        "temple",
        "monastery",
        "shrine",
        "pagoda",
    ],
    "near_religious_people_objects": [
        "priest",
        "monk",
        "nun",
        "imam",
        "rabbi",
        "bible",
        "quran",
        "rosary",
        "prayer",
        "altar",
        "candle",
        "worship ceremony",
    ],
    "near_generic_buildings": [
        "house",
        "school",
        "hospital",
        "library",
        "museum",
        "airport",
        "train station",
        "stadium",
        "skyscraper",
        "bridge",
    ],
    "far_unrelated_objects": [
        "dog",
        "car",
        "banana",
        "chair",
        "guitar",
        "bicycle",
        "laptop",
        "pizza",
        "flower",
        "camera",
    ],
}

PROMPT_TEMPLATES = [
    "a realistic photo of {concept}",
    "a detailed image of {concept}",
]

GROUP_LABELS = {
    "target_religious_buildings": "a photo of a religious building",
    "near_religious_people_objects": "a photo of a religious person or religious object",
    "near_generic_buildings": "a photo of a generic building",
    "far_unrelated_objects": "a photo of an unrelated everyday object",
}

MODEL_ORDER = ["original_sd15", "scapre_no_R", "scapre_with_R"]


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    group: str
    concept: str
    prompt: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "edit", "eval", "report"], default="all")
    parser.add_argument("--work-dir", default="reports/scapre_ased_religious_buildings")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-images-per-group", type=int, default=50)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--keep-samples", type=int, default=12)
    parser.add_argument("--seed-offset", type=int, default=12340)
    parser.add_argument("--force-edit", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--run-fid", action="store_true", help="compute optional drift FID before deleting temp images")
    parser.add_argument("--skip-delete", action="store_true", help="debug only: keep full generated image batches")
    return parser.parse_args()


def scapre_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(work_dir: Path) -> dict[str, Path]:
    dirs = {
        "checkpoints": work_dir / "checkpoints",
        "tmp_images": work_dir / "tmp_images",
        "kept_samples": work_dir / "kept_samples",
        "grids": work_dir / "grids",
        "metrics": work_dir / "metrics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def device_index(device: str) -> str:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    return device


def checkpoint_paths(work_dir: Path) -> dict[str, Path | None]:
    ckpt_dir = work_dir / "checkpoints"
    return {
        "original_sd15": None,
        "scapre_no_R": ckpt_dir / "scapre_no_R.pt",
        "scapre_with_R": ckpt_dir / "scapre_with_R.pt",
    }


def run_edit(args: argparse.Namespace, work_dir: Path) -> None:
    paths = checkpoint_paths(work_dir)
    root = scapre_root()
    common = [
        sys.executable,
        "edit/erase_scale.py",
        "--concepts",
        ", ".join(EDIT_CONCEPTS),
        "--concept_type",
        "object",
        "--device",
        device_index(args.device),
        "--base",
        "1.5",
        "--use_mi_softmask",
        "--erase_scale",
        "2",
        "--p",
        "8",
        "--bures_iters",
        "1",
        "--entropy_samples",
        "30",
        "--entropy_bins",
        "20",
    ]

    runs = [
        ("scapre_no_R", []),
        ("scapre_with_R", ["--enable_ased", "--T_sigma", "1", "--p_sigma", "1"]),
    ]
    for model_name, extra in runs:
        out_path = paths[model_name]
        assert out_path is not None
        if out_path.exists() and not args.force_edit:
            print(f"[edit] skip existing {out_path}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = common + extra + ["--output_model", str(out_path.resolve())]
        print("[edit] " + " ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True)


def build_samples(max_images_per_group: int, seed_offset: int) -> dict[str, list[SampleSpec]]:
    all_samples: dict[str, list[SampleSpec]] = {}
    for group_idx, (group, concepts) in enumerate(PROMPT_GROUPS.items()):
        prompt_specs: list[tuple[str, str]] = []
        for concept in concepts:
            for template in PROMPT_TEMPLATES:
                prompt_specs.append((concept, template.format(concept=concept)))
        samples: list[SampleSpec] = []
        rep = 0
        while len(samples) < max_images_per_group:
            for local_idx, (concept, prompt) in enumerate(prompt_specs):
                if len(samples) >= max_images_per_group:
                    break
                sample_idx = len(samples)
                seed = seed_offset + group_idx * 100000 + rep * 1000 + local_idx
                samples.append(
                    SampleSpec(
                        sample_id=f"{group}_{sample_idx:04d}",
                        group=group,
                        concept=concept,
                        prompt=prompt,
                        seed=seed,
                    )
                )
            rep += 1
        all_samples[group] = samples
    return all_samples


def load_pipe(args: argparse.Namespace, ckpt_path: Path | None) -> StableDiffusionPipeline:
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        pipe.unet.load_state_dict(state_dict, strict=False)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def load_clip(device: str) -> tuple[CLIPModel, CLIPProcessor]:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def normalized_text_features(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    texts: list[str],
    device: str,
) -> torch.Tensor:
    import torch

    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def normalized_image_features(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    images: list[Image.Image],
    device: str,
) -> torch.Tensor:
    import torch

    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def score_image(
    image: Image.Image,
    sample: SampleSpec,
    model_name: str,
    image_path: Path,
    image_feat: torch.Tensor,
    prompt_text_feats: dict[str, torch.Tensor],
    group_texts: list[str],
    group_feats: torch.Tensor,
    concept_texts: list[str],
    concept_feats: torch.Tensor,
    original_feats: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    group_scores = (image_feat @ group_feats.T).squeeze(0)
    concept_scores = (image_feat @ concept_feats.T).squeeze(0)
    prompt_score = (image_feat @ prompt_text_feats[sample.prompt].T).item()
    group_idx = int(group_scores.argmax().item())
    concept_idx = int(concept_scores.argmax().item())
    drift_clip_cosine = None
    if original_feats is not None and sample.sample_id in original_feats:
        drift_clip_cosine = float((image_feat.cpu() @ original_feats[sample.sample_id].T).item())
    return {
        "model": model_name,
        "group": sample.group,
        "sample_id": sample.sample_id,
        "concept": sample.concept,
        "prompt": sample.prompt,
        "seed": sample.seed,
        "image_path": str(image_path),
        "expected_group": sample.group,
        "pred_group": list(GROUP_LABELS.keys())[group_idx],
        "expected_group_prob_proxy": float(group_scores[list(GROUP_LABELS.keys()).index(sample.group)].item()),
        "pred_group_score": float(group_scores[group_idx].item()),
        "group_top1_hit": int(list(GROUP_LABELS.keys())[group_idx] == sample.group),
        "pred_concept": concept_texts[concept_idx].replace("a photo of ", ""),
        "expected_concept_score": float(concept_scores[concept_texts.index(f'a photo of {sample.concept}')].item()),
        "pred_concept_score": float(concept_scores[concept_idx].item()),
        "concept_top1_hit": int(concept_texts[concept_idx] == f"a photo of {sample.concept}"),
        "prompt_alignment": float(prompt_score),
        "drift_clip_cosine_to_original": drift_clip_cosine,
    }


def append_scores(scores_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    exists = scores_path.exists()
    with scores_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def copy_kept_samples(src_dir: Path, kept_dir: Path, keep_samples: int) -> None:
    kept_dir.mkdir(parents=True, exist_ok=True)
    for old_file in kept_dir.glob("*.png"):
        old_file.unlink()
    for image_path in sorted(src_dir.glob("*.png"))[:keep_samples]:
        shutil.copy2(image_path, kept_dir / image_path.name)


def maybe_compute_fid(src_a: Path, src_b: Path) -> float | None:
    import torch

    try:
        from torch_fidelity import calculate_metrics
    except Exception as exc:
        print(f"[fid] torch_fidelity unavailable: {exc}")
        return None
    try:
        metrics = calculate_metrics(
            input1=str(src_a),
            input2=str(src_b),
            cuda=torch.cuda.is_available(),
            fid=True,
            isc=False,
            kid=False,
            verbose=False,
        )
        return float(metrics["frechet_inception_distance"])
    except Exception as exc:
        print(f"[fid] failed for {src_a} vs {src_b}: {exc}")
        return None


def run_eval(args: argparse.Namespace, work_dir: Path) -> None:
    import torch
    from tqdm import tqdm

    dirs = ensure_dirs(work_dir)
    samples_by_group = build_samples(args.max_images_per_group, args.seed_offset)
    scores_path = dirs["metrics"] / "scores.csv"
    if args.force_eval and scores_path.exists():
        scores_path.unlink()
    existing_rows = read_scores(scores_path)

    group_texts = list(GROUP_LABELS.values())
    concept_texts = sorted({f"a photo of {concept}" for concepts in PROMPT_GROUPS.values() for concept in concepts})
    prompt_texts = sorted({sample.prompt for samples in samples_by_group.values() for sample in samples})

    clip_model, clip_processor = load_clip(args.device)
    group_feats = normalized_text_features(clip_model, clip_processor, group_texts, args.device)
    concept_feats = normalized_text_features(clip_model, clip_processor, concept_texts, args.device)
    prompt_feats_tensor = normalized_text_features(clip_model, clip_processor, prompt_texts, args.device)
    prompt_text_feats = {text: prompt_feats_tensor[i : i + 1] for i, text in enumerate(prompt_texts)}

    ckpts = checkpoint_paths(work_dir)
    original_feature_dir = dirs["metrics"] / "original_features"
    original_feature_dir.mkdir(parents=True, exist_ok=True)

    fid_rows: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        ckpt = ckpts[model_name]
        if ckpt is not None and not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {ckpt}")
        print(f"[eval] loading {model_name}")
        pipe = load_pipe(args, ckpt)
        for group, samples in samples_by_group.items():
            tmp_dir = dirs["tmp_images"] / model_name / group
            kept_dir = dirs["kept_samples"] / model_name / group
            completed = [
                r for r in existing_rows
                if r.get("model") == model_name and r.get("group") == group
            ]
            if len(completed) >= len(samples) and not args.force_eval:
                print(f"[eval] skip completed {model_name}/{group}")
                continue
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            rows: list[dict[str, Any]] = []
            original_feats_for_group: dict[str, torch.Tensor] | None = None
            original_feat_path = original_feature_dir / f"{group}.pt"
            if model_name != "original_sd15":
                original_feats_for_group = torch.load(original_feat_path, map_location="cpu")

            generated_feats: dict[str, torch.Tensor] = {}
            for sample in tqdm(samples, desc=f"{model_name}/{group}"):
                image_path = tmp_dir / f"{sample.sample_id}.png"
                generator = torch.Generator(device=args.device).manual_seed(sample.seed)
                with torch.no_grad():
                    image = pipe(
                        sample.prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        height=args.image_size,
                        width=args.image_size,
                        generator=generator,
                    ).images[0].convert("RGB")
                image.save(image_path)
                image_feat = normalized_image_features(clip_model, clip_processor, [image], args.device)
                generated_feats[sample.sample_id] = image_feat.cpu()
                rows.append(
                    score_image(
                        image=image,
                        sample=sample,
                        model_name=model_name,
                        image_path=image_path,
                        image_feat=image_feat,
                        prompt_text_feats=prompt_text_feats,
                        group_texts=group_texts,
                        group_feats=group_feats,
                        concept_texts=concept_texts,
                        concept_feats=concept_feats,
                        original_feats=original_feats_for_group,
                    )
                )

            append_scores(scores_path, rows)
            copy_kept_samples(tmp_dir, kept_dir, args.keep_samples)
            if model_name == "original_sd15":
                torch.save(generated_feats, original_feat_path)
            elif args.run_fid:
                original_tmp_dir = dirs["tmp_images"] / "original_sd15" / group
                if original_tmp_dir.exists():
                    fid = maybe_compute_fid(original_tmp_dir, tmp_dir)
                    fid_rows.append({"model": model_name, "group": group, "drift_fid_to_original": fid})

            keep_for_fid = args.run_fid and model_name == "original_sd15"
            if not args.skip_delete and not keep_for_fid:
                shutil.rmtree(tmp_dir)

        del pipe
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if fid_rows:
        with (dirs["metrics"] / "fid.json").open("w") as f:
            json.dump(fid_rows, f, indent=2, ensure_ascii=False)

    if args.run_fid and not args.skip_delete:
        original_tmp_root = dirs["tmp_images"] / "original_sd15"
        if original_tmp_root.exists():
            shutil.rmtree(original_tmp_root)

    make_summary_and_grids(work_dir, args)


def read_scores(scores_path: Path) -> list[dict[str, Any]]:
    if not scores_path.exists():
        return []
    with scores_path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    summary: dict[str, Any] = {}
    for model in MODEL_ORDER:
        model_rows = [r for r in rows if r["model"] == model]
        summary[model] = {}
        for group in PROMPT_GROUPS:
            group_rows = [r for r in model_rows if r["group"] == group]
            if not group_rows:
                continue
            drift_values = [safe_float(r["drift_clip_cosine_to_original"]) for r in group_rows]
            drift_values = [v for v in drift_values if v is not None]
            summary[model][group] = {
                "n": len(group_rows),
                "group_top1_rate": float(np.mean([int(r["group_top1_hit"]) for r in group_rows])),
                "concept_top1_rate": float(np.mean([int(r["concept_top1_hit"]) for r in group_rows])),
                "mean_prompt_alignment": float(np.mean([float(r["prompt_alignment"]) for r in group_rows])),
                "mean_expected_group_score": float(np.mean([float(r["expected_group_prob_proxy"]) for r in group_rows])),
                "mean_expected_concept_score": float(np.mean([float(r["expected_concept_score"]) for r in group_rows])),
                "mean_drift_clip_cosine_to_original": float(np.mean(drift_values)) if drift_values else None,
            }
    return summary


def make_grid(work_dir: Path, group: str, keep_samples: int) -> Path | None:
    from PIL import Image, ImageDraw, ImageFont

    kept_root = work_dir / "kept_samples"
    out_dir = work_dir / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    first_model_dir = kept_root / MODEL_ORDER[0] / group
    sample_files = sorted(first_model_dir.glob("*.png"))[:keep_samples]
    if not sample_files:
        return None
    cell = 192
    label_h = 28
    cols = len(MODEL_ORDER)
    rows = len(sample_files)
    canvas = Image.new("RGB", (cols * cell, rows * (cell + label_h) + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for col, model in enumerate(MODEL_ORDER):
        draw.text((col * cell + 6, 6), model, fill="black", font=font)
    for row_idx, sample_path in enumerate(sample_files):
        y0 = label_h + row_idx * (cell + label_h)
        draw.text((6, y0 + 4), sample_path.stem[-4:], fill="black", font=font)
        for col, model in enumerate(MODEL_ORDER):
            image_path = kept_root / model / group / sample_path.name
            if not image_path.exists():
                continue
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((cell, cell))
            x = col * cell + (cell - img.width) // 2
            y = y0 + label_h + (cell - img.height) // 2
            canvas.paste(img, (x, y))
    out_path = out_dir / f"{group}.png"
    canvas.save(out_path)
    return out_path


def make_report(work_dir: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    report_path = work_dir / "scapre_ased_religious_buildings_report.md"
    def metric(model: str, group: str, key: str) -> float | None:
        item = summary.get(model, {}).get(group)
        if not item:
            return None
        value = item.get(key)
        return None if value is None else float(value)

    def delta(with_key: str, without_key: str, group: str, key: str) -> float | None:
        a = metric(with_key, group, key)
        b = metric(without_key, group, key)
        if a is None or b is None:
            return None
        return a - b

    observations = []
    target_group_delta = delta("scapre_with_R", "scapre_no_R", "target_religious_buildings", "group_top1_rate")
    target_concept_delta = delta("scapre_with_R", "scapre_no_R", "target_religious_buildings", "concept_top1_rate")
    religion_preserve_delta = delta("scapre_with_R", "scapre_no_R", "near_religious_people_objects", "concept_top1_rate")
    building_preserve_delta = delta("scapre_with_R", "scapre_no_R", "near_generic_buildings", "concept_top1_rate")
    far_preserve_delta = delta("scapre_with_R", "scapre_no_R", "far_unrelated_objects", "concept_top1_rate")
    if target_group_delta is not None and target_concept_delta is not None:
        observations.append(
            f"- Target 宗教建築：with_R 的 group top1 比 no_R 低 {abs(target_group_delta):.2%}，"
            f"concept top1 低 {abs(target_concept_delta):.2%}；在這批樣本裡 R 沒有讓 erase 變弱，反而略強。"
            if target_group_delta <= 0 and target_concept_delta <= 0
            else
            f"- Target 宗教建築：with_R 相對 no_R 的 group top1 差 {target_group_delta:+.2%}，"
            f"concept top1 差 {target_concept_delta:+.2%}。"
        )
    if religion_preserve_delta is not None:
        observations.append(
            f"- 宗教人事物 preserve：with_R 的 concept top1 比 no_R 高 {religion_preserve_delta:.2%}；"
            "這是目前最符合「R 減少宗教相關概念誤傷」的訊號。"
            if religion_preserve_delta > 0
            else
            f"- 宗教人事物 preserve：with_R 的 concept top1 比 no_R 低 {abs(religion_preserve_delta):.2%}。"
        )
    if building_preserve_delta is not None:
        observations.append(
            f"- 一般建築 preserve：with_R 的 concept top1 比 no_R 低 {abs(building_preserve_delta):.2%}；"
            "這表示本次設定下 R 沒有保護一般建築，甚至可能更傷一般建築細分類。"
            if building_preserve_delta < 0
            else
            f"- 一般建築 preserve：with_R 的 concept top1 比 no_R 高 {building_preserve_delta:.2%}。"
        )
    if far_preserve_delta is not None:
        observations.append(
            f"- 無關物件 preserve：with_R 的 concept top1 比 no_R 高 {far_preserve_delta:.2%}，差異很小，"
            "表示遠距概念大致穩定。"
            if abs(far_preserve_delta) <= 0.05
            else
            f"- 無關物件 preserve：with_R 相對 no_R 的 concept top1 差 {far_preserve_delta:+.2%}。"
        )
    observations.append(
        "- 總結：這輪 50 張/類的小樣本支持「R 對宗教相關人事物有輕微保護」；"
        "但沒有支持「R 保護一般建築」，一般建築的 concept top1 反而下降。"
    )

    lines = [
        "# ScaPre ASED R 宗教建築遺忘小實驗",
        "",
        "## 實驗問題",
        "",
        "本實驗比較 ScaPre 在遺忘一組宗教建築概念時，有無 ASED regularizer "
        "`R = U diag(tilde_sigma) U^T` 對圖片結果的影響。重點不是只看 target 是否被忘掉，"
        "而是看一般建築與宗教相關人事物是否被誤傷。",
        "",
        "## 設定",
        "",
        f"- Base model: `{args.base_model}`",
        f"- 每類圖片數: `{args.max_images_per_group}`",
        f"- Inference steps: `{args.num_inference_steps}`",
        f"- Guidance scale: `{args.guidance_scale}`",
        "- 比較模型: `original_sd15`, `scapre_no_R`, `scapre_with_R`",
        "- `with_R` 額外使用 `--enable_ased --T_sigma 1 --p_sigma 1`",
        "",
        "## 指標摘要",
        "",
        "`target_religious_buildings` 的 group/concept top-1 越低代表遺忘越強；"
        "其他三類越高代表 preserve 越好。`drift_clip` 越接近 1，代表越接近 original SD1.5 的同 prompt/seed 圖片。",
        "",
    ]
    for group in PROMPT_GROUPS:
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| model | n | group top1 | concept top1 | prompt align | expected group score | drift clip |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for model in MODEL_ORDER:
            item = summary.get(model, {}).get(group)
            if not item:
                continue
            drift = item["mean_drift_clip_cosine_to_original"]
            lines.append(
                "| {model} | {n} | {group_top1:.4f} | {concept_top1:.4f} | "
                "{prompt:.4f} | {group_score:.4f} | {drift} |".format(
                    model=model,
                    n=item["n"],
                    group_top1=item["group_top1_rate"],
                    concept_top1=item["concept_top1_rate"],
                    prompt=item["mean_prompt_alignment"],
                    group_score=item["mean_expected_group_score"],
                    drift="-" if drift is None else f"{drift:.4f}",
                )
            )
        grid_path = work_dir / "grids" / f"{group}.png"
        if grid_path.exists():
            lines.append("")
            lines.append(f"![{group}](grids/{group}.png)")
        lines.append("")
    lines.extend(
        [
            "## 本次結果觀察",
            "",
            *observations,
            "",
            "## 初步判讀方式",
            "",
            "- 如果 `scapre_with_R` 在 target 類別和 `scapre_no_R` 一樣低，但 near preserve 類別更高，代表 R 有幫助。",
            "- 如果 `scapre_with_R` 的 target 類別明顯高於 `scapre_no_R`，代表 R 可能過度保守。",
            "- 如果 `scapre_no_R` 在 `near_generic_buildings` 掉很多，代表不加 R 可能傷到建築共享方向。",
            "- 如果 `scapre_no_R` 在 `near_religious_people_objects` 掉很多，代表不加 R 可能傷到宗教共享方向。",
            "",
            "## 限制",
            "",
            "- 這是小樣本實驗，CLIP 分數只能做方向性判斷。",
            "- 這裡的 drift 是 generated-to-generated 比較，不是真實資料 FID。",
            "- 最終仍需要人工看 grid，確認建築細節、宗教符號與圖片品質是否合理。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {report_path}")


def make_summary_and_grids(work_dir: Path, args: argparse.Namespace) -> None:
    scores_path = work_dir / "metrics" / "scores.csv"
    rows = read_scores(scores_path)
    summary = summarize(rows)
    summary_path = work_dir / "metrics" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for group in PROMPT_GROUPS:
        make_grid(work_dir, group, args.keep_samples)
    make_report(work_dir, summary, args)
    print(f"[summary] wrote {summary_path}")


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    ensure_dirs(work_dir)
    if args.mode in {"all", "edit"}:
        run_edit(args, work_dir)
    if args.mode in {"all", "eval"}:
        run_eval(args, work_dir)
    if args.mode == "report":
        make_summary_and_grids(work_dir, args)


if __name__ == "__main__":
    main()
