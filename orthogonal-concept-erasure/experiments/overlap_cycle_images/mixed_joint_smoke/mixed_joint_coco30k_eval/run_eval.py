from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_fidelity
from diffusers import DiffusionPipeline
from PIL import Image
from safetensors.torch import load_file
from torch_fidelity.datasets import ImagesPathDataset
from transformers import CLIPModel, CLIPProcessor


HERE = Path(__file__).resolve().parent
SMOKE_ROOT = HERE.parent
REPO_ROOT = SMOKE_ROOT.parents[2]
REFERENCE_INFRA = REPO_ROOT / "experiments" / "evaluation_references"
sys.path.insert(0, str(REFERENCE_INFRA))
from reference_registry import resolve_reference, upsert_reference  # noqa: E402


DEFAULT_PROMPTS = REPO_ROOT / "data" / "coco_30k.csv"
DEFAULT_CHECKPOINT = SMOKE_ROOT / "joint_official_subspace.safetensors"
PROMPTS_USED = HERE / "prompts_first_10000.csv"
METRICS_PATH = HERE / "metrics.json"
SUMMARY_PATH = HERE / "summary.md"
RUN_STATE_PATH = HERE / "run_state.json"
CLEANUP_PATH = HERE / "cleanup_manifest.json"
IMAGES_ROOT = HERE / "generated_images"
MILESTONE_ROOT = HERE / "milestones" / "first1000"
MILESTONE_METRICS_PATH = MILESTONE_ROOT / "metrics.json"
MILESTONE_SUMMARY_PATH = MILESTONE_ROOT / "summary.md"
METHOD_DIRS = {
    "original_sd": IMAGES_ROOT / "original_sd",
    "mixed_joint_oce": IMAGES_ROOT / "mixed_joint_oce",
}

MODEL_ID = "CompVis/stable-diffusion-v1-4"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
NUM_PROMPTS = 10_000
MILESTONE_COUNT = 1_000
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
HEIGHT = 512
WIDTH = 512
DTYPE = torch.bfloat16
PRODUCER = str(Path(__file__).resolve())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def reference_id(prompt_count: int) -> str:
    return (
        f"sd14_mscoco30k_first{prompt_count}_"
        "pndm50_cfg7p5_512_bf16"
    )


def reference_root(prompt_count: int) -> Path:
    return REFERENCE_INFRA / "references" / reference_id(prompt_count)


def reference_clip_path(prompt_count: int) -> Path:
    return reference_root(prompt_count) / "clip_baseline.json"


def reference_fid_cache_root(prompt_count: int) -> Path:
    return reference_root(prompt_count) / "torch_fidelity_cache"


def reference_prompt_manifest(prompt_count: int) -> Path:
    return reference_root(prompt_count) / f"prompts_first{prompt_count}.csv"


def reference_protocol_manifest(prompt_count: int) -> Path:
    return reference_root(prompt_count) / "protocol.json"


def fid_statistics_paths(prompt_count: int) -> list[Path]:
    return sorted(
        reference_fid_cache_root(prompt_count).glob(
            f"{reference_id(prompt_count)}-*-stat-fid-2048.pt"
        )
    )


def original_reference_identity(
    settings: dict[str, Any], prompt_count: int
) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "prompt_source_sha256": settings["prompt_source_sha256"],
        "prompt_subset": f"first {prompt_count} rows in source order",
        "prompt_count": prompt_count,
        "seed_column": "evaluation_seed",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "height": HEIGHT,
        "width": WIDTH,
        "dtype": "bfloat16",
        "scheduler": "PNDMScheduler",
        "clip_model_id": CLIP_MODEL_ID,
        "clip_implementation": "transformers logits_per_image diagonal",
        "fid_implementation": "torch_fidelity 0.3.0",
        "fid_feature_extractor": "inception-v3-compat",
        "fid_feature_layer": "2048",
    }


def reference_artifacts(prompt_count: int) -> dict[str, str]:
    return {
        "clip_baseline": str(reference_clip_path(prompt_count)),
        "fid_statistics_glob": str(
            reference_fid_cache_root(prompt_count)
            / f"{reference_id(prompt_count)}-*-stat-fid-2048.pt"
        ),
        "prompt_manifest": str(reference_prompt_manifest(prompt_count)),
        "protocol_manifest": str(reference_protocol_manifest(prompt_count)),
    }


def resolved_settings(
    prompts_path: Path, checkpoint_path: Path, batch_size: int
) -> dict[str, Any]:
    return {
        "protocol": (
            "OCE paper preservation protocol: first 10,000 prompts from "
            "the repository MSCOCO-30k source"
        ),
        "prompt_source": str(prompts_path),
        "prompt_source_sha256": sha256(prompts_path),
        "prompt_subset": "first 10,000 rows in source order",
        "prompt_count": NUM_PROMPTS,
        "milestone_prompt_count": MILESTONE_COUNT,
        "model_id": MODEL_ID,
        "mixed_checkpoint": str(checkpoint_path),
        "mixed_checkpoint_sha256": sha256(checkpoint_path),
        "methods": ["original_sd", "mixed_joint_oce"],
        "generation": {
            "scheduler": "PNDMScheduler",
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "height": HEIGHT,
            "width": WIDTH,
            "num_images_per_prompt": 1,
            "dtype": "bfloat16",
            "batch_size": batch_size,
            "safety_checker": None,
            "seed_column": "evaluation_seed",
        },
        "metrics": {
            "clip_score": (
                f"{CLIP_MODEL_ID} logits_per_image, matching "
                "metrics/eval_clip_score.py"
            ),
            "fid": (
                "torch_fidelity FID, matching metrics/eval_fid.py; "
                "Original SD generations are the reference distribution"
            ),
            "original_fid": "0.0 because Original SD is the FID reference",
        },
        "shared_references": {
            str(count): {
                "reference_id": reference_id(count),
                "registry": str(REFERENCE_INFRA / "registry.json"),
                **reference_artifacts(count),
            }
            for count in (MILESTONE_COUNT, NUM_PROMPTS)
        },
    }


def load_prompts(prompts_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(prompts_path)
    required = {"case_number", "prompt", "evaluation_seed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prompt CSV missing columns: {sorted(missing)}")
    if len(frame) < NUM_PROMPTS:
        raise ValueError(
            f"Expected at least {NUM_PROMPTS} prompts, found {len(frame)}"
        )
    subset = frame.iloc[:NUM_PROMPTS].copy()
    if subset["case_number"].duplicated().any():
        raise ValueError("case_number must be unique in the first 10,000 rows")
    PROMPTS_USED.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(PROMPTS_USED, index=False)
    return subset


def register_building_references(
    frame: pd.DataFrame, settings: dict[str, Any]
) -> None:
    for count in (MILESTONE_COUNT, NUM_PROMPTS):
        identity = original_reference_identity(settings, count)
        complete = resolve_reference(
            reference_id(count), identity, require_complete=True
        )
        if complete is not None:
            continue
        root = reference_root(count)
        root.mkdir(parents=True, exist_ok=True)
        frame.iloc[:count].to_csv(
            reference_prompt_manifest(count), index=False
        )
        write_json(
            reference_protocol_manifest(count),
            {
                "reference_id": reference_id(count),
                "identity": identity,
                "source_prompt_csv": settings["prompt_source"],
                "source_prompt_csv_sha256": settings[
                    "prompt_source_sha256"
                ],
            },
        )
        upsert_reference(
            reference_id(count),
            identity,
            "building",
            reference_artifacts(count),
            PRODUCER,
        )


def mark_reference_complete(
    settings: dict[str, Any], prompt_count: int
) -> None:
    stats = fid_statistics_paths(prompt_count)
    if not reference_clip_path(prompt_count).is_file() or len(stats) != 1:
        raise RuntimeError(
            f"Cannot complete reference {reference_id(prompt_count)}: "
            f"clip={reference_clip_path(prompt_count).is_file()}, "
            f"fid_stats={len(stats)}"
        )
    upsert_reference(
        reference_id(prompt_count),
        original_reference_identity(settings, prompt_count),
        "complete",
        reference_artifacts(prompt_count),
        PRODUCER,
    )


def reusable_original_reference_ready(
    settings: dict[str, Any], prompt_count: int
) -> bool:
    return (
        resolve_reference(
            reference_id(prompt_count),
            original_reference_identity(settings, prompt_count),
            require_complete=True,
        )
        is not None
    )


def image_path(method: str, case_number: int) -> Path:
    return METHOD_DIRS[method] / f"{case_number}.png"


def completed_count(frame: pd.DataFrame, method: str) -> int:
    return sum(
        image_path(method, int(case_number)).is_file()
        for case_number in frame["case_number"]
    )


def save_run_state(
    settings: dict[str, Any],
    frame: pd.DataFrame,
    stage: str,
    started_at: str,
    completed_images: dict[str, int] | None = None,
    **extra: Any,
) -> None:
    if completed_images is None:
        completed_images = {
            method: completed_count(frame, method) for method in METHOD_DIRS
        }
    write_json(
        RUN_STATE_PATH,
        {
            "status": "running",
            "stage": stage,
            "started_at": started_at,
            "updated_at": utc_now(),
            "settings": settings,
            "completed_images": completed_images,
            "milestone_metrics_path": (
                str(MILESTONE_METRICS_PATH)
                if MILESTONE_METRICS_PATH.is_file()
                else None
            ),
            **extra,
        },
    )


def load_pipeline() -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        local_files_only=True,
    ).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    actual_scheduler = type(pipe.scheduler).__name__
    if actual_scheduler != "PNDMScheduler":
        raise RuntimeError(f"Unexpected scheduler: {actual_scheduler}")
    return pipe


def apply_edited_checkpoint(
    pipe: DiffusionPipeline, checkpoint_path: Path
) -> dict[str, torch.Tensor]:
    state = load_file(str(checkpoint_path))
    incompatible = pipe.unet.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {incompatible.unexpected_keys}"
        )
    return state


def unload_pipeline(
    pipe: DiffusionPipeline, state: dict[str, torch.Tensor] | None = None
) -> None:
    pipe.to("cpu")
    del pipe
    if state is not None:
        del state
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_method(
    pipe: DiffusionPipeline,
    frame: pd.DataFrame,
    method: str,
    batch_size: int,
    settings: dict[str, Any],
    started_at: str,
    stop_after: int | None = None,
) -> None:
    active = frame.iloc[:stop_after] if stop_after is not None else frame
    destination = METHOD_DIRS[method]
    destination.mkdir(parents=True, exist_ok=True)
    pending = active[
        [
            not image_path(method, int(case_number)).is_file()
            for case_number in active["case_number"]
        ]
    ]
    target_total = len(active)
    if pending.empty:
        print(
            f"[generate] {method}: already complete "
            f"({target_total}/{target_total})",
            flush=True,
        )
        return
    initial_done = target_total - len(pending)
    done = initial_done
    method_started = time.monotonic()
    for batch_index, offset in enumerate(
        range(0, len(pending), batch_size), start=1
    ):
        batch = pending.iloc[offset : offset + batch_size]
        images = pipe(
            prompt=batch["prompt"].astype(str).tolist(),
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
            generator=[
                torch.Generator(device="cuda:0").manual_seed(int(seed))
                for seed in batch["evaluation_seed"]
            ],
        ).images
        for case_number, image in zip(batch["case_number"], images):
            image.save(image_path(method, int(case_number)))
        done += len(batch)
        elapsed = time.monotonic() - method_started
        rate = max(done - initial_done, 1) / max(elapsed, 1e-9)
        remaining = max(target_total - done, 0) / max(rate, 1e-9)
        print(
            f"[generate] {method}: {done}/{target_total} "
            f"({rate:.3f} images/s, ETA {remaining / 3600:.2f} h)",
            flush=True,
        )
        if batch_index % 10 == 0 or done == target_total:
            save_run_state(
                settings,
                frame,
                f"generate_{method}",
                started_at,
                completed_images={
                    method: completed_count(frame, method),
                    (
                        "mixed_joint_oce"
                        if method == "original_sd"
                        else "original_sd"
                    ): completed_count(
                        frame,
                        (
                            "mixed_joint_oce"
                            if method == "original_sd"
                            else "original_sd"
                        ),
                    ),
                },
                current_method=method,
                current_target=target_total,
                current_method_eta_seconds=remaining,
            )


def validate_selected_images(
    frame: pd.DataFrame,
    prompt_count: int,
    require_original_images: bool,
) -> dict[str, Any]:
    selected = frame.iloc[:prompt_count]
    result: dict[str, Any] = {}
    for method, directory in METHOD_DIRS.items():
        if method == "original_sd" and not require_original_images:
            result[method] = {
                "source": "repository-level reusable Original reference",
                "reference_id": reference_id(prompt_count),
                "selected_count": prompt_count,
                "missing": [],
            }
            continue
        missing = [
            int(case_number)
            for case_number in selected["case_number"]
            if not image_path(method, int(case_number)).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"{method} is missing {len(missing)} selected images: "
                f"{missing[:10]}"
            )
        result[method] = {
            "directory": str(directory),
            "selected_count": prompt_count,
            "directory_png_count": len(list(directory.glob("*.png"))),
            "missing": [],
        }
    return result


@torch.inference_mode()
def clip_scores(
    frame: pd.DataFrame,
    method: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID, local_files_only=True
    ).eval().to(device)
    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_ID, local_files_only=True
    )
    values: list[float] = []
    for offset in range(0, len(frame), batch_size):
        batch = frame.iloc[offset : offset + batch_size]
        images: list[Image.Image] = []
        for case_number in batch["case_number"]:
            with Image.open(image_path(method, int(case_number))) as image:
                images.append(image.convert("RGB"))
        inputs = processor(
            text=batch["prompt"].astype(str).tolist(),
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = model(
            **{key: value.to(device) for key, value in inputs.items()}
        )
        values.extend(
            float(value) for value in outputs.logits_per_image.diagonal().cpu()
        )
        print(
            f"[clip] {method}: {min(offset + batch_size, len(frame))}/"
            f"{len(frame)}",
            flush=True,
        )
    array = np.asarray(values, dtype=np.float64)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "count": int(array.size),
    }


def load_original_clip(
    settings: dict[str, Any], prompt_count: int
) -> dict[str, float] | None:
    path = reference_clip_path(prompt_count)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["reference_identity"] != original_reference_identity(
        settings, prompt_count
    ):
        raise RuntimeError(
            f"Original CLIP baseline identity mismatch for {prompt_count}"
        )
    return payload["clip_score"]


def save_original_clip(
    settings: dict[str, Any],
    prompt_count: int,
    clip_score: dict[str, float],
) -> None:
    write_json(
        reference_clip_path(prompt_count),
        {
            "created_at": utc_now(),
            "reference_id": reference_id(prompt_count),
            "reference_identity": original_reference_identity(
                settings, prompt_count
            ),
            "clip_model_id": CLIP_MODEL_ID,
            "clip_score": clip_score,
        },
    )


def calculate_fid(frame: pd.DataFrame, prompt_count: int) -> float:
    selected = frame.iloc[:prompt_count]
    edited_dataset = ImagesPathDataset(
        [
            str(image_path("mixed_joint_oce", int(case_number)))
            for case_number in selected["case_number"]
        ]
    )
    original_dataset = ImagesPathDataset(
        [
            str(image_path("original_sd", int(case_number)))
            for case_number in selected["case_number"]
        ]
    )
    cache_root = reference_fid_cache_root(prompt_count)
    cache_root.mkdir(parents=True, exist_ok=True)
    result = torch_fidelity.calculate_metrics(
        input1=edited_dataset,
        input2=original_dataset,
        input2_cache_name=reference_id(prompt_count),
        cache_root=str(cache_root),
        cuda=True,
        fid=True,
        verbose=False,
    )
    return float(result["frechet_inception_distance"])


def prune_reference_feature_cache(prompt_count: int) -> list[str]:
    removed: list[str] = []
    for path in reference_fid_cache_root(prompt_count).glob(
        f"{reference_id(prompt_count)}-*-features-*.pt"
    ):
        path.unlink()
        removed.append(str(path))
    return removed


def checkpoint_matches_metrics(
    path: Path, settings: dict[str, Any]
) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload.get("mixed_checkpoint_sha256")
        == settings["mixed_checkpoint_sha256"]
        and payload.get("status") == "complete"
    )


def write_metrics_summary(
    path: Path,
    metrics: dict[str, Any],
    title: str,
    cleanup_note: str,
) -> None:
    original = metrics["models"]["original_sd"]
    edited = metrics["models"]["mixed_joint_oce"]
    delta = metrics["difference_mixed_minus_original"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"Prompt count: {metrics['prompt_count']}",
                "",
                "| Model | CLIP Score ↑ | FID to Original SD ↓ |",
                "|---|---:|---:|",
                f"| Original SD | {original['clip_score']['mean']:.4f} "
                f"± {original['clip_score']['std']:.4f} | 0.0000 |",
                f"| Mixed heterogeneous joint OCE | "
                f"{edited['clip_score']['mean']:.4f} "
                f"± {edited['clip_score']['std']:.4f} | "
                f"{edited['fid_to_original_sd']:.4f} |",
                f"| Difference (mixed − original) | "
                f"{delta['clip_score_mean']:+.4f} | "
                f"{delta['fid_to_original_sd']:+.4f} |",
                "",
                "Original SD 是相同 prompt／seed 的 FID reference，"
                "因此其 FID 基準按定義為 0。",
                "",
                cleanup_note,
                "",
            ]
        ),
        encoding="utf-8",
    )


def evaluate_prompt_count(
    frame: pd.DataFrame,
    settings: dict[str, Any],
    prompt_count: int,
    clip_batch_size: int,
    metrics_path: Path,
    summary_path: Path,
    title: str,
    cleanup_note: str,
) -> dict[str, Any]:
    reuse = reusable_original_reference_ready(settings, prompt_count)
    validation = validate_selected_images(
        frame, prompt_count, require_original_images=not reuse
    )
    selected = frame.iloc[:prompt_count]
    device = torch.device("cuda:0")
    original_clip = load_original_clip(settings, prompt_count)
    if original_clip is None:
        original_clip = clip_scores(
            selected, "original_sd", device, clip_batch_size
        )
        save_original_clip(settings, prompt_count, original_clip)
    edited_clip = clip_scores(
        selected, "mixed_joint_oce", device, clip_batch_size
    )
    edited_fid = calculate_fid(frame, prompt_count)
    removed_feature_caches = prune_reference_feature_cache(prompt_count)
    mark_reference_complete(settings, prompt_count)
    metrics = {
        "status": "complete",
        "completed_at": utc_now(),
        "prompt_count": prompt_count,
        "mixed_checkpoint": settings["mixed_checkpoint"],
        "mixed_checkpoint_sha256": settings["mixed_checkpoint_sha256"],
        "prompt_manifest": str(reference_prompt_manifest(prompt_count)),
        "reference_id": reference_id(prompt_count),
        "reused_original_reference": reuse,
        "image_validation": validation,
        "models": {
            "original_sd": {
                "clip_score": original_clip,
                "fid_to_original_sd": 0.0,
                "fid_role": "reference distribution",
            },
            "mixed_joint_oce": {
                "clip_score": edited_clip,
                "fid_to_original_sd": edited_fid,
            },
        },
        "difference_mixed_minus_original": {
            "clip_score_mean": edited_clip["mean"] - original_clip["mean"],
            "fid_to_original_sd": edited_fid,
        },
        "reference_cache": {
            **reference_artifacts(prompt_count),
            "removed_per_image_feature_caches": removed_feature_caches,
        },
        "settings": settings,
    }
    write_json(metrics_path, metrics)
    write_metrics_summary(
        summary_path, metrics, title=title, cleanup_note=cleanup_note
    )
    return metrics


def cleanup_generated_images() -> dict[str, Any]:
    expected_root = (HERE / "generated_images").resolve()
    actual_root = IMAGES_ROOT.resolve()
    if actual_root != expected_root or actual_root.parent != HERE.resolve():
        raise RuntimeError(f"Refusing unsafe cleanup target: {actual_root}")
    removed: list[dict[str, Any]] = []
    for method, directory in METHOD_DIRS.items():
        resolved = directory.resolve()
        if resolved.parent != actual_root:
            raise RuntimeError(f"Refusing unsafe cleanup target: {resolved}")
        count = len(list(resolved.glob("*.png"))) if resolved.exists() else 0
        if resolved.exists():
            shutil.rmtree(resolved)
        removed.append(
            {"method": method, "directory": str(resolved), "image_count": count}
        )
    if actual_root.exists() and not any(actual_root.iterdir()):
        actual_root.rmdir()
    return {
        "completed_at": utc_now(),
        "generated_images_retained": False,
        "removed": removed,
        "preserved_shared_references": {
            str(count): reference_artifacts(count)
            for count in (MILESTONE_COUNT, NUM_PROMPTS)
        },
        "shared_resources_removed": False,
    }


def write_final_summary(
    metrics: dict[str, Any], cleanup: dict[str, Any]
) -> None:
    if cleanup["generated_images_retained"]:
        cleanup_note = (
            "因明確使用 `--keep-images`，本輪生成圖片仍保留於 "
            f"`{IMAGES_ROOT}`。"
        )
    else:
        cleanup_note = (
            f"Metrics 成功後已刪除 Original "
            f"{cleanup['removed'][0]['image_count']} 張與 mixed "
            f"{cleanup['removed'][1]['image_count']} 張。中央 1k/10k "
            "Original CLIP baseline、FID statistics、protocol 與 prompt "
            "manifest 均已保留；per-image Inception features 未保留。"
        )
    write_metrics_summary(
        SUMMARY_PATH,
        metrics,
        title="Mixed joint OCE — MSCOCO first-10k preservation",
        cleanup_note=cleanup_note,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Original SD vs the mixed heterogeneous joint official-"
            "OCE checkpoint, with a first-1k milestone and paper first-10k."
        )
    )
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep generated images after successful 10k metrics.",
    )
    parser.add_argument(
        "--stop-after-first1000",
        action="store_true",
        help=(
            "Write the first-1k screening result, retain resumable images, "
            "and exit before generating prompts 1001-10000."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts_path = args.prompts_path.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    if not prompts_path.is_file():
        raise FileNotFoundError(prompts_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.batch_size < 1 or args.clip_batch_size < 1:
        raise ValueError("Batch sizes must be positive")

    HERE.mkdir(parents=True, exist_ok=True)
    frame = load_prompts(prompts_path)
    settings = resolved_settings(prompts_path, checkpoint_path, args.batch_size)
    write_json(HERE / "resolved_settings.json", settings)
    register_building_references(frame, settings)
    started_at = utc_now()
    save_run_state(settings, frame, "initializing", started_at)

    reuse_10k_original = reusable_original_reference_ready(
        settings, NUM_PROMPTS
    )
    pipe = load_pipeline()
    if reuse_10k_original:
        print(
            "[reference] Reusing repository-level Original 10k reference; "
            "Original images will not be regenerated.",
            flush=True,
        )
    else:
        generate_method(
            pipe, frame, "original_sd", args.batch_size, settings, started_at
        )
    edited_state = apply_edited_checkpoint(pipe, checkpoint_path)

    milestone_done = checkpoint_matches_metrics(
        MILESTONE_METRICS_PATH, settings
    )
    if not milestone_done:
        generate_method(
            pipe,
            frame,
            "mixed_joint_oce",
            args.batch_size,
            settings,
            started_at,
            stop_after=MILESTONE_COUNT,
        )
        unload_pipeline(pipe, edited_state)
        save_run_state(
            settings, frame, "evaluate_first1000", started_at
        )
        evaluate_prompt_count(
            frame,
            settings,
            MILESTONE_COUNT,
            args.clip_batch_size,
            MILESTONE_METRICS_PATH,
            MILESTONE_SUMMARY_PATH,
            "Mixed joint OCE — first-1k preservation milestone",
            (
                "1k 圖片暫時保留，因同一程序將直接繼續生成至論文 "
                "first-10k；最終 10k metrics 成功後統一刪除。"
            ),
        )
        print(
            f"[milestone] metrics: {MILESTONE_METRICS_PATH}", flush=True
        )
        if args.stop_after_first1000:
            write_json(
                RUN_STATE_PATH,
                {
                    "status": "milestone_complete",
                    "stage": "first1000_complete",
                    "started_at": started_at,
                    "completed_at": utc_now(),
                    "milestone_metrics_path": str(MILESTONE_METRICS_PATH),
                    "milestone_summary_path": str(MILESTONE_SUMMARY_PATH),
                    "generated_images_retained_for_resume": True,
                    "continue_command": (
                        "conda run --no-capture-output -n py310 python "
                        f"{Path(__file__).resolve()}"
                    ),
                },
            )
            print(
                "[milestone] stopped as requested; rerun without "
                "--stop-after-first1000 to continue to 10k.",
                flush=True,
            )
            return
        pipe = load_pipeline()
        edited_state = apply_edited_checkpoint(pipe, checkpoint_path)

    generate_method(
        pipe, frame, "mixed_joint_oce", args.batch_size, settings, started_at
    )
    unload_pipeline(pipe, edited_state)

    validation = validate_selected_images(
        frame,
        NUM_PROMPTS,
        require_original_images=not reuse_10k_original,
    )
    save_run_state(
        settings,
        frame,
        "evaluate_first10000",
        started_at,
        image_validation=validation,
    )
    metrics = evaluate_prompt_count(
        frame,
        settings,
        NUM_PROMPTS,
        args.clip_batch_size,
        METRICS_PATH,
        SUMMARY_PATH,
        "Mixed joint OCE — MSCOCO first-10k preservation",
        "圖片將在 metrics 安全寫入後依 cleanup policy 處理。",
    )

    if args.keep_images:
        cleanup = {
            "completed_at": utc_now(),
            "generated_images_retained": True,
            "removed": [],
            "preserved": [str(path) for path in METHOD_DIRS.values()],
            "shared_resources_removed": False,
            "reason": "--keep-images was explicitly supplied",
        }
    else:
        cleanup = cleanup_generated_images()
    write_json(CLEANUP_PATH, cleanup)
    write_final_summary(metrics, cleanup)
    write_json(
        RUN_STATE_PATH,
        {
            "status": "complete",
            "stage": "complete",
            "started_at": started_at,
            "completed_at": utc_now(),
            "milestone_metrics_path": str(MILESTONE_METRICS_PATH),
            "metrics_path": str(METRICS_PATH),
            "summary_path": str(SUMMARY_PATH),
            "cleanup_manifest": str(CLEANUP_PATH),
            "generated_images_retained": cleanup[
                "generated_images_retained"
            ],
        },
    )
    print(f"[complete] metrics: {METRICS_PATH}", flush=True)
    print(f"[complete] summary: {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        previous: dict[str, Any] = {}
        if RUN_STATE_PATH.exists():
            try:
                previous = json.loads(RUN_STATE_PATH.read_text())
            except json.JSONDecodeError:
                pass
        previous.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "cleanup_performed": False,
                "note": (
                    "Generated images are retained after failure so the same "
                    "command can resume without repeated denoising."
                ),
            }
        )
        write_json(RUN_STATE_PATH, previous)
        raise
