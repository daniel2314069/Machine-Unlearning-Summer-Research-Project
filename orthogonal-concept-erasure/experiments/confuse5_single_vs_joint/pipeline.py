#!/usr/bin/env python3
"""Server-side Confuse5 generation and ResNet-50 evaluation pipeline.

The plan and status paths intentionally use only the Python standard library.
Torch, Diffusers, torchvision, Pillow, and safetensors are imported lazily only
by stages that perform real server-side generation or evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT_ROOT = HERE / "outputs" / "evaluation"
DEFAULT_SMOKE_ROOT = HERE / "outputs" / "smoke"
EDIT_RUNNER_PATH = HERE / "run.py"
REQUIRED_COLUMNS = ("case_number", "prompt", "class", "evaluation_seed")


class PipelineError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"Expected a JSON object in {path}")
    return payload


def load_edit_runner() -> Any:
    spec = importlib.util.spec_from_file_location("confuse5_edit_runner", EDIT_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise PipelineError(f"Cannot import edit runner: {EDIT_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_relative(raw_path: str | Path, relative_to: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def load_config(path: Path) -> tuple[dict[str, Any], Any]:
    runner = load_edit_runner()
    config = runner.load_and_validate(path)
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise PipelineError("config.evaluation must be an object")
    required = {
        "dataset_csv", "derived_dataset_csv", "derived_dataset",
        "expected_rows_per_class", "generation", "classifier",
        "expected_checkpoint_keys",
    }
    missing = sorted(required - set(evaluation))
    if missing:
        raise PipelineError(f"config.evaluation is missing fields: {missing}")
    if evaluation["expected_rows_per_class"] != 500:
        raise PipelineError("evaluation.expected_rows_per_class must be 500")
    if evaluation["expected_checkpoint_keys"] != 16:
        raise PipelineError("evaluation.expected_checkpoint_keys must be 16")
    derived = evaluation["derived_dataset"]
    if not isinstance(derived, dict) or derived.get("kind") != "internal_analysis_not_official_scapre":
        raise PipelineError(
            "evaluation.derived_dataset.kind must be internal_analysis_not_official_scapre"
        )
    for field in (
        "source_dataset_normalized_sha256", "derived_dataset_sha256", "seed_sources",
    ):
        if field not in derived:
            raise PipelineError(f"evaluation.derived_dataset is missing {field}")
    generation = evaluation["generation"]
    if not isinstance(generation, dict):
        raise PipelineError("evaluation.generation must be an object")
    expected_generation = {
        "scheduler": "PNDMScheduler",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "dtype": "bfloat16",
        "images_per_prompt": 1,
        "safety_checker": None,
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            raise PipelineError(
                f"evaluation.generation.{key} must be {expected!r}, "
                f"got {generation.get(key)!r}"
            )
    classifier = evaluation["classifier"]
    if not isinstance(classifier, dict) or classifier.get("implementation") != "torchvision_resnet50_default":
        raise PipelineError(
            "evaluation.classifier.implementation must be torchvision_resnet50_default"
        )
    if not isinstance(classifier.get("batch_size"), int) or classifier["batch_size"] < 1:
        raise PipelineError("evaluation.classifier.batch_size must be a positive integer")
    return config, runner


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise PipelineError(f"Dataset CSV has no header: {path}")
            missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
            if missing:
                raise PipelineError(f"Dataset CSV is missing columns: {missing}")
            rows = list(reader)
    except OSError as exc:
        raise PipelineError(f"Cannot read dataset CSV {path}: {exc}") from exc

    resolved: list[dict[str, Any]] = []
    by_class: dict[str, list[dict[str, Any]]] = {}
    seen_cases: set[int] = set()
    for line_number, raw in enumerate(rows, start=2):
        concept = " ".join(raw["class"].split())
        prompt = raw["prompt"].strip()
        if not concept or not prompt:
            raise PipelineError(f"Empty class or prompt at CSV line {line_number}")
        try:
            case_number = int(raw["case_number"])
            seed = int(raw["evaluation_seed"])
        except ValueError as exc:
            raise PipelineError(f"Non-integer case/seed at CSV line {line_number}") from exc
        if case_number in seen_cases:
            raise PipelineError(f"Duplicate case_number {case_number} at CSV line {line_number}")
        seen_cases.add(case_number)
        item = {
            "case_number": case_number,
            "prompt": prompt,
            "class": concept,
            "evaluation_seed": seed,
            "source_line": line_number,
        }
        resolved.append(item)
        by_class.setdefault(normalize(concept), []).append(item)
    return resolved, by_class


def selected_groups(
    config: Mapping[str, Any], raw_groups: Sequence[str] | None, runner: Any
) -> list[dict[str, Any]]:
    groups = list(config["groups"])
    if raw_groups is None:
        return groups
    requested = {runner.slug(value) for value in raw_groups}
    known = {group["id"] for group in groups}
    unknown = requested - known
    if unknown:
        raise PipelineError(
            f"Unknown group ids: {sorted(unknown)}; available={sorted(known)}"
        )
    return [group for group in groups if group["id"] in requested]


def dataset_coverage(
    groups: Sequence[Mapping[str, Any]],
    by_class: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_rows: int,
) -> dict[str, Any]:
    expected = [concept for group in groups for concept in group["concepts"]]
    available: list[str] = []
    missing: list[str] = []
    invalid_counts: dict[str, int] = {}
    for concept in expected:
        count = len(by_class.get(normalize(concept), ()))
        if count == expected_rows:
            available.append(concept)
        elif count == 0:
            missing.append(concept)
        else:
            invalid_counts[concept] = count
    return {
        "expected_classes": expected,
        "available_classes": available,
        "missing_classes": missing,
        "invalid_row_counts": invalid_counts,
        "expected_rows_per_class": expected_rows,
        "complete": not missing and not invalid_counts,
    }


def checkpoint_status(run: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = Path(run["checkpoint_path"])
    metadata_path = Path(run["metadata_path"])
    status: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "metadata_path": str(metadata_path),
        "checkpoint_present": checkpoint.is_file(),
        "metadata_present": metadata_path.is_file(),
        "metadata_complete": False,
    }
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status["metadata_complete"] = metadata.get("status") == "complete"
            status["recorded_checkpoint_sha256"] = metadata.get("checkpoint_sha256")
        except (OSError, json.JSONDecodeError):
            status["metadata_error"] = "unreadable"
    return status


def safe_component(value: str, runner: Any) -> str:
    return runner.slug(value)


def job_paths(
    output_root: Path,
    model_type: str,
    group_id: str,
    evaluated_concept: str,
    runner: Any,
    single_target: str | None = None,
) -> tuple[str, Path, Path, Path]:
    concept_slug = safe_component(evaluated_concept, runner)
    if model_type == "original":
        relative = Path("original") / group_id / concept_slug
    elif model_type == "joint":
        relative = Path("joint") / group_id / concept_slug
    else:
        if single_target is None:
            raise PipelineError("single job is missing single_target")
        relative = Path("single") / group_id / safe_component(single_target, runner) / concept_slug
    job_id = relative.as_posix().replace("/", "__")
    return (
        job_id,
        output_root / "images" / relative,
        output_root / "manifests" / f"{job_id}.json",
        output_root / "evaluations" / "shards" / f"{job_id}.json",
    )


def ordered_rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return fingerprint({
        "rows": [
            {
                "case_number": row["case_number"],
                "prompt": row["prompt"],
                "class": row["class"],
                "evaluation_seed": row["evaluation_seed"],
            }
            for row in rows
        ]
    })


def make_job(
    *,
    output_root: Path,
    group: Mapping[str, Any],
    concept: str,
    rows: Sequence[Mapping[str, Any]],
    model_type: str,
    generation: Mapping[str, Any],
    base_model: str,
    runner: Any,
    edit_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    single_target = None
    checkpoint_path = None
    checkpoint_metadata_path = None
    target_concepts: list[str] = []
    if edit_run is not None:
        target_concepts = list(edit_run["target_concepts"])
        checkpoint_path = edit_run["checkpoint_path"]
        checkpoint_metadata_path = edit_run["metadata_path"]
        if model_type == "single":
            single_target = target_concepts[0]
    checkpoint_sha256 = None
    if checkpoint_metadata_path is not None and Path(checkpoint_metadata_path).is_file():
        try:
            checkpoint_sha256 = read_json(Path(checkpoint_metadata_path)).get("checkpoint_sha256")
        except PipelineError:
            checkpoint_sha256 = None
    job_id, image_dir, manifest_path, result_path = job_paths(
        output_root, model_type, group["id"], concept, runner, single_target
    )
    identity = {
        "schema_version": 1,
        "base_model": base_model,
        "model_type": model_type,
        "group_id": group["id"],
        "single_target": single_target,
        "target_concepts": target_concepts,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluated_concept": concept,
        "ordered_rows_sha256": ordered_rows_fingerprint(rows),
        "generation": dict(generation),
    }
    return {
        "job_id": job_id,
        "job_fingerprint": fingerprint(identity),
        "model_type": model_type,
        "group_id": group["id"],
        "group_targets": list(group["targets"]),
        "group_similar_non_targets": list(group["similar_non_targets"]),
        "single_target": single_target,
        "target_concepts": target_concepts,
        "evaluated_concept": concept,
        "prompt_count": len(rows),
        "rows": list(rows),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata_path": checkpoint_metadata_path,
        "image_dir": str(image_dir),
        "manifest_path": str(manifest_path),
        "result_path": str(result_path),
    }


def build_pipeline_plan(
    *,
    config_path: Path,
    dataset_override: Path | None,
    output_root: Path,
    coverage_mode: str,
    raw_groups: Sequence[str] | None,
    rows_per_concept: int | None = None,
    smoke: bool = False,
    smoke_single_target: str = "golden retriever",
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    config, runner = load_config(config_path)
    groups = selected_groups(config, raw_groups, runner)
    evaluation = config["evaluation"]
    if dataset_override is not None:
        dataset_path = dataset_override.resolve()
    elif coverage_mode == "derived":
        dataset_path = resolve_relative(
            evaluation["derived_dataset_csv"], config_path.parent
        )
    else:
        dataset_path = resolve_relative(evaluation["dataset_csv"], config_path.parent)
    dataset_hash = sha256(dataset_path)
    if coverage_mode == "derived" and dataset_hash != evaluation["derived_dataset"]["derived_dataset_sha256"]:
        raise PipelineError(
            "Derived dataset hash does not match config provenance: "
            f"{dataset_path}"
        )
    all_rows, by_class = load_dataset(dataset_path)
    expected_rows = int(evaluation["expected_rows_per_class"])
    coverage = dataset_coverage(groups, by_class, expected_rows)
    if coverage["invalid_row_counts"]:
        raise PipelineError(
            f"Classes with non-protocol row counts: {coverage['invalid_row_counts']}"
        )

    edit_plan = runner.build_plan(
        config, config_path, (HERE / "outputs").resolve(),
        {group["id"] for group in groups}, "both",
    )
    edit_runs = edit_plan["runs"]
    edit_lookup = {
        (run["group_id"], run["mode"], tuple(run["target_concepts"])): run
        for run in edit_runs
    }

    available_norms = {normalize(value) for value in coverage["available_classes"]}
    selected_rows_by_class: dict[str, list[dict[str, Any]]] = {}
    for concept_norm, rows in by_class.items():
        if concept_norm not in available_norms:
            continue
        ordered = sorted(rows, key=lambda row: row["case_number"])
        if rows_per_concept is not None:
            ordered = ordered[:rows_per_concept]
        selected_rows_by_class[concept_norm] = ordered

    jobs: list[dict[str, Any]] = []
    generation = evaluation["generation"]
    for group in groups:
        concepts = [
            concept for concept in group["concepts"]
            if normalize(concept) in selected_rows_by_class
        ]
        if smoke:
            target_match = next(
                (value for value in group["targets"] if normalize(value) == normalize(smoke_single_target)),
                None,
            )
            if target_match is None:
                raise PipelineError(
                    f"Smoke target {smoke_single_target!r} is not a target in group {group['id']}"
                )
            preserve = next(
                (value for value in group["similar_non_targets"] if normalize(value) in selected_rows_by_class),
                None,
            )
            if preserve is None:
                raise PipelineError(f"Smoke group {group['id']} has no available similar non-target")
            concepts = [target_match, preserve]
        for concept in concepts:
            jobs.append(make_job(
                output_root=output_root, group=group, concept=concept,
                rows=selected_rows_by_class[normalize(concept)], model_type="original",
                generation=generation, base_model=config["shared"]["base_model"],
                runner=runner,
            ))
        single_targets = group["targets"]
        if smoke:
            single_targets = [
                next(value for value in group["targets"] if normalize(value) == normalize(smoke_single_target))
            ]
        for target in single_targets:
            edit_run = edit_lookup[(group["id"], "single", (target,))]
            for concept in concepts:
                jobs.append(make_job(
                    output_root=output_root, group=group, concept=concept,
                    rows=selected_rows_by_class[normalize(concept)], model_type="single",
                    generation=generation, base_model=config["shared"]["base_model"],
                    runner=runner, edit_run=edit_run,
                ))
        joint_targets = tuple(group["targets"])
        joint_run = edit_lookup[(group["id"], "joint", joint_targets)]
        for concept in concepts:
            jobs.append(make_job(
                output_root=output_root, group=group, concept=concept,
                rows=selected_rows_by_class[normalize(concept)], model_type="joint",
                generation=generation, base_model=config["shared"]["base_model"],
                runner=runner, edit_run=joint_run,
            ))

    job_ids = [job["job_id"] for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise PipelineError("Resolved evaluation job ids collide")
    image_count = sum(job["prompt_count"] for job in jobs)
    coverage_complete = bool(coverage["complete"])
    execution_allowed = (
        bool(coverage["available_classes"])
        if coverage_mode == "partial"
        else coverage_complete
    )
    if coverage_mode == "derived" and coverage_complete:
        coverage_status = "derived"
    elif coverage_mode == "complete" and coverage_complete:
        coverage_status = "complete"
    else:
        coverage_status = "partial"
    if execution_allowed:
        block_reason = None
    elif coverage_mode == "derived":
        block_reason = "Derived coverage requires all 25 configured classes with exactly 500 rows each"
    else:
        block_reason = "Complete coverage requires all 25 official classes with exactly 500 rows each"
    plan = {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id", "confuse5_single_vs_joint"),
        "created_at": utc_now(),
        "coverage_mode": coverage_mode,
        "coverage_status": coverage_status,
        "execution_allowed": execution_allowed,
        "block_reason": block_reason,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "dataset_total_rows": len(all_rows),
        "dataset_provenance": (
            evaluation["derived_dataset"]
            if coverage_mode == "derived"
            else {
                "kind": "official_rows_available_in_repository"
                if coverage_mode == "partial"
                else "externally_supplied_official_complete_rows"
            }
        ),
        "coverage": coverage,
        "selected_groups": [group["id"] for group in groups],
        "generation": generation,
        "classifier": evaluation["classifier"],
        "image_counts": {
            "total": image_count,
            "original": sum(job["prompt_count"] for job in jobs if job["model_type"] == "original"),
            "single": sum(job["prompt_count"] for job in jobs if job["model_type"] == "single"),
            "joint": sum(job["prompt_count"] for job in jobs if job["model_type"] == "joint"),
            "peak_retained_images_with_purge": max((job["prompt_count"] for job in jobs), default=0),
        },
        "checkpoint_status": [checkpoint_status(run) for run in edit_runs],
        "jobs": jobs,
    }
    plan["plan_fingerprint"] = fingerprint({
        "coverage_mode": coverage_mode,
        "dataset_sha256": plan["dataset_sha256"],
        "jobs": [job["job_fingerprint"] for job in jobs],
        "classifier": plan["classifier"],
    })
    return plan, config, runner


def public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    rendered = dict(plan)
    rendered["jobs"] = [
        {key: value for key, value in job.items() if key != "rows"}
        for job in plan["jobs"]
    ]
    return rendered


def require_execution_allowed(plan: Mapping[str, Any]) -> None:
    if not plan["execution_allowed"]:
        missing = plan["coverage"]["missing_classes"]
        raise PipelineError(
            f"Evaluation plan is blocked: {plan['block_reason']}; missing={missing}"
        )


def require_image_confirmation(plan: Mapping[str, Any], confirmed: int | None) -> None:
    expected = int(plan["image_counts"]["total"])
    if confirmed != expected:
        raise PipelineError(
            f"Refusing to generate {expected} images without "
            f"--confirm-image-count {expected}; received {confirmed!r}"
        )


def verify_checkpoint(job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    if job["model_type"] == "original":
        raise PipelineError("Original jobs do not have checkpoints")
    checkpoint = Path(job["checkpoint_path"])
    metadata_path = Path(job["checkpoint_metadata_path"])
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise PipelineError(f"Missing checkpoint or metadata for {job['job_id']}")
    metadata = read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise PipelineError(f"Checkpoint metadata is not complete: {metadata_path}")
    actual_hash = sha256(checkpoint)
    if metadata.get("checkpoint_sha256") != actual_hash:
        raise PipelineError(f"Checkpoint hash mismatch: {checkpoint}")
    if metadata.get("group_id") != job["group_id"]:
        raise PipelineError(f"Checkpoint group mismatch: {checkpoint}")
    if [normalize(v) for v in metadata.get("target_concepts", [])] != [
        normalize(v) for v in job["target_concepts"]
    ]:
        raise PipelineError(f"Checkpoint target mismatch: {checkpoint}")
    return checkpoint, metadata


def image_path(job: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    return Path(job["image_dir"]) / (
        f"case-{int(row['case_number']):06d}_seed-{int(row['evaluation_seed'])}.png"
    )


def ensure_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PipelineError(f"Refusing path outside managed image root: {path}") from exc


def new_manifest(job: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "job_fingerprint": job["job_fingerprint"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "status": "planned",
        "model_type": job["model_type"],
        "group_id": job["group_id"],
        "single_target": job["single_target"],
        "target_concepts": job["target_concepts"],
        "evaluated_concept": job["evaluated_concept"],
        "checkpoint_path": job["checkpoint_path"],
        "checkpoint_sha256": job.get("checkpoint_sha256"),
        "items": [],
        "created_at": utc_now(),
    }


def load_matching_manifest(job: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(job["manifest_path"])
    if not path.exists():
        return None
    manifest = read_json(path)
    if manifest.get("job_fingerprint") != job["job_fingerprint"]:
        raise PipelineError(
            f"Manifest protocol collision at {path}; use explicit --overwrite"
        )
    return manifest


def remove_job_artifacts(job: Mapping[str, Any], output_root: Path) -> None:
    image_root = output_root / "images"
    image_dir = Path(job["image_dir"])
    ensure_within(image_dir, image_root)
    if image_dir.exists():
        for candidate in image_dir.iterdir():
            ensure_within(candidate, image_root)
            if candidate.is_file() and (candidate.suffix == ".png" or candidate.name.endswith(".png.tmp")):
                candidate.unlink()
    for raw in (job["manifest_path"], job["result_path"]):
        path = Path(raw)
        if path.exists():
            path.unlink()


class GenerationRuntime:
    def __init__(self, config: Mapping[str, Any]):
        import torch
        from diffusers import DiffusionPipeline
        from safetensors.torch import load_file

        self.torch = torch
        self.load_file = load_file
        generation = config["evaluation"]["generation"]
        dtype = getattr(torch, generation["dtype"])
        self.pipe = DiffusionPipeline.from_pretrained(
            config["shared"]["base_model"],
            torch_dtype=dtype,
            safety_checker=None,
        ).to(config["shared"]["oce"]["device"])
        self.pipe.set_progress_bar_config(disable=True)
        actual_scheduler = type(self.pipe.scheduler).__name__
        if actual_scheduler != generation["scheduler"]:
            raise PipelineError(
                f"Unexpected scheduler {actual_scheduler}; expected {generation['scheduler']}"
            )
        self.generation = generation
        self.device = config["shared"]["oce"]["device"]
        self.expected_keys = int(config["evaluation"]["expected_checkpoint_keys"])
        self.original_weights: dict[str, Any] | None = None
        self.active_checkpoint: str | None = None

    def activate(self, job: Mapping[str, Any]) -> None:
        checkpoint_path = job["checkpoint_path"]
        if self.active_checkpoint == checkpoint_path:
            return
        if self.original_weights is not None:
            incompatible = self.pipe.unet.load_state_dict(self.original_weights, strict=False)
            if incompatible.unexpected_keys:
                raise PipelineError(f"Unexpected keys while restoring original UNet: {incompatible.unexpected_keys}")
        if checkpoint_path is None:
            self.active_checkpoint = None
            return
        checkpoint, _ = verify_checkpoint(job)
        state = dict(self.load_file(str(checkpoint), device="cpu"))
        if len(state) != self.expected_keys:
            raise PipelineError(
                f"Checkpoint {checkpoint} has {len(state)} keys; expected {self.expected_keys}"
            )
        unet_state = self.pipe.unet.state_dict()
        unknown = sorted(set(state) - set(unet_state))
        if unknown:
            raise PipelineError(f"Checkpoint has keys absent from UNet: {unknown}")
        if any(not key.endswith("attn2.to_v.weight") for key in state):
            raise PipelineError(f"Checkpoint contains a non-attn2.to_v weight: {checkpoint}")
        if self.original_weights is None:
            self.original_weights = {
                key: unet_state[key].detach().cpu().clone() for key in state
            }
        elif set(state) != set(self.original_weights):
            raise PipelineError(
                f"Checkpoint edited-key set differs from the first verified checkpoint: {checkpoint}"
            )
        incompatible = self.pipe.unet.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise PipelineError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
        self.active_checkpoint = checkpoint_path

    def generate_one(self, prompt: str, seed: int) -> Any:
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        with self.torch.inference_mode():
            return self.pipe(
                prompt=prompt,
                num_inference_steps=int(self.generation["num_inference_steps"]),
                guidance_scale=float(self.generation["guidance_scale"]),
                height=int(self.generation["height"]),
                width=int(self.generation["width"]),
                num_images_per_prompt=1,
                generator=generator,
            ).images[0]


class EvaluationRuntime:
    def __init__(self, config: Mapping[str, Any]):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        self.torch = torch
        self.device = config["shared"]["oce"]["device"]
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights).to(self.device).eval()
        self.preprocess = weights.transforms()
        self.categories = list(weights.meta["categories"])
        self.category_lookup: dict[str, list[int]] = {}
        for index, category in enumerate(self.categories):
            self.category_lookup.setdefault(normalize(category), []).append(index)
        self.batch_size = int(config["evaluation"]["classifier"]["batch_size"])

    def expected_index(self, concept: str) -> int:
        matches = self.category_lookup.get(normalize(concept), [])
        if len(matches) != 1:
            raise PipelineError(
                f"Concept {concept!r} resolves to {len(matches)} ImageNet categories; expected exactly one"
            )
        return matches[0]

    def classify(self, paths: Sequence[Path], concept: str) -> list[dict[str, Any]]:
        from PIL import Image

        expected = self.expected_index(concept)
        results: list[dict[str, Any]] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start:start + self.batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(self.preprocess(image.convert("RGB")))
            inputs = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                predictions = self.model(inputs).argmax(dim=-1).detach().cpu().tolist()
            for path, predicted in zip(batch_paths, predictions):
                results.append({
                    "image_path": str(path),
                    "expected_index": expected,
                    "expected_category": self.categories[expected],
                    "predicted_index": int(predicted),
                    "predicted_category": self.categories[int(predicted)],
                    "correct": int(predicted) == expected,
                })
        return results


def generate_job(
    job: Mapping[str, Any], plan: Mapping[str, Any], runtime: GenerationRuntime,
    *, output_root: Path, skip_existing: bool, overwrite: bool,
) -> dict[str, Any]:
    manifest_path = Path(job["manifest_path"])
    result_path = Path(job["result_path"])
    if overwrite:
        remove_job_artifacts(job, output_root)
    manifest = load_matching_manifest(job)
    if manifest is not None and not skip_existing and not overwrite:
        raise PipelineError(
            f"Generation manifest already exists at {manifest_path}; "
            "use --skip-existing or explicit --overwrite"
        )
    if result_path.is_file():
        result = read_json(result_path)
        if result.get("job_fingerprint") == job["job_fingerprint"] and skip_existing:
            print(f"[skip evaluated] {job['job_id']}", flush=True)
            return manifest or new_manifest(job, plan)
        if not overwrite:
            raise PipelineError(f"Result collision at {result_path}")
    if manifest is None:
        manifest = new_manifest(job, plan)
    if manifest.get("status") in {"evaluated", "purged"} and skip_existing:
        print(f"[skip complete] {job['job_id']}", flush=True)
        return manifest

    runtime.activate(job)
    image_dir = Path(job["image_dir"])
    image_dir.mkdir(parents=True, exist_ok=True)
    recorded = {int(item["case_number"]): item for item in manifest.get("items", [])}
    manifest["status"] = "generating"
    manifest["generation_started_at"] = manifest.get("generation_started_at", utc_now())
    write_json_atomic(manifest_path, manifest)
    for index, row in enumerate(job["rows"], start=1):
        case_number = int(row["case_number"])
        destination = image_path(job, row)
        existing = recorded.get(case_number)
        if existing is not None and existing.get("image_status") == "generated" and destination.is_file():
            if sha256(destination) == existing.get("image_sha256"):
                continue
            raise PipelineError(f"Existing image hash mismatch: {destination}")
        if destination.exists():
            raise PipelineError(f"Untracked image collision at {destination}")
        image = runtime.generate_one(row["prompt"], int(row["evaluation_seed"]))
        temporary = destination.with_suffix(".png.tmp")
        image.save(temporary, format="PNG")
        image_hash = sha256(temporary)
        item = {
            "case_number": case_number,
            "source_line": row["source_line"],
            "prompt": row["prompt"],
            "evaluated_concept": row["class"],
            "seed": int(row["evaluation_seed"]),
            "image_path": str(destination),
            "image_sha256": image_hash,
            "width": int(plan["generation"]["width"]),
            "height": int(plan["generation"]["height"]),
            "image_status": "generated",
            "generated_at": utc_now(),
        }
        recorded[case_number] = item
        manifest["items"] = [recorded[key] for key in sorted(recorded)]
        write_json_atomic(manifest_path, manifest)
        temporary.replace(destination)
        if index % 25 == 0 or index == len(job["rows"]):
            print(f"[generate] {job['job_id']} {index}/{len(job['rows'])}", flush=True)
    manifest["status"] = "generated"
    manifest["generation_finished_at"] = utc_now()
    write_json_atomic(manifest_path, manifest)
    return manifest


def evaluate_job(
    job: Mapping[str, Any], plan: Mapping[str, Any], runtime: EvaluationRuntime,
    *, output_root: Path, skip_existing: bool, overwrite: bool, purge: bool,
) -> dict[str, Any]:
    manifest_path = Path(job["manifest_path"])
    result_path = Path(job["result_path"])
    existing_result: dict[str, Any] | None = None
    if result_path.is_file() and not overwrite:
        result = read_json(result_path)
        if result.get("job_fingerprint") != job["job_fingerprint"]:
            raise PipelineError(f"Evaluation result protocol collision at {result_path}")
        if skip_existing and (not purge or result.get("images_purged")):
            print(f"[skip evaluation] {job['job_id']}", flush=True)
            return result
        if skip_existing and purge:
            existing_result = result
        else:
            raise PipelineError(f"Evaluation result already exists: {result_path}")
    manifest = load_matching_manifest(job)
    if manifest is None or manifest.get("status") not in {
        "generated", "evaluated", "purged", "purge_failed",
    }:
        raise PipelineError(f"Job has no complete generated manifest: {job['job_id']}")
    if manifest.get("status") == "purged" and result_path.is_file():
        return read_json(result_path)
    items = manifest.get("items", [])
    if len(items) != job["prompt_count"]:
        raise PipelineError(
            f"Manifest item count mismatch for {job['job_id']}: {len(items)} != {job['prompt_count']}"
        )
    paths = [Path(item["image_path"]) for item in items]
    if existing_result is not None:
        purged_at = utc_now()
        try:
            for path, item in zip(paths, items):
                ensure_within(path, output_root / "images")
                if path.exists():
                    if not path.is_file() or sha256(path) != item.get("image_sha256"):
                        raise PipelineError(f"Changed image during purge resume: {path}")
                    path.unlink()
        except OSError as exc:
            manifest["status"] = "purge_failed"
            manifest["purge_error"] = str(exc)
            write_json_atomic(manifest_path, manifest)
            raise PipelineError(f"Failed to purge evaluated job {job['job_id']}: {exc}") from exc
        for item in manifest["items"]:
            item["image_status"] = "purged"
            item["purged_at"] = purged_at
        manifest["status"] = "purged"
        manifest["purged_at"] = purged_at
        existing_result["images_purged"] = True
        existing_result["purged_at"] = purged_at
        write_json_atomic(result_path, existing_result)
        write_json_atomic(manifest_path, manifest)
        print(f"[purge resumed] {job['job_id']}", flush=True)
        return existing_result
    for path, item in zip(paths, items):
        ensure_within(path, output_root / "images")
        if not path.is_file() or sha256(path) != item.get("image_sha256"):
            raise PipelineError(f"Missing or changed image: {path}")
    predictions = runtime.classify(paths, job["evaluated_concept"])
    by_path = {item["image_path"]: item for item in predictions}
    per_image = []
    for item in items:
        prediction = by_path[item["image_path"]]
        per_image.append({
            "case_number": item["case_number"],
            "prompt": item["prompt"],
            "seed": item["seed"],
            "image_path": item["image_path"],
            "image_sha256": item["image_sha256"],
            **{key: prediction[key] for key in (
                "expected_index", "expected_category", "predicted_index",
                "predicted_category", "correct",
            )},
        })
    correct = sum(1 for item in per_image if item["correct"])
    result = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "job_fingerprint": job["job_fingerprint"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "model_type": job["model_type"],
        "group_id": job["group_id"],
        "single_target": job["single_target"],
        "target_concepts": job["target_concepts"],
        "evaluated_concept": job["evaluated_concept"],
        "checkpoint_path": job["checkpoint_path"],
        "checkpoint_sha256": job.get("checkpoint_sha256"),
        "correct": correct,
        "total": len(per_image),
        "accuracy": correct / max(len(per_image), 1),
        "classifier": plan["classifier"],
        "evaluated_at": utc_now(),
        "images_purged": False,
        "items": per_image,
    }
    write_json_atomic(result_path, result)
    manifest["status"] = "evaluated"
    manifest["evaluated_at"] = result["evaluated_at"]
    write_json_atomic(manifest_path, manifest)
    if purge:
        try:
            for path in paths:
                ensure_within(path, output_root / "images")
                path.unlink()
        except OSError as exc:
            manifest["status"] = "purge_failed"
            manifest["purge_error"] = str(exc)
            write_json_atomic(manifest_path, manifest)
            raise PipelineError(f"Failed to purge evaluated job {job['job_id']}: {exc}") from exc
        purged_at = utc_now()
        for item in manifest["items"]:
            item["image_status"] = "purged"
            item["purged_at"] = purged_at
        manifest["status"] = "purged"
        manifest["purged_at"] = purged_at
        result["images_purged"] = True
        result["purged_at"] = purged_at
        write_json_atomic(result_path, result)
        write_json_atomic(manifest_path, manifest)
    print(
        f"[evaluate] {job['job_id']} accuracy={result['accuracy']:.4f} "
        f"purged={result['images_purged']}", flush=True,
    )
    return result


def mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    return sum(collected) / len(collected) if collected else None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate(plan: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        path = Path(job["result_path"])
        if not path.is_file():
            raise PipelineError(f"Missing evaluation result: {path}")
        result = read_json(path)
        if result.get("job_fingerprint") != job["job_fingerprint"]:
            raise PipelineError(f"Evaluation fingerprint mismatch: {path}")
        results.append(result)
    lookup = {
        (r["group_id"], r["model_type"], r.get("single_target"), normalize(r["evaluated_concept"])): r
        for r in results
    }
    per_class: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for result in results:
        group_targets = next(
            job["group_targets"] for job in plan["jobs"] if job["job_id"] == result["job_id"]
        )
        concept_norm = normalize(result["evaluated_concept"])
        target_norms = {normalize(value) for value in group_targets}
        if result["model_type"] == "original":
            role = "baseline"
        elif result["model_type"] == "joint":
            role = "target" if concept_norm in target_norms else "similar_non_target"
        elif concept_norm == normalize(result["single_target"]):
            role = "target"
        elif concept_norm in target_norms:
            role = "sibling_target"
        else:
            role = "similar_non_target"
        per_class.append({
            "coverage_status": plan["coverage_status"],
            "group_id": result["group_id"],
            "model_type": result["model_type"],
            "single_target": result.get("single_target") or "",
            "evaluated_concept": result["evaluated_concept"],
            "role": role,
            "correct": result["correct"],
            "total": result["total"],
            "accuracy": result["accuracy"],
        })
        for item in result["items"]:
            per_image.append({
                "group_id": result["group_id"],
                "model_type": result["model_type"],
                "single_target": result.get("single_target") or "",
                "evaluated_concept": result["evaluated_concept"],
                **item,
            })

    comparisons: list[dict[str, Any]] = []
    group_ids = list(dict.fromkeys(job["group_id"] for job in plan["jobs"]))
    for group_id in group_ids:
        group_job = next(job for job in plan["jobs"] if job["group_id"] == group_id)
        available = {
            normalize(job["evaluated_concept"]): job["evaluated_concept"]
            for job in plan["jobs"] if job["group_id"] == group_id
        }
        preserve = [
            value for value in group_job["group_similar_non_targets"]
            if normalize(value) in available
        ]
        comparison_targets = list(dict.fromkeys(
            result["single_target"] for result in results
            if result["group_id"] == group_id
            and result["model_type"] == "single"
            and result.get("single_target")
        ))
        for target in comparison_targets:
            target_norm = normalize(target)
            original = lookup[(group_id, "original", None, target_norm)]["accuracy"]
            single = lookup[(group_id, "single", target, target_norm)]["accuracy"]
            joint = lookup[(group_id, "joint", None, target_norm)]["accuracy"]
            sibling = next(
                (value for value in group_job["group_targets"] if normalize(value) != target_norm),
                None,
            )
            sibling_accuracy = None
            if sibling is not None and normalize(sibling) in available:
                sibling_accuracy = lookup[(group_id, "single", target, normalize(sibling))]["accuracy"]
            original_preserve = mean(
                lookup[(group_id, "original", None, normalize(value))]["accuracy"] for value in preserve
            )
            single_preserve = mean(
                lookup[(group_id, "single", target, normalize(value))]["accuracy"] for value in preserve
            )
            joint_preserve = mean(
                lookup[(group_id, "joint", None, normalize(value))]["accuracy"] for value in preserve
            )
            comparisons.append({
                "coverage_status": plan["coverage_status"],
                "group_id": group_id,
                "target": target,
                "available_similar_non_targets": ";".join(preserve),
                "available_similar_non_target_count": len(preserve),
                "required_similar_non_target_count": len(group_job["group_similar_non_targets"]),
                "original_target_accuracy": original,
                "single_target_accuracy": single,
                "joint_target_accuracy": joint,
                "single_minus_original_target": single - original,
                "joint_minus_original_target": joint - original,
                "joint_minus_single_target": joint - single,
                "original_preservation_accuracy": original_preserve,
                "single_preservation_accuracy": single_preserve,
                "joint_preservation_accuracy": joint_preserve,
                "single_minus_original_preservation": (
                    None if original_preserve is None or single_preserve is None else single_preserve - original_preserve
                ),
                "joint_minus_original_preservation": (
                    None if original_preserve is None or joint_preserve is None else joint_preserve - original_preserve
                ),
                "joint_minus_single_preservation": (
                    None if single_preserve is None or joint_preserve is None else joint_preserve - single_preserve
                ),
                "sibling_target": sibling,
                "single_sibling_target_preservation_accuracy": sibling_accuracy,
            })

    eval_root = output_root / "evaluations"
    aggregate_root = output_root / "aggregates"
    write_csv(eval_root / "per_class.csv", per_class, [
        "coverage_status", "group_id", "model_type", "single_target",
        "evaluated_concept", "role", "correct", "total", "accuracy",
    ])
    write_csv(eval_root / "per_image.csv", per_image, [
        "group_id", "model_type", "single_target", "evaluated_concept",
        "case_number", "prompt", "seed", "image_path", "image_sha256",
        "expected_index", "expected_category", "predicted_index",
        "predicted_category", "correct",
    ])
    comparison_fields = list(comparisons[0]) if comparisons else []
    write_csv(aggregate_root / "all_groups.csv", comparisons, comparison_fields)
    for group_id in group_ids:
        write_csv(
            aggregate_root / "groups" / f"{group_id}.csv",
            [row for row in comparisons if row["group_id"] == group_id],
            comparison_fields,
        )
    summary = {
        "schema_version": 1,
        "coverage_status": plan["coverage_status"],
        "dataset_path": plan["dataset_path"],
        "dataset_sha256": plan["dataset_sha256"],
        "dataset_provenance": plan["dataset_provenance"],
        "missing_classes": plan["coverage"]["missing_classes"],
        "generated_image_count": plan["image_counts"]["total"],
        "comparison_rows": comparisons,
        "created_at": utc_now(),
        "interpretation": "Measured values and differences only; no automatic success/failure claim.",
    }
    write_json_atomic(aggregate_root / "summary.json", summary)
    return summary


def write_run_state(output_root: Path, stage: str, status: str, **extra: Any) -> None:
    path = output_root / "run_state.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            previous = read_json(path)
        except PipelineError:
            previous = {}
    payload = {
        **previous,
        "stage": stage,
        "status": status,
        "updated_at": utc_now(),
        **extra,
    }
    write_json_atomic(path, payload)


def update_job_progress(output_root: Path, **progress: Any) -> None:
    path = output_root / "run_state.json"
    state = read_json(path) if path.is_file() else {}
    state.update(progress)
    state["updated_at"] = utc_now()
    write_json_atomic(path, state)


def execute_edit(args: argparse.Namespace) -> None:
    command = [
        sys.executable, str(EDIT_RUNNER_PATH), "--config", str(args.config),
        "--mode", "both",
    ]
    if args.groups:
        command.extend(["--groups", *args.groups])
    if args.skip_existing:
        command.append("--skip-completed")
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_jobs(
    plan: Mapping[str, Any], config: Mapping[str, Any], output_root: Path,
    *, do_generate: bool, do_evaluate: bool, skip_existing: bool,
    overwrite: bool, purge: bool,
) -> None:
    generation_runtime = GenerationRuntime(config) if do_generate else None
    evaluation_runtime = EvaluationRuntime(config) if do_evaluate else None
    events = output_root / "logs" / "events.jsonl"
    for index, job in enumerate(plan["jobs"], start=1):
        update_job_progress(
            output_root, current_job=job["job_id"], job_index=index,
            total_jobs=len(plan["jobs"]), completed_jobs=index - 1,
        )
        append_event(events, {
            "at": utc_now(), "event": "job_started", "job_id": job["job_id"],
            "index": index, "total_jobs": len(plan["jobs"]),
        })
        try:
            if do_generate:
                assert generation_runtime is not None
                generate_job(
                    job, plan, generation_runtime, output_root=output_root,
                    skip_existing=skip_existing, overwrite=overwrite,
                )
            if do_evaluate:
                assert evaluation_runtime is not None
                evaluate_job(
                    job, plan, evaluation_runtime, output_root=output_root,
                    skip_existing=skip_existing, overwrite=overwrite, purge=purge,
                )
        except Exception as exc:
            append_event(events, {
                "at": utc_now(), "event": "job_failed", "job_id": job["job_id"],
                "error": repr(exc),
            })
            raise
        append_event(events, {
            "at": utc_now(), "event": "job_finished", "job_id": job["job_id"],
            "index": index, "total_jobs": len(plan["jobs"]),
        })
        update_job_progress(
            output_root, current_job=job["job_id"], job_index=index,
            total_jobs=len(plan["jobs"]), completed_jobs=index,
        )


def status_report(root: Path) -> dict[str, Any]:
    manifests = list((root / "manifests").glob("*.json")) if (root / "manifests").exists() else []
    counts: dict[str, int] = {}
    for path in manifests:
        try:
            status = read_json(path).get("status", "unknown")
        except PipelineError:
            status = "unreadable"
        counts[status] = counts.get(status, 0) + 1
    pngs = list((root / "images").rglob("*.png")) if (root / "images").exists() else []
    retained_bytes = sum(path.stat().st_size for path in pngs if path.is_file())
    state = read_json(root / "run_state.json") if (root / "run_state.json").is_file() else None
    return {
        "output_root": str(root),
        "run_state": state,
        "manifest_statuses": counts,
        "retained_png_count": len(pngs),
        "retained_bytes": retained_bytes,
        "retained_gib": retained_bytes / (1024 ** 3),
    }


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-csv", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--coverage", choices=("partial", "derived", "complete"), default="complete"
    )
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--plan-path", type=Path)


def add_collision_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-existing", action="store_true")
    group.add_argument("--overwrite", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    plan_parser = subparsers.add_parser("plan")
    add_common(plan_parser)

    edit_parser = subparsers.add_parser("edit")
    edit_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    edit_parser.add_argument("--groups", nargs="+")
    add_collision_flags(edit_parser)

    for stage in ("generate", "evaluate", "aggregate", "all"):
        stage_parser = subparsers.add_parser(stage)
        add_common(stage_parser)
        add_collision_flags(stage_parser)
        if stage in {"generate", "all"}:
            stage_parser.add_argument("--confirm-image-count", type=int)
        if stage == "generate":
            stage_parser.add_argument("--retain-images", action="store_true")
        if stage in {"evaluate", "all"}:
            retention = stage_parser.add_mutually_exclusive_group()
            retention.add_argument("--purge-evaluated-images", action="store_true")
            retention.add_argument("--keep-images", action="store_true")
        if stage == "all":
            stage_parser.add_argument(
                "--start-at", choices=("edit", "generate", "evaluate", "aggregate"),
                default="edit",
            )

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke_parser.add_argument("--dataset-csv", type=Path)
    smoke_parser.add_argument("--output-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    smoke_parser.add_argument("--group", default="dogs")
    smoke_parser.add_argument("--single-target", default="golden retriever")
    smoke_parser.add_argument("--rows-per-concept", type=int, choices=(1, 2), default=2)
    retention = smoke_parser.add_mutually_exclusive_group()
    retention.add_argument("--purge-evaluated-images", action="store_true")
    retention.add_argument("--keep-images", action="store_true")
    add_collision_flags(smoke_parser)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "status":
        print(json.dumps(status_report(args.output_root.resolve()), indent=2, ensure_ascii=False))
        return 0
    if args.stage == "edit":
        execute_edit(args)
        return 0

    smoke = args.stage == "smoke"
    try:
        plan, config, _ = build_pipeline_plan(
            config_path=args.config.resolve(),
            dataset_override=args.dataset_csv,
            output_root=args.output_root.resolve(),
            coverage_mode="partial" if smoke else args.coverage,
            raw_groups=[args.group] if smoke else args.groups,
            rows_per_concept=args.rows_per_concept if smoke else None,
            smoke=smoke,
            smoke_single_target=args.single_target if smoke else "golden retriever",
        )
        rendered_plan = public_plan(plan)
        if getattr(args, "plan_path", None):
            write_json_atomic(args.plan_path, rendered_plan)
        if args.stage == "plan":
            print(json.dumps(rendered_plan, indent=2, ensure_ascii=False))
            return 0 if plan["execution_allowed"] else 2
        require_execution_allowed(plan)
        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_root / "resolved_plan.json", rendered_plan)
        write_run_state(
            output_root, args.stage, "running",
            plan_fingerprint=plan["plan_fingerprint"],
            image_counts=plan["image_counts"],
        )
        if smoke:
            purge = not args.keep_images
            run_jobs(
                plan, config, output_root, do_generate=True, do_evaluate=True,
                skip_existing=args.skip_existing, overwrite=args.overwrite, purge=purge,
            )
            aggregate(plan, output_root)
            # A second pass proves that evaluated/purged jobs are terminal and
            # are safely skipped without regenerating images.
            run_jobs(
                plan, config, output_root, do_generate=True, do_evaluate=True,
                skip_existing=True, overwrite=False, purge=purge,
            )
        elif args.stage == "generate":
            if not args.retain_images:
                raise PipelineError(
                    "generate-only cannot purge before evaluation; pass --retain-images "
                    "or use the low-disk all stage"
                )
            require_image_confirmation(plan, args.confirm_image_count)
            run_jobs(
                plan, config, output_root, do_generate=True, do_evaluate=False,
                skip_existing=args.skip_existing, overwrite=args.overwrite, purge=False,
            )
        elif args.stage == "evaluate":
            purge = not args.keep_images
            run_jobs(
                plan, config, output_root, do_generate=False, do_evaluate=True,
                skip_existing=args.skip_existing, overwrite=args.overwrite, purge=purge,
            )
        elif args.stage == "aggregate":
            aggregate(plan, output_root)
        elif args.stage == "all":
            purge = not args.keep_images
            order = {"edit": 0, "generate": 1, "evaluate": 2, "aggregate": 3}
            start = order[args.start_at]
            if start <= 0:
                execute_edit(args)
                # Re-resolve checkpoint hashes after editing so evaluation
                # fingerprints bind to the actual saved artifacts.
                plan, config, _ = build_pipeline_plan(
                    config_path=args.config.resolve(),
                    dataset_override=args.dataset_csv,
                    output_root=args.output_root.resolve(),
                    coverage_mode=args.coverage,
                    raw_groups=args.groups,
                )
                rendered_plan = public_plan(plan)
                write_json_atomic(output_root / "resolved_plan.json", rendered_plan)
                write_run_state(
                    output_root, args.stage, "running",
                    plan_fingerprint=plan["plan_fingerprint"],
                    image_counts=plan["image_counts"],
                )
            if start <= 1:
                require_image_confirmation(plan, args.confirm_image_count)
                run_jobs(
                    plan, config, output_root, do_generate=True, do_evaluate=True,
                    skip_existing=args.skip_existing, overwrite=args.overwrite, purge=purge,
                )
            elif start == 2:
                run_jobs(
                    plan, config, output_root, do_generate=False, do_evaluate=True,
                    skip_existing=args.skip_existing, overwrite=args.overwrite, purge=purge,
                )
            if start <= 3:
                aggregate(plan, output_root)
        write_run_state(output_root, args.stage, "complete", finished_at=utc_now())
        return 0
    except Exception as exc:
        if hasattr(args, "output_root"):
            try:
                write_run_state(args.output_root.resolve(), args.stage, "failed", error=str(exc))
            except OSError:
                pass
        print(f"pipeline error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
