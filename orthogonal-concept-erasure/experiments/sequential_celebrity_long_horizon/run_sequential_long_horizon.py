#!/usr/bin/env python
"""Resumable 10x10 long-horizon sequential OCE celebrity experiment.

The formal run is intentionally restricted to two fixed orders and two fixed
conditions.  A non-scoring throughput benchmark locks either profile_5 or
profile_10 before formal generation.  The runner never chooses concepts,
orders, checkpoints, or metrics from observed experimental results.
"""

from __future__ import annotations

import argparse
import ast
import csv
import concurrent.futures
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
OCE_ROOT = HERE.parents[1]
PROJECT_ROOT = OCE_ROOT.parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT = HERE / "outputs" / "sequential_oce_celebrity_long_horizon_v1"
OCE_SOURCE = OCE_ROOT / "oce.py"
CELEB_SOURCE = OCE_ROOT / "generate_celeb.py"
EVAL_SOURCE = OCE_ROOT / "metrics" / "eval_celeb.py"
SHELL_RUNNER = HERE / "run_sequential_long_horizon.sh"
AUDIT_SOURCE = HERE / "audit_results.rb"
WATCHDOG_SOURCE = HERE / "lightning_studio_watchdog.sh"
H100_CONTROLLER_SOURCE = HERE / "lightning_h100_controller.sh"
E10_TRAIN_SOURCE = OCE_ROOT / "trainscripts" / "celeb_10.sh"
E100_TRAIN_SOURCE = OCE_ROOT / "trainscripts" / "celeb_100.sh"
REFERENCE_ROOT = OCE_ROOT / "experiments" / "evaluation_references"
COCO_SOURCE = OCE_ROOT / "data" / "coco_30k.csv"
MILESTONE_STEPS = (1, 5, 10)
ORDERS = ("order_a", "order_b")
CONDITIONS = ("baseline", "retain_history")
PROFILES = ("profile_5", "profile_10")

_GCD_WORKER_DETECTOR: Any = None
_GCD_WORKER_PREPROCESS: Any = None
_GCD_WORKER_IMAGE_SIZE: int = 224


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for file_path in sorted(candidate for candidate in path.glob("**/*") if candidate.is_file()):
        relative = str(file_path.relative_to(path))
        size = file_path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        digest.update(sha256_file(file_path).encode("ascii"))
        file_count += 1
        total_bytes += size
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def slug(text: str) -> str:
    return "_".join(text.casefold().replace("-", " ").split())


def package_versions() -> dict[str, str]:
    result = {}
    for name in (
        "torch", "torchvision", "diffusers", "transformers", "huggingface-hub",
        "safetensors", "Pillow", "pandas", "scikit-image", "python-dotenv",
        "tensorflow-cpu",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def gcd_worker_count(config: Mapping[str, Any]) -> int:
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    requested = max(1, int(math.floor(available * float(config["gcd"]["cpu_fraction"]))))
    return min(int(config["gcd"]["max_workers"]), requested)


def initialize_gcd_detector_worker(
    root: str, resources: str, margin: float, image_size: int,
) -> None:
    """Initialize one official CPU face detector per spawned worker."""
    global _GCD_WORKER_DETECTOR, _GCD_WORKER_PREPROCESS, _GCD_WORKER_IMAGE_SIZE

    os.environ["APP_DATA_DIR"] = resources
    os.environ["APP_USE_CUDA"] = "false"
    # APP_USE_CUDA=false is the official detector setting.  Hide the H100 from
    # detector workers before TensorFlow import so each CPU worker cannot open
    # its own CUDA context; the parent GCD recognizer still uses the H100.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    if root not in sys.path:
        sys.path.insert(0, root)
    from model_training.preprocessors.face_detection.face_detector import FaceDetector
    from model_training.utils import preprocess_image

    _GCD_WORKER_DETECTOR = FaceDetector(resources, margin=margin, use_cuda=False)
    _GCD_WORKER_PREPROCESS = preprocess_image
    _GCD_WORKER_IMAGE_SIZE = image_size


def detect_faces_worker(task: tuple[int, str]) -> tuple[int, list[Any]]:
    """Run the unmodified official detector and preprocessing for one image."""
    from skimage import io

    if _GCD_WORKER_DETECTOR is None or _GCD_WORKER_PREPROCESS is None:
        raise RuntimeError("GCD detector worker was not initialized")
    index, path = task
    detected = _GCD_WORKER_DETECTOR.perform_single(io.imread(path))
    faces = [
        _GCD_WORKER_PREPROCESS(image, _GCD_WORKER_IMAGE_SIZE)
        for image, _ in detected
    ]
    return index, faces


def git_capture(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return completed.stdout.strip()


def literal_list(path: Path, assignment: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == assignment for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise TypeError(f"{assignment} is not list[str]")
        return value
    raise KeyError(f"Missing {assignment} in {path}")


def shell_assignment(path: Path, name: str) -> str:
    pattern = re.compile(rf"^{re.escape(name)}=(?:\"([^\"]*)\"|'([^']*)'|([^#\s]+))\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            return next(value for value in match.groups() if value is not None)
    raise KeyError(f"Missing {name} in {path}")


def semicolon_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def load_config(path: Path) -> dict[str, Any]:
    return read_json(path.resolve())


def order_concepts(config: Mapping[str, Any], order: str) -> list[str]:
    targets = list(config["targets"])
    if order == "order_a":
        return targets
    if order == "order_b":
        return list(reversed(targets))
    raise ValueError(f"Unknown order: {order}")


def concept_batches(config: Mapping[str, Any], order: str) -> list[list[str]]:
    concepts = order_concepts(config, order)
    size = int(config["batch_size_concepts"])
    return [concepts[index:index + size] for index in range(0, len(concepts), size)]


def validate_config(config_path: Path, config: Mapping[str, Any], artifacts: bool) -> None:
    if config["experiment_name"] != "sequential_oce_celebrity_long_horizon_v1":
        raise ValueError("Unexpected experiment name")
    if config["model_id"] != "CompVis/stable-diffusion-v1-4":
        raise ValueError("Base model must remain Stable Diffusion v1.4")
    targets = list(config["targets"])
    retains = list(config["fixed_retains"])
    if len(targets) != 100 or len(set(targets)) != 100:
        raise ValueError("Target list must contain 100 unique celebrities")
    if len(retains) != 100 or len(set(retains)) != 100:
        raise ValueError("Retain list must contain 100 unique celebrities")
    if set(targets) & set(retains):
        raise ValueError("Target and retain sets must be disjoint")
    repository_targets = literal_list(CELEB_SOURCE, "E100_LIST")
    repository_retains = literal_list(CELEB_SOURCE, "PRESERVE_LIST")
    if targets != repository_targets or retains != repository_retains:
        raise ValueError("Config celebrity lists differ from current repository lists")
    if semicolon_list(shell_assignment(E10_TRAIN_SOURCE, "EDIT_CONCEPTS")) != targets[:10]:
        raise ValueError("Repository E10 train-script targets differ from its first ten targets")
    if semicolon_list(shell_assignment(E100_TRAIN_SOURCE, "EDIT_CONCEPTS")) != targets:
        raise ValueError("Repository E100 train-script targets differ from E100_LIST")
    for path in (E10_TRAIN_SOURCE, E100_TRAIN_SOURCE):
        if shell_assignment(path, "MODEL_ID") != config["model_id"]:
            raise ValueError(f"Model ID mismatch in {path}")
        if semicolon_list(shell_assignment(path, "PRESERVE_CONCEPTS")) != retains:
            raise ValueError(f"Retain list mismatch in {path}")
    if list(config["orders"]) != list(ORDERS) or list(config["conditions"]) != list(CONDITIONS):
        raise ValueError("Formal orders or conditions changed")
    for order in ORDERS:
        batches = concept_batches(config, order)
        if len(batches) != 10 or any(len(batch) != 10 for batch in batches):
            raise ValueError(f"{order} is not ten batches of ten")
    if order_concepts(config, "order_b") != list(reversed(order_concepts(config, "order_a"))):
        raise ValueError("Order B must be the exact reverse of Order A")
    expected_templates = [
        "a portrait of {}", "a sketch of {}", "an oil painting of {}",
        "{} in an official photo", "an image capturing {} at a public event",
    ]
    if list(config["templates"]) != expected_templates:
        raise ValueError("Celebrity templates differ from the repository evaluator protocol")
    e10 = dict(config["edit"]["e10"])
    expected_e10 = {
        "guide_concepts": ["person", "woman", "man"],
        "guide_alignment_seed": 42,
        "erase_scale": 3500.0,
        "preserve_global_scale": 50.0,
        "preserve_concept_scale": 5.0,
        "lamb": 10.0,
        "expand_prompts": False,
        "dtype": "float32",
    }
    if e10 != expected_e10:
        raise ValueError("Sequential edit settings differ from repository E10")
    if semicolon_list(shell_assignment(E10_TRAIN_SOURCE, "GUIDE_CONCEPTS")) != e10["guide_concepts"]:
        raise ValueError("Sequential guides differ from repository E10 train script")
    for shell_name, config_name in (
        ("ERASE_SCALE", "erase_scale"),
        ("PRESERVE_GLOBAL_SCALE", "preserve_global_scale"),
        ("PRESERVE_CONCEPT_SCALE", "preserve_concept_scale"),
        ("LAMB", "lamb"),
    ):
        if float(shell_assignment(E10_TRAIN_SOURCE, shell_name)) != float(e10[config_name]):
            raise ValueError(f"Sequential {config_name} differs from repository E10")
    e100 = dict(config["edit"]["joint_e100"])
    expected_e100 = {
        "guide_concepts": ["tree"], "guide_alignment_seed": 42,
        "erase_scale": 800.0, "preserve_global_scale": 70.0,
        "preserve_concept_scale": 2.0, "lamb": 10.0,
        "expand_prompts": False, "dtype": "float32",
    }
    if e100 != expected_e100:
        raise ValueError("Joint reference settings differ from repository E100")
    if semicolon_list(shell_assignment(E100_TRAIN_SOURCE, "GUIDE_CONCEPTS")) != e100["guide_concepts"]:
        raise ValueError("Joint guides differ from repository E100 train script")
    for shell_name, config_name in (
        ("ERASE_SCALE", "erase_scale"),
        ("PRESERVE_GLOBAL_SCALE", "preserve_global_scale"),
        ("PRESERVE_CONCEPT_SCALE", "preserve_concept_scale"),
        ("LAMB", "lamb"),
    ):
        if float(shell_assignment(E100_TRAIN_SOURCE, shell_name)) != float(e100[config_name]):
            raise ValueError(f"Joint {config_name} differs from repository E100")
    generation = config["generation"]
    if (
        int(generation["num_inference_steps"]) != 50
        or float(generation["guidance_scale"]) != 7.5
        or [int(generation["height"]), int(generation["width"])] != [512, 512]
        or generation["dtype"] != "bfloat16"
        or generation["scheduler"] != "PNDMScheduler"
    ):
        raise ValueError("Generation protocol changed")
    if list(generation["batch_size_candidates"]) != [8, 16, 24, 32]:
        raise ValueError("Frozen H100 batch-size candidates changed")
    if (
        int(config["budget"]["benchmark_images"]) != 200
        or int(config["budget"]["benchmark_images_per_candidate"]) != 50
    ):
        raise ValueError("Benchmark must remain four fixed 50-image trials")
    if (
        int(config["budget"]["hard_deadline_seconds"]) != 20700
        or float(config["budget"]["deadline_safety_fraction"]) != 0.85
    ):
        raise ValueError("Formal Lightning deadline must remain 20,700 seconds with 15% reserve")
    gcd = config["gcd"]
    if gcd != {
        "detector_parallelism": "ordered_spawn_process_pool",
        "max_workers": 16,
        "cpu_fraction": 0.5,
        "recognizer_top_n": 5,
    }:
        raise ValueError("Frozen official-GCD execution configuration changed")
    if config["budget"]["profile_image_counts"] != {
        "profile_5": 34900, "profile_10": 45820
    }:
        raise ValueError("Budget image-count authority changed")
    if config["budget"]["profile_prediction_counts"] != {
        "profile_5": 34800, "profile_10": 45800
    }:
        raise ValueError("Budget GCD prediction-count authority changed")
    if config["generation"]["paper_target_samples_per_prompt"] != {
        "1": 10, "5": 2, "10": 1
    }:
        raise ValueError("Paper-scale target sample counts changed")
    cg_path = (config_path.resolve().parent / str(config["cg_path"])).resolve()
    if cg_path != (OCE_ROOT / "Cg.pt").resolve():
        raise ValueError("Cg path must be the repository-level Cg.pt")
    if artifacts and not cg_path.is_file():
        raise FileNotFoundError(f"Missing Cg.pt: {cg_path}")


def output_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).resolve()


def state_path(output_dir: Path) -> Path:
    return output_dir / "run_state.json"


def update_state(output_dir: Path, **values: Any) -> None:
    path = state_path(output_dir)
    state = read_json(path) if path.is_file() else {"started_at": utc_now()}
    state.update(values)
    state["updated_at"] = utc_now()
    write_json(path, state)


def event(output_dir: Path, phase: str, message: str, **values: Any) -> None:
    row = {"timestamp": utc_now(), "phase": phase, "message": message, **values}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    suffix = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[{phase}] {message}{(' ' + suffix) if suffix else ''}", flush=True)


def resolve_snapshot(model_id: str, allow_downloads: bool) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(
        repo_id=model_id, local_files_only=not allow_downloads
    )).resolve()
    revision = path.name if path.parent.name == "snapshots" else "unresolved"
    return {"model_id": model_id, "snapshot_path": str(path), "revision": revision}


def resolve_gcd_root(config: Mapping[str, Any], override: Path | None) -> Path:
    raw = override or os.environ.get("GCD_PROJECT_ROOT") or config.get("gcd_project_root")
    if not raw:
        raise RuntimeError(
            "GCD project root is required. Pass --gcd-project-root or set GCD_PROJECT_ROOT."
        )
    root = Path(raw).resolve()
    if not (root / "model_training").is_dir():
        raise FileNotFoundError(f"No GCD model_training package under {root}")
    return root


def dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def validate_gcd_installation(root: Path) -> dict[str, Any]:
    required = {
        "labels.py": root / "model_training" / "helpers" / "labels.py",
        "face_recognizer.py": root / "model_training" / "helpers" / "face_recognizer.py",
        "face_detector.py": root / "model_training" / "preprocessors" / "face_detection" / "face_detector.py",
        "utils.py": root / "model_training" / "utils.py",
        ".env": root / ".env",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete GCD installation: {missing}")
    env_values = dotenv_values(required[".env"])
    raw_resources = os.environ.get("APP_DATA_DIR") or env_values.get("APP_DATA_DIR")
    if not raw_resources:
        raise RuntimeError("GCD .env/environment must define APP_DATA_DIR")
    resources = Path(raw_resources).expanduser()
    if not resources.is_absolute():
        resources = root / resources
    resources = resources.resolve()
    if not resources.is_dir() or not any(resources.iterdir()):
        raise FileNotFoundError(f"GCD APP_DATA_DIR is missing or empty: {resources}")
    return {
        "project_root": str(root),
        "app_data_dir": str(resources),
        "settings": {
            "APP_FACE_SIZE": os.environ.get("APP_FACE_SIZE") or env_values.get("APP_FACE_SIZE") or "224",
            "APP_FACE_MARGIN": os.environ.get("APP_FACE_MARGIN") or env_values.get("APP_FACE_MARGIN") or "0.2",
            "APP_USE_CUDA": os.environ.get("APP_USE_CUDA") or env_values.get("APP_USE_CUDA") or "false",
            "USE_CUDA": os.environ.get("USE_CUDA") or env_values.get("USE_CUDA") or "false",
        },
        "source_hashes": {name: sha256_file(path) for name, path in required.items()},
        "resource_fingerprint": directory_fingerprint(resources),
    }


def source_hashes(config_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "config.json": sha256_file(config_path),
        "oce.py": sha256_file(OCE_SOURCE),
        "generate_celeb.py": sha256_file(CELEB_SOURCE),
        "eval_celeb.py": sha256_file(EVAL_SOURCE),
        "runner": sha256_file(Path(__file__).resolve()),
        "shell_runner": sha256_file(SHELL_RUNNER),
        "independent_audit": sha256_file(AUDIT_SOURCE),
        "lightning_watchdog": sha256_file(WATCHDOG_SOURCE),
        "lightning_h100_controller": sha256_file(H100_CONTROLLER_SOURCE),
        "celeb_10.sh": sha256_file(E10_TRAIN_SOURCE),
        "celeb_100.sh": sha256_file(E100_TRAIN_SOURCE),
        "Cg.pt": sha256_file((config_path.parent / str(config["cg_path"])).resolve()),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    validate_config(config_path, config, artifacts=True)
    output_dir = output_path(args)
    if args.artifact_root is None:
        raise ValueError("--artifact-root must be a real absolute Lightning path")
    raw_artifact_root = Path(args.artifact_root).expanduser()
    if not raw_artifact_root.is_absolute() or raw_artifact_root == Path("/"):
        raise ValueError("Unsafe artifact root")
    artifact_root = raw_artifact_root.resolve()
    gcd_root = resolve_gcd_root(config, args.gcd_project_root)
    gcd_installation = validate_gcd_installation(gcd_root)
    required_packages = (
        "torch", "diffusers", "transformers", "safetensors", "PIL",
        "skimage", "huggingface_hub",
    )
    missing_packages = [
        name for name in required_packages if importlib.util.find_spec(name) is None
    ]
    if missing_packages:
        raise RuntimeError(f"Missing required packages: {missing_packages}")
    if shutil.which("ruby") is None:
        raise RuntimeError("Ruby is required for the standalone final audit")
    sources = source_hashes(config_path, config)
    snapshot = resolve_snapshot(config["model_id"], args.allow_downloads)
    git_commit = git_capture("rev-parse", "HEAD")
    base_fingerprint_input = {
        "config": config,
        "sources": sources,
        "model_snapshot": snapshot,
        "gcd_installation": gcd_installation,
    }
    protocol = {
        "status": "preflight_complete",
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "base_protocol_fingerprint": stable_hash(base_fingerprint_input),
        "active_protocol_fingerprint": None,
        "budget_profile": None,
        "base_checkpoint": snapshot,
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "gcd_project_root": str(gcd_root),
        "gcd_installation": gcd_installation,
        "git_commit": git_commit,
        "git_worktree_dirty": bool(git_capture("status", "--porcelain")),
        "source_hashes": sources,
        "paper_repo_mismatches": [
            {
                "field": "celebrity_anchor",
                "paper": "celebrity",
                "repository_e10": ["person", "woman", "man"],
                "authority": "repository E10",
            },
            {
                "field": "retain_spelling",
                "paper": "Melanie Grifftih",
                "repository": "Melanie Griffith",
                "authority": "repository string",
            },
        ],
        "orders": {
            order: concept_batches(config, order) for order in ORDERS
        },
        "conditions": list(CONDITIONS),
        "coco_status": "deferred_by_budget",
        "package_versions": package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "resolved_at": utc_now(),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("base_protocol_fingerprint") != protocol["base_protocol_fingerprint"]:
            raise RuntimeError("Existing output uses another base protocol")
        if existing.get("artifact_root") != str(artifact_root):
            raise RuntimeError("Artifact root differs from frozen manifest")
        schedule_path = output_dir / "inputs" / "target_schedule.csv"
        retain_path = output_dir / "inputs" / "retain_set.csv"
        if not schedule_path.is_file() or not retain_path.is_file() or not (output_dir / "PREFLIGHT.md").is_file():
            raise RuntimeError("Existing preflight is incomplete")
        schedule = read_csv(schedule_path)
        expected_schedule = [
            (order, position, concept)
            for order in ORDERS
            for position, concept in enumerate(order_concepts(config, order), 1)
        ]
        observed_schedule = [
            (row["order"], int(row["sequence_position"]), row["concept"])
            for row in schedule
        ]
        if observed_schedule != expected_schedule:
            raise RuntimeError("Frozen target schedule is corrupt")
        if [row["concept"] for row in read_csv(retain_path)] != list(config["fixed_retains"]):
            raise RuntimeError("Frozen retain input is corrupt")
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    schedule_rows = []
    for order in ORDERS:
        for step, batch in enumerate(concept_batches(config, order), start=1):
            for within, concept in enumerate(batch, start=1):
                schedule_rows.append({
                    "order": order, "step": step, "within_batch": within,
                    "sequence_position": (step - 1) * 10 + within,
                    "concept": concept,
                })
    write_csv(output_dir / "inputs" / "target_schedule.csv", schedule_rows)
    write_csv(
        output_dir / "inputs" / "retain_set.csv",
        [{"retain_position": i, "concept": value} for i, value in enumerate(config["fixed_retains"], 1)],
    )
    preflight_lines = [
        "# Sequential OCE Celebrity Preflight",
        "",
        "Status: protocol frozen; no model edit or image generation was launched.",
        "",
        "## Design",
        "",
        "- 100 repository E100 targets, batched as ten repository-E10-sized edits",
        "- Order A is the repository list; Order B is its exact reverse",
        "- baseline and retain-history each start independently from the same frozen SD snapshot",
        "- retain-history at batch t appends exactly the prior 10 × (t - 1) targets",
        "- the repository edit call reads each loaded parent checkpoint as its current pre-edit reference",
        "- one repository joint-100 reference checkpoint",
        "- no protocol-identical saved joint-100 result is assumed; the runner builds it once and resumes it by hash",
        f"- sequential repository guides: {config['edit']['e10']['guide_concepts']}",
        f"- sequential repository settings: {json.dumps(config['edit']['e10'], ensure_ascii=False, sort_keys=True)}",
        f"- joint repository settings: {json.dumps(config['edit']['joint_e100'], ensure_ascii=False, sort_keys=True)}",
        "",
        "## Evaluation",
        "",
        "- every sequential checkpoint: all introduced targets plus 500 fixed-retain images",
        "- paired profile candidates: 5 or 10 trajectory images per celebrity/checkpoint",
        "- 10/50/100 paper cells: 500 targets + 500 retains using official seed-42 sample streams",
        "- joint-100: 500 targets + 500 retains",
        f"- nominal core generated totals (excluding benchmark): profile_5={config['budget']['profile_image_counts']['profile_5']}, "
        f"profile_10={config['budget']['profile_image_counts']['profile_10']}",
        "- profile and formal batch size are not chosen until the fixed 200-image generation + GCD benchmark with a 20% credit reserve",
        "",
        "## Repository versus paper",
        "",
    ]
    for mismatch in protocol["paper_repo_mismatches"]:
        preflight_lines.append(
            f"- `{mismatch['field']}`: paper={mismatch['paper']}; "
            f"repository={mismatch.get('repository_e10', mismatch.get('repository'))}; "
            f"authority={mismatch['authority']}"
        )
    preflight_lines.extend([
        "",
        "## Frozen paths",
        "",
        f"- output: `{output_dir}`",
        f"- artifact root: `{artifact_root}`",
        f"- future qualitative archive: `{artifact_root / config['storage']['qualitative_tarball_name']}`",
        f"- GCD project: `{gcd_root}`",
        f"- GCD resources: `{gcd_installation['app_data_dir']}`",
        "",
        "Exact ordered sets are in `inputs/target_schedule.csv` and `inputs/retain_set.csv`.",
    ])
    (output_dir / "PREFLIGHT.md").write_text(
        "\n".join(preflight_lines) + "\n", encoding="utf-8"
    )
    write_json(manifest_path, protocol)
    update_state(
        output_dir, status="awaiting_benchmark", phase="preflight",
        order="-", condition="-", step=0, current_target="-",
        generation_completed=0, evaluator_completed=0,
    )
    event(output_dir, "preflight", "validated fixed 10x10 protocol")
    return protocol


def require_manifest(args: argparse.Namespace, active: bool = False) -> dict[str, Any]:
    manifest_path = output_path(args) / "run_manifest.json"
    if not manifest_path.is_file():
        return preflight(args)
    manifest = read_json(manifest_path)
    config_path = Path(manifest.get("config_path", args.config)).resolve()
    config = load_config(config_path)
    validate_config(config_path, config, artifacts=True)
    current_sources = source_hashes(config_path, config)
    if current_sources != manifest.get("source_hashes"):
        raise RuntimeError("Current source/config/Cg hashes differ from frozen preflight")
    snapshot = dict(manifest["base_checkpoint"])
    if not Path(snapshot["snapshot_path"]).is_dir():
        raise FileNotFoundError(f"Frozen model snapshot is unavailable: {snapshot['snapshot_path']}")
    gcd_root = resolve_gcd_root(config, Path(manifest["gcd_project_root"]))
    gcd_installation = validate_gcd_installation(gcd_root)
    if gcd_installation != manifest.get("gcd_installation"):
        raise RuntimeError("GCD evaluator/resources differ from frozen preflight")
    base_fingerprint = stable_hash({
        "config": config,
        "sources": current_sources,
        "model_snapshot": snapshot,
        "gcd_installation": gcd_installation,
    })
    if base_fingerprint != manifest.get("base_protocol_fingerprint"):
        raise RuntimeError("Frozen base protocol fingerprint cannot be reproduced")
    if manifest.get("output_dir") != str(output_path(args)):
        raise RuntimeError("Output path differs from frozen manifest")
    if active and not manifest.get("active_protocol_fingerprint"):
        raise RuntimeError("Run --benchmark before formal generation")
    return manifest


def load_pipeline(manifest: Mapping[str, Any], dtype_name: str, edit_only: bool = False) -> Any:
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "safety_checker": None}
    if edit_only:
        kwargs["vae"] = None
    pipe = DiffusionPipeline.from_pretrained(
        manifest["base_checkpoint"]["snapshot_path"], **kwargs
    ).to(manifest["config"]["device"])
    if pipe.scheduler.__class__.__name__ != "PNDMScheduler":
        raise RuntimeError(f"Unexpected scheduler: {pipe.scheduler.__class__.__name__}")
    return pipe


def release_cuda(*objects: Any) -> None:
    import torch

    for value in objects:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.remaining_credits <= 0 or args.gpu_rate <= 0:
        raise ValueError("Credits and GPU rate must be positive")
    manifest = preflight(args)
    output_dir = output_path(args)
    if manifest.get("active_protocol_fingerprint"):
        return read_json(output_dir / "budget_selection.json")
    config = manifest["config"]
    total = int(config["budget"]["benchmark_images"])
    candidates = [int(value) for value in config["generation"]["batch_size_candidates"]]
    per_candidate = int(config["budget"]["benchmark_images_per_candidate"])
    if total != len(candidates) * per_candidate:
        raise RuntimeError("Benchmark allocation does not total exactly 200 images")
    pipe = load_pipeline(manifest, "bfloat16")
    prompts = [
        config["templates"][index % 5].format(config["targets"][index % 10])
        for index in range(total)
    ]
    trials: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    try:
        for candidate_index, batch_size in enumerate(candidates):
            subset_start = candidate_index * per_candidate
            subset = prompts[subset_start:subset_start + per_candidate]
            generated_images = 0
            measured_images = 0
            measured_seconds = 0.0
            peak_memory_bytes: int | None = None
            status = "complete"
            error: str | None = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            try:
                for start in range(0, len(subset), batch_size):
                    batch = subset[start:start + batch_size]
                    absolute_start = subset_start + start
                    generators = [
                        torch.Generator(device="cpu").manual_seed(
                            900000 + absolute_start + offset
                        )
                        for offset in range(len(batch))
                    ]
                    started = time.perf_counter()
                    images = pipe(
                        prompt=batch,
                        num_inference_steps=int(config["generation"]["num_inference_steps"]),
                        guidance_scale=float(config["generation"]["guidance_scale"]),
                        generator=generators,
                        height=int(config["generation"]["height"]),
                        width=int(config["generation"]["width"]),
                    ).images
                    elapsed_batch = time.perf_counter() - started
                    for offset, image in enumerate(images):
                        absolute_index = absolute_start + offset
                        concept = config["targets"][absolute_index % 10]
                        relative = Path("benchmark") / "images" / f"batch_{batch_size:03d}" / f"image_{absolute_index:04d}.png"
                        destination = output_dir / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_suffix(".png.tmp")
                        image.save(temporary, format="PNG")
                        temporary.replace(destination)
                        benchmark_rows.append({
                            "sample_key": f"benchmark|batch={batch_size}|index={absolute_index}",
                            "batch_size": batch_size,
                            "benchmark_index": absolute_index,
                            "celebrity": concept,
                            "prompt": batch[offset],
                            "relative_image_path": str(relative),
                        })
                    generated_images += len(batch)
                    # The first invocation for each batch size is a warm-up.  All
                    # later invocations are the fixed measured subset.
                    if start > 0:
                        measured_images += len(batch)
                        measured_seconds += elapsed_batch
                if torch.cuda.is_available():
                    peak_memory_bytes = int(torch.cuda.max_memory_allocated())
            except (torch.OutOfMemoryError, RuntimeError) as caught:
                is_oom = isinstance(caught, torch.OutOfMemoryError) or "out of memory" in str(caught).casefold()
                if not is_oom:
                    raise
                status = "oom"
                error = f"{type(caught).__name__}: {caught}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            throughput = (
                measured_images / measured_seconds * 3600.0
                if status == "complete" and measured_images > 0 and measured_seconds > 0
                else None
            )
            trials.append({
                "batch_size": batch_size,
                "status": status,
                "allocated_images": per_candidate,
                "generated_images": generated_images,
                "warmup_images": min(batch_size, generated_images),
                "measured_images": measured_images,
                "measured_seconds": measured_seconds,
                "images_per_hour": throughput,
                "peak_memory_bytes": peak_memory_bytes,
                "error": error,
            })
    finally:
        release_cuda(pipe)
    benchmark_generation_seconds = time.perf_counter() - benchmark_started
    successful = [trial for trial in trials if trial["images_per_hour"] is not None]
    if not successful:
        write_json(output_dir / "batch_benchmark_trials.json", {"trials": trials})
        raise RuntimeError("No fixed batch-size candidate completed its measured subset")
    chosen_trial = max(successful, key=lambda row: float(row["images_per_hour"]))
    selected_batch_size = int(chosen_trial["batch_size"])
    throughput = float(chosen_trial["images_per_hour"])
    if len(benchmark_rows) != total:
        write_json(output_dir / "batch_benchmark_trials.json", {
            "trials": trials,
            "generated_images": len(benchmark_rows),
            "expected_images": total,
        })
        raise RuntimeError(
            "The fixed benchmark did not produce exactly 200 images; formal protocol remains unlocked"
        )
    selected_gcd_workers = gcd_worker_count(config)
    benchmark_protocol = dict(manifest)
    benchmark_protocol["budget_profile"] = {
        "selected_gcd_workers": selected_gcd_workers,
    }
    evaluator_started = time.perf_counter()
    benchmark_predictions = evaluate_gcd(
        benchmark_protocol,
        output_dir,
        benchmark_rows,
        output_dir / "benchmark" / "predictions.csv",
    )
    evaluator_seconds = time.perf_counter() - evaluator_started
    if (
        len(benchmark_predictions) != total
        or [row["sample_key"] for row in benchmark_predictions]
        != [row["sample_key"] for row in benchmark_rows]
    ):
        raise RuntimeError("Benchmark GCD predictions are incomplete or reordered")
    evaluator_throughput = total / evaluator_seconds * 3600.0
    usable_hours = (
        float(args.remaining_credits)
        * float(config["budget"]["credit_safety_fraction"])
        / float(args.gpu_rate)
    )
    deadline_budget_hours = (
        float(config["budget"]["hard_deadline_seconds"])
        / 3600.0
        * float(config["budget"]["deadline_safety_fraction"])
    )
    usable_hours = min(usable_hours, deadline_budget_hours)
    counts = config["budget"]["profile_image_counts"]
    prediction_counts = config["budget"]["profile_prediction_counts"]
    estimates = {
        profile: {
            "formal_and_qualitative_images": int(counts[profile]),
            "formal_gcd_predictions": int(prediction_counts[profile]),
            "benchmark_images": total,
            "budgeted_images": int(counts[profile]) + total,
            "generation_hours": int(counts[profile]) / throughput,
            "gcd_evaluation_hours": int(prediction_counts[profile]) / evaluator_throughput,
            "benchmark_hours": (benchmark_generation_seconds + evaluator_seconds) / 3600.0,
            "estimated_total_hours": (
                int(counts[profile]) / throughput
                + int(prediction_counts[profile]) / evaluator_throughput
                + (benchmark_generation_seconds + evaluator_seconds) / 3600.0
            ),
        }
        for profile in PROFILES
    }
    for estimate in estimates.values():
        estimate["fits_budget"] = estimate["estimated_total_hours"] <= usable_hours
    selected = next(
        (profile for profile in reversed(PROFILES) if estimates[profile]["fits_budget"]),
        None,
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda"
    selection = {
        "status": "complete" if selected else "insufficient_credits",
        "selected_profile": selected,
        "remaining_credits": float(args.remaining_credits),
        "gpu_rate_credits_per_hour": float(args.gpu_rate),
        "credit_safety_fraction": float(config["budget"]["credit_safety_fraction"]),
        "budget_hours": usable_hours,
        "hard_deadline_seconds": int(config["budget"]["hard_deadline_seconds"]),
        "deadline_safety_fraction": float(config["budget"]["deadline_safety_fraction"]),
        "deadline_budget_hours": deadline_budget_hours,
        "gpu_name": gpu_name,
        "benchmark_images": total,
        "batch_size_candidates": candidates,
        "selected_batch_size": selected_batch_size,
        "selected_gcd_workers": selected_gcd_workers,
        "batch_trials": trials,
        "benchmark_generation_seconds": benchmark_generation_seconds,
        "measured_images": int(chosen_trial["measured_images"]),
        "measured_seconds": float(chosen_trial["measured_seconds"]),
        "images_per_hour": throughput,
        "gcd_evaluator_images": total,
        "gcd_evaluator_seconds": evaluator_seconds,
        "gcd_images_per_hour": evaluator_throughput,
        "profile_estimates": estimates,
        "selected_at": utc_now(),
    }
    write_json(output_dir / "budget_selection.json", selection)
    write_json(output_dir / "benchmark" / "validation.json", {
        "status": "complete",
        "image_count": total,
        "prediction_count": len(benchmark_predictions),
        "prediction_sha256": sha256_file(output_dir / "benchmark" / "predictions.csv"),
        "non_scoring": True,
        "validated_at": utc_now(),
    })
    shutil.rmtree(output_dir / "benchmark" / "images")
    if selected is None:
        update_state(output_dir, status="insufficient_credits", phase="benchmark")
        raise RuntimeError("Neither fixed sample profile fits the protected credit budget")
    manifest["budget_profile"] = selection
    manifest["active_protocol_fingerprint"] = stable_hash({
        "base": manifest["base_protocol_fingerprint"],
        "profile": selected,
        "profile_seeds": config["generation"]["profile_seeds"][selected],
        "selected_batch_size": selected_batch_size,
        "selected_gcd_workers": selected_gcd_workers,
    })
    manifest["status"] = "ready"
    write_json(output_dir / "run_manifest.json", manifest)
    update_state(
        output_dir, status="ready", phase="benchmark", budget_profile=selected,
        planned_generated_images=int(counts[selected]),
        planned_images_including_benchmark=int(counts[selected]) + total,
    )
    event(
        output_dir, "benchmark", "locked formal sample profile and batch size",
        profile=selected, batch_size=selected_batch_size,
    )
    return selection


def selected_module_state(unet: Any) -> dict[str, Any]:
    return {
        name + ".weight": module.weight.detach().cpu().clone()
        for name, module in unet.named_modules()
        if "attn2" in name and name.endswith("to_v")
    }


def apply_state(unet: Any, state: Mapping[str, Any]) -> None:
    incompatible = unet.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")


def checkpoint_path(output_dir: Path, order: str, condition: str, step: int) -> Path:
    return output_dir / "checkpoints" / order / condition / f"step_{step:03d}.safetensors"


def joint_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoints" / "joint_100" / "joint_100.safetensors"


def checkpoint_manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def validate_checkpoint(
    path: Path, protocol: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    from safetensors.torch import load_file

    manifest_path = checkpoint_manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(path)
    metadata = read_json(manifest_path)
    checks = {
        "active_protocol_fingerprint": protocol["active_protocol_fingerprint"],
        **dict(expected),
    }
    mismatch = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in checks.items() if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Checkpoint manifest mismatch: {mismatch}")
    if metadata.get("checkpoint_sha256") != sha256_file(path):
        raise RuntimeError(f"Checkpoint hash mismatch: {path}")
    state = load_file(str(path))
    if len(state) != int(metadata["tensor_count"]):
        raise RuntimeError(f"Checkpoint tensor count mismatch: {path}")
    return dict(state)


def save_checkpoint_manifest(
    path: Path, protocol: Mapping[str, Any], state: Mapping[str, Any], **metadata: Any
) -> str:
    digest = sha256_file(path)
    write_json(checkpoint_manifest_path(path), {
        "status": "complete",
        "active_protocol_fingerprint": protocol["active_protocol_fingerprint"],
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": digest,
        "tensor_count": len(state),
        "tensor_names": sorted(state),
        "created_at": utc_now(),
        **metadata,
    })
    return digest


def evaluate_before_next_edit(
    pipe: Any, args: argparse.Namespace, protocol: Mapping[str, Any],
    order: str, condition: str, step: int,
) -> None:
    """Free GPU memory, finish this immutable cell, then resume editing."""
    import torch

    marker = raw_cell_dir(output_path(args), order, condition, step) / "cell_complete.json"
    if marker.is_file():
        evaluate_formal_cell(args, protocol, order, condition, step)
        return
    pipe.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluate_formal_cell(args, protocol, order, condition, step)
    pipe.to(protocol["config"]["device"])


def build_sequential_checkpoints(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    import torch
    from safetensors.torch import load_file

    config = protocol["config"]
    output_dir = output_path(args)
    if str(OCE_ROOT) not in sys.path:
        sys.path.insert(0, str(OCE_ROOT))
    import oce as oce_impl

    pipe = load_pipeline(protocol, "float32", edit_only=True)
    base_state = selected_module_state(pipe.unet)
    if not base_state:
        raise RuntimeError("Current repository module selection produced no editable state")
    oce_impl.device = config["device"]
    oce_impl.torch_dtype = torch.float32
    previous_cwd = Path.cwd()
    try:
        os.chdir(OCE_ROOT)
        for order in ORDERS:
            batches = concept_batches(config, order)
            for condition in CONDITIONS:
                parent_state = base_state
                parent_hash = None
                for step, batch in enumerate(batches, start=1):
                    path = checkpoint_path(output_dir, order, condition, step)
                    expected = {
                        "order": order, "condition": condition, "step": step,
                        "parent_checkpoint_sha256": parent_hash,
                    }
                    try:
                        state = validate_checkpoint(path, protocol, expected)
                        parent_state = state
                        parent_hash = sha256_file(path)
                        evaluate_before_next_edit(
                            pipe, args, protocol, order, condition, step
                        )
                        continue
                    except FileNotFoundError:
                        pass
                    if condition == "retain_history" and step == 1:
                        baseline_path = checkpoint_path(output_dir, order, "baseline", 1)
                        baseline_state = validate_checkpoint(
                            baseline_path,
                            protocol,
                            {
                                "order": order,
                                "condition": "baseline",
                                "step": 1,
                                "parent_checkpoint_sha256": None,
                            },
                        )
                        guides = oce_impl.align_guides_with_edits(
                            list(batch), list(config["edit"]["e10"]["guide_concepts"]),
                            seed=int(config["edit"]["e10"]["guide_alignment_seed"]),
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(baseline_path, path)
                        parent_hash = save_checkpoint_manifest(
                            path, protocol, baseline_state,
                            order=order, condition=condition, step=step,
                            batch_targets=list(batch), history_targets=[],
                            fixed_retain_targets=list(config["fixed_retains"]),
                            explicit_retain_concepts=list(config["fixed_retains"]),
                            parent_checkpoint=None,
                            parent_checkpoint_sha256=None,
                            retain_history_reference="current pre-edit checkpoint W0 @ Kp",
                            edit_configuration="repository E10",
                            aligned_guides=guides,
                            reused_from_identical_step1=str(baseline_path.resolve()),
                        )
                        parent_state = baseline_state
                        event(
                            output_dir,
                            "checkpoint_edit",
                            "reused protocol-identical baseline step 1 for retain-history",
                            order=order,
                        )
                        evaluate_before_next_edit(
                            pipe, args, protocol, order, condition, step
                        )
                        continue
                    apply_state(pipe.unet, parent_state)
                    history = order_concepts(config, order)[: (step - 1) * 10]
                    preserve = list(config["fixed_retains"])
                    if condition == "retain_history":
                        preserve.extend(history)
                    guides = oce_impl.align_guides_with_edits(
                        list(batch), list(config["edit"]["e10"]["guide_concepts"]),
                        seed=int(config["edit"]["e10"]["guide_alignment_seed"]),
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    update_state(
                        output_dir, status="running", phase="checkpoint_edit",
                        order=order, condition=condition, step=step,
                        current_target="; ".join(batch),
                    )
                    settings = config["edit"]["e10"]
                    oce_impl.Orthogonal_Erase(
                        pipe, list(batch), guides, preserve,
                        float(settings["erase_scale"]),
                        float(settings["preserve_global_scale"]),
                        float(settings["preserve_concept_scale"]),
                        float(settings["lamb"]), str(path.parent), path.stem,
                    )
                    state = dict(load_file(str(path)))
                    apply_state(pipe.unet, state)
                    parent_checkpoint = (
                        None if step == 1 else str(checkpoint_path(
                            output_dir, order, condition, step - 1
                        ).resolve())
                    )
                    parent_hash = save_checkpoint_manifest(
                        path, protocol, state,
                        order=order, condition=condition, step=step,
                        batch_targets=list(batch), history_targets=history,
                        fixed_retain_targets=list(config["fixed_retains"]),
                        explicit_retain_concepts=preserve,
                        parent_checkpoint=parent_checkpoint,
                        parent_checkpoint_sha256=parent_hash,
                        retain_history_reference=(
                            "current pre-edit checkpoint W0 @ Kp"
                            if condition == "retain_history"
                            else "not applicable"
                        ),
                        edit_configuration="repository E10",
                        aligned_guides=guides,
                    )
                    parent_state = state
                    event(
                        output_dir, "checkpoint_edit", "saved sequential checkpoint",
                        order=order, condition=condition, step=step,
                    )
                    evaluate_before_next_edit(
                        pipe, args, protocol, order, condition, step
                    )
    finally:
        os.chdir(previous_cwd)
        release_cuda(pipe)


def build_joint_checkpoint(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    import torch
    from safetensors.torch import load_file

    config = protocol["config"]
    output_dir = output_path(args)
    path = joint_checkpoint_path(output_dir)
    expected = {"order": "joint", "condition": "joint_100", "step": 10,
                "parent_checkpoint_sha256": None}
    try:
        validate_checkpoint(path, protocol, expected)
        return
    except FileNotFoundError:
        pass
    if str(OCE_ROOT) not in sys.path:
        sys.path.insert(0, str(OCE_ROOT))
    import oce as oce_impl

    pipe = load_pipeline(protocol, "float32", edit_only=True)
    settings = config["edit"]["joint_e100"]
    oce_impl.device = config["device"]
    oce_impl.torch_dtype = torch.float32
    previous_cwd = Path.cwd()
    try:
        os.chdir(OCE_ROOT)
        guides = oce_impl.align_guides_with_edits(
            list(config["targets"]), list(settings["guide_concepts"]),
            seed=int(settings["guide_alignment_seed"]),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        oce_impl.Orthogonal_Erase(
            pipe, list(config["targets"]), guides, list(config["fixed_retains"]),
            float(settings["erase_scale"]),
            float(settings["preserve_global_scale"]),
            float(settings["preserve_concept_scale"]),
            float(settings["lamb"]), str(path.parent), path.stem,
        )
        state = dict(load_file(str(path)))
        save_checkpoint_manifest(
            path, protocol, state, **expected,
            batch_targets=list(config["targets"]), history_targets=[],
            fixed_retain_targets=list(config["fixed_retains"]),
            explicit_retain_concepts=list(config["fixed_retains"]),
            parent_checkpoint=None, edit_configuration="repository E100",
            aligned_guides=guides,
        )
        event(output_dir, "checkpoint_edit", "saved joint-100 reference checkpoint")
    finally:
        os.chdir(previous_cwd)
        release_cuda(pipe)


def profile_seeds(protocol: Mapping[str, Any]) -> list[int]:
    profile = protocol["budget_profile"]["selected_profile"]
    return [int(value) for value in protocol["config"]["generation"]["profile_seeds"][profile]]


def paper_samples_per_prompt(protocol: Mapping[str, Any], step: int) -> int:
    return int(
        protocol["config"]["generation"]["paper_target_samples_per_prompt"].get(
            str(step), 0
        )
    )


def qualitative_slots(protocol: Mapping[str, Any], order: str) -> dict[str, list[int]]:
    config = protocol["config"]
    sequence = order_concepts(config, order)
    offsets = [int(value) for value in config["qualitative"]["later_batch_offsets"]]
    result: dict[str, list[int]] = {}
    for position in config["qualitative"]["sequence_positions"]:
        concept = sequence[int(position) - 1]
        introduction = (int(position) - 1) // 10 + 1
        steps = {introduction, 10}
        steps.update(min(10, introduction + offset) for offset in offsets)
        result[concept] = sorted(steps)
    return result


def sample_key(
    order: str, condition: str, step: int, set_name: str,
    concept: str, template_index: int, seed: int, sample_index: int,
    generator_protocol: str,
) -> str:
    return stable_hash({
        "order": order, "condition": condition, "step": step,
        "set": set_name, "concept": concept,
        "template_index": template_index, "seed": seed,
        "sample_index": sample_index, "generator_protocol": generator_protocol,
    })


def image_path(
    output_dir: Path, order: str, condition: str, step: int,
    set_name: str, concept: str, template_index: int, sample_id: str,
) -> Path:
    return (
        output_dir / "images" / order / condition / f"step_{step:03d}"
        / set_name / slug(concept)
        / f"template_{template_index}_{sample_id}.png"
    )


def raw_cell_dir(
    output_dir: Path, order: str, condition: str, step: int
) -> Path:
    return output_dir / "raw" / "gcd_cells" / order / condition / f"step_{step:03d}"


def formal_rows(
    protocol: Mapping[str, Any], output_dir: Path,
    order: str, condition: str, step: int,
) -> list[dict[str, Any]]:
    config = protocol["config"]
    if order == "joint":
        target_concepts = list(config["targets"])
    else:
        target_concepts = order_concepts(config, order)[: step * 10]
    rows: list[dict[str, Any]] = []
    for set_name, concepts in (
        ("targets", target_concepts),
        ("retains", list(config["fixed_retains"])),
    ):
        for concept in concepts:
            introduction = (
                None if set_name == "retains"
                else 1 if order == "joint"
                else order_concepts(config, order).index(concept) // 10 + 1
            )
            for template_index, template in enumerate(config["templates"]):
                if set_name == "retains" or order == "joint":
                    samples = [(42, 0, "official_stream_42", True, True)]
                else:
                    paper_count = paper_samples_per_prompt(protocol, step)
                    samples = []
                    for seed in profile_seeds(protocol):
                        samples.append((
                            seed, 0,
                            "official_stream_42" if seed == 42 else "independent_seed",
                            True,
                            bool(seed == 42 and paper_count),
                        ))
                    for sample_index in range(1, paper_count):
                        samples.append((42, sample_index, "official_stream_42", False, True))
                for seed, sample_index, generator_protocol, trajectory_sample, paper_sample in samples:
                    sample_id = (
                        f"official_seed_42_index_{sample_index}"
                        if generator_protocol == "official_stream_42"
                        else f"trajectory_seed_{seed}"
                    )
                    path = image_path(
                        output_dir, order, condition, step, set_name,
                        concept, template_index, sample_id,
                    )
                    rows.append({
                        "sample_key": sample_key(
                            order, condition, step, set_name, concept,
                            template_index, seed, sample_index, generator_protocol,
                        ),
                        "order": order, "condition": condition, "step": step,
                        "set": set_name, "celebrity": concept,
                        "concept_introduction_step": introduction,
                        "template_index": template_index, "template": template,
                        "prompt": template.format(concept), "seed": seed,
                        "sample_index": sample_index,
                        "generator_protocol": generator_protocol,
                        "generator_group": (
                            f"{set_name}|{concept}|{template_index}|official_seed_42"
                            if generator_protocol == "official_stream_42" else ""
                        ),
                        "trajectory_sample": trajectory_sample,
                        "paper_sample": paper_sample,
                        "relative_image_path": str(path.relative_to(output_dir)),
                    })
    return rows


def checkpoint_for_cell(
    output_dir: Path, order: str, condition: str, step: int
) -> Path:
    return (
        joint_checkpoint_path(output_dir)
        if order == "joint"
        else checkpoint_path(output_dir, order, condition, step)
    )


def generate_formal_images(
    protocol: Mapping[str, Any], output_dir: Path,
    order: str, condition: str, step: int,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    import torch
    from safetensors.torch import load_file

    pending = [
        row for row in rows
        if not (output_dir / str(row["relative_image_path"])).is_file()
    ]
    missing_groups = {
        str(row["generator_group"])
        for row in pending if str(row["generator_group"])
    }
    pending = [
        row for row in rows
        if not (output_dir / str(row["relative_image_path"])).is_file()
        or str(row["generator_group"]) in missing_groups
    ]
    if not pending:
        return
    pipe = load_pipeline(protocol, "bfloat16")
    checkpoint = checkpoint_for_cell(output_dir, order, condition, step)
    apply_state(pipe.unet, load_file(str(checkpoint)))
    batch_size = int(protocol["budget_profile"]["selected_batch_size"])
    generated = 0
    generator_cache: dict[str, Any] = {}
    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            generators = []
            for row in batch:
                group = str(row["generator_group"])
                if group:
                    generator = generator_cache.setdefault(
                        group, torch.Generator(device="cpu").manual_seed(42)
                    )
                else:
                    generator = torch.Generator(device="cpu").manual_seed(int(row["seed"]))
                generators.append(generator)
            images = pipe(
                prompt=[str(row["prompt"]) for row in batch],
                num_inference_steps=int(protocol["config"]["generation"]["num_inference_steps"]),
                guidance_scale=float(protocol["config"]["generation"]["guidance_scale"]),
                height=int(protocol["config"]["generation"]["height"]),
                width=int(protocol["config"]["generation"]["width"]),
                generator=generators,
            ).images
            for row, image in zip(batch, images):
                path = output_dir / str(row["relative_image_path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".tmp")
                image.save(temporary, format="PNG")
                temporary.replace(path)
                generated += 1
            update_state(
                output_dir, status="running", phase="generation",
                order=order, condition=condition, step=step,
                generation_cell_completed=generated,
                generation_cell_total=len(pending),
            )
    finally:
        release_cuda(pipe)
    missing = [
        row["relative_image_path"] for row in rows
        if not (output_dir / str(row["relative_image_path"])).is_file()
    ]
    if missing:
        raise RuntimeError(f"Generation incomplete; missing {len(missing)} images")


def load_dotenv_file(path: Path) -> None:
    for key, value in dotenv_values(path).items():
        os.environ.setdefault(key, value)


def evaluate_gcd(
    protocol: Mapping[str, Any], output_dir: Path,
    rows: Sequence[Mapping[str, Any]], destination: Path,
) -> list[dict[str, Any]]:
    root = Path(protocol["gcd_project_root"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    load_dotenv_file(root / ".env")
    os.environ["APP_DATA_DIR"] = protocol["gcd_installation"]["app_data_dir"]
    from model_training.helpers.labels import Labels
    from model_training.helpers.face_recognizer import FaceRecognizer

    resources = os.environ.get("APP_DATA_DIR")
    if not resources:
        raise RuntimeError("GCD .env must set APP_DATA_DIR")
    gcd_settings = protocol["gcd_installation"]["settings"]
    image_size = int(gcd_settings["APP_FACE_SIZE"])
    labels = Labels(resources_path=resources)
    recognizer = FaceRecognizer(
        labels=labels, resources_path=resources,
        use_cuda=str(gcd_settings["USE_CUDA"]).casefold() == "true",
        top_n=int(protocol["config"]["gcd"]["recognizer_top_n"]),
    )
    workers = int(protocol["budget_profile"]["selected_gcd_workers"])
    if workers < 1 or workers > int(protocol["config"]["gcd"]["max_workers"]):
        raise RuntimeError(f"Invalid frozen GCD worker count: {workers}")
    predictions: list[dict[str, Any]] = []
    tasks = [
        (index, str((output_dir / str(row["relative_image_path"])).resolve()))
        for index, row in enumerate(rows)
    ]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initialize_gcd_detector_worker,
        initargs=(
            str(root), resources, float(gcd_settings["APP_FACE_MARGIN"]), image_size,
        ),
    ) as executor:
        detected_stream = executor.map(detect_faces_worker, tasks, chunksize=1)
        for completed, (detected_index, faces) in enumerate(detected_stream, start=1):
            expected_index = completed - 1
            if detected_index != expected_index:
                raise RuntimeError("Parallel GCD detector returned reordered rows")
            row = rows[detected_index]
            result = recognizer.perform(faces)
            top: list[dict[str, Any]] = []
            if result:
                for label, probability in result[0][0]:
                    name = str(label).split("_[", 1)[0].replace("_", " ")
                    top.append({"name": name, "probability": float(probability)})
            top1 = top[0]["name"] if top else None
            predictions.append({
                **dict(row),
                "face_detected": bool(top),
                "top1": top1,
                "top1_probability": top[0]["probability"] if top else None,
                "correct": bool(top1 and top1.casefold() == str(row["celebrity"]).casefold()),
                "top5_json": json.dumps(top, ensure_ascii=False),
            })
            if completed % 100 == 0 or completed == len(rows):
                update_state(
                    output_dir,
                    evaluator_cell_completed=completed,
                    evaluator_cell_total=len(rows),
                )
                print(f"[gcd workers={workers}] {completed}/{len(rows)}", flush=True)
    write_csv(destination, predictions)
    release_cuda(recognizer, labels)
    return predictions


def raw_accuracy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    detected = sum(bool_value(row["face_detected"]) for row in rows)
    correct = sum(bool_value(row["correct"]) for row in rows)
    return {
        "sample_count": len(rows),
        "no_face_count": len(rows) - detected,
        "face_detected_count": detected,
        "correct_count": correct,
        "accuracy": correct / detected if detected else None,
    }


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).casefold() in {"true", "1", "yes"}


def copy_qualitative_samples(
    protocol: Mapping[str, Any], output_dir: Path,
    order: str, condition: str, step: int,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if order == "joint":
        return
    slots = qualitative_slots(protocol, order)
    by_key = {
        (str(row["celebrity"]), int(row["template_index"]), int(row["seed"])):
        output_dir / str(row["relative_image_path"])
        for row in rows
        if row["set"] == "targets"
        and bool_value(row["trajectory_sample"])
        and int(row["sample_index"]) == 0
    }
    for concept, selected_steps in slots.items():
        if step not in selected_steps:
            continue
        for seed in protocol["config"]["qualitative"]["seeds"]:
            destination = (
                output_dir / "qualitative" / "raw" / order / condition
                / slug(concept) / f"step_{step:03d}_seed_{seed}.png"
            )
            if destination.is_file():
                continue
            source = by_key.get((concept, 0, int(seed)))
            if source is None or not source.is_file():
                raise RuntimeError(
                    f"Missing qualitative source {order}/{condition}/{step}/{concept}/{seed}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def validate_and_cleanup_cell(
    protocol: Mapping[str, Any], output_dir: Path,
    order: str, condition: str, step: int,
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
    cell_dir: Path,
) -> None:
    if len(rows) != len(predictions):
        raise RuntimeError("GCD prediction count differs from generation manifest")
    expected_keys = [str(row["sample_key"]) for row in rows]
    observed_keys = [str(row["sample_key"]) for row in predictions]
    if observed_keys != expected_keys or len(set(observed_keys)) != len(observed_keys):
        raise RuntimeError("Prediction sample keys are incomplete, reordered, or duplicated")
    prediction_path = cell_dir / "predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    metrics = {
        "status": "complete",
        "active_protocol_fingerprint": protocol["active_protocol_fingerprint"],
        "order": order, "condition": condition, "step": step,
        "checkpoint": str(checkpoint_for_cell(output_dir, order, condition, step).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_for_cell(output_dir, order, condition, step)),
        "sets": {
            set_name: raw_accuracy([row for row in predictions if row["set"] == set_name])
            for set_name in ("targets", "retains")
        },
        "prediction_sha256": sha256_file(prediction_path),
        "completed_at": utc_now(),
    }
    write_json(cell_dir / "metrics.json", metrics)
    generation_path = cell_dir / "generation_manifest.csv"
    marker = {
        "status": "complete",
        "active_protocol_fingerprint": protocol["active_protocol_fingerprint"],
        "row_count": len(rows),
        "generation_manifest_sha256": sha256_file(generation_path),
        "prediction_sha256": sha256_file(prediction_path),
        "metrics_sha256": sha256_file(cell_dir / "metrics.json"),
        "cleanup_permitted": True,
        "predictions_validated_at": utc_now(),
    }
    removed = 0
    for row in rows:
        path = output_dir / str(row["relative_image_path"])
        if path.is_file():
            path.unlink()
            removed += 1
    marker["removed_formal_images"] = removed
    marker["formal_images_remaining"] = sum(
        (output_dir / str(row["relative_image_path"])).is_file() for row in rows
    )
    marker["cleanup_completed_at"] = utc_now()
    write_json(cell_dir / "cell_complete.json", marker)


def generate_missing_qualitative_seed43(
    protocol: Mapping[str, Any], output_dir: Path,
    order: str, condition: str, step: int,
) -> None:
    """Profile 5 lacks independent seed 43; generate only fixed portraits."""
    if protocol["budget_profile"]["selected_profile"] != "profile_5":
        return
    slots = qualitative_slots(protocol, order)
    concepts = [concept for concept, steps in slots.items() if step in steps]
    destinations = {
        concept: output_dir / "qualitative" / "raw" / order / condition
        / slug(concept) / f"step_{step:03d}_seed_43.png"
        for concept in concepts
    }
    pending = [concept for concept, path in destinations.items() if not path.is_file()]
    if not pending:
        return
    import torch
    from safetensors.torch import load_file

    pipe = load_pipeline(protocol, "bfloat16")
    apply_state(pipe.unet, load_file(str(checkpoint_path(output_dir, order, condition, step))))
    try:
        for concept in pending:
            image = pipe(
                prompt=protocol["config"]["templates"][0].format(concept),
                num_inference_steps=50, guidance_scale=7.5,
                generator=torch.Generator(device="cpu").manual_seed(43),
                height=512, width=512,
            ).images[0]
            destinations[concept].parent.mkdir(parents=True, exist_ok=True)
            temporary = destinations[concept].with_suffix(".png.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(destinations[concept])
    finally:
        release_cuda(pipe)


def evaluate_formal_cell(
    args: argparse.Namespace, protocol: Mapping[str, Any],
    order: str, condition: str, step: int,
) -> None:
    output_dir = output_path(args)
    cell_dir = raw_cell_dir(output_dir, order, condition, step)
    marker = cell_dir / "cell_complete.json"
    if marker.is_file():
        existing = read_json(marker)
        if existing.get("active_protocol_fingerprint") != protocol["active_protocol_fingerprint"]:
            raise RuntimeError(f"Completed cell protocol mismatch: {cell_dir}")
        generation_path = cell_dir / "generation_manifest.csv"
        prediction_path = cell_dir / "predictions.csv"
        metrics_path = cell_dir / "metrics.json"
        expected_rows = formal_rows(protocol, output_dir, order, condition, step)
        generation_rows = read_csv(generation_path)
        prediction_rows = read_csv(prediction_path)
        valid = (
            existing.get("status") == "complete"
            and existing.get("cleanup_permitted") is True
            and int(existing.get("formal_images_remaining", -1)) == 0
            and int(existing.get("row_count", -1)) == len(expected_rows)
            and [row["sample_key"] for row in expected_rows]
            == [row["sample_key"] for row in generation_rows]
            == [row["sample_key"] for row in prediction_rows]
            and existing.get("generation_manifest_sha256") == sha256_file(generation_path)
            and existing.get("prediction_sha256") == sha256_file(prediction_path)
            and existing.get("metrics_sha256") == sha256_file(metrics_path)
        )
        if not valid:
            raise RuntimeError(f"Completed cell failed resume integrity: {cell_dir}")
        return
    rows = formal_rows(protocol, output_dir, order, condition, step)
    cell_dir.mkdir(parents=True, exist_ok=True)
    generation_manifest = cell_dir / "generation_manifest.csv"
    write_csv(generation_manifest, rows)
    update_state(
        output_dir, status="running", phase="generation",
        order=order, condition=condition, step=step,
        current_target="; ".join(
            list(protocol["config"]["targets"])
            if order == "joint"
            else concept_batches(protocol["config"], order)[step - 1]
        ),
        generation_cell_completed=0, generation_cell_total=len(rows),
        evaluator_cell_completed=0, evaluator_cell_total=len(rows),
    )
    generate_missing_qualitative_seed43(protocol, output_dir, order, condition, step)
    generate_formal_images(protocol, output_dir, order, condition, step, rows)
    predictions = evaluate_gcd(
        protocol, output_dir, rows, cell_dir / "predictions.csv"
    )
    copy_qualitative_samples(protocol, output_dir, order, condition, step, rows)
    validate_and_cleanup_cell(
        protocol, output_dir, order, condition, step, rows, predictions, cell_dir
    )
    event(
        output_dir, "evaluation", "completed and cleaned formal GCD cell",
        order=order, condition=condition, step=step, images=len(rows),
    )


def generate_original_qualitative(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    import torch

    output_dir = output_path(args)
    concepts = sorted({
        concept
        for order in ORDERS
        for concept in qualitative_slots(protocol, order)
    })
    pending = []
    for concept in concepts:
        for seed in protocol["config"]["qualitative"]["seeds"]:
            path = output_dir / "qualitative" / "raw" / "original" / slug(concept) / f"seed_{seed}.png"
            if not path.is_file():
                pending.append((concept, int(seed), path))
    if not pending:
        return
    pipe = load_pipeline(protocol, "bfloat16")
    try:
        for concept, seed, path in pending:
            image = pipe(
                prompt=protocol["config"]["templates"][0].format(concept),
                num_inference_steps=50, guidance_scale=7.5,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                height=512, width=512,
            ).images[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".png.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(path)
    finally:
        release_cuda(pipe)


def completed_cell_paths(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "raw" / "gcd_cells").glob("**/cell_complete.json"))


def all_prediction_rows(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((output_dir / "raw" / "gcd_cells").glob("**/predictions.csv")):
        rows.extend(read_csv(path))
    return rows


def accuracy_for(
    predictions: Sequence[Mapping[str, Any]], *, order: str, condition: str,
    step: int, set_name: str, concept: str | None = None,
    trajectory_only: bool = False, paper_only: bool = False,
) -> dict[str, Any]:
    rows = [
        row for row in predictions
        if row["order"] == order
        and row["condition"] == condition
        and int(row["step"]) == step
        and row["set"] == set_name
        and (concept is None or row["celebrity"] == concept)
        and (not trajectory_only or bool_value(row["trajectory_sample"]))
        and (not paper_only or bool_value(row["paper_sample"]))
    ]
    if not rows:
        raise RuntimeError(
            f"No prediction rows for {order}/{condition}/{step}/{set_name}/{concept}"
        )
    return raw_accuracy(rows)


def finite_values(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def descriptive(values: Sequence[float | None]) -> dict[str, float | None]:
    finite = finite_values(values)
    if not finite:
        return {"mean": None, "median": None, "maximum": None}
    return {
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "maximum": max(finite),
    }


def harmonic_score(acc_e: float | None, acc_s: float | None) -> float | None:
    if acc_e is None or acc_s is None:
        return None
    erase_success = 1.0 - acc_e
    if erase_success <= 0.0 or acc_s <= 0.0:
        return 0.0
    return 2.0 / (1.0 / erase_success + 1.0 / acc_s)


def aggregate_results(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    output_dir = output_path(args)
    markers = completed_cell_paths(output_dir)
    if len(markers) != 41:
        raise RuntimeError(f"Expected 41 completed GCD cells, found {len(markers)}")
    predictions = all_prediction_rows(output_dir)
    expected_prediction_count = (
        34800
        if protocol["budget_profile"]["selected_profile"] == "profile_5"
        else 45800
    )
    if len(predictions) != expected_prediction_count:
        raise RuntimeError(
            f"Expected {expected_prediction_count} GCD predictions, found {len(predictions)}"
        )
    keys = [str(row["sample_key"]) for row in predictions]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate raw GCD prediction keys")
    trajectory_rows: list[dict[str, Any]] = []
    introductions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for order in ORDERS:
        sequence = order_concepts(protocol["config"], order)
        for condition in CONDITIONS:
            for step in range(1, 11):
                checkpoint = checkpoint_path(output_dir, order, condition, step)
                checkpoint_hash = sha256_file(checkpoint)
                for concept in sequence[: step * 10]:
                    introduction = sequence.index(concept) // 10 + 1
                    metric = accuracy_for(
                        predictions, order=order, condition=condition,
                        step=step, set_name="targets", concept=concept,
                        trajectory_only=True,
                    )
                    key = (order, condition, concept)
                    if step == introduction:
                        introductions[key] = metric
                    intro_metric = introductions.get(key)
                    if intro_metric is None:
                        raise RuntimeError(f"Missing introduction metric: {key}")
                    intro_accuracy = intro_metric["accuracy"]
                    success = (
                        intro_accuracy is not None
                        and float(intro_accuracy)
                        <= float(protocol["config"]["introduction_success_max_accuracy"])
                    )
                    trajectory_rows.append({
                        "order": order,
                        "condition": condition,
                        "step": step,
                        "concept": concept,
                        "concept_introduction_step": introduction,
                        "current_position_age": step - introduction,
                        "was_successfully_erased_at_introduction": success,
                        "introduction_status": (
                            "successfully_erased" if success else "failed_at_introduction"
                        ),
                        "introduction_raw_gcd_accuracy": intro_accuracy,
                        "sample_count": metric["sample_count"],
                        "face_detected_count": metric["face_detected_count"],
                        "no_face_count": metric["no_face_count"],
                        "correct_count": metric["correct_count"],
                        "raw_gcd_accuracy": metric["accuracy"],
                        "checkpoint_id": checkpoint_hash,
                    })
    write_csv(output_dir / "trajectory_per_concept.csv", trajectory_rows)

    step_rows: list[dict[str, Any]] = []
    for order in ORDERS:
        sequence = order_concepts(protocol["config"], order)
        for condition in CONDITIONS:
            for step in range(1, 11):
                current = sequence[(step - 1) * 10: step * 10]
                historical = sequence[: (step - 1) * 10]
                by_concept = {
                    row["concept"]: row
                    for row in trajectory_rows
                    if row["order"] == order
                    and row["condition"] == condition
                    and int(row["step"]) == step
                }
                current_values = [by_concept[value]["raw_gcd_accuracy"] for value in current]
                history_values = [by_concept[value]["raw_gcd_accuracy"] for value in historical]
                current_summary = descriptive(current_values)
                history_summary = descriptive(history_values)
                retain = accuracy_for(
                    predictions, order=order, condition=condition, step=step,
                    set_name="retains", trajectory_only=True,
                )
                target_samples = sum(
                    int(by_concept[value]["sample_count"])
                    for value in current + historical
                )
                step_rows.append({
                    "order": order, "condition": condition, "step": step,
                    "current_batch_targets_json": json.dumps(current, ensure_ascii=False),
                    "current_batch_individual_raw_accuracies_json": json.dumps(
                        dict(zip(current, current_values)), ensure_ascii=False
                    ),
                    "current_batch_mean_raw_accuracy": current_summary["mean"],
                    "current_batch_median_raw_accuracy": current_summary["median"],
                    "current_batch_maximum_raw_accuracy": current_summary["maximum"],
                    "number_of_historical_targets": len(historical),
                    "historical_individual_raw_accuracies_json": json.dumps(
                        dict(zip(historical, history_values)), ensure_ascii=False
                    ),
                    "mean_historical_target_raw_accuracy": history_summary["mean"],
                    "median_historical_target_raw_accuracy": history_summary["median"],
                    "maximum_historical_target_raw_accuracy": history_summary["maximum"],
                    "retain_set_gcd_accuracy": retain["accuracy"],
                    "target_sample_count": target_samples,
                    "retain_sample_count": retain["sample_count"],
                    "retain_face_detected_count": retain["face_detected_count"],
                    "checkpoint_id": sha256_file(checkpoint_path(output_dir, order, condition, step)),
                })
    write_csv(output_dir / "step_summary.csv", step_rows)

    paper_rows: list[dict[str, Any]] = []
    for order in ORDERS:
        for condition in CONDITIONS:
            for step in MILESTONE_STEPS:
                target = accuracy_for(
                    predictions, order=order, condition=condition, step=step,
                    set_name="targets", paper_only=True,
                )
                retain = accuracy_for(
                    predictions, order=order, condition=condition, step=step,
                    set_name="retains", paper_only=True,
                )
                if target["sample_count"] != 500 or retain["sample_count"] != 500:
                    raise RuntimeError("Paper checkpoint must be 500 target + 500 retain")
                paper_rows.append({
                    "order": order, "condition": condition, "step": step,
                    "erased_concept_count": step * 10,
                    "reference_type": "sequential",
                    "official_target_gcd_accuracy": target["accuracy"],
                    "official_retain_gcd_accuracy": retain["accuracy"],
                    "H_o": harmonic_score(target["accuracy"], retain["accuracy"]),
                    "target_sample_count": target["sample_count"],
                    "target_face_detected_count": target["face_detected_count"],
                    "target_no_face_count": target["no_face_count"],
                    "retain_sample_count": retain["sample_count"],
                    "retain_face_detected_count": retain["face_detected_count"],
                    "retain_no_face_count": retain["no_face_count"],
                    "checkpoint_id": sha256_file(checkpoint_path(output_dir, order, condition, step)),
                    "coco_status": "deferred_by_budget",
                })
    target = accuracy_for(
        predictions, order="joint", condition="joint_100", step=10,
        set_name="targets", paper_only=True,
    )
    retain = accuracy_for(
        predictions, order="joint", condition="joint_100", step=10,
        set_name="retains", paper_only=True,
    )
    paper_rows.append({
        "order": "joint", "condition": "joint_100", "step": 10,
        "erased_concept_count": 100, "reference_type": "joint_reference",
        "official_target_gcd_accuracy": target["accuracy"],
        "official_retain_gcd_accuracy": retain["accuracy"],
        "H_o": harmonic_score(target["accuracy"], retain["accuracy"]),
        "target_sample_count": target["sample_count"],
        "target_face_detected_count": target["face_detected_count"],
        "target_no_face_count": target["no_face_count"],
        "retain_sample_count": retain["sample_count"],
        "retain_face_detected_count": retain["face_detected_count"],
        "retain_no_face_count": retain["no_face_count"],
        "checkpoint_id": sha256_file(joint_checkpoint_path(output_dir)),
        "coco_status": "deferred_by_budget",
    })
    write_csv(output_dir / "paper_checkpoint_results.csv", paper_rows)
    write_csv(output_dir / "raw" / "all_gcd_predictions.csv", predictions)


def build_contact_sheets(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    from PIL import Image, ImageDraw

    output_dir = output_path(args)
    sheet_rows = []
    for order in ORDERS:
        for concept, steps in qualitative_slots(protocol, order).items():
            for seed in protocol["config"]["qualitative"]["seeds"]:
                original = output_dir / "qualitative" / "raw" / "original" / slug(concept) / f"seed_{seed}.png"
                columns: list[tuple[str, Path]] = [("Original", original)]
                for step in steps:
                    for condition in CONDITIONS:
                        columns.append((
                            f"{condition} s{step}",
                            output_dir / "qualitative" / "raw" / order / condition
                            / slug(concept) / f"step_{step:03d}_seed_{seed}.png",
                        ))
                missing = [str(path) for _, path in columns if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing qualitative images: {missing[:3]}")
                opened = []
                for _, path in columns:
                    with Image.open(path) as image:
                        opened.append(image.convert("RGB").copy())
                width, height = opened[0].size
                header = 42
                canvas = Image.new("RGB", (len(opened) * width, height + header), "white")
                draw = ImageDraw.Draw(canvas)
                for index, ((label, _), image) in enumerate(zip(columns, opened)):
                    canvas.paste(image, (index * width, header))
                    draw.text((index * width + 5, 8), label, fill="black")
                destination = (
                    output_dir / "qualitative" / "contact_sheets" / order
                    / slug(concept) / f"seed_{seed}.png"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".png.tmp")
                canvas.save(temporary, format="PNG")
                temporary.replace(destination)
                sheet_rows.append({
                    "order": order, "concept": concept, "seed": seed,
                    "steps": steps,
                    "contact_sheet": str(destination.relative_to(output_dir)),
                })
    raw_images = sorted((output_dir / "qualitative" / "raw").glob("**/*.png"))
    sheets = sorted((output_dir / "qualitative" / "contact_sheets").glob("**/*.png"))
    write_json(output_dir / "qualitative" / "qualitative_manifest.json", {
        "status": "complete",
        "selection_timing": "fixed before formal generation",
        "sequence_positions": protocol["config"]["qualitative"]["sequence_positions"],
        "seeds": protocol["config"]["qualitative"]["seeds"],
        "raw_image_count": len(raw_images),
        "contact_sheet_count": len(sheets),
        "selection_by_order": {
            order: qualitative_slots(protocol, order) for order in ORDERS
        },
        "raw_images": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in raw_images
        ],
        "contact_sheets": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sheets
        ],
        "sheets": sheet_rows,
    })


def package_qualitative(args: argparse.Namespace, protocol: Mapping[str, Any]) -> Path:
    output_dir = output_path(args)
    artifact_root = Path(protocol["artifact_root"])
    destination = artifact_root / protocol["config"]["storage"]["qualitative_tarball_name"]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(output_dir / "qualitative" / "raw", arcname="raw")
        archive.add(output_dir / "qualitative" / "contact_sheets", arcname="contact_sheets")
        archive.add(
            output_dir / "qualitative" / "qualitative_manifest.json",
            arcname="qualitative_manifest.json",
        )
    temporary.replace(destination)
    return destination


def run_independent_audit(args: argparse.Namespace) -> None:
    script = HERE / "audit_results.rb"
    completed = subprocess.run(
        ["ruby", str(script), str(output_path(args))], check=False, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Independent Ruby audit failed with {completed.returncode}")


def render_report(args: argparse.Namespace, protocol: Mapping[str, Any], tarball: Path) -> None:
    output_dir = output_path(args)
    paper = read_csv(output_dir / "paper_checkpoint_results.csv")
    trajectory = read_csv(output_dir / "trajectory_per_concept.csv")
    joint = next(row for row in paper if row["reference_type"] == "joint_reference")
    lines = [
        "# Long-Horizon Sequential OCE Celebrity Results",
        "",
        "## Status",
        "",
        "The fixed 10×10 sequential celebrity experiment completed its core GCD protocol. ",
        "MS-COCO evaluation is explicitly deferred by the Lightning free-credit budget and is not silently substituted.",
        "",
        "## Fixed protocol",
        "",
        f"- Budget profile: `{protocol['budget_profile']['selected_profile']}`",
        "- Orders: official order and its exact reverse",
        "- Conditions: baseline and retain full history",
        "- Introduction success threshold: raw GCD accuracy ≤ 10%",
        "- GCD denominator: images with a detected face",
        "",
        "## Paper checkpoint metrics",
        "",
        "| Order | Condition | Erased | Acc_e ↓ | Acc_s ↑ | H_o ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def display(value: str) -> str:
        return "NA" if value in {"", "None"} else f"{float(value):.4f}"

    for row in paper:
        lines.append(
            f"| {row['order']} | {row['condition']} | {row['erased_concept_count']} | "
            f"{display(row['official_target_gcd_accuracy'])} | "
            f"{display(row['official_retain_gcd_accuracy'])} | "
            f"{display(row['H_o'])} |"
        )
    lines.extend([
        "",
        "## Immediate-erasure validity",
        "",
        "| Order | Condition | Successfully erased at introduction | Failed at introduction |",
        "|---|---|---:|---:|",
    ])
    for order in ORDERS:
        for condition in CONDITIONS:
            introduced = [
                row for row in trajectory
                if row["order"] == order
                and row["condition"] == condition
                and int(row["step"]) == int(row["concept_introduction_step"])
            ]
            succeeded = sum(
                row["introduction_status"] == "successfully_erased" for row in introduced
            )
            lines.append(
                f"| {order} | {condition} | {succeeded} | {len(introduced) - succeeded} |"
            )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "Only concepts that met the predeclared introduction threshold may be described as a successful erasure that later reappeared. "
        "If baseline trajectories do not show clear cumulative reappearance, this sequential direction is a negative result; retain-history is not a necessary improvement.",
        "",
        "No celebrities, orders, metrics, or checkpoints may be selected after viewing results. "
        "The complete individual trajectories are in `trajectory_per_concept.csv`.",
        "",
        "## Repository versus paper",
        "",
        "The experiment followed the frozen current-repository edit and evaluator behavior. "
        "The manifest records the paper's `celebrity` anchor description versus the repository E10 guides, "
        "and the paper's `Melanie Grifftih` spelling versus the repository's `Melanie Griffith`; no silent reconciliation was made.",
        "",
        "## Joint-100 reference",
        "",
        f"Joint-100 Acc_e={display(joint['official_target_gcd_accuracy'])}, "
        f"Acc_s={display(joint['official_retain_gcd_accuracy'])}, "
        f"H_o={display(joint['H_o'])}.",
        "",
        "## Artifacts",
        "",
        f"Qualitative archive: `{tarball.resolve()}`",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def final_validation(args: argparse.Namespace, protocol: Mapping[str, Any], tarball: Path) -> None:
    output_dir = output_path(args)
    required = [
        output_dir / "trajectory_per_concept.csv",
        output_dir / "step_summary.csv",
        output_dir / "paper_checkpoint_results.csv",
        output_dir / "raw" / "all_gcd_predictions.csv",
        output_dir / "independent_audit.json",
        output_dir / "REPORT.md",
        output_dir / "qualitative" / "qualitative_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final artifacts: {missing}")
    audit = read_json(output_dir / "independent_audit.json")
    if audit.get("status") != "passed":
        raise RuntimeError("Independent audit did not pass")
    remaining = list((output_dir / "images").glob("**/*.png"))
    if remaining:
        raise RuntimeError(f"Formal cleanup incomplete: {len(remaining)} images remain")
    validation = {
        "status": "complete",
        "validated_at": utc_now(),
        "active_protocol_fingerprint": protocol["active_protocol_fingerprint"],
        "completed_gcd_cells": len(completed_cell_paths(output_dir)),
        "prediction_rows": len(read_csv(output_dir / "raw" / "all_gcd_predictions.csv")),
        "checkpoint_count": len(list((output_dir / "checkpoints").glob("**/*.safetensors"))),
        "formal_image_cleanup": "complete",
        "coco_status": "deferred_by_budget",
        "qualitative_tarball": str(tarball.resolve()),
        "qualitative_tarball_sha256": sha256_file(tarball),
    }
    if validation["completed_gcd_cells"] != 41 or validation["checkpoint_count"] != 41:
        raise RuntimeError("Final checkpoint/cell counts differ from 41")
    write_json(output_dir / "final_validation.json", validation)
    manifest = read_json(output_dir / "run_manifest.json")
    manifest.update({
        "status": "complete", "completed_at": utc_now(),
        "final_validation": str((output_dir / "final_validation.json").resolve()),
        "qualitative_tarball": str(tarball.resolve()),
        "coco_status": "deferred_by_budget",
    })
    write_json(output_dir / "run_manifest.json", manifest)
    update_state(
        output_dir, status="complete", phase="complete",
        order="-", condition="-", step=10, current_target="-",
        completed_checkpoints=41, completed_evaluators=41,
    )


def finalize(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    aggregate_results(args, protocol)
    build_contact_sheets(args, protocol)
    tarball = package_qualitative(args, protocol)
    run_independent_audit(args)
    render_report(args, protocol, tarball)
    final_validation(args, protocol, tarball)


def run_all(args: argparse.Namespace) -> None:
    protocol = require_manifest(args, active=True)
    output_dir = output_path(args)
    try:
        generate_original_qualitative(args, protocol)
        update_state(output_dir, status="running", phase="checkpoint_edit")
        build_sequential_checkpoints(args, protocol)
        build_joint_checkpoint(args, protocol)
        evaluate_formal_cell(args, protocol, "joint", "joint_100", 10)
        update_state(output_dir, status="running", phase="aggregation")
        finalize(args, protocol)
    except BaseException as error:
        update_state(
            output_dir, status="failed_resumable", error=repr(error),
            resume_command="rerun the same --start command",
        )
        raise


def print_plan(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    validate_config(Path(args.config), config, artifacts=False)
    payload = {
        "experiment_name": config["experiment_name"],
        "orders": {
            order: concept_batches(config, order) for order in ORDERS
        },
        "conditions": list(CONDITIONS),
        "steps_per_trajectory": 10,
        "sequential_checkpoints": 40,
        "joint_reference_checkpoints": 1,
        "budget_profiles": config["budget"]["profile_image_counts"],
        "profile_selection": "200-image benchmark, 20% credit reserve",
        "coco": "deferred; never started by the core run",
        "launches_models": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def audit_only(args: argparse.Namespace) -> None:
    run_independent_audit(args)


def show_status(args: argparse.Namespace) -> None:
    output_dir = output_path(args)
    state = read_json(state_path(output_dir)) if state_path(output_dir).is_file() else {}
    final_path = output_dir / "final_validation.json"
    complete = (
        final_path.is_file() and read_json(final_path).get("status") == "complete"
    )
    checkpoints = len(list((output_dir / "checkpoints").glob("**/*.safetensors")))
    cells = len(completed_cell_paths(output_dir))
    generation_done = state.get("generation_cell_completed", 0)
    generation_total = state.get("generation_cell_total", 0)
    evaluator_done = state.get("evaluator_cell_completed", 0)
    evaluator_total = state.get("evaluator_cell_total", 0)
    payload = {
        "experiment_status": "complete" if complete else state.get("status", "not_started"),
        "phase": state.get("phase", "-"),
        "current_order": state.get("order", "-"),
        "current_condition": state.get("condition", "-"),
        "current_step": state.get("step", 0),
        "current_target": state.get("current_target", "-"),
        "generation_progress": f"{generation_done}/{generation_total}",
        "evaluator_progress": f"{evaluator_done}/{evaluator_total}",
        "completed_checkpoints": f"{checkpoints}/41",
        "completed_evaluator_cells": f"{cells}/41",
        "budget_profile": state.get("budget_profile", "unlocked"),
        "updated_at": state.get("updated_at", "-"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def continue_coco(args: argparse.Namespace) -> None:
    """Keep the expensive phase behind an explicit, protocol-safe gate.

    The repository registry currently points at machine-local FID artifacts.
    This command refuses to generate anything until a portable complete
    reference bundle exists on the active server.  That prevents accidentally
    recreating Original SD or producing edited images that cannot be scored.
    """
    protocol = require_manifest(args, active=True)
    output_dir = output_path(args)
    validation_path = output_dir / "final_validation.json"
    if not validation_path.is_file() or read_json(validation_path).get("status") != "complete":
        raise RuntimeError("Complete the core GCD experiment before the COCO phase")
    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    from reference_registry import resolve_reference

    checks = []
    for count in (1000, 10000):
        reference_id = protocol["config"]["coco"]["reference_ids"][str(count)]
        identity = {
            "model_id": protocol["config"]["model_id"],
            "prompt_source_sha256": sha256_file(COCO_SOURCE),
            "prompt_subset": f"first {count} rows in source order",
            "prompt_count": count,
            "seed_column": "evaluation_seed",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
            "dtype": "bfloat16",
            "scheduler": "PNDMScheduler",
            "clip_model_id": "openai/clip-vit-base-patch32",
            "clip_implementation": "transformers logits_per_image diagonal",
            "fid_implementation": "torch_fidelity 0.3.0",
            "fid_feature_extractor": "inception-v3-compat",
            "fid_feature_layer": "2048",
        }
        error = None
        try:
            entry = resolve_reference(reference_id, identity, require_complete=True)
        except (FileNotFoundError, RuntimeError) as caught:
            entry = None
            error = str(caught)
        checks.append({
            "count": count, "reference_id": reference_id,
            "status": entry.get("status") if entry else "unavailable",
            "fingerprint": entry.get("fingerprint") if entry else None,
            "expected_identity": identity,
            "registry_validation_error": error,
            "portable_and_complete": entry is not None,
        })
    payload = {
        "status": "blocked_missing_portable_reference_bundle",
        "reason": (
            "COCO is intentionally deferred. Copy/register the exact complete "
            "first-1k and first-10k Original reference bundles on this server "
            "before edited-model generation."
        ),
        "checks": checks,
        "checked_at": utc_now(),
    }
    write_json(output_dir / "coco_deferred.json", payload)
    if not all(row["portable_and_complete"] for row in checks):
        raise RuntimeError(payload["reason"])
    raise RuntimeError(
        "Portable references are present, but v1 deliberately leaves the costly "
        "COCO generation disabled. Enable it only in a separately reviewed budget revision."
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--gcd-project-root", type=Path)
    parser.add_argument("--allow-downloads", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    add_common_arguments(plan_parser)
    plan_parser.set_defaults(function=print_plan)

    preflight_parser = subparsers.add_parser("preflight")
    add_common_arguments(preflight_parser)
    preflight_parser.set_defaults(function=preflight)

    benchmark_parser = subparsers.add_parser("benchmark")
    add_common_arguments(benchmark_parser)
    benchmark_parser.add_argument("--remaining-credits", type=float, required=True)
    benchmark_parser.add_argument("--gpu-rate", type=float, required=True)
    benchmark_parser.set_defaults(function=benchmark)

    run_parser = subparsers.add_parser("run")
    add_common_arguments(run_parser)
    run_parser.set_defaults(function=run_all)

    audit_parser = subparsers.add_parser("audit")
    add_common_arguments(audit_parser)
    audit_parser.set_defaults(function=audit_only)

    status_parser = subparsers.add_parser("status")
    add_common_arguments(status_parser)
    status_parser.set_defaults(function=show_status)

    coco_parser = subparsers.add_parser("continue-coco")
    add_common_arguments(coco_parser)
    coco_parser.set_defaults(function=continue_coco)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
