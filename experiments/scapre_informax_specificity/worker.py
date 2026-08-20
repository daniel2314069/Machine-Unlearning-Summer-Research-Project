#!/usr/bin/env python3
"""Resume-safe orchestration for the ScaPre Informax specificity experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SCAPRE_ROOT = REPO_ROOT / "scapre"
DERIVED_DATASET = (
    REPO_ROOT
    / "orthogonal-concept-erasure"
    / "experiments"
    / "confuse5_single_vs_joint"
    / "datasets"
    / "imagenet-confuse5-derived-25.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "formal"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "config.json")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def matched_map(config: dict, profile: str) -> dict[str, list[str]]:
    groups = config["groups"][:1] if profile == "smoke" else config["groups"]
    return {
        target: list(group["retains"])
        for group in groups
        for target in group["targets"]
    }


def validate_config(config: dict) -> None:
    if config.get("variants") != ["official", "matched_retain"]:
        raise RuntimeError("config must contain exactly the two controlled variants")
    groups = config.get("groups", [])
    if len(groups) != 5:
        raise RuntimeError("config must contain all five Confuse5 groups")
    concepts: list[str] = []
    for group in groups:
        targets = group.get("targets", [])
        retains = group.get("retains", [])
        if len(targets) != 2 or len(retains) != 3 or set(targets) & set(retains):
            raise RuntimeError(f"invalid target/retain definition for {group.get('id')}")
        concepts.extend(targets)
        concepts.extend(retains)
    if len(concepts) != 25 or len(set(concepts)) != 25:
        raise RuntimeError("Confuse5 concepts must be 25 unique labels")
    edit = config.get("edit", {})
    if edit.get("num_positive") != 5 or edit.get("num_negative") != 5:
        raise RuntimeError("repository-controlled Informax sample counts must remain 5/5")
    expected_edit = {
        "base": "1.5",
        "concept_type": "object",
        "erase_scale": 2.0,
        "p": 8.0,
        "bures_iters": 1,
        "enable_ased": True,
        "entropy_samples": 30,
        "entropy_bins": 20,
        "noise_sigma": 0.01,
        "T_sigma": 1.0,
        "p_sigma": 1.0,
        "lamb_effective": 0.5,
    }
    for key, value in expected_edit.items():
        if edit.get(key) != value:
            raise RuntimeError(f"official ScaPre control changed in config: edit.{key}")
    expected_evaluation = {
        "formal_images_per_concept": 120,
        "smoke_images_per_concept": 2,
        "prompt_template": "an image of a {concept}",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "dtype": "float16",
        "sampler": "base-model scheduler config",
        "classifier": "torchvision ResNet50_Weights.DEFAULT",
        "classifier_decision": "top-1",
        "label_match": "lowercase substring in either direction",
        "missing_seed_scheme": (
            "reuse the ordered seeds of the same-group retain class present in "
            "public imagenet-15.csv"
        ),
    }
    evaluation = config.get("evaluation", {})
    for key, value in expected_evaluation.items():
        if evaluation.get(key) != value:
            raise RuntimeError(f"formal evaluation control changed in config: {key}")
    if config.get("base_model") != "runwayml/stable-diffusion-v1-5":
        raise RuntimeError("base model must remain the repository's official SD 1.5 model")
    if config.get("edit_seed") != 20260820:
        raise RuntimeError("the preregistered edit seed changed")
    if evaluation.get("protocol_dataset") != str(DERIVED_DATASET.relative_to(REPO_ROOT)):
        raise RuntimeError("protocol dataset must remain the established project asset")
    if evaluation.get("protocol_dataset_sha256") != (
        "f473503dd5a008f989a107e5adfe0749e9e2e77d8f613f2b7a4aae8bd87301d9"
    ):
        raise RuntimeError("protocol dataset hash changed")


def edit_variant(
    args: argparse.Namespace,
    config: dict,
    assets: dict,
    variant: str,
    targets: list[str],
    matched_config_path: Path,
) -> list[str]:
    checkpoint = args.run_dir / "checkpoints" / f"{variant}.pt"
    diagnostics = args.run_dir / "diagnostics" / f"{variant}.pt"
    completed = args.run_dir / "stages" / f"edit_{variant}.completed"
    edit = config["edit"]
    mode = "official" if variant == "official" else "matched-retain"
    cmd = [
        sys.executable,
        "edit/erase_scale.py",
        "--concepts", ", ".join(targets),
        "--concept_type", edit["concept_type"],
        "--device", args.device.removeprefix("cuda:"),
        "--base", edit["base"],
        "--model-id-or-path", assets["snapshot_path"],
        "--use_mi_softmask",
        "--erase_scale", str(edit["erase_scale"]),
        "--p", str(edit["p"]),
        "--bures_iters", str(edit["bures_iters"]),
        "--enable_ased",
        "--entropy_samples", str(edit["entropy_samples"]),
        "--entropy_bins", str(edit["entropy_bins"]),
        "--noise_sigma", str(edit["noise_sigma"]),
        "--T_sigma", str(edit["T_sigma"]),
        "--p_sigma", str(edit["p_sigma"]),
        "--informax-negative-mode", mode,
        "--informax-diagnostics-path", str(diagnostics.resolve()),
        "--edit-seed", str(config["edit_seed"]),
        "--output_model", str(checkpoint.resolve()),
    ]
    if variant == "matched_retain":
        cmd.extend(["--informax-matched-retain-config", str(matched_config_path.resolve())])

    command_manifest = args.run_dir / "stages" / f"edit_{variant}_command.json"
    command_payload = {"argv": cmd}
    if command_manifest.exists():
        previous = json.loads(command_manifest.read_text())
        if previous != command_payload:
            raise RuntimeError(f"refusing to resume {variant}: edit command changed")
    else:
        command_manifest.write_text(json.dumps(command_payload, indent=2) + "\n")

    if completed.exists():
        if not checkpoint.exists() or not diagnostics.exists():
            raise RuntimeError(f"completed edit stage is missing outputs: {variant}")
        print(f"[resume] edit {variant}", flush=True)
        return cmd
    if checkpoint.exists() or diagnostics.exists():
        raise RuntimeError(
            f"unverified partial edit outputs exist for {variant}; use a new run id"
        )

    run(cmd, cwd=SCAPRE_ROOT)
    if not checkpoint.exists() or not diagnostics.exists():
        raise RuntimeError(f"edit stage did not create required outputs: {variant}")
    completed.write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint),
        "diagnostics_sha256": sha256(diagnostics),
    }, indent=2) + "\n")
    return cmd


def normalized_edit_command(command: list[str]) -> list[str]:
    """Remove the intended intervention and variant-specific output paths."""
    variant_flags = {
        "--informax-negative-mode",
        "--informax-matched-retain-config",
        "--informax-diagnostics-path",
        "--output_model",
    }
    normalized: list[str] = []
    index = 0
    while index < len(command):
        if command[index] in variant_flags:
            index += 2
            continue
        normalized.append(command[index])
        index += 1
    return normalized


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "diagnostics", "evaluation", "results", "stages"):
        (args.run_dir / name).mkdir(exist_ok=True)
    config = json.loads(args.config.read_text())
    validate_config(config)
    assets = json.loads(args.assets.read_text())
    actual_config_path = args.run_dir / "actual_config.json"
    if actual_config_path.exists() and json.loads(actual_config_path.read_text()) != config:
        raise RuntimeError("run directory contains a different actual_config.json")
    actual_config_path.write_text(json.dumps(config, indent=2) + "\n")

    source_files = [
        SCAPRE_ROOT / "edit" / "erase_scale.py",
        DERIVED_DATASET,
    ] + sorted(
        path
        for path in EXPERIMENT_DIR.iterdir()
        if path.is_file()
        and (path.suffix in {".py", ".sh", ".md", ".json", ".txt"})
    )
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files},
        "assets": assets,
        "device": args.device,
    }
    run_manifest_path = args.run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_text())
        for key in ("profile", "git_commit", "source_sha256", "assets", "device"):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(f"refusing to resume run: manifest changed for {key}")
        manifest = previous
    else:
        run_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    provenance_root = args.run_dir / "provenance"
    for source in source_files:
        destination = provenance_root / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    protocol_path = args.run_dir / "protocol.csv"
    protocol_output = subprocess.check_output([
        sys.executable, str(EXPERIMENT_DIR / "build_protocol.py"),
        "--config", str(args.config.resolve()),
        "--output", str(protocol_path),
        "--profile", args.profile,
    ], cwd=REPO_ROOT, text=True)
    protocol_manifest = json.loads(protocol_output)
    (args.run_dir / "protocol_manifest.json").write_text(
        json.dumps(protocol_manifest, indent=2) + "\n"
    )

    mapping = matched_map(config, args.profile)
    matched_config_path = args.run_dir / "matched_retain_config.json"
    matched_config_path.write_text(json.dumps({
        "matched_retain_by_target": mapping,
    }, indent=2) + "\n")
    targets = list(mapping)

    edit_commands: dict[str, list[str]] = {}
    for variant in ("official", "matched_retain"):
        edit_commands[variant] = edit_variant(
            args, config, assets, variant, targets, matched_config_path
        )

    official_normalized = normalized_edit_command(edit_commands["official"])
    matched_normalized = normalized_edit_command(edit_commands["matched_retain"])
    if official_normalized != matched_normalized:
        raise RuntimeError(
            "controlled-ablation check failed: edit commands differ outside the intervention"
        )
    (args.run_dir / "controlled_ablation_check.json").write_text(json.dumps({
        "status": "passed",
        "same_normalized_edit_command": True,
        "intended_difference": (
            "Informax negative/reference source only: official empty prompt versus "
            "balanced same-group retain concepts"
        ),
        "removed_for_comparison": [
            "--informax-negative-mode",
            "--informax-matched-retain-config",
            "--informax-diagnostics-path",
            "--output_model",
        ],
        "normalized_argv": official_normalized,
    }, indent=2) + "\n")

    for variant in ("official", "matched_retain"):
        evaluation_dir = args.run_dir / "evaluation" / variant
        if (evaluation_dir / "COMPLETED").exists():
            print(f"[resume] evaluation {variant}", flush=True)
            continue
        run([
            sys.executable, str(EXPERIMENT_DIR / "evaluate_confuse5.py"),
            "--config", str(actual_config_path),
            "--assets", str(args.assets.resolve()),
            "--protocol", str(protocol_path),
            "--checkpoint", str(args.run_dir / "checkpoints" / f"{variant}.pt"),
            "--variant", variant,
            "--output-dir", str(evaluation_dir),
            "--device", args.device,
        ])

    run([
        sys.executable, str(EXPERIMENT_DIR / "aggregate_results.py"),
        "--config", str(actual_config_path),
        "--run-dir", str(args.run_dir),
        "--profile", args.profile,
    ])
    shutil.copy2(args.run_dir / "results" / "summary.md", args.run_dir / "summary.md")
    (args.run_dir / "worker_complete.json").write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "summary": str((args.run_dir / "results" / "summary.md").resolve()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
