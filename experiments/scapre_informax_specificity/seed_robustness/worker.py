#!/usr/bin/env python3
"""Resume-safe orchestration for ScaPre Informax edit-seed robustness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SEED_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SEED_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SCAPRE_ROOT = REPO_ROOT / "scapre"
BASE_CONFIG_PATH = EXPERIMENT_DIR / "config.json"
EVALUATOR = EXPERIMENT_DIR / "evaluate_confuse5.py"
PROTOCOL_BUILDER = EXPERIMENT_DIR / "build_protocol.py"
EDITOR = SCAPRE_ROOT / "edit" / "erase_scale.py"
RNG_RUNNER = SEED_DIR / "informax_seed_runner.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "formal"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=SEED_DIR / "config.json")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path)
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


def git_status() -> list[str]:
    output = git_output("status", "--porcelain", "--untracked-files=all")
    return output.splitlines() if output else []


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def accuracy(rows: list[dict[str, str]]) -> float:
    return 100.0 * sum(int(row["correct"]) for row in rows) / len(rows)


def overall(unlearn: float, preserve: float) -> float:
    forgetting = 100.0 - unlearn
    return 2.0 * forgetting * preserve / (forgetting + preserve)


def validate_configuration(config: dict, base_config: dict) -> None:
    expected_seeds = [20260820, 20260821, 20260822, 20260823, 20260824]
    if config.get("edit_seeds") != expected_seeds:
        raise RuntimeError("the five preregistered edit seeds changed")
    if config.get("new_edit_seeds") != expected_seeds[1:]:
        raise RuntimeError("the four new edit seeds changed")
    if config.get("fixed_non_informax_seed") != 20260820:
        raise RuntimeError("the non-Informax seed must remain 20260820")
    if config.get("variants") != ["official", "matched_retain"]:
        raise RuntimeError("the controlled variants changed")
    if base_config.get("edit_seed") != 20260820:
        raise RuntimeError("the previous experiment base config changed")
    if base_config.get("variants") != config["variants"]:
        raise RuntimeError("variant definitions disagree with the previous experiment")
    if base_config["edit"].get("num_positive") != 5 or base_config["edit"].get("num_negative") != 5:
        raise RuntimeError("Informax sample counts are no longer 5 positive + 5 negative")
    if base_config["edit"].get("matched_negative_assignment") != "round-robin in listed retain order (2/2/1)":
        raise RuntimeError("matched-retain 2/2/1 allocation changed")
    if len(base_config.get("groups", [])) != 5:
        raise RuntimeError("Confuse5 must contain all five groups")
    concepts = [concept for group in base_config["groups"] for field in ("targets", "retains") for concept in group[field]]
    if len(concepts) != 25 or len(set(concepts)) != 25:
        raise RuntimeError("Confuse5 must contain 25 unique concepts")
    for relative, expected_hash in config["source_controls"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"controlled source hash changed: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["required_ancestor_commit"], "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )


def matched_map(base_config: dict, profile: str) -> dict[str, list[str]]:
    groups = base_config["groups"][:1] if profile == "smoke" else base_config["groups"]
    return {
        target: list(group["retains"])
        for group in groups
        for target in group["targets"]
    }


def validate_prior_seed(prior: Path, config: dict) -> dict[str, object]:
    required = [
        "actual_config.json",
        "protocol.csv",
        "protocol_manifest.json",
        "run_manifest.json",
        "controlled_ablation_check.json",
        "results/aggregate.csv",
        "results/per_group.csv",
        "results/per_concept.csv",
        "results/informax_diagnostics.csv",
        "results/result_manifest.json",
        "evaluation/official/evaluation_manifest.json",
        "evaluation/official/scores.csv",
        "evaluation/matched_retain/evaluation_manifest.json",
        "evaluation/matched_retain/scores.csv",
        "exit_code",
        "COMPLETED",
    ]
    missing = [relative for relative in required if not (prior / relative).is_file()]
    if missing:
        raise RuntimeError(f"prior seed is missing required files: {missing}")
    if (prior / "exit_code").read_text().strip() != "0":
        raise RuntimeError("prior seed run did not exit successfully")
    actual_config = json.loads((prior / "actual_config.json").read_text())
    if actual_config.get("edit_seed") != 20260820:
        raise RuntimeError("prior run is not edit seed 20260820")
    protocol_manifest = json.loads((prior / "protocol_manifest.json").read_text())
    if protocol_manifest.get("sha256") != config["prior_seed"]["protocol_sha256"]:
        raise RuntimeError("prior protocol hash changed")
    if sha256(prior / "protocol.csv") != config["prior_seed"]["protocol_sha256"]:
        raise RuntimeError("prior protocol file does not match its pinned hash")
    if json.loads((prior / "controlled_ablation_check.json").read_text()).get("status") != "passed":
        raise RuntimeError("prior controlled-ablation check did not pass")
    if json.loads((prior / "results/result_manifest.json").read_text()).get("judgment") != config["prior_seed"]["expected_judgment"]:
        raise RuntimeError("prior scientific judgment changed")

    prior_manifest = json.loads((prior / "run_manifest.json").read_text())
    for relative in (
        "scapre/edit/erase_scale.py",
        "experiments/scapre_informax_specificity/config.json",
        "experiments/scapre_informax_specificity/evaluate_confuse5.py",
    ):
        if prior_manifest["source_sha256"].get(relative) != config["source_controls"][relative]:
            raise RuntimeError(f"prior source hash mismatch: {relative}")

    calculated: dict[str, float] = {}
    score_hashes: dict[str, str] = {}
    generation_keys: set[tuple[str, ...]] | None = None
    for variant in ("official", "matched_retain"):
        score_path = prior / "evaluation" / variant / "scores.csv"
        rows = read_csv(score_path)
        if len(rows) != 3000:
            raise RuntimeError(f"prior {variant} has {len(rows)} scores instead of 3000")
        target_rows = [row for row in rows if row["role"] == "target"]
        retain_rows = [row for row in rows if row["role"] == "retain"]
        if (len(target_rows), len(retain_rows)) != (1200, 1800):
            raise RuntimeError(f"prior {variant} target/retain denominators changed")
        keys = {
            tuple(row[field] for field in ("group", "role", "concept", "sample_index", "prompt", "seed", "seed_source"))
            for row in rows
        }
        if len(keys) != 3000:
            raise RuntimeError(f"prior {variant} contains duplicate generation keys")
        if generation_keys is None:
            generation_keys = keys
        elif generation_keys != keys:
            raise RuntimeError("prior variants use different generation keys")
        unlearn = accuracy(target_rows)
        preserve = accuracy(retain_rows)
        metric_prefix = "official" if variant == "official" else "matched"
        calculated[f"{metric_prefix}_unlearn_acc"] = unlearn
        calculated[f"{metric_prefix}_preserve_acc"] = preserve
        calculated[f"{metric_prefix}_overall_acc"] = overall(unlearn, preserve)
        score_hashes[variant] = sha256(score_path)

    for key, expected in config["prior_seed"]["expected_metrics"].items():
        if not math_isclose(calculated[key], expected):
            raise RuntimeError(f"prior metric mismatch for {key}: {calculated[key]} != {expected}")
    return {
        "path": str(prior.resolve()),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_manifest["sha256"],
        "score_sha256": score_hashes,
        "metrics": calculated,
        "legacy_git_commit": prior_manifest["git_commit"],
        "legacy_git_dirty": prior_manifest["git_dirty"],
        "legacy_dirty_caveat": "accepted only because all controlled source and result hashes were independently verified",
    }


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-10


def import_prior(prior: Path, destination: Path, manifest: dict[str, object]) -> None:
    if destination.exists():
        existing = json.loads((destination / "seed_source_manifest.json").read_text())
        if existing != manifest:
            raise RuntimeError("existing imported seed 20260820 has different provenance")
        return
    for variant in ("official", "matched_retain"):
        target = destination / "evaluation" / variant
        target.mkdir(parents=True, exist_ok=True)
        for name in ("scores.csv", "evaluation_manifest.json"):
            shutil.copy2(prior / "evaluation" / variant / name, target / name)
        (target / "COMPLETED").write_text("imported\n")
    (destination / "results").mkdir(parents=True, exist_ok=True)
    for name in ("aggregate.csv", "per_group.csv", "per_concept.csv", "informax_diagnostics.csv"):
        shutil.copy2(prior / "results" / name, destination / "results" / name)
    shutil.copy2(prior / "actual_config.json", destination / "actual_config.json")
    shutil.copy2(prior / "run_manifest.json", destination / "prior_run_manifest.json")
    (destination / "seed_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def edit_command(
    args: argparse.Namespace,
    base_config: dict,
    assets: dict,
    seed: int,
    seed_dir: Path,
    variant: str,
    targets: list[str],
    matched_config_path: Path,
    expected_calls: int,
) -> list[str]:
    checkpoint = seed_dir / "checkpoints" / f"{variant}.pt"
    diagnostics = seed_dir / "diagnostics" / f"{variant}.pt"
    audit = seed_dir / "stages" / f"informax_rng_{variant}.json"
    mode = "official" if variant == "official" else "matched-retain"
    edit = base_config["edit"]
    editor_args = [
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
        "--edit-seed", "20260820",
        "--output_model", str(checkpoint.resolve()),
    ]
    if variant == "matched_retain":
        editor_args.extend(["--informax-matched-retain-config", str(matched_config_path.resolve())])
    return [
        sys.executable,
        str(RNG_RUNNER),
        "--informax-seed", str(seed),
        "--script", str(EDITOR),
        "--audit-output", str(audit.resolve()),
        "--expected-randn-calls", str(expected_calls),
        "--",
        *editor_args,
    ]


def normalized_command(command: list[str]) -> list[str]:
    removed = {
        "--informax-negative-mode",
        "--informax-matched-retain-config",
        "--informax-diagnostics-path",
        "--output_model",
        "--audit-output",
    }
    normalized: list[str] = []
    index = 0
    while index < len(command):
        if command[index] in removed:
            index += 2
            continue
        normalized.append(command[index])
        index += 1
    return normalized


def run_edit(command: list[str], seed_dir: Path, variant: str) -> None:
    checkpoint = seed_dir / "checkpoints" / f"{variant}.pt"
    diagnostics = seed_dir / "diagnostics" / f"{variant}.pt"
    audit = seed_dir / "stages" / f"informax_rng_{variant}.json"
    completed = seed_dir / "stages" / f"edit_{variant}.completed"
    command_path = seed_dir / "stages" / f"edit_{variant}_command.json"
    payload = {"argv": command}
    if command_path.exists() and json.loads(command_path.read_text()) != payload:
        raise RuntimeError(f"seed command changed while resuming: {variant}")
    command_path.write_text(json.dumps(payload, indent=2) + "\n")
    if completed.exists():
        if not all(path.is_file() for path in (checkpoint, diagnostics, audit)):
            raise RuntimeError(f"completed edit stage is incomplete: {variant}")
        print(f"[resume] edit {variant}", flush=True)
        return
    if checkpoint.exists() or diagnostics.exists():
        raise RuntimeError(f"unverified partial edit output exists: {variant}")
    run(command, cwd=SCAPRE_ROOT)
    audit_payload = json.loads(audit.read_text())
    if not audit_payload.get("completed") or audit_payload["intercepted_randn_calls"] != audit_payload["expected_randn_calls"]:
        raise RuntimeError(f"Informax-only RNG audit failed: {variant}")
    completed.write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint),
        "diagnostics_sha256": sha256(diagnostics),
        "rng_audit_sha256": sha256(audit),
    }, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    start_status = git_status()
    if start_status:
        raise RuntimeError(f"working tree is dirty at run start: {start_status}")

    config = json.loads(args.config.read_text())
    base_config = json.loads(BASE_CONFIG_PATH.read_text())
    assets = json.loads(args.assets.read_text())
    validate_configuration(config, base_config)
    if args.profile == "formal" and args.prior_run is None:
        raise RuntimeError("formal robustness requires the validated seed-20260820 run")

    source_files = sorted(
        [EDITOR, BASE_CONFIG_PATH, EVALUATOR, PROTOCOL_BUILDER]
        + [path for path in SEED_DIR.iterdir() if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json", ".txt"}]
    )
    source_hashes = {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files}
    manifest_path = args.run_dir / "run_manifest.json"
    new_manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_start": start_status,
        "source_sha256": source_hashes,
        "assets": assets,
        "device": args.device,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key in ("profile", "git_commit", "source_sha256", "assets", "device"):
            if manifest.get(key) != new_manifest.get(key):
                raise RuntimeError(f"refusing resume because run manifest changed: {key}")
    else:
        manifest = new_manifest
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    (args.run_dir / "actual_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.run_dir / "base_config.json").write_text(json.dumps(base_config, indent=2) + "\n")
    provenance = args.run_dir / "provenance"
    for source in source_files:
        destination = provenance / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    protocol_path = args.run_dir / "protocol.csv"
    protocol_output = subprocess.check_output([
        sys.executable, str(PROTOCOL_BUILDER),
        "--config", str(BASE_CONFIG_PATH),
        "--output", str(protocol_path),
        "--profile", args.profile,
    ], cwd=REPO_ROOT, text=True)
    protocol_manifest = json.loads(protocol_output)
    (args.run_dir / "protocol_manifest.json").write_text(json.dumps(protocol_manifest, indent=2) + "\n")
    if args.profile == "formal" and protocol_manifest["sha256"] != config["prior_seed"]["protocol_sha256"]:
        raise RuntimeError("newly built formal protocol differs from seed 20260820")

    mapping = matched_map(base_config, args.profile)
    matched_path = args.run_dir / "matched_retain_config.json"
    matched_path.write_text(json.dumps({"matched_retain_by_target": mapping}, indent=2) + "\n")
    targets = list(mapping)
    seeds = config["new_edit_seeds"] if args.profile == "formal" else [20260821]

    if args.profile == "formal":
        prior = args.prior_run.resolve()
        prior_manifest = validate_prior_seed(prior, config)
        (args.run_dir / "reproducibility").mkdir(exist_ok=True)
        (args.run_dir / "reproducibility" / "prior_seed_validation.json").write_text(json.dumps(prior_manifest, indent=2) + "\n")
        import_prior(prior, args.run_dir / "seeds" / "20260820", prior_manifest)

    expected_calls = config[
        "expected_informax_randn_calls_per_formal_edit"
        if args.profile == "formal"
        else "expected_informax_randn_calls_per_smoke_edit"
    ]
    for seed in seeds:
        seed_dir = args.run_dir / "seeds" / str(seed)
        for name in ("checkpoints", "diagnostics", "evaluation", "results", "stages"):
            (seed_dir / name).mkdir(parents=True, exist_ok=True)
        seed_config = dict(base_config)
        seed_config["informax_seed"] = seed
        seed_config["fixed_non_informax_seed"] = config["fixed_non_informax_seed"]
        seed_config_path = seed_dir / "actual_config.json"
        if seed_config_path.exists() and json.loads(seed_config_path.read_text()) != seed_config:
            raise RuntimeError(f"seed {seed} config changed while resuming")
        seed_config_path.write_text(json.dumps(seed_config, indent=2) + "\n")
        (seed_dir / "source_manifest.json").write_text(json.dumps(source_hashes, indent=2) + "\n")

        commands = {
            variant: edit_command(
                args, base_config, assets, seed, seed_dir, variant, targets,
                matched_path, expected_calls,
            )
            for variant in config["variants"]
        }
        for variant, command in commands.items():
            run_edit(command, seed_dir, variant)
        if normalized_command(commands["official"]) != normalized_command(commands["matched_retain"]):
            raise RuntimeError(f"seed {seed} variants differ outside the intended intervention")
        official_rng = json.loads((seed_dir / "stages" / "informax_rng_official.json").read_text())
        matched_rng = json.loads((seed_dir / "stages" / "informax_rng_matched_retain.json").read_text())
        for key in ("informax_seed", "intercepted_randn_calls", "expected_randn_calls", "shape_counts", "global_rng_legacy_draws_consumed"):
            if official_rng[key] != matched_rng[key]:
                raise RuntimeError(f"seed {seed} RNG audit differs between variants: {key}")
        (seed_dir / "controlled_ablation_check.json").write_text(json.dumps({
            "status": "passed",
            "seed": seed,
            "fixed_non_informax_seed": config["fixed_non_informax_seed"],
            "same_normalized_edit_command": True,
            "same_informax_noise_shape_stream": True,
            "method_source_sha256": config["source_controls"]["scapre/edit/erase_scale.py"],
            "intended_differences": ["Informax negative/reference source", "variant-specific output paths"],
        }, indent=2) + "\n")

        for variant in config["variants"]:
            evaluation_dir = seed_dir / "evaluation" / variant
            if (evaluation_dir / "COMPLETED").exists():
                print(f"[resume] seed {seed} evaluation {variant}", flush=True)
                continue
            run([
                sys.executable, str(EVALUATOR),
                "--config", str(seed_config_path),
                "--assets", str(args.assets.resolve()),
                "--protocol", str(protocol_path),
                "--checkpoint", str(seed_dir / "checkpoints" / f"{variant}.pt"),
                "--variant", variant,
                "--output-dir", str(evaluation_dir),
                "--device", args.device,
            ])

    manifest["git_status_end"] = git_status()
    manifest["calculation_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if manifest["git_status_end"]:
        raise RuntimeError(f"working tree became dirty during run: {manifest['git_status_end']}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run([
        sys.executable, str(SEED_DIR / "aggregate_seed_results.py"),
        "--run-dir", str(args.run_dir),
        "--config", str(args.config.resolve()),
        "--base-config", str(BASE_CONFIG_PATH),
        "--profile", args.profile,
    ])
    final_status = git_status()
    if final_status:
        raise RuntimeError(f"aggregation changed the tracked working tree: {final_status}")
    manifest["git_status_end"] = final_status
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(args.run_dir / "results" / "summary.md", args.run_dir / "summary.md")
    (args.run_dir / "worker_complete.json").write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "edit_seeds": config["edit_seeds"] if args.profile == "formal" else [20260821],
        "summary": str((args.run_dir / "results" / "summary.md").resolve()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
