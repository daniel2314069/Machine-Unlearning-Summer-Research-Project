#!/usr/bin/env python3
"""Resume-safe GPU worker for final Informax alpha-channel controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
REPO_ROOT = EXPERIMENT.parents[1]
SCAPRE_ROOT = REPO_ROOT / "scapre"
BASE_CONFIG = EXPERIMENT / "config.json"
PROTOCOL_BUILDER = EXPERIMENT / "build_protocol.py"
EDITOR = SCAPRE_ROOT / "edit" / "erase_scale.py"
EVALUATOR = EXPERIMENT / "evaluate_confuse5.py"
ALPHA_RUNNER = HERE / "alpha_control_runner.py"
EVAL_RUNNER = HERE / "evaluate_control_runner.py"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "formal"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--official-reference", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()


def git_status() -> list[str]:
    value = git("status", "--porcelain", "--untracked-files=all")
    return value.splitlines() if value else []


def run(command: list[str], cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_sources(config: dict) -> None:
    for relative, expected in config["source_controls"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"controlled source changed: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["required_ancestor_commit"], "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )


def controlled_assets(value: dict) -> dict:
    keys = (
        "base_model", "resolved_revision", "snapshot_path", "downloaded_files",
        "resnet_weights", "resnet_url", "packages", "config_sha256",
    )
    return {key: value.get(key) for key in keys}


def score_keys(rows: list[dict[str, str]]) -> set[tuple[str, ...]]:
    fields = ("group", "role", "concept", "sample_index", "prompt", "seed", "seed_source")
    return {tuple(row[field] for field in fields) for row in rows}


def validate_official_reference(reference: Path, config: dict, assets: dict, protocol: Path) -> dict:
    required = ["run_manifest.json", "protocol.csv", "protocol_manifest.json"]
    for seed in config["edit_seeds"]:
        required.extend([
            f"seeds/{seed}/evaluation/official/scores.csv",
            f"seeds/{seed}/evaluation/official/evaluation_manifest.json",
        ])
    missing = [name for name in required if not (reference / name).is_file()]
    if missing:
        raise RuntimeError(f"official reference is incomplete: {missing}")
    if sha256(reference / "protocol.csv") != config["official_reference"]["protocol_sha256"]:
        raise RuntimeError("official reference protocol hash mismatch")
    if sha256(protocol) != config["official_reference"]["protocol_sha256"]:
        raise RuntimeError("current formal protocol differs from official reference")
    manifest = json.loads((reference / "run_manifest.json").read_text())
    official_config = config["official_reference"]
    if manifest.get("git_commit") != official_config["run_commit"]:
        raise RuntimeError("official reference run commit changed")
    historical_sources = manifest.get("source_sha256", {})
    historical_editor = historical_sources.get("scapre/edit/erase_scale.py")
    expected_editor = official_config["editor_source_sha256"]
    if historical_editor != expected_editor:
        raise RuntimeError(
            "official reference editor source hash changed "
            f"(actual={historical_editor!r}, expected={expected_editor!r}, "
            f"manifest_sha256={sha256(reference / 'run_manifest.json')})"
        )
    evaluator_key = "experiments/scapre_informax_specificity/evaluate_confuse5.py"
    historical_evaluator = historical_sources.get(evaluator_key)
    expected_evaluator = official_config["evaluator_source_sha256"]
    if historical_evaluator != expected_evaluator:
        raise RuntimeError(
            "official reference evaluator source hash changed "
            f"(actual={historical_evaluator!r}, expected={expected_evaluator!r}, "
            f"actual_type={type(historical_evaluator).__name__}, "
            f"expected_type={type(expected_evaluator).__name__}, "
            f"actual_length={len(historical_evaluator) if isinstance(historical_evaluator, str) else None}, "
            f"expected_length={len(expected_evaluator) if isinstance(expected_evaluator, str) else None}, "
            f"manifest_sha256={sha256(reference / 'run_manifest.json')})"
        )
    compatibility_diff = subprocess.check_output([
        "git", "diff", f"{official_config['run_commit']}..HEAD", "--",
        "scapre/edit/erase_scale.py",
        "experiments/scapre_informax_specificity/evaluate_confuse5.py",
    ], cwd=REPO_ROOT)
    if hashlib.sha256(compatibility_diff).hexdigest() != official_config["compatibility_diff_sha256"]:
        raise RuntimeError(
            "production/evaluator changes since the official run are no longer "
            "the audited additive-only diff"
        )
    if controlled_assets(manifest["assets"]) != controlled_assets(assets):
        raise RuntimeError("official reference model/classifier assets differ")

    expected_keys = score_keys(read_csv(protocol))
    evaluator_control: dict | None = None
    details: dict[str, object] = {}
    for seed in config["edit_seeds"]:
        scores = reference / "seeds" / str(seed) / "evaluation" / "official" / "scores.csv"
        rows = read_csv(scores)
        if len(rows) != 3000 or len(score_keys(rows)) != 3000 or score_keys(rows) != expected_keys:
            raise RuntimeError(f"official seed {seed} does not have the frozen 3,000 rows")
        if Counter(row["role"] for row in rows) != {"target": 1200, "retain": 1800}:
            raise RuntimeError(f"official seed {seed} target/retain counts changed")
        expected_hash = config["official_reference"]["score_sha256"][str(seed)]
        if sha256(scores) != expected_hash:
            raise RuntimeError(f"official seed {seed} score hash mismatch")
        evaluation_manifest = json.loads((scores.parent / "evaluation_manifest.json").read_text())
        current = {
            key: value for key, value in evaluation_manifest.items()
            if key not in {"variant", "checkpoint_sha256"}
        }
        scheduler = current.get("scheduler_config", {})
        defaults = scheduler.get("_use_default_values")
        if isinstance(defaults, list):
            scheduler["_use_default_values"] = sorted(defaults)
        if evaluator_control is None:
            evaluator_control = current
        elif current != evaluator_control:
            raise RuntimeError("official evaluator fingerprints differ across seeds")
        details[str(seed)] = {"score_sha256": expected_hash, "rows": len(rows)}
    fingerprint = hashlib.sha256(
        json.dumps(evaluator_control, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != config["official_reference"]["evaluator_fingerprint_sha256"]:
        raise RuntimeError("official evaluator canonical fingerprint mismatch")
    return {"status": "passed", "path": str(reference), "seeds": details,
            "evaluator_fingerprint_sha256": fingerprint,
            "historical_source_hashes_verified": True,
            "official_code_path_compatibility_diff_sha256": official_config["compatibility_diff_sha256"],
            "compatibility_diff_is_additive_non_official_modes_only": True}


def import_official(reference: Path, run_dir: Path, seeds: list[int]) -> None:
    for seed in seeds:
        source = reference / "seeds" / str(seed) / "evaluation" / "official"
        destination = run_dir / "seeds" / str(seed) / "evaluation" / "official"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("scores.csv", "evaluation_manifest.json"):
            if not (destination / name).exists():
                shutil.copy2(source / name, destination / name)
        (destination / "COMPLETED").write_text("verified reference reuse\n")


def shuffle_salt(config: dict, variant: str) -> str:
    if variant == "shuffled":
        return config["formal_shuffle_salt"]
    return config["alternate_shuffle_salts"].get(variant, "not-applicable")


def edit_command(ns: argparse.Namespace, config: dict, base: dict, assets: dict,
                 seed: int, seed_dir: Path, variant: str, targets: list[str]) -> list[str]:
    edit = base["edit"]
    profile_calls = config[
        "expected_informax_randn_calls_formal" if ns.profile == "formal"
        else "expected_informax_randn_calls_smoke"
    ]
    editor_args = [
        "--concepts", ", ".join(targets),
        "--concept_type", edit["concept_type"],
        "--device", ns.device.removeprefix("cuda:"),
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
        "--informax-negative-mode", "official",
        "--informax-diagnostics-path", str((seed_dir / "diagnostics" / f"{variant}.pt").resolve()),
        "--edit-seed", str(config["fixed_non_informax_seed"]),
        "--output_model", str((seed_dir / "checkpoints" / f"{variant}.pt").resolve()),
    ]
    return [
        sys.executable, str(ALPHA_RUNNER),
        "--variant", variant,
        "--informax-seed", str(seed),
        "--informax-rng-mode", "legacy" if seed == config["legacy_informax_seed"] else "isolated",
        "--shuffle-salt", shuffle_salt(config, variant),
        "--script", str(EDITOR),
        "--audit-output", str((seed_dir / "alpha_audits" / f"{variant}.json").resolve()),
        "--expected-randn-calls", str(profile_calls),
        "--expected-alpha-intercepts", str(config["expected_alpha_intercepts_per_edit"]),
        "--layers-per-projection", str(config["expected_layers_per_projection"]),
        "--", *editor_args,
    ]


def normalized_command(command: list[str]) -> list[str]:
    remove = {"--variant", "--shuffle-salt", "--audit-output", "--informax-diagnostics-path", "--output_model"}
    output = []
    index = 0
    while index < len(command):
        if command[index] in remove:
            index += 2
        else:
            output.append(command[index])
            index += 1
    return output


def run_edit(command: list[str], seed_dir: Path, variant: str) -> None:
    paths = {
        "checkpoint": seed_dir / "checkpoints" / f"{variant}.pt",
        "diagnostics": seed_dir / "diagnostics" / f"{variant}.pt",
        "audit": seed_dir / "alpha_audits" / f"{variant}.json",
        "completed": seed_dir / "stages" / f"edit_{variant}.completed.json",
        "command": seed_dir / "stages" / f"edit_{variant}.command.json",
        "cleanup": seed_dir / "stages" / f"checkpoint_{variant}.cleanup.json",
    }
    payload = {"argv": command}
    if paths["command"].exists() and json.loads(paths["command"].read_text()) != payload:
        raise RuntimeError(f"resume command changed for {variant}")
    paths["command"].write_text(json.dumps(payload, indent=2) + "\n")
    if paths["completed"].exists():
        checkpoint_accounted = paths["checkpoint"].is_file() or paths["cleanup"].is_file()
        if not checkpoint_accounted or not all(paths[key].is_file() for key in ("diagnostics", "audit")):
            raise RuntimeError(f"completed edit is incomplete: {variant}")
        print(f"[resume] edit {variant}", flush=True)
        return
    if any(paths[key].exists() for key in ("checkpoint", "diagnostics", "audit")):
        raise RuntimeError(f"unverified partial edit exists: {variant}")
    run(command, cwd=SCAPRE_ROOT)
    audit = json.loads(paths["audit"].read_text())
    if not audit.get("completed") or not audit["checkpoint_finiteness"]["all_projection_weights_finite"]:
        raise RuntimeError(f"alpha/checkpoint audit failed: {variant}")
    paths["completed"].write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(paths["checkpoint"]),
        "diagnostics_sha256": sha256(paths["diagnostics"]),
        "audit_sha256": sha256(paths["audit"]),
    }, indent=2) + "\n")


def cleanup_checkpoint(seed_dir: Path, variant: str) -> None:
    checkpoint = seed_dir / "checkpoints" / f"{variant}.pt"
    marker = seed_dir / "stages" / f"checkpoint_{variant}.cleanup.json"
    completed = seed_dir / "stages" / f"edit_{variant}.completed.json"
    if marker.exists():
        if checkpoint.exists():
            raise RuntimeError(f"checkpoint cleanup marker exists but file returned: {variant}")
        return
    if not checkpoint.is_file() or not completed.is_file():
        raise RuntimeError(f"cannot account for checkpoint cleanup: {variant}")
    edit_record = json.loads(completed.read_text())
    if sha256(checkpoint) != edit_record["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash changed before cleanup: {variant}")
    size = checkpoint.stat().st_size
    checkpoint.unlink()
    marker.write_text(json.dumps({
        "status": "passed", "variant": variant,
        "deleted_regenerable_checkpoint": str(checkpoint),
        "sha256": edit_record["checkpoint_sha256"], "size_bytes": size,
        "deleted_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")


def record_key(record: dict) -> tuple:
    return (record.get("projection"), record.get("layer_index"), record.get("stage"),
            record.get("target_index"), record.get("target_concept"))


def assert_tensor_equal(left: object, right: object, label: str) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if not torch.equal(left, right):
            raise RuntimeError(f"diagnostic tensor changed: {label}")
    elif left != right:
        raise RuntimeError(f"diagnostic value changed: {label}")


def validate_edit_isolation(seed_dir: Path, variants: list[str], commands: dict[str, list[str]]) -> dict:
    baseline_command = normalized_command(commands[variants[0]])
    if any(normalized_command(commands[variant]) != baseline_command for variant in variants[1:]):
        raise RuntimeError("edit commands differ outside final-alpha controls/output paths")
    audits = {variant: json.loads((seed_dir / "alpha_audits" / f"{variant}.json").read_text()) for variant in variants}
    baseline_shapes = audits[variants[0]]["informax_randn_shape_counts"]
    for variant, audit in audits.items():
        if audit["informax_randn_shape_counts"] != baseline_shapes:
            raise RuntimeError(f"Informax RNG tensor signatures changed: {variant}")
        if audit["alpha_intercepts"] != audit["expected_alpha_intercepts"]:
            raise RuntimeError(f"alpha intercept coverage failed: {variant}")
        expected_mode = "legacy" if audit["informax_seed"] == 20260820 else "isolated"
        if audit["informax_rng_mode"] != expected_mode:
            raise RuntimeError(f"Informax RNG mode changed: {variant}")

    diagnostics = {variant: torch.load(seed_dir / "diagnostics" / f"{variant}.pt", map_location="cpu") for variant in variants}
    baseline = diagnostics[variants[0]]
    baseline_records = {record_key(record): record for record in baseline["records"] if record["stage"] != "aggregate-max"}
    fields = ("raw_mi", "alpha", "threshold", "negative_base_indices", "negative_concepts")
    for variant in variants[1:]:
        current = diagnostics[variant]
        if {key: value for key, value in current.items() if key != "records"} != {
            key: value for key, value in baseline.items() if key != "records"
        }:
            raise RuntimeError(f"non-record Informax metadata changed: {variant}")
        current_records = {record_key(record): record for record in current["records"] if record["stage"] != "aggregate-max"}
        if set(current_records) != set(baseline_records):
            raise RuntimeError(f"raw Informax record coverage changed: {variant}")
        for key, expected in baseline_records.items():
            for field in fields:
                assert_tensor_equal(expected.get(field), current_records[key].get(field), f"{variant}:{key}:{field}")
    report = {
        "status": "passed", "same_normalized_edit_command": True,
        "same_raw_mi": True, "same_preaggregate_alpha": True,
        "same_thresholds": True, "same_negative_base_indices": True,
        "same_informax_rng_tensor_signatures": True,
        "established_legacy_vs_isolated_seed_semantics": True,
        "official_empty_string_neutral_only": True,
        "variants": variants,
    }
    (seed_dir / "controlled_ablation_check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def evaluate(ns: argparse.Namespace, base_config_path: Path, assets: Path, protocol: Path,
             seed_dir: Path, variant: str) -> None:
    output = seed_dir / "evaluation" / variant
    if (output / "COMPLETED").exists():
        manifest = json.loads((output / "evaluation_manifest.json").read_text())
        rows = read_csv(output / "scores.csv")
        if manifest.get("variant") != variant or not rows or any(row.get("variant") != variant for row in rows):
            raise RuntimeError(f"completed evaluation has invalid variant labels: {variant}")
        print(f"[resume] evaluation {variant}", flush=True)
        return
    evaluator_prefix = (
        [sys.executable, str(EVALUATOR), "--variant", "official"]
        if variant == "official"
        else [sys.executable, str(EVAL_RUNNER), "--variant", variant,
              "--script", str(EVALUATOR), "--"]
    )
    run([*evaluator_prefix,
        "--config", str(base_config_path), "--assets", str(assets),
        "--protocol", str(protocol),
        "--checkpoint", str(seed_dir / "checkpoints" / f"{variant}.pt"),
        "--output-dir", str(output), "--device", ns.device,
    ])


def main() -> None:
    ns = args()
    ns.run_dir = ns.run_dir.resolve()
    if git_status():
        raise RuntimeError(f"working tree is dirty at start: {git_status()}")
    config = json.loads((HERE / "config.json").read_text())
    base = json.loads(BASE_CONFIG.read_text())
    assets = json.loads(ns.assets.read_text())
    validate_sources(config)
    if os.environ.get("CONDA_DEFAULT_ENV") != "MU":
        raise RuntimeError("worker requires active Conda MU")
    if base["edit_seed"] != config["fixed_non_informax_seed"] or base["edit"]["num_positive"] != 5 or base["edit"]["num_negative"] != 5:
        raise RuntimeError("established edit seed or Informax 5+5 protocol changed")
    ns.run_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        [REPO_ROOT / relative for relative in config["source_controls"]]
        + [path for path in HERE.iterdir() if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json"}]
    )
    source_hashes = {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files}
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": ns.profile, "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"), "git_status_start": [],
        "python_executable": sys.executable, "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "source_sha256": source_hashes, "assets": assets, "assets_sha256": sha256(ns.assets),
    }
    manifest_path = ns.run_dir / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("profile", "git_commit", "git_branch", "source_sha256", "assets", "assets_sha256"):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(f"resume provenance changed: {key}")
        manifest = old
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(HERE / "config.json", ns.run_dir / "actual_config.json")
    shutil.copy2(BASE_CONFIG, ns.run_dir / "base_config.json")
    for source in source_files:
        destination = ns.run_dir / "provenance" / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    protocol = ns.run_dir / "protocol.csv"
    output = subprocess.check_output([
        sys.executable, str(PROTOCOL_BUILDER), "--config", str(BASE_CONFIG),
        "--output", str(protocol), "--profile", ns.profile,
    ], cwd=REPO_ROOT, text=True)
    (ns.run_dir / "protocol_manifest.json").write_text(json.dumps(json.loads(output), indent=2) + "\n")

    official_validation = None
    if ns.profile == "formal":
        if ns.official_reference is None:
            raise RuntimeError("formal run requires an official reference")
        official_validation = validate_official_reference(ns.official_reference.resolve(), config, assets, protocol)
        (ns.run_dir / "official_reference_validation.json").write_text(json.dumps(official_validation, indent=2) + "\n")
        import_official(ns.official_reference.resolve(), ns.run_dir, config["edit_seeds"])

    groups = base["groups"] if ns.profile == "formal" else base["groups"][:1]
    targets = [target for group in groups for target in group["targets"]]
    seeds = config["edit_seeds"] if ns.profile == "formal" else [20260820]
    variants = config["formal_variants"] if ns.profile == "formal" else config["smoke_variants"]
    for seed in seeds:
        seed_dir = ns.run_dir / "seeds" / str(seed)
        for name in ("checkpoints", "diagnostics", "alpha_audits", "stages", "evaluation"):
            (seed_dir / name).mkdir(parents=True, exist_ok=True)
        commands = {variant: edit_command(ns, config, base, assets, seed, seed_dir, variant, targets) for variant in variants}
        for variant in variants:
            run_edit(commands[variant], seed_dir, variant)
        validate_edit_isolation(seed_dir, variants, commands)
        evaluate_variants = variants if ns.profile == "smoke" else config["generated_formal_variants"]
        for variant in evaluate_variants:
            evaluate(ns, ns.run_dir / "base_config.json", ns.assets.resolve(), protocol, seed_dir, variant)
            cleanup_checkpoint(seed_dir, variant)
        if ns.profile == "formal":
            cleanup_checkpoint(seed_dir, "official")

    if git_status():
        raise RuntimeError(f"working tree became dirty during calculation: {git_status()}")
    manifest["git_status_before_aggregation"] = []
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run([sys.executable, str(HERE / "aggregate_results.py"),
         "--run-dir", str(ns.run_dir), "--profile", ns.profile])
    if git_status():
        raise RuntimeError(f"aggregation changed tracked files: {git_status()}")
    manifest["git_status_end"] = []
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(ns.run_dir / "results" / "summary.md", ns.run_dir / "summary.md")
    (ns.run_dir / "worker_complete.json").write_text(json.dumps({
        "status": "passed", "profile": ns.profile, "edit_seeds": seeds,
        "official_reference_reused": ns.profile == "formal",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
