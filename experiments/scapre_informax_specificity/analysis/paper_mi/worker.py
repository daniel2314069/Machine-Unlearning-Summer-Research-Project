#!/usr/bin/env python3
"""Resume-safe worker for the fixed paper-MI versus repository experiment."""

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
RNG_RUNNER = HERE / "paper_mi_runner.py"


def parse_args() -> argparse.Namespace:
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


def score_keys(rows: list[dict[str, str]]) -> set[tuple[str, ...]]:
    fields = ("group", "role", "concept", "sample_index", "prompt", "seed", "seed_source")
    return {tuple(row[field] for field in fields) for row in rows}


def validate_sources(config: dict) -> None:
    for relative, expected in config["source_controls"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"controlled source changed: {relative}")


def controlled_assets(value: dict) -> dict:
    keys = (
        "base_model", "resolved_revision", "snapshot_path", "downloaded_files",
        "resnet_weights", "resnet_url", "packages", "config_sha256",
    )
    return {key: value.get(key) for key in keys}


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
    reference_config = config["official_reference"]
    if sha256(reference / "protocol.csv") != reference_config["protocol_sha256"]:
        raise RuntimeError("official reference protocol hash mismatch")
    if sha256(protocol) != reference_config["protocol_sha256"]:
        raise RuntimeError("current formal protocol differs from official reference")
    manifest = json.loads((reference / "run_manifest.json").read_text())
    if manifest.get("git_commit") != reference_config["run_commit"]:
        raise RuntimeError("official reference run commit changed")
    historical = manifest.get("source_sha256", {})
    if historical.get("scapre/edit/erase_scale.py") != reference_config["editor_source_sha256"]:
        raise RuntimeError("official reference editor hash mismatch")
    evaluator_key = "experiments/scapre_informax_specificity/evaluate_confuse5.py"
    if historical.get(evaluator_key) != reference_config["evaluator_source_sha256"]:
        raise RuntimeError("official reference evaluator hash mismatch")
    if controlled_assets(manifest["assets"]) != controlled_assets(assets):
        raise RuntimeError("official reference model/classifier assets differ")

    expected_keys = score_keys(read_csv(protocol))
    evaluator_control: dict | None = None
    details: dict[str, object] = {}
    for seed in config["edit_seeds"]:
        scores = reference / "seeds" / str(seed) / "evaluation" / "official" / "scores.csv"
        rows = read_csv(scores)
        if len(rows) != 3000 or score_keys(rows) != expected_keys:
            raise RuntimeError(f"official seed {seed} does not match the frozen 3,000-row protocol")
        if Counter(row["role"] for row in rows) != {"target": 1200, "retain": 1800}:
            raise RuntimeError(f"official seed {seed} target/retain counts changed")
        expected_hash = reference_config["score_sha256"][str(seed)]
        if sha256(scores) != expected_hash:
            raise RuntimeError(f"official seed {seed} score hash mismatch")
        evaluation_manifest = json.loads((scores.parent / "evaluation_manifest.json").read_text())
        current = {
            key: value for key, value in evaluation_manifest.items()
            if key not in {"variant", "checkpoint_sha256"}
        }
        defaults = current.get("scheduler_config", {}).get("_use_default_values")
        if isinstance(defaults, list):
            current["scheduler_config"]["_use_default_values"] = sorted(defaults)
        if evaluator_control is None:
            evaluator_control = current
        elif current != evaluator_control:
            raise RuntimeError("official evaluator fingerprints differ across seeds")
        details[str(seed)] = {"rows": len(rows), "score_sha256": expected_hash}
    fingerprint = hashlib.sha256(
        json.dumps(evaluator_control, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != reference_config["evaluator_fingerprint_sha256"]:
        raise RuntimeError("official evaluator canonical fingerprint mismatch")
    return {
        "status": "passed",
        "path": str(reference),
        "seeds": details,
        "protocol_sha256": reference_config["protocol_sha256"],
        "evaluator_fingerprint_sha256": fingerprint,
        "historical_source_hashes_verified": True,
        "current_paper_editor_not_used_to_regenerate_repository_baseline": True,
    }


def import_official(reference: Path, run_dir: Path, seeds: list[int]) -> None:
    for seed in seeds:
        source = reference / "seeds" / str(seed) / "evaluation" / "official"
        destination = run_dir / "seeds" / str(seed) / "evaluation" / "official"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("scores.csv", "evaluation_manifest.json"):
            if not (destination / name).exists():
                shutil.copy2(source / name, destination / name)
        (destination / "COMPLETED").write_text("verified repository reference reuse\n")


def edit_command(ns: argparse.Namespace, config: dict, base: dict, assets: dict,
                 seed: int, seed_dir: Path, variant: str, targets: list[str]) -> list[str]:
    weighting_mode = "repository" if variant == "official" else "paper"
    calls = config[
        "expected_informax_randn_calls_formal"
        if ns.profile == "formal" else "expected_informax_randn_calls_smoke"
    ]
    edit = base["edit"]
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
        "--informax-negative-mode", config["informax_negative_mode"],
        "--informax-weighting-mode", weighting_mode,
        "--informax-diagnostics-path", str((seed_dir / "diagnostics" / f"{variant}.pt").resolve()),
        "--edit-seed", str(config["fixed_non_informax_seed"]),
        "--output_model", str((seed_dir / "checkpoints" / f"{variant}.pt").resolve()),
    ]
    return [
        sys.executable, str(RNG_RUNNER),
        "--weighting-mode", weighting_mode,
        "--informax-seed", str(seed),
        "--informax-rng-mode", "legacy" if seed == config["legacy_informax_seed"] else "isolated",
        "--script", str(EDITOR),
        "--audit-output", str((seed_dir / "audits" / f"{variant}.json").resolve()),
        "--expected-randn-calls", str(calls),
        "--", *editor_args,
    ]


def run_edit(command: list[str], seed_dir: Path, variant: str) -> None:
    checkpoint = seed_dir / "checkpoints" / f"{variant}.pt"
    diagnostics = seed_dir / "diagnostics" / f"{variant}.pt"
    audit_path = seed_dir / "audits" / f"{variant}.json"
    completed = seed_dir / "stages" / f"edit_{variant}.completed.json"
    command_path = seed_dir / "stages" / f"edit_{variant}.command.json"
    payload = {"argv": command}
    if command_path.exists() and json.loads(command_path.read_text()) != payload:
        raise RuntimeError(f"resume command changed: {variant}")
    command_path.write_text(json.dumps(payload, indent=2) + "\n")
    if completed.exists():
        if not all(path.is_file() for path in (checkpoint, diagnostics, audit_path)):
            raise RuntimeError(f"completed edit is incomplete: {variant}")
        print(f"[resume] edit {variant}", flush=True)
        return
    if any(path.exists() for path in (checkpoint, diagnostics, audit_path)):
        raise RuntimeError(f"unverified partial edit output exists: {variant}")
    run(command, cwd=SCAPRE_ROOT)
    audit = json.loads(audit_path.read_text())
    if not audit.get("completed") or audit["intercepted_randn_calls"] != audit["expected_randn_calls"]:
        raise RuntimeError(f"RNG/checkpoint audit failed: {variant}")
    completed.write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint),
        "diagnostics_sha256": sha256(diagnostics),
        "audit_sha256": sha256(audit_path),
    }, indent=2) + "\n")


def validate_paper_diagnostics(seed_dir: Path, config: dict, profile: str) -> dict:
    path = seed_dir / "diagnostics" / "paper_mi.pt"
    payload = torch.load(path, map_location="cpu")
    if payload.get("informax_weighting_mode") != "paper":
        raise RuntimeError("paper diagnostics have the wrong weighting mode")
    if payload.get("informax_negative_mode") != "official":
        raise RuntimeError("paper diagnostics changed the neutral source")
    raw_records = [row for row in payload["records"] if row["stage"] == "aggregate"]
    aggregate_records = [row for row in payload["records"] if row["stage"] == "aggregate-max"]
    expected_raw = (
        config["expected_raw_mi_records_formal"]
        if profile == "formal"
        else 2 * config["expected_layers_per_projection"] * config["targets_smoke"]
    )
    if len(raw_records) != expected_raw or len(aggregate_records) != config["expected_aggregate_records_formal"]:
        raise RuntimeError("paper diagnostic record coverage changed")
    if any(row["stage"] == "accumulation" for row in payload["records"]):
        raise RuntimeError("paper mode unexpectedly performed per-concept accumulation MI")
    for row in raw_records:
        if not torch.equal(row["returned_weight"], row["raw_mi"]):
            raise RuntimeError("paper per-concept weight is not raw MI")
    for row in aggregate_records:
        concept_max = row["concept_max_raw_mi"].double()
        channel_max = row["channel_max_raw_mi"].double()
        alpha = row["alpha"].double()
        if not torch.isfinite(alpha).all() or alpha.min().item() < 0.0 or alpha.max().item() > 1.0:
            raise RuntimeError("paper alpha is outside [0, 1]")
        if alpha.max().item() != 1.0:
            raise RuntimeError("paper alpha is not normalized by its maximum channel")
        if not torch.allclose(alpha, concept_max / channel_max, rtol=0.0, atol=1e-7):
            raise RuntimeError("paper alpha does not equal max-concept raw MI / channel maximum")
    audit = json.loads((seed_dir / "audits" / "paper_mi.json").read_text())
    expected_raw_calls = config[
        "expected_raw_mi_randn_calls_formal"
        if profile == "formal" else "expected_raw_mi_randn_calls_smoke"
    ]
    caller_counts = audit.get("caller_counts", {})
    if caller_counts.get("_compute_mi_softmask_emptyneg") != expected_raw_calls:
        raise RuntimeError("paper raw-MI pseudo-sample draw count changed")
    if caller_counts.get("_consume_removed_accumulation_informax_rng") != expected_raw_calls:
        raise RuntimeError("removed-accumulation RNG pairing draw count changed")
    report = {
        "status": "passed",
        "informax_weighting_mode": "paper",
        "official_empty_string_neutral": True,
        "raw_mi_used_directly": True,
        "max_over_concepts_before_channel_normalization": True,
        "alpha_channel_max_exactly_one": True,
        "per_concept_accumulation_weighting_removed": True,
        "removed_accumulation_rng_positions_preserved": True,
        "B_only_final_objective": True,
        "raw_mi_records": len(raw_records),
        "aggregate_records": len(aggregate_records),
        "informax_randn_calls": audit["intercepted_randn_calls"],
    }
    (seed_dir / "paper_formula_check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def validate_generated_pair(seed_dir: Path, profile: str) -> None:
    repository = torch.load(seed_dir / "diagnostics" / "official.pt", map_location="cpu")
    paper = torch.load(seed_dir / "diagnostics" / "paper_mi.pt", map_location="cpu")
    repository_records = [row for row in repository["records"] if row["stage"] == "aggregate"]
    paper_records = [row for row in paper["records"] if row["stage"] == "aggregate"]
    if len(repository_records) != len(paper_records):
        raise RuntimeError(f"{profile} aggregate Informax coverage differs")
    for left, right in zip(repository_records, paper_records):
        identity_fields = ("projection", "layer_index", "target_index", "target_concept")
        if any(left[field] != right[field] for field in identity_fields):
            raise RuntimeError(f"{profile} aggregate Informax record order differs")
        for field in ("raw_mi", "threshold", "negative_base_indices"):
            if not torch.equal(left[field], right[field]):
                raise RuntimeError(f"{profile} aggregate Informax input/result changed: {field}")
    report_path = seed_dir / "paper_formula_check.json"
    report = json.loads(report_path.read_text())
    report.update({
        "direct_repository_pair_checked": True,
        "direct_repository_pair_profile": profile,
        "same_aggregate_raw_mi": True,
        "same_aggregate_thresholds": True,
        "same_aggregate_negative_base_indices": True,
    })
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def evaluate(ns: argparse.Namespace, base_config: Path, assets: Path, protocol: Path,
             seed_dir: Path, variant: str) -> None:
    output = seed_dir / "evaluation" / variant
    if (output / "COMPLETED").exists():
        rows = read_csv(output / "scores.csv")
        if rows and all(row.get("variant") == variant for row in rows):
            print(f"[resume] evaluation {variant}", flush=True)
            return
        raise RuntimeError(f"completed evaluation is invalid: {variant}")
    run([
        sys.executable, str(EVALUATOR),
        "--config", str(base_config),
        "--assets", str(assets),
        "--protocol", str(protocol),
        "--checkpoint", str(seed_dir / "checkpoints" / f"{variant}.pt"),
        "--variant", variant,
        "--output-dir", str(output),
        "--device", ns.device,
    ])


def main() -> None:
    ns = parse_args()
    ns.run_dir = ns.run_dir.resolve()
    if os.environ.get("CONDA_DEFAULT_ENV") != "MU":
        raise RuntimeError("worker requires active Conda MU")
    if git_status():
        raise RuntimeError(f"working tree is dirty at start: {git_status()}")
    config = json.loads((HERE / "config.json").read_text())
    base = json.loads(BASE_CONFIG.read_text())
    assets = json.loads(ns.assets.read_text())
    validate_sources(config)
    if config["edit_seeds"] != [20260820, 20260821, 20260822, 20260823, 20260824]:
        raise RuntimeError("the five established edit seeds changed")
    if config["variants"] != ["official", "paper_mi"]:
        raise RuntimeError("comparison variants changed")
    if config["informax_negative_mode"] != "official":
        raise RuntimeError("paper comparison must retain the official empty-string neutral")
    if config["paper_formula"].get("parameter_search") is not False:
        raise RuntimeError("parameter search is forbidden for this fixed comparison")
    expected_calls = (
        4 * config["expected_layers_per_projection"] * config["targets_formal"]
    )
    if config["expected_informax_randn_calls_formal"] != expected_calls:
        raise RuntimeError("formal Informax RNG call expectation is inconsistent")
    if base["edit_seed"] != config["fixed_non_informax_seed"]:
        raise RuntimeError("fixed non-Informax seed changed")
    if base["edit"]["num_positive"] != 5 or base["edit"]["num_negative"] != 5:
        raise RuntimeError("Informax pseudo-sample counts changed")
    ns.run_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        [REPO_ROOT / relative for relative in config["source_controls"]]
        + [path for path in HERE.iterdir() if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json"}]
    )
    source_hashes = {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files}
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": ns.profile,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status_start": [],
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "source_sha256": source_hashes,
        "assets": assets,
        "assets_sha256": sha256(ns.assets),
    }
    manifest_path = ns.run_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("profile", "git_commit", "git_branch", "source_sha256", "assets", "assets_sha256"):
            if previous.get(key) != manifest[key]:
                raise RuntimeError(f"resume provenance changed: {key}")
        manifest = previous
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

    repository_reference_reused = False
    reference_rejection: str | None = None
    if ns.profile == "formal" and ns.official_reference is not None:
        try:
            validation = validate_official_reference(ns.official_reference.resolve(), config, assets, protocol)
        except (RuntimeError, OSError, KeyError, TypeError, ValueError) as error:
            reference_rejection = str(error)
            validation = {
                "status": "rejected",
                "path": str(ns.official_reference.resolve()),
                "reason": reference_rejection,
                "action": "regenerate_repository_baseline",
            }
            print(
                f"Verified reference could not be reused ({reference_rejection}); "
                "regenerating the repository baseline.",
                flush=True,
            )
        else:
            repository_reference_reused = True
            import_official(ns.official_reference.resolve(), ns.run_dir, config["edit_seeds"])
        (ns.run_dir / "official_reference_validation.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )
    baseline_source = {
        "mode": "verified_historical_reference" if repository_reference_reused else "generated_in_run",
        "repository_reference_reused": repository_reference_reused,
        "official_reference": str(ns.official_reference.resolve()) if repository_reference_reused else None,
        "attempted_reference": str(ns.official_reference.resolve()) if ns.official_reference else None,
        "reference_rejection": reference_rejection,
        "note": (
            "Checksum-pinned historical scores were validated and reused."
            if repository_reference_reused else
            (
                "The repository baseline was generated directly for the paired smoke check."
                if ns.profile == "smoke" else
                "The repository baseline was regenerated because no verified retained reference was available."
            )
        ),
    }
    baseline_source_path = ns.run_dir / "baseline_source.json"
    if baseline_source_path.exists() and json.loads(baseline_source_path.read_text()) != baseline_source:
        raise RuntimeError("resume baseline source changed")
    baseline_source_path.write_text(json.dumps(baseline_source, indent=2) + "\n")

    groups = base["groups"] if ns.profile == "formal" else base["groups"][:1]
    targets = [target for group in groups for target in group["targets"]]
    seeds = config["edit_seeds"] if ns.profile == "formal" else [config["legacy_informax_seed"]]
    generated_variants = ["paper_mi"] if repository_reference_reused else config["variants"]
    for seed in seeds:
        seed_dir = ns.run_dir / "seeds" / str(seed)
        for name in ("checkpoints", "diagnostics", "audits", "stages", "evaluation"):
            (seed_dir / name).mkdir(parents=True, exist_ok=True)
        for variant in generated_variants:
            command = edit_command(ns, config, base, assets, seed, seed_dir, variant, targets)
            run_edit(command, seed_dir, variant)
            if variant == "paper_mi":
                validate_paper_diagnostics(seed_dir, config, ns.profile)
            evaluate(ns, ns.run_dir / "base_config.json", ns.assets.resolve(), protocol, seed_dir, variant)
        if not repository_reference_reused:
            validate_generated_pair(seed_dir, ns.profile)

    if git_status():
        raise RuntimeError(f"working tree became dirty during calculation: {git_status()}")
    manifest["git_status_before_aggregation"] = []
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run([sys.executable, str(HERE / "aggregate_results.py"),
         "--run-dir", str(ns.run_dir), "--profile", ns.profile])
    if git_status():
        raise RuntimeError("aggregation changed tracked files")
    manifest["git_status_end"] = []
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(ns.run_dir / "results" / "summary.md", ns.run_dir / "summary.md")
    (ns.run_dir / "worker_complete.json").write_text(json.dumps({
        "status": "passed",
        "profile": ns.profile,
        "edit_seeds": seeds,
        "repository_reference_reused": repository_reference_reused,
        "generated_variants": generated_variants,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
