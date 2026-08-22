#!/usr/bin/env python
"""Resume-safe orchestration for the ScaPre superclass-neutral ablation."""

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


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO = HERE.parents[2]
SCAPRE = REPO / "scapre"
BASE_CONFIG = PARENT / "config.json"
EDITOR = SCAPRE / "edit" / "erase_scale.py"
PROTOCOL_BUILDER = PARENT / "build_protocol.py"
EVALUATOR = PARENT / "evaluate_confuse5.py"
RNG_RUNNER = PARENT / "seed_robustness" / "informax_seed_runner.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "formal"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run(command: list[str], cwd: Path = REPO) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def git_status() -> list[str]:
    output = git_output("status", "--porcelain", "--untracked-files=all")
    return output.splitlines() if output else []


def validate_config(config: dict, base: dict) -> None:
    expected_targets = [target for group in base["groups"] for target in group["targets"]]
    mapping = config.get("superclass_by_target")
    if not isinstance(mapping, dict) or list(mapping) != expected_targets:
        raise RuntimeError("superclass mapping must cover the ten ordered Confuse5 targets")
    expected = {
        "golden retriever": "dog", "labrador retriever": "dog",
        "tabby": "cat", "tiger cat": "cat",
        "orange": "fruit", "lemon": "fruit",
        "yawl": "boat", "lifeboat": "boat",
        "soccer ball": "ball", "volleyball": "ball",
    }
    if mapping != expected:
        raise RuntimeError("superclass mapping differs from the preregistered mapping")
    if any(value in {item for group in base["groups"] for item in group["retains"]}
           for value in mapping.values()):
        raise RuntimeError("a superclass reference cannot be a retain label")
    if config.get("edit_seeds") != [20260820, 20260821, 20260822, 20260823, 20260824]:
        raise RuntimeError("fixed edit seeds changed")
    if config.get("fixed_non_informax_seed") != 20260820:
        raise RuntimeError("fixed non-Informax seed changed")
    if config.get("variant") != "superclass_neutral":
        raise RuntimeError("this runner supports only superclass_neutral")
    if base["edit"]["num_positive"] != 5 or base["edit"]["num_negative"] != 5:
        raise RuntimeError("Informax sample counts changed from 5+5")
    if base["evaluation"]["formal_images_per_concept"] != 120:
        raise RuntimeError("formal image count changed")


def build_editor_args(
    base: dict,
    assets: dict,
    targets: list[str],
    checkpoint: Path,
    diagnostics: Path,
    superclass_config: Path | None,
    mode: str,
) -> list[str]:
    edit = base["edit"]
    result = [
        "--concepts", ", ".join(targets),
        "--concept_type", edit["concept_type"],
        "--device", "0",
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
    if superclass_config is not None:
        result.extend(["--informax-superclass-config", str(superclass_config.resolve())])
    return result


def wrapped_edit_command(
    seed: int,
    expected_calls: int,
    audit: Path,
    editor_args: list[str],
) -> list[str]:
    return [
        sys.executable, str(RNG_RUNNER),
        "--informax-seed", str(seed),
        "--script", str(EDITOR),
        "--audit-output", str(audit.resolve()),
        "--expected-randn-calls", str(expected_calls),
        "--", *editor_args,
    ]


def normalized_command(command: list[str]) -> list[str]:
    removed = {
        "--informax-negative-mode", "--informax-superclass-config",
        "--informax-diagnostics-path", "--output_model", "--audit-output",
    }
    result: list[str] = []
    index = 0
    while index < len(command):
        if command[index] in removed:
            index += 2
            continue
        result.append(command[index])
        index += 1
    return result


def run_edit(command: list[str], seed_dir: Path, legacy_global_rng: bool) -> None:
    checkpoint = seed_dir / "checkpoints" / "superclass_neutral.pt"
    diagnostics = seed_dir / "diagnostics" / "superclass_neutral.pt"
    audit = seed_dir / "stages" / "informax_rng_superclass_neutral.json"
    completed = seed_dir / "stages" / "edit_superclass_neutral.completed"
    command_path = seed_dir / "stages" / "edit_superclass_neutral_command.json"
    payload = {"argv": command}
    if command_path.exists() and json.loads(command_path.read_text()) != payload:
        raise RuntimeError("edit command changed while resuming")
    command_path.write_text(json.dumps(payload, indent=2) + "\n")
    if completed.exists():
        if not all(path.is_file() for path in (checkpoint, diagnostics, audit)):
            raise RuntimeError("completed superclass edit is missing outputs")
        print("[resume] superclass-neutral edit", flush=True)
        return
    if checkpoint.exists() or diagnostics.exists():
        raise RuntimeError("unverified partial superclass edit exists; use a new run id")
    run(command, cwd=SCAPRE)
    if legacy_global_rng:
        audit_payload = {
            "informax_seed": 20260820,
            "completed": True,
            "mode": "legacy global RNG, identical to the imported seed-20260820 official run",
            "intercepted_randn_calls": 0,
            "global_rng_seed": 20260820,
        }
        audit.write_text(json.dumps(audit_payload, indent=2) + "\n")
    else:
        audit_payload = json.loads(audit.read_text())
        if (not audit_payload.get("completed") or
                audit_payload["intercepted_randn_calls"] != audit_payload["expected_randn_calls"]):
            raise RuntimeError("Informax-only RNG audit failed")
        callers = {key.split(":", 1)[0] for key in audit_payload["shape_counts"]}
        if callers != {"_compute_mi_softmask_matchedneg"}:
            raise RuntimeError(f"unexpected superclass Informax RNG caller: {sorted(callers)}")
    completed.write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint),
        "diagnostics_sha256": sha256(diagnostics),
        "rng_audit_sha256": sha256(audit),
    }, indent=2) + "\n")


def copy_prior_baselines(prior: Path, run_dir: Path, seeds: list[int]) -> None:
    final_complete = (
        (prior / "COMPLETED").is_file()
        and (prior / "exit_code").is_file()
        and (prior / "exit_code").read_text().strip() == "0"
    )
    calculation_complete = (
        (prior / "CALCULATION_COMPLETED").is_file()
        and (prior / "calculation_exit_code").is_file()
        and (prior / "calculation_exit_code").read_text().strip() == "0"
    )
    if not (final_complete or calculation_complete):
        raise RuntimeError(f"prior robustness run is not completed: {prior}")
    integrity = json.loads((prior / "reproducibility" / "integrity_report.json").read_text())
    if integrity.get("status") != "passed" or integrity.get("edit_seeds") != seeds:
        raise RuntimeError("prior robustness integrity report is invalid")
    destination = run_dir / "baselines"
    for seed in seeds:
        for variant in ("official", "matched_retain"):
            source = prior / "seeds" / str(seed) / "evaluation" / variant
            target = destination / str(seed) / variant
            target.mkdir(parents=True, exist_ok=True)
            for name in ("scores.csv", "evaluation_manifest.json", "COMPLETED"):
                if not (source / name).is_file():
                    raise RuntimeError(f"prior baseline is missing: {source / name}")
                shutil.copy2(source / name, target / name)
        prior_diagnostic = prior / "seeds" / str(seed) / "results" / "informax_diagnostics.csv"
        if not prior_diagnostic.is_file():
            raise RuntimeError(f"prior diagnostic summary is missing: {prior_diagnostic}")
        shutil.copy2(prior_diagnostic, destination / str(seed) / "informax_diagnostics.csv")
    repro = run_dir / "reproducibility"
    repro.mkdir(exist_ok=True)
    shutil.copy2(prior / "run_manifest.json", repro / "prior_robustness_run_manifest.json")
    shutil.copy2(prior / "results" / "summary.md", repro / "prior_robustness_summary.md")
    (repro / "baseline_reuse.json").write_text(json.dumps({
        "status": "passed",
        "source_run": str(prior.resolve()),
        "source_integrity_sha256": sha256(prior / "reproducibility" / "integrity_report.json"),
        "seeds": seeds,
        "variants": ["official", "matched_retain"],
        "regenerated_baseline_scores": False,
        "copied_score_files": 10,
    }, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    start_status = git_status()
    if start_status:
        raise RuntimeError(f"working tree is dirty at run start: {start_status}")
    config = json.loads(args.config.read_text())
    base = json.loads(BASE_CONFIG.read_text())
    assets = json.loads(args.assets.read_text())
    validate_config(config, base)
    if args.profile == "formal" and args.prior_run is None:
        raise RuntimeError("formal run requires the verified prior robustness run")

    source_files = sorted(set(
        [REPO / value for value in config["source_controls"]]
        + [path for path in HERE.iterdir() if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json"}]
    ))
    source_hashes = {str(path.relative_to(REPO)): sha256(path) for path in source_files}
    manifest_path = args.run_dir / "run_manifest.json"
    candidate = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_start": start_status,
        "source_sha256": source_hashes,
        "assets": assets,
        "assets_manifest_sha256": sha256(args.assets),
        "device": args.device,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key in ("profile", "git_commit", "source_sha256", "assets", "assets_manifest_sha256", "device"):
            if manifest.get(key) != candidate.get(key):
                raise RuntimeError(f"resume manifest changed: {key}")
    else:
        manifest = candidate
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    (args.run_dir / "actual_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.run_dir / "base_config.json").write_text(json.dumps(base, indent=2) + "\n")
    provenance = args.run_dir / "provenance"
    for source in source_files:
        destination = provenance / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    protocol = args.run_dir / "protocol.csv"
    protocol_output = subprocess.check_output([
        sys.executable, str(PROTOCOL_BUILDER), "--config", str(BASE_CONFIG),
        "--output", str(protocol), "--profile", args.profile,
    ], cwd=REPO, text=True)
    protocol_manifest = json.loads(protocol_output)
    (args.run_dir / "protocol_manifest.json").write_text(json.dumps(protocol_manifest, indent=2) + "\n")
    if args.profile == "formal" and protocol_manifest["sha256"] != config["prior_robustness"]["expected_protocol_sha256"]:
        raise RuntimeError("formal protocol differs from the verified baseline protocol")

    superclass_path = args.run_dir / "superclass_config.json"
    superclass_path.write_text(json.dumps({"superclass_by_target": config["superclass_by_target"]}, indent=2) + "\n")
    groups = base["groups"][:1] if args.profile == "smoke" else base["groups"]
    targets = [target for group in groups for target in group["targets"]]
    seeds = [20260821] if args.profile == "smoke" else config["edit_seeds"]
    if args.profile == "formal":
        copy_prior_baselines(args.prior_run.resolve(), args.run_dir, seeds)

    expected_calls = config[
        "expected_informax_randn_calls_per_smoke_edit"
        if args.profile == "smoke" else "expected_informax_randn_calls_per_formal_edit"
    ]
    for seed in seeds:
        seed_dir = args.run_dir / "seeds" / str(seed)
        for name in ("checkpoints", "diagnostics", "evaluation", "results", "stages"):
            (seed_dir / name).mkdir(parents=True, exist_ok=True)
        seed_config = dict(base)
        seed_config["informax_seed"] = seed
        seed_config["fixed_non_informax_seed"] = config["fixed_non_informax_seed"]
        seed_config["informax_negative_mode"] = "superclass-neutral"
        seed_config_path = seed_dir / "actual_config.json"
        seed_config_path.write_text(json.dumps(seed_config, indent=2) + "\n")
        checkpoint = seed_dir / "checkpoints" / "superclass_neutral.pt"
        diagnostics = seed_dir / "diagnostics" / "superclass_neutral.pt"
        audit = seed_dir / "stages" / "informax_rng_superclass_neutral.json"
        editor_args = build_editor_args(base, assets, targets, checkpoint, diagnostics, superclass_path, "superclass-neutral")
        legacy_global_rng = seed == 20260820
        if legacy_global_rng:
            command = [sys.executable, str(EDITOR), *editor_args]
            official_shadow = [
                sys.executable, str(EDITOR),
                *build_editor_args(base, assets, targets,
                                   seed_dir / "checkpoints" / "shadow_official.pt",
                                   seed_dir / "diagnostics" / "shadow_official.pt", None, "official"),
            ]
        else:
            command = wrapped_edit_command(seed, expected_calls, audit, editor_args)
            official_shadow = wrapped_edit_command(
                seed, expected_calls, seed_dir / "stages" / "shadow_official_rng.json",
                build_editor_args(base, assets, targets,
                                  seed_dir / "checkpoints" / "shadow_official.pt",
                                  seed_dir / "diagnostics" / "shadow_official.pt", None, "official"),
            )
        if normalized_command(command) != normalized_command(official_shadow):
            raise RuntimeError("controlled commands differ outside the intended negative source")
        run_edit(command, seed_dir, legacy_global_rng)
        audit_payload = json.loads(audit.read_text())
        (seed_dir / "controlled_ablation_check.json").write_text(json.dumps({
            "status": "passed", "edit_seed": seed,
            "same_normalized_edit_command": True,
            "same_informax_noise_shapes_as_official_implementation": True,
            "negative_sample_count": 5,
            "unique_negative_base_per_target": 1,
            "rng_mode": "legacy_global" if legacy_global_rng else "isolated_informax_stream",
            "intended_difference": "Informax negative base: empty prompt versus fixed target superclass",
            "informax_rng_audit": audit_payload,
            "normalized_argv": normalized_command(command),
        }, indent=2) + "\n")

        evaluation_dir = seed_dir / "evaluation" / "superclass_neutral"
        if not (evaluation_dir / "COMPLETED").exists():
            run([
                sys.executable, str(EVALUATOR), "--config", str(seed_config_path),
                "--assets", str(args.assets.resolve()), "--protocol", str(protocol),
                "--checkpoint", str(checkpoint), "--variant", "superclass_neutral",
                "--output-dir", str(evaluation_dir), "--device", args.device,
            ])
        else:
            print(f"[resume] evaluation seed {seed}", flush=True)

    if args.profile == "formal":
        run([
            sys.executable, str(HERE / "build_qualitative.py"),
            "--run-dir", str(args.run_dir), "--config", str(args.config.resolve()),
            "--base-config", str(BASE_CONFIG), "--assets", str(args.assets.resolve()),
            "--prior-run", str(args.prior_run.resolve()), "--device", args.device,
        ])

    manifest["git_status_end"] = git_status()
    manifest["calculation_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if manifest["git_status_end"]:
        raise RuntimeError(f"working tree became dirty during run: {manifest['git_status_end']}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run([
        sys.executable, str(HERE / "aggregate_results.py"), "--run-dir", str(args.run_dir),
        "--config", str(args.config.resolve()), "--base-config", str(BASE_CONFIG),
        "--profile", args.profile,
    ])
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["git_status_end"] = git_status()
    if manifest["git_status_end"]:
        raise RuntimeError("aggregation changed the tracked working tree")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(args.run_dir / "results" / "summary.md", args.run_dir / "summary.md")
    (args.run_dir / "worker_complete.json").write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile, "edit_seeds": seeds,
        "new_image_records": len(seeds) * (10 if args.profile == "smoke" else 3000),
        "baseline_score_evaluations_rerun": False,
        "qualitative_images": 90 if args.profile == "formal" else 0,
        "summary": str((args.run_dir / "results" / "summary.md").resolve()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
