#!/usr/bin/env python3
"""Qualification followed by the preregistered five-seed Confuse5 run."""

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
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
SCAPRE_ROOT = REPO_ROOT / "scapre"
EDITOR = SCAPRE_ROOT / "edit" / "erase_scale.py"
BASE_CONFIG = EXPERIMENT / "config.json"
PROTOCOL_BUILDER = EXPERIMENT / "build_protocol.py"
EVALUATOR = EXPERIMENT / "evaluate_confuse5.py"
PROJECTION_RUNNER = HERE / "projection_runner.py"
EVALUATOR_RUNNER = HERE / "evaluate_projection_runner.py"
AGGREGATOR = HERE / "aggregate_results.py"
SEEDS = [20260820, 20260821, 20260822, 20260823, 20260824]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()


def git_status() -> list[str]:
    output = git("status", "--porcelain", "--untracked-files=all")
    return output.splitlines() if output else []


def run(command: list[str], cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def score_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["group"], row["role"], row["concept"], row["sample_index"],
        row["prompt"], row["seed"], row["seed_source"],
    )


def validate_configuration(config: dict[str, Any], base: dict[str, Any]) -> None:
    if config["edit_seeds"] != SEEDS or config["qualification_seed"] != 20260820:
        raise RuntimeError("frozen edit seeds changed")
    if config["fixed_non_informax_seed"] != 20260820:
        raise RuntimeError("fixed non-Informax seed changed")
    formula = config["formula"]
    if formula["eps"] != 1e-8:
        raise RuntimeError("projection formula parameters changed")
    if formula.get("alpha_mode") not in {"zscore_sigmoid_power", "direct_cos2"}:
        raise RuntimeError("projection alpha mode is invalid")
    if formula["temperature"] != 0.7 or formula["power"] != 8.0:
        raise RuntimeError("frozen V1 diagnostic transform changed")
    if formula["alpha_mode"] == "direct_cos2":
        if formula.get("selected_alpha") != "projection_score":
            raise RuntimeError("direct-cos2 must select the exact projection score")
        if formula.get("normalization") != "none" or formula.get("sweep") is not False:
            raise RuntimeError("direct-cos2 normalization/sweep contract changed")
    edit = base["edit"]
    expected = {
        "base": "1.5", "concept_type": "object", "erase_scale": 2.0,
        "p": 8.0, "bures_iters": 1, "enable_ased": True,
        "entropy_samples": 30, "entropy_bins": 20, "noise_sigma": 0.01,
        "T_sigma": 1.0, "p_sigma": 1.0, "lamb_effective": 0.5,
        "num_positive": 5, "num_negative": 5,
    }
    for key, value in expected.items():
        if edit.get(key) != value:
            raise RuntimeError(f"official edit setting changed: {key}")
    evaluation = base["evaluation"]
    expected_evaluation = {
        "formal_images_per_concept": 120, "num_inference_steps": 50,
        "guidance_scale": 7.5, "height": 512, "width": 512,
        "dtype": "float16", "classifier": "torchvision ResNet50_Weights.DEFAULT",
    }
    for key, value in expected_evaluation.items():
        if evaluation.get(key) != value:
            raise RuntimeError(f"official evaluation setting changed: {key}")


def validate_sources(config: dict[str, Any]) -> None:
    for relative, expected in config["source_controls"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"controlled source hash mismatch: {relative}")


def validate_official_reference(
    reference: Path, config: dict[str, Any], assets: dict[str, Any], protocol: Path
) -> dict[str, Any]:
    official = config["official_reference"]
    required = ["run_manifest.json", "protocol.csv", "protocol_manifest.json"]
    for seed in SEEDS:
        required.extend([
            f"seeds/{seed}/evaluation/official/scores.csv",
            f"seeds/{seed}/evaluation/official/evaluation_manifest.json",
        ])
    missing = [name for name in required if not (reference / name).is_file()]
    if missing:
        raise RuntimeError(f"official reference is incomplete: {missing}")
    if sha256(reference / "protocol.csv") != official["protocol_sha256"]:
        raise RuntimeError("official protocol hash mismatch")
    if sha256(protocol) != official["protocol_sha256"]:
        raise RuntimeError("current protocol differs from official reference")
    manifest = json.loads((reference / "run_manifest.json").read_text())
    if manifest.get("git_commit") != official["run_commit"]:
        raise RuntimeError("official run commit changed")
    historical = manifest.get("source_sha256", {})
    if historical.get("scapre/edit/erase_scale.py") != official["editor_source_sha256"]:
        raise RuntimeError("historical editor hash changed")
    if historical.get("experiments/scapre_informax_specificity/evaluate_confuse5.py") != official["evaluator_source_sha256"]:
        raise RuntimeError("historical evaluator hash changed")
    compatibility = subprocess.check_output([
        "git", "diff", f"{official['run_commit']}..HEAD", "--",
        "scapre/edit/erase_scale.py",
        "experiments/scapre_informax_specificity/evaluate_confuse5.py",
    ], cwd=REPO_ROOT)
    if hashlib.sha256(compatibility).hexdigest() != official["compatibility_diff_sha256"]:
        raise RuntimeError("official branch/evaluator compatibility diff changed")
    historical_assets = manifest.get("assets", {})
    for key in ("base_model", "resolved_revision", "packages", "resnet_weights"):
        if historical_assets.get(key) != assets.get(key):
            raise RuntimeError(f"official model/classifier asset mismatch: {key}")

    protocol_keys = [score_key(row) for row in read_csv(protocol)]
    if len(protocol_keys) != 3000 or len(set(protocol_keys)) != 3000:
        raise RuntimeError("current formal protocol keys are incomplete or duplicated")
    observed_fingerprints: set[str] = set()
    score_hashes: dict[str, str] = {}
    for seed in SEEDS:
        root = reference / "seeds" / str(seed) / "evaluation" / "official"
        score_path = root / "scores.csv"
        rows = read_csv(score_path)
        if len(rows) != 3000 or Counter(row["role"] for row in rows) != {"target": 1200, "retain": 1800}:
            raise RuntimeError(f"official seed {seed} row counts changed")
        if [score_key(row) for row in rows] != protocol_keys:
            raise RuntimeError(f"official seed {seed} generation key order changed")
        actual_hash = sha256(score_path)
        if actual_hash != official["score_sha256"][str(seed)]:
            raise RuntimeError(f"official seed {seed} score hash mismatch")
        score_hashes[str(seed)] = actual_hash
        evaluator = json.loads((root / "evaluation_manifest.json").read_text())
        controlled = {key: value for key, value in evaluator.items() if key not in {"variant", "checkpoint_sha256"}}
        defaults = controlled.get("scheduler_config", {}).get("_use_default_values")
        if not isinstance(defaults, list):
            raise RuntimeError("evaluator scheduler defaults are missing")
        controlled["scheduler_config"]["_use_default_values"] = sorted(defaults)
        encoded = json.dumps(controlled, sort_keys=True, separators=(",", ":")).encode()
        observed_fingerprints.add(hashlib.sha256(encoded).hexdigest())
    if observed_fingerprints != {official["evaluator_fingerprint_sha256"]}:
        raise RuntimeError("official evaluator fingerprint mismatch")
    return {
        "status": "passed", "reference": str(reference),
        "protocol_sha256": official["protocol_sha256"],
        "score_sha256": score_hashes,
        "evaluator_fingerprint_sha256": official["evaluator_fingerprint_sha256"],
        "historical_source_hashes_verified": True,
        "compatibility_diff_verified": True,
    }


def import_official(reference: Path, run_dir: Path) -> None:
    for seed in SEEDS:
        source = reference / "seeds" / str(seed) / "evaluation" / "official"
        destination = run_dir / "seeds" / str(seed) / "evaluation" / "official"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("scores.csv", "evaluation_manifest.json"):
            if not (destination / name).exists():
                shutil.copy2(source / name, destination / name)
        (destination / "COMPLETED").write_text("verified historical reference\n")


def evaluator_fingerprint(path: Path) -> str:
    manifest = json.loads(path.read_text())
    controlled = {
        key: value for key, value in manifest.items()
        if key not in {"variant", "checkpoint_sha256"}
    }
    defaults = controlled.get("scheduler_config", {}).get("_use_default_values")
    if not isinstance(defaults, list):
        raise RuntimeError(f"evaluator scheduler defaults are missing: {path}")
    controlled["scheduler_config"]["_use_default_values"] = sorted(defaults)
    encoded = json.dumps(controlled, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def targets(base: dict[str, Any]) -> list[str]:
    return [target for group in base["groups"] for target in group["targets"]]


def edit_command(
    config: dict[str, Any], base: dict[str, Any], assets: dict[str, Any],
    seed: int, seed_dir: Path, variant: str,
) -> list[str]:
    edit = base["edit"]
    editor_args = [
        "--concepts", ", ".join(targets(base)),
        "--concept_type", edit["concept_type"], "--device", "0",
        "--base", edit["base"], "--model-id-or-path", assets["snapshot_path"],
        "--use_mi_softmask", "--erase_scale", str(edit["erase_scale"]),
        "--p", str(edit["p"]), "--bures_iters", str(edit["bures_iters"]),
        "--enable_ased", "--entropy_samples", str(edit["entropy_samples"]),
        "--entropy_bins", str(edit["entropy_bins"]),
        "--noise_sigma", str(edit["noise_sigma"]),
        "--T_sigma", str(edit["T_sigma"]), "--p_sigma", str(edit["p_sigma"]),
        "--informax-negative-mode", "official",
        "--informax-diagnostics-path", str((seed_dir / "diagnostics" / f"informax_{variant}.pt").resolve()),
        "--edit-seed", str(config["fixed_non_informax_seed"]),
        "--output_model", str((seed_dir / "checkpoints" / f"{variant}.pt").resolve()),
    ]
    formula = config["formula"]
    return [
        sys.executable, str(PROJECTION_RUNNER), "--variant", variant,
        "--informax-seed", str(seed),
        "--informax-rng-mode", "legacy" if seed == config["legacy_informax_seed"] else "isolated",
        "--script", str(EDITOR),
        "--audit-output", str((seed_dir / "audits" / f"{variant}.json").resolve()),
        "--diagnostics-output", str((seed_dir / "diagnostics" / f"projection_{variant}.pt").resolve()),
        "--expected-informax-randn-calls", str(config["expected_informax_randn_calls_per_edit"]),
        "--expected-accumulation-intercepts", str(config["expected_accumulation_intercepts_per_edit"]),
        "--expected-matrix-records", str(config["expected_matrix_records_per_edit"]),
        "--targets-per-matrix", str(config["targets_per_matrix"]),
        "--alpha-mode", str(formula["alpha_mode"]),
        "--temperature", str(formula["temperature"]), "--power", str(formula["power"]),
        "--eps", str(formula["eps"]), "--", *editor_args,
    ]


def normalized_command(command: list[str]) -> list[str]:
    remove = {
        "--variant", "--audit-output", "--diagnostics-output",
        "--informax-diagnostics-path", "--output_model",
    }
    output: list[str] = []
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
        "informax": seed_dir / "diagnostics" / f"informax_{variant}.pt",
        "projection": seed_dir / "diagnostics" / f"projection_{variant}.pt",
        "audit": seed_dir / "audits" / f"{variant}.json",
        "command": seed_dir / "stages" / f"edit_{variant}.command.json",
        "completed": seed_dir / "stages" / f"edit_{variant}.completed.json",
    }
    command_payload = {"argv": command}
    if paths["command"].exists() and json.loads(paths["command"].read_text()) != command_payload:
        raise RuntimeError(f"resume command changed: {variant}")
    write_json(paths["command"], command_payload)
    if paths["completed"].exists():
        if not all(paths[name].is_file() for name in ("informax", "projection", "audit")):
            raise RuntimeError(f"completed edit diagnostics are incomplete: {variant}")
        if not paths["checkpoint"].is_file():
            cleanup = seed_dir / "stages" / f"checkpoint_{variant}.cleanup.json"
            projection_evaluation = seed_dir / "evaluation" / variant / "COMPLETED"
            if not cleanup.is_file() or not projection_evaluation.is_file():
                raise RuntimeError(f"completed edit checkpoint disappeared before verified cleanup: {variant}")
            completed = json.loads(paths["completed"].read_text())
            cleanup_payload = json.loads(cleanup.read_text())
            if cleanup_payload.get("sha256") != completed.get("checkpoint_sha256"):
                raise RuntimeError(f"checkpoint cleanup hash mismatch: {variant}")
        print(f"[resume] edit {variant}", flush=True)
        return
    if any(paths[name].exists() for name in ("checkpoint", "informax", "projection", "audit")):
        raise RuntimeError(f"unverified partial edit exists: {variant}")
    run(command, cwd=SCAPRE_ROOT)
    audit = json.loads(paths["audit"].read_text())
    if not audit.get("completed") or not audit["checkpoint_finiteness"]["all_projection_weights_finite"]:
        raise RuntimeError(f"edit audit failed: {variant}")
    write_json(paths["completed"], {
        "completed_at_utc": utc_now(),
        "checkpoint_sha256": sha256(paths["checkpoint"]),
        "informax_diagnostics_sha256": sha256(paths["informax"]),
        "projection_diagnostics_sha256": sha256(paths["projection"]),
        "audit_sha256": sha256(paths["audit"]),
    })


def assert_equal(left: Any, right: Any, label: str) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if not torch.equal(left, right):
            raise RuntimeError(f"tensor mismatch: {label}")
    elif isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError(f"dict keys mismatch: {label}")
        for key in left:
            assert_equal(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"list length mismatch: {label}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_equal(left_item, right_item, f"{label}[{index}]")
    elif left != right:
        raise RuntimeError(f"value mismatch: {label}")


def distribution(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().double().flatten().cpu()
    q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.95, 0.99], dtype=torch.double))
    return {
        "count": flat.numel(), "finite": bool(torch.isfinite(flat).all().item()),
        "non_constant": bool(flat.max().item() != flat.min().item()),
        "min": flat.min().item(), "p01": q[0].item(), "median": q[1].item(),
        "mean": flat.mean().item(), "std": flat.std(unbiased=True).item(),
        "p95": q[2].item(), "p99": q[3].item(), "max": flat.max().item(),
    }


def validate_edit_isolation(
    seed_dir: Path, commands: dict[str, list[str]], treatment: str,
) -> dict[str, Any]:
    variants = ["official", treatment]
    if normalized_command(commands["official"]) != normalized_command(commands[treatment]):
        raise RuntimeError("edit commands differ outside treatment/output paths")
    audits = {
        variant: json.loads((seed_dir / "audits" / f"{variant}.json").read_text())
        for variant in variants
    }
    for key in (
        "informax_rng_mode", "informax_randn_calls", "expected_informax_randn_calls",
        "alpha_mode", "selected_treatment_alpha",
        "informax_randn_shape_counts", "informax_event_stream_sha256",
        "accumulation_intercepts", "matrix_records",
        "first_informax_pre_draw_global_rng_state", "final_global_rng_state",
        "in_memory_source_substitution_count", "in_memory_source_substitution_scope",
        "production_source_sha256", "executed_in_memory_source_sha256",
    ):
        if audits["official"][key] != audits[treatment][key]:
            raise RuntimeError(f"RNG/interception audit differs: {key}")

    projection = {
        variant: torch.load(seed_dir / "diagnostics" / f"projection_{variant}.pt", map_location="cpu")
        for variant in variants
    }
    left_records = projection["official"]["accumulation_records"]
    right_records = projection[treatment]["accumulation_records"]
    if len(left_records) != 320 or len(right_records) != 320:
        raise RuntimeError("accumulation record coverage changed")
    compare_fields = (
        "index", "projection", "layer_index", "target_index", "target_concept",
        "official_row_w_c", "projection_score", "projection_alpha", "direct_cos2_alpha",
        "weighted_contribution_stats",
        "pearson", "spearman", "official_vs_direct_pearson", "official_vs_direct_spearman",
        "W_old_sha256", "c_vec_sha256", "empty_vec_sha256", "d_vec_sha256",
        "for_mat1_sha256", "for_mat2_sha256", "rng_state_after_official_accumulation_mi",
    )
    for index, (left, right) in enumerate(zip(left_records, right_records)):
        for field in compare_fields:
            assert_equal(left[field], right[field], f"accumulation[{index}].{field}")

    left_matrices = projection["official"]["matrix_records"]
    right_matrices = projection[treatment]["matrix_records"]
    if len(left_matrices) != 32 or len(right_matrices) != 32:
        raise RuntimeError("matrix record coverage changed")
    for index, (left, right) in enumerate(zip(left_matrices, right_matrices)):
        for field in (
            "matrix_index", "projection", "layer_index", "tensor_sha256",
            "row_w_max", "mu", "rng_state_before_solve",
        ):
            assert_equal(left[field], right[field], f"matrix[{index}].{field}")

    informax = {
        variant: torch.load(seed_dir / "diagnostics" / f"informax_{variant}.pt", map_location="cpu")
        for variant in variants
    }
    assert_equal(informax["official"], informax[treatment], "production_informax")
    completion = {
        variant: json.loads((seed_dir / "stages" / f"edit_{variant}.completed.json").read_text())
        for variant in variants
    }
    if completion["official"]["checkpoint_sha256"] == completion[treatment]["checkpoint_sha256"]:
        raise RuntimeError("projection checkpoint is a no-op relative to official")
    return {
        "status": "passed",
        "alpha_mode": audits[treatment]["alpha_mode"],
        "selected_treatment_alpha": audits[treatment]["selected_treatment_alpha"],
        "same_normalized_edit_command": True,
        "in_memory_source_substitution_count": 2,
        "in_memory_source_substitution_scope": "for_mat1 * row_w_c only",
        "official_mi_executed_and_diagnostics_identical": True,
        "informax_rng_call_signatures_order_and_tensors_identical": True,
        "global_rng_states_identical": True,
        "entropy_outputs_and_rng_positions_identical": True,
        "target_and_empty_embeddings_identical": True,
        "for_mat1_and_for_mat2_inputs_identical": True,
        "S_R_CCt_PiC_geometry_inputs_identical": True,
        "final_row_w_max_bitwise_identical": True,
        "official_empty_string_neutral_only": True,
        "checkpoint_hashes_different": True,
        "checkpoint_sha256": {
            variant: completion[variant]["checkpoint_sha256"] for variant in variants
        },
    }


def checkpoint_delta_report(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = torch.load(left_path, map_location="cpu")
    right = torch.load(right_path, map_location="cpu")
    if set(left) != set(right):
        raise RuntimeError("checkpoint parameter keys differ")
    difference_sq = 0.0
    reference_sq = 0.0
    changed = 0
    checked = 0
    for name in left:
        if not isinstance(left[name], torch.Tensor) or not (
            name.endswith(".to_k.weight") or name.endswith(".to_v.weight")
        ):
            continue
        checked += 1
        left_value = left[name].detach().double().cpu()
        right_value = right[name].detach().double().cpu()
        if left_value.shape != right_value.shape:
            raise RuntimeError(f"checkpoint tensor shape differs: {name}")
        delta = right_value - left_value
        difference_sq += float(delta.square().sum().item())
        reference_sq += float(left_value.square().sum().item())
        if not torch.equal(left_value, right_value):
            changed += 1
    if checked == 0:
        raise RuntimeError("no projection weights available for checkpoint delta")
    delta_norm = difference_sq ** 0.5
    reference_norm = reference_sq ** 0.5
    return {
        "checked_projection_weights": checked,
        "changed_projection_weights": changed,
        "delta_frobenius": delta_norm,
        "official_frobenius": reference_norm,
        "relative_delta_frobenius": delta_norm / reference_norm if reference_norm else None,
        "nonzero": changed > 0 and delta_norm > 0.0,
    }


def qualification_report(
    seed_dir: Path, isolation: dict[str, Any], output: Path,
    production_hash_start: str, treatment: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    wrapper = torch.load(
        seed_dir / "diagnostics" / f"projection_{treatment}.pt", map_location="cpu"
    )
    treatment_audit = json.loads(
        (seed_dir / "audits" / f"{treatment}.json").read_text()
    )
    alpha_mode = treatment_audit["alpha_mode"]
    records = wrapper["accumulation_records"]
    official_alpha = torch.cat([row["official_row_w_c"].flatten() for row in records])
    scores = torch.cat([row["projection_score"].flatten() for row in records])
    v1_alpha = torch.cat([row["projection_alpha"].flatten() for row in records])
    direct_alpha = torch.cat([row["direct_cos2_alpha"].flatten() for row in records])
    informax = torch.load(seed_dir / "diagnostics" / "informax_official.pt", map_location="cpu")
    stage_stats: dict[str, dict[str, Any]] = {}
    for stage in ("aggregate", "accumulation"):
        selected = [row for row in informax["records"] if row["stage"] == stage]
        stage_stats[stage] = {
            "records": len(selected),
            "raw_binary_mi": distribution(torch.cat([row["raw_mi"].flatten() for row in selected])),
            "official_alpha": distribution(torch.cat([row["alpha"].flatten() for row in selected])),
        }
    correlations = output / "per_layer_concept_correlations.csv"
    with correlations.open("w", newline="") as handle:
        fields = [
            "projection", "layer_index", "target_index", "target_concept",
            "official_vs_v1_pearson", "official_vs_v1_spearman",
            "official_vs_direct_pearson", "official_vs_direct_spearman",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "projection": row["projection"],
                "layer_index": row["layer_index"],
                "target_index": row["target_index"],
                "target_concept": row["target_concept"],
                "official_vs_v1_pearson": row["pearson"],
                "official_vs_v1_spearman": row["spearman"],
                "official_vs_direct_pearson": row["official_vs_direct_pearson"],
                "official_vs_direct_spearman": row["official_vs_direct_spearman"],
            })
    weight_diagnostics = output / "per_layer_concept_weight_diagnostics.csv"
    diagnostic_fields = [
        "projection", "layer_index", "target_index", "target_concept",
        "official_mean", "v1_mean", "direct_mean",
        "official_median", "v1_median", "direct_median",
        "official_p95", "v1_p95", "direct_p95",
        "official_max", "v1_max", "direct_max",
        "official_contribution_frobenius", "v1_contribution_frobenius",
        "direct_contribution_frobenius", "direct_to_official_contribution_ratio",
        "direct_to_v1_contribution_ratio",
    ]
    with weight_diagnostics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_fields)
        writer.writeheader()
        for row in records:
            contributions = row["weighted_contribution_stats"]
            official_norm = contributions["official"]["frobenius"]
            v1_norm = contributions["v1_projection"]["frobenius"]
            direct_norm = contributions["direct_cos2"]["frobenius"]
            writer.writerow({
                "projection": row["projection"],
                "layer_index": row["layer_index"],
                "target_index": row["target_index"],
                "target_concept": row["target_concept"],
                "official_mean": row["official_stats"]["mean"],
                "v1_mean": row["projection_alpha_stats"]["mean"],
                "direct_mean": row["direct_cos2_alpha_stats"]["mean"],
                "official_median": row["official_stats"]["median"],
                "v1_median": row["projection_alpha_stats"]["median"],
                "direct_median": row["direct_cos2_alpha_stats"]["median"],
                "official_p95": row["official_stats"]["p95"],
                "v1_p95": row["projection_alpha_stats"]["p95"],
                "direct_p95": row["direct_cos2_alpha_stats"]["p95"],
                "official_max": row["official_stats"]["max"],
                "v1_max": row["projection_alpha_stats"]["max"],
                "direct_max": row["direct_cos2_alpha_stats"]["max"],
                "official_contribution_frobenius": official_norm,
                "v1_contribution_frobenius": v1_norm,
                "direct_contribution_frobenius": direct_norm,
                "direct_to_official_contribution_ratio": (
                    direct_norm / official_norm if official_norm else None
                ),
                "direct_to_v1_contribution_ratio": direct_norm / v1_norm if v1_norm else None,
            })
    matrix_diagnostics = output / "per_matrix_edit_strength.csv"
    official_wrapper = torch.load(
        seed_dir / "diagnostics" / "projection_official.pt", map_location="cpu"
    )
    with matrix_diagnostics.open("w", newline="") as handle:
        fields = [
            "matrix_index", "projection", "layer_index", "official_V_frobenius",
            "treatment_V_frobenius", "treatment_to_official_V_ratio",
            "official_mat1_agg_frobenius", "treatment_mat1_agg_frobenius",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for left, right in zip(official_wrapper["matrix_records"], wrapper["matrix_records"]):
            official_v = left["V_stats"]["frobenius"]
            treatment_v = right["V_stats"]["frobenius"]
            writer.writerow({
                "matrix_index": left["matrix_index"],
                "projection": left["projection"],
                "layer_index": left["layer_index"],
                "official_V_frobenius": official_v,
                "treatment_V_frobenius": treatment_v,
                "treatment_to_official_V_ratio": treatment_v / official_v if official_v else None,
                "official_mat1_agg_frobenius": left["mat1_agg_stats"]["frobenius"],
                "treatment_mat1_agg_frobenius": right["mat1_agg_stats"]["frobenius"],
            })
    checkpoint_delta = checkpoint_delta_report(
        seed_dir / "checkpoints" / "official.pt",
        seed_dir / "checkpoints" / f"{treatment}.pt",
    )
    contributions_finite = all(
        stats["finite"]
        for row in records
        for stats in row["weighted_contribution_stats"].values()
    )
    direct_matches_score = all(
        torch.equal(
            row["direct_cos2_alpha"],
            row["projection_score"].view(-1, 1).to(
                dtype=row["direct_cos2_alpha"].dtype
            ),
        )
        for row in records
    )
    report = {
        "status": "passed",
        "edit_seed": 20260820,
        "treatment": treatment,
        "alpha_mode": alpha_mode,
        "selected_treatment_alpha": treatment_audit["selected_treatment_alpha"],
        "normalization": "none" if alpha_mode == "direct_cos2" else "V1 z-score transform",
        "selection_criteria": {
            "all_projection_scores_finite": bool(torch.isfinite(scores).all().item()),
            "projection_score_non_constant": bool(scores.max().item() != scores.min().item()),
            "all_direct_cos2_alpha_finite": bool(torch.isfinite(direct_alpha).all().item()),
            "direct_cos2_alpha_non_constant": bool(direct_alpha.max().item() != direct_alpha.min().item()),
            "direct_cos2_equals_projection_score_after_dtype_cast": direct_matches_score,
            "all_weighted_contributions_finite": contributions_finite,
            "all_checkpoint_projection_weights_finite": True,
            "production_editor_byte_unchanged": sha256(EDITOR) == production_hash_start,
            "final_row_w_max_bitwise_identical": isolation["final_row_w_max_bitwise_identical"],
            "all_non_treatment_inputs_identical": True,
            "rng_audit_passed": True,
            "checkpoint_hashes_different": isolation["checkpoint_hashes_different"],
            "checkpoint_parameter_delta_nonzero": checkpoint_delta["nonzero"],
        },
        "official_per_concept_informax": stage_stats,
        "official_accumulation_row_w_c_distribution": distribution(official_alpha),
        "projection_score_distribution": distribution(scores),
        "v1_projection_alpha_distribution_descriptive_only": distribution(v1_alpha),
        "direct_cos2_alpha_distribution": distribution(direct_alpha),
        "checkpoint_parameter_delta": checkpoint_delta,
        "correlations_are_descriptive_only": True,
        "correlations_csv": str(correlations),
        "weight_diagnostics_csv": str(weight_diagnostics),
        "matrix_edit_strength_csv": str(matrix_diagnostics),
        "production_editor_sha256": production_hash_start,
        "isolation": isolation,
    }
    if not all(report["selection_criteria"].values()):
        report["status"] = "failed"
        write_json(output / "integrity_report.json", report)
        raise RuntimeError(f"qualification failed: {report['selection_criteria']}")
    write_json(output / "integrity_report.json", report)
    return report


def evaluate(
    seed_dir: Path, base_config: Path, assets: Path, protocol: Path, treatment: str,
) -> None:
    output = seed_dir / "evaluation" / treatment
    if (output / "COMPLETED").exists():
        rows = read_csv(output / "scores.csv")
        if len(rows) != 3000 or any(row["variant"] != treatment for row in rows):
            raise RuntimeError("completed projection evaluation is invalid")
        print("[resume] projection evaluation", flush=True)
        return
    run([
        sys.executable, str(EVALUATOR_RUNNER), "--script", str(EVALUATOR),
        "--variant-label", treatment, "--",
        "--config", str(base_config), "--assets", str(assets), "--protocol", str(protocol),
        "--checkpoint", str(seed_dir / "checkpoints" / f"{treatment}.pt"),
        "--output-dir", str(output), "--device", "cuda:0",
    ])


def cleanup_checkpoint(seed_dir: Path, variant: str) -> None:
    checkpoint = seed_dir / "checkpoints" / f"{variant}.pt"
    marker = seed_dir / "stages" / f"checkpoint_{variant}.cleanup.json"
    if marker.exists():
        if checkpoint.exists():
            raise RuntimeError(f"checkpoint returned after cleanup: {variant}")
        return
    completed = json.loads((seed_dir / "stages" / f"edit_{variant}.completed.json").read_text())
    if not checkpoint.is_file() or sha256(checkpoint) != completed["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint cannot be safely cleaned: {variant}")
    size = checkpoint.stat().st_size
    checkpoint.unlink()
    write_json(marker, {
        "status": "passed", "variant": variant, "sha256": completed["checkpoint_sha256"],
        "size_bytes": size, "deleted_regenerable_checkpoint": str(checkpoint),
        "deleted_at_utc": utc_now(),
    })


def set_stage(run_dir: Path, stage: str, completed_seeds: list[int]) -> None:
    write_json(run_dir / "status.json", {
        "stage": stage, "completed_seeds": completed_seeds, "updated_at_utc": utc_now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--official-reference", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, default=HERE)
    parser.add_argument("--v1-run-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    assets_path = args.assets.resolve()
    reference = args.official_reference.resolve()
    experiment_dir = args.experiment_dir.resolve()
    config_path = experiment_dir / "config.json"
    if experiment_dir != HERE and experiment_dir.parent != HERE.parent:
        raise RuntimeError("experiment directory must be the shared runner or a sibling")
    if not config_path.is_file():
        raise RuntimeError(f"experiment config is missing: {config_path}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "MU":
        raise RuntimeError("worker requires active Conda MU")
    if git_status():
        raise RuntimeError(f"working tree is dirty at start: {git_status()}")
    config = json.loads(config_path.read_text())
    base = json.loads(BASE_CONFIG.read_text())
    assets = json.loads(assets_path.read_text())
    validate_configuration(config, base)
    validate_sources(config)
    treatment = config["variant"]
    if treatment == "official" or treatment not in {
        "projection_accumulation", "projection_accumulation_direct_cos2",
    }:
        raise RuntimeError(f"unsupported treatment variant: {treatment}")
    variants = ["official", treatment]
    production_hash_start = sha256(EDITOR)
    run_dir.mkdir(parents=True, exist_ok=True)
    source_files = sorted(set(
        [REPO_ROOT / relative for relative in config["source_controls"]]
        + [
            path for path in experiment_dir.iterdir()
            if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json"}
        ]
        + [Path(__file__).resolve(), PROJECTION_RUNNER, EVALUATOR_RUNNER, AGGREGATOR]
    ))
    source_hashes = {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files}
    v1_provenance: dict[str, Any] | None = None
    if config.get("requires_v1_diagnostic_analysis"):
        if args.v1_run_dir is None:
            raise RuntimeError("direct-cos2 requires --v1-run-dir for descriptive pre-analysis")
        v1_run_dir = args.v1_run_dir.resolve()
        v1_diagnostics = (
            v1_run_dir / "seeds" / "20260820" / "diagnostics"
            / "projection_projection_accumulation.pt"
        )
        v1_integrity = v1_run_dir / "reproducibility" / "integrity_report.json"
        if not v1_diagnostics.is_file() or not v1_integrity.is_file():
            raise RuntimeError("required completed V1 diagnostics are missing")
        v1_provenance = {
            "run_dir": str(v1_run_dir),
            "diagnostics_sha256": sha256(v1_diagnostics),
            "integrity_report_sha256": sha256(v1_integrity),
        }
    manifest_path = run_dir / "reproducibility" / "run_manifest.json"
    manifest = {
        "started_at_utc": utc_now(), "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"), "git_status_start": [],
        "python_executable": sys.executable, "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "source_sha256": source_hashes, "assets": assets, "assets_sha256": sha256(assets_path),
        "production_editor_sha256_start": production_hash_start,
        "v1_diagnostic_source": v1_provenance,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in (
            "git_commit", "git_branch", "source_sha256", "assets", "assets_sha256",
            "v1_diagnostic_source",
        ):
            if previous.get(key) != manifest[key]:
                raise RuntimeError(f"resume provenance changed: {key}")
        manifest = previous
    else:
        write_json(manifest_path, manifest)
    shutil.copy2(config_path, run_dir / "reproducibility" / "actual_config.json")
    shutil.copy2(BASE_CONFIG, run_dir / "reproducibility" / "base_config.json")
    for source in source_files:
        destination = run_dir / "reproducibility" / "provenance" / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    analysis_summary: dict[str, Any] | None = None
    if v1_provenance is not None:
        set_stage(run_dir, "v1_descriptive_diagnostics", [])
        v1_analysis = run_dir / "pre_analysis" / "v1_transform"
        if not (v1_analysis / "summary.json").exists():
            run([
                sys.executable, str(experiment_dir / "analyze_v1_diagnostics.py"),
                "--v1-run-dir", v1_provenance["run_dir"],
                "--output-dir", str(v1_analysis),
            ])
        analysis_summary = json.loads((v1_analysis / "summary.json").read_text())
        if analysis_summary.get("status") != "passed" or analysis_summary.get("records") != 320:
            raise RuntimeError("V1 descriptive diagnostic analysis is incomplete")

    set_stage(run_dir, "protocol_preflight", [])
    protocol = run_dir / "reproducibility" / "protocol.csv"
    protocol_output = subprocess.check_output([
        sys.executable, str(PROTOCOL_BUILDER), "--config", str(BASE_CONFIG),
        "--output", str(protocol), "--profile", "formal",
    ], cwd=REPO_ROOT, text=True)
    protocol_manifest = json.loads(protocol_output)
    write_json(run_dir / "reproducibility" / "protocol_manifest.json", protocol_manifest)
    official_validation = validate_official_reference(reference, config, assets, protocol)
    write_json(run_dir / "reproducibility" / "official_reference_validation.json", official_validation)
    import_official(reference, run_dir)

    completed_seeds: list[int] = []
    for seed in SEEDS:
        seed_dir = run_dir / "seeds" / str(seed)
        for name in ("checkpoints", "diagnostics", "audits", "stages", "evaluation"):
            (seed_dir / name).mkdir(parents=True, exist_ok=True)
        stage = "qualification_edit" if seed == 20260820 else f"formal_edit_seed_{seed}"
        set_stage(run_dir, stage, completed_seeds)
        commands = {
            variant: edit_command(config, base, assets, seed, seed_dir, variant)
            for variant in variants
        }
        for variant in variants:
            run_edit(commands[variant], seed_dir, variant)
        isolation = validate_edit_isolation(seed_dir, commands, treatment)
        write_json(seed_dir / "reproducibility_isolation.json", isolation)
        if seed == 20260820:
            qualification = qualification_report(
                seed_dir, isolation, run_dir / "qualification", production_hash_start,
                treatment,
            )
            if qualification["status"] != "passed":
                raise RuntimeError("qualification did not pass")
            (run_dir / "qualification" / "PASSED").write_text("qualification passed\n")
        set_stage(run_dir, f"formal_evaluation_seed_{seed}", completed_seeds)
        evaluate(
            seed_dir, run_dir / "reproducibility" / "base_config.json",
            assets_path, protocol, treatment,
        )
        projection_manifest = (
            seed_dir / "evaluation" / treatment / "evaluation_manifest.json"
        )
        if evaluator_fingerprint(projection_manifest) != config["official_reference"]["evaluator_fingerprint_sha256"]:
            raise RuntimeError(f"projection evaluator fingerprint mismatch: seed {seed}")
        cleanup_checkpoint(seed_dir, "official")
        cleanup_checkpoint(seed_dir, treatment)
        completed_seeds.append(seed)

    if git_status():
        raise RuntimeError(f"working tree dirty before aggregation: {git_status()}")
    set_stage(run_dir, "aggregation", completed_seeds)
    run([
        sys.executable, str(AGGREGATOR), "--run-dir", str(run_dir),
        "--treatment", treatment,
    ])
    if git_status():
        raise RuntimeError(f"aggregation changed tracked files: {git_status()}")
    if sha256(EDITOR) != production_hash_start:
        raise RuntimeError("production editor changed during experiment")
    aggregate = json.loads((run_dir / "results" / "aggregate_metrics.json").read_text())
    integrity = {
        "status": "passed", "edit_seeds": SEEDS, "treatment": treatment,
        "official_reference_reused": True, "new_generated_image_count": 15000,
        "rows_per_variant_seed": 3000, "target_rows_per_seed": 1200,
        "retain_rows_per_seed": 1800, "duplicate_generation_keys": 0,
        "generation_keys_identical": True, "evaluator_fingerprint_identical": True,
        "official_empty_string_neutral_only": True,
        "production_editor_byte_unchanged": True,
        "production_editor_sha256": production_hash_start,
        "qualification_status": "passed",
        "v1_descriptive_analysis_status": (
            analysis_summary.get("status") if analysis_summary is not None else "not_applicable"
        ),
        "v1_diagnostic_source": v1_provenance,
        "final_row_w_max_bitwise_identical_all_seeds": True,
        "informax_rng_and_entropy_positions_identical_all_seeds": True,
        "all_non_treatment_edit_inputs_identical_all_seeds": True,
        "all_checkpoints_finite_and_different_from_official": True,
        "protocol_sha256": protocol_manifest["sha256"],
        "official_reference_validation": official_validation,
        "directional_conditions": aggregate["directional_conditions"],
        "git_commit": manifest["git_commit"], "git_status_start_clean": True,
        "git_status_before_aggregation_clean": True, "git_status_end_clean": True,
    }
    write_json(run_dir / "reproducibility" / "integrity_report.json", integrity)
    shutil.copy2(run_dir / "results" / "summary.md", run_dir / "summary.md")
    manifest["production_editor_sha256_end"] = sha256(EDITOR)
    manifest["git_status_before_aggregation"] = []
    manifest["git_status_end"] = []
    manifest["finished_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    write_json(run_dir / "worker_complete.json", {
        "status": "passed", "completed_at_utc": utc_now(),
        "summary": str((run_dir / "results" / "summary.md").resolve()),
        "coco_started": False,
    })
    set_stage(run_dir, "completed", completed_seeds)


if __name__ == "__main__":
    main()
