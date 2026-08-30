#!/usr/bin/env python3
"""Lightweight MU-environment self-test; no model, dataset, or generation."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SHARED_WORKER = HERE.parent / "projection_accumulation" / "worker.py"
CONFIG = HERE / "config.json"
DIRECT_RESULTS = HERE.parent / "projection_accumulation_direct_cos2" / "formal_results"


def load_worker():
    specification = importlib.util.spec_from_file_location(
        "budget_matched_projection_worker", SHARED_WORKER
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load shared worker")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def arithmetic_self_test() -> None:
    eps = 1e-8
    before = torch.random.get_rng_state().clone()
    for_mat1 = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    official_alpha = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float32)
    direct_alpha = torch.tensor([[0.01], [0.03], [0.02]], dtype=torch.float32)
    official = for_mat1 * official_alpha
    geo = for_mat1 * direct_alpha
    official_norm = torch.linalg.vector_norm(official)
    geo_norm = torch.linalg.vector_norm(geo)
    if official_norm.item() <= eps or geo_norm.item() <= eps:
        raise RuntimeError("synthetic norm unexpectedly degenerate")
    budget_lambda = official_norm / (geo_norm + eps)
    new = budget_lambda * geo
    new_norm = torch.linalg.vector_norm(new)
    if not torch.isfinite(budget_lambda).item() or budget_lambda.item() <= 0.0:
        raise RuntimeError("synthetic lambda is invalid")
    if not torch.isclose(new_norm, official_norm, rtol=1e-5, atol=1e-7).item():
        raise RuntimeError("synthetic contribution budget does not match")
    if not torch.equal(before, torch.random.get_rng_state()):
        raise RuntimeError("deterministic budget arithmetic consumed RNG")


def report_self_test() -> None:
    worker = load_worker()
    config = json.loads(CONFIG.read_text())
    historical = worker.validate_historical_comparisons(config)
    treatment = config["variant"]
    official = json.loads((DIRECT_RESULTS / "aggregate_metrics.json").read_text())[
        "official_five_seed_mean"
    ]
    aggregate = {
        "official_five_seed_mean": official,
        f"{treatment}_five_seed_mean": official,
        "treatment_minus_official": {
            "unlearn": 0.0, "preserve": 0.0, "overall": 0.0,
        },
        "favorable_seeds": {"unlearn": 0, "preserve": 0, "overall": 0},
        "directional_conditions": {
            "mean_delta_unlearn_negative": False,
            "mean_delta_preserve_positive": False,
            "mean_delta_overall_positive": False,
            "overall_favorable_at_least_4_of_5": False,
            "group_target_pattern_requires_manual_review": True,
            "automatic_directional_conditions_passed": False,
        },
    }
    with tempfile.TemporaryDirectory(prefix="budget_matched_cos2_preflight_") as temporary:
        results = Path(temporary)
        for name in ("per_group_metrics.csv", "per_target_metrics.csv"):
            shutil.copy2(DIRECT_RESULTS / name, results / name)
        worker.write_formal_reports(results, aggregate, treatment, historical)
        for name in (
            "historical_variant_comparison.csv",
            "validation_report.md",
            "result_manifest.json",
        ):
            if not (results / name).is_file():
                raise RuntimeError(f"report self-test output missing: {name}")
        manifest = json.loads((results / "result_manifest.json").read_text())
        if manifest.get("status") != "passed" or not manifest.get(
            "retrieval_validation_pending"
        ):
            raise RuntimeError("report self-test manifest is invalid")


def main() -> None:
    if REPO_ROOT != Path.cwd().resolve():
        raise RuntimeError("preflight self-test must run from repository root")
    arithmetic_self_test()
    report_self_test()
    print("Budget-matched-cos2 lightweight preflight self-test passed.")


if __name__ == "__main__":
    main()
