#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active Conda environment. Activate MU on the GPU server first." >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
    echo "This diagnostic must run in the GPU server Conda environment MU." >&2
    exit 1
fi

PYTHON_BIN="$(command -v python)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "Could not resolve the active MU environment's python executable." >&2
    exit 1
fi

exec "$PYTHON_BIN" - "$HERE" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any


here = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(here))

import pipeline
import protocol


targets = ["yawl", "volleyball"]
config, anchors = protocol.load_protocol(here / "config.json")
primary_root = Path(config["_resolved"]["output_root"])
diagnostic_root = primary_root / "target_only_diagnostic_v1"
formal_completion = primary_root / "formal" / "completion.json"
summary_path = diagnostic_root / "summary.json"

if summary_path.is_file():
    completed = protocol.read_json(summary_path)
    if completed.get("status") == "complete":
        print(json.dumps(completed, indent=2, ensure_ascii=False))
        raise SystemExit(0)

if formal_completion.exists():
    raise RuntimeError(
        "Formal completion exists; refusing to mix this target-only diagnostic "
        "with an already completed formal run."
    )

anchor_gate = pipeline._require_gate(primary_root, "anchor_sanity")
canary_gate = pipeline._require_gate(primary_root, "original_canary")
smoke_gate_path = primary_root / "smoke" / "gate.json"
smoke_gate = protocol.read_json(smoke_gate_path)
if smoke_gate.get("status") != "failed" or smoke_gate.get("failed_targets") != ["yawl"]:
    raise RuntimeError("Expected the preserved yawl Single smoke failure before diagnosis")

all_rows, rows_by_class = pipeline.load_dataset(config)
if len(all_rows) != 12500:
    raise RuntimeError("The matched evaluation dataset must contain 12,500 rows")

reference, archive_root = pipeline._legacy_reference(config)
legacy_originals = pipeline._load_legacy_original_results(config, archive_root)
checkpoint_lookup = pipeline._checkpoint_lookup(config, anchors)

specs: dict[str, dict[str, Any]] = {}
for target in targets:
    group = pipeline._group_for_target(config, target)
    spec = checkpoint_lookup[(group["id"], "single", target)]
    pipeline._validate_checkpoint(spec, config)
    specs[target] = spec

diagnostic_root.mkdir(parents=True, exist_ok=True)
protocol.write_json_atomic(diagnostic_root / "resolved_plan.json", {
    "schema_version": 1,
    "status": "resolved",
    "purpose": (
        "Distinguish true Single-OCE erasure failure from a low-baseline, "
        "32-sample exact-top1 readout artifact for yawl; complete the same "
        "target-only Single diagnostic for volleyball."
    ),
    "targets": targets,
    "rows_per_target": 500,
    "new_single_images": 1000,
    "new_joint_images": 0,
    "new_preservation_images": 0,
    "new_original_images": 0,
    "formal_pipeline_enabled": False,
    "checkpoint_modification_forbidden": True,
    "legacy_original_reference": reference,
    "anchor_gate_fingerprint": anchor_gate.get("protocol_fingerprint"),
    "canary_checked_images": canary_gate.get("checked_images"),
    "source_hashes": protocol.source_hashes([
        here / "run_yawl_volleyball_target_diagnostic.sh"
    ]),
    "created_at": protocol.utc_now(),
})

generation = pipeline.GenerationRuntime(config)
evaluator = pipeline.EvaluationRuntime(config)
results: dict[str, dict[str, Any]] = {}

for target in targets:
    group = pipeline._group_for_target(config, target)
    rows = rows_by_class[protocol.normalize(target)]
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 ordered rows for {target}, found {len(rows)}")
    spec = specs[target]
    identity = pipeline._job_identity(
        "single", group, target, target, spec, rows, config
    )
    job_id = f"single__{group['id']}__{protocol.slug(target)}__target-only-500"
    job = {
        **identity,
        "job_id": job_id,
        "job_fingerprint": protocol.fingerprint(identity),
        "checkpoint_spec": spec,
        "rows": rows,
        "image_dir": str(diagnostic_root / "images" / protocol.slug(target)),
        "manifest_path": str(diagnostic_root / "manifests" / f"{job_id}.json"),
        "result_path": str(diagnostic_root / "evaluations" / f"{job_id}.json"),
    }
    print(f"[target diagnostic] generating/evaluating {target}: 500 Single rows", flush=True)
    results[target] = pipeline._run_formal_job(
        job,
        generation,
        evaluator,
        skip_existing=True,
        purge=False,
    )

smoke_items = protocol.read_json(primary_root / "smoke" / "per_image.json")["items"]
yawl_original_smoke = {
    int(item["case_number"]): item
    for item in smoke_items
    if item.get("target") == "yawl" and item.get("model_type") == "original"
}
if len(yawl_original_smoke) != 32:
    raise RuntimeError(
        f"Expected 32 retained yawl Original smoke images, found {len(yawl_original_smoke)}"
    )

yawl_single_items = {
    int(item["case_number"]): item for item in results["yawl"]["items"]
}
qualitative_pairs: list[dict[str, Any]] = []
pair_root = diagnostic_root / "qualitative_pairs" / "yawl"

for case_number in sorted(yawl_original_smoke):
    original_item = yawl_original_smoke[case_number]
    single_item = yawl_single_items[case_number]
    original_source = Path(original_item["image_path"])
    single_source = Path(single_item["image_path"])

    case_root = pair_root / f"case-{case_number:06d}"
    case_root.mkdir(parents=True, exist_ok=True)
    original_copy = case_root / "original.png"
    single_copy = case_root / "single.png"
    for source, destination, expected_hash in (
        (original_source, original_copy, original_item["image_sha256"]),
        (single_source, single_copy, single_item["image_sha256"]),
    ):
        if destination.exists():
            if protocol.sha256(destination) != expected_hash:
                raise RuntimeError(f"Qualitative pair collision: {destination}")
        else:
            if not source.is_file():
                raise RuntimeError(
                    f"Missing matched qualitative source for case {case_number}: {source}"
                )
            if protocol.sha256(source) != expected_hash:
                raise RuntimeError(
                    f"Qualitative source hash mismatch for case {case_number}: {source}"
                )
            shutil.copy2(source, destination)
        if protocol.sha256(destination) != expected_hash:
            raise RuntimeError(f"Copied qualitative hash mismatch: {destination}")

    if original_item["correct"] and not single_item["correct"]:
        transition = "original_correct_to_single_wrong"
    elif original_item["correct"] and single_item["correct"]:
        transition = "original_correct_to_single_correct"
    elif not original_item["correct"] and single_item["correct"]:
        transition = "original_wrong_to_single_correct"
    else:
        transition = "original_wrong_to_single_wrong"
    qualitative_pairs.append({
        "case_number": case_number,
        "prompt": single_item["prompt"],
        "seed": single_item["evaluation_seed"],
        "transition": transition,
        "original_path": str(original_copy.resolve()),
        "original_sha256": protocol.sha256(original_copy),
        "single_path": str(single_copy.resolve()),
        "single_sha256": protocol.sha256(single_copy),
    })

if len(qualitative_pairs) != 32:
    raise RuntimeError("Exactly 32 matched yawl qualitative pairs must be retained")
protocol.write_json_atomic(diagnostic_root / "qualitative_pairs.json", {
    "schema_version": 1,
    "target": "yawl",
    "pair_count": len(qualitative_pairs),
    "pairs": qualitative_pairs,
})

summary_rows: list[dict[str, Any]] = []
for target in targets:
    group = pipeline._group_for_target(config, target)
    original_result = legacy_originals[(group["id"], protocol.normalize(target))]
    single_result = results[target]
    original_by_case = {
        int(item["case_number"]): item for item in original_result["items"]
    }
    single_by_case = {
        int(item["case_number"]): item for item in single_result["items"]
    }
    if set(original_by_case) != set(single_by_case) or len(single_by_case) != 500:
        raise RuntimeError(f"Matched Original/Single case mismatch for {target}")

    transitions = {
        "original_correct_to_single_wrong": 0,
        "original_correct_to_single_correct": 0,
        "original_wrong_to_single_correct": 0,
        "original_wrong_to_single_wrong": 0,
    }
    for case_number in sorted(original_by_case):
        original_correct = bool(original_by_case[case_number]["correct"])
        single_correct = bool(single_by_case[case_number]["correct"])
        if original_correct and not single_correct:
            transitions["original_correct_to_single_wrong"] += 1
        elif original_correct and single_correct:
            transitions["original_correct_to_single_correct"] += 1
        elif not original_correct and single_correct:
            transitions["original_wrong_to_single_correct"] += 1
        else:
            transitions["original_wrong_to_single_wrong"] += 1

    original_correct_count = int(original_result["correct"])
    single_correct_count = int(single_result["correct"])
    if sum(transitions.values()) != 500:
        raise RuntimeError(f"Transition count does not sum to 500 for {target}")
    if (
        transitions["original_correct_to_single_wrong"]
        + transitions["original_correct_to_single_correct"]
        != original_correct_count
    ):
        raise RuntimeError(f"Original-correct transition denominator mismatch for {target}")

    summary_rows.append({
        "target": target,
        "total": 500,
        "original_correct": original_correct_count,
        "original_target_accuracy": original_correct_count / 500,
        "single_correct": single_correct_count,
        "single_target_accuracy": single_correct_count / 500,
        "single_minus_original_accuracy": (
            single_correct_count - original_correct_count
        ) / 500,
        "matched_transitions": transitions,
        "conditional_erasure_rate_among_original_correct": (
            transitions["original_correct_to_single_wrong"]
            / original_correct_count
            if original_correct_count
            else None
        ),
        "legacy_original_auxiliary_metrics": "unavailable",
        "single_mean_target_probability": single_result["mean_target_probability"],
        "single_mean_raw_target_logit": single_result["mean_raw_target_logit"],
    })

# Preserve only the 32 copied yawl pairs. Full per-image metrics and hashes are
# already durable, so the remaining 1,000 temporary Single PNGs can be removed.
for target in targets:
    result = results[target]
    manifest_path = diagnostic_root / "manifests" / (
        f"single__{pipeline._group_for_target(config, target)['id']}__"
        f"{protocol.slug(target)}__target-only-500.json"
    )
    manifest = protocol.read_json(manifest_path)
    for item in manifest["items"]:
        image_path = Path(item["image_path"])
        if image_path.is_file():
            if protocol.sha256(image_path) != item["image_sha256"]:
                raise RuntimeError(f"Refusing to purge hash-mismatched image: {image_path}")
            image_path.unlink()
        item["image_status"] = "purged_after_durable_evaluation"
    manifest["status"] = "selectively_purged"
    manifest["qualitative_copies_retained"] = 32 if target == "yawl" else 0
    manifest["purged_at"] = protocol.utc_now()
    protocol.write_json_atomic(manifest_path, manifest)

report = {
    "schema_version": 1,
    "status": "complete",
    "formal_pipeline_enabled": False,
    "new_single_images_generated": 1000,
    "new_joint_images_generated": 0,
    "new_preservation_images_generated": 0,
    "new_original_images_generated": 0,
    "legacy_original_probability_logit_top5": "unavailable_not_regenerated",
    "yawl_qualitative_pair_count": 32,
    "yawl_qualitative_pairs_root": str(pair_root.resolve()),
    "results": summary_rows,
    "generation_runtime": generation.identity,
    "evaluator": evaluator.identity,
    "completed_at": protocol.utc_now(),
}
protocol.write_json_atomic(summary_path, report)
print(json.dumps(report, indent=2, ensure_ascii=False))
PY
