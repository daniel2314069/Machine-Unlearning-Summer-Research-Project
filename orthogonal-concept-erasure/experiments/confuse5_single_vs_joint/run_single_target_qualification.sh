#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
STATE_ROOT="$HERE/outputs/official_repo_primary_v1/single_target_qualification_v1"
PID_FILE="$STATE_ROOT/detached.pid"
LATEST_FILE="$STATE_ROOT/detached.latest"
EXIT_FILE="$STATE_ROOT/detached.exit_code"

require_mu() {
    if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
        echo "Activate the GPU server Conda environment MU first." >&2
        exit 1
    fi
    if [[ -z "$(command -v python)" ]]; then
        echo "Could not resolve python from the active MU environment." >&2
        exit 1
    fi
}

show_status() {
    local pid=""
    local latest=""
    local exit_code=""
    [[ -f "$PID_FILE" ]] && pid="$(tr -d '[:space:]' < "$PID_FILE")"
    [[ -f "$LATEST_FILE" ]] && latest="$(tr -d '\n' < "$LATEST_FILE")"
    [[ -f "$EXIT_FILE" ]] && exit_code="$(tr -d '[:space:]' < "$EXIT_FILE")"
    echo "PID: ${pid:-unknown}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Process status: RUNNING"
    else
        echo "Process status: NOT RUNNING"
    fi
    if [[ -n "$exit_code" ]]; then
        echo "Exit status: $exit_code"
    else
        echo "Exit status: pending or unavailable"
    fi
    echo "Log: ${latest:-unknown}"
    if [[ -f "$STATE_ROOT/summary.json" ]]; then
        echo "Qualification summary: COMPLETE"
    else
        echo "Qualification summary: missing"
    fi
    completed_targets="0"
    retained_images="0"
    if [[ -d "$STATE_ROOT/evaluations" ]]; then
        completed_targets="$(find "$STATE_ROOT/evaluations" -type f -name '*.json' | wc -l | tr -d '[:space:]')"
    fi
    if [[ -d "$STATE_ROOT/images" ]]; then
        retained_images="$(find "$STATE_ROOT/images" -type f -name '*.png' | wc -l | tr -d '[:space:]')"
    fi
    echo "New target evaluations complete: $completed_targets / 8"
    echo "Current target images generated: $retained_images / 500"
    if [[ -n "$latest" && -f "$latest" ]]; then
        echo "Latest log lines:"
        tail -n 20 "$latest"
    fi
}

run_worker() {
    local python_bin
    python_bin="$(command -v python)"
    "$python_bin" - "$HERE" <<'PY'
from __future__ import annotations

import copy
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


here = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(here))

import pipeline
import protocol


config, anchors = protocol.load_protocol(here / "config.json")
primary_root = Path(config["_resolved"]["output_root"])
qualification_root = primary_root / "single_target_qualification_v1"
summary_path = qualification_root / "summary.json"
audit_path = qualification_root / "scheduler_provenance_audit.json"
diagnostic_root = primary_root / "target_only_diagnostic_v1"
diagnostic_summary_path = diagnostic_root / "summary.json"
formal_completion = primary_root / "formal" / "completion.json"
reused_targets = {"yawl", "volleyball"}

if summary_path.is_file():
    completed = protocol.read_json(summary_path)
    if completed.get("status") == "complete":
        print(json.dumps(completed, indent=2, ensure_ascii=False))
        raise SystemExit(0)

if formal_completion.exists():
    raise RuntimeError(
        "Formal completion exists; refusing to run baseline qualification after formal generation."
    )

pipeline._require_gate(primary_root, "anchor_sanity")
pipeline._require_gate(primary_root, "original_canary")
smoke_gate = protocol.read_json(primary_root / "smoke" / "gate.json")
if smoke_gate.get("status") != "failed" or smoke_gate.get("failed_targets") != ["yawl"]:
    raise RuntimeError("Expected the preserved yawl Single smoke failure")

diagnostic_summary = protocol.read_json(diagnostic_summary_path)
if diagnostic_summary.get("status") != "complete":
    raise RuntimeError("Completed yawl/volleyball target-only diagnostic is required")
if {row.get("target") for row in diagnostic_summary.get("results", [])} != reused_targets:
    raise RuntimeError("Diagnostic summary must contain exactly yawl and volleyball")

all_rows, rows_by_class = pipeline.load_dataset(config)
if len(all_rows) != 12500:
    raise RuntimeError("The matched evaluation dataset must contain 12,500 rows")

reference, archive_root = pipeline._legacy_reference(config)
legacy_originals = pipeline._load_legacy_original_results(config, archive_root)
checkpoint_lookup = pipeline._checkpoint_lookup(config, anchors)
target_specs: dict[str, dict[str, Any]] = {}
ordered_targets: list[str] = []
for group in config["groups"]:
    for target in group["targets"]:
        ordered_targets.append(target)
        spec = checkpoint_lookup[(group["id"], "single", target)]
        pipeline._validate_checkpoint(spec, config)
        target_specs[target] = spec
if len(ordered_targets) != 10 or len(set(ordered_targets)) != 10:
    raise RuntimeError("Expected exactly ten unique Confuse5 targets")


def hash_scheduler_config(value: Mapping[str, Any]) -> str:
    return protocol.fingerprint({"config": dict(value)})


def reconstruct_metadata_order_variants(
    actual: Mapping[str, Any], wanted_hashes: set[str]
) -> tuple[dict[str, dict[str, Any]], int, list[str]]:
    varying_keys = [
        key
        for key, value in actual.items()
        if key.startswith("_") and isinstance(value, list) and len(value) > 1
    ]
    permutation_lists: list[list[tuple[Any, ...]]] = []
    total = 1
    for key in varying_keys:
        values = list(actual[key])
        permutations = list(dict.fromkeys(itertools.permutations(values)))
        permutation_lists.append(permutations)
        total *= len(permutations)
    if total > 100000:
        raise RuntimeError(
            f"Scheduler metadata reconstruction would require {total} variants"
        )
    found: dict[str, dict[str, Any]] = {}
    combinations = itertools.product(*permutation_lists) if permutation_lists else [()]
    checked = 0
    for combination in combinations:
        candidate = copy.deepcopy(dict(actual))
        for key, permutation in zip(varying_keys, combination):
            candidate[key] = list(permutation)
        digest = hash_scheduler_config(candidate)
        checked += 1
        if digest in wanted_hashes and digest not in found:
            found[digest] = candidate
        if set(found) == wanted_hashes:
            break
    return found, checked, varying_keys


def json_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            differences.append({
                "field": key,
                "smoke_value": left.get(key),
                "diagnostic_value": right.get(key),
                "sampling_relevant": not key.startswith("_"),
            })
    return differences


def runtime_projection(runtime: Mapping[str, Any]) -> dict[str, Any]:
    runtime_details = runtime.get("runtime", {})
    return {
        "requested_base_model": runtime.get("requested_base_model"),
        "requested_dtype": runtime.get("requested_dtype"),
        "pipeline": runtime.get("pipeline"),
        "unet": runtime.get("unet"),
        "text_encoder": runtime.get("text_encoder"),
        "tokenizer": runtime.get("tokenizer"),
        "scheduler": runtime.get("scheduler"),
        "settings": runtime.get("settings"),
        "python": runtime_details.get("python"),
        "executable": runtime_details.get("executable"),
        "platform": runtime_details.get("platform"),
        "packages": runtime_details.get("packages"),
    }


# Loading the pipeline is permitted here, but no image may be generated until this
# provenance audit reaches a passed state.
generation = pipeline.GenerationRuntime(config)
smoke_runtime = smoke_gate.get("generation_runtime", {})
diagnostic_runtime = diagnostic_summary.get("generation_runtime", {})
smoke_hash = smoke_runtime.get("scheduler_config_sha256")
diagnostic_hash = diagnostic_runtime.get("scheduler_config_sha256")
if not isinstance(smoke_hash, str) or not isinstance(diagnostic_hash, str):
    raise RuntimeError("Historical scheduler hashes are missing")

actual_scheduler_config = json.loads(
    json.dumps(dict(generation.pipe.scheduler.config), ensure_ascii=False)
)
wanted_hashes = {smoke_hash, diagnostic_hash}
reconstructed, variants_checked, varying_keys = reconstruct_metadata_order_variants(
    actual_scheduler_config, wanted_hashes
)
runtime_equal = runtime_projection(smoke_runtime) == runtime_projection(diagnostic_runtime)

repo_root = here.parents[1]
smoke_git = smoke_runtime.get("runtime", {}).get("git_hash")
diagnostic_git = diagnostic_runtime.get("runtime", {}).get("git_hash")
historical_changed_files: list[str] = []
git_diff_error: str | None = None
if isinstance(smoke_git, str) and isinstance(diagnostic_git, str):
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{smoke_git}..{diagnostic_git}"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        historical_changed_files = [
            line for line in completed.stdout.splitlines() if line.strip()
        ]
    else:
        git_diff_error = completed.stderr.strip() or completed.stdout.strip()

expected_historical_change = (
    "orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/"
    "run_yawl_volleyball_target_diagnostic.sh"
)
git_history_safe = (
    git_diff_error is None
    and historical_changed_files == [expected_historical_change]
)
hashes_reconstructed = set(reconstructed) == wanted_hashes

differences: list[dict[str, Any]] = []
if hashes_reconstructed:
    differences = json_differences(
        reconstructed[smoke_hash], reconstructed[diagnostic_hash]
    )
sampling_differences = [row for row in differences if row["sampling_relevant"]]
audit_passed = (
    runtime_equal
    and git_history_safe
    and hashes_reconstructed
    and not sampling_differences
)

audit = {
    "schema_version": 1,
    "status": "passed" if audit_passed else "failed",
    "generation_started": False,
    "smoke_scheduler_config_sha256": smoke_hash,
    "diagnostic_scheduler_config_sha256": diagnostic_hash,
    "current_scheduler_config_sha256": hash_scheduler_config(actual_scheduler_config),
    "historical_hashes_reconstructed_from_current_actual_config": hashes_reconstructed,
    "reconstruction_method": (
        "permute only list-valued underscore-prefixed scheduler metadata fields; "
        "never alter public sampling fields"
    ),
    "metadata_list_fields_permuted": varying_keys,
    "variants_checked": variants_checked,
    "actual_current_scheduler_config": actual_scheduler_config,
    "smoke_reconstructed_scheduler_config": reconstructed.get(smoke_hash),
    "diagnostic_reconstructed_scheduler_config": reconstructed.get(diagnostic_hash),
    "field_differences": differences,
    "sampling_relevant_field_differences": sampling_differences,
    "stored_runtime_sampling_context_equal": runtime_equal,
    "historical_git_range": (
        f"{smoke_git}..{diagnostic_git}" if smoke_git and diagnostic_git else None
    ),
    "historical_changed_files": historical_changed_files,
    "git_diff_error": git_diff_error,
    "only_expected_diagnostic_script_changed": git_history_safe,
    "verdict": (
        "sampling-relevant fields identical; historical hash difference is "
        "metadata list ordering only"
        if audit_passed
        else "audit could not prove a metadata-only scheduler hash difference"
    ),
    "completed_at": protocol.utc_now(),
}
protocol.write_json_atomic(audit_path, audit)
if not audit_passed:
    raise RuntimeError(
        "SCHEDULER_PROVENANCE_AUDIT_FAILURE: no new images were generated; "
        f"inspect {audit_path}"
    )
print("[scheduler audit] passed: sampling fields identical; metadata ordering only", flush=True)

qualification_root.mkdir(parents=True, exist_ok=True)
protocol.write_json_atomic(qualification_root / "resolved_plan.json", {
    "schema_version": 1,
    "status": "resolved",
    "purpose": "Qualify all ten Single targets before any further Joint evaluation",
    "ordered_targets": ordered_targets,
    "reused_targets": sorted(reused_targets),
    "new_targets": [target for target in ordered_targets if target not in reused_targets],
    "rows_per_target": 500,
    "new_single_images": 4000,
    "new_joint_images": 0,
    "new_preservation_images": 0,
    "new_original_images": 0,
    "formal_pipeline_enabled": False,
    "scheduler_audit_path": str(audit_path.resolve()),
    "legacy_original_reference": reference,
    "qualification_rule": {
        "source": "existing configured smoke_gate.required_accuracy_drop",
        "minimum_accuracy_drop": config["smoke_gate"]["required_accuracy_drop"],
        "minimum_drop_count_at_500": math.ceil(
            config["smoke_gate"]["required_accuracy_drop"] * 500
        ),
        "usable_original_minimum_correct": math.ceil(
            config["smoke_gate"]["required_accuracy_drop"] * 500
        ),
        "rationale": (
            "Original must have enough exact-top1 headroom to permit the existing "
            "12.5 percentage-point erasure criterion."
        ),
    },
    "checkpoint_modification_forbidden": True,
    "source_hashes": protocol.source_hashes([here / "run_single_target_qualification.sh"]),
    "created_at": protocol.utc_now(),
})

evaluator = pipeline.EvaluationRuntime(config)
single_results: dict[str, dict[str, Any]] = {}

for target in ordered_targets:
    group = pipeline._group_for_target(config, target)
    if target in reused_targets:
        job_id = f"single__{group['id']}__{protocol.slug(target)}__target-only-500"
        result_path = diagnostic_root / "evaluations" / f"{job_id}.json"
        result = protocol.read_json(result_path)
        if result.get("status") != "complete" or result.get("total") != 500:
            raise RuntimeError(f"Invalid reusable diagnostic result: {result_path}")
        if result.get("checkpoint_sha256") != protocol.sha256(
            Path(target_specs[target]["checkpoint_path"])
        ):
            raise RuntimeError(f"Reusable checkpoint hash mismatch for {target}")
        single_results[target] = result
        print(f"[qualification reuse] {target}: 500 Single rows", flush=True)
        continue

    rows = rows_by_class[protocol.normalize(target)]
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 ordered rows for {target}, found {len(rows)}")
    spec = target_specs[target]
    identity = pipeline._job_identity(
        "single", group, target, target, spec, rows, config
    )
    job_id = f"single__{group['id']}__{protocol.slug(target)}__qualification-500"
    job = {
        **identity,
        "job_id": job_id,
        "job_fingerprint": protocol.fingerprint(identity),
        "checkpoint_spec": spec,
        "rows": rows,
        "image_dir": str(qualification_root / "images" / protocol.slug(target)),
        "manifest_path": str(qualification_root / "manifests" / f"{job_id}.json"),
        "result_path": str(qualification_root / "evaluations" / f"{job_id}.json"),
    }
    print(f"[qualification] generating/evaluating {target}: 500 Single rows", flush=True)
    single_results[target] = pipeline._run_formal_job(
        job,
        generation,
        evaluator,
        skip_existing=True,
        purge=True,
    )


def summarize_target(target: str) -> dict[str, Any]:
    group = pipeline._group_for_target(config, target)
    original = legacy_originals[(group["id"], protocol.normalize(target))]
    single = single_results[target]
    original_by_case = {int(item["case_number"]): item for item in original["items"]}
    single_by_case = {int(item["case_number"]): item for item in single["items"]}
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

    original_correct = int(original["correct"])
    single_correct = int(single["correct"])
    if sum(transitions.values()) != 500:
        raise RuntimeError(f"Transition count does not sum to 500 for {target}")
    if (
        transitions["original_correct_to_single_wrong"]
        + transitions["original_correct_to_single_correct"]
        != original_correct
    ):
        raise RuntimeError(f"Original-correct transition mismatch for {target}")

    threshold = math.ceil(config["smoke_gate"]["required_accuracy_drop"] * 500)
    usable_original = original_correct >= threshold
    clear_net_erasure = original_correct - single_correct >= threshold
    reasons: list[str] = []
    if not usable_original:
        reasons.append("original_below_12_5pp_headroom_floor")
    if not clear_net_erasure:
        reasons.append("single_net_accuracy_drop_below_12_5pp")
    return {
        "group_id": group["id"],
        "target": target,
        "total": 500,
        "original_correct": original_correct,
        "single_correct": single_correct,
        "original_accuracy": original_correct / 500,
        "single_accuracy": single_correct / 500,
        "single_minus_original_accuracy": (single_correct - original_correct) / 500,
        "matched_transitions": transitions,
        "conditional_erasure_rate_among_original_correct": (
            transitions["original_correct_to_single_wrong"] / original_correct
            if original_correct
            else None
        ),
        "usable_original_classifier_baseline": usable_original,
        "clear_net_target_erasure": clear_net_erasure,
        "target_eligible": usable_original and clear_net_erasure,
        "ineligibility_reasons": reasons,
        "result_origin": (
            "reused_target_only_diagnostic_v1"
            if target in reused_targets
            else "single_target_qualification_v1"
        ),
        "legacy_original_auxiliary_metrics": "unavailable",
        "single_mean_target_probability": single["mean_target_probability"],
        "single_mean_raw_target_logit": single["mean_raw_target_logit"],
        "single_checkpoint_sha256": single["checkpoint_sha256"],
    }


target_rows = [summarize_target(target) for target in ordered_targets]
target_lookup = {row["target"]: row for row in target_rows}
group_rows: list[dict[str, Any]] = []
for group in config["groups"]:
    members = [target_lookup[target] for target in group["targets"]]
    eligible = all(row["target_eligible"] for row in members)
    reasons = [
        {"target": row["target"], "reasons": row["ineligibility_reasons"]}
        for row in members
        if not row["target_eligible"]
    ]
    group_rows.append({
        "group_id": group["id"],
        "targets": list(group["targets"]),
        "status": (
            "eligible_for_primary_single_vs_joint"
            if eligible
            else "ineligible_for_primary_single_vs_joint"
        ),
        "eligible": eligible,
        "reasons": reasons,
        "joint_results_used_for_inclusion": False,
    })

report = {
    "schema_version": 1,
    "status": "complete",
    "formal_pipeline_enabled": False,
    "scheduler_audit_status": "passed",
    "scheduler_audit_path": str(audit_path.resolve()),
    "new_single_images_generated": 4000,
    "reused_single_images_evaluated": 1000,
    "new_joint_images_generated": 0,
    "new_preservation_images_generated": 0,
    "new_original_images_generated": 0,
    "qualification_rule": {
        "source": "existing smoke threshold, not a new tuned setting",
        "minimum_original_correct_of_500": math.ceil(
            config["smoke_gate"]["required_accuracy_drop"] * 500
        ),
        "minimum_original_minus_single_correct_of_500": math.ceil(
            config["smoke_gate"]["required_accuracy_drop"] * 500
        ),
        "minimum_accuracy_drop": config["smoke_gate"]["required_accuracy_drop"],
    },
    "targets": target_rows,
    "groups": group_rows,
    "eligible_groups": [row["group_id"] for row in group_rows if row["eligible"]],
    "ineligible_groups": [row["group_id"] for row in group_rows if not row["eligible"]],
    "legacy_original_probability_logit_top5": "unavailable_not_regenerated",
    "generation_runtime": generation.identity,
    "evaluator": evaluator.identity,
    "completed_at": protocol.utc_now(),
}
protocol.write_json_atomic(summary_path, report)
print(json.dumps(report, indent=2, ensure_ascii=False))
PY
}

case "${1:-}" in
    --worker)
        require_mu
        run_worker
        ;;
    --detached-worker)
        require_mu
        set +e
        run_worker
        worker_status="$?"
        printf '%s\n' "$worker_status" > "$EXIT_FILE"
        exit "$worker_status"
        ;;
    --foreground)
        require_mu
        run_worker
        ;;
    --status)
        show_status
        ;;
    "")
        require_mu
        mkdir -p "$STATE_ROOT"
        if [[ -f "$PID_FILE" ]]; then
            existing_pid="$(tr -d '[:space:]' < "$PID_FILE")"
            if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
                echo "Qualification is already running with PID $existing_pid"
                show_status
                exit 0
            fi
        fi
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        log_path="$STATE_ROOT/detached_${stamp}.log"
        rm -f "$EXIT_FILE"
        printf '%s\n' "$log_path" > "$LATEST_FILE"
        nohup "$SCRIPT_PATH" --detached-worker > "$log_path" 2>&1 < /dev/null &
        child_pid="$!"
        printf '%s\n' "$child_pid" > "$PID_FILE"
        echo "Started detached Single-target qualification."
        echo "PID: $child_pid"
        echo "Log: $log_path"
        echo "You may close SSH. Check later with:"
        echo "./experiments/confuse5_single_vs_joint/run_single_target_qualification.sh --status"
        ;;
    *)
        echo "Usage: $0 [--status|--foreground]" >&2
        exit 2
        ;;
esac
