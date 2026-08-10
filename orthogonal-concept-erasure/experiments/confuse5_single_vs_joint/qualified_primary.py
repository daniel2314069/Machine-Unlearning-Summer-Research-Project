#!/usr/bin/env python3
"""Run and aggregate the pre-qualified 3/5-group Confuse5 primary subset."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline
import protocol


SCOPE_LABEL = (
    "Confuse5 baseline-qualified primary subset "
    "(3/5 groups, qualification completed before Joint evaluation)"
)
RESEARCH_QUESTION = (
    "Given targets for which OCE Single erasure is demonstrably effective, "
    "does jointly erasing two semantically similar targets with OCE's "
    "multi-concept subspace introduce additional target-erasure regression "
    "and/or collateral damage to their shared similar non-target neighborhood?"
)
ELIGIBLE_SCOPE = {
    "dogs": ["golden retriever", "labrador retriever"],
    "fruits": ["orange", "lemon"],
    "balls": ["soccer ball", "volleyball"],
}
EXCLUDED_SCOPE = {
    "cats": ["tabby", "tiger cat"],
    "boats": ["yawl", "lifeboat"],
}
EXPECTED_JOB_COUNT = 45
EXPECTED_SINGLE_IMAGES = 15000
EXPECTED_JOINT_IMAGES = 7500
EXPECTED_EDITED_IMAGES = 22500


def mean(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("Cannot compute an empty mean")
    return sum(values) / len(values)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def runtime_projection(runtime: Mapping[str, Any]) -> dict[str, Any]:
    details = runtime.get("runtime", {})
    return {
        "requested_base_model": runtime.get("requested_base_model"),
        "requested_dtype": runtime.get("requested_dtype"),
        "pipeline": runtime.get("pipeline"),
        "unet": runtime.get("unet"),
        "text_encoder": runtime.get("text_encoder"),
        "tokenizer": runtime.get("tokenizer"),
        "scheduler": runtime.get("scheduler"),
        "settings": runtime.get("settings"),
        "python": details.get("python"),
        "executable": details.get("executable"),
        "platform": details.get("platform"),
        "packages": details.get("packages"),
    }


def sampling_scheduler_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def load_and_validate_qualification(
    primary_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    path = primary_root / "single_target_qualification_v1" / "summary.json"
    qualification = protocol.read_json(path)
    if (
        qualification.get("status") != "complete"
        or qualification.get("formal_pipeline_enabled") is not False
        or qualification.get("scheduler_audit_status") != "passed"
        or qualification.get("new_joint_images_generated") != 0
        or qualification.get("new_preservation_images_generated") != 0
    ):
        raise RuntimeError("Baseline qualification is incomplete or contaminated")
    if qualification.get("eligible_groups") != list(ELIGIBLE_SCOPE):
        raise RuntimeError("Qualification eligible groups differ from frozen scope")
    if qualification.get("ineligible_groups") != list(EXCLUDED_SCOPE):
        raise RuntimeError("Qualification excluded groups differ from frozen scope")

    config_groups = {row["id"]: list(row["targets"]) for row in config["groups"]}
    if any(config_groups.get(key) != value for key, value in ELIGIBLE_SCOPE.items()):
        raise RuntimeError("Frozen eligible targets differ from resolved config")
    if any(config_groups.get(key) != value for key, value in EXCLUDED_SCOPE.items()):
        raise RuntimeError("Frozen excluded targets differ from resolved config")

    qualification_groups = {
        row["group_id"]: row for row in qualification.get("groups", [])
    }
    for group_id in ELIGIBLE_SCOPE:
        row = qualification_groups.get(group_id, {})
        if (
            row.get("eligible") is not True
            or row.get("joint_results_used_for_inclusion") is not False
        ):
            raise RuntimeError(f"Invalid eligible qualification record: {group_id}")
    for group_id in EXCLUDED_SCOPE:
        row = qualification_groups.get(group_id, {})
        if (
            row.get("eligible") is not False
            or row.get("joint_results_used_for_inclusion") is not False
        ):
            raise RuntimeError(f"Invalid excluded qualification record: {group_id}")

    target_rows = {
        protocol.normalize(row["target"]): row
        for row in qualification.get("targets", [])
    }
    if len(target_rows) != 10:
        raise RuntimeError("Qualification must retain all ten target histories")
    for targets in ELIGIBLE_SCOPE.values():
        for target in targets:
            if target_rows[protocol.normalize(target)].get("target_eligible") is not True:
                raise RuntimeError(f"Frozen eligible target did not qualify: {target}")
    if target_rows["tiger cat"].get("original_correct") != 0:
        raise RuntimeError("Frozen cats exclusion no longer matches tiger cat floor")
    yawl = target_rows["yawl"]
    if yawl.get("original_correct") != 178 or yawl.get("single_correct") != 181:
        raise RuntimeError("Frozen boats exclusion no longer matches yawl failure")
    return qualification, path


def eligible_groups(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = [row for row in config["groups"] if row["id"] in ELIGIBLE_SCOPE]
    if [row["id"] for row in groups] != list(ELIGIBLE_SCOPE):
        raise RuntimeError("Eligible group ordering differs from frozen scope")
    return groups


def build_jobs(
    config: Mapping[str, Any],
    anchors: Mapping[str, str],
    root: Path,
    qualification_hash: str,
) -> list[dict[str, Any]]:
    _, rows_by_class = pipeline.load_dataset(config)
    checkpoints = pipeline._checkpoint_lookup(config, anchors)
    jobs: list[dict[str, Any]] = []
    for group in eligible_groups(config):
        for target in group["targets"]:
            spec = checkpoints[(group["id"], "single", target)]
            pipeline._validate_checkpoint(spec, config)
            for concept in group["concepts"]:
                rows = rows_by_class[protocol.normalize(concept)]
                identity = {
                    **pipeline._job_identity(
                        "single", group, concept, target, spec, rows, config
                    ),
                    "primary_scope": SCOPE_LABEL,
                    "qualification_summary_sha256": qualification_hash,
                }
                job_id = (
                    f"single__{group['id']}__{protocol.slug(target)}__"
                    f"{protocol.slug(concept)}"
                )
                jobs.append({
                    **identity,
                    "job_id": job_id,
                    "job_fingerprint": protocol.fingerprint(identity),
                    "checkpoint_spec": spec,
                    "rows": rows,
                    "image_dir": str(
                        root / "formal" / "images" / "single" / group["id"]
                        / protocol.slug(target) / protocol.slug(concept)
                    ),
                    "manifest_path": str(
                        root / "formal" / "manifests" / f"{job_id}.json"
                    ),
                    "result_path": str(
                        root / "formal" / "evaluations" / "shards" / f"{job_id}.json"
                    ),
                })
        spec = checkpoints[(group["id"], "joint", None)]
        pipeline._validate_checkpoint(spec, config)
        for concept in group["concepts"]:
            rows = rows_by_class[protocol.normalize(concept)]
            identity = {
                **pipeline._job_identity(
                    "joint", group, concept, None, spec, rows, config
                ),
                "primary_scope": SCOPE_LABEL,
                "qualification_summary_sha256": qualification_hash,
            }
            job_id = f"joint__{group['id']}__{protocol.slug(concept)}"
            jobs.append({
                **identity,
                "job_id": job_id,
                "job_fingerprint": protocol.fingerprint(identity),
                "checkpoint_spec": spec,
                "rows": rows,
                "image_dir": str(
                    root / "formal" / "images" / "joint" / group["id"]
                    / protocol.slug(concept)
                ),
                "manifest_path": str(
                    root / "formal" / "manifests" / f"{job_id}.json"
                ),
                "result_path": str(
                    root / "formal" / "evaluations" / "shards" / f"{job_id}.json"
                ),
            })

    single_images = sum(
        len(job["rows"]) for job in jobs if job["model_type"] == "single"
    )
    joint_images = sum(
        len(job["rows"]) for job in jobs if job["model_type"] == "joint"
    )
    if (
        len(jobs) != EXPECTED_JOB_COUNT
        or single_images != EXPECTED_SINGLE_IMAGES
        or joint_images != EXPECTED_JOINT_IMAGES
        or single_images + joint_images != EXPECTED_EDITED_IMAGES
    ):
        raise RuntimeError(
            "Qualified formal plan count mismatch: "
            f"jobs={len(jobs)}, single={single_images}, joint={joint_images}"
        )
    if any(job["group_id"] in EXCLUDED_SCOPE for job in jobs):
        raise RuntimeError("Excluded group leaked into qualified formal jobs")
    return jobs


def validate_scheduler_for_formal(
    generation: pipeline.GenerationRuntime,
    qualification: Mapping[str, Any],
    audit: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    actual = json.loads(
        json.dumps(dict(generation.pipe.scheduler.config), ensure_ascii=False)
    )
    prior_actual = audit.get("actual_current_scheduler_config")
    runtime_equal = runtime_projection(generation.identity) == runtime_projection(
        qualification.get("generation_runtime", {})
    )
    sampling_equal = (
        isinstance(prior_actual, dict)
        and sampling_scheduler_config(actual)
        == sampling_scheduler_config(prior_actual)
    )
    clean_git = generation.identity.get("runtime", {}).get("git_dirty") is False
    passed = (
        audit.get("status") == "passed"
        and runtime_equal
        and sampling_equal
        and clean_git
    )
    record = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "generation_started": False,
        "qualification_scheduler_audit_sha256": protocol.sha256(
            Path(qualification["scheduler_audit_path"])
        ),
        "qualification_runtime_context_equal": runtime_equal,
        "sampling_relevant_scheduler_fields_equal": sampling_equal,
        "git_clean": clean_git,
        "prior_sampling_scheduler_config": (
            sampling_scheduler_config(prior_actual)
            if isinstance(prior_actual, dict)
            else None
        ),
        "current_sampling_scheduler_config": sampling_scheduler_config(actual),
        "current_scheduler_config_sha256": protocol.fingerprint({"config": actual}),
        "metadata_fields_excluded_from_sampling_comparison": sorted(
            key for key in actual if key.startswith("_")
        ),
        "generation_runtime": generation.identity,
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(output_path, record)
    if not passed:
        raise RuntimeError(
            "QUALIFIED_FORMAL_INTEGRITY_FAILURE: scheduler/runtime provenance "
            f"mismatch; no new images generated; inspect {output_path}"
        )
    return record


def validate_result(job: Mapping[str, Any]) -> dict[str, Any]:
    result = protocol.read_json(Path(job["result_path"]))
    if (
        result.get("status") != "complete"
        or result.get("job_fingerprint") != job["job_fingerprint"]
        or result.get("total") != 500
        or len(result.get("items", [])) != 500
    ):
        raise RuntimeError(f"Invalid completed result: {job['result_path']}")
    return result


def validate_single_target_reproduction(
    result: Mapping[str, Any],
    target: str,
    group_id: str,
    qualification: Mapping[str, Any],
    primary_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    qualification_row = next(
        row
        for row in qualification["targets"]
        if protocol.normalize(row["target"]) == protocol.normalize(target)
    )
    if qualification_row["result_origin"] == "reused_target_only_diagnostic_v1":
        prior_path = (
            primary_root / "target_only_diagnostic_v1" / "evaluations"
            / f"single__{group_id}__{protocol.slug(target)}__target-only-500.json"
        )
    elif qualification_row["result_origin"] == "single_target_qualification_v1":
        prior_path = (
            primary_root / "single_target_qualification_v1" / "evaluations"
            / f"single__{group_id}__{protocol.slug(target)}__qualification-500.json"
        )
    else:
        raise RuntimeError(f"Unknown qualification result origin for {target}")
    prior = protocol.read_json(prior_path)
    prior_items = {int(row["case_number"]): row for row in prior["items"]}
    current_items = {int(row["case_number"]): row for row in result["items"]}
    mismatches: list[dict[str, Any]] = []
    for case in sorted(set(prior_items) | set(current_items)):
        previous = prior_items.get(case, {})
        current = current_items.get(case, {})
        differing = [
            field
            for field in (
                "prompt",
                "evaluation_seed",
                "image_sha256",
                "predicted_index",
                "predicted_category",
                "correct",
            )
            if previous.get(field) != current.get(field)
        ]
        if differing:
            mismatches.append({"case_number": case, "differing_fields": differing})
            if len(mismatches) >= 20:
                break
    passed = (
        prior.get("status") == "complete"
        and prior.get("total") == 500
        and result.get("total") == 500
        and prior.get("checkpoint_sha256") == result.get("checkpoint_sha256")
        and len(prior_items) == 500
        and len(current_items) == 500
        and not mismatches
    )
    record = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "target": target,
        "group_id": group_id,
        "prior_result_path": str(prior_path.resolve()),
        "prior_result_sha256": protocol.sha256(prior_path),
        "formal_result_sha256": protocol.sha256(Path(result["result_path"]))
        if isinstance(result.get("result_path"), str)
        and Path(result["result_path"]).is_file()
        else None,
        "checked_images": 500,
        "checkpoint_sha256_equal": (
            prior.get("checkpoint_sha256") == result.get("checkpoint_sha256")
        ),
        "mismatch_count_capped_at_20": len(mismatches),
        "mismatch_examples": mismatches,
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(output_path, record)
    if not passed:
        raise RuntimeError(
            f"SINGLE_TARGET_REPRODUCTION_FAILURE: {target}; inspect {output_path}"
        )
    return record


def item_seed(item: Mapping[str, Any]) -> int:
    value = item.get("evaluation_seed", item.get("seed"))
    return int(value)


def matched_transitions(
    original: Mapping[str, Any],
    single: Mapping[str, Any],
    joint: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    original_by_case = {int(row["case_number"]): row for row in original["items"]}
    single_by_case = {int(row["case_number"]): row for row in single["items"]}
    joint_by_case = {int(row["case_number"]): row for row in joint["items"]}
    cases = sorted(original_by_case)
    if (
        len(cases) != 500
        or set(cases) != set(single_by_case)
        or set(cases) != set(joint_by_case)
    ):
        raise RuntimeError(f"Matched case mismatch: {context}")
    counts = {
        "single_correct_to_joint_wrong": 0,
        "single_correct_to_joint_correct": 0,
        "single_wrong_to_joint_correct": 0,
        "single_wrong_to_joint_wrong": 0,
    }
    rows: list[dict[str, Any]] = []
    for case in cases:
        o = original_by_case[case]
        s = single_by_case[case]
        j = joint_by_case[case]
        if (
            o.get("prompt") != s.get("prompt")
            or s.get("prompt") != j.get("prompt")
            or item_seed(o) != item_seed(s)
            or item_seed(s) != item_seed(j)
        ):
            raise RuntimeError(f"Matched prompt/seed mismatch: {context}, case={case}")
        single_correct = bool(s["correct"])
        joint_correct = bool(j["correct"])
        if single_correct and not joint_correct:
            transition = "single_correct_to_joint_wrong"
        elif single_correct and joint_correct:
            transition = "single_correct_to_joint_correct"
        elif not single_correct and joint_correct:
            transition = "single_wrong_to_joint_correct"
        else:
            transition = "single_wrong_to_joint_wrong"
        counts[transition] += 1
        rows.append({
            **context,
            "case_number": case,
            "prompt": s["prompt"],
            "seed": item_seed(s),
            "original_correct": bool(o["correct"]),
            "single_correct": single_correct,
            "joint_correct": joint_correct,
            "single_to_joint_transition": transition,
            "original_predicted_category": o.get("predicted_category"),
            "single_predicted_category": s.get("predicted_category"),
            "joint_predicted_category": j.get("predicted_category"),
            "single_target_probability": s.get("target_probability"),
            "joint_target_probability": j.get("target_probability"),
            "single_raw_target_logit": s.get("raw_target_logit"),
            "joint_raw_target_logit": j.get("raw_target_logit"),
            "single_image_sha256": s.get("image_sha256"),
            "joint_image_sha256": j.get("image_sha256"),
        })
    if sum(counts.values()) != 500:
        raise RuntimeError(f"Transition count mismatch: {context}")
    return counts, rows


def excluded_history(qualification: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = {
        protocol.normalize(row["target"]): row
        for row in qualification["targets"]
    }
    return [
        {
            "group_id": "cats",
            "targets": ["tabby", "tiger cat"],
            "status": "ineligible_for_primary_single_vs_joint",
            "decided_before_primary_joint_evaluation": True,
            "joint_results_used_for_inclusion": False,
            "reason": "tiger cat Original ResNet exact-top1 classifier floor",
            "evidence_target": "tiger cat",
            "original_correct": targets["tiger cat"]["original_correct"],
            "single_correct": targets["tiger cat"]["single_correct"],
            "total": 500,
        },
        {
            "group_id": "boats",
            "targets": ["yawl", "lifeboat"],
            "status": "ineligible_for_primary_single_vs_joint",
            "decided_before_primary_joint_evaluation": True,
            "joint_results_used_for_inclusion": False,
            "reason": "yawl failed fixed Single-erasure prerequisite",
            "evidence_target": "yawl",
            "original_correct": targets["yawl"]["original_correct"],
            "single_correct": targets["yawl"]["single_correct"],
            "total": 500,
        },
    ]


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {summary['scope_label']}",
        "",
        summary["research_question"],
        "",
        "## Scope and provenance",
        "",
        "Eligibility was frozen from Original + Single results before this primary "
        "Joint/preservation run. Joint results were not used for group inclusion. "
        "This is not a full five-group Confuse5 experiment.",
        "",
        f"New edited images: {summary['formal_new_edited_images']:,} "
        f"({summary['formal_single_images']:,} Single; "
        f"{summary['formal_joint_images']:,} Joint). Original images regenerated: 0.",
        "",
        "## Target erasure",
        "",
        "| Group | Target | Original | Single | Joint | Joint - Single | S→J transitions¹ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["target_rows"]:
        transitions = row["single_to_joint_transitions"]
        compact = "/".join(str(transitions[key]) for key in (
            "single_correct_to_joint_wrong",
            "single_correct_to_joint_correct",
            "single_wrong_to_joint_correct",
            "single_wrong_to_joint_wrong",
        ))
        lines.append(
            f"| {row['group_id']} | {row['target']} | "
            f"{row['original_target_accuracy']:.3f} | "
            f"{row['single_target_accuracy']:.3f} | "
            f"{row['joint_target_accuracy']:.3f} | "
            f"{row['joint_minus_single_target_accuracy']:+.3f} | {compact} |"
        )
    lines.extend([
        "",
        "¹ Single-correct→Joint-wrong / Single-correct→Joint-correct / "
        "Single-wrong→Joint-correct / Single-wrong→Joint-wrong.",
        "",
        "Positive Joint−Single target accuracy means additional Joint erasure failure.",
        "",
        "## Similar non-target preservation",
        "",
        "| Group | Single target | Preservation concept | Original | Single | Joint | Joint - Single |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in summary["preservation_rows"]:
        lines.append(
            f"| {row['group_id']} | {row['single_target']} | "
            f"{row['preservation_concept']} | {row['original_accuracy']:.3f} | "
            f"{row['single_preservation_accuracy']:.3f} | "
            f"{row['joint_preservation_accuracy']:.3f} | "
            f"{row['joint_minus_single_preservation_accuracy']:+.3f} |"
        )
    lines.extend([
        "",
        "Negative Joint−Single preservation accuracy means additional Joint collateral damage.",
        "",
        "## Group-level descriptive macros",
        "",
        "| Group | Target J−S | Preservation J−S | Target units | Preservation units |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary["group_rows"]:
        lines.append(
            f"| {row['group_id']} | "
            f"{row['target_joint_minus_single_macro']:+.3f} | "
            f"{row['preservation_joint_minus_single_macro']:+.3f} | "
            f"{row['target_unit_count']} | {row['preservation_comparison_unit_count']} |"
        )
    lines.extend([
        "",
        "These are descriptive concept-level macros. Images are matched observations "
        "within a target/class, not independent concept-level experimental units.",
        "",
        "## Pre-Joint baseline qualification exclusions",
        "",
        "| Group | Targets | Reason | Evidence |",
        "|---|---|---|---|",
    ])
    for row in summary["pre_joint_baseline_qualification_exclusions"]:
        lines.append(
            f"| {row['group_id']} | {', '.join(row['targets'])} | {row['reason']} | "
            f"{row['evidence_target']}: Original {row['original_correct']}/500, "
            f"Single {row['single_correct']}/500 |"
        )
    lines.extend([
        "",
        "Cats and boats remain part of the experiment history but are excluded from "
        "the primary Single-vs-Joint analysis. No rescue, anchor change, or retuning "
        "was performed.",
        "",
    ])
    return "\n".join(lines)


def aggregate(
    config: Mapping[str, Any],
    anchors: Mapping[str, str],
    qualification: Mapping[str, Any],
    qualification_path: Path,
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    generation_identity: Mapping[str, Any],
    evaluator_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _, archive_root = pipeline._legacy_reference(config)
    originals = pipeline._load_legacy_original_results(config, archive_root)
    edited: dict[tuple[str, str, str | None, str], dict[str, Any]] = {}
    per_class: list[dict[str, Any]] = []
    for job in jobs:
        result = validate_result(job)
        key = (
            job["group_id"],
            job["model_type"],
            job["single_target"],
            protocol.normalize(job["evaluated_concept"]),
        )
        if key in edited:
            raise RuntimeError(f"Duplicate edited result key: {key}")
        edited[key] = result
        per_class.append({
            "group_id": job["group_id"],
            "model_type": job["model_type"],
            "single_target": job["single_target"] or "",
            "evaluated_concept": job["evaluated_concept"],
            "correct": result["correct"],
            "total": result["total"],
            "accuracy": result["accuracy"],
            "mean_target_probability": result["mean_target_probability"],
            "mean_raw_target_logit": result["mean_raw_target_logit"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        })
    if len(edited) != EXPECTED_JOB_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_JOB_COUNT} edited shards, found {len(edited)}")

    target_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    sibling_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []

    for group in eligible_groups(config):
        group_id = group["id"]
        for target in group["targets"]:
            target_norm = protocol.normalize(target)
            original = originals[(group_id, target_norm)]
            single = edited[(group_id, "single", target, target_norm)]
            joint = edited[(group_id, "joint", None, target_norm)]
            transitions, rows = matched_transitions(
                original,
                single,
                joint,
                {
                    "analysis_role": "target_erasure",
                    "group_id": group_id,
                    "single_target": target,
                    "evaluated_concept": target,
                },
            )
            per_image_rows.extend(rows)
            target_rows.append({
                "group_id": group_id,
                "target": target,
                "anchor": anchors[target],
                "original_target_correct": original["correct"],
                "single_target_correct": single["correct"],
                "joint_target_correct": joint["correct"],
                "total": 500,
                "original_target_accuracy": original["accuracy"],
                "single_target_accuracy": single["accuracy"],
                "joint_target_accuracy": joint["accuracy"],
                "joint_minus_single_target_accuracy": (
                    joint["accuracy"] - single["accuracy"]
                ),
                "single_to_joint_transitions": transitions,
                "single_mean_target_probability": single["mean_target_probability"],
                "joint_mean_target_probability": joint["mean_target_probability"],
                "single_mean_raw_target_logit": single["mean_raw_target_logit"],
                "joint_mean_raw_target_logit": joint["mean_raw_target_logit"],
                "legacy_original_auxiliary_metrics": "unavailable",
            })

            for concept in group["similar_non_targets"]:
                concept_norm = protocol.normalize(concept)
                original_p = originals[(group_id, concept_norm)]
                single_p = edited[(group_id, "single", target, concept_norm)]
                joint_p = edited[(group_id, "joint", None, concept_norm)]
                transitions, rows = matched_transitions(
                    original_p,
                    single_p,
                    joint_p,
                    {
                        "analysis_role": "similar_non_target_preservation",
                        "group_id": group_id,
                        "single_target": target,
                        "evaluated_concept": concept,
                    },
                )
                per_image_rows.extend(rows)
                preservation_rows.append({
                    "group_id": group_id,
                    "single_target": target,
                    "preservation_concept": concept,
                    "original_correct": original_p["correct"],
                    "single_correct": single_p["correct"],
                    "joint_correct": joint_p["correct"],
                    "total": 500,
                    "original_accuracy": original_p["accuracy"],
                    "single_preservation_accuracy": single_p["accuracy"],
                    "joint_preservation_accuracy": joint_p["accuracy"],
                    "joint_minus_single_preservation_accuracy": (
                        joint_p["accuracy"] - single_p["accuracy"]
                    ),
                    "single_to_joint_transitions": transitions,
                    "single_mean_target_probability": single_p["mean_target_probability"],
                    "joint_mean_target_probability": joint_p["mean_target_probability"],
                    "single_mean_raw_target_logit": single_p["mean_raw_target_logit"],
                    "joint_mean_raw_target_logit": joint_p["mean_raw_target_logit"],
                    "legacy_original_auxiliary_metrics": "unavailable",
                })

            sibling = next(
                value
                for value in group["targets"]
                if protocol.normalize(value) != target_norm
            )
            sibling_result = edited[
                (group_id, "single", target, protocol.normalize(sibling))
            ]
            sibling_rows.append({
                "group_id": group_id,
                "single_target": target,
                "sibling_target": sibling,
                "single_sibling_correct": sibling_result["correct"],
                "single_sibling_accuracy": sibling_result["accuracy"],
                "role": "secondary_diagnostic_only",
            })

    if len(target_rows) != 6 or len(preservation_rows) != 18:
        raise RuntimeError("Primary aggregate row count mismatch")
    if len(per_image_rows) != (6 + 18) * 500:
        raise RuntimeError("Matched per-image comparison count mismatch")

    group_rows: list[dict[str, Any]] = []
    for group in eligible_groups(config):
        group_targets = [row for row in target_rows if row["group_id"] == group["id"]]
        group_preservation = [
            row for row in preservation_rows if row["group_id"] == group["id"]
        ]
        unique_joint_preservation = {}
        for row in group_preservation:
            unique_joint_preservation[row["preservation_concept"]] = row[
                "joint_preservation_accuracy"
            ]
        group_rows.append({
            "group_id": group["id"],
            "targets": list(group["targets"]),
            "target_unit_count": 2,
            "target_original_macro_accuracy": mean([
                row["original_target_accuracy"] for row in group_targets
            ]),
            "target_single_macro_accuracy": mean([
                row["single_target_accuracy"] for row in group_targets
            ]),
            "target_joint_macro_accuracy": mean([
                row["joint_target_accuracy"] for row in group_targets
            ]),
            "target_joint_minus_single_macro": mean([
                row["joint_minus_single_target_accuracy"] for row in group_targets
            ]),
            "preservation_comparison_unit_count": 6,
            "preservation_single_macro_accuracy": mean([
                row["single_preservation_accuracy"] for row in group_preservation
            ]),
            "preservation_joint_macro_accuracy": mean([
                row["joint_preservation_accuracy"] for row in group_preservation
            ]),
            "preservation_joint_minus_single_macro": mean([
                row["joint_minus_single_preservation_accuracy"]
                for row in group_preservation
            ]),
            "unique_joint_preservation_class_count": 3,
            "unique_joint_preservation_macro_accuracy": mean(
                list(unique_joint_preservation.values())
            ),
            "inference_note": (
                "descriptive concept-level macro; 500 images per class are matched "
                "observations, not independent concept-level units"
            ),
        })

    overall = {
        "group_unit_count": 3,
        "target_joint_minus_single_group_macro": mean([
            row["target_joint_minus_single_macro"] for row in group_rows
        ]),
        "preservation_joint_minus_single_group_macro": mean([
            row["preservation_joint_minus_single_macro"] for row in group_rows
        ]),
        "inference_note": (
            "descriptive macro across three baseline-qualified groups; no image-level "
            "independence assumption and no claim about excluded groups"
        ),
    }

    aggregate_root = root / "formal" / "aggregates"
    write_csv(aggregate_root / "target_erasure.csv", [
        {**row, "single_to_joint_transitions": json.dumps(
            row["single_to_joint_transitions"], sort_keys=True
        )}
        for row in target_rows
    ])
    write_csv(aggregate_root / "similar_non_target_preservation.csv", [
        {**row, "single_to_joint_transitions": json.dumps(
            row["single_to_joint_transitions"], sort_keys=True
        )}
        for row in preservation_rows
    ])
    write_csv(aggregate_root / "group_summary.csv", group_rows)
    write_csv(aggregate_root / "sibling_target_secondary.csv", sibling_rows)
    write_csv(
        root / "formal" / "evaluations" / "per_class.csv",
        per_class,
    )
    write_csv(
        root / "formal" / "evaluations" / "per_image_single_joint_transitions.csv",
        per_image_rows,
    )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope_label": SCOPE_LABEL,
        "not_full_five_group_confuse5": True,
        "research_question": RESEARCH_QUESTION,
        "primary_behavior": "official released repository behavior",
        "eligibility_source": "Original + Single baseline qualification only",
        "eligibility_completed_before_primary_joint_and_preservation_evaluation": True,
        "joint_results_used_for_group_inclusion": False,
        "qualification_summary_path": str(qualification_path.resolve()),
        "qualification_summary_sha256": protocol.sha256(qualification_path),
        "eligible_groups": list(ELIGIBLE_SCOPE),
        "eligible_targets": [
            target for targets in ELIGIBLE_SCOPE.values() for target in targets
        ],
        "pre_joint_baseline_qualification_exclusions": excluded_history(qualification),
        "formal_original_source": "conditional_reusable_original_reference",
        "formal_original_regenerated_images": 0,
        "formal_original_auxiliary_metrics": (
            "unavailable_for_legacy_original_full_baseline"
        ),
        "formal_single_images": EXPECTED_SINGLE_IMAGES,
        "formal_joint_images": EXPECTED_JOINT_IMAGES,
        "formal_new_edited_images": EXPECTED_EDITED_IMAGES,
        "interpretation": {
            "joint_minus_single_target_accuracy": (
                "positive means Joint introduces additional target-erasure failure"
            ),
            "joint_minus_single_preservation_accuracy": (
                "negative means Joint introduces additional collateral damage"
            ),
        },
        "analysis_units": {
            "target_level": "six targets",
            "preservation_level": (
                "three designated similar non-target classes per target-specific "
                "Single checkpoint"
            ),
            "group_level": "three baseline-qualified groups",
            "images": (
                "matched observations within target/class; not independent "
                "concept-level experimental units"
            ),
        },
        "target_rows": target_rows,
        "preservation_rows": preservation_rows,
        "group_rows": group_rows,
        "overall_descriptive_summary": overall,
        "sibling_target_rows": sibling_rows,
        "per_image_matched_correctness_path": str(
            (root / "formal" / "evaluations"
             / "per_image_single_joint_transitions.csv").resolve()
        ),
        "generation_runtime": generation_identity,
        "evaluator": evaluator_identity,
        "provenance_categories": config["provenance_notes"],
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(aggregate_root / "summary.json", summary)
    report_path = aggregate_root / "REPORT.md"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(render_markdown(summary), encoding="utf-8")
    temporary.replace(report_path)
    return summary


def run() -> dict[str, Any]:
    config, anchors = protocol.load_protocol(HERE / "config.json")
    primary_root = Path(config["_resolved"]["output_root"])
    root = primary_root / "baseline_qualified_primary_v1"
    formal_root = root / "formal"
    aggregate_path = formal_root / "aggregates" / "summary.json"
    if aggregate_path.is_file():
        existing = protocol.read_json(aggregate_path)
        if existing.get("status") == "complete":
            return existing

    # The original all-five-group formal path must never have completed.
    if (primary_root / "formal" / "completion.json").exists():
        raise RuntimeError(
            "Legacy five-group formal completion exists; refusing qualified subset run"
        )

    pipeline._require_gate(primary_root, "anchor_sanity")
    pipeline._require_gate(primary_root, "original_canary")
    qualification, qualification_path = load_and_validate_qualification(
        primary_root, config
    )
    qualification_hash = protocol.sha256(qualification_path)
    jobs = build_jobs(config, anchors, root, qualification_hash)

    plan_payload = {
        "schema_version": 1,
        "status": "resolved",
        "scope_label": SCOPE_LABEL,
        "research_question": RESEARCH_QUESTION,
        "eligible_groups": ELIGIBLE_SCOPE,
        "excluded_groups": EXCLUDED_SCOPE,
        "qualification_summary_path": str(qualification_path.resolve()),
        "qualification_summary_sha256": qualification_hash,
        "qualification_completed_at": qualification["completed_at"],
        "qualification_preceded_all_primary_joint_and_preservation_jobs": True,
        "joint_results_used_for_inclusion": False,
        "job_count": len(jobs),
        "single_images": EXPECTED_SINGLE_IMAGES,
        "joint_images": EXPECTED_JOINT_IMAGES,
        "total_new_edited_images": EXPECTED_EDITED_IMAGES,
        "original_images_regenerated": 0,
        "jobs": [
            {key: value for key, value in job.items() if key not in {"rows", "checkpoint_spec"}}
            for job in jobs
        ],
        "config_sha256": protocol.sha256(Path(config["_resolved"]["config_path"])),
        "anchors_sha256": protocol.sha256(Path(config["_resolved"]["anchors_path"])),
        "source_hashes": protocol.source_hashes([
            HERE / "qualified_primary.py",
            HERE / "run_qualified_primary.sh",
            HERE / "pipeline.py",
            HERE / "protocol.py",
        ]),
        "created_at": protocol.utc_now(),
    }
    plan_identity = {
        key: plan_payload[key]
        for key in (
            "scope_label",
            "eligible_groups",
            "excluded_groups",
            "qualification_summary_sha256",
            "job_count",
            "single_images",
            "joint_images",
            "total_new_edited_images",
            "config_sha256",
            "anchors_sha256",
            "source_hashes",
        )
    }
    plan_payload["plan_fingerprint"] = protocol.fingerprint(plan_identity)
    plan_path = formal_root / "resolved_plan.json"
    if plan_path.is_file():
        existing_plan = protocol.read_json(plan_path)
        if existing_plan.get("plan_fingerprint") != plan_payload["plan_fingerprint"]:
            raise RuntimeError("Qualified formal plan fingerprint collision")
    else:
        protocol.write_json_atomic(plan_path, plan_payload)

    # Load model and prove provenance before allowing the first generation call.
    generation = pipeline.GenerationRuntime(config)
    audit_path = Path(qualification["scheduler_audit_path"])
    audit = protocol.read_json(audit_path)
    formal_audit_path = formal_root / "scheduler_integrity_gate.json"
    validate_scheduler_for_formal(
        generation, qualification, audit, formal_audit_path
    )
    evaluator = pipeline.EvaluationRuntime(config)

    protocol.write_json_atomic(formal_root / "progress.json", {
        "schema_version": 1,
        "status": "running",
        "completed_jobs": sum(Path(job["result_path"]).is_file() for job in jobs),
        "total_jobs": len(jobs),
        "formal_pipeline_scope": SCOPE_LABEL,
        "updated_at": protocol.utc_now(),
    })
    for index, job in enumerate(jobs, start=1):
        protocol.write_json_atomic(formal_root / "progress.json", {
            "schema_version": 1,
            "status": "running",
            "current_job": job["job_id"],
            "job_index": index,
            "completed_jobs": sum(
                Path(candidate["result_path"]).is_file() for candidate in jobs
            ),
            "total_jobs": len(jobs),
            "updated_at": protocol.utc_now(),
        })
        print(
            f"[qualified formal {index}/{len(jobs)}] {job['job_id']}: 500 rows",
            flush=True,
        )
        result = pipeline._run_formal_job(
            job,
            generation,
            evaluator,
            skip_existing=True,
            purge=True,
        )
        if (
            job["model_type"] == "single"
            and protocol.normalize(job["evaluated_concept"])
            == protocol.normalize(job["single_target"])
        ):
            result_with_path = {
                **result,
                "result_path": job["result_path"],
            }
            validate_single_target_reproduction(
                result_with_path,
                str(job["single_target"]),
                str(job["group_id"]),
                qualification,
                primary_root,
                formal_root / "reproduction_canaries"
                / f"{protocol.slug(str(job['single_target']))}.json",
            )

    for job in jobs:
        validate_result(job)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "scope_label": SCOPE_LABEL,
        "job_count": len(jobs),
        "single_images": EXPECTED_SINGLE_IMAGES,
        "joint_images": EXPECTED_JOINT_IMAGES,
        "new_edited_image_count": EXPECTED_EDITED_IMAGES,
        "original_generation_count": 0,
        "qualification_summary_sha256": qualification_hash,
        "generation_runtime": generation.identity,
        "evaluator": evaluator.identity,
        "completed_at": protocol.utc_now(),
    }
    protocol.write_json_atomic(formal_root / "completion.json", completion)
    protocol.write_json_atomic(formal_root / "progress.json", {
        "schema_version": 1,
        "status": "aggregating",
        "completed_jobs": len(jobs),
        "total_jobs": len(jobs),
        "updated_at": protocol.utc_now(),
    })
    summary = aggregate(
        config,
        anchors,
        qualification,
        qualification_path,
        root,
        jobs,
        generation.identity,
        evaluator.identity,
    )
    protocol.write_json_atomic(formal_root / "progress.json", {
        "schema_version": 1,
        "status": "complete",
        "completed_jobs": len(jobs),
        "total_jobs": len(jobs),
        "summary_path": str(aggregate_path.resolve()),
        "completed_at": protocol.utc_now(),
    })
    return summary


def main() -> int:
    try:
        result = run()
    except Exception as exc:
        print(f"qualified primary error: {exc}", file=sys.stderr)
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
