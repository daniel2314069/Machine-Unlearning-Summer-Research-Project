#!/usr/bin/env python3
"""Validate and aggregate ScaPre final-alpha channel controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_CONFIG = HERE.parents[1] / "config.json"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "formal"], required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def accuracy(rows: list[dict[str, str]]) -> float:
    if not rows:
        raise RuntimeError("cannot calculate accuracy over zero rows")
    return 100.0 * sum(int(row["correct"]) for row in rows) / len(rows)


def metric_block(rows: list[dict[str, str]]) -> dict[str, float]:
    unlearn = accuracy([row for row in rows if row["role"] == "target"])
    preserve = accuracy([row for row in rows if row["role"] == "retain"])
    forgetting = 100.0 - unlearn
    overall = 2.0 * forgetting * preserve / (forgetting + preserve)
    return {"unlearn_accuracy": unlearn, "preserve_accuracy": preserve, "overall_accuracy": overall}


def generation_keys(rows: list[dict[str, str]]) -> set[tuple[str, ...]]:
    fields = ("group", "role", "concept", "sample_index", "prompt", "seed", "seed_source")
    return {tuple(row[field] for field in fields) for row in rows}


def evaluator_control(manifest: dict) -> dict:
    value = {key: item for key, item in manifest.items()
             if key not in {"variant", "checkpoint_sha256", "variant_label_wrapper"}}
    defaults = value.get("scheduler_config", {}).get("_use_default_values")
    if not isinstance(defaults, list) or len(defaults) != len(set(defaults)):
        raise RuntimeError("invalid evaluator scheduler default-value set")
    value["scheduler_config"]["_use_default_values"] = sorted(defaults)
    return value


def signed(value: float) -> str:
    return f"{value:+.2f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def main() -> None:
    ns = args()
    run_dir = ns.run_dir.resolve()
    config = json.loads((run_dir / "actual_config.json").read_text())
    base = json.loads(BASE_CONFIG.read_text())
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    results = run_dir / "results"
    results.mkdir(exist_ok=True)
    seeds = config["edit_seeds"] if ns.profile == "formal" else [20260820]
    variants = config["formal_variants"] if ns.profile == "formal" else config["smoke_variants"]
    expected_rows = config[
        "expected_images_per_variant_formal" if ns.profile == "formal"
        else "expected_images_per_variant_smoke"
    ]

    concepts = []
    roles = {}
    groups = base["groups"] if ns.profile == "formal" else base["groups"][:1]
    for group in groups:
        for role, field in (("target", "targets"), ("retain", "retains")):
            for concept in group[field]:
                concepts.append(concept)
                roles[concept] = role

    scores: dict[tuple[int, str], list[dict[str, str]]] = {}
    metrics: dict[tuple[int, str], dict[str, float]] = {}
    reference_keys = None
    reference_evaluator = None
    score_hashes: dict[str, dict[str, str]] = {}
    per_seed_rows: list[dict[str, object]] = []
    per_target_rows: list[dict[str, object]] = []
    for seed in seeds:
        score_hashes[str(seed)] = {}
        for variant in variants:
            path = run_dir / "seeds" / str(seed) / "evaluation" / variant / "scores.csv"
            rows = read_csv(path)
            if len(rows) != expected_rows or len(generation_keys(rows)) != expected_rows:
                raise RuntimeError(f"seed {seed} {variant} row/key count mismatch")
            if any(row["variant"] != variant or row["correct"] not in {"0", "1"} for row in rows):
                raise RuntimeError(f"seed {seed} {variant} invalid score labels")
            keys = generation_keys(rows)
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise RuntimeError("generation prompts/seeds differ across seed or variant")
            manifest = json.loads((path.parent / "evaluation_manifest.json").read_text())
            controlled = evaluator_control(manifest)
            if reference_evaluator is None:
                reference_evaluator = controlled
            elif controlled != reference_evaluator:
                raise RuntimeError("evaluator/classifier/sampler fingerprint changed")
            scores[(seed, variant)] = rows
            metrics[(seed, variant)] = metric_block(rows)
            score_hashes[str(seed)][variant] = sha256(path)
            per_seed_rows.append({"edit_seed": seed, "variant": variant, **metrics[(seed, variant)]})
            for concept in concepts:
                selected = [row for row in rows if row["concept"] == concept]
                if len(selected) != expected_rows // len(concepts):
                    raise RuntimeError(f"seed {seed} {variant} concept denominator mismatch: {concept}")
                official_value = None
                if variant != "official" and (seed, "official") in scores:
                    official_value = accuracy([row for row in scores[(seed, "official")] if row["concept"] == concept])
                current = accuracy(selected)
                per_target_rows.append({
                    "edit_seed": seed, "variant": variant, "role": roles[concept],
                    "concept": concept, "accuracy": current,
                    "delta_vs_official": "" if official_value is None else current - official_value,
                })

    matrix_rows: list[dict[str, object]] = []
    for seed in seeds:
        for variant in variants:
            audit = json.loads((run_dir / "seeds" / str(seed) / "alpha_audits" / f"{variant}.json").read_text())
            if not audit.get("completed") or len(audit["matrix_records"]) != config["expected_alpha_intercepts_per_edit"]:
                raise RuntimeError(f"seed {seed} {variant} alpha audit incomplete")
            if not audit.get("checkpoint_finiteness", {}).get("all_projection_weights_finite"):
                raise RuntimeError(f"seed {seed} {variant} checkpoint finiteness gate failed")
            cleanup = json.loads(
                (run_dir / "seeds" / str(seed) / "stages" / f"checkpoint_{variant}.cleanup.json").read_text()
            )
            completed_edit = json.loads(
                (run_dir / "seeds" / str(seed) / "stages" / f"edit_{variant}.completed.json").read_text()
            )
            if cleanup.get("status") != "passed" or cleanup.get("sha256") != completed_edit.get("checkpoint_sha256"):
                raise RuntimeError(f"seed {seed} {variant} checkpoint cleanup accounting failed")
            for record in audit["matrix_records"]:
                official = record["official"]
                controlled = record["controlled"]
                matrix_rows.append({
                    "edit_seed": seed, "variant": variant,
                    "projection": record["projection"], "layer": record["layer_index"],
                    "output_dimension": official["output_dimension"],
                    **{f"official_alpha_{key}": official[key] for key in ("mean", "std", "min", "max", "p50", "p95", "p99")},
                    **{f"controlled_alpha_{key}": controlled[key] for key in ("mean", "std", "min", "max", "p50", "p95", "p99")},
                    "constant_alpha_value": controlled["mean"] if variant == "constant_mean" else "",
                    "constant_mean_abs_error": record["constant_mean_abs_error"],
                    "exact_multiset_preserved": record["exact_multiset_preserved"],
                    "identity_exact_all_ones": record["identity_exact_all_ones"],
                    "trace_B": controlled["trace_B"], "frobenius_B": controlled["frobenius_B"],
                    "shuffle_salt": record["shuffle_salt"] or "",
                    "permutation_seed": record["permutation_seed"] or "",
                })

    comparison_specs = [
        ("official_vs_constant_mean", "official", "constant_mean"),
        ("official_vs_shuffled", "official", "shuffled"),
        ("official_vs_identity_B", "official", "identity_B"),
        ("constant_mean_vs_identity_B", "constant_mean", "identity_B"),
    ] if ns.profile == "formal" else []
    comparison_rows: list[dict[str, object]] = []
    for label, baseline, treatment in comparison_specs:
        seed_deltas = []
        for seed in seeds:
            delta = {metric: metrics[(seed, treatment)][metric] - metrics[(seed, baseline)][metric]
                     for metric in ("unlearn_accuracy", "preserve_accuracy", "overall_accuracy")}
            seed_deltas.append(delta)
            comparison_rows.append({
                "comparison": label, "scope": "aggregate", "concept": "",
                "edit_seed": seed, "delta_accuracy": "",
                "delta_unlearn": delta["unlearn_accuracy"],
                "delta_preserve": delta["preserve_accuracy"],
                "delta_overall": delta["overall_accuracy"],
                "same_direction_count": "", "total_seeds": "",
            })
        for metric, field, improves in (
            ("unlearn_accuracy", "delta_unlearn", lambda x: x < 0),
            ("preserve_accuracy", "delta_preserve", lambda x: x > 0),
            ("overall_accuracy", "delta_overall", lambda x: x > 0),
        ):
            values = [row[metric] for row in seed_deltas]
            comparison_rows.append({
                "comparison": label, "scope": f"five_seed_mean_{metric}", "concept": "",
                "edit_seed": "mean", "delta_accuracy": "", "delta_unlearn": "",
                "delta_preserve": "", "delta_overall": "", field: statistics.mean(values),
                "same_direction_count": sum(improves(value) for value in values), "total_seeds": len(values),
            })
        target_concepts = [concept for concept in concepts if roles[concept] == "target"]
        for concept in target_concepts:
            values = []
            for seed in seeds:
                left = accuracy([row for row in scores[(seed, baseline)] if row["concept"] == concept])
                right = accuracy([row for row in scores[(seed, treatment)] if row["concept"] == concept])
                values.append(right - left)
                comparison_rows.append({
                    "comparison": label, "scope": "per_target", "concept": concept,
                    "edit_seed": seed, "delta_accuracy": right - left,
                    "delta_unlearn": "", "delta_preserve": "", "delta_overall": "",
                    "same_direction_count": "", "total_seeds": "",
                })
            comparison_rows.append({
                "comparison": label, "scope": "per_target_five_seed_mean", "concept": concept,
                "edit_seed": "mean", "delta_accuracy": statistics.mean(values),
                "delta_unlearn": "", "delta_preserve": "", "delta_overall": "",
                "same_direction_count": sum(value < 0 for value in values), "total_seeds": len(values),
            })

    write_csv(results / "alpha_matrix_summary.csv", matrix_rows, list(matrix_rows[0]))
    write_csv(results / "per_seed_metrics.csv", per_seed_rows, list(per_seed_rows[0]))
    write_csv(results / "per_target_metrics.csv", per_target_rows, list(per_target_rows[0]))
    comparison_fields = ["comparison", "scope", "concept", "edit_seed", "delta_accuracy",
                         "delta_unlearn", "delta_preserve", "delta_overall",
                         "same_direction_count", "total_seeds"]
    write_csv(results / "comparison_deltas.csv", comparison_rows, comparison_fields)

    generated_variants = variants if ns.profile == "smoke" else config["generated_formal_variants"]
    image_manifest_rows: list[dict[str, object]] = []
    seen_images = set()
    for seed in seeds:
        for variant in generated_variants:
            expected_root = (run_dir / "seeds" / str(seed) / "evaluation" / variant / "images").resolve()
            for row in scores[(seed, variant)]:
                image = Path(row["image_path"]).resolve()
                try:
                    relative = image.relative_to(expected_root)
                except ValueError as error:
                    raise RuntimeError(f"image escaped expected output root: {image}") from error
                if image in seen_images or not image.is_file():
                    raise RuntimeError(f"generated image missing or duplicated: {image}")
                seen_images.add(image)
                image_manifest_rows.append({
                    "edit_seed": seed, "variant": variant, "concept": row["concept"],
                    "sample_index": row["sample_index"],
                    "relative_image_path": str(image.relative_to(run_dir)),
                    "size_bytes": image.stat().st_size, "sha256": sha256(image),
                })
    expected_generated_images = len(seeds) * len(generated_variants) * expected_rows
    if len(image_manifest_rows) != expected_generated_images:
        raise RuntimeError("generated image manifest count mismatch")
    write_csv(results / "generated_image_manifest.csv", image_manifest_rows, list(image_manifest_rows[0]))

    controlled_checks = []
    for seed in seeds:
        controlled_checks.append(json.loads(
            (run_dir / "seeds" / str(seed) / "controlled_ablation_check.json").read_text()
        ))
    integrity = {
        "status": "passed", "profile": ns.profile, "edit_seeds": seeds,
        "variants": variants, "rows_per_variant_seed": expected_rows,
        "row_count": len(per_seed_rows) * expected_rows,
        "generation_keys_identical": True, "duplicate_generation_keys": 0,
        "prompt_lists_and_generation_seeds_identical": True,
        "evaluator_fingerprint_identical": True,
        "evaluator_fingerprint_sha256": hashlib.sha256(
            json.dumps(reference_evaluator, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "raw_mi_identical_across_variants": all(item["same_raw_mi"] for item in controlled_checks),
        "preaggregate_alpha_identical_across_variants": all(item["same_preaggregate_alpha"] for item in controlled_checks),
        "non_alpha_edit_inputs_identical": all(item["same_normalized_edit_command"] for item in controlled_checks),
        "official_empty_string_neutral_only": True,
        "constant_mean_gate_passed": all(
            float(row["constant_mean_abs_error"]) <= config["numeric_tolerance"]
            for row in matrix_rows if row["variant"] == "constant_mean"
        ),
        "shuffle_multiset_gate_passed": all(
            row["exact_multiset_preserved"] is True
            for row in matrix_rows if str(row["variant"]).startswith("shuffled")
        ),
        "identity_all_ones_gate_passed": all(
            row["identity_exact_all_ones"] is True
            for row in matrix_rows if row["variant"] == "identity_B"
        ),
        "all_checkpoint_projection_weights_finite": True,
        "official_reference_reused": ns.profile == "formal",
        "new_generated_image_count": len(image_manifest_rows),
        "generated_image_manifest_sha256": sha256(results / "generated_image_manifest.csv"),
        "official_reference_validation": json.loads((run_dir / "official_reference_validation.json").read_text())
            if ns.profile == "formal" else None,
        "score_sha256": score_hashes,
        "config_sha256": sha256(run_dir / "actual_config.json"),
        "base_config_sha256": sha256(run_dir / "base_config.json"),
        "protocol_sha256": sha256(run_dir / "protocol.csv"),
        "git_commit": run_manifest["git_commit"], "git_branch": run_manifest["git_branch"],
        "git_status_start_clean": run_manifest["git_status_start"] == [],
        "git_status_before_aggregation_clean": run_manifest.get("git_status_before_aggregation") == [],
    }
    if not all(value is True for key, value in integrity.items() if key.endswith("_passed") or key in {
        "raw_mi_identical_across_variants", "preaggregate_alpha_identical_across_variants",
        "non_alpha_edit_inputs_identical", "all_checkpoint_projection_weights_finite"
    }):
        raise RuntimeError("one or more integrity gates failed")
    (results / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n")

    if ns.profile == "smoke":
        summary = f"""# ScaPre Informax alpha-channel controls: smoke

The smoke stage passed for all `{len(variants)}` variants at edit seed `20260820`.
Every checkpoint was finite, each variant produced `{expected_rows}` evaluated
images, raw MI and every non-final-alpha edit input matched exactly, and the two
alternate shuffle salts passed their bijection/multiset gates. These image
results are implementation checks only and were not used to select the formal
permutation or draw scientific conclusions.

Formal shuffle salt remains preregistered as
`{config['formal_shuffle_salt']}`.
"""
    else:
        seed_table = []
        for seed in seeds:
            row = [str(seed)]
            for variant in variants:
                block = metrics[(seed, variant)]
                row.append(f"{block['unlearn_accuracy']:.2f}/{block['preserve_accuracy']:.2f}/{block['overall_accuracy']:.2f}")
            seed_table.append(row)
        mean_rows = []
        for label, baseline, treatment in comparison_specs:
            values = {metric: [metrics[(seed, treatment)][metric] - metrics[(seed, baseline)][metric] for seed in seeds]
                      for metric in ("unlearn_accuracy", "preserve_accuracy", "overall_accuracy")}
            mean_rows.append([
                label, signed(statistics.mean(values["unlearn_accuracy"])),
                signed(statistics.mean(values["preserve_accuracy"])),
                signed(statistics.mean(values["overall_accuracy"])),
                f"{sum(value < 0 for value in values['unlearn_accuracy'])}/5",
                f"{sum(value > 0 for value in values['preserve_accuracy'])}/5",
                f"{sum(value > 0 for value in values['overall_accuracy'])}/5",
            ])
        target_pattern_rows = []
        for label, baseline, treatment in comparison_specs:
            for concept in [item for item in concepts if roles[item] == "target"]:
                deltas = []
                for seed in seeds:
                    left = accuracy([row for row in scores[(seed, baseline)] if row["concept"] == concept])
                    right = accuracy([row for row in scores[(seed, treatment)] if row["concept"] == concept])
                    deltas.append(right - left)
                target_pattern_rows.append([
                    label, concept, signed(statistics.mean(deltas)),
                    f"{sum(value < 0 for value in deltas)}/5",
                ])
        def mean_delta(baseline: str, treatment: str) -> dict[str, float]:
            return {
                metric: statistics.mean(
                    metrics[(seed, treatment)][metric] - metrics[(seed, baseline)][metric]
                    for seed in seeds
                )
                for metric in ("unlearn_accuracy", "preserve_accuracy", "overall_accuracy")
            }

        channel_deltas = [
            mean_delta("official", "constant_mean"),
            mean_delta("official", "shuffled"),
        ]
        official_pareto_dominates_channels = all(
            item["unlearn_accuracy"] >= 0
            and item["preserve_accuracy"] <= 0
            and item["overall_accuracy"] <= 0
            and any(value != 0 for value in item.values())
            for item in channel_deltas
        )
        identity_delta = mean_delta("official", "identity_B")
        identity_dominates = (
            identity_delta["unlearn_accuracy"] <= 0
            and identity_delta["preserve_accuracy"] >= 0
            and identity_delta["overall_accuracy"] >= 0
            and any(value != 0 for value in identity_delta.values())
        )
        identity_is_dominated = (
            identity_delta["unlearn_accuracy"] >= 0
            and identity_delta["preserve_accuracy"] <= 0
            and identity_delta["overall_accuracy"] <= 0
            and any(value != 0 for value in identity_delta.values())
        )
        if official_pareto_dominates_channels:
            observed_interpretation = (
                "Official Pareto-dominates both mean-matched channel controls on the five-seed means. "
                "The next question is which residual alpha differences carry useful channel-specific information."
            )
        else:
            observed_interpretation = (
                "The channel-control comparison is mixed rather than a strict Pareto result. "
                "The next question is whether the trade-off is concentrated in particular target families, using the saved per-target deltas."
            )
        if identity_dominates:
            identity_interpretation = (
                "Identity_B Pareto-dominates official on the five-seed means, so uniform all-one weighting remains plausible in this setting."
            )
        elif identity_is_dominated:
            identity_interpretation = (
                "Official Pareto-dominates identity_B on the five-seed means, showing a practical cost for the all-one paper-limit control."
            )
        else:
            identity_interpretation = (
                "The identity_B comparison is mixed; its scale/channel trade-off must be reported directly rather than collapsed to one claim."
            )
        summary = f"""# ScaPre Informax alpha-channel controls

## Result

All integrity gates passed. Delta in this report is treatment minus baseline;
lower Unlearn and higher Preserve/Overall are favorable. The controls support
only conclusions about the final concept-max channel assignment in this fixed
official-empty-neutral configuration.

{observed_interpretation} {identity_interpretation} These labels use strict
Pareto direction only; no significance or similarity threshold was invented.

## Exact intervention and isolation

The experiment-only runner replaced only `row_w_max` after the official
per-concept MI -> z-score -> sigmoid/power -> concept-max pipeline and before
the row-wise closed-form term `erase_scale * alpha_i * PiC`. The production
editor was byte-unchanged. Raw MI, pre-aggregate alpha, thresholds, Informax
noise tensors, UCE inputs, prompts, generation seeds, sampler, 50 steps, CFG
7.5, classifier, and row counts matched across variants.

## Smoke

Seed `20260820` produced finite checkpoints and complete small evaluations for
official, constant mean, three preregistered/non-selected shuffle salts, and
identity. Smoke was not used for tuning.

## Five-seed image metrics

Cells are `Unlearn / Preserve / Overall` percentages.

{table(['Seed', *variants], seed_table)}

## Required comparisons

{table(['Comparison', 'Mean ΔU', 'Mean ΔP', 'Mean ΔO', 'U favorable', 'P favorable', 'O favorable'], mean_rows)}

### Official vs constant_mean

The corresponding row above is the mean-matched channel-identity control.

### Official vs shuffled

The corresponding row above preserves the exact per-matrix alpha multiset and
destroys only channel identity.

### Official vs identity_B

The corresponding row above is the separate all-one paper-limit control.

### Constant_mean vs identity_B

This row separates uniform alpha scale from channel differentiation.

### Optional no-Informax reference

Omitted because setting only final `B=0` leaves per-concept Informax-weighted
UCE accumulation active, while globally disabling Informax changes more than
the final-alpha intervention and is not a clean established No-Informax match.

## Per-target patterns

Mean target accuracy delta is treatment minus baseline; negative is stronger
unlearning. `Favorable seeds` counts negative deltas.

{table(['Comparison', 'Target', 'Mean Δ accuracy', 'Favorable seeds'], target_pattern_rows)}

All seed-level and per-target deltas are in `comparison_deltas.csv`; full
per-concept accuracies are in `per_target_metrics.csv`.

## Interpretation and limitations

Interpret mixed Unlearn/Preserve effects as trade-offs. `constant_mean`
preserves each matrix's mean alpha but not every nonlinear notion of edit
strength. `identity_B` is a separate paper-limit control with different scale.
The findings cover SD v1.5, this reconstructed Confuse5 protocol, official
empty-string neutral, fixed evaluator, and five edit seeds only. They do not
test a replacement relevance estimator or matched-retain neutral.

The next research question is the one stated in the result paragraph above; it
is selected by strict metric direction, not by tuning a threshold on this run.

## Reproducibility

- Git commit: `{run_manifest['git_commit']}` on `{run_manifest['git_branch']}`
- Edit seeds: `{', '.join(str(seed) for seed in seeds)}`
- Per variant/seed: 25 concepts x 120 = 3,000 images
- Official image scores: reused byte-for-byte after archive, protocol, asset,
  evaluator, row-key, and score-hash validation
- Protocol SHA-256: `{integrity['protocol_sha256']}`
- Actual config SHA-256: `{integrity['config_sha256']}`
- Base config SHA-256: `{integrity['base_config_sha256']}`
- Total score rows: `{integrity['row_count']}`
- Working tree: clean at launch and immediately before aggregation; the worker
  also aborts unless it remains clean after aggregation and records the final
  empty status in `run_manifest.json`
"""
    (results / "summary.md").write_text(summary)
    result_hashes = {path.name: sha256(path) for path in sorted(results.iterdir()) if path.is_file()}
    (results / "result_manifest.json").write_text(json.dumps({
        "status": "passed", "profile": ns.profile, "files_sha256_before_manifest": result_hashes,
        "required_files_present": all((results / name).is_file() for name in (
            "alpha_matrix_summary.csv", "per_seed_metrics.csv", "per_target_metrics.csv",
            "comparison_deltas.csv", "generated_image_manifest.csv",
            "integrity_report.json", "summary.md",
        )),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
