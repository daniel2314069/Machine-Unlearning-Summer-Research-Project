#!/usr/bin/env python3
"""Validate and aggregate the fixed ScaPre Informax edit-seed experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import torch
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "formal"], required=True)
    return parser.parse_args()


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
        raise RuntimeError("cannot compute accuracy over zero rows")
    return 100.0 * sum(int(row["correct"]) for row in rows) / len(rows)


def overall(unlearn: float, preserve: float) -> float:
    forgetting = 100.0 - unlearn
    denominator = forgetting + preserve
    return 0.0 if denominator == 0 else 2.0 * forgetting * preserve / denominator


def metric_block(rows: list[dict[str, str]]) -> dict[str, float]:
    target_rows = [row for row in rows if row["role"] == "target"]
    retain_rows = [row for row in rows if row["role"] == "retain"]
    unlearn = accuracy(target_rows)
    preserve = accuracy(retain_rows)
    return {
        "unlearn_acc": unlearn,
        "preserve_acc": preserve,
        "overall_acc": overall(unlearn, preserve),
    }


def controlled_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        row[field]
        for field in (
            "group", "role", "concept", "sample_index", "prompt", "seed", "seed_source"
        )
    )


def validate_rows(
    rows: list[dict[str, str]],
    base_config: dict,
    profile: str,
    variant: str,
) -> set[tuple[str, ...]]:
    groups = base_config["groups"][:1] if profile == "smoke" else base_config["groups"]
    per_concept = base_config["evaluation"][
        "smoke_images_per_concept" if profile == "smoke" else "formal_images_per_concept"
    ]
    expected: dict[str, tuple[str, str]] = {}
    for group in groups:
        for role, field in (("target", "targets"), ("retain", "retains")):
            for concept in group[field]:
                expected[concept] = (group["id"], role)

    expected_rows = len(expected) * per_concept
    if len(rows) != expected_rows:
        raise RuntimeError(f"{variant} has {len(rows)} rows; expected {expected_rows}")
    keys: set[tuple[str, ...]] = set()
    counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    for row in rows:
        if row["variant"] != variant:
            raise RuntimeError(f"variant column mismatch in {variant}")
        concept = row["concept"]
        if concept not in expected:
            raise RuntimeError(f"unexpected concept in {variant}: {concept}")
        expected_group, expected_role = expected[concept]
        if (row["group"], row["role"]) != (expected_group, expected_role):
            raise RuntimeError(f"group/role mismatch for {concept} in {variant}")
        if row["correct"] not in {"0", "1"}:
            raise RuntimeError(f"non-binary classifier result for {concept}")
        key = controlled_key(row)
        if key in keys:
            raise RuntimeError(f"duplicate generation key in {variant}: {key}")
        keys.add(key)
        counts[concept] += 1
        role_counts[row["role"]] += 1

    if set(counts) != set(expected) or any(value != per_concept for value in counts.values()):
        raise RuntimeError(f"{variant} does not contain exactly {per_concept} rows per concept")
    if profile == "formal" and (role_counts["target"], role_counts["retain"]) != (1200, 1800):
        raise RuntimeError(f"{variant} target/retain denominators changed: {role_counts}")
    return keys


def evaluator_control(manifest: dict) -> dict:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"variant", "checkpoint_sha256"}
    }


def load_diagnostic_records(path: Path) -> dict[tuple[str, int, str], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records: dict[tuple[str, int, str], dict[str, object]] = {}
    for record in payload["records"]:
        if record["stage"] != "aggregate" or record["target_concept"] is None:
            continue
        key = (record["projection"], int(record["layer_index"]), record["target_concept"])
        if key in records:
            raise RuntimeError(f"duplicate Informax diagnostic key: {key}")
        records[key] = record
    return records


def create_diagnostic_csv(seed_dir: Path) -> list[dict[str, object]]:
    result_path = seed_dir / "results" / "informax_diagnostics.csv"
    if result_path.exists():
        return [dict(row) for row in read_csv(result_path)]

    official = load_diagnostic_records(seed_dir / "diagnostics" / "official.pt")
    matched = load_diagnostic_records(seed_dir / "diagnostics" / "matched_retain.pt")
    if official.keys() != matched.keys():
        raise RuntimeError("Informax diagnostic keys differ between variants")
    rows: list[dict[str, object]] = []
    for key in sorted(official):
        projection, layer_index, target = key
        alpha_official = official[key]["alpha"].double().flatten()
        alpha_matched = matched[key]["alpha"].double().flatten()
        raw_official = official[key]["raw_mi"].double().flatten()
        raw_matched = matched[key]["raw_mi"].double().flatten()
        if alpha_official.numel() != alpha_matched.numel():
            raise RuntimeError(f"channel count differs at {key}")
        correlation = float(
            spearmanr(alpha_official.numpy(), alpha_matched.numpy()).statistic
        )
        if math.isnan(correlation):
            correlation = 0.0
        row: dict[str, object] = {
            "projection": projection,
            "layer_index": layer_index,
            "target_concept": target,
            "channels": alpha_official.numel(),
            "spearman_alpha": correlation,
            "mean_raw_mi_official": raw_official.mean().item(),
            "mean_raw_mi_matched": raw_matched.mean().item(),
        }
        for percent in (1, 5, 10):
            count = max(1, math.ceil(alpha_official.numel() * percent / 100.0))
            left = set(torch.topk(alpha_official, count).indices.tolist())
            right = set(torch.topk(alpha_matched, count).indices.tolist())
            row[f"top_{percent}_percent_overlap"] = len(left & right) / len(left)
        rows.append(row)
    write_csv(
        result_path,
        rows,
        [
            "projection", "layer_index", "target_concept", "channels",
            "spearman_alpha", "mean_raw_mi_official", "mean_raw_mi_matched",
            "top_1_percent_overlap", "top_5_percent_overlap", "top_10_percent_overlap",
        ],
    )
    return rows


def as_float(row: dict[str, object], key: str) -> float:
    return float(row[key])


def summarize_diagnostics(seed: int, rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise RuntimeError(f"seed {seed} has no Informax diagnostics")
    return {
        "edit_seed": seed,
        "records": len(rows),
        "mean_raw_mi_official": statistics.mean(as_float(row, "mean_raw_mi_official") for row in rows),
        "mean_raw_mi_matched": statistics.mean(as_float(row, "mean_raw_mi_matched") for row in rows),
        "mean_alpha_spearman": statistics.mean(as_float(row, "spearman_alpha") for row in rows),
        "mean_top_1_percent_overlap": statistics.mean(as_float(row, "top_1_percent_overlap") for row in rows),
        "mean_top_5_percent_overlap": statistics.mean(as_float(row, "top_5_percent_overlap") for row in rows),
        "mean_top_10_percent_overlap": statistics.mean(as_float(row, "top_10_percent_overlap") for row in rows),
    }


def fmt(value: float) -> str:
    return f"{value:.2f}"


def signed(value: float) -> str:
    return f"{value:+.2f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    robustness = json.loads(args.config.read_text())
    base_config = json.loads(args.base_config.read_text())
    seeds = robustness["edit_seeds"] if args.profile == "formal" else [20260821]
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    expected_new_source_hashes = run_manifest["source_sha256"]

    group_order = [group["id"] for group in base_config["groups"]]
    if args.profile == "smoke":
        group_order = group_order[:1]
    concept_order: list[str] = []
    concept_role: dict[str, str] = {}
    concept_group: dict[str, str] = {}
    for group in base_config["groups"]:
        if group["id"] not in group_order:
            continue
        for role, field in (("target", "targets"), ("retain", "retains")):
            for concept in group[field]:
                concept_order.append(concept)
                concept_role[concept] = role
                concept_group[concept] = group["id"]

    per_seed_rows: list[dict[str, object]] = []
    per_group_rows: list[dict[str, object]] = []
    per_concept_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    reference_keys: set[tuple[str, ...]] | None = None
    reference_evaluator: dict | None = None
    score_hashes: dict[str, dict[str, str]] = {}

    import hashlib

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    for seed in seeds:
        seed_dir = run_dir / "seeds" / str(seed)
        if seed != 20260820:
            seed_source_hashes = json.loads((seed_dir / "source_manifest.json").read_text())
            if seed_source_hashes != expected_new_source_hashes:
                raise RuntimeError(f"seed {seed}: source hashes differ from the run manifest")
        by_variant: dict[str, list[dict[str, str]]] = {}
        seed_keys: set[tuple[str, ...]] | None = None
        score_hashes[str(seed)] = {}
        for variant in ("official", "matched_retain"):
            evaluation_dir = seed_dir / "evaluation" / variant
            score_path = evaluation_dir / "scores.csv"
            manifest_path = evaluation_dir / "evaluation_manifest.json"
            rows = read_csv(score_path)
            keys = validate_rows(rows, base_config, args.profile, variant)
            if seed_keys is None:
                seed_keys = keys
            elif seed_keys != keys:
                raise RuntimeError(f"seed {seed}: variants use different generation keys")
            if reference_keys is None:
                reference_keys = keys
            elif reference_keys != keys:
                raise RuntimeError(f"seed {seed}: generation keys differ across edit seeds")
            manifest = json.loads(manifest_path.read_text())
            controlled_manifest = evaluator_control(manifest)
            if reference_evaluator is None:
                reference_evaluator = controlled_manifest
            elif controlled_manifest != reference_evaluator:
                raise RuntimeError(f"seed {seed}: classifier/evaluator fingerprint changed")
            by_variant[variant] = rows
            score_hashes[str(seed)][variant] = sha256(score_path)

        seed_metrics = {variant: metric_block(rows) for variant, rows in by_variant.items()}
        per_seed_row: dict[str, object] = {"edit_seed": seed}
        for metric in ("unlearn_acc", "preserve_acc", "overall_acc"):
            per_seed_row[f"official_{metric}"] = seed_metrics["official"][metric]
            per_seed_row[f"matched_{metric}"] = seed_metrics["matched_retain"][metric]
            per_seed_row[f"delta_{metric}"] = (
                seed_metrics["matched_retain"][metric] - seed_metrics["official"][metric]
            )
        per_seed_rows.append(per_seed_row)

        seed_concept_rows: list[dict[str, object]] = []
        for concept in concept_order:
            official_accuracy = accuracy([row for row in by_variant["official"] if row["concept"] == concept])
            matched_accuracy = accuracy([row for row in by_variant["matched_retain"] if row["concept"] == concept])
            row = {
                "edit_seed": seed,
                "group": concept_group[concept],
                "role": concept_role[concept],
                "concept": concept,
                "official_acc": official_accuracy,
                "matched_acc": matched_accuracy,
                "delta_acc": matched_accuracy - official_accuracy,
            }
            per_concept_rows.append(row)
            seed_concept_rows.append(row)
        write_csv(
            seed_dir / "results" / "per_concept.csv",
            seed_concept_rows,
            ["edit_seed", "group", "role", "concept", "official_acc", "matched_acc", "delta_acc"],
        )

        seed_group_rows: list[dict[str, object]] = []
        for group in group_order:
            values = {
                variant: metric_block([row for row in rows if row["group"] == group])
                for variant, rows in by_variant.items()
            }
            row = {"edit_seed": seed, "group": group}
            for metric in ("unlearn_acc", "preserve_acc", "overall_acc"):
                row[f"official_{metric}"] = values["official"][metric]
                row[f"matched_{metric}"] = values["matched_retain"][metric]
                row[f"delta_{metric}"] = values["matched_retain"][metric] - values["official"][metric]
            per_group_rows.append(row)
            seed_group_rows.append(row)
        write_csv(
            seed_dir / "results" / "per_group.csv",
            seed_group_rows,
            list(seed_group_rows[0].keys()),
        )
        write_csv(
            seed_dir / "results" / "aggregate.csv",
            [per_seed_row],
            list(per_seed_row.keys()),
        )
        diagnostic_rows.append(summarize_diagnostics(seed, create_diagnostic_csv(seed_dir)))

    per_seed_fields = list(per_seed_rows[0].keys())
    write_csv(results_dir / "per_seed.csv", per_seed_rows, per_seed_fields)
    write_csv(results_dir / "per_group_seed.csv", per_group_rows, list(per_group_rows[0].keys()))
    write_csv(results_dir / "per_concept_seed.csv", per_concept_rows, list(per_concept_rows[0].keys()))
    write_csv(
        results_dir / "informax_seed_diagnostics.csv",
        diagnostic_rows,
        list(diagnostic_rows[0].keys()),
    )

    delta_specs = [
        ("delta_unlearn", "delta_unlearn_acc", lambda value: value <= 0, "<= 0"),
        ("delta_preserve", "delta_preserve_acc", lambda value: value > 0, "> 0"),
        ("delta_overall", "delta_overall_acc", lambda value: value > 0, "> 0"),
    ]
    aggregate_rows: list[dict[str, object]] = []
    for label, field, improves, definition in delta_specs:
        values = [float(row[field]) for row in per_seed_rows]
        aggregate_rows.append({
            "metric": label,
            "mean": statistics.mean(values),
            "std": sample_std(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "improving_seed_count": sum(improves(value) for value in values),
            "total_seeds": len(values),
            "improvement_definition": definition,
        })
    write_csv(
        results_dir / "aggregate_across_seeds.csv",
        aggregate_rows,
        list(aggregate_rows[0].keys()),
    )

    group_robustness_rows: list[dict[str, object]] = []
    for group in group_order:
        selected = [row for row in per_group_rows if row["group"] == group]
        group_robustness_rows.append({
            "group": group,
            "mean_delta_unlearn": statistics.mean(float(row["delta_unlearn_acc"]) for row in selected),
            "mean_delta_preserve": statistics.mean(float(row["delta_preserve_acc"]) for row in selected),
            "mean_delta_overall": statistics.mean(float(row["delta_overall_acc"]) for row in selected),
            "positive_preserve_seed_count": sum(float(row["delta_preserve_acc"]) > 0 for row in selected),
            "positive_overall_seed_count": sum(float(row["delta_overall_acc"]) > 0 for row in selected),
            "total_seeds": len(selected),
        })
    write_csv(
        results_dir / "per_group_robustness.csv",
        group_robustness_rows,
        list(group_robustness_rows[0].keys()),
    )

    retain_rows: list[dict[str, object]] = []
    for concept in concept_order:
        if concept_role[concept] != "retain":
            continue
        selected = [row for row in per_concept_rows if row["concept"] == concept]
        row: dict[str, object] = {"group": concept_group[concept], "concept": concept}
        for item in selected:
            seed = int(item["edit_seed"])
            row[f"official_{seed}"] = item["official_acc"]
            row[f"matched_{seed}"] = item["matched_acc"]
        deltas = [float(item["delta_acc"]) for item in selected]
        row["mean_delta"] = statistics.mean(deltas)
        row["improvement_seed_count"] = sum(value > 0 for value in deltas)
        row["total_seeds"] = len(deltas)
        retain_rows.append(row)
    write_csv(
        results_dir / "per_retain_robustness.csv",
        retain_rows,
        list(retain_rows[0].keys()),
    )

    rule = robustness["robustness_rule"]
    mean_du = float(aggregate_rows[0]["mean"])
    mean_dp = float(aggregate_rows[1]["mean"])
    mean_do = float(aggregate_rows[2]["mean"])
    unlearn_count = int(aggregate_rows[0]["improving_seed_count"])
    preserve_count = int(aggregate_rows[1]["improving_seed_count"])
    overall_count = int(aggregate_rows[2]["improving_seed_count"])
    new_positive_preserve = sum(
        int(row["edit_seed"]) != 20260820 and float(row["delta_preserve_acc"]) > 0
        for row in per_seed_rows
    )
    positive_mean_groups = sum(float(row["mean_delta_preserve"]) > 0 for row in group_robustness_rows)
    positive_mean_retains = sum(float(row["mean_delta"]) > 0 for row in retain_rows)
    majority_positive_retains = sum(
        int(row["improvement_seed_count"]) >= math.ceil(len(seeds) / 2)
        for row in retain_rows
    )
    if args.profile == "formal":
        robust = (
            preserve_count >= rule["minimum_positive_preserve_seeds"]
            and mean_dp >= rule["minimum_mean_preserve_delta_pp"]
            and mean_du <= rule["maximum_mean_unlearn_delta_pp"]
            and overall_count >= rule["minimum_positive_overall_seeds"]
        )
        not_supported = (
            preserve_count <= 1
            or mean_dp <= 0
            or (mean_dp > 0 and mean_du > rule["material_mean_unlearn_degradation_pp"])
        )
        judgment = "ROBUSTLY SUPPORTED" if robust else ("NOT SUPPORTED" if not_supported else "SEED-SENSITIVE")
    else:
        judgment = None

    seed_table = [
        [
            str(row["edit_seed"]),
            fmt(float(row["official_unlearn_acc"])), fmt(float(row["matched_unlearn_acc"])), signed(float(row["delta_unlearn_acc"])),
            fmt(float(row["official_preserve_acc"])), fmt(float(row["matched_preserve_acc"])), signed(float(row["delta_preserve_acc"])),
            signed(float(row["delta_overall_acc"])),
        ]
        for row in per_seed_rows
    ]
    aggregate_table = [
        [
            str(row["metric"]), fmt(float(row["mean"])), fmt(float(row["std"])),
            fmt(float(row["median"])), fmt(float(row["min"])), fmt(float(row["max"])),
            f"{row['improving_seed_count']}/{row['total_seeds']}",
        ]
        for row in aggregate_rows
    ]
    group_table = [
        [
            str(row["group"]), signed(float(row["mean_delta_unlearn"])),
            signed(float(row["mean_delta_preserve"])), signed(float(row["mean_delta_overall"])),
            f"{row['positive_preserve_seed_count']}/{row['total_seeds']}",
            f"{row['positive_overall_seed_count']}/{row['total_seeds']}",
        ]
        for row in group_robustness_rows
    ]
    retain_headers = ["Group", "Retain concept"]
    for seed in seeds:
        retain_headers.extend([f"Off {seed}", f"Match {seed}"])
    retain_headers.extend(["Mean Δ", "Improve seeds"])
    retain_table: list[list[str]] = []
    for row in retain_rows:
        rendered = [str(row["group"]), str(row["concept"])]
        for seed in seeds:
            rendered.extend([fmt(float(row[f"official_{seed}"])), fmt(float(row[f"matched_{seed}"]))])
        rendered.extend([signed(float(row["mean_delta"])), f"{row['improvement_seed_count']}/{row['total_seeds']}"])
        retain_table.append(rendered)
    diagnostic_table = [
        [
            str(row["edit_seed"]), fmt(float(row["mean_raw_mi_official"])),
            fmt(float(row["mean_raw_mi_matched"])), fmt(float(row["mean_alpha_spearman"])),
            fmt(float(row["mean_top_1_percent_overlap"])), fmt(float(row["mean_top_5_percent_overlap"])),
            fmt(float(row["mean_top_10_percent_overlap"])),
        ]
        for row in diagnostic_rows
    ]

    if args.profile == "formal":
        judgment_text = {
            "ROBUSTLY SUPPORTED": "matched-retain improvement is reproducible across Informax pseudo-sample randomness.",
            "SEED-SENSITIVE": "The previous positive result exists but is not sufficiently stable to support a robust method improvement.",
            "NOT SUPPORTED": "The previous result does not replicate across edit randomness.",
        }[judgment]
    else:
        judgment_text = "Smoke test only. No robustness judgment is permitted."
    if args.profile == "formal":
        reuse_text = "The legacy `20260820` image-level scores are reused after integrity validation; no legacy images are regenerated."
        denominator_text = "Formal denominator per variant/seed: 25 concepts × 120 images = 3,000 rows, comprising 1,200 target and 1,800 retain rows."
        new_seed_text = f" (`{new_positive_preserve}/4` among the newly generated seeds)"
    else:
        reuse_text = "This smoke run evaluates only seed `20260821` on two images for each dogs-group concept and does not import or judge the legacy seed."
        denominator_text = "Smoke denominator per variant: 5 dogs-group concepts × 2 images = 10 rows."
        new_seed_text = ""

    summary = f"""# ScaPre Informax Specificity: Edit-Seed Robustness

## Technical summary

This run varies only the Informax pseudo-sample random stream across the fixed seeds `{', '.join(str(seed) for seed in seeds)}`. {reuse_text} Mean ΔPreserve is `{mean_dp:+.2f}` points with `{preserve_count}/{len(seeds)}` positive seeds{new_seed_text}. Mean ΔUnlearn is `{mean_du:+.2f}` points with `{unlearn_count}/{len(seeds)}` seeds maintaining or improving target erasure, and mean ΔOverall is `{mean_do:+.2f}` with `{overall_count}/{len(seeds)}` positive seeds. The final judgment is **{judgment or 'SMOKE ONLY'}**.

## Seed-level image results

Delta is always `matched_retain - official`; therefore negative ΔU and positive ΔP/ΔOverall are improvements.

{markdown_table(['Edit Seed', 'Official U ↓', 'Matched U ↓', 'ΔU ↓', 'Official P ↑', 'Matched P ↑', 'ΔP ↑', 'ΔOverall ↑'], seed_table)}

## Effect magnitude and seed variance

Standard deviation is the sample standard deviation with an `n-1` denominator. No chart is used because the experiment has exactly five fixed observations and the exact values, signs, and extrema are more auditable in a table.

{markdown_table(['Metric', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Improving seeds'], aggregate_table)}

## Group-level robustness

Positive mean Preserve deltas occur in `{positive_mean_groups}/{len(group_order)}` groups. This table shows whether the effect is distributed across semantic groups rather than being driven by only the original seed or one group.

{markdown_table(['Group', 'Mean ΔU', 'Mean ΔP', 'Mean ΔOverall', 'ΔP > 0', 'ΔOverall > 0'], group_table)}

## Retain-concept robustness

All `{len(retain_rows)}` retain concepts are shown without selection. `{positive_mean_retains}/{len(retain_rows)}` have positive mean deltas and `{majority_positive_retains}/{len(retain_rows)}` improve in at least half of the evaluated seeds. The complete 25-concept × seed table, including all targets, is in `results/per_concept_seed.csv`.

{markdown_table(retain_headers, retain_table)}

## Experimental design and integrity

- Base method source: unchanged `scapre/edit/erase_scale.py` with SHA-256 `{robustness['source_controls']['scapre/edit/erase_scale.py']}`.
- Fixed non-Informax/global edit seed: `{robustness['fixed_non_informax_seed']}`.
- New seeds use the audited RNG wrapper: every legacy global Informax draw is still consumed to preserve all later non-Informax RNG positions, while only the tensor returned to Informax comes from the seed-specific stream.
- Generation prompts and generation seeds are identical across both variants and every edit seed.
- {denominator_text}
- Evaluator/classifier fingerprints are identical across all comparisons.
- Run commit: `{run_manifest['git_commit']}`; start and end working-tree status are both recorded in `run_manifest.json`.
- The protocol remains the project-established Confuse5 reconstruction, not an exact author-released Table 7 seed asset.

## Limitations and decision boundary

This is a deterministic five-seed robustness check over one fixed generation protocol, not an inferential population estimate or a hyperparameter sweep. Seed `20260820` comes from the legacy globally seeded execution; its controlled sources and raw outputs are verified, while the four new runs isolate Informax draws and preserve the legacy non-Informax RNG stream. Safety-checker substitutions remain part of the unchanged evaluator and can affect absolute classifier accuracy. No threshold, seed, group, concept, or method component is changed in response to observed results, and no new Informax formulation is proposed or implemented in this run.

## Informax mechanism diagnostic

These quantities are explanatory only and do not determine the judgment.

{markdown_table(['Edit Seed', 'MI official', 'MI matched', 'Alpha Spearman', 'Top 1% overlap', 'Top 5% overlap', 'Top 10% overlap'], diagnostic_table)}

## Final judgment

**{judgment or 'SMOKE ONLY'}**

{judgment_text}
"""
    (results_dir / "summary.md").write_text(summary)

    integrity = {
        "status": "passed",
        "profile": args.profile,
        "edit_seeds": seeds,
        "rows_per_variant_seed": 3000 if args.profile == "formal" else 10,
        "cross_seed_generation_keys_identical": True,
        "duplicate_generation_keys": 0,
        "evaluator_fingerprint_identical": True,
        "new_seed_source_hashes_identical": True,
        "legacy_controlled_source_hashes_verified": args.profile != "formal" or (run_dir / "reproducibility" / "prior_seed_validation.json").is_file(),
        "score_sha256": score_hashes,
        "start_git_clean": run_manifest["git_status_start"] == [],
        "end_git_clean": run_manifest.get("git_status_end") == [],
    }
    (run_dir / "reproducibility" / "integrity_report.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "reproducibility" / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n")
    (results_dir / "result_manifest.json").write_text(json.dumps({
        "profile": args.profile,
        "edit_seeds": seeds,
        "image_records": len(seeds) * (6000 if args.profile == "formal" else 20),
        "judgment": judgment,
        "preserve_improving_seed_count": preserve_count,
        "overall_improving_seed_count": overall_count,
        "mean_delta_unlearn": mean_du,
        "mean_delta_preserve": mean_dp,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
