#!/usr/bin/env python3
"""Fail-closed aggregation for the five-seed projection accumulation study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


SEEDS = [20260820, 20260821, 20260822, 20260823, 20260824]
VARIANTS = ["official", "projection_accumulation"]


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
    values = [float(row["correct"]) for row in rows]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("score contains NaN/Inf")
    if any(value not in {0.0, 1.0} for value in values):
        raise RuntimeError("score correctness value is not binary")
    return 100.0 * sum(values) / len(values)


def overall(unlearn: float, preserve: float) -> float:
    retained_erasure = 100.0 - unlearn
    denominator = retained_erasure + preserve
    return 0.0 if denominator == 0 else 2.0 * retained_erasure * preserve / denominator


def generation_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["group"], row["role"], row["concept"], row["sample_index"],
        row["prompt"], row["seed"], row["seed_source"],
    )


def metric_triplet(rows: list[dict[str, str]]) -> dict[str, float]:
    target = [row for row in rows if row["role"] == "target"]
    retain = [row for row in rows if row["role"] == "retain"]
    u = accuracy(target)
    p = accuracy(retain)
    return {"unlearn": u, "preserve": p, "overall": overall(u, p)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    results = run_dir / "results"
    results.mkdir(exist_ok=True)

    scores: dict[tuple[int, str], list[dict[str, str]]] = {}
    expected_keys: list[tuple[str, ...]] | None = None
    for seed in SEEDS:
        for variant in VARIANTS:
            path = run_dir / "seeds" / str(seed) / "evaluation" / variant / "scores.csv"
            rows = read_csv(path)
            if len(rows) != 3000:
                raise RuntimeError(f"{seed}/{variant} has {len(rows)} rows, expected 3000")
            if Counter(row["role"] for row in rows) != {"target": 1200, "retain": 1800}:
                raise RuntimeError(f"{seed}/{variant} target/retain counts changed")
            if any(row["variant"] != variant for row in rows):
                raise RuntimeError(f"{seed}/{variant} has an incorrect variant label")
            keys = [generation_key(row) for row in rows]
            if len(set(keys)) != 3000:
                raise RuntimeError(f"{seed}/{variant} contains duplicate generation keys")
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise RuntimeError(f"{seed}/{variant} generation key order differs")
            scores[(seed, variant)] = rows

    per_seed: list[dict[str, object]] = []
    metric_by_seed: dict[tuple[int, str], dict[str, float]] = {}
    deltas: list[dict[str, object]] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            metrics = metric_triplet(scores[(seed, variant)])
            metric_by_seed[(seed, variant)] = metrics
            per_seed.append({
                "edit_seed": seed,
                "variant": variant,
                "unlearn_accuracy": metrics["unlearn"],
                "preserve_accuracy": metrics["preserve"],
                "overall_accuracy": metrics["overall"],
            })
        official = metric_by_seed[(seed, "official")]
        treatment = metric_by_seed[(seed, "projection_accumulation")]
        deltas.append({
            "scope": "seed",
            "identifier": seed,
            "delta_unlearn": treatment["unlearn"] - official["unlearn"],
            "delta_preserve": treatment["preserve"] - official["preserve"],
            "delta_overall": treatment["overall"] - official["overall"],
        })

    per_target: list[dict[str, object]] = []
    concept_summary: list[dict[str, object]] = []
    concepts = sorted({row["concept"] for row in scores[(SEEDS[0], VARIANTS[0])]}, key=lambda value: next(
        index for index, row in enumerate(scores[(SEEDS[0], VARIANTS[0])]) if row["concept"] == value
    ))
    for concept in concepts:
        seed_deltas: list[float] = []
        group = ""
        role = ""
        for seed in SEEDS:
            official_rows = [row for row in scores[(seed, "official")] if row["concept"] == concept]
            treatment_rows = [row for row in scores[(seed, "projection_accumulation")] if row["concept"] == concept]
            if len(official_rows) != 120 or len(treatment_rows) != 120:
                raise RuntimeError(f"{concept}/{seed} does not have 120 rows per variant")
            group, role = official_rows[0]["group"], official_rows[0]["role"]
            official_accuracy = accuracy(official_rows)
            treatment_accuracy = accuracy(treatment_rows)
            delta = treatment_accuracy - official_accuracy
            seed_deltas.append(delta)
            per_target.append({
                "edit_seed": seed,
                "group": group,
                "role": role,
                "concept": concept,
                "official_accuracy": official_accuracy,
                "projection_accumulation_accuracy": treatment_accuracy,
                "delta_accuracy": delta,
                "favorable": delta < 0 if role == "target" else delta > 0,
            })
        concept_summary.append({
            "edit_seed": "mean",
            "group": group,
            "role": role,
            "concept": concept,
            "official_accuracy": mean(
                float(row["official_accuracy"]) for row in per_target if row["concept"] == concept
            ),
            "projection_accumulation_accuracy": mean(
                float(row["projection_accumulation_accuracy"]) for row in per_target if row["concept"] == concept
            ),
            "delta_accuracy": mean(seed_deltas),
            "favorable": sum(delta < 0 for delta in seed_deltas) if role == "target" else sum(delta > 0 for delta in seed_deltas),
        })
    per_target.extend(concept_summary)

    group_rows: list[dict[str, object]] = []
    groups = list(dict.fromkeys(row["group"] for row in scores[(SEEDS[0], VARIANTS[0])]))
    for group in groups:
        group_seed_deltas: list[dict[str, float]] = []
        for seed in SEEDS:
            variant_metrics: dict[str, dict[str, float]] = {}
            for variant in VARIANTS:
                selected = [row for row in scores[(seed, variant)] if row["group"] == group]
                variant_metrics[variant] = metric_triplet(selected)
            row = {
                "edit_seed": seed,
                "group": group,
                "official_unlearn": variant_metrics["official"]["unlearn"],
                "projection_unlearn": variant_metrics["projection_accumulation"]["unlearn"],
                "delta_unlearn": variant_metrics["projection_accumulation"]["unlearn"] - variant_metrics["official"]["unlearn"],
                "official_preserve": variant_metrics["official"]["preserve"],
                "projection_preserve": variant_metrics["projection_accumulation"]["preserve"],
                "delta_preserve": variant_metrics["projection_accumulation"]["preserve"] - variant_metrics["official"]["preserve"],
                "official_overall": variant_metrics["official"]["overall"],
                "projection_overall": variant_metrics["projection_accumulation"]["overall"],
                "delta_overall": variant_metrics["projection_accumulation"]["overall"] - variant_metrics["official"]["overall"],
            }
            group_rows.append(row)
            group_seed_deltas.append(row)  # type: ignore[arg-type]
        group_rows.append({
            "edit_seed": "mean",
            "group": group,
            **{
                field: mean(float(row[field]) for row in group_seed_deltas)
                for field in (
                    "official_unlearn", "projection_unlearn", "delta_unlearn",
                    "official_preserve", "projection_preserve", "delta_preserve",
                    "official_overall", "projection_overall", "delta_overall",
                )
            },
        })

    official_mean = {
        metric: mean(metric_by_seed[(seed, "official")][metric] for seed in SEEDS)
        for metric in ("unlearn", "preserve", "overall")
    }
    treatment_mean = {
        metric: mean(metric_by_seed[(seed, "projection_accumulation")][metric] for seed in SEEDS)
        for metric in ("unlearn", "preserve", "overall")
    }
    mean_delta = {metric: treatment_mean[metric] - official_mean[metric] for metric in official_mean}
    favorable = {
        "unlearn": sum(
            metric_by_seed[(seed, "projection_accumulation")]["unlearn"]
            < metric_by_seed[(seed, "official")]["unlearn"] for seed in SEEDS
        ),
        "preserve": sum(
            metric_by_seed[(seed, "projection_accumulation")]["preserve"]
            > metric_by_seed[(seed, "official")]["preserve"] for seed in SEEDS
        ),
        "overall": sum(
            metric_by_seed[(seed, "projection_accumulation")]["overall"]
            > metric_by_seed[(seed, "official")]["overall"] for seed in SEEDS
        ),
    }
    directional = {
        "mean_delta_unlearn_negative": mean_delta["unlearn"] < 0,
        "mean_delta_preserve_positive": mean_delta["preserve"] > 0,
        "mean_delta_overall_positive": mean_delta["overall"] > 0,
        "overall_favorable_at_least_4_of_5": favorable["overall"] >= 4,
        "group_target_pattern_requires_manual_review": True,
    }
    directional["automatic_directional_conditions_passed"] = all(
        directional[key] for key in (
            "mean_delta_unlearn_negative", "mean_delta_preserve_positive",
            "mean_delta_overall_positive", "overall_favorable_at_least_4_of_5",
        )
    )

    deltas.append({
        "scope": "five_seed_mean",
        "identifier": "projection_accumulation-minus-official",
        "delta_unlearn": mean_delta["unlearn"],
        "delta_preserve": mean_delta["preserve"],
        "delta_overall": mean_delta["overall"],
    })
    write_csv(results / "per_seed_metrics.csv", per_seed,
              ["edit_seed", "variant", "unlearn_accuracy", "preserve_accuracy", "overall_accuracy"])
    write_csv(results / "per_target_metrics.csv", per_target,
              ["edit_seed", "group", "role", "concept", "official_accuracy", "projection_accumulation_accuracy", "delta_accuracy", "favorable"])
    write_csv(results / "comparison_deltas.csv", deltas,
              ["scope", "identifier", "delta_unlearn", "delta_preserve", "delta_overall"])
    write_csv(results / "per_group_metrics.csv", group_rows,
              ["edit_seed", "group", "official_unlearn", "projection_unlearn", "delta_unlearn",
               "official_preserve", "projection_preserve", "delta_preserve",
               "official_overall", "projection_overall", "delta_overall"])

    payload = {
        "official_five_seed_mean": official_mean,
        "projection_accumulation_five_seed_mean": treatment_mean,
        "treatment_minus_official": mean_delta,
        "favorable_seeds": favorable,
        "directional_conditions": directional,
    }
    (results / "aggregate_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# ScaPre projection accumulation — Confuse5", "",
        "Delta is projection_accumulation minus official. Lower Unlearn and higher Preserve/Overall are favorable.", "",
        "| Variant | Unlearn ↓ | Preserve ↑ | Overall ↑ |", "| --- | ---: | ---: | ---: |",
        f"| official five-seed mean | {official_mean['unlearn']:.4f} | {official_mean['preserve']:.4f} | {official_mean['overall']:.4f} |",
        f"| projection_accumulation five-seed mean | {treatment_mean['unlearn']:.4f} | {treatment_mean['preserve']:.4f} | {treatment_mean['overall']:.4f} |",
        f"| treatment - official | {mean_delta['unlearn']:+.4f} | {mean_delta['preserve']:+.4f} | {mean_delta['overall']:+.4f} |", "",
        f"Favorable seeds: Unlearn {favorable['unlearn']}/5; Preserve {favorable['preserve']}/5; Overall {favorable['overall']}/5.", "",
        "Automatic directional conditions: " + ("PASS" if directional["automatic_directional_conditions_passed"] else "FAIL"),
        "The per-group/per-target concentration condition is intentionally left for manual review; no numerical cutoff was invented.", "",
        "## Group mean deltas", "", "| Group | ΔUnlearn | ΔPreserve | ΔOverall |", "| --- | ---: | ---: | ---: |",
    ]
    for row in group_rows:
        if row["edit_seed"] == "mean":
            lines.append(f"| {row['group']} | {float(row['delta_unlearn']):+.4f} | {float(row['delta_preserve']):+.4f} | {float(row['delta_overall']):+.4f} |")
    lines.extend(["", "## All concept mean deltas", "", "| Group | Role | Concept | Mean Δaccuracy | Favorable seeds |", "| --- | --- | --- | ---: | ---: |"])
    for row in concept_summary:
        lines.append(f"| {row['group']} | {row['role']} | {row['concept']} | {float(row['delta_accuracy']):+.4f} | {row['favorable']}/5 |")
    (results / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
