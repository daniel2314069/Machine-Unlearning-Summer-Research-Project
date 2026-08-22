#!/usr/bin/env python
"""Validate and aggregate superclass-neutral against reused official scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "seed_robustness"))
from evaluator_fingerprint import compare_evaluator_manifests  # noqa: E402


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


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accuracy(rows: list[dict[str, str]]) -> float:
    if not rows:
        raise RuntimeError("cannot calculate accuracy over zero rows")
    return 100.0 * sum(int(row["correct"]) for row in rows) / len(rows)


def overall(unlearn: float, preserve: float) -> float:
    forgetting = 100.0 - unlearn
    return 0.0 if forgetting + preserve == 0 else 2.0 * forgetting * preserve / (forgetting + preserve)


def metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    unlearn = accuracy([row for row in rows if row["role"] == "target"])
    preserve = accuracy([row for row in rows if row["role"] == "retain"])
    return {"unlearn_acc": unlearn, "preserve_acc": preserve, "overall_acc": overall(unlearn, preserve)}


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "group", "role", "concept", "sample_index", "prompt", "seed", "seed_source"
    ))


def validate_rows(rows: list[dict[str, str]], base: dict, profile: str, variant: str) -> set[tuple[str, ...]]:
    groups = base["groups"][:1] if profile == "smoke" else base["groups"]
    per_concept = base["evaluation"]["smoke_images_per_concept" if profile == "smoke" else "formal_images_per_concept"]
    expected = {
        concept: (group["id"], role)
        for group in groups
        for role, field in (("target", "targets"), ("retain", "retains"))
        for concept in group[field]
    }
    if len(rows) != len(expected) * per_concept:
        raise RuntimeError(f"{variant} row count changed: {len(rows)}")
    keys: set[tuple[str, ...]] = set()
    counts: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    for row in rows:
        concept = row["concept"]
        if row["variant"] != variant or concept not in expected:
            raise RuntimeError(f"invalid {variant} row for {concept}")
        if (row["group"], row["role"]) != expected[concept] or row["correct"] not in {"0", "1"}:
            raise RuntimeError(f"invalid group/role/score for {concept}")
        controlled = key(row)
        if controlled in keys:
            raise RuntimeError(f"duplicate generation key in {variant}: {controlled}")
        keys.add(controlled)
        counts[concept] += 1
        roles[row["role"]] += 1
    if set(counts) != set(expected) or any(value != per_concept for value in counts.values()):
        raise RuntimeError(f"{variant} per-concept denominator changed")
    if profile == "formal" and (roles["target"], roles["retain"]) != (1200, 1800):
        raise RuntimeError(f"{variant} target/retain denominators changed")
    return keys


def fmt(value: float) -> str:
    return f"{value:.2f}"


def signed(value: float) -> str:
    return f"{value:+.2f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def diagnostic_summary(seed_dir: Path, official_csv: Path) -> dict[str, float]:
    payload = torch.load(seed_dir / "diagnostics" / "superclass_neutral.pt", map_location="cpu", weights_only=False)
    records = [
        row for row in payload["records"]
        if row["stage"] == "aggregate" and row["target_concept"] is not None
    ]
    official = read_csv(official_csv)
    if len(records) != len(official):
        raise RuntimeError("official/superclass diagnostic coverage differs")
    for row in records:
        negatives = row.get("negative_concepts")
        if not isinstance(negatives, list) or len(negatives) != 1:
            raise RuntimeError("superclass diagnostic does not contain one negative base")
    return {
        "records": float(len(records)),
        "mean_raw_mi_official": statistics.mean(float(row["mean_raw_mi_official"]) for row in official),
        "mean_raw_mi_superclass": statistics.mean(row["raw_mi"].double().mean().item() for row in records),
        "mean_alpha_superclass": statistics.mean(row["alpha"].double().mean().item() for row in records),
    }


def robustness_rows(
    concepts: list[str],
    role: str,
    per_concept: list[dict[str, object]],
    seeds: list[int],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for concept in concepts:
        selected = [row for row in per_concept if row["concept"] == concept]
        if len(selected) != len(seeds) or any(row["role"] != role for row in selected):
            raise RuntimeError(f"per-concept seed coverage changed for {concept}")
        result: dict[str, object] = {"group": selected[0]["group"], "concept": concept}
        for row in selected:
            seed = int(row["edit_seed"])
            result[f"official_{seed}"] = row["official_acc"]
            result[f"superclass_{seed}"] = row["superclass_acc"]
        deltas = [float(row["delta_acc"]) for row in selected]
        result["mean_delta"] = statistics.mean(deltas)
        result["improvement_seed_count"] = sum(
            value < 0 if role == "target" else value > 0 for value in deltas
        )
        result["total_seeds"] = len(deltas)
        output.append(result)
    return output


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    config = json.loads(args.config.read_text())
    base = json.loads(args.base_config.read_text())
    seeds = [20260821] if args.profile == "smoke" else config["edit_seeds"]
    groups = base["groups"][:1] if args.profile == "smoke" else base["groups"]
    concepts = [concept for group in groups for field in ("targets", "retains") for concept in group[field]]
    roles = {concept: role for group in groups for role, field in (("target", "targets"), ("retain", "retains")) for concept in group[field]}
    concept_groups = {concept: group["id"] for group in groups for field in ("targets", "retains") for concept in group[field]}

    if args.profile == "smoke":
        score = run_dir / "seeds" / "20260821" / "evaluation" / "superclass_neutral" / "scores.csv"
        rows = read_csv(score)
        validate_rows(rows, base, "smoke", "superclass_neutral")
        block = metrics(rows)
        write_csv(results_dir / "per_seed.csv", [{"edit_seed": 20260821, **{f"superclass_{k}": v for k, v in block.items()}}])
        (results_dir / "summary.md").write_text(
            "# Superclass-neutral smoke test\n\n"
            "The superclass-neutral edit and unchanged evaluator completed for the dogs group "
            "with 10 image records. This smoke run is a code-path check only; it reuses no "
            "baseline and permits no scientific conclusion.\n"
        )
        repro = run_dir / "reproducibility"
        repro.mkdir(exist_ok=True)
        (repro / "integrity_report.json").write_text(json.dumps({
            "status": "passed", "profile": "smoke", "score_rows": 10,
            "duplicate_generation_keys": 0, "scientific_judgment": None,
        }, indent=2) + "\n")
        (results_dir / "result_manifest.json").write_text(json.dumps({
            "profile": "smoke", "new_image_records": 10, "judgment": None,
        }, indent=2) + "\n")
        return

    per_seed: list[dict[str, object]] = []
    per_group: list[dict[str, object]] = []
    per_concept: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    evaluator_observations: list[tuple[str, dict]] = []
    score_hashes: dict[str, dict[str, str]] = {}
    reference_keys: set[tuple[str, ...]] | None = None
    for edit_seed in seeds:
        sources = {
            "official": run_dir / "baselines" / str(edit_seed) / "official",
            "matched_retain": run_dir / "baselines" / str(edit_seed) / "matched_retain",
            "superclass_neutral": run_dir / "seeds" / str(edit_seed) / "evaluation" / "superclass_neutral",
        }
        by_variant: dict[str, list[dict[str, str]]] = {}
        seed_keys: set[tuple[str, ...]] | None = None
        score_hashes[str(edit_seed)] = {}
        for variant, directory in sources.items():
            rows = read_csv(directory / "scores.csv")
            keys = validate_rows(rows, base, "formal", variant)
            if seed_keys is None:
                seed_keys = keys
            elif keys != seed_keys:
                raise RuntimeError(f"seed {edit_seed}: generation keys differ across variants")
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise RuntimeError(f"seed {edit_seed}: generation keys differ across edit seeds")
            by_variant[variant] = rows
            score_hashes[str(edit_seed)][variant] = sha256(directory / "scores.csv")
            evaluator_observations.append((
                f"seed={edit_seed},variant={variant}",
                json.loads((directory / "evaluation_manifest.json").read_text()),
            ))

        blocks = {variant: metrics(rows) for variant, rows in by_variant.items()}
        row: dict[str, object] = {"edit_seed": edit_seed}
        for metric in ("unlearn_acc", "preserve_acc", "overall_acc"):
            row[f"official_{metric}"] = blocks["official"][metric]
            row[f"superclass_{metric}"] = blocks["superclass_neutral"][metric]
            row[f"delta_{metric}"] = blocks["superclass_neutral"][metric] - blocks["official"][metric]
            row[f"matched_{metric}"] = blocks["matched_retain"][metric]
            row[f"superclass_vs_matched_{metric}"] = blocks["superclass_neutral"][metric] - blocks["matched_retain"][metric]
        per_seed.append(row)

        for group in [item["id"] for item in groups]:
            group_blocks = {
                variant: metrics([item for item in rows if item["group"] == group])
                for variant, rows in by_variant.items()
            }
            group_row: dict[str, object] = {"edit_seed": edit_seed, "group": group}
            for metric in ("unlearn_acc", "preserve_acc", "overall_acc"):
                group_row[f"official_{metric}"] = group_blocks["official"][metric]
                group_row[f"superclass_{metric}"] = group_blocks["superclass_neutral"][metric]
                group_row[f"delta_{metric}"] = group_blocks["superclass_neutral"][metric] - group_blocks["official"][metric]
            per_group.append(group_row)

        for concept in concepts:
            official_acc = accuracy([item for item in by_variant["official"] if item["concept"] == concept])
            superclass_acc = accuracy([item for item in by_variant["superclass_neutral"] if item["concept"] == concept])
            matched_acc = accuracy([item for item in by_variant["matched_retain"] if item["concept"] == concept])
            per_concept.append({
                "edit_seed": edit_seed, "group": concept_groups[concept], "role": roles[concept],
                "concept": concept, "official_acc": official_acc,
                "superclass_acc": superclass_acc, "delta_acc": superclass_acc - official_acc,
                "matched_acc": matched_acc, "superclass_vs_matched_delta": superclass_acc - matched_acc,
            })
        diagnostic = diagnostic_summary(
            run_dir / "seeds" / str(edit_seed),
            run_dir / "baselines" / str(edit_seed) / "informax_diagnostics.csv",
        )
        diagnostics.append({"edit_seed": edit_seed, **diagnostic})

    evaluator_comparison = compare_evaluator_manifests(evaluator_observations)
    write_csv(results_dir / "per_seed.csv", per_seed)
    write_csv(results_dir / "per_group_seed.csv", per_group)
    write_csv(results_dir / "per_concept_seed.csv", per_concept)
    write_csv(results_dir / "informax_seed_diagnostics.csv", diagnostics)

    delta_specs = [
        ("delta_unlearn", "delta_unlearn_acc", lambda value: value <= 0, "<= 0"),
        ("delta_preserve", "delta_preserve_acc", lambda value: value > 0, "> 0"),
        ("delta_overall", "delta_overall_acc", lambda value: value > 0, "> 0"),
    ]
    aggregate: list[dict[str, object]] = []
    for label, field, improves, definition in delta_specs:
        values = [float(row[field]) for row in per_seed]
        aggregate.append({
            "metric": label, "mean": statistics.mean(values),
            "std": statistics.stdev(values), "median": statistics.median(values),
            "min": min(values), "max": max(values),
            "improving_seed_count": sum(improves(value) for value in values),
            "total_seeds": len(values), "improvement_definition": definition,
        })
    write_csv(results_dir / "aggregate_across_seeds.csv", aggregate)

    group_robustness: list[dict[str, object]] = []
    for group in [item["id"] for item in groups]:
        selected = [row for row in per_group if row["group"] == group]
        group_robustness.append({
            "group": group,
            "mean_delta_unlearn": statistics.mean(float(row["delta_unlearn_acc"]) for row in selected),
            "mean_delta_preserve": statistics.mean(float(row["delta_preserve_acc"]) for row in selected),
            "mean_delta_overall": statistics.mean(float(row["delta_overall_acc"]) for row in selected),
            "positive_preserve_seed_count": sum(float(row["delta_preserve_acc"]) > 0 for row in selected),
            "positive_overall_seed_count": sum(float(row["delta_overall_acc"]) > 0 for row in selected),
            "total_seeds": len(selected),
        })
    write_csv(results_dir / "per_group_robustness.csv", group_robustness)
    targets = [concept for concept in concepts if roles[concept] == "target"]
    retains = [concept for concept in concepts if roles[concept] == "retain"]
    target_robustness = robustness_rows(targets, "target", per_concept, seeds)
    retain_robustness = robustness_rows(retains, "retain", per_concept, seeds)
    write_csv(results_dir / "per_target_robustness.csv", target_robustness)
    write_csv(results_dir / "per_retain_robustness.csv", retain_robustness)

    mean_du, mean_dp, mean_do = (float(row["mean"]) for row in aggregate)
    unlearn_count, preserve_count, overall_count = (int(row["improving_seed_count"]) for row in aggregate)
    rule = config["decision_rule"]
    supported = (
        preserve_count >= rule["minimum_positive_preserve_seeds"]
        and mean_dp >= rule["minimum_mean_preserve_delta_pp"]
        and mean_du <= rule["maximum_mean_unlearn_delta_pp"]
        and overall_count >= rule["minimum_positive_overall_seeds"]
    )
    trade_off = (
        preserve_count >= rule["minimum_positive_preserve_seeds"]
        and mean_dp > 0
        and mean_du > rule["material_mean_unlearn_degradation_pp"]
    )
    judgment = "SUPPORTED" if supported else ("TRADE-OFF ONLY" if trade_off else "NOT SUPPORTED")

    seed_table = [[
        str(row["edit_seed"]), fmt(float(row["official_unlearn_acc"])),
        fmt(float(row["superclass_unlearn_acc"])), signed(float(row["delta_unlearn_acc"])),
        fmt(float(row["official_preserve_acc"])), fmt(float(row["superclass_preserve_acc"])),
        signed(float(row["delta_preserve_acc"])), signed(float(row["delta_overall_acc"])),
    ] for row in per_seed]
    aggregate_table = [[
        str(row["metric"]), fmt(float(row["mean"])), fmt(float(row["std"])),
        fmt(float(row["median"])), fmt(float(row["min"])), fmt(float(row["max"])),
        f"{row['improving_seed_count']}/{row['total_seeds']}",
    ] for row in aggregate]
    group_table = [[
        str(row["group"]), signed(float(row["mean_delta_unlearn"])),
        signed(float(row["mean_delta_preserve"])), signed(float(row["mean_delta_overall"])),
        f"{row['positive_preserve_seed_count']}/{row['total_seeds']}",
        f"{row['positive_overall_seed_count']}/{row['total_seeds']}",
    ] for row in group_robustness]
    matched_comparison = []
    for metric, direction in (("unlearn_acc", "↓"), ("preserve_acc", "↑"), ("overall_acc", "↑")):
        off = statistics.mean(float(row[f"official_{metric}"]) for row in per_seed)
        matched = statistics.mean(float(row[f"matched_{metric}"]) for row in per_seed)
        superclass = statistics.mean(float(row[f"superclass_{metric}"]) for row in per_seed)
        matched_comparison.append([metric.replace("_acc", ""), direction, fmt(off), fmt(matched), fmt(superclass)])
    positive_groups = sum(float(row["mean_delta_preserve"]) > 0 for row in group_robustness)
    positive_targets = sum(float(row["mean_delta"]) < 0 for row in target_robustness)
    positive_retains = sum(float(row["mean_delta"]) > 0 for row in retain_robustness)
    majority_retains = sum(int(row["improvement_seed_count"]) >= 3 for row in retain_robustness)
    target_table = [[
        str(row["group"]), str(row["concept"]), signed(float(row["mean_delta"])),
        f"{row['improvement_seed_count']}/{row['total_seeds']}",
    ] for row in target_robustness]
    retain_table = [[
        str(row["group"]), str(row["concept"]), signed(float(row["mean_delta"])),
        f"{row['improvement_seed_count']}/{row['total_seeds']}",
    ] for row in retain_robustness]
    qualitative_manifest = run_dir / "qualitative" / "manifest.csv"
    qualitative_rows = read_csv(qualitative_manifest)
    if len(qualitative_rows) != 90 or not (run_dir / "qualitative" / "COMPLETED").is_file():
        raise RuntimeError("formal qualitative comparison set is incomplete")

    judgment_text = {
        "SUPPORTED": "Superclass-neutral gives a stable image-level advantage over official under the frozen rule.",
        "TRADE-OFF ONLY": "Preservation rises, but mainly with materially weaker target erasure.",
        "NOT SUPPORTED": "Superclass-neutral does not show a stable, non-trade-off image-level advantage over official.",
    }[judgment]
    summary = f"""# ScaPre Informax superclass-neutral

## Technical summary

This five-edit-seed experiment changes only the Informax negative base from the empty prompt to the target superclass. Verified official score rows are reused; no 3,000-row official baseline evaluation is rerun. Mean ΔPreserve is `{mean_dp:+.2f}` pp, mean ΔUnlearn is `{mean_du:+.2f}` pp, and mean ΔOverall is `{mean_do:+.2f}` pp. The frozen decision gives **{judgment}**.

## Official versus superclass-neutral

Delta is `superclass_neutral - official`; negative ΔUnlearn and positive ΔPreserve/ΔOverall are improvements.

{table(['Edit Seed', 'Official U ↓', 'Superclass U ↓', 'ΔU ↓', 'Official P ↑', 'Superclass P ↑', 'ΔP ↑', 'ΔOverall ↑'], seed_table)}

## Across-seed stability

{table(['Metric', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Improving seeds'], aggregate_table)}

## Group distribution

`{positive_groups}/5` groups have positive mean preservation delta.

{table(['Group', 'Mean ΔU', 'Mean ΔP', 'Mean ΔOverall', 'ΔP > 0', 'ΔOverall > 0'], group_table)}

## Target and retain distribution

`{positive_targets}/10` targets have lower mean residual accuracy and `{positive_retains}/15` retains have positive mean preservation delta; `{majority_retains}/15` retains improve in at least three seeds. Complete unselected results are in `per_concept_seed.csv`, `per_target_robustness.csv`, and `per_retain_robustness.csv`.

{table(['Group', 'Target', 'Mean Δ accuracy', 'Erasure-improving seeds'], target_table)}

{table(['Group', 'Retain', 'Mean Δ accuracy', 'Preserve-improving seeds'], retain_table)}

## Context against matched-retain

This table is descriptive only; matched-retain scores are reused from the completed robustness experiment.

{table(['Metric', 'Direction', 'Official mean', 'Matched mean', 'Superclass mean'], matched_comparison)}

## Informax diagnostic

`informax_seed_diagnostics.csv` reports raw official and superclass MI summaries. These mechanism values do not determine success.

## Reproducibility and qualitative set

- Edit seeds: `{', '.join(str(seed) for seed in seeds)}`; global non-Informax seed remains `20260820`.
- Five positive and five negative pseudo-samples are retained; superclass mode uses five noisy copies of exactly one mapped superclass embedding.
- All 15,000 new superclass score rows have the same 25 concepts × 120 prompts/seeds as official and matched-retain.
- Evaluator/classifier fingerprints are substantively identical across all 15 seed/variant observations.
- The predeclared qualitative set contains 90 images plus 30 side-by-side panels: both targets and one retain per group, sample indices 0 and 1, identical prompt and generation seed across all three variants.
- Official/matched qualitative pictures are the only baseline images regenerated. Their 60 predictions are rechecked against the recorded formal rows; no full baseline evaluation or metric recomputation is performed.
- Run commit: `{json.loads((run_dir / 'run_manifest.json').read_text())['git_commit']}`.

## Final answer

**{judgment}**

{judgment_text}
"""
    (results_dir / "summary.md").write_text(summary)

    integrity = {
        "status": "passed", "profile": "formal", "edit_seeds": seeds,
        "new_superclass_score_rows": 15000, "reused_official_score_rows": 15000,
        "reused_matched_score_rows": 15000, "baseline_score_evaluations_rerun": False,
        "rows_per_variant_seed": 3000, "target_rows_per_variant_seed": 1200,
        "retain_rows_per_variant_seed": 1800, "duplicate_generation_keys": 0,
        "cross_variant_generation_keys_identical": True,
        "cross_seed_generation_keys_identical": True,
        "evaluator_comparison": evaluator_comparison,
        "score_sha256": score_hashes, "qualitative_images": len(qualitative_rows),
        "qualitative_comparison_panels": 30,
        "qualitative_selection_predeclared": True,
        "start_git_clean": json.loads((run_dir / "run_manifest.json").read_text())["git_status_start"] == [],
        "end_git_clean": json.loads((run_dir / "run_manifest.json").read_text())["git_status_end"] == [],
    }
    repro = run_dir / "reproducibility"
    repro.mkdir(exist_ok=True)
    (repro / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n")
    (results_dir / "result_manifest.json").write_text(json.dumps({
        "profile": "formal", "edit_seeds": seeds, "new_image_records": 15000,
        "judgment": judgment, "mean_delta_unlearn": mean_du,
        "mean_delta_preserve": mean_dp, "mean_delta_overall": mean_do,
        "preserve_improving_seed_count": preserve_count,
        "overall_improving_seed_count": overall_count,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
