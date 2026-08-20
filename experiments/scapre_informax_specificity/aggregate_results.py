#!/usr/bin/env python3
"""Validate paired outputs, aggregate official metrics, and render summary.md."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
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
    return 100.0 * sum(int(row["correct"]) for row in rows) / len(rows)


def overall(unlearn: float, preserve: float) -> float:
    forgetting = 100.0 - unlearn
    denominator = forgetting + preserve
    return 0.0 if denominator == 0 else 2.0 * forgetting * preserve / denominator


def metric_block(rows: list[dict[str, str]]) -> dict[str, float]:
    targets = [row for row in rows if row["role"] == "target"]
    retains = [row for row in rows if row["role"] == "retain"]
    unlearn = accuracy(targets)
    preserve = accuracy(retains)
    return {
        "unlearn_acc": unlearn,
        "preserve_acc": preserve,
        "overall_acc": overall(unlearn, preserve),
    }


def validate_pair(official: list[dict[str, str]], matched: list[dict[str, str]]) -> None:
    def keyed(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
        result = {}
        for row in rows:
            key = (row["concept"], int(row["sample_index"]))
            if key in result:
                raise RuntimeError(f"duplicate evaluation row: {key}")
            result[key] = row
        return result

    left, right = keyed(official), keyed(matched)
    if left.keys() != right.keys():
        raise RuntimeError("official and matched variants evaluated different sample sets")
    controlled = ("group", "role", "concept", "sample_index", "prompt", "seed", "seed_source")
    for key in left:
        for field in controlled:
            if left[key][field] != right[key][field]:
                raise RuntimeError(f"paired protocol mismatch at {key}: {field}")


def validate_expected_rows(
    rows: list[dict[str, str]], config: dict, profile: str, variant: str
) -> None:
    groups = config["groups"][:1] if profile == "smoke" else config["groups"]
    count = config["evaluation"][
        "smoke_images_per_concept"
        if profile == "smoke"
        else "formal_images_per_concept"
    ]
    expected: dict[str, tuple[str, str]] = {}
    for group in groups:
        for role, field in (("target", "targets"), ("retain", "retains")):
            for concept in group[field]:
                expected[concept] = (group["id"], role)

    if len(rows) != len(expected) * count:
        raise RuntimeError(
            f"{variant} has {len(rows)} rows; expected {len(expected) * count}"
        )
    by_concept: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["variant"] != variant:
            raise RuntimeError(f"variant column mismatch in {variant} scores")
        concept = row["concept"]
        if concept not in expected:
            raise RuntimeError(f"unexpected concept in {variant} scores: {concept}")
        group, role = expected[concept]
        if row["group"] != group or row["role"] != role:
            raise RuntimeError(f"group/role mismatch for {concept} in {variant}")
        if row["correct"] not in {"0", "1"}:
            raise RuntimeError(f"non-binary classifier result for {concept} in {variant}")
        by_concept.setdefault(concept, []).append(row)

    if set(by_concept) != set(expected):
        raise RuntimeError(f"{variant} concept set is incomplete")
    for concept, concept_rows in by_concept.items():
        indices = sorted(int(row["sample_index"]) for row in concept_rows)
        if indices != list(range(count)):
            raise RuntimeError(
                f"{variant} has incomplete or duplicate sample indices for {concept}"
            )


def load_diagnostics(path: Path) -> dict[tuple[str, int, str], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = {}
    for record in payload["records"]:
        if record["stage"] != "aggregate" or record["target_concept"] is None:
            continue
        key = (record["projection"], int(record["layer_index"]), record["target_concept"])
        records[key] = record
    return records


def diagnostic_comparison(run_dir: Path, results_dir: Path) -> list[dict[str, object]]:
    official = load_diagnostics(run_dir / "diagnostics" / "official.pt")
    matched = load_diagnostics(run_dir / "diagnostics" / "matched_retain.pt")
    if official.keys() != matched.keys():
        raise RuntimeError("Informax diagnostic keys differ between variants")
    rows = []
    top_channels = {"official": {}, "matched_retain": {}}
    for key in sorted(official):
        projection, layer_index, target = key
        alpha_official = official[key]["alpha"].double().flatten()
        alpha_matched = matched[key]["alpha"].double().flatten()
        raw_official = official[key]["raw_mi"].double().flatten()
        raw_matched = matched[key]["raw_mi"].double().flatten()
        if alpha_official.numel() != alpha_matched.numel():
            raise RuntimeError(f"channel count mismatch at {key}")
        correlation = float(spearmanr(alpha_official.numpy(), alpha_matched.numpy()).statistic)
        if math.isnan(correlation):
            correlation = 0.0
        base = {
            "projection": projection,
            "layer_index": layer_index,
            "target_concept": target,
            "channels": alpha_official.numel(),
            "spearman_alpha": correlation,
            "mean_raw_mi_official": raw_official.mean().item(),
            "mean_raw_mi_matched": raw_matched.mean().item(),
        }
        channel_key = f"{projection}.layer_{layer_index:02d}.{target}"
        for label, tensor in (("official", alpha_official), ("matched_retain", alpha_matched)):
            top_channels[label][channel_key] = {}
            for percent in (1, 5, 10):
                count = max(1, math.ceil(tensor.numel() * percent / 100.0))
                top_channels[label][channel_key][f"top_{percent}_percent"] = (
                    torch.topk(tensor, count).indices.tolist()
                )
        for percent in (1, 5, 10):
            a = set(top_channels["official"][channel_key][f"top_{percent}_percent"])
            b = set(top_channels["matched_retain"][channel_key][f"top_{percent}_percent"])
            base[f"top_{percent}_percent_overlap"] = len(a & b) / len(a)
        rows.append(base)

    write_csv(
        results_dir / "informax_diagnostics.csv",
        rows,
        [
            "projection", "layer_index", "target_concept", "channels",
            "spearman_alpha", "mean_raw_mi_official", "mean_raw_mi_matched",
            "top_1_percent_overlap", "top_5_percent_overlap", "top_10_percent_overlap",
        ],
    )
    (results_dir / "top_channels.json").write_text(json.dumps(top_channels, indent=2) + "\n")
    return rows


def fmt(value: float) -> str:
    return f"{value:.2f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    results_dir = args.run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    by_variant = {
        variant: read_csv(args.run_dir / "evaluation" / variant / "scores.csv")
        for variant in ("official", "matched_retain")
    }
    for variant, rows in by_variant.items():
        validate_expected_rows(rows, config, args.profile, variant)
    validate_pair(by_variant["official"], by_variant["matched_retain"])

    group_order = [group["id"] for group in config["groups"]]
    if args.profile == "smoke":
        group_order = group_order[:1]
    concept_order = []
    concept_roles = {}
    concept_groups = {}
    for group in config["groups"]:
        if group["id"] not in group_order:
            continue
        for role, field in (("target", "targets"), ("retain", "retains")):
            for concept in group[field]:
                concept_order.append(concept)
                concept_roles[concept] = role
                concept_groups[concept] = group["id"]

    concept_rows = []
    for concept in concept_order:
        values = {}
        for variant, rows in by_variant.items():
            selected = [row for row in rows if row["concept"] == concept]
            values[variant] = accuracy(selected)
        concept_rows.append({
            "group": concept_groups[concept],
            "role": concept_roles[concept],
            "concept": concept,
            "official": values["official"],
            "matched_retain": values["matched_retain"],
            "delta": values["matched_retain"] - values["official"],
        })
    write_csv(
        results_dir / "per_concept.csv", concept_rows,
        ["group", "role", "concept", "official", "matched_retain", "delta"],
    )

    group_rows = []
    for group in group_order:
        values = {}
        for variant, rows in by_variant.items():
            values[variant] = metric_block([row for row in rows if row["group"] == group])
        group_rows.append({
            "group": group,
            "official_unlearn_acc": values["official"]["unlearn_acc"],
            "matched_unlearn_acc": values["matched_retain"]["unlearn_acc"],
            "delta_unlearn_acc": values["matched_retain"]["unlearn_acc"] - values["official"]["unlearn_acc"],
            "official_preserve_acc": values["official"]["preserve_acc"],
            "matched_preserve_acc": values["matched_retain"]["preserve_acc"],
            "delta_preserve_acc": values["matched_retain"]["preserve_acc"] - values["official"]["preserve_acc"],
            "official_overall_acc": values["official"]["overall_acc"],
            "matched_overall_acc": values["matched_retain"]["overall_acc"],
            "delta_overall_acc": values["matched_retain"]["overall_acc"] - values["official"]["overall_acc"],
        })
    group_fields = list(group_rows[0].keys())
    write_csv(results_dir / "per_group.csv", group_rows, group_fields)

    aggregate = {variant: metric_block(rows) for variant, rows in by_variant.items()}
    aggregate_rows = []
    for variant in ("official", "matched_retain"):
        aggregate_rows.append({"variant": variant, **aggregate[variant]})
    aggregate_rows.append({
        "variant": "delta",
        **{
            key: aggregate["matched_retain"][key] - aggregate["official"][key]
            for key in aggregate["official"]
        },
    })
    write_csv(
        results_dir / "aggregate.csv", aggregate_rows,
        ["variant", "unlearn_acc", "preserve_acc", "overall_acc"],
    )
    diagnostics = diagnostic_comparison(args.run_dir, results_dir)

    decision = None
    if args.profile == "formal":
        rule = config["decision_rule"]
        preserve_delta = aggregate["matched_retain"]["preserve_acc"] - aggregate["official"]["preserve_acc"]
        unlearn_delta = aggregate["matched_retain"]["unlearn_acc"] - aggregate["official"]["unlearn_acc"]
        positive_groups = sum(row["delta_preserve_acc"] > 0 for row in group_rows)
        preservation_passes = (
            preserve_delta >= rule["minimum_aggregate_preserve_gain_pp"]
            and positive_groups >= rule["minimum_groups_with_positive_preserve_delta"]
        )
        if preservation_passes and unlearn_delta <= rule["unlearn_noninferiority_margin_pp"]:
            decision = "SUPPORTED"
            decision_text = "matched-retain negatives consistently improve similar-concept preservation without materially weakening target erasure."
        elif preservation_passes:
            decision = "TRADE-OFF ONLY"
            decision_text = "preservation improves, but primarily by weakening unlearning rather than improving specificity."
        else:
            decision = "NOT SUPPORTED"
            decision_text = "no consistent image-level advantage."

    main_rows = [
        [row["variant"], fmt(row["unlearn_acc"]), fmt(row["preserve_acc"]), fmt(row["overall_acc"])]
        for row in aggregate_rows
    ]
    per_group_md = []
    for row in group_rows:
        per_group_md.append([
            row["group"],
            fmt(row["official_unlearn_acc"]), fmt(row["matched_unlearn_acc"]), fmt(row["delta_unlearn_acc"]),
            fmt(row["official_preserve_acc"]), fmt(row["matched_preserve_acc"]), fmt(row["delta_preserve_acc"]),
            fmt(row["official_overall_acc"]), fmt(row["matched_overall_acc"]), fmt(row["delta_overall_acc"]),
        ])
    per_concept_md = [
        [row["group"], row["role"], row["concept"], fmt(row["official"]), fmt(row["matched_retain"]), fmt(row["delta"])]
        for row in concept_rows
    ]
    mean_spearman = statistics.mean(row["spearman_alpha"] for row in diagnostics)
    mean_overlaps = {
        percent: statistics.mean(row[f"top_{percent}_percent_overlap"] for row in diagnostics)
        for percent in (1, 5, 10)
    }

    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    protocol_manifest = json.loads((args.run_dir / "protocol_manifest.json").read_text())
    summary = f"""# ScaPre Informax Specificity Experiment

## A. Audit

The audited repository uses five target pseudo-samples against five empty-prompt pseudo-samples. It applies the repository's median binarization, empirical MI, channel z-score, sigmoid temperature, power transform, and channel-wise maximum aggregation. The complete audit is in `experiments/scapre_informax_specificity/AUDIT.md`.

The public repository does not contain the complete paper Confuse5 seed asset. This run is therefore labeled **{config['protocol_label']}**, not an exact reproduction of paper Table 7.

## B. Modification

The only algorithmic intervention is the negative base-vector source inside `scapre/edit/erase_scale.py`. The existing `_compute_mi_softmask_emptyneg` remains the `official` default; `_compute_mi_softmask_matchedneg` changes only the negative bases to the three listed same-group retain embeddings, selected by the `compute_informax` dispatcher inside `edit_model`. Both modes use exactly five negatives, identical Gaussian-noise draw shapes, and the same downstream Informax and ScaPre code. Five negatives are assigned round-robin as 2/2/1 in declared retain order. `controlled_ablation_check.json` verifies that the two normalized edit commands differ only in the intervention and variant-specific artifact paths.

## C. Reproducibility

- Profile: `{args.profile}`
- Workspace commit: `{manifest['git_commit']}`
- Working tree dirty at launch: `{manifest['git_dirty']}`
- Base model: `{manifest['assets']['base_model']}`
- Resolved model revision: `{manifest['assets']['resolved_revision']}`
- Protocol SHA-256: `{protocol_manifest['sha256']}`
- Images per concept: `{protocol_manifest['images_per_concept']}`
- Prompt template: `{config['evaluation']['prompt_template']}`
- Seed sources: `{json.dumps(protocol_manifest['seed_sources'], sort_keys=True)}`
- Generation: base-model scheduler, {config['evaluation']['num_inference_steps']} steps, CFG {config['evaluation']['guidance_scale']}, {config['evaluation']['width']}x{config['evaluation']['height']}, float16
- Classifier: `{config['evaluation']['classifier']}`, top-1, repository substring label mapping
- Informax edit seed: `{config['edit_seed']}` for both variants

## D. Main table

{markdown_table(['Variant', 'Unlearn Acc ↓', 'Preserve Acc ↑', 'Overall Acc ↑'], main_rows)}

## E. Per-group table

{markdown_table(['Group', 'Off U', 'Match U', 'Δ U', 'Off P', 'Match P', 'Δ P', 'Off O', 'Match O', 'Δ O'], per_group_md)}

## F. Per-concept table

{markdown_table(['Group', 'Role', 'Concept', 'Official', 'Matched', 'Delta'], per_concept_md)}

## G. Informax diagnostic

These are mechanism diagnostics only and are not success metrics. Across `{len(diagnostics)}` matched layer/projection/target records, mean Spearman alpha correlation is `{mean_spearman:.4f}`. Mean official-vs-matched top-channel overlap is `{mean_overlaps[1]:.4f}` at 1%, `{mean_overlaps[5]:.4f}` at 5%, and `{mean_overlaps[10]:.4f}` at 10%. Raw MI/alpha tensors remain in `diagnostics/*.pt`; exact indices are in `results/top_channels.json`.

## H. Final judgment

"""
    if args.profile == "formal":
        summary += f"**{decision}**\n\n{decision_text}\n"
    else:
        summary += "Smoke test only. No scientific judgment is permitted from this run.\n"
    (results_dir / "summary.md").write_text(summary)
    if args.profile == "formal":
        published = Path(__file__).resolve().parent / "results" / "summary.md"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text(summary)

    result_manifest = {
        "profile": args.profile,
        "paired_rows": len(by_variant["official"]),
        "concepts": len(concept_rows),
        "groups": len(group_rows),
        "judgment": decision,
    }
    (results_dir / "result_manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
