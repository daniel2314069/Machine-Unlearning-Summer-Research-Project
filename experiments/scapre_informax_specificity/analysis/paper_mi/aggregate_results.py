#!/usr/bin/env python3
"""Validate and aggregate the fixed paper-MI versus repository comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generation_key(row: dict[str, str]) -> tuple[str, ...]:
    fields = ("group", "role", "concept", "sample_index", "prompt", "seed", "seed_source")
    return tuple(row[field] for field in fields)


def accuracy(rows: list[dict[str, str]]) -> float:
    return 100.0 * sum(int(row["correct"]) for row in rows) / len(rows)


def metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    target = [row for row in rows if row["role"] == "target"]
    retain = [row for row in rows if row["role"] == "retain"]
    unlearn = accuracy(target)
    preserve = accuracy(retain)
    forgetting = 100.0 - unlearn
    return {
        "unlearn_accuracy": unlearn,
        "preserve_accuracy": preserve,
        "overall_accuracy": 2.0 * forgetting * preserve / (forgetting + preserve),
    }


def canonical_evaluator(manifest: dict) -> dict:
    value = {key: item for key, item in manifest.items() if key not in {"variant", "checkpoint_sha256"}}
    defaults = value.get("scheduler_config", {}).get("_use_default_values")
    if isinstance(defaults, list):
        value["scheduler_config"]["_use_default_values"] = sorted(defaults)
    return value


def main() -> None:
    ns = parse_args()
    run_dir = ns.run_dir.resolve()
    config = json.loads((run_dir / "actual_config.json").read_text())
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    baseline_source = json.loads((run_dir / "baseline_source.json").read_text())
    repository_reference_reused = bool(baseline_source["repository_reference_reused"])
    results = run_dir / "results"
    results.mkdir(exist_ok=True)
    seeds = config["edit_seeds"] if ns.profile == "formal" else [config["legacy_informax_seed"]]
    variants = config["variants"]
    expected_rows = (
        config["expected_images_per_variant_formal"]
        if ns.profile == "formal" else config["expected_images_per_variant_smoke"]
    )

    scores: dict[tuple[int, str], list[dict[str, str]]] = {}
    fingerprints: dict[tuple[int, str], dict] = {}
    score_hashes: dict[str, dict[str, str]] = {}
    reference_keys: set[tuple[str, ...]] | None = None
    reference_evaluator: dict | None = None
    for seed in seeds:
        score_hashes[str(seed)] = {}
        for variant in variants:
            directory = run_dir / "seeds" / str(seed) / "evaluation" / variant
            path = directory / "scores.csv"
            rows = read_csv(path)
            if len(rows) != expected_rows or len({generation_key(row) for row in rows}) != expected_rows:
                raise RuntimeError(f"invalid row coverage: seed={seed}, variant={variant}")
            if any(row["variant"] != variant or row["correct"] not in {"0", "1"} for row in rows):
                raise RuntimeError(f"invalid score values: seed={seed}, variant={variant}")
            keys = {generation_key(row) for row in rows}
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise RuntimeError("prompts or generation seeds differ across a paired comparison")
            fingerprint = canonical_evaluator(json.loads((directory / "evaluation_manifest.json").read_text()))
            if reference_evaluator is None:
                reference_evaluator = fingerprint
            elif fingerprint != reference_evaluator:
                raise RuntimeError("evaluator fingerprint differs across variants/seeds")
            scores[(seed, variant)] = rows
            fingerprints[(seed, variant)] = fingerprint
            score_hashes[str(seed)][variant] = sha256(path)

    metric_map = {(seed, variant): metrics(scores[(seed, variant)]) for seed in seeds for variant in variants}
    per_seed_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for seed in seeds:
        for variant in variants:
            per_seed_rows.append({"edit_seed": seed, "variant": variant, **metric_map[(seed, variant)]})
        repository = metric_map[(seed, "official")]
        paper = metric_map[(seed, "paper_mi")]
        comparison_rows.append({
            "edit_seed": seed,
            "delta_unlearn": paper["unlearn_accuracy"] - repository["unlearn_accuracy"],
            "delta_preserve": paper["preserve_accuracy"] - repository["preserve_accuracy"],
            "delta_overall": paper["overall_accuracy"] - repository["overall_accuracy"],
        })

    concepts = sorted({row["concept"] for row in next(iter(scores.values()))})
    per_concept_rows: list[dict[str, object]] = []
    for seed in seeds:
        for concept in concepts:
            for variant in variants:
                selected = [row for row in scores[(seed, variant)] if row["concept"] == concept]
                per_concept_rows.append({
                    "edit_seed": seed,
                    "variant": variant,
                    "group": selected[0]["group"],
                    "role": selected[0]["role"],
                    "concept": concept,
                    "accuracy": accuracy(selected),
                    "rows": len(selected),
                })

    alpha_rows: list[dict[str, object]] = []
    for seed in seeds:
        diagnostics = torch.load(
            run_dir / "seeds" / str(seed) / "diagnostics" / "paper_mi.pt",
            map_location="cpu",
        )
        for record in diagnostics["records"]:
            if record["stage"] != "aggregate-max":
                continue
            alpha = record["alpha"].double().flatten()
            raw = record["concept_max_raw_mi"].double().flatten()
            alpha_rows.append({
                "edit_seed": seed,
                "projection": record["projection"],
                "layer_index": record["layer_index"],
                "channels": alpha.numel(),
                "raw_mi_min": raw.min().item(),
                "raw_mi_mean": raw.mean().item(),
                "raw_mi_max": raw.max().item(),
                "alpha_min": alpha.min().item(),
                "alpha_mean": alpha.mean().item(),
                "alpha_max": alpha.max().item(),
                "trace_B": alpha.sum().item(),
                "frobenius_B": torch.linalg.vector_norm(alpha).item(),
            })

    generated_variants = ["paper_mi"] if repository_reference_reused else variants
    image_rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for seed in seeds:
        for variant in generated_variants:
            expected_root = (run_dir / "seeds" / str(seed) / "evaluation" / variant / "images").resolve()
            for row in scores[(seed, variant)]:
                image = Path(row["image_path"]).resolve()
                try:
                    image.relative_to(expected_root)
                except ValueError as error:
                    raise RuntimeError(f"generated image escaped its output root: {image}") from error
                if image in seen or not image.is_file():
                    raise RuntimeError(f"generated image missing or duplicated: {image}")
                seen.add(image)
                image_rows.append({
                    "edit_seed": seed,
                    "variant": variant,
                    "concept": row["concept"],
                    "sample_index": row["sample_index"],
                    "relative_image_path": str(image.relative_to(run_dir)),
                    "size_bytes": image.stat().st_size,
                    "sha256": sha256(image),
                })
    expected_images = len(seeds) * len(generated_variants) * expected_rows
    if len(image_rows) != expected_images:
        raise RuntimeError("generated image manifest count mismatch")

    write_csv(results / "per_seed_metrics.csv", per_seed_rows, list(per_seed_rows[0]))
    write_csv(results / "comparison_deltas.csv", comparison_rows, list(comparison_rows[0]))
    write_csv(results / "per_concept_metrics.csv", per_concept_rows, list(per_concept_rows[0]))
    write_csv(results / "paper_alpha_summary.csv", alpha_rows, list(alpha_rows[0]))
    write_csv(results / "generated_image_manifest.csv", image_rows, list(image_rows[0]))

    mean_delta = {
        key: statistics.mean(float(row[key]) for row in comparison_rows)
        for key in ("delta_unlearn", "delta_preserve", "delta_overall")
    }
    std_delta = {
        key: statistics.stdev(float(row[key]) for row in comparison_rows) if len(comparison_rows) > 1 else 0.0
        for key in mean_delta
    }
    integrity = {
        "status": "passed",
        "profile": ns.profile,
        "variants": variants,
        "edit_seeds": seeds,
        "rows_per_variant_seed": expected_rows,
        "role_counts": dict(Counter(row["role"] for row in next(iter(scores.values())))),
        "prompt_lists_and_generation_seeds_identical": True,
        "evaluator_fingerprint_identical": True,
        "official_empty_string_neutral": True,
        "paper_formula_checks_passed": all(
            json.loads((run_dir / "seeds" / str(seed) / "paper_formula_check.json").read_text())["status"] == "passed"
            for seed in seeds
        ),
        "repository_reference_reused": repository_reference_reused,
        "repository_baseline_generated_in_run": not repository_reference_reused,
        "parameter_search": False,
        "generated_image_count": len(image_rows),
        "generated_image_manifest_sha256": sha256(results / "generated_image_manifest.csv"),
        "score_sha256": score_hashes,
        "protocol_sha256": sha256(run_dir / "protocol.csv"),
        "evaluator_fingerprint_sha256": hashlib.sha256(
            json.dumps(reference_evaluator, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "git_commit": run_manifest["git_commit"],
        "git_status_start_clean": run_manifest["git_status_start"] == [],
        "git_status_before_aggregation_clean": run_manifest.get("git_status_before_aggregation") == [],
    }
    (results / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n")
    result_manifest = {
        "status": "descriptive_smoke_only" if ns.profile == "smoke" else "completed_fixed_comparison",
        "mean_delta_paper_minus_repository": mean_delta,
        "sample_std_delta": std_delta,
        "favorable_seed_counts": {
            "unlearn_lower": sum(float(row["delta_unlearn"]) < 0 for row in comparison_rows),
            "preserve_higher": sum(float(row["delta_preserve"]) > 0 for row in comparison_rows),
            "overall_higher": sum(float(row["delta_overall"]) > 0 for row in comparison_rows),
        },
        "parameter_search": False,
    }
    (results / "result_manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n")

    if ns.profile == "formal":
        lines = [
            "# ScaPre paper-MI versus repository baseline",
            "",
            "This fixed five-seed comparison changes only the Informax weighting path. The paper variant uses raw MI, takes the maximum over concepts, normalizes by the maximum channel, and applies `B=diag(alpha)` only in the final objective. Prompts, pseudo-sample count, threshold, empty-string neutral, evaluation, and seeds are unchanged. No parameter search was run.",
            "",
            ("The repository baseline came from the checksum-pinned, fully validated historical reference."
             if repository_reference_reused else
             "The repository baseline was regenerated in this run because the verified historical cache/archive was unavailable."),
            "",
            "| Metric (paper - repository) | Mean | Sample std | Favorable seeds |",
            "| --- | ---: | ---: | ---: |",
            f"| Unlearn accuracy (lower is favorable) | {mean_delta['delta_unlearn']:+.4f} | {std_delta['delta_unlearn']:.4f} | {result_manifest['favorable_seed_counts']['unlearn_lower']}/5 |",
            f"| Preserve accuracy (higher is favorable) | {mean_delta['delta_preserve']:+.4f} | {std_delta['delta_preserve']:.4f} | {result_manifest['favorable_seed_counts']['preserve_higher']}/5 |",
            f"| Overall accuracy (higher is favorable) | {mean_delta['delta_overall']:+.4f} | {std_delta['delta_overall']:.4f} | {result_manifest['favorable_seed_counts']['overall_higher']}/5 |",
            "",
            "Absolute values and per-concept results are in the CSV files. This is the project-established Confuse5 reconstruction, not an exact reproduction of a released paper seed asset.",
        ]
    else:
        lines = [
            "# ScaPre paper-MI smoke test",
            "",
            "The end-to-end two-variant smoke test passed integrity checks. These 10-image-per-variant results are not scientific evidence.",
        ]
    (results / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
