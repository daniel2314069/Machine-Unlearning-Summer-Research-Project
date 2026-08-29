#!/usr/bin/env python3
"""Summarize the completed V1 raw-score and transformed-alpha diagnostics.

This is descriptive only.  It reads an existing seed-20260820 diagnostics file;
it does not edit a model, generate images, select parameters, or gate the frozen
direct-cos2 formula.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


EXPECTED_RECORDS = 320
EXPECTED_TARGETS = 10


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def distribution(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().double().flatten().cpu()
    quantiles = torch.quantile(
        flat, torch.tensor([0.01, 0.5, 0.95, 0.99], dtype=torch.double)
    )
    total = float(flat.sum().item())

    def top_share(fraction: float) -> float | None:
        if total == 0.0:
            return None
        count = max(1, math.ceil(flat.numel() * fraction))
        return float(torch.topk(flat, count).values.sum().item() / total)

    return {
        "count": flat.numel(),
        "finite": bool(torch.isfinite(flat).all().item()),
        "non_constant": bool(flat.max().item() != flat.min().item()),
        "min": float(flat.min().item()),
        "p01": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=True).item()),
        "rms": float(torch.sqrt(flat.square().mean()).item()),
        "p95": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
        "max": float(flat.max().item()),
        "top_1pct_weight_share": top_share(0.01),
        "top_5pct_weight_share": top_share(0.05),
    }


def flattened(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in stats.items() if key != "count"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.v1_run_dir.resolve()
    output = args.output_dir.resolve()
    integrity_path = run_dir / "reproducibility" / "integrity_report.json"
    completion_path = run_dir / "worker_complete.json"
    diagnostic_path = (
        run_dir / "seeds" / "20260820" / "diagnostics"
        / "projection_projection_accumulation.pt"
    )
    for path in (integrity_path, completion_path, diagnostic_path):
        if not path.is_file():
            raise RuntimeError(f"required V1 artifact is missing: {path}")
    integrity = json.loads(integrity_path.read_text())
    completion = json.loads(completion_path.read_text())
    if integrity.get("status") != "passed" or completion.get("status") != "passed":
        raise RuntimeError("V1 formal run did not pass integrity/completion")
    if integrity.get("edit_seeds") != [20260820, 20260821, 20260822, 20260823, 20260824]:
        raise RuntimeError("V1 formal edit-seed set changed")
    if integrity.get("new_generated_image_count") != 15000:
        raise RuntimeError("V1 source is not the completed 15,000-image formal run")
    if integrity.get("qualification_status") != "passed":
        raise RuntimeError("V1 formal qualification did not pass")
    payload = torch.load(diagnostic_path, map_location="cpu")
    if payload.get("variant") != "projection_accumulation":
        raise RuntimeError("diagnostics are not the completed V1 treatment")
    records = payload.get("accumulation_records", [])
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(f"V1 diagnostic coverage changed: {len(records)}")
    targets = list(dict.fromkeys(row["target_concept"] for row in records))
    if len(targets) != EXPECTED_TARGETS:
        raise RuntimeError(f"V1 target coverage changed: {targets}")

    output.mkdir(parents=True, exist_ok=True)
    by_concept: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {"raw": [], "v1": [], "official": []}
    )
    layer_rows: list[dict[str, Any]] = []
    all_raw: list[torch.Tensor] = []
    all_v1: list[torch.Tensor] = []
    all_official: list[torch.Tensor] = []
    for row in records:
        raw = row["projection_score"].detach().double().flatten().cpu()
        v1 = row["projection_alpha"].detach().double().flatten().cpu()
        official = row["official_row_w_c"].detach().double().flatten().cpu()
        if not all(torch.isfinite(value).all().item() for value in (raw, v1, official)):
            raise RuntimeError("V1 diagnostics contain NaN/Inf")
        concept = row["target_concept"]
        by_concept[concept]["raw"].append(raw)
        by_concept[concept]["v1"].append(v1)
        by_concept[concept]["official"].append(official)
        all_raw.append(raw)
        all_v1.append(v1)
        all_official.append(official)
        raw_stats = distribution(raw)
        v1_stats = distribution(v1)
        official_stats = distribution(official)
        layer_rows.append({
            "projection": row["projection"],
            "layer_index": row["layer_index"],
            "target_index": row["target_index"],
            "target_concept": concept,
            **flattened("raw_cos2", raw_stats),
            **flattened("v1_alpha", v1_stats),
            **flattened("official_alpha", official_stats),
            "v1_to_raw_mean_ratio": (
                v1_stats["mean"] / raw_stats["mean"] if raw_stats["mean"] else None
            ),
        })

    layer_path = output / "per_layer_concept_distributions.csv"
    with layer_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)

    concept_rows: list[dict[str, Any]] = []
    for concept in targets:
        raw_stats = distribution(torch.cat(by_concept[concept]["raw"]))
        v1_stats = distribution(torch.cat(by_concept[concept]["v1"]))
        official_stats = distribution(torch.cat(by_concept[concept]["official"]))
        concept_rows.append({
            "target_concept": concept,
            **flattened("raw_cos2", raw_stats),
            **flattened("v1_alpha", v1_stats),
            **flattened("official_alpha", official_stats),
            "v1_to_raw_mean_ratio": (
                v1_stats["mean"] / raw_stats["mean"] if raw_stats["mean"] else None
            ),
        })
    concept_path = output / "per_concept_distributions.csv"
    with concept_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(concept_rows[0]))
        writer.writeheader()
        writer.writerows(concept_rows)

    aggregate = {
        "status": "passed",
        "descriptive_only": True,
        "used_for_formula_selection": False,
        "v1_run_dir": str(run_dir),
        "diagnostics_path": str(diagnostic_path),
        "records": len(records),
        "targets": targets,
        "raw_cos2": distribution(torch.cat(all_raw)),
        "v1_alpha": distribution(torch.cat(all_v1)),
        "official_alpha": distribution(torch.cat(all_official)),
        "v1_to_raw_mean_ratio": float(
            torch.cat(all_v1).mean().item() / torch.cat(all_raw).mean().item()
        ),
        "per_layer_concept_csv": str(layer_path),
        "per_concept_csv": str(concept_path),
    }
    write_json(output / "summary.json", aggregate)
    lines = [
        "# V1 transform diagnostics", "",
        "Descriptive only; all ten targets and all 320 layer/concept records are included.", "",
        "| Weight | Mean | Median | P95 | P99 | Max | Top 1% share | Top 5% share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("official accumulation alpha", "official_alpha"),
        ("raw cos2", "raw_cos2"),
        ("V1 transformed alpha", "v1_alpha"),
    ):
        stats = aggregate[key]
        lines.append(
            f"| {label} | {stats['mean']:.8g} | {stats['median']:.8g} | "
            f"{stats['p95']:.8g} | {stats['p99']:.8g} | {stats['max']:.8g} | "
            f"{stats['top_1pct_weight_share']:.6f} | {stats['top_5pct_weight_share']:.6f} |"
        )
    lines.extend([
        "",
        f"V1 alpha mean / raw cos2 mean: `{aggregate['v1_to_raw_mean_ratio']:.6f}`.",
        "",
        "No threshold, parameter, or treatment choice is derived from this report.",
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
