#!/usr/bin/env python
"""Fail-fast integrity audit for the completed codex_diverse experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from concept_clustering.config import all_banned_terms, load_config
from concept_clustering.text_validation import _contains_phrase, words
from concept_clustering.utils import atomic_write_text, read_jsonl
from scripts.merge_codex_diverse_rounds import _conflicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    accepted = read_jsonl(output / "accepted_descriptions.jsonl")
    concepts = [row["name"] for row in config["concepts"]]
    facets = [row["id"] for row in config["facets"]]
    expected_per_group = int(config["candidate_validation"]["accepted_per_concept_facet"])
    expected_total = len(concepts) * len(facets) * expected_per_group
    counts = Counter((row["concept"], row["facet_id"]) for row in accepted)
    expected_counts = {(concept, facet): expected_per_group for concept in concepts for facet in facets}
    check("balanced_dataset", len(accepted) == expected_total and counts == expected_counts,
          f"rows={len(accepted)}, groups={len(counts)}, expected={expected_total}")
    check("unique_candidate_ids", len({row["candidate_id"] for row in accepted}) == len(accepted),
          f"unique={len({row['candidate_id'] for row in accepted})}")
    check("single_honest_provenance", {row.get("source") for row in accepted} == {"codex_diverse"},
          f"sources={sorted({str(row.get('source')) for row in accepted})}")

    rules = config["candidate_validation"]
    invalid_lengths = [row["candidate_id"] for row in accepted
                       if not int(rules["min_words"]) <= len(words(row["description"])) <= int(rules["max_words"])]
    banned = all_banned_terms(config)
    leaked = [(row["candidate_id"], term) for row in accepted for term in banned
              if _contains_phrase(row["description"], term)]
    check("accepted_text_lengths", not invalid_lengths, f"invalid={invalid_lengths[:5]}")
    check("no_banned_concept_terms", not leaked, f"leaks={leaked[:5]}")
    conflict = ""
    for index, row in enumerate(accepted):
        conflict = _conflicts(row, accepted[:index])
        if conflict:
            break
    check("global_text_diversity", not conflict, f"first_conflict={conflict or 'none'}")

    decisions = pd.read_csv(output / "candidate_generation_decisions.csv").set_index("candidate_id")
    selected_decisions = decisions.loc[[row["candidate_id"] for row in accepted]]
    valid_generation = (
        selected_decisions["automatic_decision"].eq("accepted")
        & selected_decisions["generated_seed_count"].eq(3)
        & selected_decisions["target_top1_count"].ge(2)
    )
    check("three_seed_automatic_acceptance", bool(valid_generation.all()),
          f"passing={int(valid_generation.sum())}/{len(valid_generation)}")

    manifest = json.loads((output / "merge_manifest.json").read_text())
    solver = manifest.get("selection_solver", {})
    check("global_selection_solver", bool(solver.get("success")) and not manifest.get("shortages"),
          f"solver={solver.get('status')}, shortages={manifest.get('shortages')}")

    audit = pd.read_csv(output / "tokenization_audit.csv")
    expected_audit = expected_total * 3 + len(concepts)
    fixed = audit[audit["condition"].eq(config["readout"]["primary_suffix_name"])]
    shuffled = audit[audit["condition"].str.startswith("shuffled_words_")]
    check("tokenization_audit_cardinality", len(audit) == expected_audit,
          f"rows={len(audit)}, expected={expected_audit}")
    check("no_token_truncation", not audit["truncation_occurred"].astype(bool).any(), "all conditions")
    check("fixed_token_identity", fixed["selected_token_id"].nunique() == 1
          and fixed["selected_token"].eq("concept</w>").all(),
          f"ids={fixed['selected_token_id'].unique().tolist()}, tokens={fixed['selected_token'].unique().tolist()}")
    check("shuffled_token_identity", shuffled["selected_token_id"].nunique() == 1
          and shuffled["selected_token"].eq("concept</w>").all(),
          f"ids={shuffled['selected_token_id'].unique().tolist()}")

    raw = torch.load(output / "raw_text_embeddings.pt", map_location="cpu", weights_only=False)
    primary = config["readout"]["primary_suffix_name"]
    check("raw_embedding_shapes",
          tuple(raw["fixed_readout"][primary].shape) == (expected_total, 768)
          and tuple(raw["natural_last_token"].shape) == (expected_total, 768)
          and tuple(raw["shuffled_fixed_readout"].shape) == (expected_total, 768)
          and tuple(raw["prototypes"][primary].shape) == (len(concepts), 768),
          "fixed/natural/shuffled=(200,768), prototypes=(4,768)")
    check("original_model_metadata", raw["metadata"].get("projection") == "to_v"
          and bool(raw["metadata"].get("generation_checkpoint_verified"))
          and bool(raw["metadata"].get("generation_w0_structure_verified")),
          f"projection={raw['metadata'].get('projection')}, checkpoint/W0 verified")

    layers = torch.load(output / "layer_embeddings.pt", map_location="cpu", weights_only=False)
    layer_names = layers["layer_names"]
    layer_shapes_ok = (
        len(layer_names) == 16
        and all(".attn2.to_v" in name for name in layer_names)
        and all(layers["description_embeddings"][name].shape[0] == expected_total for name in layer_names)
        and all(layers["prototype_embeddings"][name].shape[0] == len(concepts) for name in layer_names)
    )
    check("sixteen_original_to_v_layers", layer_shapes_ok, f"layers={len(layer_names)}")

    metrics = pd.read_csv(output / "clustering_metrics.csv")
    representation_counts = metrics.groupby("representation").size().to_dict()
    expected_representations = {
        f"fixed:{primary}", "natural_last_token", f"shuffled_words:fixed:{primary}"
    }
    check("raw_clustering_runs", set(representation_counts) == expected_representations
          and set(representation_counts.values()) == {20}, f"counts={representation_counts}")
    layer_metrics = pd.read_csv(output / "layer_metrics.csv")
    check("layer_clustering_runs", len(layer_metrics) == 16 * 20
          and layer_metrics["layer_name"].nunique() == 16,
          f"rows={len(layer_metrics)}, layers={layer_metrics['layer_name'].nunique()}")
    prototypes = pd.read_csv(output / "prototype_metrics.csv")
    required_proto = {"description_distance_percentile", "bootstrap_percentile_median",
                      "bootstrap_percentile_ci_low", "bootstrap_percentile_ci_high"}
    prototype_group_sizes = prototypes.groupby("representation").size()
    check("prototype_percentiles", required_proto.issubset(prototypes.columns)
          and prototypes[list(required_proto)].notna().all().all()
          and len(prototype_group_sizes) == 16 + 2
          and prototype_group_sizes.eq(len(concepts)).all(),
          f"rows={len(prototypes)}, representations={len(prototype_group_sizes)}")

    required_files = [
        "candidate_descriptions.jsonl", "candidate_text_validation.csv", "generation_validation.csv",
        "accepted_descriptions.jsonl", "manual_review.csv", "tokenization_audit.csv",
        "raw_text_embeddings.pt", "layer_embeddings.pt", "clustering_metrics.csv",
        "clustering_assignments.csv", "prototype_metrics.csv", "facet_confounding_metrics.csv",
        "layer_metrics.csv", "final_report.md", "plots/confusion_fixed_describes_concept.png",
        "plots/concept_centroid_distances.png", "plots/prototype_to_centroid_distances.png",
        "plots/layer_clustering_metrics.png", "plots/layer_concept_vs_facet.png",
        "plots/description_length_by_concept.png", "plots/pca_by_concept.png",
        "plots/lexical_baseline_comparison.png", "launched_command.txt",
    ]
    missing = [name for name in required_files if not (output / name).is_file() or (output / name).stat().st_size == 0]
    check("required_artifacts", not missing, f"missing_or_empty={missing}")

    report = (output / "final_report.md").read_text()
    check("report_provenance", "single provenance label `codex_diverse`" in report
          and "not independent human or LLM sources" in report, "single-source disclosure present")
    check("report_no_false_manual_claim", "after three-seed visual inspection" not in report
          and "No manual overrides were used" in report, "manual-review wording audited")
    check("report_reproducibility_commands", "## Reproducibility and completion status" in report
          and "scripts.qa_final_results" in report and "launched_command.txt" in report,
          "status, cached analysis, QA, and launch-command references present")

    summary = {
        "status": "passed", "checks_passed": len(checks), "checks": checks,
        "accepted_rows": len(accepted), "concepts": concepts, "facets": facets,
        "to_v_layers": len(layer_names), "clustering_runs_per_representation": 20,
    }
    atomic_write_text(output / "qa_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = "\n".join(f"| {item['check']} | yes | {item['detail']} |" for item in checks)
    markdown = (
        "# Final result QA\n\n"
        f"All **{len(checks)}** fail-fast integrity checks passed.\n\n"
        "| Check | Passed | Detail |\n|---|---:|---|\n" + rows + "\n"
    )
    atomic_write_text(output / "qa_report.md", markdown)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
