#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from concept_clustering.config import load_config
from concept_clustering.utils import atomic_write_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source = args.source_output.resolve()
    output = args.output.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    concepts = [row["name"] for row in config["concepts"]]
    audit = pd.read_csv(output / "bare_name_tokenization_audit.csv")
    check("four_exact_bare_prompts", audit["prompt"].tolist() == concepts,
          f"prompts={audit['prompt'].tolist()}")
    check("single_unsplit_content_tokens",
          audit["selected_token_position"].eq(1).all()
          and audit["concept_token_count"].eq(1).all()
          and (~audit["concept_split_into_multiple_tokens"].astype(bool)).all(),
          f"tokens={audit['selected_decoded_token'].tolist()}")
    check("selected_tokens_match_names",
          audit["selected_decoded_token"].tolist() == [f"{name}</w>" for name in concepts],
          f"ids={audit['selected_token_id'].tolist()}")
    check("exact_repository_rule_audited",
          audit["exact_repository_function_or_rule"].str.contains("attention_mask.sum() - 2", regex=False).all()
          and audit["exact_repository_function_or_rule"].str.contains("pipe.encode_prompt", regex=False).all(),
          "OCE/UCE inline rule and encode_prompt are recorded")

    bare = torch.load(output / "bare_name_embeddings.pt", map_location="cpu", weights_only=False)
    check("bare_embedding_shapes",
          tuple(bare["raw_unnormalized"].shape) == (4, 768)
          and tuple(bare["raw_normalized"].shape) == (4, 768)
          and len(bare["layer_names"]) == 16
          and all(bare["layer_unnormalized"][name].shape[0] == 4 for name in bare["layer_names"]),
          "raw=(4,768), projected rows=4 across 16 layers")
    check("no_generation_or_edit_metadata",
          bare["metadata"].get("image_generation_performed") is False
          and bare["metadata"].get("projection") == "to_v",
          "image_generation_performed=False; projection=to_v")
    source_hashes = bare["metadata"].get("immutable_source_sha256", {})
    current_hashes = {
        name: _sha256(source / name)
        for name in ["accepted_descriptions.jsonl", "raw_text_embeddings.pt", "layer_embeddings.pt"]
    }
    check("immutable_source_hashes", source_hashes == current_hashes,
          json.dumps(current_hashes, sort_keys=True))

    centroid = pd.read_csv(output / "bare_vs_fixed_prototype_centroid_metrics.csv")
    capture = pd.read_csv(output / "bare_name_subspace_capture.csv")
    heldout = pd.read_csv(output / "heldout_description_subspace_metrics.csv")
    per_split = pd.read_csv(output / "per_split_metrics.csv")
    layerwise = pd.read_csv(output / "layerwise_bare_name_metrics.csv")
    check("centroid_metric_cardinality", len(centroid) == 17 * 2 * 4,
          f"rows={len(centroid)}")
    check("capture_metric_cardinality", len(capture) == 17 * 7 * 4 * 4,
          f"rows={len(capture)}")
    check("heldout_metric_cardinality", len(heldout) == 17 * 7 * 5,
          f"rows={len(heldout)}")
    check("per_split_cardinality", len(per_split) == 17 * 7 * 100 * 5,
          f"rows={len(per_split)}, splits={per_split['split'].nunique()}")
    check("layerwise_metric_cardinality", len(layerwise) == 17 * 7 * 5,
          f"rows={len(layerwise)}")
    capture_columns = [
        (capture, "capture"),
        (per_split, "mean_own_subspace_capture"),
        (per_split, "mean_highest_incorrect_subspace_capture"),
        (per_split, "bare_own_subspace_capture"),
    ]
    check("capture_scores_bounded",
          all(frame[column].between(0, 1).all() for frame, column in capture_columns),
          "all checked capture values are in [0,1]")
    check("balanced_heldout_splits",
          per_split.loc[per_split["concept"] != "__overall__", "n_heldout"].eq(10).all()
          and per_split.loc[per_split["concept"] == "__overall__", "n_heldout"].eq(40).all(),
          "10 per concept and 40 overall for every split/rank/space")
    check("all_required_ranks",
          set(capture["rank_label"].astype(str)) == {"1", "2", "4", "8", "16", "32", "full"},
          f"ranks={sorted(set(capture['rank_label'].astype(str)))}")

    required = [
        "bare_name_tokenization_audit.csv", "bare_name_embeddings.pt",
        "bare_vs_fixed_prototype_centroid_metrics.csv", "bare_name_subspace_capture.csv",
        "heldout_description_subspace_metrics.csv", "layerwise_bare_name_metrics.csv",
        "per_split_metrics.csv", "centroid_distance_heatmap.png",
        "bare_name_subspace_capture_heatmap.png", "layerwise_capture_curves.png",
        "rank_sweep_curves.png", "bare_name_subspace_report.md",
    ]
    missing = [name for name in required if not (output / name).is_file() or (output / name).stat().st_size == 0]
    check("required_outputs", not missing, f"missing_or_empty={missing}")
    image_files = [path.name for path in output.iterdir() if path.suffix.casefold() in {".jpg", ".jpeg", ".webp"}]
    check("no_generated_images", not image_files, f"unexpected_images={image_files}")
    report = (output / "bare_name_subspace_report.md").read_text()
    required_phrases = [
        "intentional asymmetric comparison", "attention_mask.sum() - 2",
        "Main result in plain language", "Held-out question", "full numerical rank",
        "does not test or prove OCE/UCE erasure correctness", "Layer 4 is not unusually strong",
    ]
    normalized_report = " ".join(report.split())
    check("report_scope_and_caveats", all(phrase in normalized_report for phrase in required_phrases),
          "asymmetry, exact rule, held-out controls, layer 4, and limits are explicit")

    summary = {
        "status": "passed", "checks_passed": len(checks), "checks": checks,
        "source_sha256": current_hashes, "concepts": concepts, "spaces": 17,
        "heldout_splits": 100, "rank_conditions": 7,
    }
    atomic_write_text(output / "qa_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    table = "\n".join(f"| {row['check']} | yes | {row['detail']} |" for row in checks)
    atomic_write_text(
        output / "qa_report.md",
        f"# Bare-name subspace QA\n\nAll **{len(checks)}** checks passed.\n\n"
        "| Check | Passed | Detail |\n|---|---:|---|\n" + table + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
