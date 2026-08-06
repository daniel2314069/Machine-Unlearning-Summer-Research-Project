#!/usr/bin/env python
"""Cross-tab the existing EOT spherical clusters against possible confounds.

This script does not extract a new representation or change clustering.  It
replays the deterministic raw/centered fits from the saved experiment settings,
verifies their objectives, then writes counts and within-cluster proportions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from concept_clustering.utils import atomic_write_text
from scripts.eot_spherical_clustering import (
    cluster_representations_without_labels,
    evaluate_after_clustering,
    load_description_dataset,
)


FIELDS = ("facet_id", "source", "round", "effective_token_length")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_values(series: pd.Series, field: str) -> list[Any]:
    values = series.drop_duplicates().tolist()
    if field == "effective_token_length":
        return sorted(int(value) for value in values)
    return sorted(str(value) for value in values)


def build_composition_tables(
    assignments: pd.DataFrame,
    field: str,
    cluster_columns: list[tuple[str, int, str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide count and within-cluster proportion tables."""
    value_column = field
    values = _ordered_values(assignments[value_column], field)
    counts = pd.DataFrame({value_column: values})
    proportions = pd.DataFrame({value_column: values})
    for representation, cluster_id, matched_concept, cluster_size in cluster_columns:
        column = f"{representation}_cluster_{cluster_id}__matched_{matched_concept}__n{cluster_size}"
        subset = assignments[
            (assignments["representation"] == representation)
            & (assignments["predicted_cluster"] == cluster_id)
        ]
        current_counts = subset[value_column].value_counts(dropna=False)
        if field == "effective_token_length":
            count_values = [int(current_counts.get(int(value), 0)) for value in values]
        else:
            count_values = [int(current_counts.get(str(value), 0)) for value in values]
        counts[column] = count_values
        proportions[column] = np.asarray(count_values, dtype=np.float64) / float(cluster_size)
    return counts, proportions


def run(dataset_path: Path, results_dir: Path, output_dir: Path) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    results_dir = results_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = json.loads((results_dir / "metrics.json").read_text())
    if _sha256(dataset_path) != metrics_payload["dataset"]["sha256"]:
        raise RuntimeError("Dataset hash does not match the completed EOT experiment")
    rows = load_description_dataset(dataset_path)
    metadata = pd.read_csv(results_dir / "metadata.csv").sort_values("sample_index")
    if len(rows) != len(metadata):
        raise RuntimeError("Dataset and EOT metadata row counts differ")
    if metadata["description"].tolist() != [row["description"] for row in rows]:
        raise RuntimeError("Dataset order/text differs from EOT metadata")

    raw = np.load(results_dir / "eot_raw_normalized.npy")
    centered = np.load(results_dir / "eot_centered_normalized.npy")
    settings = metrics_payload["spherical_kmeans"]
    fits = cluster_representations_without_labels(
        raw,
        centered,
        k=int(settings["k"]),
        n_init=int(settings["n_init"]),
        max_iter=int(settings["max_iter"]),
        tolerance=float(settings["tolerance"]),
        random_seed=int(settings["random_seed"]),
    )

    # As in the main experiment, labels are created only after both fits exist.
    concept_names = list(metrics_payload["dataset"]["concept_order"])
    true_ids = np.asarray([concept_names.index(str(row["concept"])) for row in rows])
    assignments: list[dict[str, Any]] = []
    cluster_columns: list[tuple[str, int, str, int]] = []
    replay_checks: dict[str, Any] = {}
    for representation, features in (("raw", raw), ("centered", centered)):
        saved_metrics = metrics_payload["representations"][representation]
        result = fits[representation]
        evaluated, predicted_ids, _ = evaluate_after_clustering(
            features, result, true_ids, concept_names
        )
        if not np.isclose(
            evaluated["spherical_objective"],
            saved_metrics["spherical_objective"],
            rtol=1e-7,
            atol=1e-7,
        ):
            raise RuntimeError(f"{representation} replay objective differs from saved experiment")
        if evaluated["cluster_sizes"] != saved_metrics["cluster_sizes"]:
            raise RuntimeError(f"{representation} replay cluster sizes differ from saved experiment")
        cluster_to_concept = {
            int(cluster): concept for cluster, concept in evaluated["cluster_to_concept"].items()
        }
        for cluster_id, cluster_size in enumerate(evaluated["cluster_sizes"]):
            cluster_columns.append(
                (representation, cluster_id, cluster_to_concept[cluster_id], int(cluster_size))
            )
        for index, row in enumerate(rows):
            assignments.append({
                "representation": representation,
                "sample_index": index,
                "candidate_id": row.get("candidate_id", f"row_{index:04d}"),
                "true_concept": row["concept"],
                "predicted_cluster": int(result.labels[index]),
                "matched_predicted_concept": concept_names[int(predicted_ids[index])],
                "facet_id": row.get("facet_id", ""),
                "source": row.get("source", ""),
                "round": row.get("round", ""),
                "effective_token_length": int(metadata.iloc[index]["effective_token_length"]),
                "description": row["description"],
            })
        replay_checks[representation] = {
            "objective": evaluated["spherical_objective"],
            "cluster_sizes": evaluated["cluster_sizes"],
            "cluster_to_concept": evaluated["cluster_to_concept"],
        }

    assignment_frame = pd.DataFrame(assignments)
    missing = {
        field: int((assignment_frame[field].astype(str).str.strip() == "").sum())
        for field in FIELDS
    }
    if any(missing.values()):
        raise RuntimeError(f"Composition fields contain missing values: {missing}")
    assignment_frame.to_csv(output_dir / "cluster_composition_assignments.csv", index=False)

    validation: dict[str, Any] = {
        "status": "passed",
        "dataset_sha256": metrics_payload["dataset"]["sha256"],
        "rows_per_representation": len(rows),
        "representations": replay_checks,
        "field_missing_counts": missing,
        "tables": {},
    }
    for field in FIELDS:
        counts, proportions = build_composition_tables(
            assignment_frame, field, cluster_columns
        )
        counts_path = output_dir / f"cluster_by_{field}_counts.csv"
        proportions_path = output_dir / f"cluster_by_{field}_proportions.csv"
        counts.to_csv(counts_path, index=False)
        proportions.to_csv(proportions_path, index=False, float_format="%.8f")
        numeric_count_columns = [column for column in counts.columns if column != field]
        numeric_proportion_columns = [column for column in proportions.columns if column != field]
        expected_sizes = {
            f"{representation}_cluster_{cluster_id}__matched_{matched_concept}__n{cluster_size}": cluster_size
            for representation, cluster_id, matched_concept, cluster_size in cluster_columns
        }
        count_sums = {column: int(counts[column].sum()) for column in numeric_count_columns}
        proportion_sums = {
            column: float(proportions[column].sum()) for column in numeric_proportion_columns
        }
        if count_sums != expected_sizes:
            raise RuntimeError(f"{field} count columns do not sum to cluster sizes")
        if not all(np.isclose(value, 1.0, atol=1e-7) for value in proportion_sums.values()):
            raise RuntimeError(f"{field} proportions do not sum to one")
        validation["tables"][field] = {
            "categories": len(counts),
            "count_column_sums": count_sums,
            "proportion_column_sums": proportion_sums,
        }

    atomic_write_text(
        output_dir / "cluster_composition_validation.json",
        json.dumps(validation, indent=2) + "\n",
    )
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-tab unchanged EOT spherical clusters against facet/source/round/length"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validation = run(args.dataset, args.results, args.output)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
