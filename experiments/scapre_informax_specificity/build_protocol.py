#!/usr/bin/env python3
"""Build and validate the declared Confuse5 reconstruction protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from worker import validate_config


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
PUBLIC_CSV = REPO_ROOT / "scapre" / "eval" / "datasets" / "imagenet-15.csv"
DERIVED_CSV = (
    REPO_ROOT
    / "orthogonal-concept-erasure"
    / "experiments"
    / "confuse5_single_vs_joint"
    / "datasets"
    / "imagenet-confuse5-derived-25.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "formal"], default="formal")
    return parser.parse_args()


def load_public_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["class"].strip().lower(), []).append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    declared_dataset = REPO_ROOT / config["evaluation"]["protocol_dataset"]
    if declared_dataset.resolve() != DERIVED_CSV.resolve():
        raise RuntimeError("protocol dataset path differs from the frozen project asset")
    dataset_hash = hashlib.sha256(DERIVED_CSV.read_bytes()).hexdigest()
    if dataset_hash != config["evaluation"]["protocol_dataset_sha256"]:
        raise RuntimeError("derived Confuse5 dataset hash mismatch")
    public_classes = set(load_public_rows(PUBLIC_CSV))
    protocol_rows = load_public_rows(DERIVED_CSV)
    count = config["evaluation"][
        "smoke_images_per_concept" if args.profile == "smoke" else "formal_images_per_concept"
    ]
    groups = config["groups"][:1] if args.profile == "smoke" else config["groups"]
    output_rows: list[dict[str, object]] = []

    for group in groups:
        for role in ("target", "retain"):
            key = "targets" if role == "target" else "retains"
            for concept in group[key]:
                source_rows = protocol_rows.get(concept, [])
                if len(source_rows) != 500:
                    raise RuntimeError(f"derived protocol must contain 500 rows for {concept}")
                for index in range(count):
                    source = source_rows[index]
                    prompt = source["prompt"]
                    seed = int(source["evaluation_seed"])
                    seed_source = (
                        "public-repo:imagenet-15.csv"
                        if concept in public_classes
                        else "project-derived:same-group-retain-seed-reuse"
                    )
                    source_case = source["case_number"]
                    output_rows.append({
                        "group": group["id"],
                        "role": role,
                        "concept": concept,
                        "sample_index": index,
                        "prompt": prompt,
                        "seed": seed,
                        "seed_source": seed_source,
                        "source_case_number": source_case,
                    })

    expected_concepts = len(groups) * 5
    counts = Counter(str(row["concept"]) for row in output_rows)
    if len(counts) != expected_concepts or set(counts.values()) != {count}:
        raise RuntimeError(f"invalid protocol counts: {dict(counts)}")
    if args.profile == "formal" and len(output_rows) != 25 * count:
        raise RuntimeError("formal protocol must contain all 25 Confuse5 concepts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group", "role", "concept", "sample_index", "prompt", "seed",
        "seed_source", "source_case_number",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "profile": args.profile,
        "path": str(args.output.resolve()),
        "sha256": digest,
        "rows": len(output_rows),
        "concepts": len(counts),
        "images_per_concept": count,
        "source_dataset": str(DERIVED_CSV.relative_to(REPO_ROOT)),
        "source_dataset_sha256": dataset_hash,
        "seed_sources": dict(Counter(str(row["seed_source"]) for row in output_rows)),
    }, indent=2))


if __name__ == "__main__":
    main()
