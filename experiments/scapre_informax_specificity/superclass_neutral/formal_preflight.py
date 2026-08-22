#!/usr/bin/env python
"""Dependency-free audit before launching the expensive formal run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO = HERE.parents[2]
BASE_CONFIG = PARENT / "config.json"
PROTOCOL_BUILDER = PARENT / "build_protocol.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def controlled_keys(path: Path) -> set[tuple[str, ...]]:
    with path.open(newline="") as handle:
        return {
            tuple(row[field] for field in (
                "group", "role", "concept", "sample_index", "prompt", "seed", "seed_source"
            ))
            for row in csv.DictReader(handle)
        }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    prior = args.prior_run.resolve()
    integrity_path = prior / "reproducibility" / "integrity_report.json"
    integrity = json.loads(integrity_path.read_text())
    if integrity.get("status") != "passed" or integrity.get("edit_seeds") != config["edit_seeds"]:
        raise RuntimeError("prior robustness integrity report did not pass")
    result_manifest = json.loads((prior / "results" / "result_manifest.json").read_text())
    if result_manifest.get("judgment") != config["prior_robustness"]["expected_judgment"]:
        raise RuntimeError("prior robustness judgment differs from the pinned completed result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="superclass_preflight_", dir=args.output.parent) as temporary:
        protocol = Path(temporary) / "protocol.csv"
        protocol_output = subprocess.check_output([
            sys.executable, str(PROTOCOL_BUILDER), "--config", str(BASE_CONFIG),
            "--output", str(protocol), "--profile", "formal",
        ], cwd=REPO, text=True)
        protocol_manifest = json.loads(protocol_output)
        if protocol_manifest["sha256"] != config["prior_robustness"]["expected_protocol_sha256"]:
            raise RuntimeError("current formal protocol differs from the pinned baseline protocol")
        expected_keys = controlled_keys(protocol)
    if len(expected_keys) != 3000:
        raise RuntimeError("current formal protocol does not contain 3,000 unique keys")
    observations = []
    for seed in config["edit_seeds"]:
        for variant in ("official", "matched_retain"):
            root = prior / "seeds" / str(seed) / "evaluation" / variant
            score = root / "scores.csv"
            manifest = root / "evaluation_manifest.json"
            if count_rows(score) != 3000 or not manifest.is_file() or not (root / "COMPLETED").is_file():
                raise RuntimeError(f"prior seed/variant is incomplete: {seed}/{variant}")
            if controlled_keys(score) != expected_keys:
                raise RuntimeError(f"prior generation keys differ from current protocol: {seed}/{variant}")
            recorded_hash = integrity.get("score_sha256", {}).get(str(seed), {}).get(variant)
            if recorded_hash != sha256(score):
                raise RuntimeError(f"prior score hash differs from its integrity report: {seed}/{variant}")
            observations.append({
                "seed": seed, "variant": variant,
                "scores_sha256": sha256(score), "manifest_sha256": sha256(manifest),
            })
    if len(observations) != 10:
        raise RuntimeError("prior baseline observation count changed")
    args.output.write_text(json.dumps({
        "status": "passed", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "prior_run": str(prior),
        "prior_integrity_sha256": sha256(integrity_path), "observations": observations,
        "protocol": protocol_manifest, "generation_keys_identical": True,
        "baseline_score_evaluations_to_rerun": 0,
        "new_superclass_image_records": 15000,
        "qualitative_checkpoint_policy": "reuse existing seed-20260820 checkpoints; deterministically recreate only missing checkpoints",
        "loaded_model": False, "generated_images": False, "downloaded_assets": False,
    }, indent=2) + "\n")
    print("Formal preflight passed without loading a model or generating images.")


if __name__ == "__main__":
    main()
