#!/usr/bin/env python
"""Safely finalize the completed 20260821 formal run after metadata-only failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluator_fingerprint import compare_evaluator_manifests


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ELIGIBLE_GENERATION_COMMIT = "9ca7b5e9c4ab626027fb8fe0bd32fca51e8faf89"
SEEDS = (20260820, 20260821, 20260822, 20260823, 20260824)
NEW_SEEDS = SEEDS[1:]
VARIANTS = ("official", "matched_retain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
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


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"required post-hoc input is missing: {path}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads(args.config.read_text())
    run_config = json.loads((run_dir / "actual_config.json").read_text())
    base_config = json.loads(args.base_config.read_text())
    run_base_config = json.loads((run_dir / "base_config.json").read_text())
    if run_config != config or run_base_config != base_config:
        raise RuntimeError("current frozen configs differ from the failed formal run")

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("profile") != "formal":
        raise RuntimeError("post-hoc finalizer accepts only a formal run")
    if manifest.get("git_commit") != ELIGIBLE_GENERATION_COMMIT:
        raise RuntimeError(
            f"generation commit is not eligible: {manifest.get('git_commit')}"
        )
    if manifest.get("git_status_start") != [] or manifest.get("git_status_end") != []:
        raise RuntimeError("generation run did not record clean start/end git status")
    if git_output("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("current repository is dirty during post-hoc finalization")

    for relative, expected in config["source_controls"].items():
        if manifest["source_sha256"].get(relative) != expected:
            raise RuntimeError(f"generation source hash changed for {relative}")
        if sha256(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"current controlled source hash changed for {relative}")

    observations: list[tuple[str, dict]] = []
    score_hashes: dict[str, dict[str, str]] = {}
    for seed in SEEDS:
        score_hashes[str(seed)] = {}
        for variant in VARIANTS:
            evaluation = run_dir / "seeds" / str(seed) / "evaluation" / variant
            score_path = evaluation / "scores.csv"
            evaluator_manifest_path = evaluation / "evaluation_manifest.json"
            require_file(evaluation / "COMPLETED")
            require_file(score_path)
            require_file(evaluator_manifest_path)
            rows = read_csv(score_path)
            if len(rows) != 3000:
                raise RuntimeError(f"seed {seed} {variant} has {len(rows)} scores")
            keys = {
                tuple(row[field] for field in (
                    "group", "role", "concept", "sample_index", "prompt", "seed", "seed_source"
                ))
                for row in rows
            }
            if len(keys) != 3000:
                raise RuntimeError(f"seed {seed} {variant} has duplicate generation keys")
            if seed in NEW_SEEDS:
                missing_images = sum(not Path(row["image_path"]).is_file() for row in rows)
                if missing_images:
                    raise RuntimeError(
                        f"seed {seed} {variant} is missing {missing_images} images before archive cleanup"
                    )
            observations.append((
                f"seed={seed},variant={variant}",
                json.loads(evaluator_manifest_path.read_text()),
            ))
            score_hashes[str(seed)][variant] = sha256(score_path)
        if seed in NEW_SEEDS:
            for variant in VARIANTS:
                require_file(
                    run_dir / "seeds" / str(seed) / "stages" / f"edit_{variant}.completed"
                )
                require_file(
                    run_dir / "seeds" / str(seed) / "stages" / f"informax_rng_{variant}.json"
                )

    evaluator_comparison = compare_evaluator_manifests(observations)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "aggregate_seed_results.py"),
            "--run-dir",
            str(run_dir),
            "--config",
            str(args.config.resolve()),
            "--base-config",
            str(args.base_config.resolve()),
            "--profile",
            "formal",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    required_outputs = {
        "results/per_seed.csv": 6,
        "results/per_group_seed.csv": 26,
        "results/per_concept_seed.csv": 126,
        "results/aggregate_across_seeds.csv": 4,
        "results/per_group_robustness.csv": 6,
        "results/per_retain_robustness.csv": 16,
        "results/informax_seed_diagnostics.csv": 6,
    }
    for relative, expected_lines in required_outputs.items():
        path = run_dir / relative
        require_file(path)
        with path.open() as handle:
            actual_lines = sum(1 for _ in handle)
        if actual_lines != expected_lines:
            raise RuntimeError(
                f"post-hoc output {relative} has {actual_lines} lines, expected {expected_lines}"
            )
    integrity_path = run_dir / "reproducibility" / "integrity_report.json"
    integrity = json.loads(integrity_path.read_text())
    if integrity.get("status") != "passed":
        raise RuntimeError("post-hoc aggregation integrity report did not pass")

    current_commit = git_output("rev-parse", "HEAD")
    provenance_dir = run_dir / "posthoc_provenance"
    provenance_dir.mkdir(exist_ok=True)
    provenance_files = (
        "aggregate_seed_results.py",
        "cleanup_images.sh",
        "evaluator_fingerprint.py",
        "package_results.sh",
        "posthoc_finalize.py",
        "finalize_server.sh",
        "json_stdlib.py",
        "posthoc_finalize_worker.sh",
    )
    provenance_hashes = {}
    for name in provenance_files:
        source = SCRIPT_DIR / name
        require_file(source)
        shutil.copy2(source, provenance_dir / name)
        provenance_hashes[name] = sha256(source)

    finalized_at = datetime.now(timezone.utc).isoformat()
    finalization = {
        "status": "passed",
        "reason": (
            "original aggregation compared the ordering of Diffusers "
            "scheduler_config._use_default_values, which is semantically unordered"
        ),
        "generation_commit": manifest["git_commit"],
        "finalizer_commit": current_commit,
        "finalized_at_utc": finalized_at,
        "evaluator_comparison": evaluator_comparison,
        "score_sha256": score_hashes,
        "reused_existing_edits": True,
        "reused_existing_24000_generated_images": True,
        "reran_model_editing": False,
        "regenerated_images": False,
        "posthoc_source_sha256": provenance_hashes,
    }
    finalization_path = run_dir / "reproducibility" / "posthoc_finalization.json"
    finalization_path.write_text(json.dumps(finalization, indent=2) + "\n")
    manifest["posthoc_finalization"] = {
        "status": "passed",
        "finalizer_commit": current_commit,
        "manifest": "reproducibility/posthoc_finalization.json",
    }
    manifest["finished_at_utc"] = finalized_at
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(run_dir / "results" / "summary.md", run_dir / "summary.md")
    (run_dir / "worker_complete.json").write_text(json.dumps({
        "completed_at_utc": finalized_at,
        "profile": "formal",
        "edit_seeds": list(SEEDS),
        "summary": str((run_dir / "results" / "summary.md").resolve()),
        "posthoc_finalized": True,
    }, indent=2) + "\n")
    print("Post-hoc aggregation validation passed without editing or generating images.")


if __name__ == "__main__":
    main()
