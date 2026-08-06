from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("confuse5_pipeline", HERE / "pipeline.py")
assert SPEC and SPEC.loader
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def test_repository_partial_plan_has_expected_coverage_and_counts(tmp_path: Path) -> None:
    plan, _, _ = PIPELINE.build_pipeline_plan(
        config_path=HERE / "config.json",
        dataset_override=None,
        output_root=tmp_path / "evaluation",
        coverage_mode="partial",
        raw_groups=None,
    )
    assert plan["coverage_status"] == "partial"
    assert plan["execution_allowed"] is True
    assert len(plan["coverage"]["available_classes"]) == 15
    assert plan["coverage"]["missing_classes"] == [
        "Chesapeake Bay retriever", "pug", "Siamese cat", "Egyptian cat",
        "fig", "Granny Smith", "catamaran", "schooner", "rugby ball",
        "ping-pong ball",
    ]
    assert len(plan["jobs"]) == 60
    assert plan["image_counts"] == {
        "total": 30000,
        "original": 7500,
        "single": 15000,
        "joint": 7500,
        "peak_retained_images_with_purge": 500,
    }


def test_complete_plan_is_blocked_while_official_rows_are_missing(tmp_path: Path) -> None:
    plan, _, _ = PIPELINE.build_pipeline_plan(
        config_path=HERE / "config.json",
        dataset_override=None,
        output_root=tmp_path / "evaluation",
        coverage_mode="complete",
        raw_groups=None,
    )
    assert plan["execution_allowed"] is False
    try:
        PIPELINE.require_execution_allowed(plan)
    except PIPELINE.PipelineError as exc:
        assert "missing=" in str(exc)
    else:
        raise AssertionError("Complete execution should be blocked")


def test_synthetic_complete_25_class_plan_has_50000_images(tmp_path: Path) -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    dataset = tmp_path / "complete.csv"
    case_number = 0
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("case_number", "prompt", "class", "evaluation_seed")
        )
        writer.writeheader()
        for group in config["groups"]:
            for concept in group["concepts"]:
                for index in range(500):
                    writer.writerow({
                        "case_number": case_number,
                        "prompt": f"synthetic prompt for {concept}",
                        "class": concept,
                        "evaluation_seed": index,
                    })
                    case_number += 1
    plan, _, _ = PIPELINE.build_pipeline_plan(
        config_path=HERE / "config.json",
        dataset_override=dataset,
        output_root=tmp_path / "evaluation",
        coverage_mode="complete",
        raw_groups=None,
    )
    assert plan["coverage_status"] == "complete"
    assert plan["execution_allowed"] is True
    assert len(plan["jobs"]) == 100
    assert plan["image_counts"]["total"] == 50000
    assert plan["image_counts"]["peak_retained_images_with_purge"] == 500


def test_one_group_partial_count_is_6000(tmp_path: Path) -> None:
    plan, _, _ = PIPELINE.build_pipeline_plan(
        config_path=HERE / "config.json",
        dataset_override=None,
        output_root=tmp_path / "evaluation",
        coverage_mode="partial",
        raw_groups=["dogs"],
    )
    assert len(plan["jobs"]) == 12
    assert plan["image_counts"]["total"] == 6000


def test_smoke_plan_is_12_images_and_three_models(tmp_path: Path) -> None:
    plan, _, _ = PIPELINE.build_pipeline_plan(
        config_path=HERE / "config.json",
        dataset_override=None,
        output_root=tmp_path / "smoke",
        coverage_mode="partial",
        raw_groups=["dogs"],
        rows_per_concept=2,
        smoke=True,
        smoke_single_target="golden retriever",
    )
    assert plan["image_counts"]["total"] == 12
    assert {job["model_type"] for job in plan["jobs"]} == {"original", "single", "joint"}
    assert {job["evaluated_concept"] for job in plan["jobs"]} == {
        "golden retriever", "german shepherd",
    }


def test_image_confirmation_must_match_exact_count() -> None:
    plan = {"image_counts": {"total": 30000}}
    PIPELINE.require_image_confirmation(plan, 30000)
    try:
        PIPELINE.require_image_confirmation(plan, 29999)
    except PIPELINE.PipelineError as exc:
        assert "--confirm-image-count 30000" in str(exc)
    else:
        raise AssertionError("Mismatched confirmation should fail")


class FakeImage:
    def save(self, path: Path, format: str) -> None:
        assert format == "PNG"
        Path(path).write_bytes(b"synthetic-png")


class FakeGenerator:
    def activate(self, job: dict) -> None:
        assert job["model_type"] == "original"

    def generate_one(self, prompt: str, seed: int) -> FakeImage:
        assert prompt
        assert seed >= 0
        return FakeImage()


class FakeEvaluator:
    def classify(self, paths: list[Path], concept: str) -> list[dict]:
        return [
            {
                "image_path": str(path),
                "expected_index": 1,
                "expected_category": concept,
                "predicted_index": 1,
                "predicted_category": concept,
                "correct": True,
            }
            for path in paths
        ]


def test_successful_evaluation_persists_results_then_purges_images(tmp_path: Path) -> None:
    output_root = tmp_path / "evaluation"
    rows = [
        {
            "case_number": index,
            "prompt": "an image of alpha",
            "class": "alpha",
            "evaluation_seed": 100 + index,
            "source_line": index + 2,
        }
        for index in range(2)
    ]
    image_dir = output_root / "images" / "original" / "group" / "alpha"
    job = {
        "job_id": "original__group__alpha",
        "job_fingerprint": "job-fingerprint",
        "model_type": "original",
        "group_id": "group",
        "group_targets": ["alpha", "beta"],
        "group_similar_non_targets": ["gamma"],
        "single_target": None,
        "target_concepts": [],
        "evaluated_concept": "alpha",
        "prompt_count": 2,
        "rows": rows,
        "checkpoint_path": None,
        "checkpoint_metadata_path": None,
        "image_dir": str(image_dir),
        "manifest_path": str(output_root / "manifests" / "job.json"),
        "result_path": str(output_root / "evaluations" / "shards" / "job.json"),
    }
    plan = {
        "plan_fingerprint": "plan-fingerprint",
        "generation": {"width": 512, "height": 512},
        "classifier": {"implementation": "synthetic"},
    }
    PIPELINE.generate_job(
        job, plan, FakeGenerator(), output_root=output_root,
        skip_existing=False, overwrite=False,
    )
    assert len(list(image_dir.glob("*.png"))) == 2
    result = PIPELINE.evaluate_job(
        job, plan, FakeEvaluator(), output_root=output_root,
        skip_existing=False, overwrite=False, purge=True,
    )
    assert result["accuracy"] == 1.0
    assert result["images_purged"] is True
    assert list(image_dir.glob("*.png")) == []
    manifest = json.loads(Path(job["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "purged"
    assert {item["image_status"] for item in manifest["items"]} == {"purged"}


def test_plan_builder_does_not_import_model_libraries(tmp_path: Path, monkeypatch) -> None:
    forbidden = {"torch", "diffusers", "torchvision", "safetensors"}
    imported: list[str] = []
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            imported.append(name)
            raise AssertionError(f"Planning imported model library {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    PIPELINE.build_pipeline_plan(
        config_path=HERE / "config.json",
        dataset_override=None,
        output_root=tmp_path / "evaluation",
        coverage_mode="partial",
        raw_groups=["dogs"],
    )
    assert imported == []


def test_managed_path_rejects_external_target(tmp_path: Path) -> None:
    root = tmp_path / "images"
    outside = tmp_path / "outside.png"
    try:
        PIPELINE.ensure_within(outside, root)
    except PIPELINE.PipelineError as exc:
        assert "outside managed image root" in str(exc)
    else:
        raise AssertionError("External deletion target should be rejected")
