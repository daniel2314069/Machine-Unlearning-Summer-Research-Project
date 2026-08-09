from __future__ import annotations

import builtins
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline  # noqa: E402
import protocol  # noqa: E402


def test_plan_has_exact_image_budget_and_hard_stops() -> None:
    plan, _, _ = pipeline.build_plan(HERE / "config.json")
    assert plan["dataset_row_count"] == 12500
    assert plan["checkpoint_count"] == 15
    assert plan["image_counts"] == {
        "anchor_sanity_original": 80,
        "smoke_original": 128,
        "smoke_single": 128,
        "smoke_joint": 128,
        "formal_original_regenerated": 0,
        "formal_single": 25000,
        "formal_joint": 12500,
        "formal_total_new_edited": 37500,
    }
    assert plan["hard_stops"] == [
        "anchor_sanity_failure",
        "original_reproduction_hash_mismatch",
        "any_single_smoke_drop_below_4_of_32",
    ]


def test_dataset_has_25_ordered_classes_of_500_rows() -> None:
    config, _ = protocol.load_protocol(HERE / "config.json")
    rows, by_class = pipeline.load_dataset(config)
    assert len(rows) == 12500
    assert len(by_class) == 25
    assert {len(values) for values in by_class.values()} == {500}
    assert rows == sorted(rows, key=lambda row: row["case_number"])
    assert all(values == sorted(values, key=lambda row: row["case_number"]) for values in by_class.values())


def test_plan_builder_never_imports_model_libraries(monkeypatch) -> None:
    forbidden = {"torch", "torchvision", "diffusers", "safetensors", "PIL"}
    imported: list[str] = []
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in forbidden:
            imported.append(name)
            raise AssertionError(f"model import during planning: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    plan, _, _ = pipeline.build_plan(HERE / "config.json")
    assert plan["image_counts"]["formal_total_new_edited"] == 37500
    assert imported == []


def test_legacy_reference_is_separate_and_conditionally_reusable() -> None:
    config, _ = protocol.load_protocol(HERE / "config.json")
    reference, archive_root = pipeline._legacy_reference(config)
    assert reference["status"] == "conditional_reusable_original_reference"
    assert reference["invalid_edited_checkpoints_are_not_part_of_this_reference"] is True
    assert archive_root.name == "invalid_for_primary__pilot_default_config"
    assert reference["unavailable_metrics"] == [
        "target_probability", "raw_target_logit", "top5_labels", "top5_probabilities"
    ]


def test_cli_exposes_only_protocol_stages() -> None:
    for stage in ("plan", "k0", "anchor-sanity", "checkpoints", "smoke", "formal", "aggregate", "all", "status"):
        assert pipeline.parse_args([stage]).stage == stage
