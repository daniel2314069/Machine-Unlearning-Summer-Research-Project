from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("confuse5_runner", HERE / "run.py")
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def config() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "test",
        "shared": {
            "base_model": "test/model",
            "concept_type": "object",
            "editable_modules": "unet.attn2.to_v",
            "anchor_policy": {"kind": "per_target", "anchors": {"Alpha": "anchor a", "Beta": "anchor b"}},
            "retain_policy": {"kind": "explicit_global", "concepts": ["global retain"]},
            "oce": {
                "erase_scale": 1.0, "preserve_global_scale": 0.5,
                "preserve_concept_scale": 0.3, "lamb": 1.5,
                "expand_prompts": False, "dtype": "float32", "seed": 42,
                "device": "cuda:0",
            },
        },
        "groups": [{
            "id": "Group One", "concepts": ["Alpha", "Beta", "Gamma"],
            "targets": ["Alpha", "Beta"], "similar_non_targets": ["Gamma"],
        }],
    }


def write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_plan_has_two_single_and_one_joint_runs(tmp_path: Path) -> None:
    path = write_config(tmp_path, config())
    validated = RUNNER.load_and_validate(path)
    plan = RUNNER.build_plan(validated, path, tmp_path / "outputs", None, "both")
    assert [run["mode"] for run in plan["runs"]] == ["single", "single", "joint"]
    assert plan["runs"][0]["resolved_anchors"] == ["anchor a"]
    assert plan["runs"][2]["resolved_anchors"] == ["anchor a", "anchor b"]
    assert plan["runs"][0]["evaluation_non_target_concepts"] == ["Beta", "Gamma"]
    assert plan["runs"][2]["evaluation_non_target_concepts"] == ["Gamma"]
    assert {tuple(run["retain_concepts"]) for run in plan["runs"]} == {("global retain",)}


def test_conflicting_roles_are_rejected(tmp_path: Path) -> None:
    payload = config()
    payload["groups"][0]["similar_non_targets"] = ["Alpha", "Gamma"]
    path = write_config(tmp_path, payload)
    try:
        RUNNER.load_and_validate(path)
    except RUNNER.ConfigError as exc:
        assert "conflicting" in str(exc)
    else:
        raise AssertionError("Expected conflicting roles to fail")


def test_duplicate_normalized_concepts_are_rejected(tmp_path: Path) -> None:
    payload = config()
    payload["groups"][0]["concepts"] = ["Alpha", " alpha ", "Beta", "Gamma"]
    path = write_config(tmp_path, payload)
    try:
        RUNNER.load_and_validate(path)
    except RUNNER.ConfigError as exc:
        assert "duplicate normalized" in str(exc)
    else:
        raise AssertionError("Expected duplicate concepts to fail")


def test_oce_default_anchor_is_not_passed_as_a_blank_cli_value(tmp_path: Path) -> None:
    payload = config()
    payload["shared"]["anchor_policy"] = {"kind": "oce_default"}
    path = write_config(tmp_path, payload)
    validated = RUNNER.load_and_validate(path)
    plan = RUNNER.build_plan(validated, path, tmp_path / "outputs", None, "joint")
    command = RUNNER.oce_command(plan["runs"][0], validated["shared"])
    assert "--guide_concepts" not in command
    assert plan["runs"][0]["resolved_anchors"] == [" ", " "]
