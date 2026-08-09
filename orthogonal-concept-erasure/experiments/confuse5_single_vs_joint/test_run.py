from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol  # noqa: E402
import run  # noqa: E402


def test_primary_protocol_resolves_ten_single_and_five_joint() -> None:
    config, anchors = protocol.load_protocol(HERE / "config.json")
    specs = protocol.checkpoint_specs(config, anchors)
    assert sum(item["mode"] == "single" for item in specs) == 10
    assert sum(item["mode"] == "joint" for item in specs) == 5
    assert all(item["retain_concepts"] for item in specs)
    assert all(len(item["retain_concepts"]) == 3 for item in specs)


def test_single_and_joint_reuse_identical_target_anchors() -> None:
    config, anchors = protocol.load_protocol(HERE / "config.json")
    specs = protocol.checkpoint_specs(config, anchors)
    for group in config["groups"]:
        joint = next(item for item in specs if item["group_id"] == group["id"] and item["mode"] == "joint")
        singles = [item for item in specs if item["group_id"] == group["id"] and item["mode"] == "single"]
        assert joint["anchors"] == [anchors[target] for target in group["targets"]]
        assert [item["anchors"][0] for item in singles] == joint["anchors"]
        assert all(item["retain_concepts"] == joint["retain_concepts"] for item in singles)


def test_official_expansion_order_is_all_bare_then_extras() -> None:
    config, _ = protocol.load_protocol(HERE / "config.json")
    prompts = protocol.expanded_prompts(["alpha", "beta"], config)
    assert prompts[:2] == ["alpha", "beta"]
    assert prompts[2:7] == [
        "image of alpha", "photo of alpha", "portrait of alpha",
        "picture of alpha", "painting of alpha",
    ]
    assert prompts[7:] == [
        "image of beta", "photo of beta", "portrait of beta",
        "picture of beta", "painting of beta",
    ]


def test_locked_hyperparameters_are_not_legacy_parser_defaults() -> None:
    config, _ = protocol.load_protocol(HERE / "config.json")
    oce = config["oce"]
    assert (oce["lambda_e"], oce["lambda_0"], oce["lambda_r"], oce["lamb_repo_regularizer"]) == (1000.0, 50.0, 1.0, 10.0)
    assert (oce["lambda_e"], oce["lambda_0"], oce["lambda_r"], oce["lamb_repo_regularizer"]) != (1.0, 0.5, 0.3, 1.5)


def test_checkpoint_plan_contains_full_provenance() -> None:
    plan, config, anchors = run.build_plan(HERE / "config.json")
    assert plan["resolved_config"] == config
    assert plan["full_anchor_mapping"] == anchors
    assert plan["single_count"] == 10
    assert plan["joint_count"] == 5
    assert plan["k0_path"].endswith("official_repo_primary_v1/artifacts/K0.pt")


def test_anchor_file_is_exactly_the_approved_mapping() -> None:
    payload = json.loads((HERE / "anchors.json").read_text(encoding="utf-8"))
    assert payload["anchors"] == {
        "golden retriever": "cocker spaniel",
        "labrador retriever": "beagle",
        "tabby": "lynx",
        "tiger cat": "lion",
        "orange": "banana",
        "lemon": "pineapple",
        "yawl": "canoe",
        "lifeboat": "ferry",
        "soccer ball": "basketball",
        "volleyball": "baseball",
    }
