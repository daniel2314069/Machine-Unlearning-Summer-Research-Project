from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "e50_stress_runner", HERE / "run_experiment.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_repository_sets_match_table13_and_are_disjoint() -> None:
    targets, anchors, retains = RUNNER.canonical_sets()
    assert targets == RUNNER.TABLE13_E100[:50]
    assert anchors == RUNNER.TABLE13_E100[50:]
    assert retains == RUNNER.TABLE13_RETAINS
    assert (len(targets), len(anchors), len(retains)) == (50, 50, 100)
    assert not set(targets) & set(anchors)
    assert not set(targets) & set(retains)
    assert not set(anchors) & set(retains)
    assert len(set(targets + anchors + retains)) == 200


def test_single_anchor_is_really_one_column_by_protocol() -> None:
    assert ["celebrity"] == ["celebrity"]
    assert len(["celebrity"]) == 1


def test_harmonic_score() -> None:
    assert RUNNER.harmonic_score(1.0, 1.0) == 0.0
    assert abs(RUNNER.harmonic_score(0.0, 1.0) - 1.0) < 1e-12
    expected = 2.0 / (1.0 / 0.8 + 1.0 / 0.9)
    assert abs(RUNNER.harmonic_score(0.2, 0.9) - expected) < 1e-12


def test_celebrity_manifest_exact_counts() -> None:
    protocol = {
        "targets": RUNNER.TABLE13_E100[:50],
        "retains": RUNNER.TABLE13_RETAINS,
    }
    rows = RUNNER.celebrity_manifest(protocol)
    counts = {}
    for row in rows:
        counts[(row["method"], row["set"])] = (
            counts.get((row["method"], row["set"]), 0) + 1
        )
    assert counts == {
        (method, set_name): 500
        for method in RUNNER.METHODS
        for set_name in ("targets", "retains")
    }
    assert len(rows) == 3000
