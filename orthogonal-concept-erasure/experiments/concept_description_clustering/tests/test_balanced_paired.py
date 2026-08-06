from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

from concept_clustering.balanced_paired import (
    FACETS,
    FIXED_SUFFIX,
    LENGTH_RANGES,
    LENGTH_SCHEDULE,
    _forbidden_hits,
    _load_config,
    create_shared_slots,
    fit_spherical_kmeans,
)


ROOT = Path(__file__).resolve().parents[1]


def test_primary_config_and_shared_slots_are_exactly_balanced(tmp_path):
    config = _load_config(ROOT / "balanced_paired/configs/primary_4x50.json")
    slots = create_shared_slots(config, tmp_path)
    assert len(slots) == 50
    assert [row["facet"] for row in slots[::5]] == FACETS
    for facet in FACETS:
        local = [row for row in slots if row["facet"] == facet]
        assert [row["length_bin"] for row in local] == LENGTH_SCHEDULE
    assert Counter(row["generation_round"] for row in slots) == {
        "round_1": 10, "round_2": 10, "round_3": 10, "round_4": 10, "round_5": 10
    }
    assert config["readout"]["fixed_suffix"] == FIXED_SUFFIX


def test_secondary_config_contains_the_requested_eight_animals():
    config = _load_config(ROOT / "balanced_paired/configs/secondary_8x50.json")
    assert [row["name"] for row in config["concepts"]] == [
        "cat", "dog", "fox", "bear", "wolf", "rabbit", "deer", "horse"
    ]
    assert config["spherical_kmeans"]["k"] == 8


def test_forbidden_terms_detect_punctuation_and_hyphens():
    forbidden = {"cat": ["cat", "feline"], "bear": ["bear", "cub"]}
    assert _forbidden_hits("A FELINE-like outline watches quietly.", forbidden) == ["cat:feline"]
    assert _forbidden_hits("The bear, waits beside a cub.", forbidden) == ["bear:bear", "bear:cub"]
    assert _forbidden_hits("A broad woodland forager waits quietly.", forbidden) == []


def test_spherical_fit_boundary_has_no_label_parameter():
    parameters = set(inspect.signature(fit_spherical_kmeans).parameters)
    assert not ({"labels", "true_labels", "concepts", "targets"} & parameters)


def test_length_ranges_have_intentional_gaps():
    assert LENGTH_RANGES == {"short": (14, 19), "medium": (21, 26), "long": (28, 33)}
