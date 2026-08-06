from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pandas as pd

from concept_clustering import cat_to_dog_description_shift as shift


def test_projection_uses_fixed_original_anchors_and_expected_signs():
    w0 = np.eye(2, dtype=np.float32)
    wcat = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    c_cat = np.asarray([1.0, 0.0], dtype=np.float32)
    c_dog = np.asarray([0.0, 1.0], dtype=np.float32)
    descriptions = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    result = shift._project_descriptions(descriptions, w0, wcat, c_cat, c_dog)

    assert result["delta_dog"][0] > 0
    assert result["delta_cat"][0] < 0
    assert bool(result["intended_direction"][0])
    assert np.allclose(result["before_norms"], 1.0)
    assert np.allclose(result["after_norms"], 1.0)
    assert np.allclose(result["cat_anchor_norm"], 1.0)
    assert np.allclose(result["dog_anchor_norm"], 1.0)


def test_canonical_cat_sanity_uses_same_requested_deltas():
    w0 = np.eye(2, dtype=np.float32)
    wcat = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    c_cat = np.asarray([1.0, 0.0], dtype=np.float32)
    c_dog = np.asarray([0.0, 1.0], dtype=np.float32)

    result = shift._canonical_cat_shift(w0, wcat, c_cat, c_dog)

    assert np.isclose(result["canonical_delta_dog"], 1.0)
    assert np.isclose(result["canonical_delta_cat"], -1.0)
    assert result["intended_direction"] is True


def test_aggregation_retains_requested_statistics_and_proportions():
    values = pd.DataFrame({
        "layer_index": [0, 0, 0, 0],
        "layer_name": ["layer"] * 4,
        "true_concept": ["cat"] * 4,
        "delta_cat": [-0.2, -0.1, 0.1, -0.3],
        "delta_dog": [0.3, 0.2, 0.1, -0.2],
    })

    row = shift._aggregate_shifts(values).iloc[0]

    assert row["count"] == 4
    assert np.isclose(row["mean_delta_cat"], -0.125)
    assert np.isclose(row["median_delta_dog"], 0.15)
    assert np.isclose(row["proportion_toward_dog"], 0.75)
    assert np.isclose(row["proportion_away_from_cat"], 0.75)
    assert np.isclose(row["joint_intended_direction"], 0.5)


def test_edit_metadata_requires_verified_cat_to_dog_hash(tmp_path):
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"verified checkpoint fixture")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata = {
        "method": "OCE",
        "model_id": "CompVis/stable-diffusion-v1-4",
        "edit_concept": "cat",
        "guide_concept": "dog",
        "sha256": {"edited_weights": digest},
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    loaded, actual = shift._validate_edit_metadata(
        metadata_path, weights, "CompVis/stable-diffusion-v1-4"
    )

    assert loaded["edit_concept"] == "cat"
    assert loaded["guide_concept"] == "dog"
    assert actual == digest


def test_module_contains_no_clustering_or_edited_dog_anchor():
    audit = shift._strict_scope_source_audit()
    projection_source = inspect.getsource(shift._project_descriptions)
    canonical_source = inspect.getsource(shift._canonical_cat_shift)
    assert audit["no_clustering_calls"]
    assert audit["edited_dog_anchor_absent_from_projection_functions"]
    assert "wcat @ c_dog" not in projection_source
    assert "wcat @ c_dog" not in canonical_source
