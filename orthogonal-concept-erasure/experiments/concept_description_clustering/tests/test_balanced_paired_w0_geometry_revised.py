from __future__ import annotations

import inspect

import numpy as np

from scripts.eot_spherical_clustering import normalize_rows
from concept_clustering.balanced_paired_w0_geometry_revised import (
    CONDITIONS,
    CONCEPTS,
    _fit_all_clustering_without_labels,
    _project_raw,
    _prototype_analysis,
)


def test_revised_fit_boundary_has_no_labels_or_concepts():
    parameters = inspect.signature(_fit_all_clustering_without_labels).parameters
    assert not any("label" in name or "concept" in name for name in parameters)


def test_three_clustering_controls_keep_the_required_normalization_distinction():
    raw = np.asarray([
        [3.0, 0.0], [2.0, 0.2], [2.5, -0.1],
        [0.0, 4.0], [0.2, 3.0], [-0.1, 2.5],
    ], dtype=np.float32)
    raw_spaces = {("text", "eot"): raw}
    normalized_spaces = {("text", "eot"): normalize_rows(raw).astype(np.float32)}
    settings = {"k": 2, "n_init": 3, "max_iter": 30, "tolerance": 1e-6, "random_seed": 7}
    fits = _fit_all_clustering_without_labels(raw_spaces, normalized_spaces, settings)

    spherical = fits[("text", "eot", "spherical_normalized")]
    euclidean_raw = fits[("text", "eot", "euclidean_raw")]
    euclidean_normalized = fits[("text", "eot", "euclidean_normalized")]
    assert np.allclose(np.linalg.norm(spherical["centers"], axis=1), 1.0)
    assert spherical["input_normalized"] is True
    assert euclidean_raw["input_normalized"] is False
    assert euclidean_normalized["input_normalized"] is True
    assert euclidean_normalized["centers_normalized_during_fit"] is False


def test_projection_is_raw_until_the_separate_normalization_step():
    source = np.asarray([[3.0, 4.0]], dtype=np.float32)
    projected = _project_raw(source, None, "cpu")
    assert np.array_equal(projected, source)
    assert np.linalg.norm(projected[0]) == 5.0


def test_prototype_output_retains_every_animal_and_condition():
    basis = np.eye(len(CONCEPTS), dtype=np.float32)
    features = np.repeat(basis, 2, axis=0)
    rows = [
        {"concept": concept, "candidate_id": f"{concept}_{copy}"}
        for concept in CONCEPTS
        for copy in range(2)
    ]
    descriptions = {
        ("text", "eot"): features,
        ("text", "fixed_suffix"): features,
    }
    names = {
        ("text", "last"): basis,
        ("text", "eot"): basis,
        ("text", "fixed"): basis,
    }
    distances, summary, comparison, checks = _prototype_analysis(
        rows, [], descriptions, names
    )
    assert len(distances) == len(CONDITIONS) * len(CONCEPTS)
    assert set(distances["condition"]) == set(CONDITIONS)
    assert set(distances["animal"]) == set(CONCEPTS)
    assert (distances["own_centroid_rank"] == 1).all()
    assert (distances["nearest_other_concept"] != distances["animal"]).all()
    assert len(summary) == len(CONDITIONS)
    assert len(comparison) == len(CONCEPTS)
    assert np.isclose(checks["centroid_norm_min"], 1.0)
