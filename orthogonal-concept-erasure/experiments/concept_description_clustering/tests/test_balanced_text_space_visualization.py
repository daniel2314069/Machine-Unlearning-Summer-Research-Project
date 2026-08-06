from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd

from concept_clustering.balanced_text_space_visualization import (
    ANIMAL_COLORS,
    CONCEPTS,
    TSNE_CONFIG,
    _coordinate_rows,
    _padded_limits,
)


def test_visualization_configuration_matches_the_prespecified_design():
    assert list(ANIMAL_COLORS) == CONCEPTS
    assert TSNE_CONFIG == {
        "n_components": 2,
        "perplexity": 30,
        "init": "pca",
        "learning_rate": "auto",
        "max_iter": 2000,
        "metric": "cosine",
    }


def test_true_and_predicted_plots_can_share_exact_axis_limits():
    coordinates = np.asarray([[-2.0, 1.0], [3.0, -4.0], [0.5, 2.0]])
    limits = _padded_limits(coordinates)
    assert limits["x"][0] < -2.0 and limits["x"][1] > 3.0
    assert limits["y"][0] < -4.0 and limits["y"][1] > 2.0


def test_coordinate_rows_preserve_cached_assignment_fields():
    frame = pd.DataFrame({
        "sample_index": [0, 1],
        "candidate_id": ["a", "b"],
        "description": ["first", "second"],
        "true_concept": ["cat", "dog"],
        "predicted_cluster": [7, 1],
        "matched_predicted_concept": ["cat", "dog"],
        "prediction_correct": [True, True],
    })
    coordinates = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    rows = _coordinate_rows("eot", "tsne", "42", coordinates, frame, None)
    assert [row["description_id"] for row in rows] == ["a", "b"]
    assert [row["raw_cluster_id"] for row in rows] == [7, 1]
    assert all(row["prediction_correct"] for row in rows)


def test_visualization_module_has_no_clustering_or_w0_imports():
    import concept_clustering.balanced_text_space_visualization as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "sklearn.cluster" not in imports
    assert not ({"KMeans", "fit_spherical_kmeans", "original_projection_modules"} & calls)
