from __future__ import annotations

import ast
import inspect

import numpy as np

from concept_clustering.eot_local_geometry_diagnostic import (
    SPECTRAL_RUNS,
    build_cosine_knn_graph_without_labels,
    find_cosine_neighbors_without_labels,
    fit_spectral_without_labels,
)


def test_predetermined_spectral_runs_are_exact_and_primary_is_unique():
    assert SPECTRAL_RUNS == [
        {"n_neighbors": 5, "random_state": 42, "primary": False},
        {"n_neighbors": 10, "random_state": 0, "primary": False},
        {"n_neighbors": 10, "random_state": 1, "primary": False},
        {"n_neighbors": 10, "random_state": 42, "primary": True},
        {"n_neighbors": 15, "random_state": 42, "primary": False},
    ]
    assert sum(bool(run["primary"]) for run in SPECTRAL_RUNS) == 1


def test_neighbor_graph_and_fit_boundaries_do_not_accept_labels():
    for function in (
        find_cosine_neighbors_without_labels,
        build_cosine_knn_graph_without_labels,
        fit_spectral_without_labels,
    ):
        parameters = inspect.signature(function).parameters
        assert not any("label" in name or "concept" in name for name in parameters)


def test_cosine_neighbors_exclude_self_and_graph_is_symmetric_nonnegative():
    features = np.asarray([
        [1.0, 0.0],
        [0.99, 0.1],
        [0.0, 1.0],
        [0.1, 0.99],
    ], dtype=np.float64)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    indices, distances = find_cosine_neighbors_without_labels(features, 1)
    assert indices.shape == (4, 1)
    assert distances.shape == (4, 1)
    assert not np.any(indices == np.arange(4)[:, None])
    assert indices[:, 0].tolist() == [1, 0, 3, 2]
    graph, diagnostics = build_cosine_knn_graph_without_labels(features, 1)
    assert (graph != graph.T).nnz == 0
    assert np.all(graph.data >= 0)
    assert np.allclose(graph.diagonal(), 0.0)
    assert diagnostics["labels_used"] is False


def test_spectral_fit_returns_one_assignment_per_sample_without_labels():
    features = np.asarray([
        [1.0, 0.00], [1.0, 0.05], [1.0, -0.05],
        [0.0, 1.00], [0.05, 1.0], [-0.05, 1.0],
    ], dtype=np.float64)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    graph, _ = build_cosine_knn_graph_without_labels(features, 2)
    clusters, warning_messages = fit_spectral_without_labels(
        graph, n_clusters=2, random_state=42, n_init=5
    )
    assert clusters.shape == (6,)
    assert len(np.unique(clusters)) == 2
    assert isinstance(warning_messages, list)


def test_module_has_no_dimensionality_reduction_or_oce_calls():
    import concept_clustering.eot_local_geometry_diagnostic as module

    tree = ast.parse(inspect.getsource(module))
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
    assert not ({"sklearn.decomposition", "sklearn.manifold"} & imports)
    assert not ({"PCA", "TSNE", "fit_spherical_kmeans", "Orthogonal_Erase"} & calls)
