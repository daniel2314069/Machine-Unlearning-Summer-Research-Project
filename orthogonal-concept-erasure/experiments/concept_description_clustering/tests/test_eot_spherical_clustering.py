from __future__ import annotations

import inspect

import numpy as np
import torch

from scripts.eot_spherical_clustering import (
    _validate_eot_batch,
    build_eot_representations,
    cluster_representations_without_labels,
    fit_spherical_kmeans,
)


def test_attention_mask_selects_actual_eot_not_last_padding_position():
    hidden = torch.zeros(2, 7, 4)
    hidden[0, 3] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    hidden[1, 5] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
    ])
    input_ids = torch.tensor([
        [49406, 10, 11, 49407, 49407, 49407, 49407],
        [49406, 20, 21, 22, 23, 49407, 49407],
    ])
    vectors, indices = _validate_eot_batch(hidden, mask, input_ids, eos_token_id=49407)
    assert indices.tolist() == [3, 5]
    assert vectors.tolist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_centering_uses_one_global_mean_and_outputs_unit_rows():
    embeddings = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 1.0],
    ])
    raw, centered, mean = build_eot_representations(embeddings)
    np.testing.assert_allclose(mean, embeddings.mean(axis=0))
    np.testing.assert_allclose(np.linalg.norm(raw, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(centered, axis=1), 1.0)


def test_true_spherical_kmeans_is_deterministic_and_recovers_separable_groups():
    rng = np.random.default_rng(9)
    features = []
    for group in range(4):
        center = np.zeros(12)
        center[group] = 1.0
        for _ in range(15):
            vector = center + rng.normal(0, 0.015, 12)
            vector /= np.linalg.norm(vector)
            features.append(vector)
    features = np.asarray(features)
    first = fit_spherical_kmeans(features, k=4, n_init=8, random_seed=123)
    second = fit_spherical_kmeans(features, k=4, n_init=8, random_seed=123)
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_allclose(first.centers, second.centers)
    np.testing.assert_allclose(np.linalg.norm(first.centers, axis=1), 1.0)
    assert len(set(first.labels.tolist())) == 4
    assert first.objective > 59.0


def test_unsupervised_boundary_has_no_label_parameter():
    fit_parameters = set(inspect.signature(fit_spherical_kmeans).parameters)
    boundary_parameters = set(inspect.signature(cluster_representations_without_labels).parameters)
    forbidden = {"labels", "true_labels", "true_ids", "concepts", "concept_labels"}
    assert not fit_parameters.intersection(forbidden)
    assert not boundary_parameters.intersection(forbidden)
