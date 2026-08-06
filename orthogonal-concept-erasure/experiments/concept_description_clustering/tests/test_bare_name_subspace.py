import numpy as np
import torch

from concept_clustering.bare_name_subspace import (
    _basis_from_rows,
    _capture,
    _deterministic_splits,
)
from concept_clustering.oce_uce_bare import oce_uce_last_token_position


def test_exact_oce_uce_attention_mask_rule_selects_last_content_token():
    # SOT, two content tokens, EOT, then padding.
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    assert oce_uce_last_token_position(attention_mask) == 2


def test_capture_is_bounded_and_rank_sensitive():
    descriptions = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [0.0, 1.0, 0.0],
    ])
    basis, numerical_rank = _basis_from_rows(descriptions)
    assert numerical_rank == 2
    query = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    captured = _capture(query, basis, None)
    assert np.all((captured >= 0.0) & (captured <= 1.0))
    assert captured[0] < 1e-10
    assert captured[1] > 1.0 - 1e-10


def test_deterministic_balanced_splits_are_disjoint():
    labels = np.repeat(np.arange(4), 50)
    left = _deterministic_splits(labels, 4, 3, 40, 10, 123)
    right = _deterministic_splits(labels, 4, 3, 40, 10, 123)
    for split_left, split_right in zip(left, right):
        for concept in range(4):
            train, heldout = split_left[concept]
            train_again, heldout_again = split_right[concept]
            assert np.array_equal(train, train_again)
            assert np.array_equal(heldout, heldout_again)
            assert len(train) == 40
            assert len(heldout) == 10
            assert not set(train).intersection(heldout)
