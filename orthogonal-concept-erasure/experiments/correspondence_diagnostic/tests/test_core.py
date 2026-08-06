from __future__ import annotations

import numpy as np
import pytest
import torch

from oce_correspondence.core import (
    PairSpec,
    _upstream_procrustes,
    build_subspace_objective,
    build_vector_objective,
    compute_correspondence_metrics,
    validate_experiment_sets,
)


def test_disjoint_set_validation() -> None:
    pairs = [
        PairSpec("cat", "dog", "a photo of a cat"),
        PairSpec("airplane", "sky", "a photo of an airplane"),
    ]
    result = validate_experiment_sets(pairs, ["horse", "ship"])
    assert result["valid"] is True
    with pytest.raises(ValueError, match="disjoint"):
        validate_experiment_sets(
            [PairSpec("cat", "dog", "x"), PairSpec("dog", "sky", "y")],
            ["horse"],
        )
    with pytest.raises(ValueError, match="Control"):
        validate_experiment_sets(pairs, ["dog"])


def test_vector_objective_matches_weighted_cross_covariance() -> None:
    weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([1.0, 0.0])
    anchor = torch.tensor([0.0, 1.0])
    retain = torch.tensor([1.0, 1.0])
    cg = torch.eye(2)
    actual = build_vector_objective(
        weight,
        [target],
        [anchor],
        [retain],
        cg,
        erase_scale=3.0,
        preserve_global_scale=5.0,
        preserve_concept_scale=7.0,
        lamb=11.0,
    )
    expected = 3.0 * torch.outer(weight @ anchor, weight @ target)
    expected += 7.0 * torch.outer(weight @ retain, weight @ retain)
    expected += 5.0 * weight @ cg @ weight.T
    expected += 11.0 * weight @ weight.T
    torch.testing.assert_close(actual, expected)


def test_subspace_objective_matches_upstream_formula() -> None:
    weight = torch.eye(3)
    target = torch.tensor([1.0, 0.0, 0.0])
    anchor = torch.tensor([0.0, 1.0, 0.0])
    cg = torch.zeros(3, 3)
    actual = build_subspace_objective(
        weight,
        [target],
        [anchor],
        [],
        cg,
        erase_scale=2.0,
        preserve_global_scale=0.0,
        preserve_concept_scale=0.0,
        lamb=0.0,
    )
    target_projector = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
    anchor_projector = torch.diag(torch.tensor([0.0, 1.0, 0.0]))
    expected = -2.0 * target_projector @ (torch.eye(3) - anchor_projector)
    torch.testing.assert_close(actual, expected)


def test_subspace_objective_is_anchor_permutation_invariant() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(4, 5, generator=generator)
    targets = [
        torch.randn(5, generator=generator),
        torch.randn(5, generator=generator),
    ]
    anchors = [
        torch.randn(5, generator=generator),
        torch.randn(5, generator=generator),
    ]
    cg = torch.eye(5)
    reference = build_subspace_objective(
        weight,
        targets,
        anchors,
        [],
        cg,
        erase_scale=2.0,
        preserve_global_scale=3.0,
        preserve_concept_scale=0.0,
        lamb=5.0,
    )
    permuted = build_subspace_objective(
        weight,
        targets,
        list(reversed(anchors)),
        [],
        cg,
        erase_scale=2.0,
        preserve_global_scale=3.0,
        preserve_concept_scale=0.0,
        lamb=5.0,
    )
    torch.testing.assert_close(reference, permuted, rtol=1e-5, atol=1e-6)


def test_procrustes_is_orthogonal() -> None:
    matrix = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    rotation, _ = _upstream_procrustes(matrix, "upstream")
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(2))


def test_correspondence_metrics_and_single_anchor_edge_case() -> None:
    similarities = np.array(
        [
            [[0.8, 0.2], [0.4, 0.6]],
            [[0.1, 0.9], [0.3, 0.7]],
        ]
    )
    rows, aggregate = compute_correspondence_metrics(similarities)
    assert len(rows) == 4
    assert aggregate["own_anchor_top1_rate"] == pytest.approx(0.75)
    assert aggregate["positive_margin_fraction"] == pytest.approx(0.75)
    single_rows, single_aggregate = compute_correspondence_metrics(
        np.array([[[0.5], [0.7]]])
    )
    assert single_rows[0]["correspondence_margin"] is None
    assert single_aggregate["mean_margin"] is None
    assert single_aggregate["own_anchor_top1_rate"] == 1.0
