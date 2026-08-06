from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from concept_clustering.balanced_paired_w0_geometry import (
    _fit_feature_sets_without_labels,
    _inspect_oce_source,
    _project_and_normalize,
)


def test_oce_source_selects_only_to_v_and_last_content_token():
    repo = Path(__file__).resolve().parents[3]
    audit = _inspect_oce_source(repo)
    assert audit["verified_matrix_type"] == "to_v"
    assert "sum()-2" in audit["verified_name_readout_rule"].replace(" ", "")
    assert audit["edited_checkpoint_loaded"] is False


def test_projection_normalizes_only_after_w0():
    raw = np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    weight = torch.tensor([[2.0, 0.0], [0.0, 0.5]], dtype=torch.float32)
    actual = _project_and_normalize(raw, weight, "cpu")
    projected = raw @ weight.numpy().T
    expected = projected / np.linalg.norm(projected, axis=1, keepdims=True)
    assert np.allclose(actual, expected)
    pre_normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    assert not np.allclose(projected, pre_normalized @ weight.numpy().T)


def test_unsupervised_fit_boundary_has_no_label_argument():
    parameters = set(inspect.signature(_fit_feature_sets_without_labels).parameters)
    assert parameters == {"description_raw", "modules", "settings", "device"}
    assert not any("label" in name or "concept" in name for name in parameters)
