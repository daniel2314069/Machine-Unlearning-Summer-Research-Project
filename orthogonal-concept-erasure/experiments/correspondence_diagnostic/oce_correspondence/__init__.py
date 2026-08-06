"""OCE target-anchor correspondence diagnostic."""

from .core import (
    PairSpec,
    compute_correspondence_metrics,
    edit_projection_weights,
    validate_experiment_sets,
)

__all__ = [
    "PairSpec",
    "compute_correspondence_metrics",
    "edit_projection_weights",
    "validate_experiment_sets",
]
