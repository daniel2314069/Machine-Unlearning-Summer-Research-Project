import numpy as np

from concept_clustering.clustering import _fit_runs, _prototype_distribution_metrics


def test_separable_clusters_recover_concepts_not_facets():
    rng = np.random.default_rng(7)
    features, concepts, facets = [], [], []
    for concept in range(3):
        center = np.zeros(12)
        center[concept] = 1.0
        for item in range(15):
            vector = center + rng.normal(0, 0.03, 12)
            vector /= np.linalg.norm(vector)
            features.append(vector)
            concepts.append(concept)
            facets.append(item % 3)
    config = {
        "clustering": {
            "k": 3,
            "n_runs": 2,
            "n_init": 5,
            "random_seeds": [0, 1],
        }
    }
    metrics, _ = _fit_runs(
        np.asarray(features), np.asarray(concepts), np.asarray(facets), config, "synthetic"
    )
    assert min(row["ari_concept"] for row in metrics) > 0.99
    assert max(row["ari_facet"] for row in metrics) < 0.1
    assert min(row["hungarian_accuracy"] for row in metrics) == 1.0


def test_prototype_percentile_distinguishes_center_from_outlier():
    rng = np.random.default_rng(11)
    first = np.column_stack([np.ones(50), rng.normal(0, 0.03, (50, 3))])
    second = np.column_stack([rng.normal(0, 0.03, (50, 3)), np.ones(50)])
    features = np.vstack([first, second])
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    true_ids = np.repeat([0, 1], 50)
    prototypes = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    result = _prototype_distribution_metrics(
        features,
        true_ids,
        prototypes,
        {"n_bootstrap": 100, "reference_fraction": 0.8, "seed": 3},
    )
    assert result[0]["bootstrap_percentile_median"] < 60
    assert result[1]["bootstrap_percentile_median"] > 95
