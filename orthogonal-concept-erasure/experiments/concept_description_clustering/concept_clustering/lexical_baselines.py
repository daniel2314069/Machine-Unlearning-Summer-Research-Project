from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import read_jsonl, set_reproducible_seed, write_csv


DEFAULT_VECTORIZERS = [
    {"name": "tfidf_word_unigram", "analyzer": "word", "ngram_range": [1, 1], "min_df": 1},
    {"name": "tfidf_word_unigram_bigram", "analyzer": "word", "ngram_range": [1, 2], "min_df": 1},
    {"name": "tfidf_char_3_5", "analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 1},
]


def _mapping(true_ids: np.ndarray, cluster_ids: np.ndarray, k: int):
    matrix = confusion_matrix(true_ids, cluster_ids, labels=np.arange(k))
    true_rows, cluster_columns = linear_sum_assignment(-matrix)
    cluster_to_true = {int(cluster): int(true) for true, cluster in zip(true_rows, cluster_columns)}
    predicted = np.array([cluster_to_true[int(cluster)] for cluster in cluster_ids])
    return cluster_to_true, predicted


def _vectorizer(spec: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        analyzer=spec.get("analyzer", "word"),
        ngram_range=tuple(spec.get("ngram_range", [1, 1])),
        min_df=spec.get("min_df", 1),
        max_df=spec.get("max_df", 1.0),
        sublinear_tf=True,
        norm="l2",
    )


def _classification_row(
    name: str,
    features,
    true_ids: np.ndarray,
    cv: StratifiedKFold,
    chance: float,
) -> dict[str, Any]:
    estimator = LogisticRegression(max_iter=5000, random_state=0, solver="liblinear")
    scores = cross_val_score(estimator, features, true_ids, cv=cv, scoring="accuracy")
    return {
        "representation": name,
        "task": "supervised_linear_classification",
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std(ddof=1)),
        "chance_accuracy": chance,
        "cv_folds": cv.n_splits,
        "cv_seed": cv.random_state,
    }


def run_lexical_baselines(config: dict[str, Any], output_dir: str | Path) -> None:
    settings = config.get("lexical_baselines", {})
    if not bool(settings.get("enabled", False)):
        return
    set_reproducible_seed(0)
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    accepted = read_jsonl(output_dir / "accepted_descriptions.jsonl")
    raw = torch.load(output_dir / "raw_text_embeddings.pt", map_location="cpu", weights_only=False)
    descriptions = [row["description"] for row in accepted]
    concept_names = [item["name"] for item in config["concepts"]]
    facet_names = [item["id"] for item in config["facets"]]
    true_ids = np.array([concept_names.index(row["concept"]) for row in accepted])
    facet_ids = np.array([facet_names.index(row["facet_id"]) for row in accepted])
    candidate_ids = [row["candidate_id"] for row in accepted]
    k = len(concept_names)
    clustering = config["clustering"]
    seeds = clustering["random_seeds"][: int(settings.get("n_runs", clustering["n_runs"]))]
    n_init = int(settings.get("n_init", clustering["n_init"]))
    canonical_seed = int(clustering["canonical_seed"])
    cv_folds = int(settings.get("classification_cv_folds", 5))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    chance = 1.0 / k

    metric_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    confusion_dir = output_dir / "baseline_confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    for spec in settings.get("vectorizers", DEFAULT_VECTORIZERS):
        vectorizer = _vectorizer(spec)
        features = vectorizer.fit_transform(descriptions)
        representation = str(spec["name"])
        canonical = None
        for seed in seeds:
            model = KMeans(
                n_clusters=k,
                n_init=n_init,
                random_state=int(seed),
                algorithm="lloyd",
            )
            clusters = model.fit_predict(features)
            mapping, predicted = _mapping(true_ids, clusters, k)
            metric_rows.append({
                "representation": representation,
                "seed": int(seed),
                "ari_concept": adjusted_rand_score(true_ids, clusters),
                "nmi_concept": normalized_mutual_info_score(true_ids, clusters),
                "hungarian_accuracy": accuracy_score(true_ids, predicted),
                "silhouette": silhouette_score(features, clusters, metric="euclidean"),
                "ari_facet": adjusted_rand_score(facet_ids, clusters),
                "nmi_facet": normalized_mutual_info_score(facet_ids, clusters),
                "feature_count": features.shape[1],
                "inertia": float(model.inertia_),
            })
            if int(seed) == canonical_seed:
                canonical = (clusters, mapping, predicted)
        if canonical is None:
            raise ValueError("canonical_seed must be included in lexical baseline seeds")
        clusters, mapping, predicted = canonical
        for index, candidate_id in enumerate(candidate_ids):
            assignment_rows.append({
                "representation": representation,
                "candidate_id": candidate_id,
                "concept": accepted[index]["concept"],
                "facet_id": accepted[index]["facet_id"],
                "cluster": int(clusters[index]),
                "matched_cluster_concept": concept_names[mapping[int(clusters[index])]],
                "correct_after_hungarian": bool(predicted[index] == true_ids[index]),
            })
        matrix = confusion_matrix(true_ids, predicted, labels=np.arange(k))
        pd.DataFrame(matrix, index=concept_names, columns=concept_names).to_csv(
            confusion_dir / f"{representation}.csv"
        )

        pipeline = make_pipeline(
            _vectorizer(spec),
            LogisticRegression(max_iter=5000, random_state=0, solver="liblinear"),
        )
        scores = cross_val_score(pipeline, descriptions, true_ids, cv=cv, scoring="accuracy")
        classification_rows.append({
            "representation": representation,
            "task": "supervised_text_classification",
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std(ddof=1)),
            "chance_accuracy": chance,
            "cv_folds": cv_folds,
            "cv_seed": 0,
        })

        names = vectorizer.get_feature_names_out()
        for concept_id, concept in enumerate(concept_names):
            mean_weights = np.asarray(features[true_ids == concept_id].mean(axis=0)).ravel()
            top = np.argsort(mean_weights)[-20:][::-1]
            feature_rows.extend({
                "representation": representation,
                "concept": concept,
                "rank": rank,
                "feature": str(names[feature_id]),
                "mean_tfidf": float(mean_weights[feature_id]),
            } for rank, feature_id in enumerate(top, start=1))

    primary = config["readout"]["primary_suffix_name"]
    fixed = raw["fixed_readout"][primary].numpy()
    classification_rows.append(
        _classification_row(f"fixed:{primary}", fixed, true_ids, cv, chance)
    )
    if raw.get("shuffled_fixed_readout") is not None:
        classification_rows.append(
            _classification_row(
                f"shuffled_words:fixed:{primary}",
                raw["shuffled_fixed_readout"].numpy(),
                true_ids,
                cv,
                chance,
            )
        )

    write_csv(output_dir / "lexical_baseline_clustering_metrics.csv", metric_rows)
    write_csv(output_dir / "lexical_baseline_assignments.csv", assignment_rows)
    write_csv(output_dir / "baseline_classification_metrics.csv", classification_rows)
    write_csv(output_dir / "tfidf_top_features.csv", feature_rows)

    frame = pd.DataFrame(metric_rows)
    summary = frame.groupby("representation", as_index=False).agg(
        ari_concept=("ari_concept", "mean"),
        nmi_concept=("nmi_concept", "mean"),
        accuracy=("hungarian_accuracy", "mean"),
        ari_facet=("ari_facet", "mean"),
    )
    summary.to_csv(output_dir / "lexical_baseline_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    width = 0.25
    ax.bar(x - width, summary["ari_concept"], width, label="Concept ARI")
    ax.bar(x, summary["accuracy"], width, label="Matched accuracy")
    ax.bar(x + width, summary["ari_facet"], width, label="Facet ARI")
    ax.set_xticks(x, summary["representation"], rotation=20, ha="right")
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Surface-lexical TF-IDF baselines")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "lexical_baseline_comparison.png", dpi=180)
    plt.close(fig)
