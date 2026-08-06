import torch

from concept_clustering.lexical_baselines import run_lexical_baselines
from concept_clustering.utils import write_jsonl


def test_lexical_baseline_writes_unsupervised_and_supervised_outputs(tmp_path):
    rows = []
    for concept, noun in [("red", "scarlet"), ("blue", "azure")]:
        for index in range(6):
            rows.append({
                "candidate_id": f"{concept}_{index}",
                "concept": concept,
                "facet_id": "appearance" if index % 2 == 0 else "scene",
                "description": f"{noun} texture fills a distinct visual scene number {index}",
            })
    write_jsonl(tmp_path / "accepted_descriptions.jsonl", rows)
    generator = torch.Generator().manual_seed(4)
    fixed = torch.randn(12, 8, generator=generator)
    fixed = fixed / fixed.norm(dim=1, keepdim=True)
    torch.save({
        "fixed_readout": {"describes_concept": fixed},
        "shuffled_fixed_readout": fixed.clone(),
    }, tmp_path / "raw_text_embeddings.pt")
    config = {
        "concepts": [{"name": "red"}, {"name": "blue"}],
        "facets": [{"id": "appearance"}, {"id": "scene"}],
        "readout": {"primary_suffix_name": "describes_concept"},
        "clustering": {
            "k": 2,
            "n_runs": 1,
            "n_init": 2,
            "random_seeds": [0],
            "canonical_seed": 0,
        },
        "lexical_baselines": {
            "enabled": True,
            "n_runs": 1,
            "n_init": 2,
            "classification_cv_folds": 2,
            "vectorizers": [{
                "name": "tfidf_word_unigram",
                "analyzer": "word",
                "ngram_range": [1, 1],
                "min_df": 1,
            }],
        },
    }
    run_lexical_baselines(config, tmp_path)
    assert (tmp_path / "lexical_baseline_clustering_metrics.csv").exists()
    assert (tmp_path / "baseline_classification_metrics.csv").exists()
    assert (tmp_path / "tfidf_top_features.csv").exists()
