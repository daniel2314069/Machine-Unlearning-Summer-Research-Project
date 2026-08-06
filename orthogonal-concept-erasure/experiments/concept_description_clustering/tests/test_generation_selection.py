from collections import Counter

from concept_clustering.generation_validation import finalize_accepted
from concept_clustering.utils import read_jsonl, write_csv, write_jsonl


def test_final_selection_enforces_source_quota(tmp_path):
    candidates = []
    decisions = []
    reviews = []
    for source in ["human", "model_a", "model_b"]:
        for index in range(2):
            candidate_id = f"{source}_{index}"
            candidates.append({
                "candidate_id": candidate_id,
                "concept": "cat",
                "facet_id": "appearance",
                "source": source,
                "description": f"description {candidate_id}",
            })
            decisions.append({
                "candidate_id": candidate_id,
                "concept": "cat",
                "facet_id": "appearance",
                "automatic_decision": "accepted",
                "automatic_reason": "test",
                "mean_target_score": 0.8,
                "mean_target_margin": 0.2 - index * 0.01,
            })
            reviews.append({"candidate_id": candidate_id, "manual_decision": "unset"})
    write_jsonl(tmp_path / "candidate_descriptions.jsonl", candidates)
    write_csv(tmp_path / "candidate_generation_decisions.csv", decisions)
    write_csv(tmp_path / "manual_review.csv", reviews)
    config = {
        "concepts": [{"name": "cat"}],
        "facets": [{"id": "appearance"}],
        "candidate_validation": {
            "accepted_per_concept_facet": 3,
            "diversity": {
                "accepted_source_quotas": {"human": 1, "model_a": 1, "model_b": 1}
            },
        },
    }
    finalize_accepted(config, tmp_path)
    selected = read_jsonl(tmp_path / "accepted_descriptions.jsonl")
    assert Counter(row["source"] for row in selected) == Counter(
        {"human": 1, "model_a": 1, "model_b": 1}
    )
