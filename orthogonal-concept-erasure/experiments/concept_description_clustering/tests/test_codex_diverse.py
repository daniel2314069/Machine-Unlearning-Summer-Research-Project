from collections import Counter

import pandas as pd

from concept_clustering.codex_diverse import (
    build_codex_diverse_candidates,
    score_and_select_tfidf_hard,
)
from concept_clustering.config import load_config
from concept_clustering.text_validation import validate_candidates


def test_single_source_builder_and_oof_selection_are_balanced(tmp_path):
    pool_config = load_config("configs/codex_diverse_smoke_pool.json")
    run_config = load_config("configs/codex_diverse_smoke.json")
    pool = build_codex_diverse_candidates(pool_config, tmp_path / "pool.jsonl")
    assert len(pool) == 8
    assert {row["source"] for row in pool} == {"codex_diverse"}
    assert all(row["generation_metadata"]["single_source"] for row in pool)

    validate_candidates(pool_config, tmp_path / "pool.jsonl", tmp_path / "validation")
    selected = score_and_select_tfidf_hard(
        run_config,
        tmp_path / "validation",
        tmp_path / "selected.jsonl",
        tmp_path / "tfidf_hardness.csv",
    )
    assert len(selected) == 4
    assert Counter(row["concept"] for row in selected) == Counter(
        {"cat": 1, "dog": 1, "fox": 1, "bear": 1}
    )
    scores = pd.read_csv(tmp_path / "tfidf_hardness.csv")
    assert len(scores) == 8
    assert {
        "tfidf_word_unigram__predicted_class",
        "tfidf_word_unigram__target_probability",
        "tfidf_word_unigram__target_rank",
        "tfidf_word_unigram__target_margin",
        "selected_for_generation",
    }.issubset(scores.columns)
