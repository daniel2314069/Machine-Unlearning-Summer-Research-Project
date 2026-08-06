from pathlib import Path

from concept_clustering.config import load_config
from concept_clustering.text_validation import prepare_generation_requests, validate_candidates


ROOT = Path(__file__).resolve().parents[1]


def test_config_inheritance_and_balanced_pilot():
    config = load_config(ROOT / "configs/pilot.json")
    assert [item["name"] for item in config["concepts"]] == ["cat", "dog", "rabbit"]
    assert [item["id"] for item in config["facets"]] == ["visual_appearance", "body_parts", "movement"]
    assert config["clustering"]["k"] == 3
    assert config["model"]["model_id"] == "CompVis/stable-diffusion-v1-4"


def test_curated_pilot_text_is_valid(tmp_path):
    config = load_config(ROOT / "configs/pilot.json")
    candidates, validation = validate_candidates(
        config, ROOT / "data/pilot_candidates.jsonl", tmp_path
    )
    assert len(candidates) == 45
    candidate_rows = [row for row in validation if row["candidate_id"] != "__group_count__"]
    assert len(candidate_rows) == 45
    failures = [row for row in validation if not row["text_valid"]]
    assert failures == []


def test_banned_synonym_fails(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"candidate_id":"bad","concept":"cat","facet_id":"visual_appearance",'
        '"candidate_index":1,"description":"A fluffy feline with pointed ears rests beside a warm window today.",'
        '"source":"test"}\n'
    )
    config = load_config(ROOT / "configs/pilot.json")
    config["candidate_validation"]["candidates_per_concept_facet"] = 0
    _, validation = validate_candidates(config, path, tmp_path / "out")
    bad = next(row for row in validation if row["candidate_id"] == "bad")
    assert not bad["text_valid"]
    assert "banned_term:feline" in bad["failure_reasons"]


def test_non_english_sentence_fails(tmp_path):
    path = tmp_path / "spanish.jsonl"
    path.write_text(
        '{"candidate_id":"spanish","concept":"cat","facet_id":"visual_appearance",'
        '"candidate_index":1,"description":"Una criatura peluda descansa junto a la ventana cálida durante la tarde.",'
        '"source":"test"}\n'
    )
    config = load_config(ROOT / "configs/pilot.json")
    config["candidate_validation"]["candidates_per_concept_facet"] = 0
    _, validation = validate_candidates(config, path, tmp_path / "out")
    row = next(item for item in validation if item["candidate_id"] == "spanish")
    assert not row["text_valid"]
    assert "language_not_english" in row["failure_reasons"]


def test_syntax_independent_config_and_requests(tmp_path):
    config = load_config(ROOT / "configs/syntax_independent_4x100.json")
    assert [item["name"] for item in config["concepts"]] == ["cat", "dog", "fox", "bear"]
    assert config["candidate_validation"]["accepted_per_concept_facet"] == 10
    assert sum(config["candidate_validation"]["diversity"]["accepted_source_quotas"].values()) == 10
    output = tmp_path / "requests.jsonl"
    prepare_generation_requests(config, output)
    rows = [line for line in output.read_text().splitlines() if line]
    assert len(rows) == 4 * 10 * 3
    assert 'syntax_family' in rows[0]


def test_repeated_trigram_is_rejected(tmp_path):
    path = tmp_path / "repeated.jsonl"
    path.write_text(
        '{"candidate_id":"one","concept":"cat","facet_id":"visual_appearance",'
        '"candidate_index":1,"description":"Soft whiskered creature watches sunlight shimmer across curtains beside a quiet window.",'
        '"source":"writer_a","syntax_family":"fronted_subject"}\n'
        '{"candidate_id":"two","concept":"cat","facet_id":"visual_appearance",'
        '"candidate_index":2,"description":"Soft whiskered creature studies drifting dust above cushions during a calm afternoon.",'
        '"source":"writer_b","syntax_family":"scene_first"}\n'
    )
    config = load_config(ROOT / "configs/syntax_independent_4x50.json")
    config["concepts"] = [config["concepts"][0]]
    config["facets"] = [config["facets"][0]]
    config["candidate_validation"]["candidates_per_concept_facet"] = 2
    config["candidate_validation"]["diversity"]["candidate_source_quotas"] = {}
    _, validation = validate_candidates(config, path, tmp_path / "out")
    second = next(row for row in validation if row["candidate_id"] == "two")
    assert not second["text_valid"]
    assert "repeated_3gram:soft whiskered creature:one" in second["failure_reasons"]
