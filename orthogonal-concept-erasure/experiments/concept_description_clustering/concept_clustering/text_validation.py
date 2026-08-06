from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langdetect import DetectorFactory, LangDetectException, detect_langs

from .config import all_banned_terms, config_concept_names, config_facet_ids
from .utils import read_jsonl, write_csv, write_jsonl


WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
SENTENCE_END_RE = re.compile(r"[.!?]")
NON_ASCII_LETTER_RE = re.compile(r"[^\x00-\x7F]")
DetectorFactory.seed = 0

# Used only for lexical opening signatures.  This is deliberately small and
# deterministic: it is a template-leakage check, not a linguistic parser.
OPENING_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "its", "their",
    "in", "on", "at", "by", "near", "beside", "with", "from", "under",
}


def words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def normalized_tokens(text: str) -> set[str]:
    return set(words(text))


def token_jaccard(left: str, right: str) -> float:
    a, b = normalized_tokens(left), normalized_tokens(right)
    return len(a & b) / max(1, len(a | b))


def token_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    token_list = words(text)
    return [tuple(token_list[index:index + n]) for index in range(max(0, len(token_list) - n + 1))]


def opening_signature(text: str, size: int) -> str:
    content = [token for token in words(text) if token not in OPENING_STOPWORDS]
    return " ".join(content[:size])


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<![A-Za-z])" + re.escape(phrase.casefold()).replace(r"\ ", r"\s+") + r"(?![A-Za-z])"
    return re.search(pattern, text.casefold()) is not None


def _detect_language(text: str) -> tuple[str, float]:
    try:
        probabilities = detect_langs(text)
    except LangDetectException:
        return "unknown", 0.0
    english = next((item.prob for item in probabilities if item.lang == "en"), 0.0)
    return probabilities[0].lang, float(english)


def validate_candidates(
    config: dict[str, Any], candidates_path: str | Path, output_dir: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir = Path(output_dir)
    candidates = read_jsonl(candidates_path)
    concepts = set(config_concept_names(config))
    facets = set(config_facet_ids(config))
    banned = all_banned_terms(config)
    rules = config["candidate_validation"]
    diversity = rules.get("diversity", {})
    diversity_enabled = bool(diversity.get("enabled", False))
    ngram_size = int(diversity.get("ngram_size", 3))
    max_ngram_occurrences = int(diversity.get("max_ngram_occurrences", 1))
    opening_words = int(diversity.get("opening_content_words", 3))
    max_opening_occurrences = int(diversity.get("max_opening_occurrences", 1))
    max_syntax_per_group = int(diversity.get("max_syntax_family_per_concept_facet", 1))
    require_syntax_family = bool(diversity.get("require_syntax_family", False))
    source_quotas = {str(key): int(value) for key, value in diversity.get("candidate_source_quotas", {}).items()}

    seen_ids: set[str] = set()
    earlier_valid: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    normalized_candidates: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    seen_ngrams: dict[tuple[str, ...], str] = {}
    ngram_counts: Counter[tuple[str, ...]] = Counter()
    opening_counts: Counter[str] = Counter()
    opening_first: dict[str, str] = {}
    syntax_counts: Counter[tuple[str, str, str]] = Counter()

    required = {"candidate_id", "concept", "facet_id", "candidate_index", "description", "source"}
    for row_number, raw in enumerate(candidates, start=1):
        candidate = dict(raw)
        reasons: list[str] = []
        missing = sorted(required - set(candidate))
        if missing:
            reasons.append("missing_fields:" + "|".join(missing))
        candidate_id = str(candidate.get("candidate_id", f"row_{row_number}"))
        description = str(candidate.get("description", "")).strip()
        concept = str(candidate.get("concept", ""))
        facet = str(candidate.get("facet_id", ""))
        source = str(candidate.get("source", ""))
        metadata = candidate.get("generation_metadata") or {}
        syntax_family = str(candidate.get("syntax_family") or metadata.get("syntax_family") or "").strip()

        if candidate_id in seen_ids:
            reasons.append("duplicate_candidate_id")
        seen_ids.add(candidate_id)
        if concept not in concepts:
            reasons.append("unknown_concept")
        if facet not in facets:
            reasons.append("unknown_facet")
        if NON_ASCII_LETTER_RE.search(description):
            reasons.append("non_english_character")
        detected_language, english_probability = _detect_language(description)
        if rules.get("language_backend", "langdetect") != "langdetect":
            raise ValueError("candidate_validation.language_backend must be 'langdetect'")
        if detected_language != "en" or english_probability < float(rules["min_english_probability"]):
            reasons.append(f"language_not_english:{detected_language}:{english_probability:.3f}")
        if not description or description[-1:] not in ".!?":
            reasons.append("missing_sentence_terminal")
        if len(SENTENCE_END_RE.findall(description)) != 1:
            reasons.append("not_exactly_one_sentence")

        token_list = words(description)
        word_count = len(token_list)
        if word_count < int(rules["min_words"]):
            reasons.append("too_short")
        if word_count > int(rules["max_words"]):
            reasons.append("too_long")

        found_banned = sorted(term for term in banned if _contains_phrase(description, term))
        if found_banned:
            reasons.append("banned_term:" + "|".join(found_banned))
        found_negation = sorted(term for term in rules["negation_terms"] if _contains_phrase(description, term))
        if found_negation:
            reasons.append("negation:" + "|".join(found_negation))
        found_comparison = sorted(
            phrase for phrase in rules["comparison_phrases"] if _contains_phrase(description, phrase)
        )
        if found_comparison:
            reasons.append("explicit_comparison:" + "|".join(found_comparison))

        informative = [
            token for token in token_list
            if token not in set(rules["vague_terms"]) and len(token) > 2
        ]
        if len(informative) < int(rules["min_informative_words"]):
            reasons.append("insufficient_visual_information")

        duplicate_of = ""
        for earlier in earlier_valid:
            similarity = token_jaccard(description, earlier["description"])
            if similarity >= float(rules["near_duplicate_jaccard"]):
                reasons.append(f"near_duplicate:{earlier['candidate_id']}:{similarity:.3f}")
                duplicate_of = earlier["candidate_id"]
                break

        local_ngrams = token_ngrams(description, ngram_size) if diversity_enabled else []
        repeated_ngrams = []
        opening = opening_signature(description, opening_words) if diversity_enabled else ""
        opening_collision = ""
        if diversity_enabled:
            if source_quotas and source not in source_quotas:
                reasons.append(f"unconfigured_source:{source}")
            if require_syntax_family and not syntax_family:
                reasons.append("missing_syntax_family")
            if syntax_family and syntax_counts[(concept, facet, syntax_family)] >= max_syntax_per_group:
                reasons.append(f"repeated_syntax_family:{syntax_family}")
            for ngram, local_count in Counter(local_ngrams).items():
                if local_count > max_ngram_occurrences:
                    repeated_ngrams.append((ngram, candidate_id))
                if ngram_counts[ngram] >= max_ngram_occurrences:
                    repeated_ngrams.append((ngram, seen_ngrams[ngram]))
            if repeated_ngrams:
                ngram, first_id = repeated_ngrams[0]
                reasons.append(f"repeated_{ngram_size}gram:{' '.join(ngram)}:{first_id}")
            if opening and opening_counts[opening] >= max_opening_occurrences:
                opening_collision = opening_first[opening]
                reasons.append(f"repeated_opening:{opening}:{opening_collision}")

        is_valid = not reasons
        normalized = {
            **candidate,
            "candidate_id": candidate_id,
            "concept": concept,
            "facet_id": facet,
            "description": description,
        }
        normalized_candidates.append(normalized)
        if is_valid:
            earlier_valid.append(normalized)
            if diversity_enabled:
                for ngram in local_ngrams:
                    seen_ngrams.setdefault(ngram, candidate_id)
                    ngram_counts[ngram] += 1
                if opening:
                    opening_counts[opening] += 1
                    opening_first.setdefault(opening, candidate_id)
                if syntax_family:
                    syntax_counts[(concept, facet, syntax_family)] += 1
        validation_rows.append({
            "candidate_id": candidate_id,
            "concept": concept,
            "facet_id": facet,
            "word_count": word_count,
            "detected_language": detected_language,
            "english_probability": english_probability,
            "text_valid": is_valid,
            "failure_reasons": ";".join(reasons),
            "near_duplicate_of": duplicate_of,
            "source": source,
            "syntax_family": syntax_family,
            "description": description,
        })
        diversity_rows.append({
            "candidate_id": candidate_id,
            "concept": concept,
            "facet_id": facet,
            "source": source,
            "syntax_family": syntax_family,
            "opening_signature": opening,
            "opening_collision_with": opening_collision,
            "ngram_size": ngram_size if diversity_enabled else "",
            "ngram_count": len(local_ngrams),
            "first_repeated_ngram": " ".join(repeated_ngrams[0][0]) if repeated_ngrams else "",
            "ngram_collision_with": repeated_ngrams[0][1] if repeated_ngrams else "",
            "text_valid": is_valid,
        })

    expected = int(rules["candidates_per_concept_facet"])
    counts = Counter((row["concept"], row["facet_id"]) for row in normalized_candidates)
    for concept in concepts:
        for facet in facets:
            actual = counts[(concept, facet)]
            if actual != expected:
                validation_rows.append({
                    "candidate_id": "__group_count__",
                    "concept": concept,
                    "facet_id": facet,
                    "word_count": "",
                    "detected_language": "",
                    "english_probability": "",
                    "text_valid": False,
                    "failure_reasons": f"candidate_count_expected_{expected}_got_{actual}",
                    "near_duplicate_of": "",
                    "source": "",
                    "syntax_family": "",
                    "description": "",
                })

    if diversity_enabled and source_quotas:
        source_counts = Counter(
            (row["concept"], row["facet_id"], str(row.get("source", "")))
            for row in normalized_candidates
        )
        for concept in concepts:
            for facet in facets:
                for source, source_expected in source_quotas.items():
                    actual = source_counts[(concept, facet, source)]
                    if actual != source_expected:
                        validation_rows.append({
                            "candidate_id": "__source_count__",
                            "concept": concept,
                            "facet_id": facet,
                            "word_count": "",
                            "detected_language": "",
                            "english_probability": "",
                            "text_valid": False,
                            "failure_reasons": f"source_{source}_count_expected_{source_expected}_got_{actual}",
                            "near_duplicate_of": "",
                            "source": source,
                            "syntax_family": "",
                            "description": "",
                        })

    write_jsonl(output_dir / "candidate_descriptions.jsonl", normalized_candidates)
    write_csv(output_dir / "candidate_text_validation.csv", validation_rows)
    failures = [row for row in validation_rows if not row["text_valid"]]
    write_jsonl(output_dir / "candidate_text_validation_failures.jsonl", failures)
    write_csv(output_dir / "candidate_diversity_audit.csv", diversity_rows)
    summary_rows = []
    if diversity_enabled:
        scopes = [("overall", "all", earlier_valid)]
        scopes.extend(("concept", concept, [row for row in earlier_valid if row["concept"] == concept]) for concept in sorted(concepts))
        scopes.extend(("source", source, [row for row in earlier_valid if row.get("source") == source]) for source in sorted(source_quotas))
        for scope_type, scope, scope_rows in scopes:
            for n in (1, 2, 3):
                grams = [gram for row in scope_rows for gram in token_ngrams(row["description"], n)]
                summary_rows.append({
                    "scope_type": scope_type,
                    "scope": scope,
                    "n": n,
                    "valid_descriptions": len(scope_rows),
                    "ngram_tokens": len(grams),
                    "distinct_ngrams": len(set(grams)),
                    "distinct_ratio": len(set(grams)) / max(1, len(grams)),
                })
    write_csv(
        output_dir / "candidate_diversity_summary.csv",
        summary_rows,
        ["scope_type", "scope", "n", "valid_descriptions", "ngram_tokens", "distinct_ngrams", "distinct_ratio"],
    )
    return normalized_candidates, validation_rows


def prepare_generation_requests(config: dict[str, Any], output_path: str | Path) -> None:
    banned = sorted(all_banned_terms(config))
    count = int(config["candidate_validation"]["candidates_per_concept_facet"])
    sources = config.get("candidate_generation", {}).get("sources") or [
        {"id": "optional_llm", "candidates_per_group": count}
    ]
    if sum(int(source["candidates_per_group"]) for source in sources) != count:
        raise ValueError("candidate_generation source counts must sum to candidates_per_concept_facet")
    requests = []
    for concept in config["concepts"]:
        for facet in config["facets"]:
            for source in sources:
                source_id = str(source["id"])
                source_count = int(source["candidates_per_group"])
                prompt = f"""Independently write {source_count} English descriptions of the visual concept {concept['name']!r}.
Facet: {facet['name']}.
Source/author slot: {source_id!r}. Do not imitate text from any other source slot.
Return JSON only as an array of objects with keys `description` and `syntax_family`.
Each item must be exactly one sentence of 8-20 words and visually useful for text-to-image generation.
Do not use any of these banned words or phrases: {', '.join(banned)}.
Do not use negation, comparisons, labels, taxonomic names, or vague-only descriptions.
Within this response, use a different subject construction and syntax_family for every sentence.
Avoid repeated three-word sequences, repeated openings, and reusable concept-specific subject frames.
Keep syntax and length varied while staying within the requested facet."""
                requests.append({
                    "request_id": f"{source_id}__{concept['name']}__{facet['id']}",
                    "source": source_id,
                    "concept": concept["name"],
                    "facet_id": facet["id"],
                    "n_requested": source_count,
                    "prompt": prompt,
                })
    write_jsonl(output_path, requests)


def import_generated_responses(
    responses_path: str | Path, output_path: str | Path, source: str = "optional_llm"
) -> None:
    responses = read_jsonl(responses_path)
    candidates = []
    for response in responses:
        descriptions = response.get("descriptions")
        if isinstance(descriptions, str):
            descriptions = json.loads(descriptions)
        if not isinstance(descriptions, list):
            raise ValueError(f"{response.get('request_id')} has no descriptions array")
        source_id = str(response.get("source") or source)
        for index, item in enumerate(descriptions, start=1):
            if isinstance(item, dict):
                description = item.get("description", "")
                syntax_family = str(item.get("syntax_family", "")).strip()
            else:
                description = item
                syntax_family = ""
            candidate_id = response.get("candidate_id_prefix") or response.get("request_id")
            if not candidate_id:
                candidate_id = f"{source_id}__{response['concept']}__{response['facet_id']}"
            candidates.append({
                "candidate_id": f"{candidate_id}_{index:02d}",
                "concept": response["concept"],
                "facet_id": response["facet_id"],
                "candidate_index": index,
                "description": str(description).strip(),
                "source": source_id,
                "generation_metadata": {
                    "request_id": response.get("request_id"),
                    "syntax_family": syntax_family,
                    "source_metadata": response.get("source_metadata", {}),
                },
            })
    write_jsonl(output_path, candidates, overwrite=False)
