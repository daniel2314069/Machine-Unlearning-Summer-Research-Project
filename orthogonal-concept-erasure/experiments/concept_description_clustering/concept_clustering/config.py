from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if "extends" in data:
        base = load_config(path.parent / data["extends"])
        data = _deep_merge(base, data)

    if "concept_names" in data:
        wanted = set(data.pop("concept_names"))
        data["concepts"] = [item for item in data["concepts"] if item["name"] in wanted]
    if "facet_ids" in data:
        wanted = set(data.pop("facet_ids"))
        data["facets"] = [item for item in data["facets"] if item["id"] in wanted]

    expected_k = len(data["concepts"])
    if data["clustering"]["k"] != expected_k:
        raise ValueError(f"clustering.k={data['clustering']['k']} but config contains {expected_k} concepts")

    validation = data["candidate_validation"]
    diversity = validation.get("diversity", {})
    candidate_quotas = diversity.get("candidate_source_quotas", {})
    accepted_quotas = diversity.get("accepted_source_quotas", {})
    if candidate_quotas and sum(int(value) for value in candidate_quotas.values()) != int(
        validation["candidates_per_concept_facet"]
    ):
        raise ValueError("candidate_source_quotas must sum to candidates_per_concept_facet")
    if accepted_quotas and sum(int(value) for value in accepted_quotas.values()) != int(
        validation["accepted_per_concept_facet"]
    ):
        raise ValueError("accepted_source_quotas must sum to accepted_per_concept_facet")
    generation_sources = data.get("candidate_generation", {}).get("sources", [])
    if generation_sources and candidate_quotas:
        generation_counts = {str(row["id"]): int(row["candidates_per_group"]) for row in generation_sources}
        expected_counts = {str(key): int(value) for key, value in candidate_quotas.items()}
        if generation_counts != expected_counts:
            raise ValueError("candidate_generation sources must exactly match candidate_source_quotas")
    return data


def config_concept_names(config: dict[str, Any]) -> list[str]:
    return [item["name"] for item in config["concepts"]]


def config_facet_ids(config: dict[str, Any]) -> list[str]:
    return [item["id"] for item in config["facets"]]


def all_banned_terms(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for concept in config["concepts"]:
        for term in concept["banned_terms"]:
            result[term.casefold()] = concept["name"]
    return result
