#!/usr/bin/env python
"""Strict semantic normalization for ScaPre evaluator manifests."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


VARIANT_ONLY_FIELDS = {"variant", "checkpoint_sha256"}
UNORDERED_LIST_PATHS = {("scheduler_config", "_use_default_values")}


def canonical_evaluator_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove intended variant fields and sort one semantically unordered list.

    Diffusers records ``_use_default_values`` as a list even though it denotes a
    set of configuration keys. No other list or value is normalized.
    """
    controlled = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key not in VARIANT_ONLY_FIELDS
    }
    scheduler = controlled.get("scheduler_config")
    if not isinstance(scheduler, dict):
        raise RuntimeError("evaluator manifest is missing scheduler_config")
    defaults = scheduler.get("_use_default_values")
    if not isinstance(defaults, list) or not all(isinstance(value, str) for value in defaults):
        raise RuntimeError("scheduler _use_default_values must be a list of strings")
    if len(defaults) != len(set(defaults)):
        raise RuntimeError("scheduler _use_default_values contains duplicates")
    scheduler["_use_default_values"] = sorted(defaults)
    return controlled


def fingerprint_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def differing_paths(left: Any, right: Any, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if type(left) is not type(right):
        return {path}
    if isinstance(left, dict):
        paths: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            child = path + (str(key),)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(differing_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        return set() if left == right else {path}
    return set() if left == right else {path}


def compare_evaluator_manifests(
    observations: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not observations:
        raise RuntimeError("no evaluator manifests were supplied")
    baseline_label, baseline_raw = observations[0]
    baseline = canonical_evaluator_fingerprint(baseline_raw)
    canonical_hash = fingerprint_sha256(baseline)
    raw_variations: list[dict[str, Any]] = []
    observed_raw_hashes: dict[str, str] = {}
    for label, raw in observations:
        canonical = canonical_evaluator_fingerprint(raw)
        if canonical != baseline:
            paths = sorted(".".join(path) for path in differing_paths(baseline, canonical))
            raise RuntimeError(
                f"substantive evaluator fingerprint difference for {label}: {paths}"
            )
        raw_controlled = {
            key: value for key, value in raw.items() if key not in VARIANT_ONLY_FIELDS
        }
        baseline_controlled = {
            key: value
            for key, value in baseline_raw.items()
            if key not in VARIANT_ONLY_FIELDS
        }
        paths = differing_paths(baseline_controlled, raw_controlled)
        disallowed = paths - UNORDERED_LIST_PATHS
        if disallowed:
            rendered = sorted(".".join(path) for path in disallowed)
            raise RuntimeError(f"raw evaluator difference is not allowlisted for {label}: {rendered}")
        if paths:
            raw_variations.append({
                "observation": label,
                "paths": sorted(".".join(path) for path in paths),
                "baseline": baseline_label,
            })
        observed_raw_hashes[label] = fingerprint_sha256(raw_controlled)
    return {
        "status": "passed",
        "canonical_sha256": canonical_hash,
        "observations": len(observations),
        "unique_raw_sha256_count": len(set(observed_raw_hashes.values())),
        "raw_sha256": observed_raw_hashes,
        "raw_variations": raw_variations,
        "allowlisted_unordered_paths": [
            ".".join(path) for path in sorted(UNORDERED_LIST_PATHS)
        ],
        "substantive_fields_identical": True,
    }
