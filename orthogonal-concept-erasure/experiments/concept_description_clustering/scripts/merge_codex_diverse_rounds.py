#!/usr/bin/env python
"""Merge independently cached codex_diverse rounds into one balanced analysis corpus.

Only candidates that passed the unchanged automatic three-seed rule are eligible.
Round-local IDs are namespaced; cached images and score files are read, never moved
or rewritten.  Selection prefers TF-IDF-hard candidates, then larger generation
margins, while enforcing exact text/opening/trigram/near-duplicate diversity over
the final 200 rows.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from concept_clustering.config import load_config
from concept_clustering.text_validation import opening_signature, token_jaccard, token_ngrams
from concept_clustering.utils import atomic_write_text, read_csv, read_jsonl, write_csv, write_jsonl


def _namespace(row: dict[str, Any], round_name: str) -> dict[str, Any]:
    result = dict(row)
    original_id = str(result["candidate_id"])
    result["candidate_id"] = f"{round_name}__{original_id}"
    result["round"] = round_name
    result["round_candidate_id"] = original_id
    return result


def _difficulty(row: dict[str, Any]) -> float:
    hardness = row.get("tfidf_hardness", {})
    try:
        return float(hardness.get("tfidf_difficulty_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _margin(row: dict[str, Any]) -> float:
    try:
        return float(row.get("mean_target_margin", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _conflicts(row: dict[str, Any], selected: list[dict[str, Any]]) -> str:
    description = str(row["description"])
    grams = set(token_ngrams(description, 3))
    opening = opening_signature(description, 3)
    for earlier in selected:
        earlier_text = str(earlier["description"])
        if description.casefold() == earlier_text.casefold():
            return f"exact_duplicate:{earlier['candidate_id']}"
        shared = grams.intersection(token_ngrams(earlier_text, 3))
        if shared:
            return f"repeated_3gram:{' '.join(sorted(shared)[0])}:{earlier['candidate_id']}"
        if opening and opening == opening_signature(earlier_text, 3):
            return f"repeated_opening:{opening}:{earlier['candidate_id']}"
        similarity = token_jaccard(description, earlier_text)
        if similarity >= 0.72:
            return f"near_duplicate:{earlier['candidate_id']}:{similarity:.3f}"
    return ""


def _global_selection(
    by_group: dict[tuple[str, str], list[dict[str, Any]]],
    groups: list[tuple[str, str]], per_group: int,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Solve the final diversity-constrained selection globally with SciPy/HiGHS.

    Greedy group ordering can report a false shortage when changing one earlier
    selection would free a later candidate.  The binary program enforces exact
    group quotas and every pairwise diversity conflict simultaneously.
    """
    if any(len(by_group[group]) < per_group for group in groups):
        return None, {"status": "insufficient_automatic_candidates"}
    candidates = [row for group in groups for row in by_group[group]]
    candidate_index = {id(row): index for index, row in enumerate(candidates)}
    conflict_pairs: list[tuple[int, int, str]] = []
    for right_index, right in enumerate(candidates):
        for left_index in range(right_index):
            reason = _conflicts(right, [candidates[left_index]])
            if reason:
                conflict_pairs.append((left_index, right_index, reason))

    n_group_rows = len(groups)
    matrix = lil_matrix((n_group_rows + len(conflict_pairs), len(candidates)), dtype=np.float64)
    lower = np.full(matrix.shape[0], -np.inf, dtype=np.float64)
    upper = np.ones(matrix.shape[0], dtype=np.float64)
    for row_index, group in enumerate(groups):
        for candidate in by_group[group]:
            matrix[row_index, candidate_index[id(candidate)]] = 1.0
        lower[row_index] = per_group
        upper[row_index] = per_group
    for offset, (left, right, _) in enumerate(conflict_pairs, start=n_group_rows):
        matrix[offset, left] = 1.0
        matrix[offset, right] = 1.0

    # Minimize the negative quality score. Difficulty dominates, generation
    # margin breaks ties, and the tiny index term makes equivalent optima stable.
    quality = np.array([
        _difficulty(row) + 0.01 * _margin(row) - 1e-9 * index
        for index, row in enumerate(candidates)
    ])
    result = milp(
        c=-quality,
        integrality=np.ones(len(candidates), dtype=np.int8),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 300.0, "mip_rel_gap": 0.0},
    )
    audit = {
        "status": str(result.message), "success": bool(result.success),
        "candidate_variables": len(candidates), "pairwise_conflicts": len(conflict_pairs),
        "objective": float(-result.fun) if result.fun is not None else None,
    }
    if not result.success or result.x is None:
        return None, audit
    selected = [row for index, row in enumerate(candidates) if result.x[index] > 0.5]
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--round", action="append", nargs=2, metavar=("NAME", "ROOT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    accepted_path = args.output / "accepted_descriptions.jsonl"
    if accepted_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace {accepted_path}; pass --force")
    config = load_config(args.config)
    concepts = [item["name"] for item in config["concepts"]]
    facets = [item["id"] for item in config["facets"]]
    per_group = int(config["candidate_validation"]["accepted_per_concept_facet"])

    candidates: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    text_validation: list[dict[str, Any]] = []
    round_summaries = []
    total_failure_reasons: Counter[str] = Counter()
    for round_name, root_text in args.round:
        root = Path(root_text).resolve()
        local_candidates = read_jsonl(root / "candidate_descriptions.jsonl")
        candidate_map = {row["candidate_id"]: _namespace(row, round_name) for row in local_candidates}
        local_decisions = read_csv(root / "candidate_generation_decisions.csv")
        accepted_count = 0
        for decision in local_decisions:
            namespaced = _namespace(decision, round_name)
            decisions.append(namespaced)
            candidate = candidate_map[decision["candidate_id"]]
            candidate["automatic_decision"] = decision["automatic_decision"]
            candidate["automatic_reason"] = decision.get("automatic_reason", "")
            candidate["mean_target_score"] = decision.get("mean_target_score", "")
            candidate["mean_target_margin"] = decision.get("mean_target_margin", "")
            if decision["automatic_decision"] == "accepted":
                accepted_count += 1
        candidates.extend(candidate_map.values())
        for row in read_csv(root / "generation_validation.csv"):
            generation.append(_namespace(row, round_name))
        for row in read_csv(root / "manual_review.csv"):
            namespaced = _namespace(row, round_name)
            namespaced["manual_decision"] = "unset"
            namespaced["manual_notes"] = "merged_round; automatic decisions only"
            reviews.append(namespaced)
        for row in read_csv(root / "candidate_text_validation.csv"):
            if row.get("candidate_id") not in {"__group_count__", "__source_count__"}:
                text_validation.append(_namespace(row, round_name))

        pool_validation_path = root / "pool_validation" / "candidate_text_validation.csv"
        audit_validation = read_csv(pool_validation_path if pool_validation_path.exists() else root / "candidate_text_validation.csv")
        audit_rows = [
            row for row in audit_validation
            if row.get("candidate_id") not in {"__group_count__", "__source_count__"}
        ]
        text_valid = sum(str(row.get("text_valid", "")).casefold() == "true" for row in audit_rows)
        for row in audit_rows:
            if str(row.get("text_valid", "")).casefold() != "true":
                for reason in str(row.get("failure_reasons", "")).split(";"):
                    if reason:
                        total_failure_reasons[reason.split(":", 1)[0]] += 1
        raw_pool_path = root / "candidate_pool.jsonl"
        raw_candidate_count = len(read_jsonl(raw_pool_path)) if raw_pool_path.exists() else len(audit_rows)
        stage1_pass = sum(row.get("stage1_status") == "pass" for row in local_decisions)
        stage2_complete = sum(int(float(row.get("generated_seed_count", 0))) == int(float(row.get("expected_seed_count", 3))) for row in local_decisions)
        state = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {}
        round_summaries.append({
            "round": round_name,
            "root": str(root),
            "raw_candidates": raw_candidate_count,
            "text_valid": text_valid,
            "text_rejected": len(audit_rows) - text_valid,
            "selected_generation_candidates": len(local_candidates),
            "stage1_pass": stage1_pass,
            "stage1_pass_rate": stage1_pass / max(1, len(local_decisions)),
            "stage2_complete": stage2_complete,
            "automatic_accepted": accepted_count,
            "automatic_accept_rate_among_stage2_complete": accepted_count / max(1, stage2_complete),
            "runner_status": state.get("status", "targeted_no_runner_state"),
            "runner_elapsed_seconds": state.get("elapsed_seconds"),
            "runner_started_utc": state.get("started_utc"),
        })

    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row.get("automatic_decision") == "accepted":
            by_group[(row["concept"], row["facet_id"])].append(row)
    for rows in by_group.values():
        rows.sort(key=lambda row: (-_difficulty(row), -_margin(row), row["candidate_id"]))

    # Try an exact global selection first. A deterministic greedy fallback keeps
    # per-group shortage diagnostics available while replenishment is incomplete.
    group_order = sorted(
        [(concept, facet) for concept in concepts for facet in facets],
        key=lambda key: (len(by_group[key]), key),
    )
    selected, solver_audit = _global_selection(by_group, group_order, per_group)
    shortages = []
    if selected is None:
        selected = []
        for concept, facet in group_order:
            chosen = 0
            for row in by_group[(concept, facet)]:
                if _conflicts(row, selected):
                    continue
                selected.append(row)
                chosen += 1
                if chosen == per_group:
                    break
            if chosen < per_group:
                shortages.append({
                    "concept": concept, "facet_id": facet, "required": per_group,
                    "automatic_available": len(by_group[(concept, facet)]),
                    "diversity_compatible": chosen, "shortage": per_group - chosen,
                })
    selected_ids: set[str] = {row["candidate_id"] for row in selected}
    conflicts = []
    for group_rows in by_group.values():
        for row in group_rows:
            if row["candidate_id"] in selected_ids:
                continue
            reason = _conflicts(row, selected)
            conflicts.append({
                "candidate_id": row["candidate_id"], "concept": row["concept"],
                "facet_id": row["facet_id"], "reason": reason or "objective_not_selected",
            })

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "candidate_descriptions.jsonl", candidates, overwrite=True)
    write_csv(args.output / "candidate_generation_decisions.csv", decisions)
    write_csv(args.output / "generation_validation.csv", generation)
    write_csv(args.output / "manual_review.csv", reviews)
    write_csv(args.output / "candidate_text_validation.csv", text_validation)
    write_csv(args.output / "merge_diversity_conflicts.csv", conflicts)
    write_csv(
        args.output / "facet_shortages.csv", shortages,
        ["concept", "facet_id", "required", "automatic_available", "diversity_compatible", "shortage"],
    )
    selection_rows = [
        {
            "candidate_id": row["candidate_id"], "concept": row["concept"],
            "facet_id": row["facet_id"], "eligible": row.get("automatic_decision") == "accepted",
            "selected": row["candidate_id"] in selected_ids,
            "reason": "automatic_accepted_merged" if row["candidate_id"] in selected_ids else "not_selected",
        }
        for row in candidates
    ]
    write_csv(args.output / "final_selection.csv", selection_rows)
    atomic_write_text(
        args.output / "merge_manifest.json",
        json.dumps({
            "config": str(args.config.resolve()), "rounds": round_summaries,
            "selected": len(selected), "shortages": shortages,
            "merged_utc": datetime.now(timezone.utc).isoformat(),
            "totals": {
                "raw_candidates": sum(row["raw_candidates"] for row in round_summaries),
                "text_valid": sum(row["text_valid"] for row in round_summaries),
                "text_rejected": sum(row["text_rejected"] for row in round_summaries),
                "selected_generation_candidates": sum(row["selected_generation_candidates"] for row in round_summaries),
                "stage1_pass": sum(row["stage1_pass"] for row in round_summaries),
                "stage2_complete": sum(row["stage2_complete"] for row in round_summaries),
                "automatic_accepted": sum(row["automatic_accepted"] for row in round_summaries),
            },
            "text_rejection_reasons": dict(total_failure_reasons.most_common()),
            "selection_solver": solver_audit,
            "selection_policy": "automatic three-seed accepted only; TF-IDF difficulty then margin; global final diversity",
        }, indent=2) + "\n",
    )
    shutil.copy2(args.config, args.output / "merge_config_source.json")
    if shortages:
        raise RuntimeError(f"Merged corpus still has {len(shortages)} deficient groups; see facet_shortages.csv")
    expected = len(concepts) * len(facets) * per_group
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} selected rows, got {len(selected)}")
    write_jsonl(accepted_path, sorted(selected, key=lambda row: (row["concept"], row["facet_id"], row["candidate_id"])), overwrite=True)
    print(f"merged={len(candidates)} accepted_pool={sum(len(rows) for rows in by_group.values())} selected={len(selected)}")


if __name__ == "__main__":
    main()
