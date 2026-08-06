from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from .config import config_concept_names
from .modeling import ClipEnsembleClassifier, load_original_pipeline, model_metadata
from .utils import atomic_write_text, read_csv, read_jsonl, set_reproducible_seed, write_csv, write_jsonl


GENERATION_FIELDS = [
    "candidate_id", "concept", "facet_id", "description", "seed", "stage", "image_path",
    "top1_concept", "top1_probability", "runner_up_concept", "runner_up_probability",
    "target_rank", "target_score", "target_margin", "normalized_entropy", "low_confidence",
    "confident_wrong", "class_probabilities", "class_logits",
]


def _bool(value: Any) -> bool:
    return str(value).casefold() in {"1", "true", "yes"}


def _valid_candidate_ids(output_dir: Path) -> set[str]:
    rows = read_csv(output_dir / "candidate_text_validation.csv")
    return {row["candidate_id"] for row in rows if _bool(row["text_valid"]) and row["candidate_id"] != "__group_count__"}


def _score_row(candidate: dict[str, Any], seed: int, stage: int, image_path: Path, score: dict[str, Any], config):
    probabilities = score["class_probabilities"]
    target = candidate["concept"]
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    target_score = probabilities[target]
    best_other = max(value for name, value in probabilities.items() if name != target)
    classifier = config["classifier"]
    low_confidence = (
        score["top1_probability"] < float(classifier["min_top1_probability"])
        or score["normalized_entropy"] > float(classifier["max_normalized_entropy"])
    )
    confident_wrong = (
        score["top1_concept"] != target
        and score["top1_probability"] >= float(classifier["high_confidence_wrong_probability"])
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "concept": target,
        "facet_id": candidate["facet_id"],
        "description": candidate["description"],
        "seed": seed,
        "stage": stage,
        "image_path": str(image_path),
        "top1_concept": score["top1_concept"],
        "top1_probability": score["top1_probability"],
        "runner_up_concept": score["runner_up_concept"],
        "runner_up_probability": score["runner_up_probability"],
        "target_rank": ranked.index(target) + 1,
        "target_score": target_score,
        "target_margin": target_score - best_other,
        "normalized_entropy": score["normalized_entropy"],
        "low_confidence": low_confidence,
        "confident_wrong": confident_wrong,
        "class_probabilities": json.dumps(probabilities, sort_keys=True),
        "class_logits": json.dumps(score["class_logits"], sort_keys=True),
    }


def _stage1_status(row: dict[str, Any], config: dict[str, Any]) -> str:
    classifier = config["classifier"]
    if row["top1_concept"] == row["concept"] and not _bool(row["low_confidence"]):
        if float(row["target_score"]) >= float(classifier["stage1_min_target_probability"]) and float(row["target_margin"]) >= float(classifier["stage1_min_margin"]):
            return "pass"
    if _bool(row["confident_wrong"]):
        return "reject"
    return "borderline"


def candidate_decisions(generation_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        grouped[row["candidate_id"]].append(row)
    expected_seeds = {int(seed) for seed in config["model"]["generation_seeds"]}
    classifier = config["classifier"]
    decisions = []
    for candidate_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        first = rows[0]
        present = {int(row["seed"]) for row in rows}
        target_top1 = sum(row["top1_concept"] == row["concept"] for row in rows)
        mean_target = sum(float(row["target_score"]) for row in rows) / len(rows)
        mean_margin = sum(float(row["target_margin"]) for row in rows) / len(rows)
        high_wrong = sum(_bool(row["confident_wrong"]) for row in rows)
        low_confidence = sum(_bool(row["low_confidence"]) for row in rows)
        stage1 = _stage1_status(first, config)
        complete = present == expected_seeds
        if not complete:
            automatic = "incomplete" if stage1 == "pass" else stage1
            reason = "missing_generation_seeds" if stage1 == "pass" else f"stage1_{stage1}"
        elif (
            target_top1 >= int(classifier["min_top1_seeds"])
            and mean_target >= float(classifier["min_target_probability"])
            and mean_margin >= float(classifier["min_average_margin"])
            and high_wrong == 0
            and low_confidence == 0
        ):
            automatic = "accepted"
            reason = "meets_all_automatic_thresholds"
        elif high_wrong > 0 or target_top1 == 0:
            automatic = "rejected"
            reason = "high_confidence_wrong_class" if high_wrong else "target_never_top1"
        else:
            automatic = "borderline"
            reason = "ambiguous_or_below_margin"
        decisions.append({
            "candidate_id": candidate_id,
            "concept": first["concept"],
            "facet_id": first["facet_id"],
            "description": first["description"],
            "stage1_status": stage1,
            "generated_seed_count": len(present),
            "expected_seed_count": len(expected_seeds),
            "target_top1_count": target_top1,
            "mean_target_score": mean_target,
            "mean_target_margin": mean_margin,
            "low_confidence_count": low_confidence,
            "confident_wrong_count": high_wrong,
            "automatic_decision": automatic,
            "automatic_reason": reason,
        })
    return decisions


def _contact_sheets(output_dir: Path, generation_rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    decision_map = {row["candidate_id"]: row["automatic_decision"] for row in decisions}
    categories = {"accepted": [], "rejected": [], "borderline": []}
    for row in generation_rows:
        decision = decision_map.get(row["candidate_id"], "borderline")
        category = decision if decision in categories else "borderline"
        categories[category].append(row)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    for category, rows in categories.items():
        target_dir = output_dir / "contact_sheets" / category
        target_dir.mkdir(parents=True, exist_ok=True)
        for page_index in range(0, len(rows), 80):
            page = rows[page_index:page_index + 80]
            canvas = Image.new("RGB", (5 * 220, math.ceil(len(page) / 5) * 250), "white")
            draw = ImageDraw.Draw(canvas)
            for slot, row in enumerate(page):
                x, y = (slot % 5) * 220, (slot // 5) * 250
                image = Image.open(row["image_path"]).convert("RGB")
                image.thumbnail((210, 205))
                canvas.paste(image, (x + 5, y + 5))
                label = f"{row['candidate_id']} s={row['seed']}\n{row['top1_concept']} p={float(row['top1_probability']):.2f} m={float(row['target_margin']):+.2f}"
                draw.multiline_text((x + 5, y + 212), label, fill="black", font=font, spacing=2)
            canvas.save(target_dir / f"page_{page_index // 80 + 1:03d}.jpg", quality=88)


def run_generation_validation(
    config: dict[str, Any], output_dir: str | Path, stage: str = "all", resume: bool = True
) -> bool:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(output_dir / "candidate_descriptions.jsonl")
    valid_ids = _valid_candidate_ids(output_dir)
    candidates = [row for row in candidates if row["candidate_id"] in valid_ids]
    candidate_map = {row["candidate_id"]: row for row in candidates}
    existing = read_csv(output_dir / "generation_validation.csv") if resume else []
    rows_by_key = {(row["candidate_id"], int(row["seed"])): row for row in existing}

    set_reproducible_seed(0)
    pipe = load_original_pipeline(config, purpose="generation", include_vae=True)
    metadata = model_metadata(pipe, config, projection="to_v")
    metadata.update({
        "global_seed": 0,
        "generation_dtype": config["model"]["generation_dtype"],
        "device": config["model"]["device"],
        "num_inference_steps": config["model"]["num_inference_steps"],
        "guidance_scale": config["model"]["guidance_scale"],
        "height": config["model"]["height"],
        "width": config["model"]["width"],
        "generation_seeds": config["model"]["generation_seeds"],
        "classifier": config["classifier"],
        "classifier_limitation": "CLIP probabilities are closed-set and not calibrated open-set probabilities; confidence, entropy, and margin gates are therefore required.",
    })
    metadata_path = output_dir / "run_metadata.json"
    if resume and existing and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        stable_keys = [
            "model_id", "checkpoint_revision", "projection_layers", "generation_dtype", "num_inference_steps",
            "guidance_scale", "height", "width", "generation_seeds", "classifier",
        ]
        mismatches = [key for key in stable_keys if previous.get(key) != metadata.get(key)]
        if mismatches:
            raise RuntimeError(f"Cached generation settings differ for keys: {mismatches}; use a new output directory")
    atomic_write_text(metadata_path, json.dumps(metadata, indent=2, default=str) + "\n")
    classifier = ClipEnsembleClassifier(config, config_concept_names(config))
    seeds = [int(seed) for seed in config["model"]["generation_seeds"]]
    deadline_epoch = float(os.environ.get("CONCEPT_CLUSTER_DEADLINE_EPOCH", "0") or 0)
    deadline_grace = float(os.environ.get("CONCEPT_CLUSTER_DEADLINE_GRACE_SECONDS", "300") or 300)
    stop_file_value = os.environ.get("CONCEPT_CLUSTER_STOP_FILE", "")
    stop_file = Path(stop_file_value) if stop_file_value else None

    def should_stop() -> bool:
        return bool(
            (stop_file is not None and stop_file.exists())
            or (deadline_epoch and time.time() >= deadline_epoch - deadline_grace)
        )

    def generate_one(candidate, seed, generation_stage):
        key = (candidate["candidate_id"], seed)
        image_path = output_dir / "generated_images" / candidate["candidate_id"] / f"seed_{seed}.png"
        if key in rows_by_key and image_path.exists():
            if rows_by_key[key].get("description") != candidate["description"]:
                raise RuntimeError(f"Cached description mismatch for {candidate['candidate_id']}; use a new output directory")
            return
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists() and resume:
            image = Image.open(image_path).convert("RGB")
        else:
            # Match the repository's generate_object.py use of torch.manual_seed: a CPU generator.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.inference_mode():
                image = pipe(
                    prompt=candidate["description"],
                    num_inference_steps=int(config["model"]["num_inference_steps"]),
                    guidance_scale=float(config["model"]["guidance_scale"]),
                    height=int(config["model"]["height"]),
                    width=int(config["model"]["width"]),
                    generator=generator,
                ).images[0]
            image.save(image_path)
        score = classifier.score(image)
        rows_by_key[key] = _score_row(candidate, seed, generation_stage, image_path, score, config)
        write_csv(output_dir / "generation_validation.csv", list(rows_by_key.values()), GENERATION_FIELDS)

    stopped = False
    if stage in {"1", "all"}:
        for candidate in candidates:
            if should_stop():
                stopped = True
                break
            generate_one(candidate, seeds[0], 1)

    current_rows = list(rows_by_key.values())
    decisions = candidate_decisions(current_rows, config)
    stage1_pass = {row["candidate_id"] for row in decisions if row["stage1_status"] == "pass"}
    # A human/vision review can rescue a visually correct Stage-1 image that the
    # independent classifier mislabeled. It is only promoted to Stage 2 here;
    # the two remaining seeds still have to be generated and reviewed.
    manual_reviews = {row["candidate_id"]: row for row in read_csv(output_dir / "manual_review.csv")}
    stage1_pass.difference_update(
        candidate_id
        for candidate_id, row in manual_reviews.items()
        if str(row.get("manual_decision", "unset")).strip().casefold() == "reject"
    )
    stage1_pass.update(
        candidate_id
        for candidate_id, row in manual_reviews.items()
        if str(row.get("manual_decision", "unset")).strip().casefold() == "accept"
    )
    if stage in {"2", "all"} and len(seeds) > 1:
        for candidate_id in sorted(stage1_pass):
            for seed in seeds[1:]:
                if should_stop():
                    stopped = True
                    break
                generate_one(candidate_map[candidate_id], seed, 2)
            if stopped:
                break

    final_rows = list(rows_by_key.values())
    final_rows.sort(key=lambda row: (row["candidate_id"], int(row["seed"])))
    write_csv(output_dir / "generation_validation.csv", final_rows, GENERATION_FIELDS)
    decisions = candidate_decisions(final_rows, config)
    write_csv(output_dir / "candidate_generation_decisions.csv", decisions)
    _initialize_manual_review(output_dir, decisions)
    _contact_sheets(output_dir, final_rows, decisions)
    return not stopped


def _initialize_manual_review(output_dir: Path, decisions: list[dict[str, Any]]) -> None:
    path = output_dir / "manual_review.csv"
    existing = {row["candidate_id"]: row for row in read_csv(path)}
    rows = []
    for decision in decisions:
        old = existing.get(decision["candidate_id"], {})
        rows.append({
            "candidate_id": decision["candidate_id"],
            "concept": decision["concept"],
            "facet_id": decision["facet_id"],
            "automatic_decision": decision["automatic_decision"],
            "mean_target_score": decision["mean_target_score"],
            "mean_target_margin": decision["mean_target_margin"],
            "manual_decision": old.get("manual_decision", "unset") or "unset",
            "manual_notes": old.get("manual_notes", ""),
            "description": decision["description"],
        })
    write_csv(path, rows)


def finalize_accepted(config: dict[str, Any], output_dir: str | Path, force: bool = False) -> None:
    output_dir = Path(output_dir)
    accepted_path = output_dir / "accepted_descriptions.jsonl"
    if accepted_path.exists() and not force:
        raise FileExistsError(f"Refusing to regenerate accepted data: {accepted_path}; pass --force explicitly")
    decisions = {row["candidate_id"]: row for row in read_csv(output_dir / "candidate_generation_decisions.csv")}
    reviews = {row["candidate_id"]: row for row in read_csv(output_dir / "manual_review.csv")}
    candidates = {row["candidate_id"]: row for row in read_jsonl(output_dir / "candidate_descriptions.jsonl")}
    per_group = int(config["candidate_validation"]["accepted_per_concept_facet"])
    accepted_source_quotas = {
        str(key): int(value)
        for key, value in config["candidate_validation"].get("diversity", {}).get(
            "accepted_source_quotas", {}
        ).items()
    }
    if accepted_source_quotas and sum(accepted_source_quotas.values()) != per_group:
        raise ValueError("accepted_source_quotas must sum to accepted_per_concept_facet")
    eligible: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    selection_rows = []
    for candidate_id, decision in decisions.items():
        manual = reviews.get(candidate_id, {}).get("manual_decision", "unset").strip().casefold() or "unset"
        if manual not in {"accept", "reject", "unset"}:
            raise ValueError(f"Invalid manual_decision={manual!r} for {candidate_id}")
        accepted = manual == "accept" or (manual == "unset" and decision["automatic_decision"] == "accepted")
        if manual == "reject":
            accepted = False
        row = {
            **candidates[candidate_id],
            "automatic_decision": decision["automatic_decision"],
            "manual_decision": manual,
            "mean_target_score": float(decision["mean_target_score"]),
            "mean_target_margin": float(decision["mean_target_margin"]),
        }
        if accepted:
            eligible[(row["concept"], row["facet_id"])].append(row)
        selection_rows.append({
            "candidate_id": candidate_id,
            "concept": row["concept"],
            "facet_id": row["facet_id"],
            "eligible": accepted,
            "selected": False,
            "reason": "manual_accept" if manual == "accept" else decision["automatic_reason"],
        })

    selected_ids: set[str] = set()
    shortages = []
    source_shortages = []
    for concept in [item["name"] for item in config["concepts"]]:
        for facet in [item["id"] for item in config["facets"]]:
            group = eligible[(concept, facet)]
            if config.get("tfidf_hard_selection"):
                group.sort(key=lambda row: (
                    row["manual_decision"] != "accept",
                    -float(row.get("tfidf_hardness", {}).get("tfidf_difficulty_score", 0.0)),
                    -row["mean_target_margin"],
                    row["candidate_id"],
                ))
            else:
                group.sort(key=lambda row: (row["manual_decision"] != "accept", -row["mean_target_margin"], row["candidate_id"]))
            if len(group) < per_group:
                shortages.append({
                    "concept": concept,
                    "facet_id": facet,
                    "required": per_group,
                    "eligible": len(group),
                    "shortage": per_group - len(group),
                })
            if accepted_source_quotas:
                for source, required in accepted_source_quotas.items():
                    source_rows = [row for row in group if row.get("source") == source]
                    if len(source_rows) < required:
                        source_shortages.append({
                            "concept": concept,
                            "facet_id": facet,
                            "source": source,
                            "required": required,
                            "eligible": len(source_rows),
                            "shortage": required - len(source_rows),
                        })
                    selected_ids.update(row["candidate_id"] for row in source_rows[:required])
            else:
                selected_ids.update(row["candidate_id"] for row in group[:per_group])
    for row in selection_rows:
        row["selected"] = row["candidate_id"] in selected_ids
        if row["eligible"] and not row["selected"]:
            row["reason"] = "eligible_not_selected_lower_margin"
    write_csv(output_dir / "final_selection.csv", selection_rows)
    write_csv(output_dir / "facet_shortages.csv", shortages, ["concept", "facet_id", "required", "eligible", "shortage"])
    write_csv(
        output_dir / "source_shortages.csv",
        source_shortages,
        ["concept", "facet_id", "source", "required", "eligible", "shortage"],
    )
    if shortages or source_shortages:
        raise RuntimeError(
            f"Final dataset has {len(shortages)} concept/facet shortages and "
            f"{len(source_shortages)} source-quota shortages; see facet_shortages.csv and source_shortages.csv"
        )
    selected = [candidates[candidate_id] for candidate_id in sorted(selected_ids)]
    write_jsonl(accepted_path, selected, overwrite=True)
