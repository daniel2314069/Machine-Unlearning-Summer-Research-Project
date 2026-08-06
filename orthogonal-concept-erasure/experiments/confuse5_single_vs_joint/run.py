#!/usr/bin/env python3
"""Plan and launch group-wise Confuse5 single-vs-joint baseline OCE runs.

The planning path intentionally uses only the Python standard library. It does
not import torch/diffusers or inspect model-dependent tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT_ROOT = HERE / "outputs"
OCE_ENTRYPOINT = REPO_ROOT / "oce.py"
CG_PATH = REPO_ROOT / "Cg.pt"
EDITABLE_MODULES = "unet.attn2.to_v"


class ConfigError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", normalize(value)).strip("-")
    if not result:
        raise ConfigError(f"Cannot form a safe slug from {value!r}")
    return result


def unique_strings(values: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ConfigError(f"{field} must be a list of strings")
    cleaned = [" ".join(item.split()) for item in values]
    if not allow_empty and not cleaned:
        raise ConfigError(f"{field} must not be empty")
    if any(not item for item in cleaned):
        raise ConfigError(f"{field} contains an empty concept")
    normalized = [normalize(item) for item in cleaned]
    duplicates = sorted({item for item in normalized if normalized.count(item) > 1})
    if duplicates:
        raise ConfigError(f"{field} contains duplicate normalized concepts: {duplicates}")
    return cleaned


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_and_validate(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read configuration {path}: {exc}") from exc
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    shared = config.get("shared")
    if not isinstance(shared, dict):
        raise ConfigError("shared must be an object")
    required_shared = {"base_model", "concept_type", "editable_modules", "anchor_policy", "retain_policy", "oce"}
    missing = sorted(required_shared - set(shared))
    if missing:
        raise ConfigError(f"shared is missing fields: {missing}")
    if shared["concept_type"] not in {"art", "object"}:
        raise ConfigError("shared.concept_type must be 'art' or 'object'")
    if not isinstance(shared["base_model"], str) or not shared["base_model"].strip():
        raise ConfigError("shared.base_model must be a non-empty string")
    if shared["editable_modules"] != EDITABLE_MODULES:
        raise ConfigError(f"Current oce.py only supports editable_modules={EDITABLE_MODULES!r}")
    retain_policy = shared["retain_policy"]
    if not isinstance(retain_policy, dict) or retain_policy.get("kind") not in {
        "group_similar_non_targets", "explicit_global"
    }:
        raise ConfigError(
            "shared.retain_policy.kind must be 'group_similar_non_targets' or 'explicit_global'"
        )
    if retain_policy["kind"] == "explicit_global":
        retain_policy["concepts"] = unique_strings(
            retain_policy.get("concepts"), "shared.retain_policy.concepts"
        )
    oce = shared["oce"]
    if not isinstance(oce, dict):
        raise ConfigError("shared.oce must be an object")
    required_oce = {"erase_scale", "preserve_global_scale", "preserve_concept_scale", "lamb", "expand_prompts", "dtype", "seed", "device"}
    missing = sorted(required_oce - set(oce))
    if missing:
        raise ConfigError(f"shared.oce is missing fields: {missing}")
    if oce["dtype"] != "float32":
        raise ConfigError("Current SD OCE editing path requires dtype='float32'")
    for field in ("erase_scale", "preserve_global_scale", "preserve_concept_scale", "lamb"):
        if not isinstance(oce[field], (int, float)) or isinstance(oce[field], bool):
            raise ConfigError(f"shared.oce.{field} must be numeric")
    if not isinstance(oce["expand_prompts"], bool):
        raise ConfigError("shared.oce.expand_prompts must be boolean")
    if oce["seed"] != 42:
        raise ConfigError("Current oce.py fixes guide alignment seed at 42; shared.oce.seed must be 42")
    if not isinstance(oce["device"], str) or not oce["device"].strip():
        raise ConfigError("shared.oce.device must be a non-empty string")
    anchor_policy = shared["anchor_policy"]
    if not isinstance(anchor_policy, dict) or anchor_policy.get("kind") not in {"oce_default", "per_target"}:
        raise ConfigError("anchor_policy.kind must be 'oce_default' or 'per_target'")
    per_target = anchor_policy.get("anchors", {})
    if anchor_policy["kind"] == "per_target" and not isinstance(per_target, dict):
        raise ConfigError("per_target anchor policy requires an anchors object")
    if anchor_policy["kind"] == "per_target":
        normalized_anchor_keys = [normalize(key) for key in per_target]
        if len(normalized_anchor_keys) != len(set(normalized_anchor_keys)):
            raise ConfigError("anchor_policy.anchors contains duplicate normalized target keys")

    groups = config.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ConfigError(
            "groups must contain at least one finalized Confuse5 group; the repository has no source list, so fill config.json with the scientific assignments"
        )
    seen_ids: set[str] = set()
    for index, group in enumerate(groups):
        prefix = f"groups[{index}]"
        if not isinstance(group, dict):
            raise ConfigError(f"{prefix} must be an object")
        group_id = group.get("id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ConfigError(f"{prefix}.id must be a non-empty string")
        safe_id = slug(group_id)
        if safe_id in seen_ids:
            raise ConfigError(f"Duplicate normalized group id: {safe_id}")
        seen_ids.add(safe_id)
        concepts = unique_strings(group.get("concepts"), f"{prefix}.concepts")
        targets = unique_strings(group.get("targets"), f"{prefix}.targets")
        similar = unique_strings(group.get("similar_non_targets"), f"{prefix}.similar_non_targets")
        concept_norms = {normalize(item) for item in concepts}
        target_norms = {normalize(item) for item in targets}
        similar_norms = {normalize(item) for item in similar}
        if not target_norms <= concept_norms or not similar_norms <= concept_norms:
            raise ConfigError(f"{prefix}: targets and similar_non_targets must be included in concepts")
        overlap = sorted(target_norms & similar_norms)
        if overlap:
            raise ConfigError(f"{prefix}: concepts have conflicting target/non-target roles: {overlap}")
        if target_norms | similar_norms != concept_norms:
            unassigned = sorted(concept_norms - target_norms - similar_norms)
            raise ConfigError(f"{prefix}: every concept needs exactly one role; unassigned={unassigned}")
        if anchor_policy["kind"] == "per_target":
            normalized_anchors = {normalize(key): value for key, value in per_target.items()}
            missing_anchors = sorted(target_norms - set(normalized_anchors))
            if missing_anchors:
                raise ConfigError(f"{prefix}: missing per-target anchors for {missing_anchors}")
            for target in targets:
                anchor = normalized_anchors[normalize(target)]
                if not isinstance(anchor, str) or not anchor.strip():
                    raise ConfigError(f"Anchor for {target!r} must be a non-empty string")
        group["id"] = safe_id
        group["concepts"] = concepts
        group["targets"] = targets
        group["similar_non_targets"] = similar
    return config


def resolve_anchors(targets: Iterable[str], shared: dict[str, Any]) -> list[str]:
    targets = list(targets)
    policy = shared["anchor_policy"]
    if policy["kind"] == "oce_default":
        anchor = "art" if shared["concept_type"] == "art" else " "
        return [anchor] * len(targets)
    lookup = {normalize(key): value for key, value in policy["anchors"].items()}
    return [" ".join(lookup[normalize(target)].split()) for target in targets]


def resolve_retains(group: dict[str, Any], shared: dict[str, Any]) -> list[str]:
    policy = shared["retain_policy"]
    if policy["kind"] == "group_similar_non_targets":
        return list(group["similar_non_targets"])
    return list(policy["concepts"])


def make_run(group: dict[str, Any], mode: str, targets: list[str], shared: dict[str, Any], output_root: Path) -> dict[str, Any]:
    variant = "joint" if mode == "joint" else f"single/{slug(targets[0])}"
    run_dir = output_root / group["id"] / variant
    return {
        "mode": mode,
        "group_id": group["id"],
        "target_concepts": targets,
        "similar_non_target_concepts": group["similar_non_targets"],
        "evaluation_non_target_concepts": [
            concept for concept in group["concepts"]
            if normalize(concept) not in {normalize(target) for target in targets}
        ],
        "resolved_anchors": resolve_anchors(targets, shared),
        "retain_policy": shared["retain_policy"]["kind"],
        "retain_concepts": resolve_retains(group, shared),
        "base_model": shared["base_model"],
        "editable_modules": shared["editable_modules"],
        "oce_hyperparameters": shared["oce"],
        "checkpoint_path": str(run_dir / "weights.safetensors"),
        "metadata_path": str(run_dir / "metadata.json"),
    }


def build_plan(config: dict[str, Any], config_path: Path, output_root: Path, selected: set[str] | None, mode: str) -> dict[str, Any]:
    groups = config["groups"]
    if selected is not None:
        known = {group["id"] for group in groups}
        unknown = selected - known
        if unknown:
            raise ConfigError(f"Unknown group ids: {sorted(unknown)}; available={sorted(known)}")
        groups = [group for group in groups if group["id"] in selected]
    runs: list[dict[str, Any]] = []
    for group in groups:
        if mode in {"single", "both"}:
            runs.extend(make_run(group, "single", [target], config["shared"], output_root) for target in group["targets"])
        if mode in {"joint", "both"}:
            runs.append(make_run(group, "joint", group["targets"], config["shared"], output_root))
    checkpoint_paths = [run["checkpoint_path"] for run in runs]
    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        collisions = sorted({path for path in checkpoint_paths if checkpoint_paths.count(path) > 1})
        raise ConfigError(f"Resolved output path collision(s): {collisions}")
    return {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id", "confuse5_single_vs_joint"),
        "requested_mode": mode,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "code": {"git_head": git_head(), "oce_py_sha256": sha256(OCE_ENTRYPOINT), "runner_sha256": sha256(Path(__file__))},
        "required_server_artifacts": {
            "generic_preservation_term": str(CG_PATH),
            "present_when_planned": CG_PATH.is_file(),
        },
        "shared_parameters": config["shared"],
        "selected_groups": [group["id"] for group in groups],
        "runs": runs,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def oce_command(run: dict[str, Any], shared: dict[str, Any]) -> list[str]:
    checkpoint = Path(run["checkpoint_path"])
    oce = shared["oce"]
    command = [
        sys.executable, str(OCE_ENTRYPOINT),
        "--edit_concepts", ";".join(run["target_concepts"]),
        "--preserve_concepts", ";".join(run["retain_concepts"]),
        "--concept_type", shared["concept_type"],
        "--model_id", shared["base_model"],
        "--device", str(oce["device"]),
        "--erase_scale", str(oce["erase_scale"]),
        "--preserve_global_scale", str(oce["preserve_global_scale"]),
        "--preserve_concept_scale", str(oce["preserve_concept_scale"]),
        "--lamb", str(oce["lamb"]),
        "--expand_prompts", str(oce["expand_prompts"]).lower(),
        "--save_dir", str(checkpoint.parent),
        "--exp_name", checkpoint.stem,
    ]
    # Omitting this flag is semantically important for oce.py's default object
    # anchor: its single blank string would otherwise be stripped by CLI parsing.
    if shared["anchor_policy"]["kind"] == "per_target":
        command[4:4] = ["--guide_concepts", ";".join(run["resolved_anchors"])]
    return command


def execute(plan: dict[str, Any], *, skip_completed: bool, overwrite: bool) -> None:
    if not CG_PATH.is_file():
        raise FileNotFoundError(
            f"Required OCE generic preservation term is missing: {CG_PATH}. "
            "Place the protocol-matching server artifact there before running edits."
        )
    shared = plan["shared_parameters"]
    for run in plan["runs"]:
        checkpoint = Path(run["checkpoint_path"])
        metadata_path = Path(run["metadata_path"])
        existing_complete = False
        if checkpoint.is_file() and metadata_path.is_file():
            try:
                existing_complete = json.loads(metadata_path.read_text(encoding="utf-8")).get("status") == "complete"
            except json.JSONDecodeError:
                existing_complete = False
        if skip_completed and existing_complete:
            print(f"[skip complete] {checkpoint}")
            continue
        if (checkpoint.exists() or metadata_path.exists()) and not overwrite:
            raise FileExistsError(f"Refusing output collision at {checkpoint.parent}; use --skip-completed or explicit --overwrite")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**run, "status": "running", "started_at": utc_now(), "code": plan["code"], "config_sha256": plan["config_sha256"]}
        write_json_atomic(metadata_path, metadata)
        command = oce_command(run, shared)
        print("[run]", " ".join(command), flush=True)
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except Exception:
            metadata.update(status="failed", finished_at=utc_now())
            write_json_atomic(metadata_path, metadata)
            raise
        if not checkpoint.is_file():
            raise RuntimeError(f"OCE exited successfully but checkpoint is missing: {checkpoint}")
        metadata.update(status="complete", finished_at=utc_now(), checkpoint_sha256=sha256(checkpoint))
        write_json_atomic(metadata_path, metadata)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("single", "joint", "both"), default="both")
    parser.add_argument("--groups", nargs="+", help="Selected group ids; default: all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without importing/loading OCE")
    parser.add_argument("--plan-path", type=Path, help="Optionally save the machine-readable plan")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing an existing run output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_and_validate(args.config)
        selected = {slug(item) for item in args.groups} if args.groups else None
        plan = build_plan(config, args.config, args.output_root.resolve(), selected, args.mode)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(plan, indent=2, ensure_ascii=False)
    if args.dry_run:
        print(rendered)
        if args.plan_path:
            write_json_atomic(args.plan_path, plan)
        return 0
    if args.plan_path:
        write_json_atomic(args.plan_path, plan)
    execute(plan, skip_completed=args.skip_completed, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
