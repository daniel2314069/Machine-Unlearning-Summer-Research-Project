from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PROJECT_ROOT = REPO_ROOT.parent
CELEB_SOURCE = REPO_ROOT / "generate_celeb.py"
OCE_SOURCE = REPO_ROOT / "oce.py"
CG_PATH = REPO_ROOT / "Cg.pt"
COCO_SOURCE = REPO_ROOT / "data" / "coco_30k.csv"
REFERENCE_ROOT = REPO_ROOT / "experiments" / "evaluation_references"

MODEL_ID = "CompVis/stable-diffusion-v1-4"
EDIT_DTYPE = "float32"
GENERATION_DTYPE = "bfloat16"
DEVICE = "cuda:0"
ERASE_SCALE = 1500.0
PRESERVE_GLOBAL_SCALE = 70.0
PRESERVE_CONCEPT_SCALE = 2.5
LAMB = 10.0
EXPECTED_MODULE_COUNT = 16
SMOKE_SEEDS = [42, 43, 44]
TEMPLATES = [
    "a portrait of {}",
    "a sketch of {}",
    "an oil painting of {}",
    "{} in an official photo",
    "an image capturing {} at a public event",
]
METHODS = ("original_sd", "single_anchor", "unique_anchor")
EDITED_METHODS = ("single_anchor", "unique_anchor")
CHECKPOINTS = {
    "single_anchor": HERE / "single_anchor" / "e50_single_celebrity.safetensors",
    "unique_anchor": HERE / "unique_anchor" / "e50_unique_50.safetensors",
}
REFERENCE_IDS = {
    1000: "sd14_mscoco30k_first1000_pndm50_cfg7p5_512_bf16",
    10000: "sd14_mscoco30k_first10000_pndm50_cfg7p5_512_bf16",
}
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Verbatim repository/Table 13 snapshot. Repository strings remain canonical;
# preflight records any mismatch and stops before loading the diffusion model.
TABLE13_E100 = [
    "Adam Driver", "Adriana Lima", "Amber Heard", "Amy Adams", "Andrew Garfield",
    "Angelina Jolie", "Anjelica Huston", "Anna Faris", "Anna Kendrick", "Anne Hathaway",
    "Arnold Schwarzenegger", "Barack Obama", "Beth Behrs", "Bill Clinton", "Bob Dylan",
    "Bob Marley", "Bradley Cooper", "Bruce Willis", "Bryan Cranston", "Cameron Diaz",
    "Channing Tatum", "Charlie Sheen", "Charlize Theron", "Chris Evans",
    "Chris Hemsworth", "Chris Pine", "Chuck Norris", "Courteney Cox", "Demi Lovato",
    "Drake", "Drew Barrymore", "Dwayne Johnson", "Ed Sheeran", "Elon Musk",
    "Elvis Presley", "Emma Stone", "Frida Kahlo", "George Clooney", "Glenn Close",
    "Gwyneth Paltrow", "Harrison Ford", "Hillary Clinton", "Hugh Jackman",
    "Idris Elba", "Jake Gyllenhaal", "James Franco", "Jared Leto", "Jason Momoa",
    "Jennifer Aniston", "Jennifer Lawrence", "Jennifer Lopez", "Jeremy Renner",
    "Jessica Biel", "Jessica Chastain", "John Oliver", "John Wayne", "Johnny Depp",
    "Julianne Hough", "Justin Timberlake", "Kate Bosworth", "Kate Winslet",
    "Leonardo Dicaprio", "Margot Robbie", "Mariah Carey", "Melania Trump",
    "Meryl Streep", "Mick Jagger", "Mila Kunis", "Milla Jovovich", "Morgan Freeman",
    "Nick Jonas", "Nicolas Cage", "Nicole Kidman", "Octavia Spencer", "Olivia Wilde",
    "Oprah Winfrey", "Paul Mccartney", "Paul Walker", "Peter Dinklage",
    "Philip Seymour Hoffman", "Reese Witherspoon", "Richard Gere", "Ricky Gervais",
    "Rihanna", "Robin Williams", "Ronald Reagan", "Ryan Gosling", "Ryan Reynolds",
    "Shia Labeouf", "Shirley Temple", "Spike Lee", "Stan Lee", "Theresa May",
    "Tom Cruise", "Tom Hanks", "Tom Hardy", "Tom Hiddleston", "Whoopi Goldberg",
    "Zac Efron", "Zayn Malik",
]
TABLE13_RETAINS = [
    "Aaron Paul", "Alec Baldwin", "Amanda Seyfried", "Amy Poehler", "Amy Schumer",
    "Amy Winehouse", "Andy Samberg", "Aretha Franklin", "Avril Lavigne", "Aziz Ansari",
    "Barry Manilow", "Ben Affleck", "Ben Stiller", "Benicio Del Toro", "Bette Midler",
    "Betty White", "Bill Murray", "Bill Nye", "Britney Spears", "Brittany Snow",
    "Bruce Lee", "Burt Reynolds", "Charles Manson", "Christie Brinkley",
    "Christina Hendricks", "Clint Eastwood", "Countess Vaughn", "Dakota Johnson",
    "Dane Dehaan", "David Bowie", "David Tennant", "Denise Richards", "Doris Day",
    "Dr Dre", "Elizabeth Taylor", "Emma Roberts", "Fred Rogers", "Gal Gadot",
    "George Bush", "George Takei", "Gillian Anderson", "Gordon Ramsey", "Halle Berry",
    "Harry Dean Stanton", "Harry Styles", "Hayley Atwell", "Heath Ledger",
    "Henry Cavill", "Jackie Chan", "Jada Pinkett Smith", "James Garner",
    "Jason Statham", "Jeff Bridges", "Jennifer Connelly", "Jensen Ackles",
    "Jim Morrison", "Jimmy Carter", "Joan Rivers", "John Lennon", "Johnny Cash",
    "Jon Hamm", "Judy Garland", "Julianne Moore", "Justin Bieber", "Kaley Cuoco",
    "Kate Upton", "Keanu Reeves", "Kim Jong Un", "Kirsten Dunst", "Kristen Stewart",
    "Krysten Ritter", "Lana Del Rey", "Leslie Jones", "Lily Collins", "Lindsay Lohan",
    "Liv Tyler", "Lizzy Caplan", "Maggie Gyllenhaal", "Matt Damon", "Matt Smith",
    "Matthew Mcconaughey", "Maya Angelou", "Megan Fox", "Mel Gibson",
    "Melanie Griffith", "Michael Cera", "Michael Ealy", "Natalie Portman",
    "Neil Degrasse Tyson", "Niall Horan", "Patrick Stewart", "Paul Rudd",
    "Paul Wesley", "Pierce Brosnan", "Prince", "Queen Elizabeth", "Rachel Dratch",
    "Rachel Mcadams", "Reba Mcentire", "Robert De Niro",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(text: str) -> str:
    return "_".join(text.casefold().replace("-", " ").split())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_state(stage: str, status: str = "complete", **extra: Any) -> None:
    path = HERE / "run_state.json"
    payload = read_json(path) if path.is_file() else {"started_at": utc_now()}
    timestamp = utc_now()
    payload.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_at": timestamp,
        **extra,
    }
    overall = (
        "complete"
        if stage == "complete" and status == "complete"
        else "failed"
        if status == "failed"
        else "in_progress"
        if status == "running"
        else "stage_complete"
    )
    if status == "complete":
        for stale_key in (
            "error_type",
            "error",
            "cleanup_performed",
            "resume_note",
        ):
            payload.pop(stale_key, None)
    payload.update(
        {
            "status": overall,
            "stage": stage,
            "updated_at": timestamp,
            **extra,
        }
    )
    write_json(path, payload)


def literal_assignment(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = ast.literal_eval(node.value)
                if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    raise TypeError(f"{name} is not a literal list[str]")
                return value
    raise KeyError(f"Could not find {name} in {path}")


def canonical_sets() -> tuple[list[str], list[str], list[str]]:
    e50 = literal_assignment(CELEB_SOURCE, "E50_LIST")
    e100 = literal_assignment(CELEB_SOURCE, "E100_LIST")
    retains = literal_assignment(CELEB_SOURCE, "PRESERVE_LIST")
    if e50 != e100[:50]:
        raise ValueError("Repository E50_LIST differs from E100_LIST[:50]")
    return e50, e100[50:], retains


def package_versions() -> dict[str, str | None]:
    names = ["torch", "diffusers", "transformers", "safetensors", "torch-fidelity"]
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def ensure_output_dirs() -> None:
    for path in (
        HERE / "single_anchor",
        HERE / "unique_anchor",
        HERE / "smoke_images",
        HERE / "smoke_grids",
        HERE / "celebrity_images",
        HERE / "gcd_metrics",
        HERE / "coco10k_metrics",
        HERE / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def preflight() -> dict[str, Any]:
    import torch
    from transformers import CLIPTextConfig

    ensure_output_dirs()
    update_state("preflight", "running")
    targets, anchors, retains = canonical_sets()
    groups = {"targets": targets, "unique_anchors": anchors, "retains": retains}
    checks: dict[str, Any] = {
        "e50_equals_e100_prefix": True,
        "counts": {name: len(values) for name, values in groups.items()},
        "within_group_unique": {
            name: len(values) == len(set(values)) for name, values in groups.items()
        },
    }
    sets = {name: set(values) for name, values in groups.items()}
    pairwise = {
        "targets_unique_anchors": sorted(sets["targets"] & sets["unique_anchors"]),
        "targets_retains": sorted(sets["targets"] & sets["retains"]),
        "unique_anchors_retains": sorted(
            sets["unique_anchors"] & sets["retains"]
        ),
    }
    checks["pairwise_intersections"] = pairwise
    checks["union_count"] = len(set().union(*sets.values()))
    table_diffs = {
        "e100": {
            "matches_exactly": targets + anchors == TABLE13_E100,
            "repo_only": sorted(set(targets + anchors) - set(TABLE13_E100)),
            "table_only": sorted(set(TABLE13_E100) - set(targets + anchors)),
            "ordered_mismatches": [
                {"index": i, "repository": repo, "table13": paper}
                for i, (repo, paper) in enumerate(
                    zip(targets + anchors, TABLE13_E100), start=1
                )
                if repo != paper
            ],
        },
        "retains": {
            "matches_exactly": retains == TABLE13_RETAINS,
            "repo_only": sorted(set(retains) - set(TABLE13_RETAINS)),
            "table_only": sorted(set(TABLE13_RETAINS) - set(retains)),
            "ordered_mismatches": [
                {"index": i, "repository": repo, "table13": paper}
                for i, (repo, paper) in enumerate(
                    zip(retains, TABLE13_RETAINS), start=1
                )
                if repo != paper
            ],
        },
    }
    checks["table13_comparison"] = table_diffs
    failures = []
    if checks["counts"] != {"targets": 50, "unique_anchors": 50, "retains": 100}:
        failures.append("group counts")
    if not all(checks["within_group_unique"].values()):
        failures.append("within-group uniqueness")
    if any(pairwise.values()):
        failures.append("pairwise disjointness")
    if checks["union_count"] != 200:
        failures.append("union count")
    if not all(row["matches_exactly"] for row in table_diffs.values()):
        failures.append("Table 13 exact ordered comparison")

    if not CG_PATH.is_file():
        failures.append("Cg.pt missing")
        cg_info: dict[str, Any] = {"path": str(CG_PATH), "readable": False}
    else:
        payload = torch.load(CG_PATH, map_location="cpu", weights_only=False)
        matrix = payload.get("C") if isinstance(payload, dict) else None
        cg_info = {
            "path": str(CG_PATH.resolve()),
            "sha256": sha256(CG_PATH),
            "readable": True,
            "keys": sorted(payload) if isinstance(payload, dict) else None,
            "shape": list(matrix.shape) if isinstance(matrix, torch.Tensor) else None,
            "dtype": str(matrix.dtype) if isinstance(matrix, torch.Tensor) else None,
            "finite": bool(torch.isfinite(matrix).all()) if isinstance(matrix, torch.Tensor) else False,
        }
        text_config = CLIPTextConfig.from_pretrained(
            MODEL_ID, subfolder="text_encoder", local_files_only=True
        )
        cg_info["model_text_hidden_size"] = int(text_config.hidden_size)
        cg_info["dimension_compatible"] = (
            isinstance(matrix, torch.Tensor)
            and matrix.ndim == 2
            and matrix.shape[0] == matrix.shape[1] == text_config.hidden_size
        )
        if not cg_info["finite"] or not cg_info["dimension_compatible"]:
            failures.append("Cg.pt finite/dimension validation")

    validation = {
        "status": "failed" if failures else "complete",
        "validated_at": utc_now(),
        "canonical_source": str(CELEB_SOURCE.resolve()),
        "table13_source": "https://arxiv.org/html/2605.28902#Sx9.T13",
        "repository_strings_are_canonical": True,
        "checks": checks,
        "cg": cg_info,
        "failures": failures,
    }
    write_json(HERE / "set_validation.json", validation)
    write_csv(
        HERE / "pairs.csv",
        [
            {"pair_index": index, "target": target, "unique_anchor": anchor}
            for index, (target, anchor) in enumerate(zip(targets, anchors), start=1)
        ],
    )
    md = [
        "# E50 unique-anchor set validation",
        "",
        f"Status: **{validation['status']}**",
        "",
        "| Group | Count | Unique |",
        "|---|---:|---:|",
        f"| Targets (E100 first 50) | {len(targets)} | {len(set(targets))} |",
        f"| Unique anchors (E100 last 50) | {len(anchors)} | {len(set(anchors))} |",
        f"| Retains | {len(retains)} | {len(set(retains))} |",
        "",
        f"Pairwise disjoint: `{not any(pairwise.values())}`. Union count: "
        f"`{checks['union_count']}`.",
        "",
        f"Table 13 ordered exact match: "
        f"`{all(row['matches_exactly'] for row in table_diffs.values())}`.",
        "",
        f"`Cg.pt`: `{cg_info.get('sha256')}`, shape "
        f"`{cg_info.get('shape')}`, finite `{cg_info.get('finite')}`.",
        "",
        "`pairs.csv` is an audit mapping only. The subspace objective consumes "
        "sets and does not use pairwise correspondence.",
        "",
    ]
    (HERE / "set_validation.md").write_text("\n".join(md), encoding="utf-8")
    protocol = {
        "experiment_name": "OCE E50 Unique-Anchor Stress Test",
        "claim_scope": "current official repository implementation stress test",
        "not_a_paper_number_reproduction": True,
        "base_model": MODEL_ID,
        "targets": targets,
        "anchor_sets": {"single_anchor": ["celebrity"], "unique_anchor": anchors},
        "retains": retains,
        "objective": "unchanged current oce.py subspace objective",
        "module_selector": "attn2 modules whose names end with to_v",
        "expected_module_count": EXPECTED_MODULE_COUNT,
        "edit": {
            "erase_scale": ERASE_SCALE,
            "preserve_global_scale": PRESERVE_GLOBAL_SCALE,
            "preserve_concept_scale": PRESERVE_CONCEPT_SCALE,
            "lamb": LAMB,
            "expand_prompts": False,
            "dtype": EDIT_DTYPE,
            "reflection_correction": "upstream oce.py behavior",
        },
        "generation": {
            "scheduler": "PNDMScheduler",
            "steps": 50,
            "cfg": 7.5,
            "resolution": [512, 512],
            "dtype": GENERATION_DTYPE,
        },
        "smoke": {"prompt_template": "a portrait of {}", "seeds": SMOKE_SEEDS},
        "celebrity": {
            "templates": TEMPLATES,
            "target_images_per_template": 2,
            "retain_images_per_template": 1,
            "seed_reset_per_prompt": 42,
        },
        "coco": {
            "first_1k_screening_required": True,
            "first_10k_requires_explicit_continue_flag": True,
            "original_reference_ids": REFERENCE_IDS,
        },
        "source_hashes": {
            "oce.py": sha256(OCE_SOURCE),
            "generate_celeb.py": sha256(CELEB_SOURCE),
            "Cg.pt": cg_info.get("sha256"),
            "coco_30k.csv": sha256(COCO_SOURCE),
            "runner": sha256(Path(__file__)),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "git_head": git_head(),
        },
        "resolved_at": utc_now(),
    }
    write_json(HERE / "resolved_protocol.json", protocol)
    if failures:
        update_state("preflight", "failed", failures=failures)
        raise RuntimeError(f"Preflight failed: {failures}")
    update_state("preflight")
    return protocol


def require_preflight() -> dict[str, Any]:
    path = HERE / "resolved_protocol.json"
    validation_path = HERE / "set_validation.json"
    if not path.is_file() or not validation_path.is_file():
        return preflight()
    validation = read_json(validation_path)
    if validation.get("status") != "complete":
        raise RuntimeError("Existing preflight did not complete successfully")
    protocol = read_json(path)
    current_hashes = {
        "oce.py": sha256(OCE_SOURCE),
        "generate_celeb.py": sha256(CELEB_SOURCE),
        "Cg.pt": sha256(CG_PATH),
        "coco_30k.csv": sha256(COCO_SOURCE),
    }
    changed = {
        name: {
            "resolved": protocol["source_hashes"].get(name),
            "current": digest,
        }
        for name, digest in current_hashes.items()
        if protocol["source_hashes"].get(name) != digest
    }
    if changed:
        raise RuntimeError(f"Protocol sources changed since preflight: {changed}")
    return protocol


def load_pipeline(dtype_name: str, vae_none: bool = False):
    import torch
    from diffusers import DiffusionPipeline

    torch.set_grad_enabled(False)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "local_files_only": True,
    }
    if vae_none:
        kwargs["vae"] = None
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, **kwargs).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    if type(pipe.scheduler).__name__ != "PNDMScheduler":
        raise RuntimeError(f"Unexpected scheduler: {type(pipe.scheduler).__name__}")
    return pipe


def projection_modules(unet: Any) -> list[tuple[str, Any]]:
    modules = [
        (name, module)
        for name, module in unet.named_modules()
        if "attn2" in name and name.endswith("to_v")
    ]
    if len(modules) != EXPECTED_MODULE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_MODULE_COUNT} attn2.to_v modules, got {len(modules)}"
        )
    return modules


def encode_last_tokens(pipe: Any, prompts: Iterable[str]) -> dict[str, Any]:
    import torch

    result = {}
    for prompt in dict.fromkeys(prompts):
        prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]
        tokenized = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        last_index = int(tokenized["attention_mask"].sum().item()) - 2
        result[prompt] = prompt_embeds[:, last_index, :].squeeze(0).detach().to(
            device=DEVICE, dtype=torch.float32
        )
    return result


def edit_weights(
    pipe: Any,
    embeddings: Mapping[str, Any],
    targets: Sequence[str],
    anchors: Sequence[str],
    retains: Sequence[str],
    cg: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    def basis(weight: Any, concepts: Sequence[str]):
        columns = []
        for concept in concepts:
            vector = weight @ embeddings[concept]
            columns.append(vector / (vector.norm() + 1e-8))
        return torch.linalg.qr(torch.stack(columns, dim=1), mode="reduced")[0]

    result, audit = {}, []
    for index, (name, module) in enumerate(projection_modules(pipe.unet), start=1):
        weight = module.weight.detach().to(DEVICE, torch.float32)
        target_basis = basis(weight, targets)
        anchor_basis = basis(weight, anchors)
        target_projector = target_basis @ target_basis.T
        anchor_projector = anchor_basis @ anchor_basis.T
        identity = torch.eye(weight.shape[0], device=DEVICE, dtype=torch.float32)
        objective = -ERASE_SCALE * target_projector @ (
            identity - anchor_projector
        )
        for concept in retains:
            projected = weight @ embeddings[concept]
            objective += PRESERVE_CONCEPT_SCALE * torch.outer(projected, projected)
        objective += PRESERVE_GLOBAL_SCALE * (weight @ cg @ weight.T)
        objective += LAMB * (weight @ weight.T)
        u, singular, vh = torch.linalg.svd(objective, full_matrices=False)
        rotation = u @ vh
        reflection = bool(torch.linalg.det(rotation).item() < 0)
        if reflection:
            # Exact current oce.py correction (column of R, not U).
            rotation[:, -1] *= -1
        new_weight = rotation @ weight
        result[f"{name}.weight"] = new_weight.cpu()
        orth_error = torch.linalg.matrix_norm(
            rotation.T @ rotation
            - torch.eye(rotation.shape[0], device=DEVICE, dtype=torch.float32)
        ) / rotation.shape[0]
        gram_error = torch.linalg.matrix_norm(
            new_weight.T @ new_weight - weight.T @ weight
        ) / torch.linalg.matrix_norm(weight.T @ weight).clamp_min(1e-12)
        audit.append(
            {
                "layer_index": index,
                "layer": name,
                "weight_shape": list(weight.shape),
                "reflection_detected": reflection,
                "objective_singular_min": float(singular[-1].item()),
                "objective_singular_max": float(singular[0].item()),
                "orthogonality_error": float(orth_error.item()),
                "weight_gram_relative_error": float(gram_error.item()),
            }
        )
    return result, audit


def span_stats(
    pipe: Any,
    embeddings: Mapping[str, Any],
    anchor_sets: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    import torch

    output: dict[str, Any] = {"computed_on": "unedited W0", "methods": {}}
    for method, anchors in anchor_sets.items():
        rows = []
        for index, (name, module) in enumerate(projection_modules(pipe.unet), start=1):
            weight = module.weight.detach().to(DEVICE, torch.float32)
            feature = weight @ torch.stack(
                [embeddings[anchor] for anchor in anchors], dim=1
            )
            expected_columns = 1 if method == "single_anchor" else 50
            if feature.shape[1] != expected_columns:
                raise RuntimeError(
                    f"{method} feature matrix has {feature.shape[1]} columns, "
                    f"expected {expected_columns}"
                )
            singular = torch.linalg.svdvals(feature)
            tolerance = (
                max(feature.shape)
                * torch.finfo(feature.dtype).eps
                * float(singular[0].item())
            )
            rank = int(torch.linalg.matrix_rank(feature).item())
            rows.append(
                {
                    "layer_index": index,
                    "layer": name,
                    "shape": list(feature.shape),
                    "dtype": str(feature.dtype),
                    "rank": rank,
                    "rank_api": "torch.linalg.matrix_rank default tolerance",
                    "reported_default_tolerance": tolerance,
                    "singular_values": [float(value) for value in singular.cpu()],
                }
            )
        output["methods"][method] = {"anchors": list(anchors), "layers": rows}
    return output


def checkpoint_manifest(
    method: str,
    path: Path,
    anchors: Sequence[str],
    targets: Sequence[str],
    retains: Sequence[str],
    audit: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "complete",
        "method": method,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256(path),
        "base_model": MODEL_ID,
        "cg_path": str(CG_PATH.resolve()),
        "cg_sha256": protocol["source_hashes"]["Cg.pt"],
        "targets": list(targets),
        "anchors": list(anchors),
        "retains": list(retains),
        "parameters": protocol["edit"],
        "module_names": [f"{row['layer']}.weight" for row in audit],
        "module_count": len(audit),
        "layer_audit": list(audit),
        "source_hashes": protocol["source_hashes"],
        "created_at": utc_now(),
    }


def validate_checkpoints(protocol: Mapping[str, Any]) -> None:
    from safetensors.torch import load_file

    manifests = {}
    for method in EDITED_METHODS:
        path = CHECKPOINTS[method]
        manifest_path = path.parent / "checkpoint_manifest.json"
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint/manifest for {method}")
        manifest = read_json(manifest_path)
        state = load_file(str(path))
        if sha256(path) != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"{method} checkpoint hash mismatch")
        if len(state) != EXPECTED_MODULE_COUNT:
            raise RuntimeError(f"{method} checkpoint contains {len(state)} keys")
        if not all(
            key.endswith("attn2.to_v.weight") and ".attn2." in key
            for key in state
        ):
            raise RuntimeError(f"{method} contains a non-attn2.to_v key")
        if sorted(state) != sorted(manifest["module_names"]):
            raise RuntimeError(f"{method} checkpoint keys differ from manifest")
        manifests[method] = manifest
    if manifests["single_anchor"]["module_names"] != manifests["unique_anchor"]["module_names"]:
        raise RuntimeError("Edited module sets differ between checkpoints")
    invariant_keys = ["base_model", "cg_sha256", "targets", "retains", "parameters", "module_names"]
    for key in invariant_keys:
        if manifests["single_anchor"][key] != manifests["unique_anchor"][key]:
            raise RuntimeError(f"Checkpoint invariant differs: {key}")
    if manifests["single_anchor"]["anchors"] != ["celebrity"]:
        raise RuntimeError("Single anchor was not preserved as a true one-column input")
    if manifests["unique_anchor"]["anchors"] != protocol["anchor_sets"]["unique_anchor"]:
        raise RuntimeError("Unique anchor set differs from protocol")


def prepare_weights() -> None:
    import torch
    from safetensors.torch import save_file

    protocol = require_preflight()
    update_state("weights", "running")
    targets = protocol["targets"]
    retains = protocol["retains"]
    anchor_sets = protocol["anchor_sets"]
    pipe = load_pipeline(EDIT_DTYPE, vae_none=True)
    all_prompts = targets + retains + ["celebrity"] + anchor_sets["unique_anchor"]
    embeddings = encode_last_tokens(pipe, all_prompts)
    if len(anchor_sets["single_anchor"]) != 1:
        raise RuntimeError("Single anchor list must contain exactly one string")
    stats = span_stats(pipe, embeddings, anchor_sets)
    write_json(HERE / "anchor_span_stats.json", stats)
    md = [
        "# Anchor span sanity",
        "",
        "Computed as `W0 C_anchor` on the same unedited float32 weights.",
        "",
        "| Method | Columns | Per-layer ranks |",
        "|---|---:|---|",
    ]
    for method in EDITED_METHODS:
        layers = stats["methods"][method]["layers"]
        md.append(
            f"| {method} | {layers[0]['shape'][1]} | "
            f"{', '.join(str(row['rank']) for row in layers)} |"
        )
    md.append("")
    (HERE / "anchor_span_stats.md").write_text("\n".join(md), encoding="utf-8")
    cg = torch.load(CG_PATH, map_location=DEVICE, weights_only=False)["C"].to(
        device=DEVICE, dtype=torch.float32
    )
    for method in EDITED_METHODS:
        path = CHECKPOINTS[method]
        manifest_path = path.parent / "checkpoint_manifest.json"
        if path.is_file() and manifest_path.is_file():
            print(f"[weights] validating existing {path}", flush=True)
            continue
        anchors = anchor_sets[method]
        print(f"[weights] building {method} with {len(anchors)} anchors", flush=True)
        state, audit = edit_weights(
            pipe, embeddings, targets, anchors, retains, cg
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, str(path))
        write_json(
            manifest_path,
            checkpoint_manifest(
                method, path, anchors, targets, retains, audit, protocol
            ),
        )
    validate_checkpoints(protocol)
    pipe.to("cpu")
    del pipe, cg, embeddings
    gc.collect()
    torch.cuda.empty_cache()
    update_state("weights")


def apply_state(pipe: Any, state: Mapping[str, Any]) -> None:
    incompatible = pipe.unet.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")


def base_projection_state(pipe: Any) -> dict[str, Any]:
    return {
        f"{name}.weight": module.weight.detach().clone()
        for name, module in projection_modules(pipe.unet)
    }


def unload(pipe: Any, *objects: Any) -> None:
    import torch

    pipe.to("cpu")
    del pipe
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def smoke_image_path(target: str, method: str, seed: int) -> Path:
    return HERE / "smoke_images" / slug(target) / method / f"seed_{seed}.png"


def build_smoke_grids(targets: Sequence[str]) -> None:
    from PIL import Image, ImageDraw

    for target in targets:
        destination = HERE / "smoke_grids" / f"{slug(target)}.png"
        if destination.is_file():
            continue
        cell = 512
        top, left = 28, 110
        canvas = Image.new("RGB", (left + 3 * cell, top + 3 * cell), "white")
        draw = ImageDraw.Draw(canvas)
        for column, seed in enumerate(SMOKE_SEEDS):
            draw.text((left + column * cell + 8, 7), f"seed {seed}", fill="black")
        for row, method in enumerate(METHODS):
            draw.text((5, top + row * cell + 8), method, fill="black")
            for column, seed in enumerate(SMOKE_SEEDS):
                with Image.open(smoke_image_path(target, method, seed)) as image:
                    canvas.paste(image.convert("RGB"), (left + column * cell, top + row * cell))
        canvas.save(destination)


def validate_smoke(targets: Sequence[str]) -> None:
    expected = [
        smoke_image_path(target, method, seed)
        for target in targets
        for method in METHODS
        for seed in SMOKE_SEEDS
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    corrupt = []
    from PIL import Image
    for path in expected:
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            corrupt.append(str(path))
    grids = list((HERE / "smoke_grids").glob("*.png"))
    if missing or corrupt or len(expected) != 450 or len(grids) != 50:
        raise RuntimeError(
            f"Smoke validation failed: expected={len(expected)}, "
            f"missing={len(missing)}, corrupt={len(corrupt)}, grids={len(grids)}"
        )


def run_smoke() -> None:
    import torch
    from safetensors.torch import load_file

    protocol = require_preflight()
    update_state("smoke", "running")
    validate_checkpoints(protocol)
    pipe = load_pipeline(GENERATION_DTYPE)
    states = {"original_sd": base_projection_state(pipe)}
    states.update(
        {method: load_file(str(CHECKPOINTS[method])) for method in EDITED_METHODS}
    )
    for method in METHODS:
        apply_state(pipe, states[method])
        for target in protocol["targets"]:
            prompt = f"a portrait of {target}"
            pending = [
                seed
                for seed in SMOKE_SEEDS
                if not smoke_image_path(target, method, seed).is_file()
            ]
            if not pending:
                continue
            for seed in pending:
                smoke_image_path(target, method, seed).parent.mkdir(
                    parents=True, exist_ok=True
                )
            images = pipe(
                prompt=[prompt] * len(pending),
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=[
                    torch.Generator(device=DEVICE).manual_seed(seed)
                    for seed in pending
                ],
            ).images
            for seed, image in zip(pending, images):
                path = smoke_image_path(target, method, seed)
                image.save(path)
                print(f"[smoke] {method} {target} seed={seed}", flush=True)
    build_smoke_grids(protocol["targets"])
    validate_smoke(protocol["targets"])
    unload(pipe, states)
    update_state("smoke", smoke_images=450, smoke_grids=50)


def celebrity_manifest(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for set_name, concepts, n_images in (
            ("targets", protocol["targets"], 2),
            ("retains", protocol["retains"], 1),
        ):
            for concept in concepts:
                for template_index, template in enumerate(TEMPLATES):
                    prompt = template.format(concept)
                    for sample_index in range(n_images):
                        relative = (
                            Path("celebrity_images")
                            / method
                            / set_name
                            / slug(concept)
                            / f"template_{template_index}_sample_{sample_index}.png"
                        )
                        rows.append(
                            {
                                "method": method,
                                "set": set_name,
                                "celebrity": concept,
                                "template_index": template_index,
                                "template": template,
                                "prompt": prompt,
                                "sample_index": sample_index,
                                "seed_reset_per_prompt": 42,
                                "relative_path": str(relative),
                            }
                        )
    return rows


def validate_celebrity_images(rows: Sequence[Mapping[str, Any]]) -> None:
    counts = {}
    missing = []
    for row in rows:
        key = (row["method"], row["set"])
        counts[key] = counts.get(key, 0) + 1
        if not (HERE / row["relative_path"]).is_file():
            missing.append(row["relative_path"])
    expected = {
        (method, set_name): 500
        for method in METHODS
        for set_name in ("targets", "retains")
    }
    if counts != expected or missing:
        raise RuntimeError(
            f"Celebrity validation failed: counts={counts}, missing={len(missing)}"
        )


def generate_celebrity(batch_size: int) -> None:
    import torch
    from safetensors.torch import load_file

    protocol = require_preflight()
    update_state("celebrity_generation", "running")
    validate_checkpoints(protocol)
    rows = celebrity_manifest(protocol)
    write_csv(HERE / "celebrity_images" / "manifest.csv", rows)
    pipe = load_pipeline(GENERATION_DTYPE)
    states = {"original_sd": base_projection_state(pipe)}
    states.update(
        {method: load_file(str(CHECKPOINTS[method])) for method in EDITED_METHODS}
    )
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["method"], row["set"], row["celebrity"], row["template_index"])
        grouped.setdefault(key, []).append(row)
    for method in METHODS:
        apply_state(pipe, states[method])
        method_groups = [
            prompt_rows
            for key, prompt_rows in grouped.items()
            if key[0] == method
            and not all((HERE / row["relative_path"]).is_file() for row in prompt_rows)
        ]
        packed: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        for prompt_rows in method_groups:
            # Never split a two-image target prompt across batches: both images
            # must consume one seed-42 CPU generator sequentially, exactly as
            # generate_celeb.py does for num_images_per_prompt=2.
            if current and len(current) + len(prompt_rows) > batch_size:
                packed.append(current)
                current = []
            current.extend(prompt_rows)
        if current:
            packed.append(current)
        for batch_index, batch_rows in enumerate(packed, start=1):
            generators = []
            by_prompt: dict[tuple[str, int], Any] = {}
            for row in batch_rows:
                key = (row["celebrity"], int(row["template_index"]))
                generator = by_prompt.setdefault(
                    key, torch.Generator(device="cpu").manual_seed(42)
                )
                generators.append(generator)
            images = pipe(
                prompt=[str(row["prompt"]) for row in batch_rows],
                num_inference_steps=50,
                guidance_scale=7.5,
                num_images_per_prompt=1,
                generator=generators,
            ).images
            for row, image in zip(batch_rows, images):
                path = HERE / row["relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
            print(
                f"[celebrity] {method} batch={batch_index}/{len(packed)} "
                f"images={len(batch_rows)}",
                flush=True,
            )
    validate_celebrity_images(rows)
    unload(pipe, states)
    update_state("celebrity_generation", celebrity_images=3000)


def harmonic_score(acc_e: float, acc_s: float) -> float:
    erasure_success = 1.0 - acc_e
    if erasure_success <= 0 or acc_s <= 0:
        return 0.0
    return 2.0 / (1.0 / erasure_success + 1.0 / acc_s)


def evaluate_gcd(gcd_project_root: Path | None) -> None:
    import pandas as pd
    from dotenv import load_dotenv
    from skimage import io
    from tqdm import tqdm

    protocol = require_preflight()
    manifest_path = HERE / "celebrity_images" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Generate celebrity images before GCD evaluation")
    validate_celebrity_images(celebrity_manifest(protocol))
    update_state("gcd_evaluation", "running")
    root = gcd_project_root or (
        Path(os.environ["GCD_PROJECT_ROOT"]) if "GCD_PROJECT_ROOT" in os.environ else None
    )
    if root is None:
        raise RuntimeError(
            "GCD evaluator is external to this repository. Supply "
            "--gcd-project-root or GCD_PROJECT_ROOT."
        )
    root = root.resolve()
    if not (root / "model_training").is_dir():
        raise FileNotFoundError(f"No model_training package under {root}")
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from model_training.helpers.labels import Labels
    from model_training.preprocessors.face_detection.face_detector import FaceDetector
    from model_training.helpers.face_recognizer import FaceRecognizer
    from model_training.utils import preprocess_image

    resources = os.getenv("APP_DATA_DIR")
    if not resources:
        raise RuntimeError("APP_DATA_DIR is not set by the GCD .env")
    image_size = int(os.getenv("APP_FACE_SIZE", "224"))
    labels = Labels(resources_path=resources)
    detector = FaceDetector(
        resources,
        margin=float(os.getenv("APP_FACE_MARGIN", "0.2")),
        use_cuda=os.getenv("APP_USE_CUDA", "").casefold() == "true",
    )
    recognizer = FaceRecognizer(
        labels=labels,
        resources_path=resources,
        use_cuda=os.getenv("USE_CUDA", "").casefold() == "true",
        top_n=5,
    )
    manifest = pd.read_csv(manifest_path)
    raw_rows = []
    for row in tqdm(manifest.to_dict("records"), desc="GCD"):
        path = HERE / row["relative_path"]
        detected = detector.perform_single(io.imread(path))
        faces = [preprocess_image(image, image_size) for image, _ in detected]
        predictions = recognizer.perform(faces)
        top = []
        if predictions:
            for label, probability in predictions[0][0]:
                name = str(label).split("_[", 1)[0].replace("_", " ")
                top.append({"name": name, "probability": float(probability)})
        top1 = top[0]["name"] if top else None
        raw_rows.append(
            {
                **row,
                "face_detected": bool(top),
                "top1": top1,
                "top1_probability": top[0]["probability"] if top else None,
                "correct": bool(top1 and top1.casefold() == row["celebrity"].casefold()),
                "top5_json": json.dumps(top, ensure_ascii=False),
            }
        )
    write_csv(HERE / "gcd_metrics" / "predictions.csv", raw_rows)
    per_rows = []
    for (method, set_name, celebrity), group in _group_rows(
        raw_rows, ("method", "set", "celebrity")
    ):
        denominator = sum(bool(row["face_detected"]) for row in group)
        correct = sum(bool(row["correct"]) for row in group)
        per_rows.append(
            {
                "method": method,
                "set": set_name,
                "celebrity": celebrity,
                "image_count": len(group),
                "no_face_count": len(group) - denominator,
                "denominator_face_detected": denominator,
                "correct_count": correct,
                "accuracy": correct / denominator if denominator else None,
            }
        )
    write_csv(HERE / "gcd_metrics" / "per_celebrity_accuracy.csv", per_rows)
    models = {}
    for method in METHODS:
        method_rows = [row for row in raw_rows if row["method"] == method]
        sets_summary = {}
        for set_name in ("targets", "retains"):
            subset = [row for row in method_rows if row["set"] == set_name]
            denominator = sum(bool(row["face_detected"]) for row in subset)
            correct = sum(bool(row["correct"]) for row in subset)
            sets_summary[set_name] = {
                "image_count": len(subset),
                "no_face_count": len(subset) - denominator,
                "denominator_face_detected": denominator,
                "correct_count": correct,
                "accuracy": correct / denominator if denominator else 0.0,
            }
        acc_e = sets_summary["targets"]["accuracy"]
        acc_s = sets_summary["retains"]["accuracy"]
        models[method] = {
            "Acc_e": acc_e,
            "Acc_s": acc_s,
            "H_o": harmonic_score(acc_e, acc_s),
            "sets": sets_summary,
        }
    metrics = {
        "status": "complete",
        "completed_at": utc_now(),
        "denominator_policy": "repo GCD behavior: exclude images with no detected face",
        "H_o_formula": "2 / ((1 - Acc_e)^-1 + Acc_s^-1)",
        "models": models,
        "predictions_csv": str((HERE / "gcd_metrics" / "predictions.csv").resolve()),
        "per_celebrity_csv": str(
            (HERE / "gcd_metrics" / "per_celebrity_accuracy.csv").resolve()
        ),
        "gcd_project_root": str(root),
        "protocol_hash": sha256(HERE / "resolved_protocol.json"),
    }
    write_json(HERE / "gcd_metrics" / "metrics.json", metrics)
    lines = [
        "# Celebrity GCD metrics",
        "",
        "| Model | Acc_e ↓ | Acc_s ↑ | H_o ↑ | Target no-face | Retain no-face |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = models[method]
        lines.append(
            f"| {method} | {row['Acc_e']:.4f} | {row['Acc_s']:.4f} | "
            f"{row['H_o']:.4f} | {row['sets']['targets']['no_face_count']} | "
            f"{row['sets']['retains']['no_face_count']} |"
        )
    lines.append("")
    (HERE / "gcd_metrics" / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    update_state("gcd_evaluation")


def _group_rows(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Iterable[tuple[tuple[Any, ...], list[Mapping[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return groups.items()


def expected_reference_identity(count: int) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "prompt_source_sha256": sha256(COCO_SOURCE),
        "prompt_subset": f"first {count} rows in source order",
        "prompt_count": count,
        "seed_column": "evaluation_seed",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "dtype": "bfloat16",
        "scheduler": "PNDMScheduler",
        "clip_model_id": CLIP_MODEL_ID,
        "clip_implementation": "transformers logits_per_image diagonal",
        "fid_implementation": "torch_fidelity 0.3.0",
        "fid_feature_extractor": "inception-v3-compat",
        "fid_feature_layer": "2048",
    }


def resolve_original_reference(count: int) -> dict[str, Any]:
    sys.path.insert(0, str(REFERENCE_ROOT))
    from reference_registry import resolve_reference

    entry = resolve_reference(
        REFERENCE_IDS[count], expected_reference_identity(count), require_complete=True
    )
    if entry is None:
        raise RuntimeError(
            f"Complete matching Original reference unavailable: {REFERENCE_IDS[count]}"
        )
    return entry


def coco_image_path(method: str, case_number: int) -> Path:
    return (
        HERE
        / "coco10k_metrics"
        / "generated_images"
        / method
        / f"{case_number}.png"
    )


def load_reference_prompts(entry: Mapping[str, Any], count: int):
    import pandas as pd

    manifest = Path(entry["artifacts"]["prompt_manifest"])
    frame = pd.read_csv(manifest)
    source = pd.read_csv(COCO_SOURCE).iloc[:count].reset_index(drop=True)
    frame = frame.iloc[:count].reset_index(drop=True)
    columns = ["case_number", "prompt", "evaluation_seed"]
    if len(frame) != count or not frame[columns].equals(source[columns]):
        raise RuntimeError("Registry prompt manifest is not aligned to source subset")
    return frame


def generate_coco_images(
    frame: Any,
    count: int,
    batch_size: int,
    methods: Sequence[str],
) -> None:
    import torch
    from safetensors.torch import load_file

    protocol = require_preflight()
    validate_checkpoints(protocol)
    pipe = load_pipeline(GENERATION_DTYPE)
    for method in methods:
        apply_state(pipe, load_file(str(CHECKPOINTS[method])))
        pending = frame[
            [
                not coco_image_path(method, int(case)).is_file()
                for case in frame["case_number"]
            ]
        ]
        done = count - len(pending)
        for offset in range(0, len(pending), batch_size):
            batch = pending.iloc[offset : offset + batch_size]
            images = pipe(
                prompt=batch["prompt"].astype(str).tolist(),
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=[
                    torch.Generator(device=DEVICE).manual_seed(int(seed))
                    for seed in batch["evaluation_seed"]
                ],
            ).images
            for case_number, image in zip(batch["case_number"], images):
                path = coco_image_path(method, int(case_number))
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
            done += len(batch)
            print(f"[coco] {method}: {done}/{count}", flush=True)
    unload(pipe)


def clip_score(frame: Any, method: str, batch_size: int) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID, local_files_only=True
    ).eval().to(DEVICE)
    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_ID, local_files_only=True
    )
    values = []
    for offset in range(0, len(frame), batch_size):
        batch = frame.iloc[offset : offset + batch_size]
        images = []
        for case in batch["case_number"]:
            with Image.open(coco_image_path(method, int(case))) as image:
                images.append(image.convert("RGB"))
        inputs = processor(
            text=batch["prompt"].astype(str).tolist(),
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = model(**{key: value.to(DEVICE) for key, value in inputs.items()})
        values.extend(float(value) for value in outputs.logits_per_image.diagonal().cpu())
    array = np.asarray(values, dtype=np.float64)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return {"mean": float(array.mean()), "std": float(array.std()), "count": len(values)}


def fid_to_reference(
    frame: Any, method: str, entry: Mapping[str, Any], count: int
) -> float:
    import numpy as np
    import torch
    import torch_fidelity
    from torch_fidelity.datasets import ImagesPathDataset

    paths = [
        str(coco_image_path(method, int(case))) for case in frame["case_number"]
    ]
    dataset = ImagesPathDataset(paths)
    stats_glob = Path(entry["artifacts"]["fid_statistics_glob"])
    matches = list(stats_glob.parent.glob(stats_glob.name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one reusable FID stats file, got {matches}")
    # input2 is required by the API but is never read on a cache hit.
    # torch_fidelity 0.3.0 stores FID statistics as NumPy arrays and calls
    # torch.load without an explicit weights_only argument. PyTorch >= 2.6
    # defaults that argument to True, so allowlist only the NumPy types present
    # in the repository-owned, fingerprint-validated reference cache.
    safe_numpy_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_numpy_globals):
        result = torch_fidelity.calculate_metrics(
            input1=dataset,
            input2=dataset,
            input2_cache_name=REFERENCE_IDS[count],
            cache_root=str(stats_glob.parent),
            cuda=True,
            fid=True,
            verbose=False,
        )
    return float(result["frechet_inception_distance"])


def evaluate_coco(
    count: int,
    batch_size: int,
    clip_batch_size: int,
    continue_to_10k: bool,
    methods: Sequence[str],
) -> None:
    if count not in REFERENCE_IDS:
        raise ValueError("COCO count must be 1000 or 10000")
    methods = tuple(dict.fromkeys(methods))
    if not methods or any(method not in EDITED_METHODS for method in methods):
        raise ValueError(f"Invalid edited methods: {methods}")
    stage_name = f"coco_{count}_{'_'.join(methods)}"
    update_state(stage_name, "running", methods=list(methods))
    if count == 10000:
        milestone = HERE / "coco10k_metrics" / "milestones" / "first1000" / "metrics.json"
        if not milestone.is_file() or read_json(milestone).get("status") != "complete":
            raise RuntimeError("Complete first-1k screening is required before first-10k")
        if not continue_to_10k:
            raise RuntimeError(
                "first-10k is gated. Review first-1k metrics, then pass "
                "--continue-to-10k explicitly."
            )
    entry = resolve_original_reference(count)
    frame = load_reference_prompts(entry, count)
    generate_coco_images(frame, count, batch_size, methods)
    for method in methods:
        missing = [
            int(case)
            for case in frame["case_number"]
            if not coco_image_path(method, int(case)).is_file()
        ]
        if missing:
            raise RuntimeError(f"{method} is missing {len(missing)} COCO images")
    original_payload = read_json(Path(entry["artifacts"]["clip_baseline"]))
    if original_payload["reference_identity"] != expected_reference_identity(count):
        raise RuntimeError("Original CLIP baseline identity mismatch")
    models: dict[str, Any] = {
        "original_sd": {
            "clip_score": original_payload["clip_score"],
            "fid_to_original_sd": 0.0,
        }
    }
    for method in methods:
        models[method] = {
            "clip_score": clip_score(frame, method, clip_batch_size),
            "fid_to_original_sd": fid_to_reference(frame, method, entry, count),
        }
    original_mean = models["original_sd"]["clip_score"]["mean"]
    metrics = {
        "status": "complete",
        "completed_at": utc_now(),
        "prompt_count": count,
        "reference_id": REFERENCE_IDS[count],
        "reference_fingerprint": entry["fingerprint"],
        "reused_original_reference": True,
        "original_images_regenerated": False,
        "models": models,
        "evaluated_methods": list(methods),
        "differences": {
            f"{method}_minus_original": {
                "clip_score_mean": models[method]["clip_score"]["mean"] - original_mean,
                "fid_to_original_sd": models[method]["fid_to_original_sd"],
            }
            for method in methods
        },
        "image_counts": {method: count for method in methods},
        "reference_artifacts": entry["artifacts"],
    }
    if set(methods) == set(EDITED_METHODS):
        metrics["unique_minus_single"] = {
            "clip_score_mean": (
                models["unique_anchor"]["clip_score"]["mean"]
                - models["single_anchor"]["clip_score"]["mean"]
            ),
            "fid_to_original_sd": (
                models["unique_anchor"]["fid_to_original_sd"]
                - models["single_anchor"]["fid_to_original_sd"]
            ),
        }
    if count == 1000 and set(methods) == set(EDITED_METHODS):
        root = HERE / "coco10k_metrics" / "milestones" / "first1000"
        write_json(root / "metrics.json", metrics)
        write_json(
            HERE / "coco10k_metrics" / "screening_gate.json",
            {
                "status": "awaiting_user_confirmation",
                "first1000_metrics": str((root / "metrics.json").resolve()),
                "note": "No automatic worth threshold was specified.",
                "continue_flag": "--continue-to-10k",
            },
        )
    elif count == 10000 and set(methods) == set(EDITED_METHODS):
        root = HERE / "coco10k_metrics"
        write_json(root / "metrics.json", metrics)
    else:
        root = (
            HERE
            / "coco10k_metrics"
            / "methods"
            / "_".join(methods)
            / f"first{count}"
        )
        write_json(root / "metrics.json", metrics)
    write_coco_summary(root / "summary.md", metrics)
    update_state(
        stage_name,
        coco_prompt_count=count,
        methods=list(methods),
        metrics_path=str((root / "metrics.json").resolve()),
        summary_path=str((root / "summary.md").resolve()),
    )


def write_coco_summary(path: Path, metrics: Mapping[str, Any]) -> None:
    lines = [
        f"# MSCOCO first-{metrics['prompt_count']} preservation",
        "",
        "| Model | CLIP ↑ | FID to identical Original reference ↓ |",
        "|---|---:|---:|",
    ]
    for method in metrics["models"]:
        row = metrics["models"][method]
        lines.append(
            f"| {method} | {row['clip_score']['mean']:.4f} ± "
            f"{row['clip_score']['std']:.4f} | {row['fid_to_original_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Reference: `{metrics['reference_id']}` "
            f"(`{metrics['reference_fingerprint']}`). Original images were not regenerated.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def final_validation() -> dict[str, Any]:
    protocol = require_preflight()
    validate_checkpoints(protocol)
    validate_smoke(protocol["targets"])
    celeb_rows = celebrity_manifest(protocol)
    validate_celebrity_images(celeb_rows)
    gcd = read_json(HERE / "gcd_metrics" / "metrics.json")
    coco = read_json(HERE / "coco10k_metrics" / "metrics.json")
    if gcd.get("status") != "complete" or coco.get("status") != "complete":
        raise RuntimeError("GCD and COCO 10k metrics must both be complete")
    if coco.get("prompt_count") != 10000:
        raise RuntimeError("Final COCO metrics are not first-10k")
    if coco.get("image_counts") != {"single_anchor": 10000, "unique_anchor": 10000}:
        raise RuntimeError("COCO metric image counts are invalid")
    return {
        "checkpoints": 2,
        "smoke_images": 450,
        "smoke_grids": 50,
        "celebrity_images": 3000,
        "coco_images": 20000,
        "gcd_complete": True,
        "coco10k_complete": True,
    }


def cleanup_coco_images() -> dict[str, Any]:
    root = (HERE / "coco10k_metrics" / "generated_images").resolve()
    expected = HERE.resolve() / "coco10k_metrics" / "generated_images"
    if root != expected or root.parent != (HERE / "coco10k_metrics").resolve():
        raise RuntimeError(f"Unsafe cleanup target: {root}")
    removed = []
    for method in EDITED_METHODS:
        directory = (root / method).resolve()
        if directory.parent != root:
            raise RuntimeError(f"Unsafe cleanup target: {directory}")
        count = len(list(directory.glob("*.png"))) if directory.is_dir() else 0
        if count != 10000:
            raise RuntimeError(
                f"Refusing cleanup: {method} has {count} PNGs, expected 10000"
            )
        shutil.rmtree(directory)
        removed.append({"method": method, "directory": str(directory), "count": count})
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()
    payload = {
        "status": "complete",
        "completed_at": utc_now(),
        "removed": removed,
        "removed_total": sum(row["count"] for row in removed),
        "preserved": [
            "checkpoints",
            "smoke images and grids",
            "celebrity images",
            "all JSON/CSV/Markdown metrics and manifests",
            "repository Original reference artifacts",
        ],
    }
    write_json(HERE / "coco10k_metrics" / "cleanup_manifest.json", payload)
    return payload


def finalize(keep_coco_images: bool) -> None:
    validation = final_validation()
    gcd = read_json(HERE / "gcd_metrics" / "metrics.json")
    coco = read_json(HERE / "coco10k_metrics" / "metrics.json")
    cleanup = (
        {"status": "skipped", "reason": "--keep-coco-images", "removed_total": 0}
        if keep_coco_images
        else cleanup_coco_images()
    )
    lines = [
        "# OCE E50 unique-anchor stress test",
        "",
        "This is a **current official repository implementation stress test**, "
        "not a reproduction of the paper's Wk / 1200 / 50 / 3 numbers.",
        "",
        "## Celebrity GCD",
        "",
        "| Model | Acc_e ↓ | Acc_s ↑ | H_o ↑ |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = gcd["models"][method]
        lines.append(
            f"| {method} | {row['Acc_e']:.4f} | {row['Acc_s']:.4f} | "
            f"{row['H_o']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## MSCOCO first-10k",
            "",
            "| Model | CLIP ↑ | FID to Original ↓ |",
            "|---|---:|---:|",
        ]
    )
    for method in METHODS:
        row = coco["models"][method]
        lines.append(
            f"| {method} | {row['clip_score']['mean']:.4f} | "
            f"{row['fid_to_original_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Final validation: `{json.dumps(validation, ensure_ascii=False)}`.",
            "",
            f"COCO cleanup removed `{cleanup['removed_total']}` PNG files.",
            "",
        ]
    )
    (HERE / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if keep_coco_images:
        write_json(HERE / "coco10k_metrics" / "cleanup_manifest.json", cleanup)
    update_state("complete", "complete", validation=validation, cleanup=cleanup)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable OCE E50 single-vs-unique anchor stress test."
    )
    parser.add_argument(
        "stage",
        choices=[
            "preflight",
            "weights",
            "smoke",
            "celebrity",
            "gcd",
            "coco",
            "finalize",
            "all",
        ],
    )
    parser.add_argument("--gcd-project-root", type=Path)
    parser.add_argument("--coco-count", type=int, choices=[1000, 10000], default=1000)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(EDITED_METHODS),
        default=list(EDITED_METHODS),
        help="Edited COCO methods to generate and evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--continue-to-10k", action="store_true")
    parser.add_argument("--keep-coco-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.clip_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    stage = args.stage
    if stage in {"preflight", "all"}:
        preflight()
    if stage in {"weights", "all"}:
        prepare_weights()
    if stage in {"smoke", "all"}:
        run_smoke()
    if stage in {"celebrity", "all"}:
        generate_celebrity(args.batch_size)
    if stage in {"gcd", "all"}:
        evaluate_gcd(args.gcd_project_root)
    if stage == "coco":
        evaluate_coco(
            args.coco_count,
            args.batch_size,
            args.clip_batch_size,
            args.continue_to_10k,
            args.methods,
        )
    if stage == "all":
        evaluate_coco(
            1000,
            args.batch_size,
            args.clip_batch_size,
            False,
            EDITED_METHODS,
        )
        if not args.continue_to_10k:
            print(
                "[gate] first-1k complete. Review metrics and rerun with "
                "--continue-to-10k.",
                flush=True,
            )
            return
        evaluate_coco(
            10000,
            args.batch_size,
            args.clip_batch_size,
            True,
            EDITED_METHODS,
        )
        finalize(args.keep_coco_images)
    if stage == "finalize":
        finalize(args.keep_coco_images)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update_state(
            parse_args().stage,
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            cleanup_performed=False,
            resume_note="Rerun the same stage; completed artifacts are reused.",
        )
        raise
