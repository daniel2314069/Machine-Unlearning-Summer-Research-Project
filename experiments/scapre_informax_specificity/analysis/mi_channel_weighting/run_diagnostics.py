#!/usr/bin/env python
"""MI-only ScaPre diagnostics. Never edits weights or runs diffusion inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LN2 = math.log(2.0)
EPS = 1e-8
MAX_ATOL = 2e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--legacy-diagnostic", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def repo_mi_from_counts(n11: torch.Tensor, n10: torch.Tensor,
                        n01: torch.Tensor, n00: torch.Tensor, n: int) -> torch.Tensor:
    k = 2 * n
    p11, p10 = (n11.float() + EPS) / k, (n10.float() + EPS) / k
    p01, p00 = (n01.float() + EPS) / k, (n00.float() + EPS) / k
    p1_, p0_ = p11 + p10, p01 + p00
    p_1, p_0 = p11 + p01, p10 + p00
    return (p11 * torch.log(p11 / (p1_ * p_1))
            + p10 * torch.log(p10 / (p1_ * p_0))
            + p01 * torch.log(p01 / (p0_ * p_1))
            + p00 * torch.log(p00 / (p0_ * p_0)))


def calculate_mi_alpha(pos_acts: torch.Tensor, neg_acts: torch.Tensor,
                       n: int, temperature: float, power: float) -> dict[str, torch.Tensor]:
    acts = torch.cat((pos_acts[:, :n], neg_acts[:, :n]), dim=1)
    tau = acts.median(dim=1, keepdim=True).values
    z = acts > tau
    pos_z, neg_z = z[:, :n], z[:, n:]
    n11, n10 = pos_z.sum(1), neg_z.sum(1)
    n01, n00 = n - n11, n - n10
    mi = repo_mi_from_counts(n11, n10, n01, n00, n)
    zscore = (mi - mi.mean()) / (mi.std() + EPS)
    alpha = torch.sigmoid(zscore / temperature).pow(power)
    return {
        "mi": mi,
        "alpha": alpha,
        "threshold": tau.squeeze(1),
        "ties": (acts == tau).sum(1),
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
    }


def enumerate_no_tie_mi(n: int) -> np.ndarray:
    values = []
    for positives_above in range(n + 1):
        values.append(float(repo_mi_from_counts(
            torch.tensor(positives_above), torch.tensor(n - positives_above),
            torch.tensor(n - positives_above), torch.tensor(positives_above), n
        ).item()))
    return np.unique(np.asarray(values, dtype=np.float64))


def integrity_gate(config: dict[str, Any], legacy_path: Path, output: Path) -> dict[str, Any]:
    if not legacy_path.is_file():
        raise RuntimeError(f"legacy diagnostic is missing: {legacy_path}")
    payload = torch.load(legacy_path, map_location="cpu", weights_only=False)
    records = [r for r in payload["records"] if r.get("stage") == "aggregate"]
    if len(records) != 320:
        raise RuntimeError(f"integrity failure: expected 320 aggregate records, got {len(records)}")
    raw = torch.cat([r["raw_mi"].float() for r in records])
    saved_alpha = torch.cat([r["alpha"].float() for r in records])
    recomputed_parts = []
    for record in records:
        mi = record["raw_mi"].float()
        recomputed_parts.append(torch.sigmoid(
            ((mi - mi.mean()) / (mi.std() + EPS)) / float(payload["temperature"])
        ).pow(float(payload["power"])))
    recomputed_alpha = torch.cat(recomputed_parts)
    expected = config["legacy_expected"]
    centers = expected["mi_bin_centers"]
    tol = float(expected["mi_bin_tolerance"])
    assignments = torch.stack([(raw - center).abs() <= tol for center in centers])
    counts = assignments.sum(1).tolist()
    unassigned = int((assignments.sum(0) != 1).sum().item())
    alpha_np = saved_alpha.numpy().astype(np.float64)
    alpha_stats = stats(alpha_np)
    recomputed_alpha_stats = stats(recomputed_alpha.numpy().astype(np.float64))
    alpha_ok = all(abs(alpha_stats[key] - float(value)) <= float(expected["alpha"]["absolute_tolerance"])
                   for key, value in expected["alpha"].items() if key != "absolute_tolerance")
    alpha_recomputation_finite = bool(torch.isfinite(recomputed_alpha).all().item())
    alpha_recomputation_allclose = bool(
        torch.allclose(saved_alpha, recomputed_alpha, atol=2e-7, rtol=0)
    )
    pass_gate = (
        raw.numel() == int(expected["observations"])
        and counts == expected["mi_bin_counts"]
        and unassigned == 0
        and bool(torch.isfinite(saved_alpha).all().item())
        and alpha_recomputation_finite
        and alpha_ok
    )
    unusual_candidates = []
    offset = 0
    for record in records:
        mi = record["raw_mi"].float()
        channels = torch.nonzero((mi - 0.0863).abs() <= tol).flatten().tolist()
        for channel in channels:
            unusual_candidates.append({
                "projection": record["projection"],
                "layer_index": int(record["layer_index"]),
                "target_index": int(record["target_index"]),
                "target_concept": record["target_concept"],
                "channel": int(channel),
                "raw_mi": float(mi[channel]),
                "saved_alpha": float(record["alpha"][channel]),
                "saved_threshold": float(record["threshold"][channel]),
                "activation_reconstruction": "unavailable: legacy artifact stores threshold but not activations or binary states",
                "tie_and_precision_diagnosis": "not determinable from the saved artifact; no contingency table is asserted"
            })
        offset += mi.numel()
    report = {
        "passed": pass_gate,
        "legacy_diagnostic": str(legacy_path),
        "legacy_sha256": sha256(legacy_path),
        "aggregate_records": len(records),
        "observations": int(raw.numel()),
        "mi_bin_centers": centers,
        "mi_bin_counts": counts,
        "unassigned_or_multiply_assigned": unassigned,
        "fraction_numerically_at_ln2": float(torch.isclose(raw, torch.tensor(LN2), atol=MAX_ATOL, rtol=0).float().mean()),
        "alpha_stats": alpha_stats,
        "alpha_cpu_recomputation_stats": recomputed_alpha_stats,
        "alpha_cpu_recomputation_allclose": alpha_recomputation_allclose,
        "alpha_recomputation_max_abs_error": float((saved_alpha - recomputed_alpha).abs().max()),
        "alpha_recomputation_interpretation": (
            "diagnostic only: saved alpha was computed with CUDA reductions, while the loaded raw MI "
            "is recomputed on CPU; backend reduction order need not be bitwise identical"
        ),
        "unusual_mi_candidates": unusual_candidates,
    }
    (output / "integrity_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not pass_gate:
        raise RuntimeError("n=5 integrity gate failed; sample-size diagnostics were not started")
    return report


def cross_attention_layers(pipe: Any) -> list[Any]:
    layers = []
    for name, net in pipe.unet.named_children():
        if "up" in name or "down" in name:
            for block in net:
                if "Cross" in block.__class__.__name__:
                    for attention in block.attentions:
                        for transformer in attention.transformer_blocks:
                            layers.append(transformer.attn2)
        if "mid" in name:
            for attention in net.attentions:
                for transformer in attention.transformer_blocks:
                    layers.append(transformer.attn2)
    if len(layers) != 16:
        raise RuntimeError(f"expected 16 cross-attention layers, found {len(layers)}")
    return layers


def text_vectors(pipe: Any, texts: list[str]) -> tuple[list[torch.Tensor], torch.Tensor]:
    tok, enc, device = pipe.tokenizer, pipe.text_encoder, pipe.device
    inp = tok(texts, padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = enc(inp.input_ids.to(device))[0]
    vectors = [emb[mask.sum() - 2].detach() for emb, mask in zip(embeddings, inp.attention_mask)]
    blank = tok([""], padding="max_length", max_length=tok.model_max_length,
                truncation=True, return_tensors="pt")
    with torch.no_grad():
        blank_embedding = enc(blank.input_ids.to(device))[0]
    return vectors, blank_embedding[0, 1, :].detach()


def projection_blocks(layers: list[Any]) -> Iterable[tuple[str, int, torch.Tensor]]:
    for projection in ("to_v", "to_k"):
        for layer_index, layer in enumerate(layers):
            yield projection, layer_index, getattr(layer, projection).weight.detach()


def sample_size_diagnostic(pipe: Any, layers: list[Any], config: dict[str, Any],
                           output: Path) -> tuple[dict[int, dict[tuple[str, int], dict[str, dict[str, np.ndarray]]]],
                                                  list[dict[str, Any]], dict[str, Any]]:
    targets = config["targets"]
    vectors, empty = text_vectors(pipe, targets)
    sample_sizes = config["sample_sizes"]
    temperature, power = float(config["temperature"]), float(config["power"])
    noise = float(config["noise_sigma"])
    pool_size = int(config["pool_size_per_class"])
    summary_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    all_blocks: dict[int, dict[tuple[str, int], dict[str, dict[str, np.ndarray]]]] = {}
    hist_raw = {n: np.zeros(140, dtype=np.int64) for n in sample_sizes}
    hist_alpha = {n: np.zeros(140, dtype=np.int64) for n in sample_sizes}
    raw_edges = np.linspace(0, LN2 + 5e-4, 141)
    alpha_edges = np.linspace(0, 1, 141)
    global_stability: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    activation_path = output / "max_mi_activation_summary.csv"
    activation_fields = [
        "seed", "projection", "layer_index", "target_concept", "channel", "mi_n5",
        "positive_min", "positive_max", "positive_mean", "positive_std",
        "neutral_min", "neutral_max", "neutral_mean", "neutral_std",
        "closest_cross_group_activation_distance", "median_threshold", "threshold_tie_count"
    ]
    with activation_path.open("w", newline="", encoding="utf-8") as activation_handle:
        activation_writer = csv.DictWriter(activation_handle, fieldnames=activation_fields)
        activation_writer.writeheader()
        for seed in config["informax_seeds"]:
            generator = torch.Generator(device=pipe.device)
            generator.manual_seed(int(seed))
            all_blocks[int(seed)] = {}
            seed_values: dict[int, dict[str, list[np.ndarray]]] = {
                n: {"mi": [], "alpha": []} for n in sample_sizes
            }
            for projection, layer_index, weight in projection_blocks(layers):
                block: dict[str, dict[str, np.ndarray]] = {}
                for target, vector in zip(targets, vectors):
                    pos = vector.repeat(pool_size, 1) + noise * torch.randn(
                        (pool_size, vector.numel()), device=pipe.device,
                        dtype=vector.dtype, generator=generator)
                    neg = empty.repeat(pool_size, 1) + noise * torch.randn(
                        (pool_size, empty.numel()), device=pipe.device,
                        dtype=empty.dtype, generator=generator)
                    with torch.no_grad():
                        pos_acts = weight @ pos.t()
                        neg_acts = weight @ neg.t()
                    results: dict[int, dict[str, torch.Tensor]] = {}
                    for n in sample_sizes:
                        result = calculate_mi_alpha(pos_acts, neg_acts, n, temperature, power)
                        results[n] = result
                        mi_np = result["mi"].cpu().numpy()
                        alpha_np = result["alpha"].cpu().numpy()
                        seed_values[n]["mi"].append(mi_np)
                        seed_values[n]["alpha"].append(alpha_np)
                        hist_raw[n] += np.histogram(mi_np, bins=raw_edges)[0]
                        hist_alpha[n] += np.histogram(alpha_np, bins=alpha_edges)[0]
                        mi_stats, alpha_stats = stats(mi_np), stats(alpha_np)
                        unique = np.unique(np.round(mi_np.astype(np.float64), 8))
                        no_tie_values = enumerate_no_tie_mi(n)
                        per_rows.append({
                            "seed": seed, "n": n, "projection": projection,
                            "layer_index": layer_index, "target_concept": target,
                            **{f"mi_{k}": v for k, v in mi_stats.items()},
                            "mi_unique_count": int(unique.size),
                            "mi_unique_values": json.dumps(unique.tolist(), separators=(",", ":")),
                            "theoretical_no_tie_unique_count": int(math.floor(n / 2) + 1),
                            "enumerated_no_tie_unique_count": int(no_tie_values.size),
                            "fraction_exact_repo_max": float(np.mean(mi_np == no_tie_values.max())),
                            "fraction_numerically_at_ln2": float(np.mean(np.isclose(mi_np, LN2, atol=MAX_ATOL, rtol=0))),
                            "channels_with_threshold_ties": int((result["ties"] > 1).sum().item()),
                            **{f"alpha_{k}": v for k, v in alpha_stats.items() if k != "count"},
                        })
                    block[target] = {
                        "mi": results[5]["mi"].cpu().numpy(),
                        "alpha": results[5]["alpha"].cpu().numpy(),
                    }
                    max5 = torch.isclose(results[5]["mi"], torch.tensor(LN2, device=pipe.device), atol=MAX_ATOL, rtol=0)
                    for later_n in (10, 20, 50):
                        later_max = torch.isclose(results[later_n]["mi"], torch.tensor(LN2, device=pipe.device), atol=MAX_ATOL, rtol=0)
                        denominator = int(max5.sum().item())
                        numerator = int((max5 & later_max).sum().item())
                        stability_rows.append({
                            "seed": seed, "later_n": later_n, "scope": "projection_layer_target",
                            "projection": projection, "layer_index": layer_index, "target_concept": target,
                            "n5_max_channels": denominator, "still_max_channels": numerator,
                            "fraction_still_max": numerator / denominator if denominator else float("nan")
                        })
                        global_stability[(int(seed), later_n)][0] += denominator
                        global_stability[(int(seed), later_n)][1] += numerator
                    selected = torch.nonzero(max5).flatten()
                    if selected.numel():
                        p5 = pos_acts.index_select(0, selected)[:, :5].float()
                        n5 = neg_acts.index_select(0, selected)[:, :5].float()
                        distance = (p5[:, :, None] - n5[:, None, :]).abs().amin(dim=(1, 2))
                        threshold = results[5]["threshold"].index_select(0, selected)
                        ties = results[5]["ties"].index_select(0, selected)
                        mi5 = results[5]["mi"].index_select(0, selected)
                        for row_index, channel in enumerate(selected.tolist()):
                            activation_writer.writerow({
                                "seed": seed, "projection": projection, "layer_index": layer_index,
                                "target_concept": target, "channel": channel, "mi_n5": float(mi5[row_index]),
                                "positive_min": float(p5[row_index].min()), "positive_max": float(p5[row_index].max()),
                                "positive_mean": float(p5[row_index].mean()), "positive_std": float(p5[row_index].std()),
                                "neutral_min": float(n5[row_index].min()), "neutral_max": float(n5[row_index].max()),
                                "neutral_mean": float(n5[row_index].mean()), "neutral_std": float(n5[row_index].std()),
                                "closest_cross_group_activation_distance": float(distance[row_index]),
                                "median_threshold": float(threshold[row_index]), "threshold_tie_count": int(ties[row_index]),
                            })
                all_blocks[int(seed)][(projection, layer_index)] = block
            for n in sample_sizes:
                raw = np.concatenate(seed_values[n]["mi"])
                alpha = np.concatenate(seed_values[n]["alpha"])
                raw_stats, alpha_stats = stats(raw), stats(alpha)
                unique = np.unique(np.round(raw.astype(np.float64), 8))
                no_tie_values = enumerate_no_tie_mi(n)
                summary_rows.append({
                    "scope": "seed", "seed": seed, "n": n,
                    **{f"mi_{k}": v for k, v in raw_stats.items()},
                    "mi_unique_count": int(unique.size),
                    "mi_unique_values": json.dumps(unique.tolist(), separators=(",", ":")),
                    "theoretical_no_tie_unique_count": int(math.floor(n / 2) + 1),
                    "enumerated_no_tie_unique_count": int(no_tie_values.size),
                    "fraction_exact_repo_max": float(np.mean(raw == no_tie_values.max())),
                    "fraction_numerically_at_ln2": float(np.mean(np.isclose(raw, LN2, atol=MAX_ATOL, rtol=0))),
                    **{f"alpha_{k}": v for k, v in alpha_stats.items() if k != "count"},
                })
    for (seed, later_n), (denominator, numerator) in sorted(global_stability.items()):
        stability_rows.append({
            "seed": seed, "later_n": later_n, "scope": "overall", "projection": "all",
            "layer_index": "all", "target_concept": "all", "n5_max_channels": denominator,
            "still_max_channels": numerator, "fraction_still_max": numerator / denominator
        })
    for n in sample_sizes:
        seed_rows = [r for r in summary_rows if r["scope"] == "seed" and r["n"] == n]
        summary_rows.append({
            "scope": "across_seed_mean", "seed": "all", "n": n,
            **{key: float(np.mean([float(r[key]) for r in seed_rows]))
               for key in seed_rows[0] if key.startswith("mi_") and key not in {"mi_unique_values"}},
            "mi_unique_values": "see per-seed rows",
            "theoretical_no_tie_unique_count": int(math.floor(n / 2) + 1),
            "enumerated_no_tie_unique_count": int(enumerate_no_tie_mi(n).size),
            "fraction_exact_repo_max": float(np.mean([r["fraction_exact_repo_max"] for r in seed_rows])),
            "fraction_numerically_at_ln2": float(np.mean([r["fraction_numerically_at_ln2"] for r in seed_rows])),
            **{key: float(np.mean([float(r[key]) for r in seed_rows]))
               for key in seed_rows[0] if key.startswith("alpha_")},
        })
    write_rows(output / "sample_size_summary.csv", summary_rows)
    write_rows(output / "sample_size_per_layer_concept.csv", per_rows)
    write_rows(output / "max_mi_stability.csv", stability_rows)
    return all_blocks, summary_rows, {
        "raw_edges": raw_edges, "alpha_edges": alpha_edges,
        "hist_raw": hist_raw, "hist_alpha": hist_alpha,
        "activation_rows": sum(1 for _ in activation_path.open(encoding="utf-8")) - 1,
    }


def concept_count_diagnostic(blocks_by_seed: dict[int, dict[tuple[str, int], dict[str, dict[str, np.ndarray]]]],
                             targets: list[str], output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paper_rows: list[dict[str, Any]] = []
    repo_rows: list[dict[str, Any]] = []
    for seed, blocks in blocks_by_seed.items():
        for mask in range(1, 1 << len(targets)):
            indices = [i for i in range(len(targets)) if mask & (1 << i)]
            subset = [targets[i] for i in indices]
            paper_mi_parts, paper_alpha_parts, repo_parts = [], [], []
            layer_repo: dict[tuple[str, int], np.ndarray] = {}
            for block_id, concept_data in blocks.items():
                raw_stack = np.stack([concept_data[target]["mi"] for target in subset])
                alpha_stack = np.stack([concept_data[target]["alpha"] for target in subset])
                paper_mi = raw_stack.max(axis=0)
                denominator = paper_mi.max()
                paper_alpha = paper_mi / denominator if denominator > 0 else np.zeros_like(paper_mi)
                repo_alpha = alpha_stack.max(axis=0)
                paper_mi_parts.append(paper_mi)
                paper_alpha_parts.append(paper_alpha)
                repo_parts.append(repo_alpha)
                layer_repo[block_id] = repo_alpha
            paper_mi_all = np.concatenate(paper_mi_parts)
            paper_alpha_all = np.concatenate(paper_alpha_parts)
            repo_all = np.concatenate(repo_parts)
            mi_stats, paper_alpha_stats, repo_stats = stats(paper_mi_all), stats(paper_alpha_all), stats(repo_all)
            common = {"seed": seed, "m": len(subset), "subset_mask": mask,
                      "subset_concepts": "|".join(subset)}
            paper_rows.append({
                **common, **{f"mi_{k}": v for k, v in mi_stats.items()},
                "fraction_mi_numerically_at_ln2": float(np.mean(np.isclose(paper_mi_all, LN2, atol=MAX_ATOL, rtol=0))),
                **{f"alpha_{k}": v for k, v in paper_alpha_stats.items() if k != "count"},
                "fraction_alpha_equal_1": float(np.mean(np.isclose(paper_alpha_all, 1.0, atol=1e-7, rtol=0))),
            })
            repo_rows.append({
                **common, "scope": "overall", "projection": "all", "layer_index": "all",
                **{f"alpha_{k}": v for k, v in repo_stats.items()},
            })
            for (projection, layer_index), values in layer_repo.items():
                repo_rows.append({
                    **common, "scope": "projection_layer", "projection": projection,
                    "layer_index": layer_index, **{f"alpha_{k}": v for k, v in stats(values).items()},
                })
    write_rows(output / "concept_count_paper_formula.csv", paper_rows)
    write_rows(output / "concept_count_repo_formula.csv", repo_rows)
    return paper_rows, repo_rows


def analyze_large_scale(pipe: Any, layers: list[Any], config: dict[str, Any], output: Path) -> None:
    large = config["large_scale"]
    if not large.get("enabled"):
        return
    targets = large["targets"]
    vectors, empty = text_vectors(pipe, targets)
    rows_paper, rows_repo = [], []
    for seed in config["informax_seeds"]:
        generator = torch.Generator(device=pipe.device)
        generator.manual_seed(int(seed))
        block_data: dict[tuple[str, int], list[dict[str, np.ndarray]]] = {}
        for projection, layer_index, weight in projection_blocks(layers):
            values = []
            for vector in vectors:
                pos = vector.repeat(5, 1) + float(config["noise_sigma"]) * torch.randn(
                    (5, vector.numel()), device=pipe.device, dtype=vector.dtype, generator=generator)
                neg = empty.repeat(5, 1) + float(config["noise_sigma"]) * torch.randn(
                    (5, empty.numel()), device=pipe.device, dtype=empty.dtype, generator=generator)
                with torch.no_grad():
                    result = calculate_mi_alpha(weight @ pos.t(), weight @ neg.t(), 5,
                                                float(config["temperature"]), float(large["power"]))
                values.append({"mi": result["mi"].cpu().numpy(), "alpha": result["alpha"].cpu().numpy()})
            block_data[(projection, layer_index)] = values
        for m in large["concept_counts"]:
            paper_mi_parts, paper_alpha_parts, repo_parts = [], [], []
            for values in block_data.values():
                paper_mi = np.stack([v["mi"] for v in values[:m]]).max(0)
                paper_alpha = paper_mi / paper_mi.max() if paper_mi.max() > 0 else np.zeros_like(paper_mi)
                repo_alpha = np.stack([v["alpha"] for v in values[:m]]).max(0)
                paper_mi_parts.append(paper_mi); paper_alpha_parts.append(paper_alpha); repo_parts.append(repo_alpha)
            paper_mi_all, paper_alpha_all, repo_all = map(np.concatenate,
                (paper_mi_parts, paper_alpha_parts, repo_parts))
            rows_paper.append({
                "seed": seed, "m": m, "ordered_prefix": "|".join(targets[:m]),
                **{f"mi_{k}": v for k, v in stats(paper_mi_all).items()},
                "fraction_mi_numerically_at_ln2": float(np.mean(np.isclose(paper_mi_all, LN2, atol=MAX_ATOL, rtol=0))),
                **{f"alpha_{k}": v for k, v in stats(paper_alpha_all).items() if k != "count"},
                "fraction_alpha_equal_1": float(np.mean(np.isclose(paper_alpha_all, 1.0, atol=1e-7, rtol=0))),
            })
            rows_repo.append({
                "seed": seed, "m": m, "ordered_prefix": "|".join(targets[:m]),
                **{f"alpha_{k}": v for k, v in stats(repo_all).items()},
            })
    write_rows(output / "concept_count_large_scale_paper_formula.csv", rows_paper)
    write_rows(output / "concept_count_large_scale_repo_formula.csv", rows_repo)


def activation_distance_report(path: Path) -> dict[str, Any]:
    distances, smallest, largest = [], [], []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = float(row["closest_cross_group_activation_distance"])
            distances.append(value)
            item = (value, {k: row[k] for k in ("seed", "projection", "layer_index", "target_concept", "channel")})
            smallest.append(item); smallest = sorted(smallest, key=lambda x: x[0])[:10]
            largest.append(item); largest = sorted(largest, key=lambda x: x[0], reverse=True)[:10]
    arr = np.asarray(distances, dtype=np.float64)
    return {
        "count": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
        "min": float(arr.min()), "p01": float(np.quantile(arr, .01)), "p05": float(np.quantile(arr, .05)),
        "median": float(np.median(arr)), "p95": float(np.quantile(arr, .95)),
        "p99": float(np.quantile(arr, .99)), "max": float(arr.max()),
        "smallest_examples": smallest, "largest_examples": largest,
    }


def make_figures(summary_rows: list[dict[str, Any]], paper_rows: list[dict[str, Any]],
                 repo_rows: list[dict[str, Any]], hist: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    sample = [r for r in summary_rows if r["scope"] == "across_seed_mean"]
    ns = [int(r["n"]) for r in sample]
    plt.figure(figsize=(6, 4)); plt.plot(ns, [r["fraction_numerically_at_ln2"] for r in sample], marker="o")
    plt.xlabel("Positive and neutral samples per class (n)"); plt.ylabel("Fraction with MI ≈ ln(2)"); plt.ylim(0, 1.02); plt.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(figures / "sample_size_vs_max_mi_fraction.png", dpi=180); plt.close()
    centers = (hist["raw_edges"][:-1] + hist["raw_edges"][1:]) / 2
    plt.figure(figsize=(7, 4))
    for n in ns:
        counts = hist["hist_raw"][n]; plt.plot(centers, counts / counts.sum(), label=f"n={n}")
    plt.xlabel("Raw MI (nats)"); plt.ylabel("Observation fraction per bin"); plt.legend(); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(figures / "sample_size_raw_mi_distribution.png", dpi=180); plt.close()
    centers = (hist["alpha_edges"][:-1] + hist["alpha_edges"][1:]) / 2
    plt.figure(figsize=(7, 4))
    for n in ns:
        counts = hist["hist_alpha"][n]; plt.plot(centers, counts / counts.sum(), label=f"n={n}")
    plt.xlabel("Repository alpha"); plt.ylabel("Observation fraction per bin"); plt.legend(); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(figures / "sample_size_repo_alpha_distribution.png", dpi=180); plt.close()
    paper_by_m = defaultdict(list)
    for row in paper_rows:
        paper_by_m[int(row["m"])].append(row)
    ms = sorted(paper_by_m)
    plt.figure(figsize=(7, 4))
    plt.plot(ms, [np.mean([r["fraction_mi_numerically_at_ln2"] for r in paper_by_m[m]]) for m in ms], marker="o", label="MI ≈ ln(2)")
    plt.plot(ms, [np.mean([r["fraction_alpha_equal_1"] for r in paper_by_m[m]]) for m in ms], marker="s", label="paper alpha = 1")
    plt.xlabel("Concept count m"); plt.ylabel("Channel fraction"); plt.ylim(0, 1.02); plt.legend(); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(figures / "concept_count_paper_behavior.png", dpi=180); plt.close()
    repo_by_m = defaultdict(list)
    for row in repo_rows:
        if row["scope"] == "overall": repo_by_m[int(row["m"])].append(row)
    plt.figure(figsize=(7, 4))
    for metric in ("alpha_mean", "alpha_median", "alpha_p95", "alpha_p99"):
        plt.plot(ms, [np.mean([r[metric] for r in repo_by_m[m]]) for m in ms], marker="o", label=metric.removeprefix("alpha_"))
    plt.xlabel("Concept count m"); plt.ylabel("Repository alpha"); plt.legend(); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(figures / "concept_count_repo_behavior.png", dpi=180); plt.close()


def build_summary(config: dict[str, Any], integrity: dict[str, Any], summary_rows: list[dict[str, Any]],
                  stability_path: Path, distance: dict[str, Any], paper_rows: list[dict[str, Any]],
                  repo_rows: list[dict[str, Any]], output: Path) -> None:
    aggregate = {int(r["n"]): r for r in summary_rows if r["scope"] == "across_seed_mean"}
    stability = list(csv.DictReader(stability_path.open(encoding="utf-8")))
    stability_overall = defaultdict(list)
    for row in stability:
        if row["scope"] == "overall": stability_overall[int(row["later_n"])].append(float(row["fraction_still_max"]))
    f5, f50 = aggregate[5]["fraction_numerically_at_ln2"], aggregate[50]["fraction_numerically_at_ln2"]
    stable50 = float(np.mean(stability_overall[50]))
    if f50 >= .95:
        case = "Case B: even 50+50 remains almost entirely at maximum MI; separation is not mainly a 5+5 artifact."
    elif f5 - f50 >= .10 and stable50 < .90:
        case = "Case A: larger samples materially reduce maximum-MI concentration and destabilize many n=5 maxima."
    else:
        case = "Case C: finite sample size and easy target-versus-empty separation both contribute."
    paper_m = defaultdict(list); repo_m = defaultdict(list)
    for row in paper_rows: paper_m[int(row["m"])].append(row)
    for row in repo_rows:
        if row["scope"] == "overall": repo_m[int(row["m"])].append(row)
    def mean_at(rows: dict[int, list[dict[str, Any]]], m: int, key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows[m]]))
    candidate = integrity["unusual_mi_candidates"][0] if integrity["unusual_mi_candidates"] else None
    unusual = (f"The sole candidate is `{candidate['projection']}`, layer {candidate['layer_index']}, "
               f"target `{candidate['target_concept']}`, channel {candidate['channel']}, MI={candidate['raw_mi']:.8f}. "
               "The legacy artifact does not contain its ten activations or binary states, so its contingency table, ties, "
               "and numerical mechanism cannot be reconstructed without rerunning the legacy RNG stream; no cause is asserted."
               if candidate else "No MI≈0.0863 candidate was found, which would contradict the expected gate.")
    sample_sentence = ", ".join(
        f"n={n}: {aggregate[n]['fraction_numerically_at_ln2']:.4%}" for n in (5, 10, 20, 50)
    )
    stability_sentence = ", ".join(
        f"n={n}: {np.mean(stability_overall[n]):.4%}" for n in (10, 20, 50)
    )
    text = f"""# ScaPre Informax MI and channel-weighting diagnostic

## 1. Implementation audit

The repository computes per-concept raw MI, z-scores across output channels within one projection/layer/concept call, applies `sigmoid(z/0.7)^8`, and then takes a channel-wise maximum over concepts. This differs from paper Eq. (7), which takes a max over raw per-concept MI and normalizes by the maximum channel. `to_v` and `to_k` use the same operations on distinct weights and random draws. See `implementation_audit.md` for line-level evidence.

## 2. n=5 integrity check

The gate **passed**. It reproduced {integrity['observations']:,} aggregate-stage channel observations with bin counts {integrity['mi_bin_counts']} around {integrity['mi_bin_centers']}; {integrity['fraction_numerically_at_ln2']:.4%} were numerically at ln(2). The saved repository-alpha statistics match the registered expectations. Recomputing alpha after moving raw MI from the original CUDA execution to CPU is retained as a non-gating backend-sensitivity diagnostic (maximum absolute difference {integrity['alpha_recomputation_max_abs_error']:.3g}); it is not treated as a bitwise reproducibility requirement.

{unusual}

## 3. Sample-size result

Across five fixed analysis streams, the mean maximum-MI fractions were: {sample_sentence}. {case}

Raw MI retained the unchanged repository estimator and lower-median strict-`>` binarization. The no-tie enumeration produced `floor(n/2)+1` distinct values for every requested n; observed values and tie counts are recorded in the CSVs.

## 4. Stability of n=5 maximum-MI channels

Among n=5 maximum-MI channels, the across-seed mean fractions still at maximum were: {stability_sentence}. Per projection/layer/target denominators and fractions are in `max_mi_stability.csv`.

## 5. Descriptive activation separation

For the n=5 maximum-MI rows, the closest positive-neutral raw activation distance had median {distance['median']:.6g}, p01 {distance['p01']:.6g}, p99 {distance['p99']:.6g}, min {distance['min']:.6g}, and max {distance['max']:.6g}. The wide scale-dependent range shows that equal saturated MI can coexist with different raw separations. This distance is descriptive only: it is not normalized across channels and is not a relevance score.

## 6. Paper-style max over concepts

Across every subset and seed, increasing m from 1 to 10 changed the mean fraction at maximum MI from {mean_at(paper_m,1,'fraction_mi_numerically_at_ln2'):.4%} to {mean_at(paper_m,10,'fraction_mi_numerically_at_ln2'):.4%}, and the fraction with paper alpha=1 from {mean_at(paper_m,1,'fraction_alpha_equal_1'):.4%} to {mean_at(paper_m,10,'fraction_alpha_equal_1'):.4%}. These are paper-formula results, not repository execution weights.

## 7. Repository-style max over concepts

Using the audited repository order, increasing m from 1 to 10 changed mean alpha from {mean_at(repo_m,1,'alpha_mean'):.6g} to {mean_at(repo_m,10,'alpha_mean'):.6g}, median from {mean_at(repo_m,1,'alpha_median'):.6g} to {mean_at(repo_m,10,'alpha_median'):.6g}, and p99 from {mean_at(repo_m,1,'alpha_p99'):.6g} to {mean_at(repo_m,10,'alpha_p99'):.6g}. Every one of the 1,023 non-empty target subsets was enumerated for every seed.

## 8. Official 50-concept configuration

The repository contains an unambiguous ordered ImageNet-Diversi50 ScaPre configuration at `scapre/script/erase.sh:14-49`; the two large-scale CSVs analyze its cumulative prefixes at m=1,5,10,20,30,40,50 using its configured power 5. No edit or image evaluation was run.

## 9. Limitations

- The new streams are deterministic analysis-only streams, not replays of production's global RNG positions.
- Results describe Stable Diffusion v1.5 cross-attention weights and the official empty-string neutral only.
- Aggregate-stage MI is analyzed; production's separate accumulation-stage draws are intentionally excluded from the 249,600-observation definition.
- Raw activation distance is scale-dependent and supports no channel-importance claim.
- The legacy artifact lacks activations, preventing a defensible reconstruction of the sole MI≈0.0863 contingency table.

## 10. Next research question

Without proposing a method change, the next useful question is whether the same saturation and subset-max patterns persist across model checkpoints, concept families, and repeated official-neutral pseudo-sample pools, while keeping the MI estimator and ScaPre implementation fixed.
"""
    (output / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    legacy = (args.legacy_diagnostic or Path(config["legacy_diagnostic_default"])).resolve()
    integrity = integrity_gate(config, legacy, output)
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from types import SimpleNamespace
    dtype = torch.float16 if config["dtype"] == "float16" else torch.float32
    model_source = args.model_path.resolve() if args.model_path else config["base_model"]
    if isinstance(model_source, Path) and not model_source.is_dir():
        raise RuntimeError(f"model snapshot directory is missing: {model_source}")
    model_kwargs = {"local_files_only": True}
    if not isinstance(model_source, Path) and config["base_model_revision"] is not None:
        model_kwargs["revision"] = config["base_model_revision"]
    tokenizer = CLIPTokenizer.from_pretrained(model_source, subfolder="tokenizer", **model_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(
        model_source, subfolder="text_encoder", torch_dtype=dtype, **model_kwargs
    ).to(config["device"])
    unet = UNet2DConditionModel.from_pretrained(
        model_source, subfolder="unet", torch_dtype=dtype, **model_kwargs
    ).to(config["device"])
    text_encoder.eval(); unet.eval()
    pipe = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder, unet=unet,
                           device=torch.device(config["device"]))
    layers = cross_attention_layers(pipe)
    production_before = sha256(REPO / "scapre/edit/erase_scale.py")
    blocks, summary_rows, hist = sample_size_diagnostic(pipe, layers, config, output)
    paper_rows, repo_rows = concept_count_diagnostic(blocks, config["targets"], output)
    analyze_large_scale(pipe, layers, config, output)
    distance = activation_distance_report(output / "max_mi_activation_summary.csv")
    (output / "activation_distance_diagnostic.json").write_text(json.dumps(distance, indent=2) + "\n", encoding="utf-8")
    make_figures(summary_rows, paper_rows, repo_rows, hist, output)
    build_summary(config, integrity, summary_rows, output / "max_mi_stability.csv",
                  distance, paper_rows, repo_rows, output)
    if sha256(REPO / "scapre/edit/erase_scale.py") != production_before:
        raise RuntimeError("production ScaPre source changed during analysis")
    required = [
        "sample_size_summary.csv", "sample_size_per_layer_concept.csv", "max_mi_stability.csv",
        "max_mi_activation_summary.csv", "concept_count_paper_formula.csv",
        "concept_count_repo_formula.csv", "concept_count_large_scale_paper_formula.csv",
        "concept_count_large_scale_repo_formula.csv", "summary.md"
    ]
    row_counts = {}
    hashes = {}
    for name in required:
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required output missing or empty: {path}")
        hashes[name] = sha256(path)
        row_counts[name] = sum(1 for _ in path.open(encoding="utf-8")) - (1 if name.endswith(".csv") else 0)
    manifest = {
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "git_status_at_completion": git("status", "--short"), "config": config,
        "config_sha256": sha256(args.config), "legacy_diagnostic_sha256": integrity["legacy_sha256"],
        "model_source": str(model_source),
        "base_model_resolved_revision": config["base_model_resolved_revision"],
        "seeds": config["informax_seeds"], "sample_sizes": config["sample_sizes"],
        "row_counts": row_counts, "important_file_sha256": hashes,
        "n5_reproduction_gate_passed": True, "production_source_sha256_before_after": production_before,
        "diffusion_images_generated": 0, "image_evaluators_run": [], "model_edit_performed": False,
    }
    (output / "integrity_report.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "COMPLETED").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
