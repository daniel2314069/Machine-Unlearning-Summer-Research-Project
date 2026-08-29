#!/usr/bin/env python3
"""Run the unchanged ScaPre editor with an accumulation-only projection mask.

The official Informax helper always runs first.  This wrapper replaces only the
``for_mat1 * row_w_c`` multiplication during the accumulation stage; aggregate
Informax and ``row_w_max`` are never intercepted.  It also preserves the
established five-seed Informax RNG semantics and records enough tensors to
compare every non-treatment input against an official edit.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


INFORMAX_CALLER = "_compute_mi_softmask_emptyneg"
VARIANTS = {
    "official",
    "projection_accumulation",
    "projection_accumulation_direct_cos2",
}
ALPHA_MODES = {"zscore_sigmoid_power", "direct_cos2"}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--alpha-mode", choices=sorted(ALPHA_MODES), required=True)
    parser.add_argument("--informax-seed", type=int, required=True)
    parser.add_argument("--informax-rng-mode", choices=["legacy", "isolated"], required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--expected-informax-randn-calls", type=int, required=True)
    parser.add_argument("--expected-accumulation-intercepts", type=int, required=True)
    parser.add_argument("--expected-matrix-records", type=int, required=True)
    parser.add_argument("--targets-per-matrix", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--power", type=float, required=True)
    parser.add_argument("--eps", type=float, required=True)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("wrapped editor arguments are missing")
    return args, remainder


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    header = f"{array.dtype}|{array.shape}|".encode("utf-8")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def rng_state_sha256() -> dict[str, Any]:
    cpu = tensor_sha256(torch.random.get_rng_state())
    cuda = []
    if torch.cuda.is_available():
        cuda = [tensor_sha256(state) for state in torch.cuda.get_rng_state_all()]
    return {"cpu": cpu, "cuda": cuda}


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().double().flatten().cpu()
    quantiles = torch.quantile(
        flat, torch.tensor([0.01, 0.5, 0.95, 0.99], dtype=torch.double)
    )
    return {
        "count": flat.numel(),
        "finite": bool(torch.isfinite(flat).all().item()),
        "non_constant": bool(flat.numel() > 1 and flat.max().item() != flat.min().item()),
        "min": flat.min().item(),
        "p01": quantiles[0].item(),
        "median": quantiles[1].item(),
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=True).item() if flat.numel() > 1 else 0.0,
        "frobenius": float(torch.linalg.vector_norm(flat).item()),
        "p95": quantiles[2].item(),
        "p99": quantiles[3].item(),
        "max": flat.max().item(),
        "sha256": tensor_sha256(value),
    }


def contribution_stats(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().double().flatten().cpu()
    return {
        "count": flat.numel(),
        "finite": bool(torch.isfinite(flat).all().item()),
        "frobenius": float(torch.linalg.vector_norm(flat).item()),
        "l1": float(flat.abs().sum().item()),
        "max_abs": float(flat.abs().max().item()),
        "sha256": tensor_sha256(value),
    }


def average_ranks(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().double().flatten().cpu()
    order = torch.argsort(flat, stable=True)
    sorted_value = flat.index_select(0, order)
    ranks = torch.empty_like(flat)
    start = 0
    while start < flat.numel():
        end = start + 1
        while end < flat.numel() and sorted_value[end].item() == sorted_value[start].item():
            end += 1
        ranks.index_fill_(0, order[start:end], (start + end - 1) / 2.0)
        start = end
    return ranks


def pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    x = left.detach().double().flatten().cpu()
    y = right.detach().double().flatten().cpu()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denominator.item() == 0.0:
        return None
    return float(torch.dot(x, y).item() / denominator.item())


def checkpoint_finiteness(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    checked: list[str] = []
    invalid: list[str] = []
    for name, value in state.items():
        if isinstance(value, torch.Tensor) and (
            name.endswith(".to_k.weight") or name.endswith(".to_v.weight")
        ):
            checked.append(name)
            if not torch.isfinite(value).all().item():
                invalid.append(name)
    if not checked:
        raise RuntimeError("checkpoint contains no to_k/to_v projection weights")
    return {
        "checked_projection_weights": len(checked),
        "invalid_projection_weights": invalid,
        "all_projection_weights_finite": not invalid,
    }


def output_checkpoint(editor_args: list[str]) -> Path:
    try:
        index = editor_args.index("--output_model")
        return Path(editor_args[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise RuntimeError("wrapped editor command lacks --output_model") from error


def main() -> None:
    args, editor_args = parse_args()
    script = args.script.resolve()
    if not script.is_file():
        raise RuntimeError(f"editor does not exist: {script}")
    if args.temperature != 0.7 or args.power != 8.0 or args.eps != 1e-8:
        raise RuntimeError("diagnostic V1 transform is frozen at T=0.7, p=8, eps=1e-8")
    if args.expected_accumulation_intercepts != (
        args.expected_matrix_records * args.targets_per_matrix
    ):
        raise RuntimeError("accumulation/matrix/target counts are inconsistent")

    original_randn = torch.randn
    original_cholesky = torch.linalg.cholesky
    generators: dict[str, torch.Generator] = {}
    randn_shape_counts: Counter[str] = Counter()
    randn_events: list[dict[str, Any]] = []
    accumulation_records: list[dict[str, Any]] = []
    matrix_records: list[dict[str, Any]] = []
    completed = False
    checkpoint_report: dict[str, Any] | None = None
    executed_source_sha256: str | None = None
    production_source_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
    initial_rng = rng_state_sha256()

    def controlled_randn(*shape: object, **kwargs: object) -> torch.Tensor:
        caller = sys._getframe(1).f_code.co_name
        if caller != INFORMAX_CALLER:
            return original_randn(*shape, **kwargs)
        before = rng_state_sha256()
        discarded_sha = None
        if args.informax_rng_mode == "legacy":
            result = original_randn(*shape, **kwargs)
        else:
            discarded = original_randn(*shape, **kwargs)
            discarded_sha = tensor_sha256(discarded)
            device = torch.device(kwargs.get("device", "cpu"))
            key = str(device)
            if key not in generators:
                generators[key] = torch.Generator(device=device)
                generators[key].manual_seed(args.informax_seed)
            actual = dict(kwargs)
            if "generator" in actual:
                raise RuntimeError("Informax randn unexpectedly supplied a generator")
            actual["generator"] = generators[key]
            result = original_randn(*shape, **actual)
        after = rng_state_sha256()
        signature = f"{tuple(result.shape)}:{result.device}:{result.dtype}"
        randn_shape_counts[signature] += 1
        randn_events.append({
            "index": len(randn_events),
            "signature": signature,
            "global_state_before": before,
            "global_state_after": after,
            "discarded_legacy_sha256": discarded_sha,
            "returned_sha256": tensor_sha256(result),
        })
        return result

    def accumulation_hook(
        for_mat1: torch.Tensor,
        row_w_c: torch.Tensor,
        W_old: torch.Tensor,
        c_vec: torch.Tensor,
        empty_vec: torch.Tensor,
        for_mat2: torch.Tensor,
        target_concept: str,
        layer_num: int,
        target_number: int,
        projection: str,
    ) -> torch.Tensor:
        if len(accumulation_records) >= args.expected_accumulation_intercepts:
            raise RuntimeError("unexpected extra accumulation mask multiplication")

        d_vec = c_vec - empty_vec
        W_score = W_old.detach().to(dtype=torch.float32)
        d_score = d_vec.detach().to(dtype=torch.float32)
        numerator = torch.matmul(W_score, d_score).square()
        denominator = (
            W_score.square().sum(dim=1) + args.eps
        ) * (d_score.square().sum() + args.eps)
        score = numerator / denominator
        z_score = (score - score.mean()) / (score.std() + args.eps)
        projection_alpha = torch.sigmoid(z_score / args.temperature).pow(args.power).view(-1, 1)
        projection_alpha = projection_alpha.to(device=row_w_c.device, dtype=row_w_c.dtype)
        direct_alpha = score.view(-1, 1).to(device=row_w_c.device, dtype=row_w_c.dtype)

        if (
            not torch.isfinite(score).all().item()
            or not torch.isfinite(projection_alpha).all().item()
            or not torch.isfinite(direct_alpha).all().item()
        ):
            raise RuntimeError("projection score or alpha contains NaN/Inf")
        if (
            score.max().item() == score.min().item()
            or projection_alpha.max().item() == projection_alpha.min().item()
            or direct_alpha.max().item() == direct_alpha.min().item()
        ):
            raise RuntimeError("projection score or alpha is constant")

        index = len(accumulation_records)
        target_index = target_number - 1
        if projection not in {"to_v", "to_k"}:
            raise RuntimeError(f"unexpected accumulation projection: {projection}")
        if target_index != index % args.targets_per_matrix:
            raise RuntimeError("accumulation target ordering changed")
        official_cpu = row_w_c.detach().cpu()
        score_cpu = score.detach().cpu()
        alpha_cpu = projection_alpha.detach().cpu()
        record = {
            "index": index,
            "projection": projection,
            "layer_index": layer_num - 1,
            "target_index": target_index,
            "target_concept": target_concept,
            "official_row_w_c": official_cpu,
            "projection_score": score_cpu,
            "projection_alpha": alpha_cpu,
            "direct_cos2_alpha": direct_alpha.detach().cpu(),
            "official_stats": tensor_stats(official_cpu),
            "score_stats": tensor_stats(score_cpu),
            "projection_alpha_stats": tensor_stats(alpha_cpu),
            "direct_cos2_alpha_stats": tensor_stats(direct_alpha),
            "weighted_contribution_stats": {
                "official": contribution_stats(for_mat1 * row_w_c),
                "v1_projection": contribution_stats(for_mat1 * projection_alpha),
                "direct_cos2": contribution_stats(for_mat1 * direct_alpha),
            },
            "pearson": pearson(official_cpu, alpha_cpu),
            "spearman": pearson(average_ranks(official_cpu), average_ranks(alpha_cpu)),
            "official_vs_direct_pearson": pearson(official_cpu, direct_alpha),
            "official_vs_direct_spearman": pearson(
                average_ranks(official_cpu), average_ranks(direct_alpha)
            ),
            "W_old_sha256": tensor_sha256(W_old),
            "c_vec_sha256": tensor_sha256(c_vec),
            "empty_vec_sha256": tensor_sha256(empty_vec),
            "d_vec_sha256": tensor_sha256(d_vec),
            "for_mat1_sha256": tensor_sha256(for_mat1),
            "for_mat2_sha256": tensor_sha256(for_mat2),
            "rng_state_after_official_accumulation_mi": rng_state_sha256(),
        }
        accumulation_records.append(record)
        treatment_alpha = (
            projection_alpha
            if args.alpha_mode == "zscore_sigmoid_power"
            else direct_alpha
        )
        selected = row_w_c if args.variant == "official" else treatment_alpha
        return for_mat1 * selected

    def controlled_cholesky(input_tensor: torch.Tensor, *positional: object, **kwargs: object):
        frame = sys._getframe(1)
        candidate = (
            frame.f_code.co_name == "edit_model"
            and frame.f_locals.get("i") == 0
            and frame.f_locals.get("M_i") is input_tensor
        )
        if candidate:
            if len(matrix_records) >= args.expected_matrix_records:
                raise RuntimeError("unexpected extra edited matrix")
            matrix_index = len(matrix_records)
            projection = "to_v" if matrix_index < args.expected_matrix_records // 2 else "to_k"
            layer_index = matrix_index % (args.expected_matrix_records // 2)
            names = (
                "W_old", "C_full", "CCt", "PiC", "e_i", "H_row", "R_ased",
                "S", "G_base", "row_w_max", "mat2_agg", "M_i",
            )
            hashes = {
                name: tensor_sha256(frame.f_locals[name])
                for name in names
                if isinstance(frame.f_locals.get(name), torch.Tensor)
            }
            matrix_records.append({
                "matrix_index": matrix_index,
                "projection": projection,
                "layer_index": layer_index,
                "tensor_sha256": hashes,
                "row_w_max": frame.f_locals["row_w_max"].detach().cpu(),
                "row_w_max_stats": tensor_stats(frame.f_locals["row_w_max"]),
                "V_stats": tensor_stats(frame.f_locals["V"]),
                "mat1_agg_stats": tensor_stats(frame.f_locals["mat1_agg"]),
                "mu": float(frame.f_locals["mu"]),
                "rng_state_before_solve": rng_state_sha256(),
            })
        return original_cholesky(input_tensor, *positional, **kwargs)

    def write_outputs() -> None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        event_digest = hashlib.sha256(
            json.dumps(randn_events, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload = {
            "format_version": 1,
            "variant": args.variant,
            "informax_seed": args.informax_seed,
            "informax_rng_mode": args.informax_rng_mode,
            "formula": "((W @ (c-empty))**2) / ((row_norm_sq+eps)*(d_norm_sq+eps))",
            "alpha_mode": args.alpha_mode,
            "selected_treatment_alpha": (
                "projection_score" if args.alpha_mode == "direct_cos2"
                else "sigmoid(zscore(projection_score)/0.7)**8"
            ),
            "score_dtype": "float32",
            "temperature": args.temperature,
            "power": args.power,
            "eps": args.eps,
            "torch_std_sample_semantics": True,
            "completed": completed,
            "informax_randn_calls": len(randn_events),
            "expected_informax_randn_calls": args.expected_informax_randn_calls,
            "informax_randn_shape_counts": dict(sorted(randn_shape_counts.items())),
            "informax_event_stream_sha256": event_digest,
            "accumulation_intercepts": len(accumulation_records),
            "expected_accumulation_intercepts": args.expected_accumulation_intercepts,
            "matrix_records": len(matrix_records),
            "expected_matrix_records": args.expected_matrix_records,
            "process_startup_rng_state_descriptive_only": initial_rng,
            "first_informax_pre_draw_global_rng_state": (
                randn_events[0]["global_state_before"] if randn_events else None
            ),
            "final_global_rng_state": rng_state_sha256(),
            "checkpoint_finiteness": checkpoint_report,
            "production_source_modified": False,
            "official_mi_executed_before_every_intercept": True,
            "aggregate_row_w_max_intercepted": False,
            "in_memory_source_substitution_count": 2,
            "in_memory_source_substitution_scope": "for_mat1 * row_w_c only",
            "production_source_sha256": production_source_sha256,
            "executed_in_memory_source_sha256": executed_source_sha256,
        }
        args.audit_output.write_text(json.dumps(payload, indent=2) + "\n")
        torch.save({
            "format_version": 1,
            "variant": args.variant,
            "accumulation_records": accumulation_records,
            "matrix_records": matrix_records,
            "informax_randn_events": randn_events,
        }, args.diagnostics_output)

    atexit.register(write_outputs)
    torch.randn = controlled_randn
    torch.linalg.cholesky = controlled_cholesky
    sys.argv = [str(script), *editor_args]
    source = script.read_text()
    needle = "for_mat1 * row_w_c"
    if source.count(needle) != 2:
        raise RuntimeError("production accumulation source sites changed")
    replacement_v = (
        "_projection_accumulation_hook(for_mat1, row_w_c, W_old, c_vec, empty_vec, "
        "for_mat2, ot, layer_num, idx_concept, 'to_v')"
    )
    replacement_k = (
        "_projection_accumulation_hook(for_mat1, row_w_c, W_old, c_vec, empty_vec, "
        "for_mat2, ot, layer_num, idx_concept, 'to_k')"
    )
    source = source.replace(needle, replacement_v, 1)
    source = source.replace(needle, replacement_k, 1)
    if needle in source:
        raise RuntimeError("unexpected accumulation source site remained")
    executed_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    namespace = {
        "__name__": "__main__", "__file__": str(script), "__package__": None,
        "__cached__": None, "_projection_accumulation_hook": accumulation_hook,
    }
    exec(compile(source, str(script), "exec"), namespace)

    if len(randn_events) != args.expected_informax_randn_calls:
        raise RuntimeError("Informax randn call count changed")
    if len(accumulation_records) != args.expected_accumulation_intercepts:
        raise RuntimeError("accumulation intercept count changed")
    if len(matrix_records) != args.expected_matrix_records:
        raise RuntimeError("matrix audit count changed")
    checkpoint_report = checkpoint_finiteness(output_checkpoint(editor_args))
    if not checkpoint_report["all_projection_weights_finite"]:
        raise RuntimeError("edited checkpoint contains NaN/Inf")
    completed = True


if __name__ == "__main__":
    main()
