#!/usr/bin/env python3
"""Experiment-only final-alpha control and Informax RNG isolation wrapper."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import runpy
import sys
from collections import Counter, namedtuple
from pathlib import Path

import torch


INFORMAX_CALLER = "_compute_mi_softmask_emptyneg"
MAX_RESULT = namedtuple("AlphaControlledMaxResult", "values indices")
VARIANTS = {
    "official", "constant_mean", "shuffled", "shuffled_alt1",
    "shuffled_alt2", "identity_B",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--informax-seed", type=int, required=True)
    parser.add_argument("--informax-rng-mode", choices=["legacy", "isolated"], required=True)
    parser.add_argument("--shuffle-salt", required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-randn-calls", type=int, required=True)
    parser.add_argument("--expected-alpha-intercepts", type=int, required=True)
    parser.add_argument("--layers-per-projection", type=int, required=True)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("wrapped editor arguments are missing")
    return args, remainder


def tensor_stats(value: torch.Tensor) -> dict[str, float | int | list[float]]:
    flat = value.detach().double().flatten().cpu()
    quantiles = torch.quantile(flat, torch.tensor([0.5, 0.95, 0.99], dtype=torch.double))
    return {
        "output_dimension": flat.numel(),
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=True).item() if flat.numel() > 1 else 0.0,
        "min": flat.min().item(),
        "max": flat.max().item(),
        "p50": quantiles[0].item(),
        "p95": quantiles[1].item(),
        "p99": quantiles[2].item(),
        "trace_B": flat.sum().item(),
        "frobenius_B": torch.linalg.vector_norm(flat).item(),
    }


def permutation_seed(salt: str, edit_seed: int, projection: str, layer: int) -> int:
    key = f"{salt}|{edit_seed}|{projection}|{layer}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def checkpoint_finiteness(path: Path) -> dict[str, object]:
    state = torch.load(path, map_location="cpu")
    checked = []
    invalid = []
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if name.endswith(".to_k.weight") or name.endswith(".to_v.weight"):
            checked.append(name)
            if not torch.isfinite(value).all().item():
                invalid.append(name)
    if not checked:
        raise RuntimeError("saved checkpoint contains no to_k/to_v weights")
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
        raise RuntimeError(f"editor script does not exist: {script}")
    if args.expected_alpha_intercepts != 2 * args.layers_per_projection:
        raise RuntimeError("expected alpha intercept count must equal two projections x layers")

    original_randn = torch.randn
    original_max = torch.max
    generators: dict[str, torch.Generator] = {}
    shape_counts: Counter[str] = Counter()
    randn_intercepted = 0
    alpha_intercepted = 0
    matrix_records: list[dict[str, object]] = []
    completed = False
    checkpoint_report: dict[str, object] | None = None

    def controlled_randn(*shape: object, **kwargs: object) -> torch.Tensor:
        nonlocal randn_intercepted
        caller = sys._getframe(1).f_code.co_name
        if caller != INFORMAX_CALLER:
            return original_randn(*shape, **kwargs)
        if args.informax_rng_mode == "legacy":
            result = original_randn(*shape, **kwargs)
        else:
            # Consume the legacy draw so later non-Informax global RNG state is
            # unchanged, then return the established isolated-seed draw.
            original_randn(*shape, **kwargs)
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
        randn_intercepted += 1
        shape_counts[f"{tuple(result.shape)}:{result.device}:{result.dtype}"] += 1
        return result

    def controlled_max(input_tensor: torch.Tensor, *positional: object, **kwargs: object):
        nonlocal alpha_intercepted
        result = original_max(input_tensor, *positional, **kwargs)
        caller = sys._getframe(1)
        dim = kwargs.get("dim", positional[0] if positional else None)
        candidate = (
            caller.f_code.co_name == "edit_model"
            and isinstance(input_tensor, torch.Tensor)
            and input_tensor.ndim == 3
            and dim == -1
            and hasattr(result, "values")
        )
        if not candidate:
            return result
        if alpha_intercepted >= args.expected_alpha_intercepts:
            raise RuntimeError("unexpected extra final-alpha max reduction")
        projection = "to_v" if alpha_intercepted < args.layers_per_projection else "to_k"
        layer = alpha_intercepted % args.layers_per_projection
        official = result.values
        if official.ndim != 2 or official.shape[1] != 1:
            raise RuntimeError(f"unexpected final-alpha shape: {tuple(official.shape)}")

        permutation = None
        if args.variant == "official":
            controlled = official
        elif args.variant == "constant_mean":
            controlled = torch.full_like(official, official.mean())
        elif args.variant.startswith("shuffled"):
            seed = permutation_seed(args.shuffle_salt, args.informax_seed, projection, layer)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            permutation = torch.randperm(official.numel(), generator=generator)
            controlled = official.flatten().index_select(0, permutation.to(official.device)).view_as(official)
        elif args.variant == "identity_B":
            controlled = torch.ones_like(official)
        else:
            raise AssertionError(args.variant)

        official_cpu = official.detach().cpu().flatten()
        controlled_cpu = controlled.detach().cpu().flatten()
        sorted_equal = torch.equal(*[torch.sort(item).values for item in (official_cpu, controlled_cpu)])
        mean_error = abs(controlled_cpu.double().mean().item() - official_cpu.double().mean().item())
        identity_exact = torch.equal(controlled_cpu, torch.ones_like(controlled_cpu))
        record: dict[str, object] = {
            "variant": args.variant,
            "edit_seed": args.informax_seed,
            "projection": projection,
            "layer_index": layer,
            "shuffle_salt": args.shuffle_salt if args.variant.startswith("shuffled") else None,
            "permutation_seed": permutation_seed(args.shuffle_salt, args.informax_seed, projection, layer)
                if args.variant.startswith("shuffled") else None,
            "official": tensor_stats(official),
            "controlled": tensor_stats(controlled),
            "constant_mean_abs_error": mean_error,
            "exact_multiset_preserved": sorted_equal,
            "identity_exact_all_ones": identity_exact,
            "raw_concept_alpha_input_shape": list(input_tensor.shape),
        }
        if permutation is not None:
            record["permutation_is_bijection"] = torch.equal(
                torch.sort(permutation).values, torch.arange(permutation.numel())
            )
        matrix_records.append(record)
        alpha_intercepted += 1
        return MAX_RESULT(controlled, result.indices)

    def write_audit() -> None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "variant": args.variant,
            "informax_seed": args.informax_seed,
            "informax_rng_mode": args.informax_rng_mode,
            "shuffle_salt": args.shuffle_salt,
            "completed": completed,
            "informax_randn_calls": randn_intercepted,
            "expected_informax_randn_calls": args.expected_randn_calls,
            "informax_randn_shape_counts": dict(sorted(shape_counts.items())),
            "global_rng_legacy_draws_consumed": randn_intercepted,
            "alpha_intercepts": alpha_intercepted,
            "expected_alpha_intercepts": args.expected_alpha_intercepts,
            "layers_per_projection": args.layers_per_projection,
            "matrix_records": matrix_records,
            "checkpoint_finiteness": checkpoint_report,
            "production_source_modified": False,
            "separate_permutation_rng": True,
        }
        args.audit_output.write_text(json.dumps(payload, indent=2) + "\n")

    atexit.register(write_audit)
    torch.randn = controlled_randn
    torch.max = controlled_max
    sys.argv = [str(script), *editor_args]
    runpy.run_path(str(script), run_name="__main__")
    if randn_intercepted != args.expected_randn_calls:
        raise RuntimeError(
            f"Informax RNG calls {randn_intercepted} != {args.expected_randn_calls}"
        )
    if alpha_intercepted != args.expected_alpha_intercepts:
        raise RuntimeError(
            f"final-alpha intercepts {alpha_intercepted} != {args.expected_alpha_intercepts}"
        )
    checkpoint_report = checkpoint_finiteness(output_checkpoint(editor_args))
    if not checkpoint_report["all_projection_weights_finite"]:
        raise RuntimeError("edited checkpoint contains NaN/Inf projection weights")
    if args.variant == "constant_mean" and any(
        float(row["constant_mean_abs_error"]) > 1e-7 for row in matrix_records
    ):
        raise RuntimeError("constant_mean failed its per-matrix mean gate")
    if args.variant.startswith("shuffled") and any(
        not row["exact_multiset_preserved"] or not row.get("permutation_is_bijection")
        for row in matrix_records
    ):
        raise RuntimeError("shuffled alpha failed exact multiset/bijection gate")
    if args.variant == "identity_B" and any(
        not row["identity_exact_all_ones"] for row in matrix_records
    ):
        raise RuntimeError("identity_B is not exactly all ones")
    completed = True


if __name__ == "__main__":
    main()
