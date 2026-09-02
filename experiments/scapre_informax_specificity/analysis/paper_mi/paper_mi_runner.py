#!/usr/bin/env python3
"""Run paired ScaPre weighting modes with the established Informax RNG protocol."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import runpy
import sys
from collections import Counter
from pathlib import Path

import torch


INFORMAX_CALLERS = {
    "_compute_mi_softmask_emptyneg",
    "_consume_removed_accumulation_informax_rng",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighting-mode", choices=["repository", "paper"], required=True)
    parser.add_argument("--informax-seed", type=int, required=True)
    parser.add_argument("--informax-rng-mode", choices=["legacy", "isolated"], required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-randn-calls", type=int, required=True)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("wrapped editor arguments are missing")
    return args, remainder


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_finiteness(path: Path) -> dict[str, object]:
    state = torch.load(path, map_location="cpu")
    checked, invalid = [], []
    for name, value in state.items():
        if isinstance(value, torch.Tensor) and (
            name.endswith(".to_k.weight") or name.endswith(".to_v.weight")
        ):
            checked.append(name)
            if not torch.isfinite(value).all().item():
                invalid.append(name)
    if not checked:
        raise RuntimeError("checkpoint contains no edited projection weights")
    return {
        "checked_projection_weights": len(checked),
        "invalid_projection_weights": invalid,
        "all_projection_weights_finite": not invalid,
    }


def main() -> None:
    args, editor_args = parse_args()
    script = args.script.resolve()
    if not script.is_file():
        raise RuntimeError(f"editor does not exist: {script}")
    if "--informax-weighting-mode" not in editor_args:
        raise RuntimeError("editor command lacks --informax-weighting-mode")
    mode_index = editor_args.index("--informax-weighting-mode")
    if editor_args[mode_index + 1] != args.weighting_mode:
        raise RuntimeError("runner/editor Informax weighting modes disagree")
    output_index = editor_args.index("--output_model")
    checkpoint = Path(editor_args[output_index + 1]).resolve()

    original_randn = torch.randn
    generators: dict[str, torch.Generator] = {}
    shape_counts: Counter[str] = Counter()
    caller_counts: Counter[str] = Counter()
    intercepted = 0
    completed = False
    checkpoint_report: dict[str, object] | None = None

    def controlled_randn(*shape: object, **kwargs: object) -> torch.Tensor:
        nonlocal intercepted
        caller = sys._getframe(1).f_code.co_name
        if caller not in INFORMAX_CALLERS:
            return original_randn(*shape, **kwargs)
        if args.informax_rng_mode == "legacy":
            result = original_randn(*shape, **kwargs)
        else:
            # Consume the legacy draw so entropy and every later global RNG
            # position remain identical to the repository five-seed protocol.
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
        intercepted += 1
        caller_counts[caller] += 1
        shape_counts[f"{caller}:{tuple(result.shape)}:{result.device}:{result.dtype}"] += 1
        return result

    def write_audit() -> None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps({
            "informax_seed": args.informax_seed,
            "informax_rng_mode": args.informax_rng_mode,
            "completed": completed,
            "intercepted_randn_calls": intercepted,
            "expected_randn_calls": args.expected_randn_calls,
            "shape_counts": dict(sorted(shape_counts.items())),
            "caller_counts": dict(sorted(caller_counts.items())),
            "informax_weighting_mode": args.weighting_mode,
            "paper_aggregate_calls_only": args.weighting_mode == "paper",
            "production_editor_sha256": sha256(script),
            "checkpoint_finiteness": checkpoint_report,
        }, indent=2) + "\n")

    atexit.register(write_audit)
    torch.randn = controlled_randn
    sys.argv = [str(script), *editor_args]
    runpy.run_path(str(script), run_name="__main__")
    if intercepted != args.expected_randn_calls:
        raise RuntimeError(
            f"Informax RNG calls were {intercepted}; expected {args.expected_randn_calls}"
        )
    checkpoint_report = checkpoint_finiteness(checkpoint)
    if not checkpoint_report["all_projection_weights_finite"]:
        raise RuntimeError("checkpoint contains non-finite projection weights")
    completed = True


if __name__ == "__main__":
    main()
