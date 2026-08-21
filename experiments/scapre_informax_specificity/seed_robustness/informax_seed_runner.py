#!/usr/bin/env python3
"""Run the unchanged ScaPre editor while varying only Informax noise draws.

The legacy editor uses the global Torch RNG for both entropy sampling and
Informax pseudo-samples.  This wrapper keeps that global stream on the legacy
20260820 path.  For every Informax ``torch.randn`` call it first consumes and
discards the legacy draw (so all later non-Informax draws remain unchanged),
then returns a draw from an Informax-only generator seeded by ``--informax-seed``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import runpy
import sys
from collections import Counter
from pathlib import Path

import torch


INFORMAX_CALLERS = {
    "_compute_mi_softmask_emptyneg",
    "_compute_mi_softmask_matchedneg",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--informax-seed", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-randn-calls", type=int, required=True)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("the wrapped editor arguments are missing")
    return args, remainder


def main() -> None:
    args, editor_args = parse_args()
    script = args.script.resolve()
    if not script.is_file():
        raise RuntimeError(f"editor script does not exist: {script}")

    original_randn = torch.randn
    generators: dict[str, torch.Generator] = {}
    shape_counts: Counter[str] = Counter()
    intercepted = 0
    completed = False

    def controlled_randn(*shape: object, **kwargs: object) -> torch.Tensor:
        nonlocal intercepted
        caller = sys._getframe(1).f_code.co_name
        if caller not in INFORMAX_CALLERS:
            return original_randn(*shape, **kwargs)

        # Preserve the legacy global RNG position for all later non-Informax
        # operations.  The returned tensor is intentionally discarded.
        original_randn(*shape, **kwargs)

        device = torch.device(kwargs.get("device", "cpu"))
        generator_key = str(device)
        if generator_key not in generators:
            generators[generator_key] = torch.Generator(device=device)
            generators[generator_key].manual_seed(args.informax_seed)

        actual_kwargs = dict(kwargs)
        if "generator" in actual_kwargs:
            raise RuntimeError("wrapped Informax randn unexpectedly supplied a generator")
        actual_kwargs["generator"] = generators[generator_key]
        result = original_randn(*shape, **actual_kwargs)
        intercepted += 1
        shape_counts[f"{caller}:{tuple(result.shape)}:{result.device}:{result.dtype}"] += 1
        return result

    def write_audit() -> None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "informax_seed": args.informax_seed,
            "completed": completed,
            "intercepted_randn_calls": intercepted,
            "expected_randn_calls": args.expected_randn_calls,
            "shape_counts": dict(sorted(shape_counts.items())),
            "global_rng_legacy_draws_consumed": intercepted,
            "method_source_modified": False,
        }
        args.audit_output.write_text(json.dumps(payload, indent=2) + "\n")

    atexit.register(write_audit)
    torch.randn = controlled_randn
    sys.argv = [str(script), *editor_args]
    runpy.run_path(str(script), run_name="__main__")
    if intercepted != args.expected_randn_calls:
        raise RuntimeError(
            f"Informax RNG interception count was {intercepted}; "
            f"expected {args.expected_randn_calls}"
        )
    completed = True


if __name__ == "__main__":
    main()
