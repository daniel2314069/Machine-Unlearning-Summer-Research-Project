#!/usr/bin/env python3
"""Run the unchanged Confuse5 evaluator with experiment-only variant labels."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


VARIANTS = {
    "constant_mean", "shuffled", "shuffled_alt1", "shuffled_alt2", "identity_B"
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--script", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("wrapped evaluator arguments are missing")
    return args, remainder


def main() -> None:
    args, evaluator_args = parse_args()
    script = args.script.resolve()
    if not script.is_file():
        raise RuntimeError(f"evaluator does not exist: {script}")
    if "--variant" in evaluator_args:
        raise RuntimeError("variant must be supplied only to this wrapper")

    # Pass an allowed label through the production parser, then replace only
    # the parsed label. All generation, classifier, resume, row, and fingerprint
    # code runs byte-unchanged and directly writes the intended control label.
    original_parse_args = argparse.ArgumentParser.parse_args

    def controlled_parse_args(parser_self, args_override=None, namespace=None):
        parsed = original_parse_args(parser_self, args_override, namespace)
        if getattr(parsed, "variant", None) == "official":
            parsed.variant = args.variant
        return parsed

    argparse.ArgumentParser.parse_args = controlled_parse_args
    sys.argv = [str(script), *evaluator_args, "--variant", "official"]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
