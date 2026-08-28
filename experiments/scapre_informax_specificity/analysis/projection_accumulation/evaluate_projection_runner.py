#!/usr/bin/env python3
"""Run the byte-unchanged Confuse5 evaluator with the projection label."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder or "--variant" in remainder:
        parser.error("wrapped evaluator args are missing or include --variant")
    script = args.script.resolve()
    if not script.is_file():
        raise RuntimeError(f"evaluator does not exist: {script}")

    original_parse_args = argparse.ArgumentParser.parse_args

    def controlled_parse_args(parser_self, args_override=None, namespace=None):
        parsed = original_parse_args(parser_self, args_override, namespace)
        if getattr(parsed, "variant", None) == "official":
            parsed.variant = "projection_accumulation"
        return parsed

    argparse.ArgumentParser.parse_args = controlled_parse_args
    sys.argv = [str(script), *remainder, "--variant", "official"]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
