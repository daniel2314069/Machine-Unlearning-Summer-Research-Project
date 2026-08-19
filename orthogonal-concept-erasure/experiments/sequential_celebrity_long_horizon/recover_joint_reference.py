#!/usr/bin/env python3
"""Integrity-safe recovery shim for the joint-100 reference cell.

The frozen runner incorrectly routes the evaluation-only joint control through
the sequential qualitative-slot helper.  This shim imports the exact frozen
runner, skips that helper only for ``order == "joint"``, and delegates every
other operation to the frozen implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


RUNNER = Path(__file__).with_name("run_sequential_long_horizon.py")
spec = importlib.util.spec_from_file_location("frozen_long_horizon_runner", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import frozen runner: {RUNNER}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

original_generate_qualitative = module.generate_missing_qualitative_seed43


def generate_qualitative_except_joint(protocol, output_dir, order, condition, step):
    if order == "joint":
        return None
    return original_generate_qualitative(protocol, output_dir, order, condition, step)


module.generate_missing_qualitative_seed43 = generate_qualitative_except_joint


def main() -> None:
    sys.argv[0] = str(RUNNER)
    args = module.parse_args()
    protocol = module.require_manifest(args, active=True)
    module.evaluate_formal_cell(args, protocol, "joint", "joint_100", 10)
    module.update_state(module.output_path(args), status="running", phase="aggregation")
    module.finalize(args, protocol)


if __name__ == "__main__":
    main()
