#!/usr/bin/env python
"""Dependency-free formal input audit run before launching expensive GPU work."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from worker import (
    BASE_CONFIG_PATH,
    PROTOCOL_BUILDER,
    REPO_ROOT,
    validate_configuration,
    validate_prior_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    base_config = json.loads(BASE_CONFIG_PATH.read_text())
    assets = json.loads(args.assets.read_text())
    validate_configuration(config, base_config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="formal_preflight_", dir=args.output.parent
    ) as temporary:
        protocol_path = Path(temporary) / "protocol.csv"
        protocol_output = subprocess.check_output(
            [
                sys.executable,
                str(PROTOCOL_BUILDER),
                "--config",
                str(BASE_CONFIG_PATH),
                "--output",
                str(protocol_path),
                "--profile",
                "formal",
            ],
            cwd=REPO_ROOT,
            text=True,
        )
        protocol_manifest = json.loads(protocol_output)
        if protocol_manifest["sha256"] != config["prior_seed"]["protocol_sha256"]:
            raise RuntimeError("current formal protocol differs from the pinned legacy protocol")
        prior_validation = validate_prior_seed(
            args.prior_run.resolve(), config, assets, protocol_path
        )

    payload = {
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "prior_run": str(args.prior_run.resolve()),
        "protocol": protocol_manifest,
        "prior_validation": prior_validation,
        "loaded_model": False,
        "generated_images": False,
        "downloaded_assets": False,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("Formal preflight passed without loading a model or generating images.")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
