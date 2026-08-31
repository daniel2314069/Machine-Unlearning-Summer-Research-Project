#!/usr/bin/env python3
"""Verify a completed storage inventory without jq or third-party packages."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} COMPLETED_RUN_DIRECTORY", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve(strict=True)
    manifest_path = run_dir / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise RuntimeError(f"result manifest status is not passed: {manifest.get('status')}")

    failures: list[str] = []
    for relative, expected in manifest.get("files", {}).items():
        path = run_dir / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual_size = path.stat().st_size
        actual_sha256 = sha256(path)
        if actual_size != int(expected["size_bytes"]):
            failures.append(
                f"size mismatch: {relative}: expected {expected['size_bytes']}, got {actual_size}"
            )
        if actual_sha256 != expected["sha256"]:
            failures.append(
                f"sha256 mismatch: {relative}: expected {expected['sha256']}, got {actual_sha256}"
            )
    if failures:
        print("Inventory result verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"Inventory result verification passed: {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
