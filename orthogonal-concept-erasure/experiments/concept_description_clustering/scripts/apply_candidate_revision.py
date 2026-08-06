#!/usr/bin/env python
"""Archive and remove cached rows whose candidate descriptions changed.

The next normal validate-text / validate-generation calls can then resume all
unchanged candidates while generating only the explicitly revised failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-candidates", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    old_path = args.output / "candidate_descriptions.jsonl"
    old_rows = read_jsonl(old_path)
    new_rows = read_jsonl(args.new_candidates)
    old = {row["candidate_id"]: row for row in old_rows}
    new = {row["candidate_id"]: row for row in new_rows}
    if set(old) != set(new):
        raise RuntimeError("A revision must preserve the exact 1,500 candidate IDs")
    changed = sorted(cid for cid in old if old[cid]["description"] != new[cid]["description"])
    if not changed:
        raise RuntimeError("No candidate descriptions changed")

    archive = args.output / "revision_archive" / args.revision
    if archive.exists():
        raise FileExistsError(f"Revision archive already exists: {archive}")
    archive.mkdir(parents=True)
    write_jsonl(archive / "old_candidates.jsonl", [old[cid] for cid in changed])
    write_jsonl(archive / "new_candidates.jsonl", [new[cid] for cid in changed])

    changed_set = set(changed)
    for filename in ["generation_validation.csv", "candidate_generation_decisions.csv", "manual_review.csv"]:
        path = args.output / filename
        rows = read_csv(path)
        if not rows:
            continue
        fields = list(rows[0])
        archived = [row for row in rows if row["candidate_id"] in changed_set]
        retained = [row for row in rows if row["candidate_id"] not in changed_set]
        write_csv(archive / filename, archived, fields)
        write_csv(path, retained, fields)

    images_archive = archive / "generated_images"
    images_archive.mkdir()
    for candidate_id in changed:
        source = args.output / "generated_images" / candidate_id
        if source.exists():
            shutil.move(str(source), str(images_archive / candidate_id))

    (archive / "revision_summary.json").write_text(
        json.dumps({"revision": args.revision, "changed_candidate_ids": changed}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"archived_and_invalidated={len(changed)} archive={archive}")


if __name__ == "__main__":
    main()
