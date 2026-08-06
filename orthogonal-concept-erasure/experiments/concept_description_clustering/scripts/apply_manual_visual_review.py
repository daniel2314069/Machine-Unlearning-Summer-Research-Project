#!/usr/bin/env python
"""Apply auditable manual/vision decisions from a small JSON file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--reset-old-provisional", action="store_true")
    args = parser.parse_args()

    path = args.output / "manual_review.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    by_id = {row["candidate_id"]: row for row in rows}
    missing = sorted(set(decisions) - set(by_id))
    if missing:
        raise KeyError(f"Unknown candidate IDs: {missing}")

    if args.reset_old_provisional:
        for row in rows:
            if "provisional_stage2_only" in row["manual_notes"]:
                row["manual_decision"] = "unset"
                row["manual_notes"] = ""
    for candidate_id, review in decisions.items():
        decision = review["decision"].strip().casefold()
        if decision not in {"accept", "reject", "unset"}:
            raise ValueError(f"Invalid decision {decision!r} for {candidate_id}")
        by_id[candidate_id]["manual_decision"] = decision
        by_id[candidate_id]["manual_notes"] = review.get("notes", "")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"applied={len(decisions)} reset_old_provisional={args.reset_old_provisional}")


if __name__ == "__main__":
    main()
