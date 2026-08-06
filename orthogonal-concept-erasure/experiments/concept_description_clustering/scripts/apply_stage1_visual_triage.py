#!/usr/bin/env python
"""Apply the documented fox/bear Stage-1 visual-review rule.

Indices 10-12 repeatedly rendered as deer/kangaroo-like animals across facets.
Other failed fox/bear candidates are provisionally promoted only to generate the
two additional seeds. Provisional accepts must be reset/reviewed after Stage 2.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review_path = args.output / "manual_review.csv"
    decision_path = args.output / "candidate_generation_decisions.csv"
    rows = list(csv.DictReader(review_path.open()))
    decisions = {row["candidate_id"]: row for row in csv.DictReader(decision_path.open())}
    fields = list(rows[0])
    changed = []
    for row in rows:
        concept = row["concept"]
        if concept not in {"fox", "bear"}:
            continue
        index = int(row["candidate_id"].rsplit("_", 1)[-1])
        if 10 <= index <= 12:
            row["manual_decision"] = "reject"
            row["manual_notes"] = "stage1_visual_rule: repeated deer_or_kangaroo morphology"
            changed.append({**row, "triage_action": "reject_before_stage2"})
        elif decisions[row["candidate_id"]]["stage1_status"] != "pass":
            row["manual_decision"] = "accept"
            row["manual_notes"] = "provisional_stage2_only: visually plausible or classifier-ambiguous"
            changed.append({**row, "triage_action": "promote_to_stage2"})
    write_csv(review_path, rows, fields)
    audit_fields = fields + ["triage_action"]
    write_csv(args.output / "stage1_manual_triage.csv", changed, audit_fields)
    print(f"triaged={len(changed)} promoted={sum(r['triage_action']=='promote_to_stage2' for r in changed)} "
          f"rejected={sum(r['triage_action']=='reject_before_stage2' for r in changed)}")


if __name__ == "__main__":
    main()
