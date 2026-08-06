#!/usr/bin/env python
"""Create compact three-seed visual-review sheets per concept/facet.

Each five-candidate band has one column per candidate and one row per seed.  The
sheet deliberately shows classifier predictions alongside pixels so manual
decisions can be audited instead of silently replacing the automatic cascade.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-provisional", action="store_true")
    args = parser.parse_args()

    validation = list(csv.DictReader((args.output / "generation_validation.csv").open()))
    reviews = {
        row["candidate_id"]: row
        for row in csv.DictReader((args.output / "manual_review.csv").open())
    }
    by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in validation:
        by_candidate[row["candidate_id"]].append(row)

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for candidate_id, rows in by_candidate.items():
        review = reviews[candidate_id]
        if args.only_provisional and "provisional_stage2_only" not in review["manual_notes"]:
            continue
        groups[(rows[0]["concept"], rows[0]["facet_id"])].append(candidate_id)

    sheet_dir = args.output / (
        "three_seed_provisional_review" if args.only_provisional else "three_seed_review_sheets"
    )
    sheet_dir.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 11) if font_path.exists() else ImageFont.load_default()

    cell_w, image_h, header_h = 210, 190, 45
    for (concept, facet), candidate_ids in sorted(groups.items()):
        candidate_ids.sort()
        bands = (len(candidate_ids) + 4) // 5
        band_h = header_h + 3 * image_h
        canvas = Image.new("RGB", (5 * cell_w, bands * band_h), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, candidate_id in enumerate(candidate_ids):
            band, col = divmod(slot, 5)
            x, y = col * cell_w, band * band_h
            review = reviews[candidate_id]
            suffix = candidate_id.rsplit("_", 1)[-1]
            note = "PROVISIONAL" if "provisional_stage2_only" in review["manual_notes"] else review["manual_decision"]
            draw.text((x + 4, y + 3), f"{suffix}  manual={note}", fill="navy", font=font)
            draw.text((x + 4, y + 22), f"auto={review['automatic_decision']}", fill="black", font=small_font)
            rows = sorted(by_candidate[candidate_id], key=lambda row: int(row["seed"]))
            for seed_slot, row in enumerate(rows[:3]):
                iy = y + header_h + seed_slot * image_h
                image = Image.open(row["image_path"]).convert("RGB")
                image.thumbnail((cell_w - 8, image_h - 20))
                canvas.paste(image, (x + 4 + (cell_w - 8 - image.width) // 2, iy))
                correct = row["top1_concept"] == concept
                label = f"s{row['seed']} pred={row['top1_concept']} m={float(row['target_margin']):+.2f}"
                draw.rectangle((x + 2, iy + image_h - 20, x + cell_w - 2, iy + image_h - 2), fill="white")
                draw.text((x + 4, iy + image_h - 19), label, fill="darkgreen" if correct else "darkred", font=small_font)
        canvas.save(sheet_dir / f"{concept}__{facet}.jpg", quality=92)

    print(f"wrote {len(groups)} sheets to {sheet_dir}")


if __name__ == "__main__":
    main()
