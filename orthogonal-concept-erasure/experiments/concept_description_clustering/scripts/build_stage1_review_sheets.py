#!/usr/bin/env python
"""Create one readable 5x3 Stage-1 review sheet per concept/facet."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.output / "generation_validation.csv").open()))
    rows = [row for row in rows if int(row["seed"]) == 42]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["concept"], row["facet_id"])].append(row)
    review_dir = args.output / "stage1_review_sheets"
    review_dir.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    for (concept, facet), group in sorted(groups.items()):
        group.sort(key=lambda row: row["candidate_id"])
        canvas = Image.new("RGB", (5 * 260, math.ceil(len(group) / 5) * 300), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(group):
            x, y = (slot % 5) * 260, (slot // 5) * 300
            image = Image.open(row["image_path"]).convert("RGB")
            image.thumbnail((250, 250))
            canvas.paste(image, (x + 5, y + 5))
            correct = row["top1_concept"] == concept
            label = (
                f"{row['candidate_id'].rsplit('_', 1)[-1]}  pred={row['top1_concept']}\n"
                f"target={float(row['target_score']):.2f} margin={float(row['target_margin']):+.2f} "
                f"{'AUTO' if correct else 'REVIEW'}"
            )
            draw.multiline_text((x + 5, y + 255), label, fill="darkgreen" if correct else "darkred", font=font)
        canvas.save(review_dir / f"{concept}__{facet}.jpg", quality=90)


if __name__ == "__main__":
    main()
