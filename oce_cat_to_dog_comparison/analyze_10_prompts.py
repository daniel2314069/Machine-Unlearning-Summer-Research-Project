import csv
import json
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "oce_cat_to_dog_comparison"
RESULTS = BASE / "indirect_10_results"
PROMPTS = [line.strip() for line in (BASE / "prompts_10_indirect.txt").read_text().splitlines() if line.strip()]
VARIANTS = ("W_0", "W")
LABELS_2 = ("cat", "dog")
LABELS_5 = ("cat", "dog", "mouse", "rabbit", "fox")


def get_font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


paths = [RESULTS / variant / f"{i:02d}_seed_{41 + i}.png" for i in range(1, 11) for variant in VARIANTS]
images = [Image.open(path).convert("RGB") for path in paths]
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)


def classify(labels):
    inputs = processor(
        text=[f"a photo of a {label}" for label in labels],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
    return logits.softmax(dim=1).cpu()


probs_2 = classify(LABELS_2)
probs_5 = classify(LABELS_5)
rows = []
for n, (path, p2, p5) in enumerate(zip(paths, probs_2, probs_5)):
    prompt_id = n // 2 + 1
    variant = VARIANTS[n % 2]
    row = {
        "prompt_id": prompt_id,
        "seed": 41 + prompt_id,
        "weight": variant,
        "cat_2way": float(p2[0]),
        "dog_2way": float(p2[1]),
        **{f"{label}_5way": float(p5[i]) for i, label in enumerate(LABELS_5)},
        "predicted_5way": LABELS_5[int(p5.argmax())],
    }
    rows.append(row)

with (RESULTS / "clip_scores.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for variant in VARIANTS:
    selected = [row for row in rows if row["weight"] == variant]
    summary[variant] = {
        "mean_cat_2way": sum(row["cat_2way"] for row in selected) / len(selected),
        "mean_dog_2way": sum(row["dog_2way"] for row in selected) / len(selected),
        "mean_5way": {
            label: sum(row[f"{label}_5way"] for row in selected) / len(selected)
            for label in LABELS_5
        },
        "predicted_5way_counts": {
            label: sum(row["predicted_5way"] == label for row in selected)
            for label in LABELS_5
        },
    }
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

# Compact paired visual: prompt text, W_0, W, and CLIP annotations per row.
thumb = 256
left = 500
header = 68
row_h = 286
canvas = Image.new("RGB", (left + thumb * 2, header + row_h * 10), "white")
draw = ImageDraw.Draw(canvas)
draw.text((18, 14), "10 indirect descriptions: W_0 vs OCE W (cat -> dog)", fill="black", font=get_font(26))
draw.text((left + 18, 40), "W_0 (original)", fill="#174a7e", font=get_font(18))
draw.text((left + thumb + 18, 40), "W (edited)", fill="#216b35", font=get_font(18))

for i, prompt in enumerate(PROMPTS, start=1):
    y = header + (i - 1) * row_h
    if i % 2 == 0:
        draw.rectangle((0, y, canvas.width, y + row_h - 1), fill="#f5f5f5")
    draw.text((14, y + 12), f"{i:02d} | seed {41 + i}", fill="black", font=get_font(18))
    wrapped = textwrap.wrap(prompt, width=48)
    draw.multiline_text((14, y + 43), "\n".join(wrapped), fill="#222222", font=get_font(17), spacing=5)
    for j, variant in enumerate(VARIANTS):
        image = Image.open(RESULTS / variant / f"{i:02d}_seed_{41 + i}.png").convert("RGB").resize((thumb, thumb))
        x = left + j * thumb
        canvas.paste(image, (x, y))
        row = rows[(i - 1) * 2 + j]
        label = (
            f"cat/dog: {row['cat_2way']:.0%}/{row['dog_2way']:.0%}  "
            f"5-way: {row['predicted_5way']}"
        )
        draw.rectangle((x, y + 256, x + thumb - 1, y + row_h - 1), fill="white")
        draw.text((x + 6, y + 261), label, fill="#222222", font=get_font(13))

canvas.save(ROOT / "OCE_cat_to_dog_10_prompts_W0_vs_W.png")
print(json.dumps(summary, indent=2))
