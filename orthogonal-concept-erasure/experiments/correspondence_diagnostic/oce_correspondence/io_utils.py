from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_rgb_std(path: Path) -> float:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return float(image.std())


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_seed_grid(
    output_path: Path,
    method_images: Mapping[str, Sequence[Path]],
    seeds: Sequence[int],
    title: str,
    cell_size: int = 256,
) -> None:
    methods = list(method_images)
    if not methods:
        raise ValueError("At least one method is required for a grid")
    if any(len(method_images[method]) != len(seeds) for method in methods):
        raise ValueError("Each grid method must contain one image per seed")
    label_width = 100
    header_height = 78
    row_height = cell_size + 8
    canvas = Image.new(
        "RGB",
        (label_width + cell_size * len(methods), header_height + row_height * len(seeds)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill="#17212b", font=_font(22))
    for column, method in enumerate(methods):
        x = label_width + column * cell_size
        draw.text((x + 8, 43), method, fill="#17212b", font=_font(16))
    for row_index, seed in enumerate(seeds):
        y = header_height + row_index * row_height
        draw.text((10, y + cell_size // 2 - 8), str(seed), fill="#3f4a56", font=_font(16))
        for column, method in enumerate(methods):
            image = Image.open(method_images[method][row_index]).convert("RGB")
            image.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = label_width + column * cell_size
            canvas.paste(image, (x + (cell_size - image.width) // 2, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def plot_heatmap(
    output_path: Path,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    value_format: str = ".3f",
    center: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix)
    width = max(5.5, 1.0 + 1.15 * len(column_labels))
    height = max(4.5, 1.2 + 0.8 * len(row_labels))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    if center is None:
        image = ax.imshow(values, cmap="Blues", aspect="auto")
    else:
        bound = max(
            abs(float(np.nanmin(values)) - center),
            abs(float(np.nanmax(values)) - center),
            1e-6,
        )
        image = ax.imshow(
            values,
            cmap="RdBu_r",
            aspect="auto",
            vmin=center - bound,
            vmax=center + bound,
        )
    ax.set_xticks(np.arange(len(column_labels)), labels=column_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="x", rotation=35)
    ax.set_xlabel("Anchor")
    ax.set_ylabel("Target prompt")
    ax.set_title(title)
    threshold = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "NA" if not np.isfinite(value) else format(value, value_format)
            color = "white" if np.isfinite(value) and value > threshold else "#17212b"
            ax.text(column, row, label, ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
