from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
UNIQUE_METRICS = (
    HERE
    / "coco10k_metrics"
    / "methods"
    / "unique_anchor"
    / "first10000"
    / "metrics.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify_unique() -> None:
    payload = read_json(UNIQUE_METRICS)
    expected = {
        "status": "complete",
        "prompt_count": 10000,
        "evaluated_methods": ["unique_anchor"],
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"Unique first-10k metrics are not complete: expected {expected}, got {actual}"
        )
    if payload.get("image_counts") != {"unique_anchor": 10000}:
        raise RuntimeError(
            f"Unexpected unique image counts: {payload.get('image_counts')}"
        )
    model = payload.get("models", {}).get("unique_anchor", {})
    if model.get("clip_score", {}).get("count") != 10000:
        raise RuntimeError("Unique first-10k CLIP count is not 10000")
    if not isinstance(model.get("fid_to_original_sd"), (int, float)):
        raise RuntimeError("Unique first-10k FID is missing")


def verify_gcd() -> None:
    metrics_path = HERE / "gcd_metrics" / "metrics.json"
    predictions_path = HERE / "gcd_metrics" / "predictions.csv"
    per_celebrity_path = HERE / "gcd_metrics" / "per_celebrity_accuracy.csv"
    summary_path = HERE / "gcd_metrics" / "summary.md"
    payload = read_json(metrics_path)
    if payload.get("status") != "complete":
        raise RuntimeError("GCD metrics status is not complete")
    expected_models = {"original_sd", "single_anchor", "unique_anchor"}
    if set(payload.get("models", {})) != expected_models:
        raise RuntimeError("GCD metrics do not contain all three expected models")
    for method, model in payload["models"].items():
        for name in ("Acc_e", "Acc_s", "H_o"):
            value = model.get(name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"Invalid {method} {name}: {value}")
        if model.get("sets", {}).get("targets", {}).get("image_count") != 500:
            raise RuntimeError(f"{method} target image count is not 500")
        if model.get("sets", {}).get("retains", {}).get("image_count") != 500:
            raise RuntimeError(f"{method} retain image count is not 500")
    if count_csv_rows(predictions_path) != 3000:
        raise RuntimeError("GCD predictions.csv does not contain 3000 rows")
    if count_csv_rows(per_celebrity_path) != 450:
        raise RuntimeError(
            "GCD per_celebrity_accuracy.csv does not contain 450 rows"
        )
    if not summary_path.is_file() or summary_path.stat().st_size == 0:
        raise RuntimeError("GCD summary.md is missing or empty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=("unique", "gcd"))
    args = parser.parse_args()
    if args.check == "unique":
        verify_unique()
    else:
        verify_gcd()


if __name__ == "__main__":
    main()
