import argparse
import json
import os
import re

import pandas as pd


CONTENTS = ["Snoopy", "Mickey", "Spongebob", "Pikachu", "Hello Kitty", "coco"]
NON_TARGETS = ["Mickey", "Spongebob", "Pikachu", "Hello Kitty"]


def parse_record(path):
    out = {}
    if not os.path.exists(path):
        return out
    pattern = re.compile(r"^(.*): CS is ([^,]+), FID is ([^\s]+)")
    with open(path, "r") as f:
        for line in f:
            match = pattern.search(line.strip())
            if match:
                out[match.group(1)] = {
                    "CS": float(match.group(2)),
                    "FID": float(match.group(3)),
                }
    return out


def metrics_for_mode(root, target, mode_name, sub_root):
    mode_root = os.path.join(root, mode_name, target)
    return parse_record(os.path.join(mode_root, "record_metrics.txt"))


def metrics_for_original(pretrain_root):
    metrics = parse_record(os.path.join(pretrain_root, "instance", "record_metrics.txt"))
    coco_record = parse_record(os.path.join(pretrain_root, "coco", "record_metrics.txt"))
    if "coco" in coco_record:
        metrics["coco"] = coco_record["coco"]
    return metrics


def make_row(mode, metrics, all_speed=None, original=None, sweep_type=None):
    row = {"mode": mode}
    row["Snoopy_CS"] = metrics.get("Snoopy", {}).get("CS")
    for content, column in [
        ("Mickey", "Mickey_FID"),
        ("Spongebob", "SpongeBob_FID"),
        ("Pikachu", "Pikachu_FID"),
        ("Hello Kitty", "HelloKitty_FID"),
    ]:
        row[column] = metrics.get(content, {}).get("FID")
    fid_values = [row[x] for x in ["Mickey_FID", "SpongeBob_FID", "Pikachu_FID", "HelloKitty_FID"] if row[x] is not None]
    row["avg_non_target_FID"] = sum(fid_values) / len(fid_values) if fid_values else None
    row["MSCOCO_CS"] = metrics.get("coco", {}).get("CS")
    row["MSCOCO_FID"] = metrics.get("coco", {}).get("FID")

    if all_speed and mode.startswith("loo_"):
        row["delta_target_CS"] = None if row["Snoopy_CS"] is None or all_speed.get("Snoopy_CS") is None else row["Snoopy_CS"] - all_speed["Snoopy_CS"]
        row["delta_avg_non_target_FID"] = None if row["avg_non_target_FID"] is None or all_speed.get("avg_non_target_FID") is None else row["avg_non_target_FID"] - all_speed["avg_non_target_FID"]
    if original and mode.startswith("single_"):
        row["target_CS_reduction_vs_original"] = None if row["Snoopy_CS"] is None or original.get("Snoopy_CS") is None else original["Snoopy_CS"] - row["Snoopy_CS"]
    if sweep_type:
        row["sweep_type"] = sweep_type
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--target", default="Snoopy")
    parser.add_argument("--modes", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--pretrain_root", default="data/pretrain")
    parser.add_argument("--all_speed_table", default=None)
    parser.add_argument("--layer_map", default=None)
    args = parser.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    rows = []
    original_row = None
    all_speed_row = None

    for mode in modes:
        if mode == "original":
            metrics = metrics_for_original(args.pretrain_root)
        else:
            metrics = metrics_for_mode(args.root, args.target, mode, "edit")
        sweep_type = None
        if mode.startswith("single_"):
            sweep_type = "single_layer"
        elif mode.startswith("loo_"):
            sweep_type = "leave_one_out"
        elif mode in {"down_only", "mid_only", "up_only", "down_mid", "mid_up", "down_up"}:
            sweep_type = "group"
        row = make_row(mode, metrics, all_speed=all_speed_row, original=original_row, sweep_type=sweep_type)
        rows.append(row)
        if mode == "original":
            original_row = row
        if mode in {"speed", "all_speed"}:
            all_speed_row = row

    if args.all_speed_table and all_speed_row is None and os.path.exists(args.all_speed_table):
        all_df = pd.read_csv(args.all_speed_table)
        candidates = all_df[all_df["mode"].isin(["speed", "all_speed", "SPEED"])]
        if not candidates.empty:
            all_speed_row = candidates.iloc[0].to_dict()
            rows = [
                make_row(r["mode"], metrics_for_mode(args.root, args.target, r["mode"], "edit"), all_speed=all_speed_row, original=original_row, sweep_type=r.get("sweep_type"))
                if str(r["mode"]).startswith("loo_") else r
                for r in rows
            ]

    df = pd.DataFrame(rows)
    if args.layer_map and os.path.exists(args.layer_map):
        layer_df = pd.read_csv(args.layer_map)
        layer_rows = []
        for _, row in df.iterrows():
            mode = row["mode"]
            layer_index = None
            if isinstance(mode, str) and (mode.startswith("single_") or mode.startswith("loo_")):
                layer_index = int(mode.split("_")[1])
            if layer_index is not None:
                layer_info = layer_df[layer_df["layer_index"] == layer_index]
                if not layer_info.empty:
                    row["layer_index"] = layer_index
                    row["layer_name"] = layer_info.iloc[0]["layer_name"]
                    row["group"] = layer_info.iloc[0]["group"]
            layer_rows.append(row.to_dict())
        df = pd.DataFrame(layer_rows)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
