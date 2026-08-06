import os
import shutil
import subprocess
import sys
from pathlib import Path


TARGET = "Snoopy"
TARGET_DIR = "Snoopy"
ANCHOR = " "
CONTENTS = ["Snoopy", "Mickey", "Spongebob", "Pikachu", "Hello Kitty"]
ROOT = Path(os.environ.get("ROOT", "logs/formal_layer_groups"))
SD_CKPT = os.environ.get("SD_CKPT", "CompVis/stable-diffusion-v1-4")
RETAIN_SCALE = os.environ.get("RETAIN_SCALE", "1.0")
LAMB = os.environ.get("LAMB", "0.5")
AUG_NUM = os.environ.get("AUG_NUM", "10")
THRESHOLD = os.environ.get("THRESHOLD", "1e-1")
CLEAN_IMAGES_AFTER_METRICS = os.environ.get("CLEAN_IMAGES_AFTER_METRICS", "1") == "1"
LOG_PATH = ROOT / "run.log"
GROUPS = [("down_only", "down"), ("mid_only", "mid"), ("up_only", "up")]


def log(message):
    print(message, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(message + "\n")


def run(args):
    log("$ " + " ".join(str(x) for x in args))
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        proc = subprocess.Popen(
            [sys.executable, *map(str, args)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, args)


def record_has_metrics(record, contents):
    record = Path(record)
    if not record.exists():
        return False
    text = record.read_text()
    return all(f"{content}: CS is " in text for content in contents)


def cleanup_mode_images(mode_root):
    if not CLEAN_IMAGES_AFTER_METRICS:
        return
    for content in [*CONTENTS, "coco"]:
        for subdir in ["edit", "combine"]:
            shutil.rmtree(Path(mode_root) / TARGET_DIR / content / subdir, ignore_errors=True)


def run_group(mode_name, group):
    mode_root = ROOT / mode_name
    target_root = mode_root / TARGET_DIR
    target_root.mkdir(parents=True, exist_ok=True)

    if record_has_metrics(target_root / "record_metrics.txt", [*CONTENTS, "coco"]):
        log(f"Skipping {mode_name}: metrics already complete.")
        return

    if not (target_root / "weight.pt").exists():
        run([
            "train_erase_null.py",
            "--sd_ckpt", SD_CKPT,
            "--baseline", "SPEED",
            "--target_concepts", TARGET,
            "--anchor_concepts", ANCHOR,
            "--retain_path", "data/instance.csv",
            "--heads", "concept",
            "--save_path", target_root,
            "--file_name", "weight",
            "--params", "V",
            "--aug_num", AUG_NUM,
            "--threshold", THRESHOLD,
            "--retain_scale", RETAIN_SCALE,
            "--lamb", LAMB,
            "--layer_groups", group,
            "--layer_map_path", ROOT / "layer_map.csv",
            "--diagnostics_path", target_root / "diagnostics.json",
        ])

    run([
        "sample.py",
        "--sd_ckpt", SD_CKPT,
        "--erase_type", "instance",
        "--target_concept", TARGET_DIR,
        "--contents", ", ".join(CONTENTS),
        "--mode", "edit",
        "--num_samples", "10",
        "--batch_size", "10",
        "--save_root", mode_root,
        "--edit_ckpt", target_root / "weight.pt",
    ])

    run([
        "sample2.py",
        "--sd_ckpt", SD_CKPT,
        "--erase_type", "coco",
        "--target_concept", TARGET_DIR,
        "--contents", "coco",
        "--mode", "edit",
        "--num_samples", "1",
        "--batch_size", "10",
        "--save_root", mode_root,
        "--edit_ckpt", target_root / "weight.pt",
    ])

    run([
        "src/clip_score_cal.py",
        "--contents", ", ".join([*CONTENTS, "coco"]),
        "--root_path", target_root,
        "--sub_root", "edit",
        "--pretrained_path", "data/pretrain/instance",
    ])
    cleanup_mode_images(mode_root)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")
    for mode_name, group in GROUPS:
        run_group(mode_name, group)
    run([
        "src/aggregate_table1.py",
        "--root", ROOT,
        "--target", TARGET_DIR,
        "--modes", "original,down_only,mid_only,up_only",
        "--output_csv", ROOT / "tables/layer_group_table1.csv",
        "--output_json", ROOT / "tables/layer_group_table1.json",
        "--all_speed_table", "logs/formal_cprime/tables/cprime_table1.csv",
        "--layer_map", ROOT / "layer_map.csv",
    ])


if __name__ == "__main__":
    main()
