import os
import shutil
import subprocess
import sys
from pathlib import Path


TARGET = "Snoopy"
TARGET_DIR = "Snoopy"
ANCHOR = " "
CONTENTS = ["Snoopy", "Mickey", "Spongebob", "Pikachu", "Hello Kitty"]
ROOT = Path(os.environ.get("ROOT", "logs/formal_cprime"))
SD_CKPT = os.environ.get("SD_CKPT", "CompVis/stable-diffusion-v1-4")
RETAIN_SCALE = os.environ.get("RETAIN_SCALE", "1.0")
LAMB = os.environ.get("LAMB", "0.5")
AUG_NUM = os.environ.get("AUG_NUM", "10")
THRESHOLD = os.environ.get("THRESHOLD", "1e-1")
CLEAN_IMAGES_AFTER_METRICS = os.environ.get("CLEAN_IMAGES_AFTER_METRICS", "0") == "1"
LOG_PATH = ROOT / "run.log"


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


def count_png(path):
    path = Path(path)
    if not path.is_dir():
        return 0
    return len(list(path.glob("*.png")))


def record_has_metrics(record, contents):
    record = Path(record)
    if not record.exists():
        return False
    text = record.read_text()
    return all(f"{content}: CS is " in text for content in contents)


def prepare_pretrain():
    if any(count_png(Path("data/pretrain/instance") / c / "original") < 800 for c in CONTENTS):
        run([
            "sample.py",
            "--sd_ckpt", SD_CKPT,
            "--erase_type", "instance",
            "--target_concept", "instance",
            "--contents", ", ".join(CONTENTS),
            "--mode", "original",
            "--num_samples", "10",
            "--batch_size", "10",
            "--save_root", "data/pretrain",
        ])

    if count_png("data/pretrain/coco/coco/original") < 1000:
        run([
            "sample2.py",
            "--sd_ckpt", SD_CKPT,
            "--erase_type", "coco",
            "--target_concept", "coco",
            "--contents", "coco",
            "--mode", "original",
            "--num_samples", "1",
            "--batch_size", "10",
            "--save_root", "data/pretrain",
        ])

    if not record_has_metrics("data/pretrain/instance/record_metrics.txt", CONTENTS):
        run([
            "src/clip_score_cal.py",
            "--contents", ", ".join(CONTENTS),
            "--root_path", "data/pretrain/instance",
            "--sub_root", "original",
            "--pretrained_path", "data/pretrain/instance",
        ])

    if not record_has_metrics("data/pretrain/coco/record_metrics.txt", ["coco"]):
        run([
            "src/clip_score_cal.py",
            "--contents", "coco",
            "--root_path", "data/pretrain/coco",
            "--sub_root", "original",
            "--pretrained_path", "data/pretrain/coco",
        ])


def cleanup_mode_images(mode_root):
    if not CLEAN_IMAGES_AFTER_METRICS:
        return
    for content in [*CONTENTS, "coco"]:
        for subdir in ["edit", "combine"]:
            shutil.rmtree(Path(mode_root) / TARGET_DIR / content / subdir, ignore_errors=True)


def run_mode(mode_name, baseline):
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
            "--baseline", baseline,
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
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")
    prepare_pretrain()
    run_mode("speed", "SPEED")
    run_mode("cprime_null_no_iec", "cprime_null_no_iec")
    run_mode("cprime_direct_eq_no_null", "cprime_direct_eq_no_null")
    run([
        "src/aggregate_table1.py",
        "--root", ROOT,
        "--target", TARGET_DIR,
        "--modes", "original,speed,cprime_null_no_iec,cprime_direct_eq_no_null",
        "--output_csv", ROOT / "tables/cprime_table1.csv",
        "--output_json", ROOT / "tables/cprime_table1.json",
        "--layer_map", ROOT / "layer_map.csv",
    ])


if __name__ == "__main__":
    main()
