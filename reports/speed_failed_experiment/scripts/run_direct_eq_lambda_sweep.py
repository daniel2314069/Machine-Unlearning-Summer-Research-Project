import subprocess
import sys
from pathlib import Path


LAMBS = ["0.1", "0.01", "0.001", "0.0001", "0.000001", "0.00000001"]


def safe_name(value):
    return value.replace(".", "p").replace("-", "m")


def main():
    root = Path("logs/formal_cprime_lambda_sweep")
    for lamb in LAMBS:
        out = root / f"direct_eq_lamb_{safe_name(lamb)}" / "Snoopy"
        diag = out / "diagnostics.json"
        if diag.exists():
            print(f"skip lamb={lamb}: diagnostics exists", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "train_erase_null.py",
            "--sd_ckpt", "CompVis/stable-diffusion-v1-4",
            "--baseline", "cprime_direct_eq_no_null",
            "--target_concepts", "Snoopy",
            "--anchor_concepts", " ",
            "--retain_path", "data/instance.csv",
            "--heads", "concept",
            "--save_path", str(out),
            "--file_name", "weight",
            "--params", "V",
            "--aug_num", "10",
            "--threshold", "1e-1",
            "--retain_scale", "1.0",
            "--lamb", lamb,
            "--layer_map_path", str(root / "layer_map.csv"),
            "--diagnostics_path", str(diag),
        ]
        print("run lamb=" + lamb, flush=True)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
