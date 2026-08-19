#!/usr/bin/env python
"""Server-only smoke test for ordered parallel official-GCD face detection."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import sys
from pathlib import Path

from run_sequential_long_horizon import (
    detect_faces_worker,
    initialize_gcd_detector_worker,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcd-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    root = args.gcd_root.resolve()
    resources = root / "examples" / "resources"
    os.environ["APP_DATA_DIR"] = str(resources)
    os.environ["APP_RECOGNITION_WEIGHTS_FILE"] = "face_recognition/best_model_states.pkl"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from model_training.helpers.face_recognizer import FaceRecognizer
    from model_training.helpers.labels import Labels

    labels = Labels(resources_path=str(resources))
    recognizer = FaceRecognizer(
        labels=labels, resources_path=str(resources), use_cuda=False, top_n=5,
    )
    tasks = [(index, str(args.image.resolve())) for index in range(4)]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=initialize_gcd_detector_worker,
        initargs=(str(root), str(resources), 0.2, 224),
    ) as executor:
        results = list(executor.map(detect_faces_worker, tasks, chunksize=1))
    if [index for index, _ in results] != list(range(4)):
        raise RuntimeError("Parallel detector order changed")
    top1 = []
    for _, faces in results:
        prediction = recognizer.perform(faces)
        top1.append(str(prediction[0][0][0][0]) if prediction else None)
    if not all(value and value.startswith("Brad_Pitt_") for value in top1):
        raise RuntimeError(f"Unexpected GCD smoke outputs: {top1}")
    print({"status": "passed", "workers": args.workers, "top1": top1})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
