from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import atomic_write_text


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _run(
    command: list[str], log_path: Path, env: dict[str, str] | None = None,
    *, check: bool = True,
) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc()}] command={command!r}\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode and check:
        raise RuntimeError(f"Command exited {result.returncode}: {command!r}; see {log_path}")
    return result.returncode


def _merge(
    config: Path, rounds: list[tuple[str, Path]], output: Path, log_path: Path,
    *, check: bool,
) -> int:
    command = [
        sys.executable, "-m", "scripts.merge_codex_diverse_rounds",
        "--config", str(config),
    ]
    for name, root in rounds:
        command.extend(["--round", name, str(root)])
    command.extend(["--output", str(output), "--force"])
    return _run(command, log_path, check=check)


def _round_has_decisions(root: Path) -> bool:
    return (root / "candidate_generation_decisions.csv").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for generation rounds and finish the balanced analysis")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--round2", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--round3-config", type=Path)
    parser.add_argument("--round3-pool-config", type=Path)
    parser.add_argument("--round3-output", type=Path)
    parser.add_argument(
        "--target", action="append", nargs=4,
        metavar=("NAME", "CONFIG", "CANDIDATES", "OUTPUT"), default=[],
        help="Run a targeted cached generation replenishment after round 3 if merging is still deficient.",
    )
    args = parser.parse_args()

    supervisor_root = args.final_output
    supervisor_root.mkdir(parents=True, exist_ok=True)
    state_path = supervisor_root / "supervisor_state.json"
    heartbeat_path = supervisor_root / "supervisor_heartbeat.json"
    log_path = supervisor_root / "supervisor.log"
    state: dict[str, Any] = {
        "status": "running", "stage": "waiting_for_round2", "started_utc": _utc(),
        "round1": str(args.round1.resolve()), "round2": str(args.round2.resolve()),
        "final_output": str(args.final_output.resolve()), "config": str(args.config.resolve()),
    }
    _write(state_path, state)
    try:
        while True:
            round_state_path = args.round2 / "state.json"
            round_state = _load(round_state_path) if round_state_path.exists() else {"status": "starting"}
            _write(heartbeat_path, {
                "timestamp_utc": _utc(), "pid": os.getpid(), "supervisor_stage": state["stage"],
                "round2_status": round_state.get("status"),
                "round2_stage": round_state.get("current_stage"),
                "round2_counts": round_state.get("counts", {}),
            })
            if round_state.get("status") in {"complete", "failed", "timed_out"}:
                break
            time.sleep(max(10, args.poll_seconds))

        state["stage"] = "merging_automatic_accepts"
        _write(state_path, state)
        rounds: list[tuple[str, Path]] = [("round1", args.round1), ("round2", args.round2)]
        merge_code = _merge(args.config, rounds, args.final_output, log_path, check=False)

        if merge_code and args.round3_config and args.round3_pool_config and args.round3_output:
            state["stage"] = "round3_replenishment"
            _write(state_path, state)
            args.round3_output.mkdir(parents=True, exist_ok=True)
            action = "resume" if (args.round3_output / "state.json").exists() else "run"
            _run([
                sys.executable, "-m", "concept_clustering.overnight_runner", action,
                "--config", str(args.round3_config),
                "--pool-config", str(args.round3_pool_config),
                "--root", str(args.round3_output), "--max-hours", "720",
            ], log_path, check=False)
            if not _round_has_decisions(args.round3_output):
                raise RuntimeError(f"Round 3 produced no generation decisions: {args.round3_output}")
            rounds.append(("round3", args.round3_output))
            state["stage"] = "merging_after_round3"
            _write(state_path, state)
            merge_code = _merge(args.config, rounds, args.final_output, log_path, check=False)

        for target_name, config_text, candidates_text, output_text in args.target:
            if not merge_code:
                break
            target_config = Path(config_text)
            candidates = Path(candidates_text)
            target_output = Path(output_text)
            state["stage"] = f"targeted_replenishment:{target_name}"
            _write(state_path, state)
            target_output.mkdir(parents=True, exist_ok=True)
            if not (target_output / "candidate_text_validation.csv").exists():
                _run([
                    sys.executable, "-m", "concept_clustering.cli", "validate-text",
                    "--config", str(target_config), "--candidates", str(candidates),
                    "--output", str(target_output),
                ], log_path)
            _run([
                sys.executable, "-m", "concept_clustering.cli", "validate-generation",
                "--config", str(target_config), "--output", str(target_output),
                "--stage", "all", "--resume",
            ], log_path)
            if not _round_has_decisions(target_output):
                raise RuntimeError(f"Targeted round produced no generation decisions: {target_output}")
            rounds.append((target_name, target_output))
            state["stage"] = f"merging_after:{target_name}"
            _write(state_path, state)
            merge_code = _merge(args.config, rounds, args.final_output, log_path, check=False)

        if merge_code:
            shortage_path = args.final_output / "facet_shortages.csv"
            raise RuntimeError(
                f"All configured replenishment rounds were exhausted and the corpus is still deficient; "
                f"see {shortage_path}"
            )

        for name in ["run_metadata.json"]:
            source = args.round2 / name
            if source.exists():
                shutil.copy2(source, args.final_output / name)
        atomic_write_text(
            args.final_output / "resolved_config.json",
            json.dumps(__import__("concept_clustering.config", fromlist=["load_config"]).load_config(args.config), indent=2) + "\n",
        )

        state["stage"] = "extracting_embeddings"
        _write(state_path, state)
        _run([
            sys.executable, "-m", "concept_clustering.cli", "extract",
            "--config", str(args.config), "--output", str(args.final_output), "--projection", "to_v",
        ], log_path)
        state["stage"] = "clustering_and_reporting"
        _write(state_path, state)
        _run([
            sys.executable, "-m", "concept_clustering.cli", "analyze",
            "--config", str(args.config), "--output", str(args.final_output),
        ], log_path)
        state.update({
            "status": "complete", "stage": "finished", "completed_utc": _utc(),
            "analysis_output": str(args.final_output.resolve()),
            "completion_mode": "merged_automatic_accepts_with_replenishment",
            "rounds_used": [{"name": name, "root": str(root.resolve())} for name, root in rounds],
        })
        _write(state_path, state)
        _write(heartbeat_path, {"timestamp_utc": _utc(), **state})
    except Exception as exc:
        state.update({
            "status": "needs_replenishment", "stage": "stopped", "updated_utc": _utc(),
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
        })
        _write(state_path, state)
        _write(heartbeat_path, {"timestamp_utc": _utc(), **state})
        raise


if __name__ == "__main__":
    main()
