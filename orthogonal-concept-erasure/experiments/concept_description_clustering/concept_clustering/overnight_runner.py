from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .codex_diverse import build_codex_diverse_candidates, score_and_select_tfidf_hard
from .config import load_config
from .generation_validation import finalize_accepted
from .text_validation import validate_candidates
from .utils import atomic_write_text, read_csv, read_jsonl, write_jsonl


GRACEFUL_TIMEOUT_EXIT = 75


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            pass
    return total


def _gpu_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or result.stderr.strip() or f"exit={result.returncode}"
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


class OvernightRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.exceptions_path = self.root / "exceptions.jsonl"
        self.stage_log = self.root / "stage_subprocess.log"
        self.config = load_config(args.config)
        self.pool_config = load_config(args.pool_config)
        self.runner_config = self.config.get("runner", {})
        self.heartbeat_seconds = int(self.runner_config.get("heartbeat_seconds", 30))
        self.max_retries = int(self.runner_config.get("max_retries", 2))
        self.retry_delay = int(self.runner_config.get("retry_delay_seconds", 20))
        self.grace = int(self.runner_config.get("deadline_grace_seconds", 300))
        self.stop_requested = False
        self.state = self._load_or_create_state()
        self.stop_file = self.root / f"STOP_REQUESTED_{self.state['run_id']}"
        self.state["active_stop_file"] = str(self.stop_file)
        signal.signal(signal.SIGTERM, self._signal_stop)
        signal.signal(signal.SIGINT, self._signal_stop)

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if self.args.command == "run" and state.get("status") not in {"complete", "timed_out", "failed"}:
                raise RuntimeError("An unfinished state already exists; use the resume command")
            if self.args.command == "resume":
                state["resume_count"] = int(state.get("resume_count", 0)) + 1
                state["status"] = "running"
                state["deadline_epoch"] = time.time() + self.args.max_hours * 3600
                state["last_resume_utc"] = _utc()
                state["run_id"] = f"resume_{int(time.time())}"
                return state
            if self.args.command in {"status", "cluster-only"}:
                return state
            raise FileExistsError(f"Refusing to replace existing run state: {self.state_path}")
        if self.args.command != "run":
            raise FileNotFoundError(f"No state exists at {self.state_path}; launch with run first")
        now = time.time()
        return {
            "experiment": "codex_diverse_single_source",
            "provenance": "codex_diverse",
            "run_id": f"run_{int(now)}",
            "status": "running",
            "started_utc": _utc(),
            "started_epoch": now,
            "deadline_epoch": now + self.args.max_hours * 3600,
            "max_hours": self.args.max_hours,
            "resume_count": 0,
            "current_stage": "initializing",
            "stages": {},
            "config": str(self.args.config.resolve()),
            "pool_config": str(self.args.pool_config.resolve()),
            "output": str(self.root),
            "commands": {
                "resume": self._resume_command(),
                "status": self._status_command(),
                "cluster_only": self._cluster_command(),
            },
        }

    def _resume_command(self) -> str:
        return (
            f"./scripts/run_py310.sh -m concept_clustering.overnight_runner resume "
            f"--config {self.args.config} --pool-config {self.args.pool_config} "
            f"--root {self.args.root} --max-hours {self.args.max_hours}"
        )

    def _status_command(self) -> str:
        return f"./scripts/run_py310.sh -m concept_clustering.overnight_runner status --root {self.args.root}"

    def _cluster_command(self) -> str:
        return (
            f"./scripts/run_py310.sh -m concept_clustering.overnight_runner cluster-only "
            f"--config {self.args.config} --pool-config {self.args.pool_config} --root {self.args.root}"
        )

    def _signal_stop(self, signum, _frame):
        self.stop_requested = True
        atomic_write_text(self.stop_file, f"signal={signum} utc={_utc()}\n")

    def _save_state(self) -> None:
        self.state["updated_utc"] = _utc()
        self.state["elapsed_seconds"] = time.time() - float(self.state["started_epoch"])
        self.state["remaining_seconds"] = max(0.0, float(self.state["deadline_epoch"]) - time.time())
        _atomic_json(self.state_path, self.state)

    def _counts(self) -> dict[str, Any]:
        generation = read_csv(self.root / "generation_validation.csv")
        decisions = read_csv(self.root / "candidate_generation_decisions.csv")
        text_rows = read_csv(self.root / "pool_validation" / "candidate_text_validation.csv")
        return {
            "candidate_pool": _line_count(self.root / "candidate_pool.jsonl"),
            "text_valid": sum(str(row.get("text_valid", "")).casefold() == "true" for row in text_rows),
            "selected_for_generation": _line_count(self.root / "selected_candidates.jsonl"),
            "generation_score_rows": len(generation),
            "stage1_images": sum(str(row.get("stage", row.get("generation_stage"))) == "1" for row in generation),
            "stage2_images": sum(str(row.get("stage", row.get("generation_stage"))) == "2" for row in generation),
            "automatic_accepted": sum(row.get("automatic_decision") == "accepted" for row in decisions),
            "accepted_descriptions": _line_count(self.root / "accepted_descriptions.jsonl"),
        }

    def heartbeat(self) -> None:
        usage = shutil.disk_usage(self.root)
        payload = {
            "timestamp_utc": _utc(),
            "pid": os.getpid(),
            "status": self.state.get("status"),
            "current_stage": self.state.get("current_stage"),
            "elapsed_seconds": time.time() - float(self.state["started_epoch"]),
            "remaining_seconds": max(0.0, float(self.state["deadline_epoch"]) - time.time()),
            "counts": self._counts(),
            "experiment_bytes": _directory_size(self.root),
            "filesystem_free_bytes": usage.free,
            "gpu_memory_used_free_util_temperature": _gpu_status(),
        }
        _atomic_json(self.heartbeat_path, payload)
        self.state["last_heartbeat_utc"] = payload["timestamp_utc"]
        self.state["counts"] = payload["counts"]
        self._save_state()

    def _deadline_near(self) -> bool:
        return self.stop_requested or time.time() >= float(self.state["deadline_epoch"]) - self.grace

    def _record_exception(self, stage: str, exc: BaseException) -> None:
        record = {
            "timestamp_utc": _utc(), "stage": stage,
            "exception_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with self.exceptions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        self.state.setdefault("exceptions", []).append({key: record[key] for key in ["timestamp_utc", "stage", "exception_type", "message"]})
        self._save_state()

    def _run_stage(self, name: str, function: Callable[[], None], outputs: list[Path]) -> bool:
        if outputs and all(path.exists() for path in outputs):
            self.state["stages"][name] = {"status": "skipped_cached", "updated_utc": _utc()}
            self.heartbeat()
            return True
        if self._deadline_near():
            return False
        self.state["current_stage"] = name
        attempts = int(self.state["stages"].get(name, {}).get("attempts", 0))
        while attempts <= self.max_retries:
            attempts += 1
            self.state["stages"][name] = {"status": "running", "attempts": attempts, "started_utc": _utc()}
            self.heartbeat()
            try:
                function()
                self.state["stages"][name].update({"status": "complete", "completed_utc": _utc()})
                self.heartbeat()
                return True
            except Exception as exc:
                self._record_exception(name, exc)
                self.state["stages"][name].update({"status": "failed_attempt", "failed_utc": _utc(), "message": str(exc)})
                self.heartbeat()
                if attempts > self.max_retries or self._deadline_near():
                    return False
                time.sleep(self.retry_delay)
        return False

    def _subprocess(self, arguments: list[str], stage: str) -> None:
        env = os.environ.copy()
        env.update({
            "CONCEPT_CLUSTER_DEADLINE_EPOCH": str(self.state["deadline_epoch"]),
            "CONCEPT_CLUSTER_DEADLINE_GRACE_SECONDS": str(self.grace),
            "CONCEPT_CLUSTER_STOP_FILE": str(self.stop_file),
            "MPLCONFIGDIR": str(self.root / ".matplotlib"),
        })
        (self.root / ".matplotlib").mkdir(exist_ok=True)
        command = [sys.executable, "-m", "concept_clustering.cli", *arguments]
        with self.stage_log.open("a", encoding="utf-8") as log:
            log.write(f"\n[{_utc()}] stage={stage} command={command!r}\n")
            log.flush()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            self.state["child_pid"] = process.pid
            self._save_state()
            while process.poll() is None:
                self.heartbeat()
                if self._deadline_near() and not self.stop_file.exists():
                    atomic_write_text(self.stop_file, f"deadline_stop utc={_utc()}\n")
                time.sleep(self.heartbeat_seconds)
            self.state.pop("child_pid", None)
            code = int(process.returncode)
        if code == GRACEFUL_TIMEOUT_EXIT:
            raise TimeoutError(f"Stage {stage} stopped gracefully near the wall-clock deadline")
        if code != 0:
            raise RuntimeError(f"Stage {stage} exited with status {code}; see {self.stage_log}")

    def _archive_analysis_200(self) -> None:
        target = self.root / "analysis_200"
        target.mkdir(exist_ok=True)
        names = [
            "accepted_descriptions.jsonl", "raw_text_embeddings.pt", "layer_embeddings.pt",
            "clustering_metrics.csv", "clustering_assignments.csv", "prototype_metrics.csv",
            "facet_confounding_metrics.csv", "layer_metrics.csv", "lexical_baseline_clustering_metrics.csv",
            "lexical_baseline_assignments.csv", "baseline_classification_metrics.csv", "tfidf_top_features.csv",
            "final_report.md", "tokenization_audit.csv", "word_shuffle_audit.csv",
        ]
        for name in names:
            source = self.root / name
            destination = target / name
            if source.exists() and not destination.exists():
                shutil.copy2(source, destination)
        for directory in ["plots", "confusion_matrices", "baseline_confusion_matrices"]:
            source = self.root / directory
            destination = target / directory
            if source.exists() and not destination.exists():
                shutil.copytree(source, destination)

    def run(self) -> None:
        self._save_state()
        stages = [
            ("build_candidate_pool", lambda: build_codex_diverse_candidates(self.pool_config, self.root / "candidate_pool.jsonl"), [self.root / "candidate_pool.jsonl"]),
            ("validate_candidate_pool", lambda: validate_candidates(self.pool_config, self.root / "candidate_pool.jsonl", self.root / "pool_validation"), [self.root / "pool_validation" / "candidate_text_validation.csv"]),
            ("tfidf_hard_selection", lambda: score_and_select_tfidf_hard(self.config, self.root / "pool_validation", self.root / "selected_candidates.jsonl", self.root / "tfidf_hardness.csv"), [self.root / "selected_candidates.jsonl", self.root / "tfidf_hardness.csv"]),
            ("validate_selected_candidates", lambda: validate_candidates(self.config, self.root / "selected_candidates.jsonl", self.root), [self.root / "candidate_text_validation.csv"]),
            ("generation_stage1", lambda: self._subprocess(["validate-generation", "--config", str(self.args.config), "--output", str(self.root), "--stage", "1", "--resume"], "generation_stage1"), []),
            ("generation_stage2", lambda: self._subprocess(["validate-generation", "--config", str(self.args.config), "--output", str(self.root), "--stage", "2", "--resume"], "generation_stage2"), []),
            ("finalize_200", lambda: finalize_accepted(self.config, self.root, force=False), [self.root / "accepted_descriptions.jsonl"]),
            ("extract_embeddings", lambda: self._subprocess(["extract", "--config", str(self.args.config), "--output", str(self.root), "--projection", "to_v"], "extract_embeddings"), [self.root / "raw_text_embeddings.pt", self.root / "layer_embeddings.pt"]),
            ("cluster_and_report", lambda: self._subprocess(["analyze", "--config", str(self.args.config), "--output", str(self.root)], "cluster_and_report"), [self.root / "final_report.md"]),
        ]
        success = True
        for name, function, outputs in stages:
            if not self._run_stage(name, function, outputs):
                success = False
                break

        if success and not self._deadline_near():
            self._archive_analysis_200()
            if self._eligible_for_400():
                self.state["upgrade_400"] = "eligible_but_not_started_unless_six_hours_remain"
                if float(self.state["deadline_epoch"]) - time.time() >= 6 * 3600:
                    config_400_path = self.args.config.with_name("codex_diverse_4x100.json")
                    config_400 = load_config(config_400_path)
                    success = self._run_stage(
                        "finalize_400",
                        lambda: finalize_accepted(config_400, self.root, force=True),
                        [self.root / "accepted_descriptions_400.marker"],
                    )
                    if success:
                        atomic_write_text(self.root / "accepted_descriptions_400.marker", _utc() + "\n")
                        success = self._run_stage(
                            "extract_embeddings_400",
                            lambda: self._subprocess(["extract", "--config", str(config_400_path), "--output", str(self.root), "--projection", "to_v", "--force"], "extract_embeddings_400"),
                            [],
                        )
                    if success:
                        success = self._run_stage(
                            "cluster_and_report_400",
                            lambda: self._subprocess(["analyze", "--config", str(config_400_path), "--output", str(self.root)], "cluster_and_report_400"),
                            [],
                        )
            else:
                self.state["upgrade_400"] = "insufficient_automatic_accepts_per_facet"

        if self._deadline_near():
            self.state["status"] = "timed_out"
        elif success:
            self.state["status"] = "complete"
        else:
            self.state["status"] = "failed"
        self.state["current_stage"] = "finished"
        self.heartbeat()
        self.write_overview_report()

    def _eligible_for_400(self) -> bool:
        decisions = read_csv(self.root / "candidate_generation_decisions.csv")
        counts = Counter(
            (row["concept"], row["facet_id"])
            for row in decisions if row.get("automatic_decision") == "accepted"
        )
        return all(
            counts[(concept["name"], facet["id"])] >= 10
            for concept in self.config["concepts"] for facet in self.config["facets"]
        )

    def write_overview_report(self) -> None:
        text_validation = read_csv(self.root / "pool_validation" / "candidate_text_validation.csv")
        rejection_reasons = Counter()
        for row in text_validation:
            if str(row.get("text_valid", "")).casefold() != "true":
                for reason in str(row.get("failure_reasons", "")).split(";"):
                    if reason:
                        rejection_reasons[reason.split(":", 1)[0]] += 1
        generation = read_csv(self.root / "generation_validation.csv")
        decisions = read_csv(self.root / "candidate_generation_decisions.csv")
        accepted = read_jsonl(self.root / "accepted_descriptions.jsonl") if (self.root / "accepted_descriptions.jsonl").exists() else []
        accepted_counts = Counter((row["concept"], row["facet_id"]) for row in accepted)
        oof = read_csv(self.root / "tfidf_oof_summary.csv")
        hardness = pd.read_csv(self.root / "tfidf_hardness.csv") if (self.root / "tfidf_hardness.csv").exists() else pd.DataFrame()
        clustering = pd.read_csv(self.root / "clustering_metrics.csv") if (self.root / "clustering_metrics.csv").exists() else pd.DataFrame()
        lexical = pd.read_csv(self.root / "lexical_baseline_clustering_metrics.csv") if (self.root / "lexical_baseline_clustering_metrics.csv").exists() else pd.DataFrame()
        classification = pd.read_csv(self.root / "baseline_classification_metrics.csv") if (self.root / "baseline_classification_metrics.csv").exists() else pd.DataFrame()

        stage1_rows = sum(str(row.get("stage", row.get("generation_stage"))) == "1" for row in generation)
        stage2_rows = sum(str(row.get("stage", row.get("generation_stage"))) == "2" for row in generation)
        stage1_survivors = sum(row.get("stage1_status") == "pass" for row in decisions)
        automatic_accepted = sum(row.get("automatic_decision") == "accepted" for row in decisions)
        stage1_pass_rate = stage1_survivors / len(decisions) if decisions else float("nan")
        final_pass_rate = automatic_accepted / len(decisions) if decisions else float("nan")

        fixed_lines = []
        if not clustering.empty:
            for representation, frame in clustering.groupby("representation"):
                fixed_lines.append(
                    f"| {representation} | {frame.ari_concept.mean():.4f} | {frame.nmi_concept.mean():.4f} | "
                    f"{frame.hungarian_accuracy.mean():.4f} | {frame.silhouette.mean():.4f} | {frame.ari_facet.mean():.4f} |"
                )
        lexical_lines = []
        if not lexical.empty:
            for representation, frame in lexical.groupby("representation"):
                lexical_lines.append(
                    f"| {representation} | {frame.ari_concept.mean():.4f} | {frame.nmi_concept.mean():.4f} | "
                    f"{frame.hungarian_accuracy.mean():.4f} | {frame.silhouette.mean():.4f} | {frame.ari_facet.mean():.4f} |"
                )
        report = f"""# Codex-diverse unattended concept experiment

Status: **{self.state.get('status')}**  
Started: {self.state.get('started_utc')}  
Updated: {_utc()}  
Elapsed: {(time.time() - float(self.state['started_epoch'])) / 3600:.2f} hours  
Wall-clock budget per launch/resume: {self.args.max_hours:.2f} hours

## Provenance and scope

Every candidate in this experiment has the single honest provenance label `codex_diverse`. This corpus does not represent independent human authors or multiple language models. It uses only original, unedited SD 1.4 and never edits W0 or applies OCE.

## Candidate and text validation

- Initial candidates: {_line_count(self.root / 'candidate_pool.jsonl')}
- Text-valid candidates: {sum(str(row.get('text_valid', '')).casefold() == 'true' for row in text_validation)}
- TF-IDF-hard candidates sent to generation: {_line_count(self.root / 'selected_candidates.jsonl')}
- Text rejection reasons: `{json.dumps(rejection_reasons, sort_keys=True)}`

OOF TF-IDF selection summaries: `{json.dumps(oof, sort_keys=True)}`

Selected-candidate TF-IDF difficulty means: `{json.dumps({column: float(hardness.loc[hardness.selected_for_generation.astype(str).str.casefold() == 'true', column].mean()) for column in ['tfidf_incorrect_model_count', 'tfidf_mean_target_probability', 'tfidf_mean_target_margin', 'tfidf_difficulty_score']} if not hardness.empty else {}, sort_keys=True)}`

## Generation cascade

- Stage-1 image rows: {stage1_rows}
- Stage-2 image rows: {stage2_rows}
- Stage-1 survivors: {stage1_survivors} ({stage1_pass_rate:.2%} of evaluated candidates)
- Automatically accepted after formal rules: {automatic_accepted} ({final_pass_rate:.2%} of evaluated candidates)
- Final accepted descriptions: {len(accepted)}
- Accepted per concept/facet: `{json.dumps({f'{key[0]}/{key[1]}': value for key, value in sorted(accepted_counts.items())})}`

## Contextual and surface baselines

| Representation | Concept ARI | Concept NMI | Matched accuracy | Silhouette | Facet ARI |
|---|---:|---:|---:|---:|---:|
{chr(10).join(fixed_lines + lexical_lines) if fixed_lines or lexical_lines else '| unavailable | | | | | |'}

Supervised five-fold results: `{classification.to_dict(orient='records') if not classification.empty else 'unavailable'}`

The central claim is supported only if fixed-readout materially exceeds all three TF-IDF clustering baselines on this generation-validated subset. Word-shuffle similarity indicates how much fixed-readout depends on word order rather than the preserved word bag.

## Incomplete work and failures

- Current stage: {self.state.get('current_stage')}
- Upgrade toward 400: {self.state.get('upgrade_400', 'not reached')}
- Recorded exceptions: `{json.dumps(self.state.get('exceptions', []), sort_keys=True)}`
- Facet shortages: `{(self.root / 'facet_shortages.csv').read_text() if (self.root / 'facet_shortages.csv').exists() else 'not evaluated'}`

## Resume and cached analysis

```bash
{self.state['commands']['resume']}
```

```bash
{self.state['commands']['status']}
```

```bash
{self.state['commands']['cluster_only']}
```

## Plain-language interpretation

This experiment deliberately selects text that surface TF-IDF models find difficult, then asks whether SD 1.4 can still render it and whether the fixed contextual token clusters by concept. The selection makes the comparison useful but conditional: it does not estimate performance on ordinary prose, and generation filtering introduces survivorship bias. Any incomplete facet prevents a balanced final claim.
"""
        atomic_write_text(self.root / "OVERNIGHT_REPORT.md", report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable 20-hour codex_diverse experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["run", "resume", "cluster-only"]:
        current = sub.add_parser(command)
        current.add_argument("--config", type=Path, required=True)
        current.add_argument("--pool-config", type=Path, required=True)
        current.add_argument("--root", type=Path, required=True)
        current.add_argument("--max-hours", type=float, default=20.0)
    status = sub.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "status":
        state_path = args.root / "state.json"
        heartbeat_path = args.root / "heartbeat.json"
        print(state_path.read_text() if state_path.exists() else "state.json not found")
        print(heartbeat_path.read_text() if heartbeat_path.exists() else "heartbeat.json not found")
        return
    runner = OvernightRunner(args)
    if args.command == "cluster-only":
        runner._subprocess(["analyze", "--config", str(args.config), "--output", str(args.root)], "cluster_only")
        runner.write_overview_report()
        return
    try:
        runner.run()
    except Exception as exc:
        runner._record_exception("runner", exc)
        runner.state["status"] = "failed"
        runner.state["current_stage"] = "finished"
        runner.heartbeat()
        runner.write_overview_report()
        raise


if __name__ == "__main__":
    main()
