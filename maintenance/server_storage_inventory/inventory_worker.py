#!/usr/bin/env python3
"""Read-only, full-tree storage inventory for the GPU-server repository."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import heapq
import json
import os
import shutil
import stat
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TOP_LIMIT = 5000
PROGRESS_INTERVAL_SECONDS = 5.0
EXPECTED_SERVER_ROOT = Path(
    "/home/tslin/Documents/jupyter_data/anLi/machine_unlearning"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory every entry under the repository without deleting anything."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-branch", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def iso_mtime(timestamp: float) -> str:
    try:
        return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ancestors(relative_directory: str) -> Iterable[str]:
    current = relative_directory or "."
    while True:
        yield current
        if current == ".":
            break
        parent = os.path.dirname(current)
        current = parent if parent else "."


def extension_key(relative_path: str) -> str:
    name = Path(relative_path).name.lower()
    for compound in (".tar.gz", ".tar.xz", ".tar.bz2", ".jsonl.gz", ".csv.gz", ".tsv.gz"):
        if name.endswith(compound):
            return compound
    suffix = Path(name).suffix
    return suffix if suffix else "[no extension]"


def storage_category(relative_path: str, entry_type: str) -> str:
    lower = relative_path.lower()
    parts = set(Path(lower).parts)
    suffix = extension_key(lower)

    if ".git" in parts:
        return "git_internal"
    if ".local_artifacts" in parts:
        return "local_artifacts"
    if entry_type == "symlink":
        return "symlinks"
    if "__pycache__" in parts or suffix in {".pyc", ".pyo"}:
        return "python_cache"
    if parts.intersection({"checkpoint", "checkpoints"}) or suffix in {
        ".ckpt",
        ".safetensors",
        ".pth",
        ".pt",
        ".bin",
    }:
        return "checkpoints_or_model_weights"
    if parts.intersection({"image", "images", "imgs", "samples", "generations"}) or suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }:
        return "images"
    if parts.intersection({"cache", ".cache", "caches"}):
        return "cache"
    if parts.intersection({"logs", "log"}) or suffix == ".log":
        return "logs"
    if suffix in {".zip", ".gz", ".tgz", ".xz", ".bz2", ".7z", ".rar", ".tar.gz", ".tar.xz", ".tar.bz2"}:
        return "archives"
    if parts.intersection({"runs", "outputs", "output", ".server"}):
        return "experiment_run_outputs"
    if parts.intersection({"results", "reports", "evaluation", "evaluations", "metrics"}):
        return "results_reports_evaluation"
    if suffix in {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".npy", ".npz", ".pkl", ".pickle", ".csv.gz", ".tsv.gz", ".jsonl.gz"}:
        return "data_and_metadata"
    if suffix in {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "source_and_documentation"
    return "other"


def age_bucket(mtime: float, now: float) -> str:
    days = max(0.0, (now - mtime) / 86400.0)
    if days < 7:
        return "<7 days"
    if days < 30:
        return "7-29 days"
    if days < 90:
        return "30-89 days"
    if days < 180:
        return "90-179 days"
    if days < 365:
        return "180-364 days"
    return ">=365 days"


def entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "other"


def empty_stat() -> dict[str, int]:
    return {
        "entry_count": 0,
        "file_count": 0,
        "symlink_count": 0,
        "other_count": 0,
        "logical_bytes": 0,
        "allocated_bytes_charged": 0,
    }


def add_stat(bucket: dict[str, int], kind: str, logical: int, allocated: int) -> None:
    bucket["entry_count"] += 1
    if kind == "file":
        bucket["file_count"] += 1
    elif kind == "symlink":
        bucket["symlink_count"] += 1
    else:
        bucket["other_count"] += 1
    bucket["logical_bytes"] += logical
    bucket["allocated_bytes_charged"] += allocated


def push_largest(heap: list[tuple[int, int, str, str, float]], item: tuple[int, int, str, str, float]) -> None:
    if len(heap) < TOP_LIMIT:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def human_bytes(value: int | float) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(number) < 1024.0 or unit == "PiB":
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{number:.2f} PiB"


def markdown_cell(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("|", "\\|")
        .replace("`", "\\`")
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=True)

    if root != EXPECTED_SERVER_ROOT:
        raise RuntimeError(f"refusing unexpected scan root: {root}")
    if os.path.commonpath([str(root), str(output_dir)]) == str(root):
        raise RuntimeError("output directory must be outside the scanned root")

    started_at = utc_now()
    started_monotonic = time.monotonic()
    now_epoch = time.time()
    filesystem_start = shutil.disk_usage(root)
    progress_path = output_dir / "progress.json"
    stage_path = output_dir / "stage"
    stage_path.write_text("scanning\n", encoding="utf-8")

    all_files_path = output_dir / "all_files.tsv.gz"
    errors_path = output_dir / "scan_errors.csv"
    errors_handle = errors_path.open("w", newline="", encoding="utf-8")
    errors_writer = csv.DictWriter(
        errors_handle, fieldnames=["operation", "relative_path", "error_type", "message"]
    )
    errors_writer.writeheader()

    files_handle = gzip.open(all_files_path, "wt", newline="", encoding="utf-8")
    files_writer = csv.DictWriter(
        files_handle,
        delimiter="\t",
        quoting=csv.QUOTE_MINIMAL,
        fieldnames=[
            "entry_type",
            "logical_size_bytes",
            "allocated_size_bytes",
            "allocated_size_bytes_charged",
            "mtime_utc",
            "mode_octal",
            "hardlink_count",
            "storage_category",
            "relative_path",
            "symlink_target",
        ],
    )
    files_writer.writeheader()

    directory_stats: dict[str, dict[str, int]] = defaultdict(empty_stat)
    directory_direct: dict[str, dict[str, int]] = defaultdict(empty_stat)
    root_children: dict[str, dict[str, int]] = defaultdict(empty_stat)
    extension_stats: dict[str, dict[str, int]] = defaultdict(empty_stat)
    category_stats: dict[str, dict[str, int]] = defaultdict(empty_stat)
    age_stats: dict[str, dict[str, int]] = defaultdict(empty_stat)
    largest: list[tuple[int, int, str, str, float]] = []
    seen_multilink_inodes: set[tuple[int, int]] = set()

    counts = {
        "directories": 0,
        "files": 0,
        "symlinks": 0,
        "other_entries": 0,
        "errors": 0,
        "logical_bytes": 0,
        "allocated_file_bytes_charged": 0,
        "directory_metadata_allocated_bytes": 0,
        "hardlink_paths_not_charged": 0,
    }
    stack: list[Path] = [root]
    last_progress = 0.0
    current_relative = "."

    def record_error(operation: str, path: Path, exc: BaseException) -> None:
        counts["errors"] += 1
        try:
            relative = os.path.relpath(path, root)
        except ValueError:
            relative = str(path)
        errors_writer.writerow(
            {
                "operation": operation,
                "relative_path": relative,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )

    def write_progress(force: bool = False) -> None:
        nonlocal last_progress
        current = time.monotonic()
        if not force and current - last_progress < PROGRESS_INTERVAL_SECONDS:
            return
        last_progress = current
        atomic_json(
            progress_path,
            {
                "status": "running",
                "stage": "scanning",
                "updated_at_utc": utc_now(),
                "elapsed_seconds": current - started_monotonic,
                "current_relative_path": current_relative,
                "directories_scanned": counts["directories"],
                "entries_recorded": counts["files"] + counts["symlinks"] + counts["other_entries"],
                "regular_files": counts["files"],
                "logical_bytes_seen": counts["logical_bytes"],
                "allocated_bytes_charged_seen": counts["allocated_file_bytes_charged"],
                "errors": counts["errors"],
            },
        )

    while stack:
        directory = stack.pop()
        current_relative = os.path.relpath(directory, root)
        try:
            directory_lstat = directory.lstat()
            directory_allocated = int(getattr(directory_lstat, "st_blocks", 0)) * 512
            counts["directory_metadata_allocated_bytes"] += directory_allocated
            directory_stats[current_relative]
            directory_direct[current_relative]
            for parent in ancestors(current_relative):
                directory_stats[parent]["allocated_bytes_charged"] += directory_allocated
        except (OSError, PermissionError) as exc:
            record_error("lstat_directory", directory, exc)
            write_progress()
            continue

        counts["directories"] += 1
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except (OSError, PermissionError) as exc:
            record_error("scandir", directory, exc)
            write_progress()
            continue

        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except (OSError, PermissionError) as exc:
                record_error("lstat_entry", path, exc)
                continue

            kind = entry_type(info.st_mode)
            relative = os.path.relpath(path, root)
            if kind == "directory":
                stack.append(path)
                continue

            logical = int(info.st_size)
            allocated = int(getattr(info, "st_blocks", 0)) * 512
            charged = allocated
            if info.st_nlink > 1:
                inode_key = (int(info.st_dev), int(info.st_ino))
                if inode_key in seen_multilink_inodes:
                    charged = 0
                    counts["hardlink_paths_not_charged"] += 1
                else:
                    seen_multilink_inodes.add(inode_key)

            if kind == "file":
                counts["files"] += 1
            elif kind == "symlink":
                counts["symlinks"] += 1
            else:
                counts["other_entries"] += 1
            counts["logical_bytes"] += logical
            counts["allocated_file_bytes_charged"] += charged

            category = storage_category(relative, kind)
            suffix = extension_key(relative)
            bucket = age_bucket(info.st_mtime, now_epoch)
            parent_directory = os.path.dirname(relative) or "."
            first_component = Path(relative).parts[0]

            add_stat(directory_direct[parent_directory], kind, logical, charged)
            for parent in ancestors(parent_directory):
                add_stat(directory_stats[parent], kind, logical, charged)
            add_stat(root_children[first_component], kind, logical, charged)
            add_stat(extension_stats[suffix], kind, logical, charged)
            add_stat(category_stats[category], kind, logical, charged)
            add_stat(age_stats[bucket], kind, logical, charged)

            symlink_target = ""
            if kind == "symlink":
                try:
                    symlink_target = os.readlink(path)
                except OSError as exc:
                    record_error("readlink", path, exc)

            files_writer.writerow(
                {
                    "entry_type": kind,
                    "logical_size_bytes": logical,
                    "allocated_size_bytes": allocated,
                    "allocated_size_bytes_charged": charged,
                    "mtime_utc": iso_mtime(info.st_mtime),
                    "mode_octal": oct(stat.S_IMODE(info.st_mode)),
                    "hardlink_count": info.st_nlink,
                    "storage_category": category,
                    "relative_path": relative,
                    "symlink_target": symlink_target,
                }
            )
            push_largest(largest, (charged, logical, relative, category, info.st_mtime))

        write_progress()

    files_handle.close()
    errors_handle.close()
    write_progress(force=True)
    stage_path.write_text("summarizing\n", encoding="utf-8")

    directory_rows = []
    for relative, inclusive in directory_stats.items():
        direct = directory_direct[relative]
        directory_rows.append(
            {
                "relative_directory": relative,
                "inclusive_entry_count": inclusive["entry_count"],
                "inclusive_file_count": inclusive["file_count"],
                "inclusive_symlink_count": inclusive["symlink_count"],
                "inclusive_logical_bytes": inclusive["logical_bytes"],
                "inclusive_allocated_bytes_charged": inclusive["allocated_bytes_charged"],
                "direct_entry_count": direct["entry_count"],
                "direct_logical_bytes": direct["logical_bytes"],
                "direct_allocated_bytes_charged": direct["allocated_bytes_charged"],
            }
        )
    directory_rows.sort(
        key=lambda row: (int(row["inclusive_allocated_bytes_charged"]), row["relative_directory"]),
        reverse=True,
    )

    directory_fields = [
        "relative_directory",
        "inclusive_entry_count",
        "inclusive_file_count",
        "inclusive_symlink_count",
        "inclusive_logical_bytes",
        "inclusive_allocated_bytes_charged",
        "direct_entry_count",
        "direct_logical_bytes",
        "direct_allocated_bytes_charged",
    ]
    with gzip.open(
        output_dir / "all_directories.tsv.gz", "wt", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=directory_fields)
        writer.writeheader()
        writer.writerows(directory_rows)
    write_csv(output_dir / "largest_directories.csv", directory_fields, directory_rows[:TOP_LIMIT])

    largest_rows = [
        {
            "relative_path": relative,
            "entry_type": "file_or_non_directory",
            "storage_category": category,
            "logical_size_bytes": logical,
            "allocated_size_bytes_charged": allocated,
            "mtime_utc": iso_mtime(mtime),
        }
        for allocated, logical, relative, category, mtime in sorted(largest, reverse=True)
    ]
    write_csv(
        output_dir / "largest_files.csv",
        [
            "relative_path",
            "entry_type",
            "storage_category",
            "logical_size_bytes",
            "allocated_size_bytes_charged",
            "mtime_utc",
        ],
        largest_rows,
    )

    summary_fields = [
        "name",
        "entry_count",
        "file_count",
        "symlink_count",
        "other_count",
        "logical_bytes",
        "allocated_bytes_charged",
    ]

    def named_rows(mapping: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
        rows = [{"name": name, **values} for name, values in mapping.items()]
        return sorted(
            rows,
            key=lambda row: (int(row["allocated_bytes_charged"]), row["name"]),
            reverse=True,
        )

    write_csv(output_dir / "root_children.csv", summary_fields, named_rows(root_children))
    write_csv(output_dir / "extension_summary.csv", summary_fields, named_rows(extension_stats))
    write_csv(output_dir / "category_summary.csv", summary_fields, named_rows(category_stats))
    write_csv(output_dir / "age_summary.csv", summary_fields, named_rows(age_stats))

    filesystem_end = shutil.disk_usage(root)
    finished_at = utc_now()
    duration_seconds = time.monotonic() - started_monotonic
    total_charged = (
        counts["allocated_file_bytes_charged"]
        + counts["directory_metadata_allocated_bytes"]
    )
    summary = {
        "status": "passed" if counts["errors"] == 0 else "failed",
        "read_only": True,
        "root": str(root),
        "output_directory": str(output_dir),
        "run_id": args.run_id,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_seconds": duration_seconds,
        "git_commit": args.git_commit,
        "git_branch": args.git_branch,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "hidden_entries_included": True,
        "symlinks_followed": False,
        "deletion_performed": False,
        "counts": counts,
        "total_allocated_bytes_charged_including_directory_metadata": total_charged,
        "filesystem_at_start": {
            "total_bytes": filesystem_start.total,
            "used_bytes": filesystem_start.used,
            "free_bytes": filesystem_start.free,
        },
        "filesystem_at_end": {
            "total_bytes": filesystem_end.total,
            "used_bytes": filesystem_end.used,
            "free_bytes": filesystem_end.free,
        },
        "completeness": {
            "scan_errors": counts["errors"],
            "complete": counts["errors"] == 0,
            "error_details": "scan_errors.csv",
        },
    }
    atomic_json(output_dir / "summary.json", summary)

    top_directories = [row for row in directory_rows if row["relative_directory"] != "."][:30]
    top_files = largest_rows[:30]
    markdown = [
        "# Machine-unlearning server storage inventory",
        "",
        f"- Status: **{summary['status']}**",
        f"- Root: `{root}`",
        f"- Run: `{args.run_id}`",
        f"- Files: `{counts['files']:,}`; symlinks: `{counts['symlinks']:,}`; directories: `{counts['directories']:,}`",
        f"- Logical file bytes: `{human_bytes(counts['logical_bytes'])}`",
        f"- Charged allocated bytes including directory metadata: `{human_bytes(total_charged)}`",
        f"- Scan errors: `{counts['errors']}`",
        "- Hidden entries were included; symlinks were recorded but not followed.",
        "- This inventory did not delete or modify any scanned file.",
        "",
        "## Largest directories by charged allocated bytes",
        "",
        "| Directory | Allocated | Logical | Entries |",
        "| --- | ---: | ---: | ---: |",
    ]
    markdown.extend(
        f"| `{markdown_cell(row['relative_directory'])}` | {human_bytes(int(row['inclusive_allocated_bytes_charged']))} | {human_bytes(int(row['inclusive_logical_bytes']))} | {int(row['inclusive_entry_count']):,} |"
        for row in top_directories
    )
    markdown.extend(
        [
            "",
            "## Largest files by charged allocated bytes",
            "",
            "| File | Category | Allocated | Logical |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    markdown.extend(
        f"| `{markdown_cell(row['relative_path'])}` | {markdown_cell(row['storage_category'])} | {human_bytes(int(row['allocated_size_bytes_charged']))} | {human_bytes(int(row['logical_size_bytes']))} |"
        for row in top_files
    )
    markdown.extend(
        [
            "",
            "## Review notes",
            "",
            "No path is labeled safe-to-delete by this scanner. Deletion decisions require reviewing reproducibility, Git tracking, active references, and whether an artifact is regenerable.",
            "The complete per-entry inventory is `all_files.tsv.gz`; the complete per-directory rollup is `all_directories.tsv.gz`.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    result_files = [
        "age_summary.csv",
        "all_directories.tsv.gz",
        "all_files.tsv.gz",
        "category_summary.csv",
        "extension_summary.csv",
        "largest_directories.csv",
        "largest_files.csv",
        "root_children.csv",
        "scan_errors.csv",
        "summary.json",
        "summary.md",
    ]
    atomic_json(
        output_dir / "result_manifest.json",
        {
            "status": "passed" if counts["errors"] == 0 else "failed",
            "algorithm": "sha256",
            "files": {
                name: {
                    "size_bytes": (output_dir / name).stat().st_size,
                    "sha256": sha256(output_dir / name),
                }
                for name in result_files
            },
        },
    )

    final_status = "completed" if counts["errors"] == 0 else "failed"
    stage_path.write_text(final_status + "\n", encoding="utf-8")
    atomic_json(
        progress_path,
        {
            "status": final_status,
            "stage": final_status,
            "updated_at_utc": utc_now(),
            "elapsed_seconds": duration_seconds,
            "directories_scanned": counts["directories"],
            "entries_recorded": counts["files"] + counts["symlinks"] + counts["other_entries"],
            "regular_files": counts["files"],
            "logical_bytes_seen": counts["logical_bytes"],
            "allocated_bytes_charged_seen": total_charged,
            "errors": counts["errors"],
        },
    )

    print(f"Inventory status: {summary['status']}", flush=True)
    print(f"Files: {counts['files']:,}", flush=True)
    print(f"Directories: {counts['directories']:,}", flush=True)
    print(f"Allocated (charged): {human_bytes(total_charged)}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    if counts["errors"]:
        print(f"Refusing success: {counts['errors']} scan errors; see scan_errors.csv", flush=True)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
