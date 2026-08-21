#!/usr/bin/env python
"""Small dependency-free JSON and SHA-256 helper for server shell scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: Any, dotted_path: str) -> Any:
    for component in dotted_path.split("."):
        if not component:
            continue
        if isinstance(value, list):
            value = value[int(component)]
        else:
            value = value[component]
    return value


def print_value(value: Any) -> None:
    if isinstance(value, bool):
        print("true" if value else "false")
    elif value is None:
        print("null")
    elif isinstance(value, (dict, list)):
        print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    else:
        print(value)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str, value: Any) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def command_get(args: argparse.Namespace) -> None:
    print_value(resolve_path(load_json(args.file), args.path))


def command_lines(args: argparse.Namespace) -> None:
    values = resolve_path(load_json(args.file), args.path)
    if not isinstance(values, list):
        raise TypeError(f"{args.path} is not a list")
    for value in values:
        if isinstance(value, (dict, list)):
            print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        else:
            print_value(value)


def command_keys(args: argparse.Namespace) -> None:
    value = resolve_path(load_json(args.file), args.path)
    if not isinstance(value, dict):
        raise TypeError(f"{args.path} is not an object")
    for key in sorted(value):
        print(key)


def command_sha256(args: argparse.Namespace) -> None:
    print(sha256_file(args.file))


def command_archive_manifest(args: argparse.Namespace) -> None:
    write_json(
        args.output,
        {
            "archive": args.archive,
            "created_at_utc": args.created_at_utc,
            "profile": args.profile,
            "sha256": args.sha256,
            "size_bytes": args.size_bytes,
            "verified": True,
        },
    )


def command_cleanup_manifest(args: argparse.Namespace) -> None:
    records = []
    with Path(args.records_tsv).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                raise ValueError(
                    f"cleanup record line {line_number} has {len(fields)} fields, expected 5"
                )
            status, path, detail, deleted_files, deleted_bytes = fields
            record = {
                "path": path,
                "status": status,
                "deleted_files": int(deleted_files),
                "deleted_bytes": int(deleted_bytes),
            }
            record["reason" if status == "skipped" else "context"] = detail
            records.append(record)
    write_json(
        args.output,
        {
            "archive": args.archive,
            "archive_sha256": args.archive_sha256,
            "completed_at_utc": args.completed_at_utc,
            "deleted_bytes": args.deleted_bytes,
            "deleted_files": args.deleted_files,
            "profile": args.profile,
            "records": records,
            "status": "passed",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("file")
    get_parser.add_argument("path")
    get_parser.set_defaults(function=command_get)

    lines_parser = subparsers.add_parser("lines")
    lines_parser.add_argument("file")
    lines_parser.add_argument("path")
    lines_parser.set_defaults(function=command_lines)

    keys_parser = subparsers.add_parser("keys")
    keys_parser.add_argument("file")
    keys_parser.add_argument("path")
    keys_parser.set_defaults(function=command_keys)

    sha_parser = subparsers.add_parser("sha256")
    sha_parser.add_argument("file")
    sha_parser.set_defaults(function=command_sha256)

    archive_parser = subparsers.add_parser("archive-manifest")
    archive_parser.add_argument("output")
    archive_parser.add_argument("archive")
    archive_parser.add_argument("sha256")
    archive_parser.add_argument("size_bytes", type=int)
    archive_parser.add_argument("profile")
    archive_parser.add_argument("created_at_utc")
    archive_parser.set_defaults(function=command_archive_manifest)

    cleanup_parser = subparsers.add_parser("cleanup-manifest")
    cleanup_parser.add_argument("output")
    cleanup_parser.add_argument("profile")
    cleanup_parser.add_argument("archive")
    cleanup_parser.add_argument("archive_sha256")
    cleanup_parser.add_argument("completed_at_utc")
    cleanup_parser.add_argument("deleted_files", type=int)
    cleanup_parser.add_argument("deleted_bytes", type=int)
    cleanup_parser.add_argument("records_tsv")
    cleanup_parser.set_defaults(function=command_cleanup_manifest)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
