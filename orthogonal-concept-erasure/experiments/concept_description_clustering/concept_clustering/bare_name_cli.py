from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated bare OCE/UCE-name versus cached description-subspace analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("extract", "Extract four bare-name vectors with the exact OCE/UCE token rule"),
        ("analyze", "Run raw and all original to_v subspace analyses from caches"),
        ("stats", "Rerun only statistics, plots, and report from cached vectors"),
    ]:
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--source-output", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if command == "extract":
            child.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    source_required = [
        args.source_output / "accepted_descriptions.jsonl",
        args.source_output / "raw_text_embeddings.pt",
        args.source_output / "layer_embeddings.pt",
    ]
    missing = [str(path) for path in source_required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required immutable source caches are missing: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "extract":
        from .oce_uce_bare import extract_bare_name_embeddings

        extract_bare_name_embeddings(
            config, args.source_output, args.output, force=bool(args.force)
        )
        return

    cache_required = [
        args.output / "bare_name_embeddings.pt",
        args.output / "bare_name_tokenization_audit.csv",
    ]
    missing = [str(path) for path in cache_required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run the extract command first; missing caches: {missing}")
    from .bare_name_subspace import run_bare_name_subspace_analysis

    run_bare_name_subspace_analysis(config, args.source_output, args.output)


if __name__ == "__main__":
    main()
