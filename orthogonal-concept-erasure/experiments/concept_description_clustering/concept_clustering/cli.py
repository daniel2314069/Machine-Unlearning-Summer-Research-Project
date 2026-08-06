from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .text_validation import import_generated_responses, prepare_generation_requests, validate_candidates
from .utils import atomic_write_text


def _common(parser):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _resolved(args):
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.output / "resolved_config.json", json.dumps(config, indent=2) + "\n")
    return config


def build_parser():
    parser = argparse.ArgumentParser(description="Original-SD1.4 concept-description clustering")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-prompts", help="Create optional LLM request JSONL; no API is called")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-file", type=Path, required=True)

    p = sub.add_parser("import-candidates", help="Import externally generated JSONL responses")
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--output-file", type=Path, required=True)
    p.add_argument("--source", default="optional_llm")

    p = sub.add_parser("validate-text")
    _common(p)
    p.add_argument("--candidates", type=Path, required=True)

    p = sub.add_parser("validate-generation")
    _common(p)
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    p = sub.add_parser("finalize")
    _common(p)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("extract")
    _common(p)
    p.add_argument("--projection", choices=["to_v", "to_k"], default="to_v")
    p.add_argument("--suffix-name", action="append", dest="suffix_names")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("cluster")
    _common(p)

    p = sub.add_parser("report")
    _common(p)

    p = sub.add_parser("smoke", help="Text validation plus one-seed Stage-1 generation smoke test")
    _common(p)
    p.add_argument("--candidates", type=Path, required=True)

    p = sub.add_parser("run", help="Run formal pipeline from candidates through report")
    _common(p)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--projection", choices=["to_v", "to_k"], default="to_v")
    p.add_argument("--suffix-name", action="append", dest="suffix_names")
    p.add_argument("--reuse-embeddings", action="store_true")
    p.add_argument("--force-accepted", action="store_true")
    p.add_argument("--force-embeddings", action="store_true")

    p = sub.add_parser("analyze", help="Rerun clustering/report from cached accepted data and embeddings")
    _common(p)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare-prompts":
        prepare_generation_requests(load_config(args.config), args.output_file)
        return
    if args.command == "import-candidates":
        import_generated_responses(args.responses, args.output_file, args.source)
        return

    config = _resolved(args)
    if args.command == "validate-text":
        validate_candidates(config, args.candidates, args.output)
    elif args.command == "validate-generation":
        from .generation_validation import run_generation_validation
        completed = run_generation_validation(config, args.output, stage=args.stage, resume=args.resume)
        if not completed:
            raise SystemExit(75)
    elif args.command == "finalize":
        from .generation_validation import finalize_accepted
        finalize_accepted(config, args.output, force=args.force)
    elif args.command == "extract":
        from .embeddings import extract_embeddings
        extract_embeddings(config, args.output, args.projection, args.suffix_names, args.force)
    elif args.command == "cluster":
        from .clustering import run_clustering
        run_clustering(config, args.output)
    elif args.command == "report":
        from .reporting import build_report
        build_report(config, args.output)
    elif args.command == "smoke":
        from .generation_validation import run_generation_validation
        validate_candidates(config, args.candidates, args.output)
        run_generation_validation(config, args.output, stage="1", resume=True)
    elif args.command == "run":
        from .clustering import run_clustering
        from .embeddings import extract_embeddings
        from .generation_validation import finalize_accepted, run_generation_validation
        from .reporting import build_report
        validate_candidates(config, args.candidates, args.output)
        run_generation_validation(config, args.output, stage="all", resume=True)
        finalize_accepted(config, args.output, force=args.force_accepted)
        raw_cache = args.output / "raw_text_embeddings.pt"
        layer_cache = args.output / "layer_embeddings.pt"
        if not (args.reuse_embeddings and raw_cache.exists() and layer_cache.exists()):
            extract_embeddings(config, args.output, args.projection, args.suffix_names, args.force_embeddings)
        run_clustering(config, args.output)
        build_report(config, args.output)
    elif args.command == "analyze":
        from .clustering import run_clustering
        from .reporting import build_report
        for path in [args.output / "accepted_descriptions.jsonl", args.output / "raw_text_embeddings.pt", args.output / "layer_embeddings.pt"]:
            if not path.exists():
                raise FileNotFoundError(f"Cached analysis input missing: {path}")
        run_clustering(config, args.output)
        build_report(config, args.output)


if __name__ == "__main__":
    main()
