from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "registry.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_fingerprint(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"schema_version": 1, "references": {}}
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported evaluation reference registry schema")
    return payload


def write_registry(payload: Mapping[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = REGISTRY_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REGISTRY_PATH)


def upsert_reference(
    reference_id: str,
    identity: Mapping[str, Any],
    status: str,
    artifacts: Mapping[str, Any],
    producer: str,
) -> dict[str, Any]:
    if status not in {"building", "complete", "invalid"}:
        raise ValueError(f"Unsupported reference status: {status}")
    registry = load_registry()
    existing = registry["references"].get(reference_id)
    fingerprint = canonical_fingerprint(identity)
    if existing is not None and existing["fingerprint"] != fingerprint:
        raise RuntimeError(
            f"Reference id {reference_id!r} already has a different protocol"
        )
    entry = {
        "reference_id": reference_id,
        "status": status,
        "identity": dict(identity),
        "fingerprint": fingerprint,
        "artifacts": dict(artifacts),
        "producer": producer,
        "updated_at": utc_now(),
    }
    if existing is not None and "created_at" in existing:
        entry["created_at"] = existing["created_at"]
    else:
        entry["created_at"] = entry["updated_at"]
    registry["references"][reference_id] = entry
    write_registry(registry)
    return entry


def resolve_reference(
    reference_id: str,
    expected_identity: Mapping[str, Any],
    require_complete: bool = True,
) -> dict[str, Any] | None:
    entry = load_registry()["references"].get(reference_id)
    if entry is None:
        return None
    expected_fingerprint = canonical_fingerprint(expected_identity)
    if entry["fingerprint"] != expected_fingerprint:
        raise RuntimeError(
            f"Reference {reference_id!r} exists but its protocol fingerprint "
            "does not match the requested evaluation"
        )
    if require_complete and entry["status"] != "complete":
        return None
    for label, raw_path in entry["artifacts"].items():
        if label.endswith("_glob"):
            matches = list(Path(raw_path).parent.glob(Path(raw_path).name))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Reference artifact {label!r} expected exactly one file, "
                    f"found {len(matches)}"
                )
        elif not Path(raw_path).is_file():
            raise RuntimeError(
                f"Reference artifact {label!r} is missing: {raw_path}"
            )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect repository-level reusable evaluation references."
    )
    parser.add_argument(
        "command", choices=["list", "show"], nargs="?", default="list"
    )
    parser.add_argument("--reference-id")
    args = parser.parse_args()
    registry = load_registry()
    if args.command == "list":
        rows = [
            {
                "reference_id": key,
                "status": value["status"],
                "fingerprint": value["fingerprint"],
            }
            for key, value in registry["references"].items()
        ]
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not args.reference_id:
        parser.error("--reference-id is required for show")
    entry = registry["references"].get(args.reference_id)
    if entry is None:
        raise SystemExit(f"Unknown reference: {args.reference_id}")
    print(json.dumps(entry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
