#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARCHIVE="/home/tslin/Documents/jupyter_data/anLi/tmp/scapre_informax_seed_robustness_formal_20260821T081723Z_20260822T092030Z.tar.gz"
CONFIG="$SCRIPT_DIR/config.json"
COMMITTED_MANIFEST="$REPO_ROOT/experiments/scapre_informax_specificity/seed_robustness/reproducibility/run_manifest.json"
BASE_CONFIG="$REPO_ROOT/experiments/scapre_informax_specificity/config.json"
PROTOCOL_BUILDER="$REPO_ROOT/experiments/scapre_informax_specificity/build_protocol.py"
ASSETS="$REPO_ROOT/experiments/scapre_informax_specificity/.server/assets_manifest.json"
EXPECTED_ARCHIVE_SHA="df0874fea7c0998bbaf52782c763025c4ce7968134e8334e0688adec95453708"

[[ "${CONDA_DEFAULT_ENV:-}" == "MU" ]] || {
  echo "ERROR: run 'conda activate MU' before diagnosis" >&2
  exit 2
}
PYTHON_BIN="$(command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: python is unavailable in Conda MU" >&2; exit 2; }
for command_name in awk cmp git sha256sum tar; do
  command -v "$command_name" >/dev/null || { echo "ERROR: missing command: $command_name" >&2; exit 2; }
done
[[ -f "$ARCHIVE" ]] || { echo "ERROR: reference archive missing: $ARCHIVE" >&2; exit 2; }
[[ -f "$CONFIG" && -f "$COMMITTED_MANIFEST" && -f "$BASE_CONFIG" && \
   -f "$PROTOCOL_BUILDER" && -f "$ASSETS" ]] || {
  echo "ERROR: repository diagnostic inputs are incomplete" >&2
  exit 2
}

cd "$REPO_ROOT"
ACTUAL_ARCHIVE_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
[[ "$ACTUAL_ARCHIVE_SHA" == "$EXPECTED_ARCHIVE_SHA" ]] || {
  echo "ERROR: archive SHA-256 mismatch" >&2
  exit 2
}
REFERENCE="$($SCRIPT_DIR/resolve_official_reference.sh)"
[[ -f "$REFERENCE/run_manifest.json" ]] || {
  echo "ERROR: resolved reference manifest missing: $REFERENCE/run_manifest.json" >&2
  exit 2
}

echo "archive: $ARCHIVE"
echo "archive_sha256: $ACTUAL_ARCHIVE_SHA"
echo "resolved_reference: $REFERENCE"
echo "archive_run_manifest_members:"
tar -tzf "$ARCHIVE" | awk '$0 ~ /(^|\/)run_manifest\.json$/ {print "  " $0}'
if tar -xOzf "$ARCHIVE" run_manifest.json | cmp -s - "$REFERENCE/run_manifest.json"; then
  echo "archive_top_level_manifest_matches_resolved: true"
else
  echo "archive_top_level_manifest_matches_resolved: false"
fi

"$PYTHON_BIN" - "$CONFIG" "$REFERENCE/run_manifest.json" "$COMMITTED_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

config_path, reference_path, committed_path = map(Path, sys.argv[1:])
config = json.loads(config_path.read_text())
expected = config["official_reference"]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def report(label, path):
    manifest = json.loads(path.read_text())
    sources = manifest.get("source_sha256", {})
    print(f"{label}_manifest_path: {path}")
    print(f"{label}_manifest_sha256: {digest(path)}")
    print(f"{label}_git_commit: {manifest.get('git_commit', '<missing>')}")
    print(f"{label}_evaluator_source_sha256: {sources.get('experiments/scapre_informax_specificity/evaluate_confuse5.py', '<missing>')}")
    print(f"{label}_editor_source_sha256: {sources.get('scapre/edit/erase_scale.py', '<missing>')}")
    return manifest

print(f"expected_run_commit: {expected['run_commit']}")
print(f"expected_evaluator_source_sha256: {expected['evaluator_source_sha256']}")
print(f"expected_editor_source_sha256: {expected['editor_source_sha256']}")
print(f"expected_compatibility_diff_sha256: {expected['compatibility_diff_sha256']}")
reference = report("reference", reference_path)
committed = report("committed", committed_path)
print(f"reference_manifest_equals_committed: {str(reference == committed).lower()}")
PY

RUN_COMMIT="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["official_reference"]["run_commit"])' "$CONFIG")"
HISTORICAL_EVALUATOR_SHA="$(git show "$RUN_COMMIT:experiments/scapre_informax_specificity/evaluate_confuse5.py" | sha256sum | awk '{print $1}')"
HISTORICAL_EDITOR_SHA="$(git show "$RUN_COMMIT:scapre/edit/erase_scale.py" | sha256sum | awk '{print $1}')"
CURRENT_EVALUATOR_SHA="$(sha256sum experiments/scapre_informax_specificity/evaluate_confuse5.py | awk '{print $1}')"
CURRENT_EDITOR_SHA="$(sha256sum scapre/edit/erase_scale.py | awk '{print $1}')"
COMPATIBILITY_DIFF_SHA="$(git diff "$RUN_COMMIT..HEAD" -- scapre/edit/erase_scale.py experiments/scapre_informax_specificity/evaluate_confuse5.py | sha256sum | awk '{print $1}')"
echo "historical_evaluator_source_sha256: $HISTORICAL_EVALUATOR_SHA"
echo "historical_editor_source_sha256: $HISTORICAL_EDITOR_SHA"
echo "current_evaluator_source_sha256: $CURRENT_EVALUATOR_SHA"
echo "current_editor_source_sha256: $CURRENT_EDITOR_SHA"
echo "actual_compatibility_diff_sha256: $COMPATIBILITY_DIFF_SHA"

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
PROTOCOL="$TEMP_DIR/protocol.csv"
"$PYTHON_BIN" "$PROTOCOL_BUILDER" \
  --config "$BASE_CONFIG" --output "$PROTOCOL" --profile formal >/dev/null
"$PYTHON_BIN" - "$SCRIPT_DIR/worker.py" "$CONFIG" "$ASSETS" "$PROTOCOL" "$REFERENCE" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

worker_path, config_path, assets_path, protocol_path, reference_path = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("alpha_controls_worker_preflight", worker_path)
worker = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(worker)
diagnostic_config = json.loads(config_path.read_text())
diagnostic_manifest = json.loads((reference_path / "run_manifest.json").read_text())
diagnostic_key = "experiments/scapre_informax_specificity/evaluate_confuse5.py"
diagnostic_actual = diagnostic_manifest["source_sha256"].get(diagnostic_key)
diagnostic_expected = diagnostic_config["official_reference"]["evaluator_source_sha256"]
print(f"precall_evaluator_actual_repr: {diagnostic_actual!r}")
print(f"precall_evaluator_expected_repr: {diagnostic_expected!r}")
print(f"precall_evaluator_values_equal: {str(diagnostic_actual == diagnostic_expected).lower()}")
print(f"precall_evaluator_actual_length: {len(diagnostic_actual)}")
print(f"precall_evaluator_expected_length: {len(diagnostic_expected)}")
result = worker.validate_official_reference(
    reference_path.resolve(),
    diagnostic_config,
    json.loads(assets_path.read_text()),
    protocol_path,
)
print(f"exact_worker_reference_validation: {result['status']}")
print(f"exact_worker_evaluator_fingerprint_sha256: {result['evaluator_fingerprint_sha256']}")
print(f"exact_worker_validated_seeds: {','.join(result['seeds'])}")
PY
trap - EXIT
rm -rf -- "$TEMP_DIR"
echo "diagnosis_complete: true"
