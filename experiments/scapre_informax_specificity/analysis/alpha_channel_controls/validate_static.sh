#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
for script in "$SCRIPT_DIR"/*.sh; do
  bash -n "$script"
done
ruby -rjson -e 'JSON.parse(File.read(ARGV.fetch(0)))' "$SCRIPT_DIR/config.json"
for file in README.md implementation_audit.md config.json alpha_control_runner.py \
  evaluate_control_runner.py worker.py aggregate_results.py resolve_official_reference.sh \
  diagnose_official_reference.sh \
  run_server.sh server_worker.sh status_server.sh package_results.sh cleanup_images.sh \
  download_results.sh; do
  [[ -f "$SCRIPT_DIR/$file" ]] || { echo "ERROR: missing $file" >&2; exit 2; }
done
EXPECTED_EDITOR="$(ruby -rjson -rdigest -e 'c=JSON.parse(File.read(ARGV[0])); puts c.fetch("source_controls").fetch("scapre/edit/erase_scale.py")' "$SCRIPT_DIR/config.json")"
ACTUAL_EDITOR="$(shasum -a 256 "$REPO_ROOT/scapre/edit/erase_scale.py" | awk '{print $1}')"
[[ "$EXPECTED_EDITOR" == "$ACTUAL_EDITOR" ]] || { echo "ERROR: production editor hash drift" >&2; exit 2; }
rg -q 'caller.f_code.co_name == "edit_model"' "$SCRIPT_DIR/alpha_control_runner.py"
rg -q 'input_tensor.ndim == 3' "$SCRIPT_DIR/alpha_control_runner.py"
rg -q 'official_empty_string_neutral_only' "$SCRIPT_DIR/worker.py"
rg -q 'generated_formal_variants.*constant_mean.*shuffled.*identity_B' "$SCRIPT_DIR/config.json"
rg -q 'official_reference_.*EXPECTED_SHA:0:16' "$SCRIPT_DIR/resolve_official_reference.sh"
rg -q '\.archive_sha256' "$SCRIPT_DIR/resolve_official_reference.sh"
rg -q 'reference_manifest_equals_committed' "$SCRIPT_DIR/diagnose_official_reference.sh"
rg -q 'actual_compatibility_diff_sha256' "$SCRIPT_DIR/diagnose_official_reference.sh"
rg -q 'worker.validate_official_reference' "$SCRIPT_DIR/diagnose_official_reference.sh"
if rg -q 'RUN_DIR=.*/seed_robustness/runs|printf.*RUN_DIR' "$SCRIPT_DIR/resolve_official_reference.sh"; then
  echo "ERROR: official reference resolver must not prefer a mutable run directory" >&2
  exit 2
fi
git -C "$REPO_ROOT" diff --check
echo "Static validation passed without invoking Python or model workloads."
