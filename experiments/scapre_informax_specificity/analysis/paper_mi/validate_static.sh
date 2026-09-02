#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
for script in "$SCRIPT_DIR"/*.sh; do
  bash -n "$script"
done
ruby -rjson -e 'JSON.parse(File.read(ARGV.fetch(0)))' "$SCRIPT_DIR/config.json"
EXPECTED_EDITOR="$(ruby -rjson -e 'puts JSON.parse(File.read(ARGV.fetch(0))).fetch("source_controls").fetch("scapre/edit/erase_scale.py")' "$SCRIPT_DIR/config.json")"
EXPECTED_EVALUATOR="$(ruby -rjson -e 'puts JSON.parse(File.read(ARGV.fetch(0))).fetch("source_controls").fetch("experiments/scapre_informax_specificity/evaluate_confuse5.py")' "$SCRIPT_DIR/config.json")"
[[ "$(shasum -a 256 "$REPO_ROOT/scapre/edit/erase_scale.py" | awk '{print $1}')" == "$EXPECTED_EDITOR" ]]
[[ "$(shasum -a 256 "$REPO_ROOT/experiments/scapre_informax_specificity/evaluate_confuse5.py" | awk '{print $1}')" == "$EXPECTED_EVALUATOR" ]]
rg -q "informax_weighting_mode == 'paper'" "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q "concept_max / channel_max" "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q "informax_weighting_mode == 'repository'" "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q '"parameter_search": false' "$SCRIPT_DIR/config.json"
rg -q 'setup_server.sh' "$SCRIPT_DIR/ensure_assets.sh"
rg -q 'repository baseline will be regenerated' "$SCRIPT_DIR/resolve_official_reference.sh"
echo "paper-MI static validation passed (no Python/model execution)"
