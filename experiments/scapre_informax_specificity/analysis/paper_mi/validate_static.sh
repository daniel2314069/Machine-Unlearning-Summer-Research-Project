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
FORMAL_RESULTS="$SCRIPT_DIR/formal_results"
if [[ -d "$FORMAL_RESULTS" ]]; then
  for result_json in "$FORMAL_RESULTS"/*.json; do
    ruby -rjson -e 'JSON.parse(File.read(ARGV.fetch(0)))' "$result_json"
  done
  ruby -rcsv -rjson -e '
    deltas = CSV.read(ARGV.fetch(0), headers: true)
    result = JSON.parse(File.read(ARGV.fetch(1)))
    abort "expected five paired seeds" unless deltas.length == 5
    result.fetch("mean_delta_paper_minus_repository").each do |name, expected|
      actual = deltas.sum { |row| row.fetch(name).to_f } / deltas.length
      abort "formal mean mismatch: #{name}" unless (actual - expected).abs < 1e-10
    end
  ' "$FORMAL_RESULTS/comparison_deltas.csv" "$FORMAL_RESULTS/result_manifest.json"
  [[ "$(($(wc -l < "$FORMAL_RESULTS/per_seed_metrics.csv") - 1))" -eq 10 ]]
  [[ "$(($(wc -l < "$FORMAL_RESULTS/per_concept_mean_metrics.csv") - 1))" -eq 25 ]]
fi
echo "paper-MI static validation passed (no Python/model execution)"
