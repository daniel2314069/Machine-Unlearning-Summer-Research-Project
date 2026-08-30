#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$SCRIPT_DIR/../projection_accumulation"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.json"
EDITOR="$REPO_ROOT/scapre/edit/erase_scale.py"
EXPECTED_EDITOR_SHA="cc454407a70de5b403344f8e3d0372044fed156cf78a74fa04121473674ada20"

for command_name in bash git rg ruby shasum; do
  command -v "$command_name" >/dev/null || {
    echo "ERROR: static validator requires $command_name" >&2
    exit 2
  }
done
for script in "$SCRIPT_DIR"/*.sh; do
  bash -n "$script"
done

ruby -rjson -rdigest -e '
  config = JSON.parse(File.read(ARGV.fetch(0)))
  repo = File.expand_path(ARGV.fetch(1))
  expected_seeds = [20260820, 20260821, 20260822, 20260823, 20260824]
  abort("wrong variant") unless config["variant"] == "projection_accumulation_budget_matched_cos2"
  abort("not exploratory") unless config["confirmatory_status"].start_with?("exploratory")
  formula = config.fetch("formula")
  abort("wrong alpha mode") unless formula["alpha_mode"] == "budget_matched_cos2"
  abort("wrong selected alpha") unless formula["selected_alpha"] == "lambda * projection_score"
  abort("wrong match scope") unless formula["matching_scope"] == "per-concept per-matrix contribution Frobenius norm"
  abort("lambda clamp enabled") unless formula["lambda_clamp"] == false
  abort("wrong eps") unless formula["eps"] == 1e-8
  abort("wrong rtol") unless formula["norm_match_rtol"] == 1e-5
  abort("wrong atol") unless formula["norm_match_atol"] == 1e-7
  abort("sweep enabled") unless formula["sweep"] == false
  abort("wrong seeds") unless config["edit_seeds"] == expected_seeds
  abort("wrong qualification seed") unless config["qualification_seed"] == 20260820
  abort("wrong intercept count") unless config["expected_accumulation_intercepts_per_edit"] == 320
  abort("wrong RNG call count") unless config["expected_informax_randn_calls_per_edit"] == 1280
  abort("COCO auto-launch enabled") unless config.dig("post_confuse5_contract", "automatic_coco_launch") == false
  expected_history = ["projection_accumulation", "projection_accumulation_direct_cos2"]
  abort("historical comparison set changed") unless config.fetch("historical_comparisons").keys.sort == expected_history.sort
  config.fetch("historical_comparisons").each_value do |entry|
    path = File.join(repo, entry.fetch("path"))
    abort("missing historical aggregate") unless File.file?(path)
    abort("historical hash mismatch") unless Digest::SHA256.file(path).hexdigest == entry.fetch("sha256")
  end

  eps = formula["eps"]
  official_norm = 5.0
  geo_norm = 2.0
  lambda_value = official_norm / (geo_norm + eps)
  new_norm = lambda_value * geo_norm
  tolerance = formula["norm_match_atol"] + formula["norm_match_rtol"] * official_norm.abs
  abort("synthetic budget match failed") unless (new_norm - official_norm).abs <= tolerance
' "$CONFIG" "$REPO_ROOT"

OBSERVED_EDITOR_SHA="$(shasum -a 256 "$EDITOR" | awk '{print $1}')"
[[ "$OBSERVED_EDITOR_SHA" == "$EXPECTED_EDITOR_SHA" ]] || {
  echo "ERROR: production editor hash changed: $OBSERVED_EDITOR_SHA" >&2
  exit 2
}
git -C "$REPO_ROOT" diff --exit-code -- scapre/edit/erase_scale.py >/dev/null

rg -F '"projection_accumulation_budget_matched_cos2"' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F '"budget_matched_cos2"' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'official_contribution = for_mat1 * row_w_c' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'geo_contribution = for_mat1 * direct_alpha' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'budget_lambda = official_contribution_norm / (geo_contribution_norm + args.eps)' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'budget_matched_contribution = budget_lambda * geo_contribution' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'return budget_matched_contribution' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'aggregate_row_w_max_intercepted": False' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'in_memory_source_substitution_scope": "for_mat1 * row_w_c only"' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'all_budget_norms_match_official_within_tolerance' "$SHARED_DIR/worker.py" >/dev/null
rg -F 'per_layer_concept_budget_matching.csv' "$SHARED_DIR/worker.py" >/dev/null
rg -F 'historical_variant_comparison.csv' "$SHARED_DIR/worker.py" >/dev/null
rg -F 'preflight_selftest.py' "$SCRIPT_DIR/run_budget_matched_cos2_confuse5.sh" >/dev/null

if rg -i 'run.*coco|coco.*runner' \
  "$SCRIPT_DIR/run_budget_matched_cos2_confuse5.sh" \
  "$SCRIPT_DIR/server_worker.sh" >/dev/null; then
  echo "ERROR: budget-matched Confuse5 execution scripts reference a COCO runner" >&2
  exit 2
fi
if rg -n '\bjq\b' "$SCRIPT_DIR/package_results.sh" >/dev/null; then
  echo "ERROR: package script unexpectedly depends on jq" >&2
  exit 2
fi

git -C "$REPO_ROOT" diff --check -- ':(exclude)**/*.csv'
echo "Static validation passed."
echo "Production editor SHA-256: $OBSERVED_EDITOR_SHA"
echo "No Python, model, generation, evaluation, or server job was run."
