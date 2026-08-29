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

ruby -rjson -e '
  c = JSON.parse(File.read(ARGV.fetch(0)))
  expected_seeds = [20260820, 20260821, 20260822, 20260823, 20260824]
  abort("wrong variant") unless c["variant"] == "projection_accumulation_direct_cos2"
  abort("not exploratory") unless c["confirmatory_status"].start_with?("exploratory")
  abort("wrong formula") unless c.dig("formula", "alpha_mode") == "direct_cos2"
  abort("wrong selected alpha") unless c.dig("formula", "selected_alpha") == "projection_score"
  abort("normalization present") unless c.dig("formula", "normalization") == "none"
  abort("wrong eps") unless c.dig("formula", "eps") == 1e-8
  abort("sweep enabled") unless c.dig("formula", "sweep") == false
  abort("wrong seeds") unless c["edit_seeds"] == expected_seeds
  abort("wrong qualification seed") unless c["qualification_seed"] == 20260820
  abort("wrong intercept count") unless c["expected_accumulation_intercepts_per_edit"] == 320
  abort("wrong RNG call count") unless c["expected_informax_randn_calls_per_edit"] == 1280
  abort("V1 analysis disabled") unless c["requires_v1_diagnostic_analysis"] == true
  abort("COCO auto-launch enabled") unless c.dig("post_confuse5_contract", "automatic_coco_launch") == false
' "$CONFIG"

OBSERVED_EDITOR_SHA="$(shasum -a 256 "$EDITOR" | awk '{print $1}')"
[[ "$OBSERVED_EDITOR_SHA" == "$EXPECTED_EDITOR_SHA" ]] || {
  echo "ERROR: production editor hash changed: $OBSERVED_EDITOR_SHA" >&2
  exit 2
}
git -C "$REPO_ROOT" diff --exit-code -- scapre/edit/erase_scale.py >/dev/null

rg -F 'd_vec = c_vec - empty_vec' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'else direct_alpha' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F '"projection_score" if args.alpha_mode == "direct_cos2"' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'aggregate_row_w_max_intercepted": False' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'in_memory_source_substitution_scope": "for_mat1 * row_w_c only"' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F '"weighted_contribution_stats"' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F '"V_stats"' "$SHARED_DIR/projection_runner.py" >/dev/null
rg -F 'checkpoint_parameter_delta_nonzero' "$SHARED_DIR/worker.py" >/dev/null
rg -F 'projection_projection_accumulation.pt' "$SCRIPT_DIR/analyze_v1_diagnostics.py" >/dev/null

if rg -i 'run.*coco|coco.*runner' "$SCRIPT_DIR/run_direct_cos2_confuse5.sh" "$SCRIPT_DIR/server_worker.sh" >/dev/null; then
  echo "ERROR: direct-cos2 Confuse5 execution scripts reference a COCO runner" >&2
  exit 2
fi

git -C "$REPO_ROOT" diff --check
echo "Static validation passed."
echo "Production editor SHA-256: $OBSERVED_EDITOR_SHA"
echo "No Python, model, generation, evaluation, or server job was run."

