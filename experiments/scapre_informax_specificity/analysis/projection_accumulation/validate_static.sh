#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
  abort("wrong variant") unless c["variant"] == "projection_accumulation"
  abort("wrong alpha mode") unless c.dig("formula", "alpha_mode") == "zscore_sigmoid_power"
  abort("wrong formula") unless c.dig("formula", "difference") == "concept_vecs[k] - empty_vec"
  abort("wrong transform") unless c.dig("formula", "temperature") == 0.7 && c.dig("formula", "power") == 8.0
  abort("wrong eps") unless c.dig("formula", "eps") == 1e-8
  abort("wrong std") unless c.dig("formula", "std") == "torch.std default sample-standard-deviation semantics"
  abort("sweep enabled") unless c.dig("formula", "sweep") == false
  abort("wrong seeds") unless c["edit_seeds"] == [20260820,20260821,20260822,20260823,20260824]
  abort("wrong qualification") unless c["qualification_seed"] == 20260820
  abort("wrong counts") unless c["expected_accumulation_intercepts_per_edit"] == 320 && c["expected_informax_randn_calls_per_edit"] == 1280
  abort("wrong COCO contract") unless c.dig("coco", "label") == "project-defined secondary general-generation safeguard" && c.dig("coco", "edit_seed") == 20260820 && c.dig("coco", "ordered_subsets") == [1000,10000] && c.dig("coco", "paper_reproduction") == false
' "$CONFIG"

OBSERVED_EDITOR_SHA="$(shasum -a 256 "$EDITOR" | awk '{print $1}')"
[[ "$OBSERVED_EDITOR_SHA" == "$EXPECTED_EDITOR_SHA" ]] || {
  echo "ERROR: production editor hash changed: $OBSERVED_EDITOR_SHA" >&2
  exit 2
}
git -C "$REPO_ROOT" diff --exit-code -- scapre/edit/erase_scale.py >/dev/null

rg -F 'd_vec = c_vec - empty_vec' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'score.std() + args.eps' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'if source.count(needle) != 2:' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'if args.alpha_mode == "zscore_sigmoid_power"' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'return official_contribution' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'return for_mat1 * projection_alpha' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'in_memory_source_substitution_scope": "for_mat1 * row_w_c only"' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'aggregate_row_w_max_intercepted": False' "$SCRIPT_DIR/projection_runner.py" >/dev/null

if rg -F 'run_projection_coco.sh' "$SCRIPT_DIR/run_projection_confuse5.sh" >/dev/null; then
  echo "ERROR: Confuse5 runner references the COCO runner" >&2
  exit 2
fi
if rg -F 'run_projection_confuse5.sh' "$SCRIPT_DIR/run_projection_coco.sh" >/dev/null; then
  echo "ERROR: COCO runner references the Confuse5 runner" >&2
  exit 2
fi

git -C "$REPO_ROOT" diff --check
echo "Static validation passed."
echo "Production editor SHA-256: $OBSERVED_EDITOR_SHA"
echo "No Python, model, generation, evaluation, or server job was run."
