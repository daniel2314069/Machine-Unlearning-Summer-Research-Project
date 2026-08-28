#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.json"
EDITOR="$REPO_ROOT/scapre/edit/erase_scale.py"
EXPECTED_EDITOR_SHA="cc454407a70de5b403344f8e3d0372044fed156cf78a74fa04121473674ada20"

for command_name in bash git jq rg shasum; do
  command -v "$command_name" >/dev/null || {
    echo "ERROR: static validator requires $command_name" >&2
    exit 2
  }
done

for script in "$SCRIPT_DIR"/*.sh; do
  bash -n "$script"
done

jq -e '
  .variant == "projection_accumulation" and
  .formula.difference == "concept_vecs[k] - empty_vec" and
  .formula.temperature == 0.7 and .formula.power == 8 and .formula.eps == 1e-8 and
  .formula.std == "torch.std default sample-standard-deviation semantics" and
  .formula.sweep == false and
  .qualification_seed == 20260820 and
  .edit_seeds == [20260820,20260821,20260822,20260823,20260824] and
  .expected_accumulation_intercepts_per_edit == 320 and
  .expected_informax_randn_calls_per_edit == 1280 and
  .coco.label == "project-defined secondary general-generation safeguard" and
  .coco.edit_seed == 20260820 and .coco.ordered_subsets == [1000,10000] and
  .coco.paper_reproduction == false
' "$CONFIG" >/dev/null

OBSERVED_EDITOR_SHA="$(shasum -a 256 "$EDITOR" | awk '{print $1}')"
[[ "$OBSERVED_EDITOR_SHA" == "$EXPECTED_EDITOR_SHA" ]] || {
  echo "ERROR: production editor hash changed: $OBSERVED_EDITOR_SHA" >&2
  exit 2
}
git -C "$REPO_ROOT" diff --exit-code -- scapre/edit/erase_scale.py >/dev/null

rg -F 'd_vec = c_vec - empty_vec' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'score.std() + args.eps' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'if source.count(needle) != 2:' "$SCRIPT_DIR/projection_runner.py" >/dev/null
rg -F 'selected = row_w_c if args.variant == "official" else projection_alpha' "$SCRIPT_DIR/projection_runner.py" >/dev/null
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
