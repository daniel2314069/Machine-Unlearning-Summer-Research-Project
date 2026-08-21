#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.json"
PARENT_CONFIG="$SCRIPT_DIR/../config.json"

for command_name in jq rg awk sed; do
  command -v "$command_name" >/dev/null || {
    echo "ERROR: required static-check command is unavailable: $command_name" >&2
    exit 2
  }
done

hash_file() {
  if command -v shasum >/dev/null; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

jq -e '
  .edit_seeds == [20260820,20260821,20260822,20260823,20260824] and
  .new_edit_seeds == [20260821,20260822,20260823,20260824] and
  .fixed_non_informax_seed == 20260820 and
  .variants == ["official","matched_retain"] and
  .expected_images_per_variant == 3000 and
  .expected_target_rows == 1200 and
  .expected_retain_rows == 1800 and
  .robustness_rule.minimum_positive_preserve_seeds == 4 and
  .robustness_rule.minimum_mean_preserve_delta_pp == 1.0 and
  .robustness_rule.maximum_mean_unlearn_delta_pp == 0.0 and
  .robustness_rule.minimum_positive_overall_seeds == 4
' "$CONFIG" >/dev/null
jq -e '
  .edit_seed == 20260820 and
  .variants == ["official","matched_retain"] and
  .edit.num_positive == 5 and
  .edit.num_negative == 5 and
  .edit.matched_negative_assignment == "round-robin in listed retain order (2/2/1)" and
  .evaluation.formal_images_per_concept == 120
' "$PARENT_CONFIG" >/dev/null

while IFS=$'\t' read -r relative expected; do
  actual="$(hash_file "$REPO_ROOT/$relative")"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: controlled source hash changed: $relative" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 2
  fi
done < <(jq -r '.source_controls | to_entries[] | [.key,.value] | @tsv' "$CONFIG")

rg -q "choices=\['official', 'matched-retain'\]" "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q 'num_pos=5' "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q 'global_rng_legacy_draws_consumed' "$SCRIPT_DIR/informax_seed_runner.py"
rg -q 'git pull --ff-only origin main' "$SCRIPT_DIR/run_server.sh"
rg -q 'SCAPRE_INTERNAL_FINALIZE=1' "$SCRIPT_DIR/server_worker.sh"
rg -q 'archive_manifest.json' "$SCRIPT_DIR/cleanup_images.sh"

for script in "$SCRIPT_DIR"/*.sh; do
  bash -n "$script"
done
for runner in "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.py; do
  [[ "$runner" == "$SCRIPT_DIR/validate_local.sh" ]] && continue
  if rg -n '^[[:space:]]*conda[[:space:]]+activate|/anaconda3/.*/python|/miniconda.*/python' "$runner"; then
    echo "ERROR: robustness runners must inherit the active environment" >&2
    exit 2
  fi
done

awk -F, '
  NR==2 && ($1!="official" || $2!=19.75) {exit 1}
  NR==3 && ($1!="matched_retain" || $2<17.4166 || $2>17.4167) {exit 1}
  END {if (NR!=4) exit 1}
' "$SCRIPT_DIR/../results/aggregate.csv"

required=(
  README.md AUDIT.md config.json informax_seed_runner.py worker.py
  aggregate_seed_results.py resolve_prior_seed.sh run_server.sh server_worker.sh
  status_server.sh package_results.sh cleanup_images.sh download_results.sh
  results/summary.md reproducibility/README.md
)
for relative in "${required[@]}"; do
  [[ -f "$SCRIPT_DIR/$relative" ]] || {
    echo "ERROR: required file is missing: $relative" >&2
    exit 2
  }
done

echo "Static seed-robustness validation passed. No Python, model, or image code was executed."
