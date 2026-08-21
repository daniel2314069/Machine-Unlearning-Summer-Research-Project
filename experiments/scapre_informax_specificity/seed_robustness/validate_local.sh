#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.json"
PARENT_CONFIG="$SCRIPT_DIR/../config.json"

for command_name in rg awk sed; do
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

for required_pattern in \
  '"edit_seeds": \[20260820, 20260821, 20260822, 20260823, 20260824\]' \
  '"new_edit_seeds": \[20260821, 20260822, 20260823, 20260824\]' \
  '"fixed_non_informax_seed": 20260820' \
  '"variants": \["official", "matched_retain"\]' \
  '"expected_images_per_variant": 3000' \
  '"expected_target_rows": 1200' \
  '"expected_retain_rows": 1800' \
  '"expected_edited_layers_per_projection": 16' \
  '"expected_diagnostic_records_per_formal_seed": 320' \
  '"expected_diagnostic_records_per_smoke_seed": 64' \
  '"minimum_positive_preserve_seeds": 4' \
  '"minimum_mean_preserve_delta_pp": 1.0' \
  '"maximum_mean_unlearn_delta_pp": 0.0' \
  '"minimum_positive_overall_seeds": 4'; do
  rg -q "$required_pattern" "$CONFIG"
done
for required_pattern in \
  '"edit_seed": 20260820' \
  '"official"' \
  '"matched_retain"' \
  '"num_positive": 5' \
  '"num_negative": 5' \
  '"matched_negative_assignment": "round-robin in listed retain order \(2/2/1\)"' \
  '"formal_images_per_concept": 120'; do
  rg -q "$required_pattern" "$PARENT_CONFIG"
done

while IFS=$'\t' read -r relative expected; do
  actual="$(hash_file "$REPO_ROOT/$relative")"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: controlled source hash changed: $relative" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 2
  fi
done < <(
  sed -n '/"source_controls"[[:space:]]*:/,/^[[:space:]]*},[[:space:]]*$/p' "$CONFIG" |
    sed -n 's/^[[:space:]]*"\([^"]*\)"[[:space:]]*:[[:space:]]*"\([0-9a-f][0-9a-f]*\)"[,]*[[:space:]]*$/\1\	\2/p'
)

rg -q "choices=\['official', 'matched-retain'\]" "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q 'num_pos=5' "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q 'global_rng_legacy_draws_consumed' "$SCRIPT_DIR/informax_seed_runner.py"
rg -q 'git pull --ff-only origin main' "$SCRIPT_DIR/run_server.sh"
rg -q 'SCAPRE_INTERNAL_FINALIZE=1' "$SCRIPT_DIR/server_worker.sh"
rg -q 'archive_manifest.json' "$SCRIPT_DIR/cleanup_images.sh"
rg -q 'generation_keys_match_frozen_protocol' "$SCRIPT_DIR/worker.py"
rg -q 'stable_prior_manifest' "$SCRIPT_DIR/worker.py"
rg -q 'validate_diagnostic_rows' "$SCRIPT_DIR/aggregate_seed_results.py"
rg -q 'diagnostic_layer_target_keys_identical' "$SCRIPT_DIR/aggregate_seed_results.py"
rg -q '\^  \\"\$key' "$SCRIPT_DIR/status_server.sh"
rg -q '\^  \\"\$key' "$SCRIPT_DIR/download_results.sh"
rg -q 'formal_preflight.py' "$SCRIPT_DIR/run_server.sh"
if rg -n 'import torch|diffusers|huggingface|pip install|apt[[:space:]]+install' "$SCRIPT_DIR/formal_preflight.py"; then
  echo "ERROR: formal preflight must remain lightweight and dependency-free" >&2
  exit 2
fi
if rg -n '\bjq\b' "$SCRIPT_DIR" --glob '*.sh' --glob '*.py' --glob '!validate_local.sh'; then
  echo "ERROR: jq must not be required by seed-robustness scripts" >&2
  exit 2
fi

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
  README.md AUDIT.md config.json json_stdlib.py informax_seed_runner.py
  formal_preflight.py worker.py
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
