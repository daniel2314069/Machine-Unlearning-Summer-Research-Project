#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REQUIRED=(
  config.json AUDIT.md README.md worker.py build_qualitative.py
  aggregate_results.py formal_preflight.py resolve_prior_robustness.sh
  run_server.sh server_worker.sh status_server.sh package_results.sh
  cleanup_images.sh download_results.sh
)
for relative in "${REQUIRED[@]}"; do
  [[ -f "$SCRIPT_DIR/$relative" ]] || { echo "ERROR: missing $relative" >&2; exit 1; }
done
for script in "$SCRIPT_DIR"/*.sh; do bash -n "$script"; done
if rg -n '(^|[^[:alnum:]_])jq([^[:alnum:]_]|$)' "$SCRIPT_DIR" \
  --glob '*.sh' --glob '*.py' --glob '!validate_local.sh'; then
  echo "ERROR: superclass-neutral must not depend on jq" >&2
  exit 1
fi
rg -q "choices=\['official', 'matched-retain', 'superclass-neutral'\]" "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q "default='official'" "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q '"golden retriever": "dog"' "$SCRIPT_DIR/config.json"
rg -q '"yawl": "boat"' "$SCRIPT_DIR/config.json"
rg -q '"soccer ball": "ball"' "$SCRIPT_DIR/config.json"
rg -q '"expected_total_images": 90' "$SCRIPT_DIR/config.json"
rg -q 'baseline_score_evaluations_rerun.*False' "$SCRIPT_DIR/worker.py"
rg -q 'qualitative/images qualitative/comparisons' "$SCRIPT_DIR/package_results.sh"
rg -q 'qualitative images and paired panels' "$SCRIPT_DIR/cleanup_images.sh"
if [[ -d "$SCRIPT_DIR/results" || -d "$SCRIPT_DIR/qualitative" || -d "$SCRIPT_DIR/reproducibility" ]]; then
  RESULT_FILES=(
    summary.md per_seed.csv per_group_seed.csv per_concept_seed.csv
    aggregate_across_seeds.csv per_group_robustness.csv
    per_target_robustness.csv per_retain_robustness.csv
    informax_seed_diagnostics.csv result_manifest.json
  )
  for relative in "${RESULT_FILES[@]}"; do
    [[ -f "$SCRIPT_DIR/results/$relative" ]] || {
      echo "ERROR: missing formal result $relative" >&2
      exit 1
    }
  done
  REPRO_FILES=(
    README.md actual_config.json base_config.json superclass_config.json
    formal_preflight.json baseline_reuse.json integrity_report.json
    protocol_manifest.json run_manifest.json cleanup_manifest.json
    archive_sha256.txt
  )
  for relative in "${REPRO_FILES[@]}"; do
    [[ -f "$SCRIPT_DIR/reproducibility/$relative" ]] || {
      echo "ERROR: missing reproducibility record $relative" >&2
      exit 1
    }
  done
  rg -q '^\*\*NOT SUPPORTED\*\*$' "$SCRIPT_DIR/results/summary.md"
  if [[ -d "$SCRIPT_DIR/qualitative/images" || -d "$SCRIPT_DIR/qualitative/comparisons" ]]; then
    [[ "$(find "$SCRIPT_DIR/qualitative/images" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')" == 90 ]] || {
      echo "ERROR: expected 90 local qualitative images" >&2
      exit 1
    }
    [[ "$(find "$SCRIPT_DIR/qualitative/comparisons" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')" == 30 ]] || {
      echo "ERROR: expected 30 local qualitative comparison panels" >&2
      exit 1
    }
  fi
fi
if git -C "$REPO_ROOT" diff --check; then
  echo "Static superclass-neutral validation passed (no Python/model execution on this Mac)."
fi
