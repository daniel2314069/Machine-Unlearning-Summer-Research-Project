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
if git -C "$REPO_ROOT" diff --check; then
  echo "Static superclass-neutral validation passed (no Python/model execution on this Mac)."
fi
