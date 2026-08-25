#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
for path in \
  config.json implementation_audit.md README.md run_diagnostics.py run_server.sh \
  server_worker.sh status_server.sh package_results.sh download_results.sh; do
  [[ -s "$SCRIPT_DIR/$path" ]] || { echo "ERROR: missing file: $path" >&2; exit 1; }
done
bash -n "$SCRIPT_DIR/run_server.sh" "$SCRIPT_DIR/server_worker.sh" \
  "$SCRIPT_DIR/status_server.sh" "$SCRIPT_DIR/package_results.sh" \
  "$SCRIPT_DIR/download_results.sh" "$SCRIPT_DIR/validate_static.sh"
ruby -rjson -e 'JSON.parse(File.read(ARGV.fetch(0)))' "$SCRIPT_DIR/config.json"
if rg -n "StableDiffusionPipeline|text2image|\.images|ResNet|FID|CLIPModel|edit_model\(" \
    "$SCRIPT_DIR/run_diagnostics.py"; then
  echo "ERROR: analysis runner contains a forbidden editing/generation/evaluation path" >&2
  exit 1
fi
git -C "$REPO_ROOT" diff --check
echo "Static validation passed. Python was not executed."
