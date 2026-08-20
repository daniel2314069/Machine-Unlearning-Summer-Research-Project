#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <completed-run-dir>" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$(cd "$SCRIPT_DIR/runs" 2>/dev/null && pwd || true)"
RUN_DIR="$(cd "$1" 2>/dev/null && pwd || true)"
ARCHIVE_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"

if [[ -z "$RUNS_DIR" || -z "$RUN_DIR" || "$RUN_DIR" != "$RUNS_DIR"/* ]]; then
  echo "ERROR: run must be one explicit directory under $SCRIPT_DIR/runs" >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/COMPLETED" || ! -f "$RUN_DIR/exit_code" ]]; then
  echo "ERROR: run has no verified completion state" >&2
  exit 2
fi
if [[ "$(tr -d '[:space:]' < "$RUN_DIR/exit_code")" != "0" ]]; then
  echo "ERROR: run exit code is not zero" >&2
  exit 2
fi

FILES=(
  actual_config.json
  matched_retain_config.json
  protocol.csv
  protocol_manifest.json
  run_manifest.json
  controlled_ablation_check.json
  worker_complete.json
  summary.md
  server.log
  exit_code
  COMPLETED
  results/summary.md
  results/aggregate.csv
  results/per_group.csv
  results/per_concept.csv
  results/informax_diagnostics.csv
  results/top_channels.json
  results/result_manifest.json
  diagnostics/official.pt
  diagnostics/matched_retain.pt
  evaluation/official/evaluation_manifest.json
  evaluation/official/scores.csv
  evaluation/matched_retain/evaluation_manifest.json
  evaluation/matched_retain/scores.csv
  stages/edit_official.completed
  stages/edit_matched_retain.completed
  stages/edit_official_command.json
  stages/edit_matched_retain_command.json
  provenance/scapre/edit/erase_scale.py
  provenance/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/datasets/imagenet-confuse5-derived-25.csv
  provenance/experiments/scapre_informax_specificity/AUDIT.md
  provenance/experiments/scapre_informax_specificity/README.md
  provenance/experiments/scapre_informax_specificity/aggregate_results.py
  provenance/experiments/scapre_informax_specificity/build_protocol.py
  provenance/experiments/scapre_informax_specificity/config.json
  provenance/experiments/scapre_informax_specificity/download_results.sh
  provenance/experiments/scapre_informax_specificity/evaluate_confuse5.py
  provenance/experiments/scapre_informax_specificity/package_results.sh
  provenance/experiments/scapre_informax_specificity/prefetch_assets.py
  provenance/experiments/scapre_informax_specificity/requirements_server.txt
  provenance/experiments/scapre_informax_specificity/run_server.sh
  provenance/experiments/scapre_informax_specificity/server_worker.sh
  provenance/experiments/scapre_informax_specificity/setup_server.sh
  provenance/experiments/scapre_informax_specificity/status_server.sh
  provenance/experiments/scapre_informax_specificity/validate_local.sh
  provenance/experiments/scapre_informax_specificity/worker.py
)
for path in "${FILES[@]}"; do
  if [[ ! -f "$RUN_DIR/$path" ]]; then
    echo "ERROR: required result is missing: $RUN_DIR/$path" >&2
    exit 2
  fi
done

mkdir -p "$ARCHIVE_DIR"
RUN_NAME="$(basename "$RUN_DIR")"
TIMESTAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
ARCHIVE="$ARCHIVE_DIR/scapre_informax_specificity_${RUN_NAME}_${TIMESTAMP}.tar.gz"
if [[ -e "$ARCHIVE" ]]; then
  echo "ERROR: archive already exists: $ARCHIVE" >&2
  exit 2
fi

tar -C "$RUN_DIR" -czf "$ARCHIVE" "${FILES[@]}"
tar -tzf "$ARCHIVE" >/dev/null
CHECKSUM="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
SIZE="$(du -h "$ARCHIVE" | awk '{print $1}')"
printf '%s  %s\n' "$CHECKSUM" "$ARCHIVE" > "$ARCHIVE.sha256"

echo "Archive: $ARCHIVE"
echo "Size: $SIZE"
echo "SHA-256: $CHECKSUM"
echo "Checksum file: $ARCHIVE.sha256"
echo "Original server outputs were preserved."
