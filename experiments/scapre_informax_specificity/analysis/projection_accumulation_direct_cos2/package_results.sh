#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="$(cd "$SCRIPT_DIR/runs/confuse5" 2>/dev/null && pwd || true)"
TREATMENT="projection_accumulation_direct_cos2"
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  [[ -f "$SCRIPT_DIR/.server/confuse5/latest_run" ]] || { echo "ERROR: no recorded run" >&2; exit 2; }
  RUN_DIR="$(tr -d '\r\n' < "$SCRIPT_DIR/.server/confuse5/latest_run")"
fi
RUN_DIR="$(cd "$RUN_DIR" 2>/dev/null && pwd || true)"
[[ -n "$RUNS_ROOT" && -n "$RUN_DIR" && "$RUN_DIR" == "$RUNS_ROOT"/* ]] || {
  echo "ERROR: package target must be one explicit direct-cos2 run" >&2; exit 2;
}
[[ -f "$RUN_DIR/COMPLETED" && "$(tr -d '[:space:]' < "$RUN_DIR/exit_code")" == "0" ]] || {
  echo "ERROR: package requires a completed successful run" >&2; exit 2;
}
[[ -f "$RUN_DIR/reproducibility/integrity_report.json" ]] || { echo "ERROR: integrity report missing" >&2; exit 2; }
INTEGRITY_STATUS="$(sed -n 's/^  "status": "\([^"]*\)",\{0,1\}$/\1/p' "$RUN_DIR/reproducibility/integrity_report.json" | head -n 1)"
[[ "$INTEGRITY_STATUS" == "passed" ]] || {
  echo "ERROR: integrity report did not pass" >&2; exit 2;
}
FILES=(
  COMPLETED exit_code finished_at_utc started_at_utc python_path logs/server.log status.json
  summary.md worker_complete.json
  pre_analysis/v1_transform/summary.json pre_analysis/v1_transform/summary.md
  pre_analysis/v1_transform/per_layer_concept_distributions.csv
  pre_analysis/v1_transform/per_concept_distributions.csv
  qualification/PASSED qualification/integrity_report.json
  qualification/per_layer_concept_correlations.csv
  qualification/per_layer_concept_weight_diagnostics.csv
  qualification/per_matrix_edit_strength.csv
  reproducibility/actual_config.json reproducibility/base_config.json
  reproducibility/protocol.csv reproducibility/protocol_manifest.json
  reproducibility/official_reference_validation.json reproducibility/integrity_report.json
  reproducibility/run_manifest.json
  results/per_seed_metrics.csv results/per_target_metrics.csv results/comparison_deltas.csv
  results/per_group_metrics.csv results/aggregate_metrics.json results/summary.md
)
for seed in 20260820 20260821 20260822 20260823 20260824; do
  FILES+=(
    "seeds/$seed/reproducibility_isolation.json"
    "seeds/$seed/audits/official.json" "seeds/$seed/audits/$TREATMENT.json"
    "seeds/$seed/diagnostics/informax_official.pt" "seeds/$seed/diagnostics/informax_$TREATMENT.pt"
    "seeds/$seed/diagnostics/projection_official.pt" "seeds/$seed/diagnostics/projection_$TREATMENT.pt"
    "seeds/$seed/stages/edit_official.command.json" "seeds/$seed/stages/edit_$TREATMENT.command.json"
    "seeds/$seed/stages/edit_official.completed.json" "seeds/$seed/stages/edit_$TREATMENT.completed.json"
    "seeds/$seed/stages/checkpoint_official.cleanup.json" "seeds/$seed/stages/checkpoint_$TREATMENT.cleanup.json"
    "seeds/$seed/evaluation/official/scores.csv" "seeds/$seed/evaluation/official/evaluation_manifest.json"
    "seeds/$seed/evaluation/$TREATMENT/scores.csv" "seeds/$seed/evaluation/$TREATMENT/evaluation_manifest.json"
  )
done
while IFS= read -r source; do FILES+=("reproducibility/provenance/$source"); done \
  < <(sed -n '/"source_sha256": {/,/^  }/s/^    "\([^"]*\)".*/\1/p' "$RUN_DIR/reproducibility/run_manifest.json")
for relative in "${FILES[@]}"; do
  [[ -f "$RUN_DIR/$relative" ]] || { echo "ERROR: required package file missing: $relative" >&2; exit 2; }
done
printf '%s\n' "${FILES[@]}" > "$RUN_DIR/package_file_manifest.txt"
FILES+=(package_file_manifest.txt)
ARCHIVE_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
mkdir -p "$ARCHIVE_DIR"
ARCHIVE="$ARCHIVE_DIR/scapre_projection_direct_cos2_$(basename "$RUN_DIR")_$(date -u +'%Y%m%dT%H%M%SZ').tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo "ERROR: archive already exists: $ARCHIVE" >&2; exit 2; }
tar -C "$RUN_DIR" -czf "$ARCHIVE" "${FILES[@]}"
tar -tzf "$ARCHIVE" >/dev/null
CHECKSUM="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
SIZE_BYTES="$(wc -c < "$ARCHIVE" | tr -d ' ')"
printf '%s  %s\n' "$CHECKSUM" "$ARCHIVE" > "$ARCHIVE.sha256"
echo "Archive: $ARCHIVE"
echo "Size: $(du -h "$ARCHIVE" | awk '{print $1}') ($SIZE_BYTES bytes)"
echo "SHA-256: $CHECKSUM"
echo "Mac download: scp 'tslin:$ARCHIVE' ~/Downloads/"
echo "Generated images and regenerable checkpoints were excluded; server image outputs were preserved."

