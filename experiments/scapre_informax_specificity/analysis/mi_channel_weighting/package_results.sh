#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <completed-run-directory>" >&2
  exit 2
fi
RUN_DIR="$(cd "$1" && pwd)"
OUTPUT="$RUN_DIR/output"
[[ -f "$RUN_DIR/COMPLETED" && -f "$RUN_DIR/exit_code" && "$(<"$RUN_DIR/exit_code")" == "0" ]] || {
  echo "ERROR: run is not recorded as successfully completed: $RUN_DIR" >&2; exit 1;
}
REQUIRED=(
  sample_size_summary.csv sample_size_per_layer_concept.csv max_mi_stability.csv
  max_mi_activation_summary.csv concept_count_paper_formula.csv concept_count_repo_formula.csv
  concept_count_large_scale_paper_formula.csv concept_count_large_scale_repo_formula.csv
  summary.md integrity_report.json integrity_gate.json activation_distance_diagnostic.json COMPLETED
  figures/sample_size_vs_max_mi_fraction.png figures/sample_size_raw_mi_distribution.png
  figures/sample_size_repo_alpha_distribution.png figures/concept_count_paper_behavior.png
  figures/concept_count_repo_behavior.png
)
for relative in "${REQUIRED[@]}"; do
  [[ -s "$OUTPUT/$relative" ]] || { echo "ERROR: required output missing or empty: $OUTPUT/$relative" >&2; exit 1; }
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="/home/tslin/Documents/jupyter_data/anLi/tmp"
[[ -d "$ARCHIVE_DIR" && -w "$ARCHIVE_DIR" ]] || { echo "ERROR: archive directory unavailable: $ARCHIVE_DIR" >&2; exit 1; }
RUN_ID="$(basename "$RUN_DIR")"
ARCHIVE="$ARCHIVE_DIR/scapre_informax_mi_channel_weighting_${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo "ERROR: refusing to overwrite archive: $ARCHIVE" >&2; exit 1; }
FILE_LIST="$RUN_DIR/package_file_manifest.txt"
printf '%s\n' \
  output/sample_size_summary.csv output/sample_size_per_layer_concept.csv output/max_mi_stability.csv \
  output/max_mi_activation_summary.csv output/concept_count_paper_formula.csv output/concept_count_repo_formula.csv \
  output/concept_count_large_scale_paper_formula.csv output/concept_count_large_scale_repo_formula.csv \
  output/summary.md output/integrity_report.json output/integrity_gate.json output/activation_distance_diagnostic.json \
  output/COMPLETED output/figures/sample_size_vs_max_mi_fraction.png output/figures/sample_size_raw_mi_distribution.png \
  output/figures/sample_size_repo_alpha_distribution.png output/figures/concept_count_paper_behavior.png \
  output/figures/concept_count_repo_behavior.png server.log exit_code run_id started_at_utc python_executable \
  legacy_diagnostic model_snapshot log_path output_path provenance/config.json provenance/implementation_audit.md \
  provenance/run_diagnostics.py provenance/run_server.sh provenance/server_worker.sh provenance/status_server.sh \
  provenance/package_results.sh package_file_manifest.txt > "$FILE_LIST"
tar -C "$RUN_DIR" -czf "$ARCHIVE" -T "$FILE_LIST"
if command -v sha256sum >/dev/null; then
  CHECKSUM="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
else
  CHECKSUM="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
fi
echo "Archive: $ARCHIVE"
echo "Size: $(du -h "$ARCHIVE" | awk '{print $1}')"
echo "SHA-256: $CHECKSUM"
echo "Original server outputs were retained."
