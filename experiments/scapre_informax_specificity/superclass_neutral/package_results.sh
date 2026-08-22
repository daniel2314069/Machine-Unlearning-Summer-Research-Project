#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <completed-or-calculation-complete-run-dir>" >&2
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
PYTHON_BIN="$(tr -d '\r\n' < "$RUN_DIR/python_path" 2>/dev/null || true)"
JSON_HELPER="$SCRIPT_DIR/../seed_robustness/json_stdlib.py"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: recorded MU Python interpreter is unavailable" >&2
  exit 2
fi
if [[ "${SCAPRE_INTERNAL_FINALIZE:-0}" == "1" ]]; then
  [[ -f "$RUN_DIR/CALCULATION_COMPLETED" && "$(tr -d '[:space:]' < "$RUN_DIR/calculation_exit_code")" == "0" ]] || {
    echo "ERROR: internal packaging requires successful calculation" >&2; exit 2;
  }
else
  [[ -f "$RUN_DIR/COMPLETED" && "$(tr -d '[:space:]' < "$RUN_DIR/exit_code")" == "0" ]] || {
    echo "ERROR: external packaging requires a completed successful run" >&2; exit 2;
  }
fi
if [[ -f "$RUN_DIR/archive_manifest.json" ]]; then
  ARCHIVE="$($PYTHON_BIN "$JSON_HELPER" get "$RUN_DIR/archive_manifest.json" archive)"
  EXPECTED_SHA="$($PYTHON_BIN "$JSON_HELPER" get "$RUN_DIR/archive_manifest.json" sha256)"
  [[ -f "$ARCHIVE" && "$($PYTHON_BIN "$JSON_HELPER" sha256 "$ARCHIVE")" == "$EXPECTED_SHA" ]] || {
    echo "ERROR: existing archive is missing or changed" >&2; exit 2;
  }
  echo "Archive: $ARCHIVE"
  echo "Size: $(du -h "$ARCHIVE" | awk '{print $1}')"
  echo "SHA-256: $EXPECTED_SHA"
  echo "Checksum file: $ARCHIVE.sha256"
  echo "Reused the already verified archive."
  exit 0
fi

PROFILE="$(tr -d '\r\n' < "$RUN_DIR/profile")"
FILES=(
  actual_config.json base_config.json protocol.csv protocol_manifest.json
  superclass_config.json run_manifest.json worker_complete.json summary.md
  server.log profile command.txt python_path output_path log_path prior_run_path
  started_at_utc calculation_exit_code calculation_finished_at_utc
  CALCULATION_COMPLETED results/summary.md results/per_seed.csv
  results/result_manifest.json reproducibility/integrity_report.json
)
if [[ "$PROFILE" == "formal" ]]; then
  SEEDS=(20260820 20260821 20260822 20260823 20260824)
  FILES+=(
    formal_preflight.json reproducibility/baseline_reuse.json
    reproducibility/prior_robustness_run_manifest.json
    reproducibility/prior_robustness_summary.md
    results/per_group_seed.csv results/per_concept_seed.csv
    results/aggregate_across_seeds.csv results/per_group_robustness.csv
    results/per_target_robustness.csv results/per_retain_robustness.csv
    results/informax_seed_diagnostics.csv qualitative/README.md
    qualitative/manifest.csv qualitative/provenance.json qualitative/COMPLETED
  )
else
  SEEDS=(20260821)
fi
for seed in "${SEEDS[@]}"; do
  FILES+=(
    "seeds/$seed/actual_config.json"
    "seeds/$seed/evaluation/superclass_neutral/scores.csv"
    "seeds/$seed/evaluation/superclass_neutral/evaluation_manifest.json"
    "seeds/$seed/evaluation/superclass_neutral/COMPLETED"
    "seeds/$seed/controlled_ablation_check.json"
    "seeds/$seed/stages/edit_superclass_neutral.completed"
    "seeds/$seed/stages/edit_superclass_neutral_command.json"
    "seeds/$seed/stages/informax_rng_superclass_neutral.json"
  )
  if [[ "$PROFILE" == "formal" ]]; then
    FILES+=(
      "baselines/$seed/informax_diagnostics.csv"
      "baselines/$seed/official/scores.csv"
      "baselines/$seed/official/evaluation_manifest.json"
      "baselines/$seed/official/COMPLETED"
      "baselines/$seed/matched_retain/scores.csv"
      "baselines/$seed/matched_retain/evaluation_manifest.json"
      "baselines/$seed/matched_retain/COMPLETED"
    )
  fi
done
if [[ "$PROFILE" == "formal" ]]; then
  while IFS= read -r relative; do FILES+=("$relative"); done < <(
    cd "$RUN_DIR" && find qualitative/images qualitative/comparisons -type f -name '*.png' -print | LC_ALL=C sort
  )
  if [[ -d "$RUN_DIR/qualitative/recreated_checkpoints" ]]; then
    while IFS= read -r relative; do FILES+=("$relative"); done < <(
      cd "$RUN_DIR" && find qualitative/recreated_checkpoints -type f \
        \( -name '*_command.json' -o -name '*_rng.json' -o -name 'matched_retain_config.json' \) -print | LC_ALL=C sort
    )
  fi
fi
while IFS= read -r source_path; do FILES+=("provenance/$source_path"); done < <(
  "$PYTHON_BIN" "$JSON_HELPER" keys "$RUN_DIR/run_manifest.json" source_sha256
)
for relative in "${FILES[@]}"; do
  [[ -f "$RUN_DIR/$relative" ]] || { echo "ERROR: required result is missing: $relative" >&2; exit 2; }
done
printf '%s\n' "${FILES[@]}" > "$RUN_DIR/package_file_manifest.txt"
FILES+=(package_file_manifest.txt)

mkdir -p "$ARCHIVE_DIR"
RUN_NAME="$(basename "$RUN_DIR")"
TIMESTAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
ARCHIVE="$ARCHIVE_DIR/scapre_informax_superclass_neutral_${RUN_NAME}_${TIMESTAMP}.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo "ERROR: archive already exists: $ARCHIVE" >&2; exit 2; }
tar -C "$RUN_DIR" -czf "$ARCHIVE" "${FILES[@]}"
tar -tzf "$ARCHIVE" >/dev/null
CHECKSUM="$($PYTHON_BIN "$JSON_HELPER" sha256 "$ARCHIVE")"
SIZE_BYTES="$(wc -c < "$ARCHIVE" | tr -d ' ')"
printf '%s  %s\n' "$CHECKSUM" "$ARCHIVE" > "$ARCHIVE.sha256"
"$PYTHON_BIN" "$JSON_HELPER" archive-manifest \
  "$RUN_DIR/archive_manifest.json" "$ARCHIVE" "$CHECKSUM" "$SIZE_BYTES" \
  "$PROFILE" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Archive: $ARCHIVE"
echo "Size: $(du -h "$ARCHIVE" | awk '{print $1}')"
echo "SHA-256: $CHECKSUM"
echo "Checksum file: $ARCHIVE.sha256"
if [[ "$PROFILE" == "formal" ]]; then
  echo "Included: result tables, raw scores, provenance, 90 qualitative images, and 30 side-by-side panels."
fi
echo "Excluded: model checkpoints, full evaluation images, caches, and downloaded weights."
