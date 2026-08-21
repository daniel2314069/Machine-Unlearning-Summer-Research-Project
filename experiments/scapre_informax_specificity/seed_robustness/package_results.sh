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
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: recorded MU Python interpreter is unavailable" >&2
  exit 2
fi
JSON_HELPER="$SCRIPT_DIR/json_stdlib.py"
if [[ "${SCAPRE_INTERNAL_FINALIZE:-0}" == "1" ]]; then
  if [[ ! -f "$RUN_DIR/CALCULATION_COMPLETED" || "$(tr -d '[:space:]' < "$RUN_DIR/calculation_exit_code")" != "0" ]]; then
    echo "ERROR: internal packaging requires a successful calculation" >&2
    exit 2
  fi
else
  if [[ ! -f "$RUN_DIR/COMPLETED" || "$(tr -d '[:space:]' < "$RUN_DIR/exit_code")" != "0" ]]; then
    echo "ERROR: external packaging requires a completed successful run" >&2
    exit 2
  fi
fi

if [[ -f "$RUN_DIR/archive_manifest.json" ]]; then
  EXISTING_ARCHIVE="$("$PYTHON_BIN" "$JSON_HELPER" get "$RUN_DIR/archive_manifest.json" archive)"
  EXPECTED_SHA="$("$PYTHON_BIN" "$JSON_HELPER" get "$RUN_DIR/archive_manifest.json" sha256)"
  if [[ ! -f "$EXISTING_ARCHIVE" ]]; then
    echo "ERROR: archive manifest points to a missing archive: $EXISTING_ARCHIVE" >&2
    exit 2
  fi
  ACTUAL_SHA="$("$PYTHON_BIN" "$JSON_HELPER" sha256 "$EXISTING_ARCHIVE")"
  if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    echo "ERROR: existing archive checksum no longer matches" >&2
    exit 2
  fi
  echo "Archive: $EXISTING_ARCHIVE"
  echo "Size: $(du -h "$EXISTING_ARCHIVE" | awk '{print $1}')"
  echo "SHA-256: $EXPECTED_SHA"
  echo "Checksum file: $EXISTING_ARCHIVE.sha256"
  echo "Reused the already verified archive."
  exit 0
fi

PROFILE="$(tr -d '\r\n' < "$RUN_DIR/profile")"
if [[ "$PROFILE" == "formal" ]]; then
  SEEDS=(20260820 20260821 20260822 20260823 20260824)
  NEW_SEEDS=(20260821 20260822 20260823 20260824)
elif [[ "$PROFILE" == "smoke" ]]; then
  SEEDS=(20260821)
  NEW_SEEDS=(20260821)
else
  echo "ERROR: unknown run profile: $PROFILE" >&2
  exit 2
fi

FILES=(
  actual_config.json base_config.json protocol.csv protocol_manifest.json
  matched_retain_config.json run_manifest.json worker_complete.json summary.md
  server.log profile calculation_exit_code calculation_finished_at_utc
  CALCULATION_COMPLETED
  results/summary.md results/per_seed.csv results/per_group_seed.csv
  results/per_concept_seed.csv results/aggregate_across_seeds.csv
  results/per_group_robustness.csv results/per_retain_robustness.csv
  results/informax_seed_diagnostics.csv results/result_manifest.json
  reproducibility/integrity_report.json
)
if [[ "$PROFILE" == "formal" ]]; then
  FILES+=(formal_preflight.json reproducibility/prior_seed_validation.json)
fi

for seed in "${SEEDS[@]}"; do
  FILES+=(
    "seeds/$seed/actual_config.json"
    "seeds/$seed/results/aggregate.csv"
    "seeds/$seed/results/per_group.csv"
    "seeds/$seed/results/per_concept.csv"
    "seeds/$seed/results/informax_diagnostics.csv"
    "seeds/$seed/evaluation/official/evaluation_manifest.json"
    "seeds/$seed/evaluation/official/scores.csv"
    "seeds/$seed/evaluation/official/COMPLETED"
    "seeds/$seed/evaluation/matched_retain/evaluation_manifest.json"
    "seeds/$seed/evaluation/matched_retain/scores.csv"
    "seeds/$seed/evaluation/matched_retain/COMPLETED"
  )
done
if [[ "$PROFILE" == "formal" ]]; then
  FILES+=(
    seeds/20260820/prior_run_manifest.json
    seeds/20260820/seed_source_manifest.json
  )
fi
for seed in "${NEW_SEEDS[@]}"; do
  FILES+=(
    "seeds/$seed/source_manifest.json"
    "seeds/$seed/controlled_ablation_check.json"
    "seeds/$seed/stages/edit_official.completed"
    "seeds/$seed/stages/edit_matched_retain.completed"
    "seeds/$seed/stages/edit_official_command.json"
    "seeds/$seed/stages/edit_matched_retain_command.json"
    "seeds/$seed/stages/informax_rng_official.json"
    "seeds/$seed/stages/informax_rng_matched_retain.json"
  )
done

while IFS= read -r source_path; do
  FILES+=("provenance/$source_path")
done < <("$PYTHON_BIN" "$JSON_HELPER" keys "$RUN_DIR/run_manifest.json" source_sha256)

for relative in "${FILES[@]}"; do
  if [[ ! -f "$RUN_DIR/$relative" ]]; then
    echo "ERROR: required result is missing: $RUN_DIR/$relative" >&2
    exit 2
  fi
done
printf '%s\n' "${FILES[@]}" > "$RUN_DIR/package_file_manifest.txt"
FILES+=(package_file_manifest.txt)

mkdir -p "$ARCHIVE_DIR"
RUN_NAME="$(basename "$RUN_DIR")"
TIMESTAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
ARCHIVE="$ARCHIVE_DIR/scapre_informax_seed_robustness_${RUN_NAME}_${TIMESTAMP}.tar.gz"
if [[ -e "$ARCHIVE" ]]; then
  echo "ERROR: archive already exists: $ARCHIVE" >&2
  exit 2
fi
tar -C "$RUN_DIR" -czf "$ARCHIVE" "${FILES[@]}"
tar -tzf "$ARCHIVE" >/dev/null
CHECKSUM="$("$PYTHON_BIN" "$JSON_HELPER" sha256 "$ARCHIVE")"
SIZE_BYTES="$(wc -c < "$ARCHIVE" | tr -d ' ')"
SIZE_HUMAN="$(du -h "$ARCHIVE" | awk '{print $1}')"
printf '%s  %s\n' "$CHECKSUM" "$ARCHIVE" > "$ARCHIVE.sha256"
"$PYTHON_BIN" "$JSON_HELPER" archive-manifest \
  "$RUN_DIR/archive_manifest.json" "$ARCHIVE" "$CHECKSUM" "$SIZE_BYTES" \
  "$PROFILE" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

echo "Archive: $ARCHIVE"
echo "Size: $SIZE_HUMAN"
echo "SHA-256: $CHECKSUM"
echo "Checksum file: $ARCHIVE.sha256"
echo "Generated images, model checkpoints, caches, and downloaded weights were excluded."
echo "Original non-image server outputs were preserved."
