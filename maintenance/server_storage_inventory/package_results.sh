#!/usr/bin/env bash
set -euo pipefail

TRANSFER_ROOT="/home/tslin/Documents/jupyter_data/anLi/tmp"
RUNS_ROOT="$TRANSFER_ROOT/storage_inventory_runs"
LATEST_FILE="$TRANSFER_ROOT/storage_inventory_state/latest_run"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_results.py"
RUN_DIR="${1:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "MU" ]]; then
    echo "Refusing to package: activate the GPU-server Conda environment first." >&2
    echo "Expected: conda activate MU" >&2
    exit 2
fi
PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" || ! -f "$VERIFY_SCRIPT" ]]; then
    echo "Missing MU Python or verifier: $VERIFY_SCRIPT" >&2
    exit 2
fi

if [[ -z "$RUN_DIR" ]]; then
    if [[ ! -s "$LATEST_FILE" ]]; then
        echo "No storage inventory run has been registered." >&2
        exit 2
    fi
    RUN_DIR="$(tr -d '\n' < "$LATEST_FILE")"
fi
RUN_DIR="$(cd "$RUN_DIR" 2>/dev/null && pwd -P || true)"
if [[ -z "$RUN_DIR" || "$RUN_DIR" != "$RUNS_ROOT"/* || ! -d "$RUN_DIR" ]]; then
    echo "Refusing invalid inventory run path: ${RUN_DIR:-[empty]}" >&2
    exit 2
fi
if [[ ! -f "$RUN_DIR/COMPLETED" || ! -s "$RUN_DIR/exit_code" || "$(tr -d '[:space:]' < "$RUN_DIR/exit_code")" != "0" ]]; then
    echo "Refusing to package an incomplete or failed inventory: $RUN_DIR" >&2
    exit 2
fi

FILES=(
    COMPLETED
    exit_code
    finished_at_utc
    started_at_utc
    pid
    log_path
    stage
    progress.json
    launch_metadata.txt
    git_status_porcelain.txt
    server.log
    summary.md
    summary.json
    result_manifest.json
    all_files.tsv.gz
    all_directories.tsv.gz
    largest_files.csv
    largest_directories.csv
    root_children.csv
    category_summary.csv
    extension_summary.csv
    age_summary.csv
    scan_errors.csv
)
for REQUIRED in "${FILES[@]}"; do
    if [[ ! -f "$RUN_DIR/$REQUIRED" ]]; then
        echo "Refusing incomplete inventory: missing $RUN_DIR/$REQUIRED" >&2
        exit 2
    fi
done

RESULT_MANIFEST_STATUS="$(sed -n 's/^[[:space:]]*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$RUN_DIR/result_manifest.json" | head -n 1)"
if [[ "$RESULT_MANIFEST_STATUS" != "passed" ]]; then
    echo "Refusing inventory with non-passed result manifest: $RESULT_MANIFEST_STATUS" >&2
    exit 2
fi
"$PYTHON_BIN" "$VERIFY_SCRIPT" "$RUN_DIR"

printf '%s\n' "${FILES[@]}" > "$RUN_DIR/package_file_manifest.txt"
FILES+=(package_file_manifest.txt)

RUN_ID="$(basename "$RUN_DIR")"
PACKAGE_TIME="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$TRANSFER_ROOT/machine_unlearning_storage_${RUN_ID}_${PACKAGE_TIME}.tar.gz"
if [[ -e "$ARCHIVE" ]]; then
    echo "Refusing to overwrite existing archive: $ARCHIVE" >&2
    exit 3
fi
tar -C "$RUN_DIR" -czf "$ARCHIVE" "${FILES[@]}"
gzip -t "$ARCHIVE"

if command -v sha256sum >/dev/null 2>&1; then
    CHECKSUM="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
else
    CHECKSUM="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
fi
SIZE_BYTES="$(wc -c < "$ARCHIVE" | tr -d '[:space:]')"
CHECKSUM_FILE="$ARCHIVE.sha256"
printf '%s  %s\n' "$CHECKSUM" "$(basename "$ARCHIVE")" > "$CHECKSUM_FILE"

echo "Inventory archive created; original scan outputs were preserved."
echo "Archive: $ARCHIVE"
echo "Size: $SIZE_BYTES bytes"
echo "SHA-256: $CHECKSUM"
echo "Checksum file: $CHECKSUM_FILE"
echo "Mac download: scp 'tslin:$ARCHIVE' 'tslin:$CHECKSUM_FILE' ~/Downloads/"
