#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/teamspace/studios/this_studio/runs/sequential_oce_celebrity_long_horizon_v1"
ARTIFACT_ROOT="/teamspace/studios/this_studio/artifacts/sequential_oce_celebrity_long_horizon_v1"
GCD_ROOT="/teamspace/studios/this_studio/external/celeb-detection-oss"
PYTHON="/home/zeus/miniconda3/envs/cloudspace/bin/python"
RECOVERY_DIR="$OUTPUT_DIR/.joint_recovery"
LOG_FILE="$OUTPUT_DIR/logs/joint_recovery.log"

mkdir -p "$RECOVERY_DIR" "$(dirname "$LOG_FILE")"
printf '%s\n' "$$" > "$RECOVERY_DIR/worker.pid"
printf '[%s] joint recovery worker start pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" >> "$LOG_FILE"

export PATH="/home/zeus/miniconda3/envs/cloudspace/bin:$PATH"
export CONDA_DEFAULT_ENV="cloudspace"
export HF_HOME="/teamspace/studios/this_studio/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

set +e
"$PYTHON" "$SCRIPT_DIR/recover_joint_reference.py" run \
  --output-dir "$OUTPUT_DIR" \
  --artifact-root "$ARTIFACT_ROOT" \
  --gcd-project-root "$GCD_ROOT" >> "$LOG_FILE" 2>&1
exit_code=$?
set -e
printf '%s\n' "$exit_code" > "$RECOVERY_DIR/worker.exit_code"
printf '[%s] joint recovery worker exit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_code" >> "$LOG_FILE"
sync
exit "$exit_code"
