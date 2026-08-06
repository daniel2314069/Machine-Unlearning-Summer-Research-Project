#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${TMPDIR:-/tmp}/oce-correspondence-matplotlib-${USER:-user}"
mkdir -p "$MPLCONFIGDIR"

exec conda run -n py310 bash -c '
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
exec python "$@"
' bash "$@"
