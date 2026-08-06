#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${TMPDIR:-/tmp}/concept-clustering-matplotlib-${USER:-user}"
mkdir -p "$MPLCONFIGDIR"

# Keep every Python process inside the repository-mandated py310 environment.
# The CUDA-13 wheel layout is added when present; harmless paths are ignored by ld.so.
exec conda run -n py310 bash -c '
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
exec python "$@"
' bash "$@"
