#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
    "")
        exec "$SCRIPT_DIR/run_server.sh"
        ;;
    --status)
        shift
        if [[ "$#" -ne 0 ]]; then
            echo "--status accepts no additional arguments" >&2
            exit 2
        fi
        exec "$SCRIPT_DIR/status_server.sh"
        ;;
    --package)
        shift
        if [[ "$#" -gt 1 ]]; then
            echo "Usage: $0 --package [completed_run_directory]" >&2
            exit 2
        fi
        exec "$SCRIPT_DIR/package_results.sh" "$@"
        ;;
    --help|-h)
        cat <<'EOF'
Usage:
  maintenance/server_storage_inventory/audit_server_storage.sh
  maintenance/server_storage_inventory/audit_server_storage.sh --status
  maintenance/server_storage_inventory/audit_server_storage.sh --package

The default mode launches a detached, read-only scan of the entire GPU-server
repository. It records hidden files and symlinks but never follows symlinks and
never deletes or changes scanned content. Results are written outside the repo
under /home/tslin/Documents/jupyter_data/anLi/tmp/storage_inventory_runs/.
EOF
        ;;
    *)
        echo "Unknown argument: $1" >&2
        echo "Use --help for supported modes." >&2
        exit 2
        ;;
esac
