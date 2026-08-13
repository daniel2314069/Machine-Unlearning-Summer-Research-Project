#!/usr/bin/env bash
set -euo pipefail

# Create a non-destructive, single-entry-point snapshot on the GPU server.
# Large runtime artifacts remain at their provenance-preserving locations and
# are exposed through links; compact tracked evidence is copied into snapshot.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${experiment_root}/../.." && pwd)"
default_snapshot="${experiment_root}/outputs/failed_oce_afr_study_archive_v1"
snapshot_root="${1:-${default_snapshot}}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "This inventory is intended for the GPU server, not the local Mac." >&2
  exit 2
fi

for command_name in git find sha256sum stat du tar; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 2
  fi
done

if [[ -e "${snapshot_root}" ]]; then
  echo "Refusing to overwrite existing snapshot: ${snapshot_root}" >&2
  exit 3
fi

mkdir -p "${snapshot_root}/compact" "${snapshot_root}/payload_links" \
  "${snapshot_root}/inventory"

compact_sources=(
  "${experiment_root}/study_archive"
  "${experiment_root}/solver_audit/REPORT.md"
  "${experiment_root}/solver_audit/results.csv"
  "${experiment_root}/solver_audit/exact_orthogonal_control/REPORT_exact_control.md"
  "${experiment_root}/solver_audit/exact_orthogonal_control/results_exact_control.csv"
  "${experiment_root}/solver_audit/exact_orthogonal_control/anchor_min_control/REPORT_anchor_min_control.md"
  "${experiment_root}/solver_audit/exact_orthogonal_control/anchor_min_control/results_anchor_min_control.csv"
  "${experiment_root}/afr/results/afr_balls_smoke_v1"
)

for source_path in "${compact_sources[@]}"; do
  if [[ ! -e "${source_path}" ]]; then
    echo "Missing compact artifact: ${source_path}" >&2
    exit 4
  fi
done

tar -C "${repo_root}" -cf - \
  "experiments/confuse5_single_vs_joint/study_archive" \
  "experiments/confuse5_single_vs_joint/solver_audit/REPORT.md" \
  "experiments/confuse5_single_vs_joint/solver_audit/results.csv" \
  "experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/REPORT_exact_control.md" \
  "experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/results_exact_control.csv" \
  "experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/anchor_min_control/REPORT_anchor_min_control.md" \
  "experiments/confuse5_single_vs_joint/solver_audit/exact_orthogonal_control/anchor_min_control/results_anchor_min_control.csv" \
  "experiments/confuse5_single_vs_joint/afr/results/afr_balls_smoke_v1" \
  | tar -C "${snapshot_root}/compact" -xf -

declare -a payload_names=()
declare -a payload_paths=()

register_payload() {
  local payload_name="$1"
  local payload_path="$2"
  if [[ -e "${payload_path}" ]]; then
    payload_names+=("${payload_name}")
    payload_paths+=("${payload_path}")
    ln -s "${payload_path}" "${snapshot_root}/payload_links/${payload_name}"
  fi
}

register_payload \
  "official_primary_runtime" \
  "${experiment_root}/outputs/official_repo_primary_v1"
register_payload \
  "afr_balls_smoke_runtime" \
  "${experiment_root}/afr/outputs/afr_balls_smoke_v1"
register_payload \
  "legacy_invalid_pilot_archive" \
  "${experiment_root}/archives/invalid_for_primary__pilot_default_config"

if [[ ${#payload_paths[@]} -eq 0 ]]; then
  echo "No runtime payload trees were found." >&2
  exit 5
fi

{
  echo -e "name\tsource_path\tdisk_usage"
  for index in "${!payload_paths[@]}"; do
    usage="$(du -sh "${payload_paths[$index]}" | cut -f1)"
    echo -e "${payload_names[$index]}\t${payload_paths[$index]}\t${usage}"
  done
} > "${snapshot_root}/inventory/ROOTS.tsv"

{
  echo -e "payload\tbytes\tsha256\tpath"
  for index in "${!payload_paths[@]}"; do
    payload_name="${payload_names[$index]}"
    payload_path="${payload_paths[$index]}"
    while IFS= read -r -d '' artifact; do
      bytes="$(stat -c '%s' "${artifact}")"
      digest="$(sha256sum "${artifact}" | cut -d ' ' -f1)"
      echo -e "${payload_name}\t${bytes}\t${digest}\t${artifact}"
    done < <(find "${payload_path}" -type f -print0 | sort -z)
  done
} > "${snapshot_root}/inventory/FILES.tsv"

(
  cd "${repo_root}"
  git rev-parse HEAD
) > "${snapshot_root}/inventory/GIT_HEAD.txt"

(
  cd "${repo_root}"
  git status --short
) > "${snapshot_root}/inventory/GIT_STATUS.txt"

cat > "${snapshot_root}/README.txt" <<EOF
OCE solver-audit to AFR failed-study server snapshot

Created: $(date --iso-8601=seconds)
Repository: ${repo_root}
Git HEAD: $(cat "${snapshot_root}/inventory/GIT_HEAD.txt")

compact/       copied reports and machine-readable results
payload_links/ non-destructive links to large runtime trees
inventory/     roots, disk usage, per-file SHA-256, and Git state

Nothing was moved or deleted. Removing this snapshot removes only the compact
copy, links, and inventory; it does not remove the linked runtime payloads.
EOF

echo "Created non-destructive study snapshot: ${snapshot_root}"
echo "Runtime payloads remain in place. No images or checkpoints were deleted."
