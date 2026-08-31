#!/usr/bin/env bash
set -euo pipefail

EXPECTED_ANLI_ROOT="/home/tslin/Documents/jupyter_data/anLi"
EXPECTED_REPO_ROOT="$EXPECTED_ANLI_ROOT/machine_unlearning"
EXPECTED_TMP_ROOT="$EXPECTED_ANLI_ROOT/tmp"
RECORD_ROOT="$EXPECTED_ANLI_ROOT/storage_cleanup_records"
LOCK_DIR="$EXPECTED_ANLI_ROOT/.server_storage_cleanup.lock"

SCAPRE_ROOTS=(
    "experiments/scapre_informax_specificity/runs"
    "experiments/scapre_informax_specificity/seed_robustness/runs"
    "experiments/scapre_informax_specificity/superclass_neutral/runs"
    "experiments/scapre_informax_specificity/analysis/mi_channel_weighting/runs"
    "experiments/scapre_informax_specificity/analysis/alpha_channel_controls/runs"
    "experiments/scapre_informax_specificity/analysis/projection_accumulation/runs"
    "experiments/scapre_informax_specificity/analysis/projection_accumulation_direct_cos2/runs"
    "experiments/scapre_informax_specificity/analysis/projection_accumulation_budget_matched_cos2/runs"
)

OCE_ROOTS=(
    "orthogonal-concept-erasure/experiments/concept_description_clustering"
    "orthogonal-concept-erasure/experiments/confuse5_single_vs_joint"
    "orthogonal-concept-erasure/experiments/correspondence_diagnostic"
    "orthogonal-concept-erasure/experiments/oce_failure_image_qualification"
    "orthogonal-concept-erasure/experiments/overlap_cycle_images"
    "orthogonal-concept-erasure/experiments/sequential_object_followup"
    "orthogonal-concept-erasure/experiments/sequential_object_pair_retain"
    "orthogonal-concept-erasure/experiments/sequential_object_persistence"
)

usage() {
    cat <<'EOF'
Usage:
  maintenance/server_storage_cleanup/cleanup_server_storage.sh --preview [all|scapre|oce|tmp]
  maintenance/server_storage_cleanup/cleanup_server_storage.sh --apply scapre --confirm DELETE_SCAPRE_RUNS
  maintenance/server_storage_cleanup/cleanup_server_storage.sh --apply oce --confirm DELETE_OCE_GENERATED_BINARIES
  maintenance/server_storage_cleanup/cleanup_server_storage.sh --apply tmp --confirm EMPTY_ANLI_TMP_AFTER_LOCAL_BACKUP
  maintenance/server_storage_cleanup/cleanup_server_storage.sh --status

The default behavior is not destructive: an explicit --preview or --apply is
required. Apply accepts exactly one scope; there is deliberately no destructive
"all" mode. Every candidate is resolved and checked under
/home/tslin/Documents/jupyter_data/anLi before deletion.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

format_bytes() {
    awk -v value="$1" 'BEGIN {
        split("B KiB MiB GiB TiB", units, " ");
        i = 1;
        while (value >= 1024 && i < 5) { value /= 1024; i += 1 }
        printf "%.3f %s", value, units[i]
    }'
}

assert_no_unsafe_name() {
    local root="$1"
    local bad=""
    bad="$(find "$root" -xdev \( -name $'*\n*' -o -name $'*\t*' \) -print -quit 2>/dev/null || true)"
    [[ -z "$bad" ]] || die "tab/newline filename requires manual review under: $root"
}

assert_no_nested_mount() {
    local root="$1"
    local directory=""
    while IFS= read -r -d '' directory; do
        if [[ "$directory" != "$root" ]] && mountpoint -q -- "$directory"; then
            die "nested mount point found inside deletion target: $directory"
        fi
    done < <(find "$root" -xdev -type d -print0)
}

assert_only_regular_files_and_directories() {
    local root="$1"
    local special=""
    special="$(find "$root" -xdev -mindepth 1 ! -type f ! -type d -print -quit 2>/dev/null || true)"
    [[ -z "$special" ]] || die "special entry requires manual review before recursive cleanup: $special"
}

assert_no_live_pid_files() {
    local root="$1"
    local pid_file=""
    local pid_value=""
    while IFS= read -r -d '' pid_file; do
        pid_value="$(tr -d '[:space:]' < "$pid_file" 2>/dev/null || true)"
        if [[ "$pid_value" =~ ^[0-9]+$ ]] && kill -0 "$pid_value" 2>/dev/null; then
            die "live PID $pid_value is referenced by $pid_file"
        fi
    done < <(find "$root" -xdev -type f \( -name pid -o -name '*.pid' \) -print0)
}

assert_repo_relative_root() {
    local relative="$1"
    local absolute="$REPO_ROOT/$relative"
    local resolved=""
    [[ "$relative" != /* && "$relative" != *".."* ]] || die "unsafe relative target: $relative"
    if [[ -e "$absolute" ]]; then
        [[ ! -L "$absolute" ]] || die "target root is a symlink: $absolute"
        resolved="$(realpath -e "$absolute")"
        [[ "$resolved" == "$REPO_ROOT/"* ]] || die "target resolves outside repository: $absolute -> $resolved"
    fi
}

assert_inside_anli() {
    local path="$1"
    local resolved=""
    [[ "$path" == "$ANLI_ROOT/"* ]] || die "deletion path is outside anLi: $path"
    [[ -e "$path" ]] || die "deletion path disappeared before use: $path"
    resolved="$(realpath -e "$path")"
    [[ "$resolved" == "$ANLI_ROOT/"* ]] || die "deletion path resolves outside anLi: $path -> $resolved"
}

assert_integrity_status_passed() {
    local relative="$1"
    local file="$REPO_ROOT/$relative"
    local status=""
    [[ -f "$file" ]] || die "missing required integrity artifact: $relative"
    status="$(sed -n 's/^[[:space:]]*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/{p;q;}' "$file")"
    [[ "$status" == "passed" ]] || die "integrity status is not passed: $relative (${status:-missing})"
}

assert_required_scapre_results() {
    local required_files=(
        "experiments/scapre_informax_specificity/results/aggregate.csv"
        "experiments/scapre_informax_specificity/results/per_concept.csv"
        "experiments/scapre_informax_specificity/results/reproducibility/formal_archive.sha256"
        "experiments/scapre_informax_specificity/analysis/mi_channel_weighting/results/max_mi_activation_summary.csv.gz"
        "experiments/scapre_informax_specificity/analysis/alpha_channel_controls/formal_results/per_seed_metrics.csv"
        "experiments/scapre_informax_specificity/analysis/projection_accumulation/formal_results/per_seed_metrics.csv"
        "experiments/scapre_informax_specificity/analysis/projection_accumulation_direct_cos2/formal_results/per_seed_metrics.csv"
        "experiments/scapre_informax_specificity/analysis/projection_accumulation_budget_matched_cos2/formal_results/per_seed_metrics.csv"
    )
    local required=""
    for required in "${required_files[@]}"; do
        [[ -s "$REPO_ROOT/$required" ]] || die "missing or empty retained ScaPre result: $required"
        git -C "$REPO_ROOT" ls-files --error-unmatch -- "$required" >/dev/null 2>&1 \
            || die "retained ScaPre result is not tracked: $required"
    done

    assert_integrity_status_passed "experiments/scapre_informax_specificity/seed_robustness/reproducibility/integrity_report.json"
    assert_integrity_status_passed "experiments/scapre_informax_specificity/superclass_neutral/reproducibility/integrity_report.json"
    assert_integrity_status_passed "experiments/scapre_informax_specificity/analysis/mi_channel_weighting/results/integrity_report.json"
    assert_integrity_status_passed "experiments/scapre_informax_specificity/analysis/alpha_channel_controls/formal_results/integrity_report.json"
    assert_integrity_status_passed "experiments/scapre_informax_specificity/analysis/projection_accumulation/formal_results/integrity_report.json"
    assert_integrity_status_passed "experiments/scapre_informax_specificity/analysis/projection_accumulation_direct_cos2/formal_results/integrity_report.json"
    assert_integrity_status_passed "experiments/scapre_informax_specificity/analysis/projection_accumulation_budget_matched_cos2/formal_results/integrity_report.json"
}

assert_oce_references() {
    local registry="orthogonal-concept-erasure/experiments/evaluation_references/registry.json"
    local reference_root="$REPO_ROOT/orthogonal-concept-erasure/experiments/evaluation_references/references"
    [[ -s "$REPO_ROOT/$registry" ]] || die "missing OCE reference registry"
    git -C "$REPO_ROOT" ls-files --error-unmatch -- "$registry" >/dev/null 2>&1 \
        || die "OCE reference registry is not tracked"
    [[ -d "$reference_root" ]] || die "missing OCE reference artifact root"
    [[ "$(awk '/"status"[[:space:]]*:[[:space:]]*"complete"/ { count += 1 } END { print count + 0 }' "$REPO_ROOT/$registry")" -ge 2 ]] \
        || die "expected complete first-1k and first-10k OCE references"
}

assert_oce_review_archive_if_needed() {
    local qualification="$REPO_ROOT/orthogonal-concept-erasure/experiments/oce_failure_image_qualification/outputs/qualification_v1"
    if [[ -d "$qualification/d1/images" || -d "$qualification/d3/images" ]]; then
        if ! compgen -G "$TMP_ROOT/oce_failure_image_review_v1_*.tar.gz" >/dev/null; then
            die "OCE failure-qualification raw images exist without the required review-image archive in anLi/tmp"
        fi
    fi
}

build_scapre_manifest() {
    local output="$1"
    local temporary="$output.tmp"
    local root=""
    : > "$temporary"
    (
        cd "$REPO_ROOT"
        for root in "${SCAPRE_ROOTS[@]}"; do
            [[ -d "$root" ]] || continue
            find "$root" -xdev -type f -printf '%p\t%s\t%b\t%T@\n'
        done
    ) | LC_ALL=C sort -u > "$temporary"
    mv "$temporary" "$output"
}

build_oce_manifest() {
    local output="$1"
    local temporary="$output.tmp"
    local existing=()
    local root=""
    for root in "${OCE_ROOTS[@]}"; do
        [[ -d "$REPO_ROOT/$root" ]] && existing+=("$root")
    done
    : > "$temporary"
    if [[ "${#existing[@]}" -gt 0 ]]; then
        (
            cd "$REPO_ROOT"
            find "${existing[@]}" -xdev -type f \
                \( \
                    \( \
                        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
                           -o -iname '*.gif' -o -iname '*.webp' -o -iname '*.bmp' \
                           -o -iname '*.tif' -o -iname '*.tiff' \) \
                        -a \( -path '*/images/*' -o -path '*/generated_images/*' \) \
                    \) \
                    -o -iname '*.safetensors' -o -iname '*.pt' \
                    -o -iname '*.pth' -o -iname '*.ckpt' \
                \) \
                -printf '%p\t%s\t%b\t%T@\n'
        ) | LC_ALL=C sort -u > "$temporary"
    fi
    mv "$temporary" "$output"
}

build_tmp_manifest() {
    local output="$1"
    local temporary="$output.tmp"
    : > "$temporary"
    if [[ -d "$TMP_ROOT" ]]; then
        (
            cd "$TMP_ROOT"
            find . -xdev -mindepth 1 -type f -printf '%p\t%s\t%b\t%T@\n'
        ) | LC_ALL=C sort -u > "$temporary"
    fi
    mv "$temporary" "$output"
}

manifest_count() {
    awk 'END { print NR + 0 }' "$1"
}

manifest_logical_bytes() {
    awk -F '\t' '{ total += $2 } END { printf "%.0f\n", total + 0 }' "$1"
}

manifest_allocated_bytes() {
    awk -F '\t' '{ total += ($3 * 512) } END { printf "%.0f\n", total + 0 }' "$1"
}

assert_manifest_has_no_tracked_files() {
    local manifest="$1"
    local label="$2"
    local candidates="$WORK_DIR/${label}_candidate_paths.txt"
    local tracked="$WORK_DIR/tracked_paths.txt"
    local overlap="$RUN_DIR/${label}_tracked_overlap.txt"
    cut -f 1 "$manifest" | LC_ALL=C sort -u > "$candidates"
    git -C "$REPO_ROOT" ls-files | LC_ALL=C sort -u > "$tracked"
    comm -12 "$candidates" "$tracked" > "$overlap"
    if [[ -s "$overlap" ]]; then
        echo "Tracked overlap:" >&2
        sed -n '1,40p' "$overlap" >&2
        die "$label candidate set contains Git-tracked files"
    fi
}

record_manifest_summary() {
    local scope="$1"
    local manifest="$2"
    local count=""
    local logical=""
    local allocated=""
    local digest=""
    count="$(manifest_count "$manifest")"
    logical="$(manifest_logical_bytes "$manifest")"
    allocated="$(manifest_allocated_bytes "$manifest")"
    digest="$(sha256_file "$manifest")"
    {
        printf 'scope=%s\n' "$scope"
        printf 'candidate_files=%s\n' "$count"
        printf 'logical_bytes=%s\n' "$logical"
        printf 'allocated_file_bytes=%s\n' "$allocated"
        printf 'manifest_sha256=%s\n' "$digest"
    } > "$RUN_DIR/${scope}_summary.txt"
    printf '%-7s %8s files  %12s logical  %12s allocated-file-bytes  sha256=%s\n' \
        "$scope" "$count" "$(format_bytes "$logical")" "$(format_bytes "$allocated")" "$digest"
}

verify_manifest_unchanged() {
    local scope="$1"
    local first="$2"
    local second="$WORK_DIR/${scope}_recheck.tsv"
    case "$scope" in
        scapre) build_scapre_manifest "$second" ;;
        oce) build_oce_manifest "$second" ;;
        tmp) build_tmp_manifest "$second" ;;
        *) die "internal unknown scope: $scope" ;;
    esac
    if ! cmp -s "$first" "$second"; then
        diff -u "$first" "$second" > "$RUN_DIR/${scope}_candidate_change.diff" || true
        die "$scope candidates changed between preflight and deletion"
    fi
}

delete_scapre() {
    local root=""
    for root in "${SCAPRE_ROOTS[@]}"; do
        [[ -d "$REPO_ROOT/$root" ]] || continue
        assert_inside_anli "$REPO_ROOT/$root"
        find "$REPO_ROOT/$root" -xdev -depth -mindepth 1 -delete
        rmdir -- "$REPO_ROOT/$root"
    done
}

delete_oce() {
    local relative=""
    while IFS=$'\t' read -r relative _logical _blocks _mtime; do
        [[ -n "$relative" ]] || continue
        [[ "$relative" != /* && "$relative" != *".."* ]] || die "unsafe OCE manifest path: $relative"
        [[ -f "$REPO_ROOT/$relative" && ! -L "$REPO_ROOT/$relative" ]] \
            || die "OCE candidate is no longer a regular file: $relative"
        assert_inside_anli "$REPO_ROOT/$relative"
        rm -- "$REPO_ROOT/$relative"
    done < "$RUN_DIR/oce_candidates.tsv"
}

delete_tmp_contents() {
    assert_inside_anli "$TMP_ROOT"
    [[ "$TMP_ROOT" == "$ANLI_ROOT/tmp" ]] || die "tmp deletion target changed unexpectedly: $TMP_ROOT"
    find "$TMP_ROOT" -xdev -depth -mindepth 1 -delete
}

show_status() {
    local latest_file="$RECORD_ROOT/latest_run"
    local latest=""
    if [[ ! -s "$latest_file" ]]; then
        echo "No storage cleanup run has been recorded."
        exit 0
    fi
    latest="$(tr -d '\n' < "$latest_file")"
    [[ "$latest" == "$RECORD_ROOT/"* && -d "$latest" ]] || die "invalid latest cleanup record: $latest"
    echo "Latest cleanup record: $latest"
    [[ -s "$latest/stage" ]] && echo "Stage: $(tr -d '\n' < "$latest/stage")"
    [[ -s "$latest/pid" ]] && echo "PID: $(tr -d '[:space:]' < "$latest/pid")"
    [[ -s "$latest/exit_code" ]] && echo "Exit code: $(tr -d '[:space:]' < "$latest/exit_code")"
    [[ -s "$latest/summary.txt" ]] && sed -n '1,240p' "$latest/summary.txt"
}

ACTION=""
SCOPE=""
CONFIRMATION=""

case "${1:-}" in
    --preview)
        ACTION="preview"
        SCOPE="${2:-all}"
        [[ "$#" -le 2 ]] || die "--preview accepts at most one scope"
        ;;
    --apply)
        ACTION="apply"
        SCOPE="${2:-}"
        [[ "$#" -eq 4 && "${3:-}" == "--confirm" ]] \
            || die "--apply requires one scope and --confirm TOKEN"
        CONFIRMATION="$4"
        ;;
    --status)
        [[ "$#" -eq 1 ]] || die "--status accepts no additional arguments"
        ACTION="status"
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    "")
        usage >&2
        exit 2
        ;;
    *)
        die "unknown argument: $1"
        ;;
esac

case "$SCOPE" in
    all|scapre|oce|tmp|"") ;;
    *) die "unknown scope: $SCOPE" ;;
esac
if [[ "$ACTION" == "apply" && "$SCOPE" == "all" ]]; then
    die "destructive all-scope mode is intentionally unsupported"
fi
if [[ "$ACTION" == "apply" ]]; then
    case "$SCOPE:$CONFIRMATION" in
        scapre:DELETE_SCAPRE_RUNS) ;;
        oce:DELETE_OCE_GENERATED_BINARIES) ;;
        tmp:EMPTY_ANLI_TMP_AFTER_LOCAL_BACKUP) ;;
        *) die "confirmation token does not match scope $SCOPE" ;;
    esac
fi

for command_name in realpath git find sort comm awk sed cut sha256sum cmp diff mountpoint tr tee rm rmdir mv mkdir; do
    require_command "$command_name"
done
find "$EXPECTED_REPO_ROOT" -maxdepth 0 -printf '' >/dev/null 2>&1 \
    || die "GNU find with -printf support is required"

ANLI_ROOT="$(realpath -e "$EXPECTED_ANLI_ROOT" 2>/dev/null || true)"
REPO_ROOT="$(realpath -e "$EXPECTED_REPO_ROOT" 2>/dev/null || true)"
TMP_ROOT="$(realpath -e "$EXPECTED_TMP_ROOT" 2>/dev/null || true)"
[[ "$ANLI_ROOT" == "$EXPECTED_ANLI_ROOT" ]] || die "unexpected or missing anLi root: ${ANLI_ROOT:-missing}"
[[ "$REPO_ROOT" == "$EXPECTED_REPO_ROOT" ]] || die "unexpected repository root: ${REPO_ROOT:-missing}"
[[ "$TMP_ROOT" == "$EXPECTED_TMP_ROOT" ]] || die "unexpected tmp root: ${TMP_ROOT:-missing}"
[[ ! -L "$EXPECTED_ANLI_ROOT" && ! -L "$EXPECTED_REPO_ROOT" && ! -L "$EXPECTED_TMP_ROOT" ]] \
    || die "anLi, repository, and tmp roots must not be symlinks"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
[[ "$SCRIPT_REPO_ROOT" == "$REPO_ROOT" ]] || die "script is not running from the expected server repository"

if [[ "$ACTION" == "status" ]]; then
    show_status
    exit 0
fi

[[ "${CONDA_DEFAULT_ENV:-}" == "MU" ]] || die "activate Conda environment MU before running cleanup"
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "main" ]] || die "server repository must be on branch main"
git -C "$REPO_ROOT" diff --quiet -- || die "tracked working tree has unstaged changes"
git -C "$REPO_ROOT" diff --cached --quiet -- || die "tracked working tree has staged changes"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=normal)" ]]; then
    die "server working tree is not clean; commit, ignore, or remove unrelated untracked files first"
fi

mkdir -p "$RECORD_ROOT"
[[ "$(realpath -e "$RECORD_ROOT")" == "$RECORD_ROOT" ]] || die "cleanup record root resolved unexpectedly"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    die "another storage cleanup holds lock: $LOCK_DIR"
fi

RUN_DIR=""
cleanup_exit() {
    local code=$?
    trap - EXIT
    set +e
    if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
        printf '%s\n' "$code" > "$RUN_DIR/exit_code"
        printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/finished_at_utc"
        if [[ "$code" -eq 0 ]]; then
            printf '%s\n' "completed" > "$RUN_DIR/stage"
            : > "$RUN_DIR/COMPLETED"
        else
            printf '%s\n' "failed" > "$RUN_DIR/stage"
            : > "$RUN_DIR/FAILED"
        fi
    fi
    rmdir "$LOCK_DIR" 2>/dev/null || true
    exit "$code"
}
trap cleanup_exit EXIT

RUN_ID="cleanup_$(date -u +%Y%m%dT%H%M%SZ)_${ACTION}_${SCOPE}"
RUN_DIR="$RECORD_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || die "cleanup record already exists: $RUN_DIR"
mkdir "$RUN_DIR"
WORK_DIR="$RUN_DIR/work"
mkdir "$WORK_DIR"
printf '%s\n' "$RUN_DIR" > "$RECORD_ROOT/latest_run"
printf '%s\n' "starting" > "$RUN_DIR/stage"
printf '%s\n' "$$" > "$RUN_DIR/pid"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/started_at_utc"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'action=%s\n' "$ACTION"
    printf 'scope=%s\n' "$SCOPE"
    printf 'anli_root=%s\n' "$ANLI_ROOT"
    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'tmp_root=%s\n' "$TMP_ROOT"
    printf 'git_commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git -C "$REPO_ROOT" branch --show-current)"
    printf 'conda_default_env=%s\n' "${CONDA_DEFAULT_ENV:-}"
} > "$RUN_DIR/run_metadata.txt"
git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=normal > "$RUN_DIR/git_status_before.txt"

if [[ "$SCOPE" == "all" || "$SCOPE" == "scapre" ]]; then
    assert_required_scapre_results
    for root in "${SCAPRE_ROOTS[@]}"; do
        hardlink_candidate=""
        assert_repo_relative_root "$root"
        if [[ -d "$REPO_ROOT/$root" ]]; then
            assert_no_unsafe_name "$REPO_ROOT/$root"
            assert_no_nested_mount "$REPO_ROOT/$root"
            assert_no_live_pid_files "$REPO_ROOT/$root"
            assert_only_regular_files_and_directories "$REPO_ROOT/$root"
            hardlink_candidate="$(find "$REPO_ROOT/$root" -xdev -type f -links +1 -print -quit)"
            if [[ -n "$hardlink_candidate" ]]; then
                die "hardlinked file found inside ScaPre deletion root: $root"
            fi
        fi
    done
    build_scapre_manifest "$RUN_DIR/scapre_candidates.tsv"
    assert_manifest_has_no_tracked_files "$RUN_DIR/scapre_candidates.tsv" "scapre"
    record_manifest_summary "scapre" "$RUN_DIR/scapre_candidates.tsv"
fi

if [[ "$SCOPE" == "all" || "$SCOPE" == "oce" ]]; then
    assert_oce_references
    assert_oce_review_archive_if_needed
    for root in "${OCE_ROOTS[@]}"; do
        assert_repo_relative_root "$root"
        if [[ -d "$REPO_ROOT/$root" ]]; then
            assert_no_unsafe_name "$REPO_ROOT/$root"
            assert_no_live_pid_files "$REPO_ROOT/$root"
        fi
    done
    build_oce_manifest "$RUN_DIR/oce_candidates.tsv"
    assert_manifest_has_no_tracked_files "$RUN_DIR/oce_candidates.tsv" "oce"
    record_manifest_summary "oce" "$RUN_DIR/oce_candidates.tsv"
fi

if [[ "$SCOPE" == "all" || "$SCOPE" == "tmp" ]]; then
    assert_no_unsafe_name "$TMP_ROOT"
    assert_no_nested_mount "$TMP_ROOT"
    assert_no_live_pid_files "$TMP_ROOT"
    assert_only_regular_files_and_directories "$TMP_ROOT"
    build_tmp_manifest "$RUN_DIR/tmp_candidates.tsv"
    record_manifest_summary "tmp" "$RUN_DIR/tmp_candidates.tsv"
fi

if [[ "$ACTION" == "preview" ]]; then
    {
        echo "Read-only cleanup preview completed."
        echo "No file was deleted."
        echo "Record: $RUN_DIR"
    } | tee "$RUN_DIR/summary.txt"
    exit 0
fi

SELECTED_MANIFEST="$RUN_DIR/${SCOPE}_candidates.tsv"
[[ -s "$SELECTED_MANIFEST" ]] || die "no files remain for scope $SCOPE"
verify_manifest_unchanged "$SCOPE" "$SELECTED_MANIFEST"
printf '%s\n' "deleting" > "$RUN_DIR/stage"

case "$SCOPE" in
    scapre) delete_scapre ;;
    oce) delete_oce ;;
    tmp) delete_tmp_contents ;;
    *) die "internal unsupported apply scope: $SCOPE" ;;
esac

printf '%s\n' "verifying" > "$RUN_DIR/stage"
POST_MANIFEST="$WORK_DIR/${SCOPE}_post.tsv"
case "$SCOPE" in
    scapre) build_scapre_manifest "$POST_MANIFEST" ;;
    oce) build_oce_manifest "$POST_MANIFEST" ;;
    tmp) build_tmp_manifest "$POST_MANIFEST" ;;
esac
[[ ! -s "$POST_MANIFEST" ]] || die "$SCOPE cleanup left allowlisted candidates behind"

git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=normal > "$RUN_DIR/git_status_after.txt"
[[ ! -s "$RUN_DIR/git_status_after.txt" ]] || die "Git working tree changed during cleanup"
[[ -s "$REPO_ROOT/orthogonal-concept-erasure/experiments/evaluation_references/registry.json" ]] \
    || die "OCE reference registry missing after cleanup"

DELETED_FILES="$(manifest_count "$SELECTED_MANIFEST")"
DELETED_LOGICAL="$(manifest_logical_bytes "$SELECTED_MANIFEST")"
DELETED_ALLOCATED="$(manifest_allocated_bytes "$SELECTED_MANIFEST")"
{
    echo "Cleanup completed and postconditions passed."
    echo "Scope: $SCOPE"
    echo "Deleted files: $DELETED_FILES"
    echo "Deleted logical bytes: $DELETED_LOGICAL ($(format_bytes "$DELETED_LOGICAL"))"
    echo "Deleted allocated file bytes: $DELETED_ALLOCATED ($(format_bytes "$DELETED_ALLOCATED"))"
    echo "Git status remained clean."
    echo "OCE evaluation reference registry remained present."
    echo "Record: $RUN_DIR"
    echo "Next: rerun maintenance/server_storage_inventory/audit_server_storage.sh"
} | tee "$RUN_DIR/summary.txt"
