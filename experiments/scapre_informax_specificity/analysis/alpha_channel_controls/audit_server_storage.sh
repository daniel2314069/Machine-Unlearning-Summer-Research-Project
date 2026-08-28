#!/usr/bin/env bash
set -euo pipefail

SCAN_ROOT="/home/tslin/Documents/jupyter_data/anLi"
REPORT_DIR="$SCAN_ROOT/tmp"
TIMESTAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
REPORT="$REPORT_DIR/server_storage_inventory_${TIMESTAMP}.txt"

[[ -d "$SCAN_ROOT" ]] || {
  echo "ERROR: expected server storage root not found: $SCAN_ROOT" >&2
  exit 2
}
mkdir -p "$REPORT_DIR"
[[ ! -e "$REPORT" ]] || {
  echo "ERROR: refusing to overwrite existing report: $REPORT" >&2
  exit 2
}
command -v find >/dev/null
command -v du >/dev/null
command -v sort >/dev/null
command -v numfmt >/dev/null
command -v sha256sum >/dev/null

human_size() {
  numfmt --to=iec-i --suffix=B "$1"
}

print_file_ranking() {
  local title="$1"
  shift
  echo "$title"
  while IFS=$'\t' read -r bytes path; do
    [[ -n "${bytes:-}" && -n "${path:-}" ]] || continue
    printf '%12s  %s\n' "$(human_size "$bytes")" "$path"
  done < <(find "$SCAN_ROOT" -xdev -type f "$@" -printf '%s\t%p\n' | sort -nr)
  echo
}

{
  echo "SERVER STORAGE INVENTORY"
  echo "generated_at_utc: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "hostname: $(hostname)"
  echo "scan_root: $SCAN_ROOT"
  echo "mode: read-only metadata scan"
  echo

  echo "FILESYSTEM CAPACITY"
  df -h "$SCAN_ROOT"
  echo

  echo "TOP-LEVEL TOTALS"
  while IFS= read -r path; do
    bytes="$(du -s -B1 "$path" | cut -f1)"
    printf '%12s  %s\n' "$(human_size "$bytes")" "$path"
  done < <(find "$SCAN_ROOT" -xdev -mindepth 1 -maxdepth 1 -print | sort)
  echo

  echo "DIRECTORY TOTALS THROUGH DEPTH 4 (LARGEST FIRST)"
  while IFS=$'\t' read -r bytes path; do
    [[ -n "${bytes:-}" && -n "${path:-}" ]] || continue
    printf '%12s  %s\n' "$(human_size "$bytes")" "$path"
  done < <(du -x -B1 "$SCAN_ROOT" |
    awk -F $'\t' -v root="$SCAN_ROOT" '
      {
        relative=$2
        sub("^" root "/?", "", relative)
        if (relative == "") next
        depth=split(relative, parts, "/")
        if (depth <= 4) print $0
      }
    ' | sort -nr)
  echo

  print_file_ranking "ALL FILES AT LEAST 100 MiB (LARGEST FIRST)" -size +100M
  print_file_ranking "ALL .pt FILES (LARGEST FIRST)" -name '*.pt'

  echo "ALL DIRECTORY TOTALS (PATH ORDER)"
  while IFS=$'\t' read -r bytes path; do
    [[ -n "${bytes:-}" && -n "${path:-}" ]] || continue
    printf '%12s  %s\n' "$(human_size "$bytes")" "$path"
  done < <(du -x -B1 "$SCAN_ROOT" | sort -t $'\t' -k2,2)
  echo

  echo "FILE COUNTS BY SUFFIX (TOP 80)"
  find "$SCAN_ROOT" -xdev -type f -printf '%f\n' |
    awk '
      {
        name=$0
        if (name !~ /\./) { suffix="[no suffix]" }
        else { sub(/^.*\./, ".", name); suffix=tolower(name) }
        count[suffix]++
      }
      END { for (suffix in count) print count[suffix], suffix }
    ' | sort -nr | sed -n '1,80p'
  echo

  echo "FULL TREE / INVENTORY"
  if command -v tree >/dev/null; then
    TREE_HELP="$(tree --help 2>&1 || true)"
    if [[ "$TREE_HELP" == *"--du"* ]]; then
      tree -a -h --du "$SCAN_ROOT"
    else
      tree -a -h "$SCAN_ROOT"
    fi
  else
    echo "tree command unavailable; using type, byte size, and absolute path."
    find "$SCAN_ROOT" -xdev -printf '%y\t%s\t%p\n' | sort -t $'\t' -k3,3
  fi
} > "$REPORT"

REPORT_SHA256="$(sha256sum "$REPORT" | awk '{print $1}')"
REPORT_SIZE="$(wc -c < "$REPORT" | tr -d ' ')"

echo "Storage inventory completed. No files were modified or deleted."
echo "Report: $REPORT"
echo "Size: $(human_size "$REPORT_SIZE")"
echo "SHA-256: $REPORT_SHA256"
echo
echo "Run this exact command on the Mac:"
echo "scp tslin:$REPORT ~/Downloads/"
