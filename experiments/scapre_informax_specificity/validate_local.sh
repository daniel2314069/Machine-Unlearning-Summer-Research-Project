#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.json"
PUBLIC_CSV="$REPO_ROOT/scapre/eval/datasets/imagenet-15.csv"
DERIVED_CSV="$REPO_ROOT/orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/datasets/imagenet-confuse5-derived-25.csv"

required=(
  AUDIT.md
  README.md
  aggregate_results.py
  build_protocol.py
  config.json
  download_results.sh
  evaluate_confuse5.py
  package_results.sh
  prefetch_assets.py
  requirements_server.txt
  run_server.sh
  server_worker.sh
  setup_server.sh
  status_server.sh
  worker.py
)
for path in "${required[@]}"; do
  [[ -f "$SCRIPT_DIR/$path" ]] || {
    echo "ERROR: missing $SCRIPT_DIR/$path" >&2
    exit 1
  }
done

command -v jq >/dev/null || {
  echo "ERROR: jq is required for the static JSON checks" >&2
  exit 1
}

jq -e '
  .variants == ["official", "matched_retain"] and
  (.groups | length) == 5 and
  (all(.groups[]; (.targets | length) == 2 and (.retains | length) == 3)) and
  ([.groups[].targets[]] | length) == 10 and
  ([.groups[].retains[]] | length) == 15 and
  .edit.num_positive == 5 and
  .edit.num_negative == 5 and
  .evaluation.prompt_template == "an image of a {concept}" and
  .evaluation.protocol_dataset_sha256 == "f473503dd5a008f989a107e5adfe0749e9e2e77d8f613f2b7a4aae8bd87301d9" and
  .evaluation.formal_images_per_concept == 120
' "$CONFIG" >/dev/null

awk -F, '
  NR == 1 {
    sub(/\r$/, "", $4)
    if ($1 != "case_number" || $2 != "prompt" || $3 != "class" || $4 != "evaluation_seed") exit 10
    next
  }
  {
    sub(/\r$/, "", $4)
    if (NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 !~ /^[0-9]+$/) exit 11
    if (tolower($2) != "an image of a " tolower($3)) exit 15
    if (seen_case[$1]++) exit 12
    class_count[tolower($3)]++
    rows++
  }
  END {
    if (rows != 7500 || length(class_count) != 15) exit 13
    for (class_name in class_count) if (class_count[class_name] != 500) exit 14
  }
' "$PUBLIC_CSV"

present=0
missing=0
while IFS= read -r concept; do
  if awk -F, -v wanted="$concept" 'NR > 1 && tolower($3) == wanted {found=1; exit} END {exit !found}' "$PUBLIC_CSV"; then
    present=$((present + 1))
  else
    missing=$((missing + 1))
  fi
done < <(jq -r '.groups[] | (.targets[], .retains[])' "$CONFIG")
[[ "$present" -eq 15 && "$missing" -eq 10 ]] || {
  echo "ERROR: expected public Confuse coverage of 15 present and 10 missing concepts" >&2
  exit 1
}

derived_hash="$(shasum -a 256 "$DERIVED_CSV" | awk '{print $1}')"
[[ "$derived_hash" == "f473503dd5a008f989a107e5adfe0749e9e2e77d8f613f2b7a4aae8bd87301d9" ]] || {
  echo "ERROR: project-derived 25-class dataset hash mismatch" >&2
  exit 1
}
awk -F, '
  NR == 1 { next }
  {
    if (NF != 4 || seen_case[$1]++) exit 20
    class_name = tolower($3)
    if (tolower($2) != "an image of a " class_name) exit 21
    sub(/\r$/, "", $4)
    if ($4 !~ /^[0-9]+$/) exit 22
    class_seed[class_name, class_count[class_name]] = $4
    class_count[class_name]++
    rows++
  }
  END {
    if (rows != 12500 || length(class_count) != 25) exit 23
    for (name in class_count) if (class_count[name] != 500) exit 24
    pairs["chesapeake bay retriever"] = "german shepherd"
    pairs["pug"] = "german shepherd"
    pairs["siamese cat"] = "persian cat"
    pairs["egyptian cat"] = "persian cat"
    pairs["fig"] = "pomegranate"
    pairs["granny smith"] = "pomegranate"
    pairs["catamaran"] = "speedboat"
    pairs["schooner"] = "speedboat"
    pairs["rugby ball"] = "tennis ball"
    pairs["ping-pong ball"] = "tennis ball"
    for (derived in pairs) {
      source = pairs[derived]
      for (row_index = 0; row_index < 500; row_index++) {
        if (class_seed[derived, row_index] != class_seed[source, row_index]) exit 25
      }
    }
  }
' "$DERIVED_CSV"

diff -u <(tr -d '\r' < "$PUBLIC_CSV") <(head -n 7501 "$DERIVED_CSV") >/dev/null

for script in "$SCRIPT_DIR"/*.sh; do
  bash -n "$script"
done

rg -q "default='official'" "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q "choices=\['official', 'matched-retain'\]" "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q 'negative_base_indices = torch.arange\(num_pos' "$REPO_ROOT/scapre/edit/erase_scale.py"
rg -q 'default=130' "$REPO_ROOT/scapre/eval/benchmarking/object_erase.py"

git -C "$REPO_ROOT" diff --check
echo "Static validation passed. No Python or model code was executed."
