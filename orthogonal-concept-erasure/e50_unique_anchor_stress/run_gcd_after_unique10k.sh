#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OFFICIAL_URL="https://github.com/Giphy/celeb-detection-oss.git"
OFFICIAL_COMMIT="1f28b24370f67e47e806cc57b3c38fbe42e302ab"
MODEL_URL="https://s3.amazonaws.com/giphy-public/models/celeb-detection/resources.tar.gz"
EXTERNAL_ROOT="$HERE/external"
GCD_ROOT="$EXTERNAL_ROOT/celeb-detection-oss"
RUNTIME_ROOT="$HERE/gcd_runtime"
SITE_PACKAGES="$RUNTIME_ROOT/site-packages"
ARCHIVE="$RUNTIME_ROOT/resources.tar.gz"
PATCH_FILE="$HERE/gcd_official_py310.patch"
SETUP_MANIFEST="$HERE/gcd_metrics/setup_manifest.json"

run_py310() {
  conda run --no-capture-output -n py310 "$@"
}

run_project_python() {
  conda run --no-capture-output -n py310 bash -c '
export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
shift
exec python "$@"
' bash "$SITE_PACKAGES" "$@"
}

run_py310 python "$HERE/verify_gcd_automation.py" unique

mkdir -p "$EXTERNAL_ROOT" "$RUNTIME_ROOT" "$HERE/gcd_metrics"
if [[ ! -d "$GCD_ROOT/.git" ]]; then
  clone_target="$EXTERNAL_ROOT/celeb-detection-oss.partial.$(date -u +%Y%m%dT%H%M%SZ).$$"
  git clone --no-tags "$OFFICIAL_URL" "$clone_target"
  git -C "$clone_target" checkout --detach "$OFFICIAL_COMMIT"
  mv "$clone_target" "$GCD_ROOT"
fi

actual_commit="$(git -C "$GCD_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$OFFICIAL_COMMIT" ]]; then
  echo "Unexpected GCD commit: $actual_commit" >&2
  exit 1
fi

if git -C "$GCD_ROOT" apply --check "$PATCH_FILE"; then
  git -C "$GCD_ROOT" apply "$PATCH_FILE"
elif ! git -C "$GCD_ROOT" apply --reverse --check "$PATCH_FILE"; then
  echo "Official GCD checkout is neither pristine nor compat-patched." >&2
  exit 1
fi

if [[ ! -f "$ARCHIVE" ]]; then
  curl --fail --location --retry 5 --retry-delay 10 \
    --continue-at - --output "$ARCHIVE.partial" "$MODEL_URL"
  mv "$ARCHIVE.partial" "$ARCHIVE"
fi
tar -tzf "$ARCHIVE" >/dev/null

if [[ ! -f "$GCD_ROOT/examples/resources/face_recognition/best_model_states.pkl" ]]; then
  tar -xzf "$ARCHIVE" -C "$GCD_ROOT/examples"
fi

for required in \
  "$GCD_ROOT/examples/resources/face_detection/det1.npy" \
  "$GCD_ROOT/examples/resources/face_detection/det2.npy" \
  "$GCD_ROOT/examples/resources/face_detection/det3.npy" \
  "$GCD_ROOT/examples/resources/face_recognition/labels.csv" \
  "$GCD_ROOT/examples/resources/face_recognition/best_model_states.pkl"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing GCD resource: $required" >&2
    exit 1
  fi
done

if [[ ! -f "$SITE_PACKAGES/tensorflow/__init__.py" ]]; then
  mkdir -p "$SITE_PACKAGES"
  run_py310 python -m pip install \
    --disable-pip-version-check \
    --target "$SITE_PACKAGES" \
    "numpy==1.26.4" \
    "tensorflow-cpu==2.15.1" \
    "opencv-python-headless==4.11.0.86" \
    "python-dotenv==1.0.1"
fi

resources_path="$GCD_ROOT/examples/resources"
{
  printf 'APP_USE_CUDA=false\n'
  printf 'USE_CUDA=true\n'
  printf 'APP_DATA_DIR=%s\n' "$resources_path"
  printf 'APP_RECOGNITION_WEIGHTS_FILE=face_recognition/best_model_states.pkl\n'
  printf 'APP_FACE_MARGIN=0.2\n'
  printf 'APP_FACE_SIZE=224\n'
} >"$GCD_ROOT/.env.tmp"
mv "$GCD_ROOT/.env.tmp" "$GCD_ROOT/.env"

archive_sha256="$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)"
patch_sha256="$(sha256sum "$PATCH_FILE" | cut -d ' ' -f 1)"
cat >"$SETUP_MANIFEST.tmp" <<EOF
{
  "status": "complete",
  "official_repository": "$OFFICIAL_URL",
  "official_commit": "$OFFICIAL_COMMIT",
  "model_archive_url": "$MODEL_URL",
  "model_archive_sha256": "$archive_sha256",
  "compatibility_patch_sha256": "$patch_sha256",
  "python_environment": "conda py310 with experiment-local site-packages",
  "tensorflow_device": "CPU",
  "recognizer_device": "CUDA",
  "resources_preserved": true
}
EOF
mv "$SETUP_MANIFEST.tmp" "$SETUP_MANIFEST"

run_project_python "$HERE/run_experiment.py" gcd --gcd-project-root "$GCD_ROOT"
run_project_python "$HERE/verify_gcd_automation.py" gcd
