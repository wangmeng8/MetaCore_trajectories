#!/usr/bin/env bash
# Run INSIDE the serving container/environment. Never in .venv-hle.
set -euo pipefail
ROOT_DIR="${METACOREBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${VLLM_HOOK_PYTHON:-python3}"
WORK_DIR="${VLLM_HOOK_WORK_DIR:-/tmp/hle-vllm-hook}"
HOOK_COMMIT=e3d6885899264c81ac61c403c8b56efaf5a02dab
HOOK_SOURCE="${VLLM_HOOK_SOURCE_DIR:-https://github.com/IBM/vLLM-Hook.git}"
ENV_FILE="${VLLM_HOOK_ENV_FILE:-$WORK_DIR/outcome_caa.env}"
"$PYTHON_BIN" -c 'import vllm, torch, safetensors; print("Serving environment:", vllm.__version__, torch.__version__)'
mkdir -p "$WORK_DIR"
BUILD_DIR="$(mktemp -d "$WORK_DIR/build.XXXXXX")"
SOURCE_DIR="$BUILD_DIR/source"
INSTALL_DIR="$BUILD_DIR/site-packages"
# Clone a local read-only source or the public upstream into a writable build.
git clone --no-hardlinks "$HOOK_SOURCE" "$SOURCE_DIR"
if ! git -C "$SOURCE_DIR" cat-file -e "$HOOK_COMMIT^{commit}"; then
  git -C "$SOURCE_DIR" fetch --depth 1 origin "$HOOK_COMMIT"
fi
git -C "$SOURCE_DIR" checkout --detach "$HOOK_COMMIT"
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$HOOK_COMMIT" ]]
git -C "$SOURCE_DIR" apply --check "$ROOT_DIR/patches/vllm_hook/0001-hle-outcome-caa-worker.patch"
git -C "$SOURCE_DIR" apply "$ROOT_DIR/patches/vllm_hook/0001-hle-outcome-caa-worker.patch"
# No torch/vLLM upgrades, no writes/build artifacts in a read-only mount.
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation --target "$INSTALL_DIR" \
  "$SOURCE_DIR/vllm_hook_plugins" 'zstandard==0.23.0'
export PYTHONPATH="$INSTALL_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -c 'from vllm_hook_plugins.workers.steer_activation_worker import OUTCOME_CAA_VERSION; assert OUTCOME_CAA_VERSION == "hle-outcome-caa-v2"; print(OUTCOME_CAA_VERSION)'
"$PYTHON_BIN" -m pip freeze > "$BUILD_DIR/requirements-serving.txt"
mkdir -p "$(dirname "$ENV_FILE")"
cat > "$ENV_FILE" <<EOF
export VLLM_HOOK_INSTALL_DIR=$(printf '%q' "$INSTALL_DIR")
export VLLM_HOOK_PYTHON=$(printf '%q' "$(command -v "$PYTHON_BIN")")
export PYTHONPATH=$(printf '%q' "$INSTALL_DIR")\${PYTHONPATH:+:\$PYTHONPATH}
export VLLM_HOOK_WORKER=steer
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
EOF
printf 'Prepared %s\nVersions: %s\nEnvironment: source %q\n' "$HOOK_COMMIT" "$BUILD_DIR/requirements-serving.txt" "$ENV_FILE"
