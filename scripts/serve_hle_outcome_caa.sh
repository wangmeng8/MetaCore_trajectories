#!/usr/bin/env bash
# Invoke inside the GPU serving container, with vectors mounted read-only.
set -euo pipefail
ROOT_DIR="${METACOREBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${MODEL_PATH:?Set local MODEL_PATH inside container}"
: "${SERVED_MODEL_NAME:?Set the model id used by the HLE evaluator}"
: "${OUTCOME_CAA_VECTOR_PATH:?Set this model's HLE global steering_vector.pt}"
: "${OUTCOME_CAA_VECTOR_ROOT:?Set the read-only vector mount root}"
: "${VLLM_VANILLA_ARGS_FILE:?Set a JSON array of this model's validated Vanilla serving arguments}"
: "${OUTCOME_CAA_SERVICE_MANIFEST:?Set a writable service JSON manifest path}"
: "${OUTCOME_CAA_AUDIT_DIR:?Set a writable directory for per-worker hook audit logs}"
ENV_FILE="${VLLM_HOOK_ENV_FILE:-${VLLM_HOOK_WORK_DIR:-/tmp/hle-vllm-hook}/outcome_caa.env}"
[[ -f "$ENV_FILE" ]] || { printf 'Missing plugin environment: %s\n' "$ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"
export MODEL_PATH SERVED_MODEL_NAME OUTCOME_CAA_VECTOR_PATH OUTCOME_CAA_VECTOR_ROOT
export VLLM_VANILLA_ARGS_FILE OUTCOME_CAA_SERVICE_MANIFEST OUTCOME_CAA_AUDIT_DIR
export VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOOK_WORKER=steer
# V1 engine and V1 model runner are separate settings. CAA requires input_batch.
export VLLM_USE_V2_MODEL_RUNNER=0
# If VLLM_PLUGINS is restricted, both upstream entry points must be enabled.
export VLLM_PLUGINS=hook_registry,vllm_hook
exec "${VLLM_HOOK_PYTHON:-python3}" "$ROOT_DIR/scripts/serve_hle_outcome_caa.py"
