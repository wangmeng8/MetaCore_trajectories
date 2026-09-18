#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${METACOREBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${OUTCOME_CAA_MODEL_KEY:?Set qwen36_27b, gemma4_31b_it or gptoss_20b}"
: "${OUTCOME_CAA_VECTOR_PATH:?Set matching HLE global steering_vector.pt at the same absolute mount path}"
: "${OUTCOME_CAA_SERVICE_MANIFEST:?Set service manifest produced by serve_hle_outcome_caa.sh}"
: "${HLE_MODEL:?Set served model name}"
: "${HLE_BASE_URL:?Set agent API base URL}"
# Require the original per-model budgets; do not borrow another model's values.
: "${HLE_MAX_COMPLETION_TOKENS:?Set the value used for the target model HLE Vanilla run}"
: "${HLE_MAX_ITERATIONS:?Set the Vanilla max iterations}"
: "${HLE_MAX_WORKERS:?Set the Vanilla concurrency}"
: "${HLE_MAX_RETRIES:?Set the Vanilla API retries}"
: "${HLE_PROCESS_RETRIES:?Set the Vanilla process retries}"
: "${HLE_QUESTION_TIMEOUT_SECONDS:?Set the Vanilla question timeout}"
: "${HLE_API_TIMEOUT_SECONDS:?Set the Vanilla API timeout}"
: "${HLE_STALL_TIMEOUT_SECONDS:?Set the Vanilla stall timeout}"
PYTHON_BIN="${HLE_PYTHON:-$ROOT_DIR/.venv-hle/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="${HLE_PYTHON:-python3}"
case "$OUTCOME_CAA_MODEL_KEY" in
  qwen36_27b) DEFAULT_LAYER=23 ;;
  gemma4_31b_it) DEFAULT_LAYER=30 ;;
  gptoss_20b) DEFAULT_LAYER=12 ;;
  *) printf 'Unsupported model key\n' >&2; exit 2 ;;
esac
LAYER="${OUTCOME_CAA_VECTOR_LAYER:-$DEFAULT_LAYER}"
ALPHA="${OUTCOME_CAA_ALPHA:-1.0}"
export OUTCOME_CAA_MODEL_KEY OUTCOME_CAA_SERVICE_MANIFEST
# The Python runner also rejects Prompt-Reg loaded from a .env file.
[[ -z "${HLE_SYSTEM_PROMPT_FILE:-}" ]] || { printf 'Remove HLE_SYSTEM_PROMPT_FILE: Outcome-CAA cannot include Prompt-Reg.\n' >&2; exit 2; }
BODY="$("$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_hle_outcome_caa.py" request-body \
  --vector "$OUTCOME_CAA_VECTOR_PATH" --layer "$LAYER" --alpha "$ALPHA")"
ARGS=(
  --repo-dir "${HLE_WITH_TOOLS_REPO_DIR:-$ROOT_DIR/.external/hle_with_tools}"
  --data-path "${HLE_DATA_PATH:-$ROOT_DIR/data/HLE/WebThinker_test_500_hle_with_tools.json}"
  --model "$HLE_MODEL" --base-url "$HLE_BASE_URL" --num-rollouts "${HLE_NUM_ROLLOUTS:-1}"
  --max-workers "$HLE_MAX_WORKERS" --max-retries "$HLE_MAX_RETRIES" --process-retries "$HLE_PROCESS_RETRIES"
  --max-completion-tokens "$HLE_MAX_COMPLETION_TOKENS" --max-iterations "$HLE_MAX_ITERATIONS"
  --question-timeout-seconds "$HLE_QUESTION_TIMEOUT_SECONDS" --api-timeout-seconds "$HLE_API_TIMEOUT_SECONDS"
  --stall-timeout-seconds "$HLE_STALL_TIMEOUT_SECONDS"
  --run-id "${HLE_RUN_ID:-hle_${OUTCOME_CAA_MODEL_KEY}_outcome_caa_l${LAYER}_a${ALPHA}_v2}"
  --output-dir "${OUTPUT_DIR:-$ROOT_DIR/outputs/runs}" --no-uv --agent-extra-body-json "$BODY"
)
if [[ -n "${HLE_NUM_TASKS:-}" ]]; then
  export HLE_ALL_TASKS=0
  ARGS+=(--num-tasks "$HLE_NUM_TASKS")
else
  ARGS+=(--all-tasks)
fi
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_hle_with_tools.py" "${ARGS[@]}" "$@"
