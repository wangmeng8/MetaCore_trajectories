#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${METACOREBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEFAULT_HLE_PYTHON="$ROOT_DIR/.external/hle_with_tools/.venv/bin/python"
if [[ ! -x "$DEFAULT_HLE_PYTHON" ]]; then
  DEFAULT_HLE_PYTHON="python"
fi
PYTHON_BIN="${HLE_PYTHON:-$DEFAULT_HLE_PYTHON}"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_hle_with_tools.py" \
  --repo-dir "$ROOT_DIR/.external/hle_with_tools" \
  --data-path "$ROOT_DIR/data/HLE/WebThinker_test_500_hle_with_tools.json" \
  --model "${HLE_MODEL:-qwen3.6-27b}" \
  --all-tasks \
  --max-workers "${HLE_MAX_WORKERS:-8}" \
  --max-retries "${HLE_MAX_RETRIES:-1}" \
  --process-retries "${HLE_PROCESS_RETRIES:-1}" \
  --max-completion-tokens "${HLE_MAX_COMPLETION_TOKENS:-40000}" \
  --max-iterations "${HLE_MAX_ITERATIONS:-15}" \
  --api-timeout-seconds "${HLE_API_TIMEOUT_SECONDS:-300}" \
  --question-timeout-seconds "${HLE_QUESTION_TIMEOUT_SECONDS:-900}" \
  --stall-timeout-seconds "${HLE_STALL_TIMEOUT_SECONDS:-1200}" \
  --system-prompt-file "$ROOT_DIR/prompts/hle/prompt_reg.txt" \
  --run-id "${HLE_RUN_ID:-hle_qwen36_27b_WebThinker_prompt_reg}" \
  --output-dir "${OUTPUT_DIR:-$ROOT_DIR/result}" \
  --no-uv \
  --num-rollouts 1
