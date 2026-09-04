#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${METACOREBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${HLE_PYTHON:-python}"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_hle_with_tools.py" \
  --repo-dir "$ROOT_DIR/.external/hle_with_tools" \
  --data-path "$ROOT_DIR/data/HLE/WebThinker_test_500_hle_with_tools.json" \
  --model "${HLE_MODEL:-qwen3.6-27b}" \
  --all-tasks \
  --max-workers "${HLE_MAX_WORKERS:-8}" \
  --max-retries "${HLE_MAX_RETRIES:-3}" \
  --process-retries "${HLE_PROCESS_RETRIES:-1}" \
  --max-completion-tokens "${HLE_MAX_COMPLETION_TOKENS:-100000}" \
  --max-iterations "${HLE_MAX_ITERATIONS:-15}" \
  --question-timeout-seconds "${HLE_QUESTION_TIMEOUT_SECONDS:-1800}" \
  --system-prompt-file "$ROOT_DIR/prompts/hle/prompt_reg.txt" \
  --run-id "${HLE_RUN_ID:-hle_qwen36_27b_WebThinker_prompt_reg}" \
  --output-dir "${OUTPUT_DIR:-$ROOT_DIR/result}" \
  --no-uv \
  --num-rollouts 1
