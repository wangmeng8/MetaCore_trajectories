# HLE With Tools

This repository keeps the official `activeloopai/hle_with_tools` checkout
outside Git under `.external/`. Clone the official repository and apply the
versioned local patch before installing it:

```bash
python -m pip install uv
git clone https://github.com/activeloopai/hle_with_tools .external/hle_with_tools
git -C .external/hle_with_tools checkout 7a348ce
git -C .external/hle_with_tools apply ../../patches/hle_with_tools/local_hle_with_tools_fixes.patch
cd .external/hle_with_tools
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
cd ../..
```

The patch contains the Prompt-Reg system-message support, resumable
per-question state, whole-question timeouts, code-execution process isolation,
subprocess-isolated DDGS search timeouts, and subprocess-isolated webpage
content fetches. It is based on official commit `7a348ce` (`bigmain`).

Configure an OpenAI-compatible model endpoint. Do not store real keys in Git:

```bash
export OPENAI_API_KEY='your-model-api-key'
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'

export HLE_CODE_EXECUTION_TIMEOUT=30
export HLE_CODE_EXECUTION_STARTUP_TIMEOUT=120
export HLE_CODE_EXECUTION_START_METHOD=spawn
export HLE_DUPLICATE_TOOL_THRESHOLD=2
export HLE_WEB_SEARCH_TIMEOUT=30
export HLE_WEB_FETCH_TIMEOUT=10
export HLE_SCIENTIFIC_SEARCH_TIMEOUT=20
```

Run the 500-question Prompt-Reg condition:

```bash
bash scripts/run_hle_prompt_reg.sh
```

The launcher uses:

- `data/HLE/WebThinker_test_500_hle_with_tools.json`
- `prompts/hle/prompt_reg.txt`
- eight workers and one rollout by default
- a 1,800-second whole-question timeout
- run id `hle_qwen36_27b_WebThinker_prompt_reg`

Override defaults with environment variables such as `HLE_MODEL`,
`HLE_MAX_WORKERS`, `HLE_RUN_ID`, `HLE_QUESTION_TIMEOUT_SECONDS`, and
`OUTPUT_DIR`.

Run the regression tests with:

```bash
python -m unittest discover -s tests -p test_run_hle_with_tools.py
.external/hle_with_tools/.venv/bin/python -m unittest discover \
  -s .external/hle_with_tools/tests
```
