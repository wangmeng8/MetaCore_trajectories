# Tau2 GPT-5.5 Trajectory MVP

This is a minimal local project for running Tau2-bench with OpenAI `gpt-5.5` and saving complete raw and normalized agentic trajectories for later analysis or training-data construction.

The MVP goal is local reproducibility, not leaderboard submission.

## What It Does

- Runs a small Tau2 smoke test by default:
  - domain: `airline`
  - num tasks: `2`
  - num trials: `1`
- Uses environment variables for model and run settings.
- Keeps Tau2 raw simulation files.
- Builds normalized per-task trajectory JSON files.
- Exports a benchmark-oriented JSON/JSONL format with `benchmark`, `task.instruction`, and serialized `trajectory_text`.
- Writes `trajectories.jsonl`, `summary.json`, and `manifest.json`.
- Also includes MVP runners for Terminal-Bench 2.1, OSWorld-Verified, and official HLE-with-tools collection.
- Includes an integration wrapper for the external MetaCoreBench representation visualization toolkit.

## Install

Use Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install official Tau2-bench in the same environment. If the package is not on PyPI in your environment, install it from the upstream repository:

```bash
python -m pip install tau2-bench
```

or:

```bash
python -m pip install git+https://github.com/sierra-research/tau2-bench.git
```

Verify:

```bash
tau2 run --help
```

## Cluster Migration Package

Create a clean archive for moving this project to a Linux cluster:

```bash
python scripts/create_migration_package.py
```

The archive includes project code, `data/`, `.external/`, and dependency snapshots. It recursively excludes `.env`, virtual environments, run outputs, Git metadata, and cache directories, including nested environments such as `.external/*/.venv/`.

## Reproducible Setup From a Fresh Clone

The root repository contains the MetaCoreBench runners, collectors, tests, and
some small sample files. It intentionally does **not** contain virtual
environments, API keys, run outputs, the full HLE dataset, or the benchmark
checkouts under `.external/`. Prepare the benchmark-specific dependencies below
before starting a real run.

### Common setup

On a Linux server:

```bash
git clone https://github.com/wangmeng8/MetaCore_trajectories.git
cd MetaCore_trajectories

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
cp .env.example .env
```

The root `pyproject.toml` is intentionally lightweight. Install the
dependencies for the benchmark you are actually running rather than assuming
that `pip install -e .` installs Tau2, HLE-with-tools, Harbor, or OSWorld.
Use `python scripts/<runner> --dry-run` to check the wrapper without making
model calls.

### Tau2-bench

Clone and install the official Tau2 checkout. Tau2 currently requires Python
3.12 or newer and uses `uv` for installation:

```bash
python -m pip install uv
git clone https://github.com/sierra-research/tau2-bench .external/tau2-bench
cd .external/tau2-bench
uv sync
cd ../..
```

Set the model API in `.env` or in the shell. `OPENAI_BASE_URL` may point to any
OpenAI-compatible endpoint; the runner maps it to `OPENAI_API_BASE` when Tau2
needs the older LiteLLM variable:

```bash
export OPENAI_API_KEY='your-model-api-key'
export OPENAI_BASE_URL='https://your-provider.example/v1'
```

Run a one-task smoke test:

```bash
python scripts/run_tau2_gpt55.py \
  --tau2-command "uv run tau2" \
  --tau2-repo-dir .external/tau2-bench \
  --domain airline \
  --agent-model openai/gpt-5.5 \
  --user-model openai/gpt-5.5 \
  --num-tasks 1 \
  --num-rollouts 1 \
  --max-workers 2 \
  --run-id tau2_airline_smoke \
  --output-dir outputs/runs
```

Run all tasks in all three supported text domains sequentially:

```bash
python scripts/run_tau2_gpt55.py \
  --tau2-command "uv run tau2" \
  --tau2-repo-dir .external/tau2-bench \
  --domains airline retail telecom \
  --agent-model openai/gpt-5.5 \
  --user-model openai/gpt-5.5 \
  --num-rollouts 1 \
  --all-tasks \
  --max-workers 8 \
  --run-id tau2_full \
  --progress-interval 30 \
  --output-dir outputs/runs
```

Important Tau2 options:

- `--domains airline retail telecom` runs the domains one after another. Use
  `--domain airline` for one domain.
- `--num-rollouts N` is an alias for Tau2's `--num-trials N` and runs `N`
  independent trials per task.
- `--max-workers N` controls concurrency when the installed Tau2 CLI exposes a
  compatible worker flag. Start with `2` or `4` for rate-limited endpoints.
- `--one-by-one --max-workers 8` runs one task process at a time while keeping
  up to eight task processes in flight, with checkpoints after each task.
- Use a fixed `--run-id` with `--resume` to rebuild the normalized files from
  existing raw files. In `--one-by-one` mode, completed task folders are also
  skipped.

For a safer long run:

```bash
python scripts/run_tau2_gpt55.py \
  --tau2-command "uv run tau2" \
  --tau2-repo-dir .external/tau2-bench \
  --domain airline \
  --agent-model openai/gpt-5.5 \
  --user-model openai/gpt-5.5 \
  --num-rollouts 1 \
  --all-tasks \
  --one-by-one \
  --resume \
  --run-id tau2_airline_full \
  --max-workers 8 \
  --output-dir outputs/runs
```

Tau2 writes raw simulations under the checkout's `data/simulations/` by
default. The wrapper copies changed raw files and writes normalized results to
`outputs/runs/<run-id>/`, including `summary.json`, `manifest.json`,
`trajectories.jsonl`, and `benchmark_trajectories.jsonl`.

### HLE with tools

Use the official `activeloopai/hle_with_tools` checkout. The official project
requires Python 3.12 or newer. Create its isolated environment and install the
official package:

```bash
python -m pip install uv
git clone https://github.com/activeloopai/hle_with_tools .external/hle_with_tools
git -C .external/hle_with_tools apply ../../patches/hle_with_tools/local_hle_with_tools_fixes.patch
cd .external/hle_with_tools
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
cd ../..
```

HLE can load `cais/hle` from Hugging Face directly:

```bash
export HF_HOME="$PWD/.hf-cache"
export HF_DATASETS_CACHE="$PWD/.hf-cache/datasets"
```

Alternatively, provide a local parquet/json/jsonl file with `--data-path`.
The full HLE data files are ignored by this repository and must be downloaded
or copied separately.

Configure the model endpoint and the official scientific-search service:

```bash
export OPENAI_API_KEY='your-model-api-key'
export OPENAI_BASE_URL='https://your-provider.example/v1'

export ACTIVELOOP_API_KEY='your-activeloop-api-key'
export ACTIVELOOP_WORKSPACE='default'
export ACTIVELOOP_BASE_URL='https://science-api.activeloop.ai'
```

Run a one-example smoke test using the isolated official environment:

```bash
.external/hle_with_tools/.venv/bin/python scripts/run_hle_with_tools.py \
  --repo-dir .external/hle_with_tools \
  --dataset cais/hle \
  --model gpt-5.5 \
  --num-tasks 1 \
  --max-workers 2 \
  --max-retries 3 \
  --process-retries 1 \
  --max-completion-tokens 40000 \
  --max-iterations 15 \
  --run-id hle_smoke \
  --output-dir outputs/runs \
  --no-uv
```

Run all text-only examples with four workers:

```bash
.external/hle_with_tools/.venv/bin/python scripts/run_hle_with_tools.py \
  --repo-dir .external/hle_with_tools \
  --dataset cais/hle \
  --model gpt-5.5 \
  --all-tasks \
  --max-workers 4 \
  --max-retries 3 \
  --process-retries 1 \
  --max-completion-tokens 40000 \
  --max-iterations 15 \
  --run-id hle_gpt55_full \
  --progress-interval 30 \
  --output-dir outputs/runs \
  --no-uv
```

Run the 500-example Prompt-Reg condition with the same HLE harness and model
settings as Vanilla:

```bash
export OPENAI_API_KEY='your-model-api-key'
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
bash scripts/run_hle_prompt_reg.sh
```

The launcher reads `prompts/hle/prompt_reg.txt` and appends its complete text
to the existing HLE system message before the user question. Override launcher
defaults through `HLE_MODEL`, `HLE_MAX_WORKERS`, `HLE_RUN_ID`, `OUTPUT_DIR`, and
the other `HLE_*` variables. For an ad hoc prompt, call
`scripts/run_hle_with_tools.py --system-prompt-file <path>` directly. Each run
also snapshots the supplied text as `system_prompt.txt` in its output directory.

Important HLE options and behavior:

- `--num-rollouts N` runs `N` independent rollouts per selected example.
- `--system-prompt-file PATH` appends a UTF-8 prompt to the system message;
  omit it for the unchanged Vanilla condition.
- `--max-workers` is the official async worker count. The official runner
  requires at least two workers; use `2` first when the endpoint is unstable.
- Reuse the same `--run-id` after an interruption. The official temp prediction
  file is checked incrementally and completed predictions are resumed.
- `--max-completion-tokens 40000` and `--max-iterations 15` match the official
  README defaults. The wrapper omits `temperature` unless explicitly supplied.
- The official tools are `code_interpreter`, `web_browsing`, and
  `scientific_search`. Scientific search needs the ActiveLoop variables above;
  web browsing also depends on outbound network access from the server.
- The wrapper preserves official raw files and traces and additionally writes
  normalized `trajectory_text` records under the run directory.

The root repository does not track `.external/hle_with_tools`. The local
compatibility, Prompt-Reg, per-question timeout, and hard tool-timeout changes
are stored in
`patches/hle_with_tools/local_hle_with_tools_fixes.patch`. Apply that patch to
a clean official checkout with the command above before running experiments.

### Terminal-Bench 2.1

Terminal-Bench 2.1 uses the official Harbor dataset `terminal-bench/terminal-bench-2-1`.
It is a revised benchmark snapshot with 89 tasks; use `--dataset terminal-bench/terminal-bench-2`
explicitly when reproducing the earlier 2.0 run.

Terminal-Bench is executed through Harbor and requires Docker or another
supported Harbor sandbox provider. Install Harbor outside the root Python
environment:

```bash
python -m pip install uv
uv tool install harbor
harbor --help
docker info
```

Set the provider API key required by the selected Harbor agent. For the default
OpenAI-compatible setup:

```bash
export OPENAI_API_KEY='your-model-api-key'
export OPENAI_BASE_URL='https://your-provider.example/v1'
```

Run one task:

```bash
python scripts/run_terminal_bench.py \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent terminus-2 \
  --model openai/gpt-5.5 \
  --num-tasks 1 \
  --num-trials 1 \
  --run-id terminal_smoke \
  --jobs-dir jobs \
  --output-dir outputs/runs
```

Run the complete Terminal-Bench 2.1 task set:

```bash
python scripts/run_terminal_bench.py \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent terminus-2 \
  --model openai/gpt-5.5 \
  --all-tasks \
  --num-trials 1 \
  --run-id terminal_gpt55_full \
  --jobs-dir jobs \
  --output-dir outputs/runs
```

Useful Terminal-Bench options:

- `--num-trials N` controls repeated trials per task.
- `--max-workers N` maps to Harbor's concurrent trial count. Add it only after
  a one-task smoke test succeeds.
- `--task-name openssl-selfsigned-cert` runs one named task.
- `--harbor-env daytona` selects an alternative supported sandbox provider.
- `--run-id` fixes the output location. To retry only tasks missing a valid
  trajectory, use `--resume-missing-from-jobs <previous-jobs-dir>`.

The wrapper snapshots Harbor's `jobs/` directory, copies job artifacts under
`outputs/runs/<run-id>/raw/`, and exports normalized trajectories plus
`summary.json` and `manifest.json`. The Harbor job directory is separate from
the Git repository and should not be committed.

## Configure

Copy `.env.example` to `.env` and set your API key:

```bash
copy .env.example .env
```

Required for real runs:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Defaults:

```bash
AGENT_MODEL=gpt-5.5
USER_MODEL=gpt-5.5
BENCHMARK=tau2-bench
TAU2_DOMAIN=airline
NUM_TASKS=2
NUM_TRIALS=1
TAU2_ALL_TASKS=false
TAU2_RESUME=false
# TAU2_RUN_ID=tau2_airline_full
# TAU2_MAX_WORKERS=4
TAU2_PROGRESS_INTERVAL=30
OUTPUT_DIR=outputs/runs
REASONING_EFFORT=medium
TEMPERATURE=0

# Terminal-Bench 2.1 via Harbor.
TERMINAL_BENCH_DATASET=terminal-bench/terminal-bench-2-1
TERMINAL_BENCH_AGENT=terminus-2
HARBOR_JOBS_DIR=jobs

# OSWorld-Verified via a local OSWorld checkout.
OSWORLD_REPO_DIR=.external/OSWorld
OSWORLD_PROVIDER_NAME=docker
OSWORLD_OBSERVATION_TYPE=screenshot
OSWORLD_CLIENT_PASSWORD=password

# HLE w/ tools via activeloopai/hle_with_tools.
HLE_WITH_TOOLS_REPO_DIR=.external/hle_with_tools
HLE_DATASET=cais/hle
HLE_DATA_PATH=data/modelscope/cais_hle/data/test-00000-of-00001.parquet
HLE_WITH_TOOLS_TEXT_ONLY=true
HLE_ALL_TASKS=false
HLE_MAX_WORKERS=2
HLE_MAX_RETRIES=3
HLE_PROCESS_RETRIES=3
HLE_MAX_COMPLETION_TOKENS=40000
HLE_MAX_ITERATIONS=15
HLE_PROGRESS_INTERVAL=30
ACTIVELOOP_API_KEY=
ACTIVELOOP_WORKSPACE=default
ACTIVELOOP_BASE_URL=https://science-api.activeloop.ai
OPENAI_BASE_URL=https://api.openai.com/v1

# Representation visualization with .external/MetaCoreBench-visualization.
METACOREBENCH_MODEL_PATH=/data/oss_bucket_0/modelscope/hub/models/Qwen/Qwen3.6-27B
```

If your LiteLLM/Tau2 setup requires a provider prefix, set:

```bash
AGENT_MODEL=openai/gpt-5.5
USER_MODEL=openai/gpt-5.5
```

## Smoke Test

Dry-run first. This does not call OpenAI:

```bash
python scripts/run_tau2_gpt55.py --dry-run
```

Real run:

```bash
python scripts/run_tau2_gpt55.py
```

Override values from the CLI:

```bash
python scripts/run_tau2_gpt55.py --benchmark tau2-bench --domain airline --num-tasks 5 --num-trials 1 --agent-model gpt-5.5
```

Run every task in a Tau2 domain by adding `--all-tasks`. This intentionally omits Tau2's `--num-tasks` option:

```bash
python scripts/run_tau2_gpt55.py \
  --domain airline \
  --agent-model openai/gpt-5.4 \
  --user-model openai/gpt-5.4 \
  --num-trials 1 \
  --all-tasks
```

If Tau2 only works from a local checkout with `uv run`, point the runner at that checkout:

```bash
python scripts/run_tau2_gpt55.py \
  --tau2-command "uv run tau2" \
  --tau2-repo-dir .external/tau2-bench \
  --domain airline \
  --agent-model openai/gpt-5.4 \
  --user-model openai/gpt-5.4 \
  --num-tasks 1 \
  --num-trials 1
```

For a full Tau2 run on that kind of server, replace `--num-tasks 1` with `--all-tasks`:

```bash
python scripts/run_tau2_gpt55.py \
  --tau2-command "uv run tau2" \
  --tau2-repo-dir /path/to/tau2-bench \
  --domain airline \
  --agent-model openai/gpt-5.4 \
  --user-model openai/gpt-5.4 \
  --num-trials 1 \
  --all-tasks
```

Run a full Tau2 domain into a fixed output folder and reuse existing raw files during collection:

```bash
python scripts/run_tau2_gpt55.py \
  --tau2-command "uv run tau2" \
  --tau2-repo-dir /path/to/tau2-bench \
  --domain airline \
  --agent-model openai/gpt-5.4 \
  --user-model openai/gpt-5.4 \
  --num-trials 1 \
  --all-tasks \
  --resume \
  --run-id tau2_airline_full \
  --max-workers 4 \
  --progress-interval 30 \
  --output-dir /home/wm496616/MetaCoreBench_wm/outputs/runs
```

For long runs, prefer one-by-one checkpoint mode. The wrapper discovers Tau2 task IDs, runs official `tau2 run --task-ids <id>` processes, passes a deterministic `--save-to`, copies raw files immediately after each task, and rebuilds JSONL outputs after each task. In this mode, `--max-workers 8` means task-level concurrency: if there are 100 tasks, the wrapper starts tasks 1-8 first, then starts task 9 as soon as one of those finishes:

```bash
python scripts/run_tau2_gpt55.py \
  --tau2-command "uv run tau2" \
  --tau2-repo-dir /path/to/tau2-bench \
  --domain airline \
  --agent-model openai/gpt-5.4 \
  --user-model openai/gpt-5.4 \
  --num-trials 1 \
  --all-tasks \
  --one-by-one \
  --resume \
  --run-id tau2_airline_full \
  --max-workers 8 \
  --progress-interval 30 \
  --output-dir /home/wm496616/MetaCoreBench_wm/outputs/runs
```

If you want to run a small explicit subset in this safer mode:

```bash
python scripts/run_tau2_gpt55.py \
  --tau2-command "uv run tau2" \
  --tau2-repo-dir /path/to/tau2-bench \
  --domain airline \
  --agent-model openai/gpt-5.4 \
  --user-model openai/gpt-5.4 \
  --task-ids 0 1 2 \
  --one-by-one \
  --resume \
  --run-id tau2_airline_debug \
  --output-dir /home/wm496616/MetaCoreBench_wm/outputs/runs
```

`--resume` for Tau2 is a collection-level resume: existing `raw/*.json` and `raw/*.jsonl` files in the fixed run folder are included when rebuilding `trajectories.jsonl` and `benchmark_trajectories.jsonl`. The wrapper still launches Tau2 again, because Tau2's public CLI controls task selection. If Tau2 exits with an infrastructure error, the wrapper writes `raw/tau2_process_error.json` so the failed run is still visible in the outputs.

In `--one-by-one` mode, `--resume` also skips a task when its deterministic raw folder already exists under `raw/<run_id>/<task_id>/`. This is the safest mode for unstable cluster jobs because completed tasks remain available even if the next task is interrupted.

`--max-workers` has two meanings depending on mode. In normal single-process Tau2 mode, it is passed to Tau2 only if `tau2 run --help` shows a compatible worker/concurrency flag, such as `--max-concurrency`, `--max-workers`, or `--num-workers`. If your installed Tau2 version does not expose one, the option is omitted and recorded in `manifest.json` under `ignored_options`. In `--one-by-one` mode, it controls the wrapper's task-level process pool; each single-task Tau2 process is forced to internal concurrency `1` when Tau2 exposes a compatible flag, so `--max-workers 8` means at most 8 task processes at once rather than nested Tau2 concurrency.

While Tau2 is running, the wrapper prints progress lines like `Tau2 progress ... | elapsed ... | raw changed=...`. Tau2 is an external CLI, so this is a heartbeat plus copied/raw-file progress estimate rather than an exact per-task callback. Use `--progress-interval 10` for more frequent updates.

Equivalent `.env` settings:

```bash
TAU2_COMMAND=uv run tau2
TAU2_REPO_DIR=.external/tau2-bench
TAU2_ALL_TASKS=true
TAU2_ONE_BY_ONE=true
TAU2_RESUME=true
TAU2_RUN_ID=tau2_airline_full
TAU2_MAX_WORKERS=8
TAU2_PROGRESS_INTERVAL=30
```

When `TAU2_REPO_DIR` is set and `TAU2_SIMULATIONS_DIR` is not set, the runner scans `.external/tau2-bench/data/simulations` for new raw Tau2 trajectory files.

Use `BENCHMARK` or `--benchmark` to label which dataset produced the trajectories. For Tau2 runs, keep the default `tau2-bench`; for another benchmark later, set a different name.

## Output Layout

Each run is written under `OUTPUT_DIR`:

```text
outputs/
  runs/
    2026-xx-xx_xxxxxx_gpt-5.5_airline/
      manifest.json
      raw/
        ...
      trajectories/
        task_xxx_trial_0.json
      benchmark_trajectories/
        task_xxx.json
      trajectories.jsonl
      benchmark_trajectories.jsonl
      summary.json
      tau2.stdout.log
      tau2.stderr.log
```

`manifest.json` records run config, command, environment, ignored optional options, and raw file paths.

`raw/` contains new or modified files copied from Tau2's simulation output directory, normally `data/simulations/`.

`trajectories/*.json` contains one normalized JSON trajectory per raw record.

`trajectories.jsonl` contains the same normalized trajectories as one JSON object per line.

`benchmark_trajectories/*.json` contains one training/export record per task. Multiple trials or reruns for the same task are grouped into the `trajectories` array:

```json
{
  "task_id": "0",
  "benchmark": "tau2-bench",
  "task": {
    "instruction": "Original task instruction or request."
  },
  "trajectories": [
    {
      "trajectory_id": "0_openai_gpt-5.4_2026-..._trial_0",
      "source_model": "openai/gpt-5.4",
      "trajectory_text": "USER: ...\nASSISTANT TOOL_CALL ...",
      "metadata": {}
    }
  ]
}
```

`benchmark_trajectories.jsonl` contains the same benchmark-oriented task records as one JSON object per line. `trajectory_text` serializes all recognized message, tool call, and tool result events into one text field for downstream training-data construction.

`summary.json` reports total trajectories, success count/rate when available, average turns, average tool calls, and average score when available.

## Representation Visualization

The visualization repository is placed at `.external/MetaCoreBench-visualization`. It consumes the `benchmark_trajectories.jsonl` files produced by this project and extracts Qwen-compatible hidden-state representations before producing token/layer, role/layer, anchor/JSD, and cohort-level plots.

Generate a config for the latest collected run without launching the model:

```bash
python scripts/run_metacorebench_visualization.py
```

Generate a config for a specific run:

```bash
python scripts/run_metacorebench_visualization.py \
  --run-dir outputs/runs/YOUR_RUN \
  --benchmark hle_with_tools \
  --model-path /data/oss_bucket_0/modelscope/hub/models/Qwen/Qwen3.6-27B
```

Add `--execute` to run the external extraction and validation steps. Outputs go under `outputs/runs/YOUR_RUN/metacorebench_results/<benchmark>` by default:

```bash
python scripts/run_metacorebench_visualization.py \
  --run-dir outputs/runs/YOUR_RUN \
  --benchmark tau2_airline \
  --gpu-numbers 0,1,2,3 \
  --execute
```

Supported visualization benchmark labels are `tau2_airline`, `tau2_retail`, `tau2_telecom`, `hle_with_tools`, and `terminal_bench_2`. The wrapper maps existing collection labels such as `tau2-bench`, `hle-with-tools`, and Terminal-Bench 2.0/2.1 labels when it can infer the domain from the run metadata.

## Collector Only

You can normalize an existing raw directory:

```bash
python scripts/collect_trajectories.py --raw-dir outputs/runs/YOUR_RUN/raw --output-dir outputs/runs/YOUR_RUN --benchmark tau2-bench
```

If `--run-id` is omitted, the collector uses the output directory name, such as `YOUR_RUN`.

The collector is schema-tolerant. It looks for common Tau2-like fields such as `messages`, `conversation`, `trajectory`, `events`, `turns`, `tool_calls`, `observations`, `reward`, `success`, `score`, and `task_id`. If it cannot recognize the schema, it still preserves the full raw object and writes it into JSONL.

## Additional Benchmarks

These runners use the same output contract as the Tau2 runner:

```text
outputs/runs/<run_id>/
  manifest.json
  raw/
  trajectories/
  trajectories.jsonl
  benchmark_trajectories/
  benchmark_trajectories.jsonl
  summary.json
```

### OSWorld-Verified

OSWorld-Verified requires a working OSWorld checkout plus its VM/Docker environment. Clone and install OSWorld separately:

```bash
git clone https://github.com/xlang-ai/OSWorld .external/OSWorld
cd .external/OSWorld
pip install -r requirements.txt
```

Dry-run:

```bash
python scripts/run_osworld_verified.py --dry-run
```

Run a small targeted task when your OSWorld environment is ready:

```bash
python scripts/run_osworld_verified.py --repo-dir .external/OSWorld --provider-name docker --model openai/gpt-5.4 --domain libreoffice_impress --example-id a669ef01-ded5-4099-9ea9-25e99b569840 --max-steps 3
```

OSWorld writes screenshots, actions, and recordings under its `result_dir`; this wrapper sets `result_dir` inside the run folder, copies those files into `raw/`, and exports any JSON/JSONL records it finds.

## Common Issues

### `OPENAI_API_KEY` is not set

Dry-run works without an API key. A real run exits with a clear error. Add `OPENAI_API_KEY` to `.env` or your shell environment.

### `tau2` command not found

Install Tau2-bench in the active Python environment, then verify `tau2 run --help`.

On clusters where Tau2 must run from the cloned repository, use:

```bash
python scripts/run_tau2_gpt55.py --tau2-command "uv run tau2" --tau2-repo-dir .external/tau2-bench --dry-run
```

or add this to `.env`:

```bash
TAU2_COMMAND=uv run tau2
TAU2_REPO_DIR=.external/tau2-bench
```

### LiteLLM or OpenAI does not recognize the model name

The default model is `gpt-5.5`. If your Tau2/LiteLLM version expects provider-prefixed model names, set `AGENT_MODEL=openai/gpt-5.5` and `USER_MODEL=openai/gpt-5.5`.

### Optional settings are ignored

`REASONING_EFFORT`, `TEMPERATURE`, and `MAX_STEPS` are passed only when `tau2 run --help` shows matching CLI flags. Ignored options are recorded in `manifest.json`.

`TAU2_MAX_WORKERS` / `--max-workers` follows the same rule. It is passed only when the installed Tau2 CLI exposes a compatible worker or concurrency option.

### Tau2 resume does not skip tasks

`--resume` reuses existing raw files in the run folder while collecting trajectories. It does not force Tau2 itself to skip completed tasks. Use it with a fixed `--run-id` when you want to rebuild `trajectories.jsonl` and `benchmark_trajectories.jsonl` from old raw files plus any newly copied Tau2 simulations.

### No `data/simulations/` directory found

Tau2 normally writes simulation trajectories under `data/simulations/`. If your version writes elsewhere, set:

```bash
TAU2_SIMULATIONS_DIR=path/to/simulations
```

or pass:

```bash
python scripts/run_tau2_gpt55.py --simulations-dir path/to/simulations
```

### `harbor` command not found

Install Harbor with `uv tool install harbor`, then run `harbor --help`. Terminal-Bench 2.1 requires Docker or a supported Harbor sandbox provider.

### OSWorld run script not found

Clone OSWorld and set `OSWORLD_REPO_DIR` or pass `--repo-dir`. OSWorld-Verified also needs a configured VM/Docker provider before real runs will work.

### HLE data file not found

Set `HLE_DATA_PATH` or pass `--data-path` to a JSON, JSONL, or parquet file. If you omit it, `scripts/run_hle_with_tools.py` asks the official runner to load `cais/hle` from Hugging Face.

### HLE endpoint disconnects or rate-limits

Use a fixed `--run-id` and reduce `--max-workers`:

```bash
python scripts/run_hle_with_tools.py --all-tasks --run-id hle_gpt55_full --max-workers 2
```

The official runner writes `raw/official_run/hle_<model>.json.temp` after each completed prediction. If the process is interrupted, rerun the same command with the same `--run-id` and the official runner will skip predictions already present in that file.

## Validation

Local checks:

```bash
python scripts/run_tau2_gpt55.py --dry-run
python scripts/collect_trajectories.py --help
python scripts/run_terminal_bench.py --dry-run
python scripts/run_osworld_verified.py --dry-run
python scripts/run_hle_with_tools.py --dry-run
python -m compileall .
python -m unittest discover -s tests
```
