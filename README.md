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
- Also includes MVP runners for Terminal-Bench 2.0, OSWorld-Verified, and official HLE-with-tools collection.
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

# Terminal-Bench 2.0 via Harbor.
TERMINAL_BENCH_DATASET=terminal-bench/terminal-bench-2
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

Supported visualization benchmark labels are `tau2_airline`, `tau2_retail`, `tau2_telecom`, `hle_with_tools`, and `terminal_bench_2`. The wrapper maps existing collection labels such as `tau2-bench`, `hle-with-tools`, and `terminal-bench-2.0` when it can infer the domain from the run metadata.

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

### Terminal-Bench 2.0

Terminal-Bench 2.0 is run through Harbor. Install Harbor and make sure Docker is running:

```bash
uv tool install harbor
harbor --help
```

Dry-run:

```bash
python scripts/run_terminal_bench.py --dry-run
```

Run one task:

```bash
python scripts/run_terminal_bench.py --model openai/gpt-5.4 --agent terminus-2 --num-tasks 1 --num-trials 1
```

Run a named task:

```bash
python scripts/run_terminal_bench.py --model openai/gpt-5.4 --agent terminus-2 --task-name openssl-selfsigned-cert
```

The script snapshots Harbor's `jobs/` directory, copies new or modified job files into `raw/`, and then exports normalized and benchmark-style trajectories.

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

### HLE W/ Tools

Use the official `activeloopai/hle_with_tools` checkout through `scripts/run_hle_with_tools.py`. The wrapper calls the official runner, keeps its raw prediction/temp/code/trace files, and exports our unified `trajectory_text` format.

Official tools exposed in the run are `code_interpreter`, `web_browsing`, and `scientific_search`. The wrapper defaults to the official README settings `--max_completion_tokens 40000` and `--max_iterations 15`; it filters multimodal examples by default.

The official repo declares Python 3.12+ in its `pyproject.toml`. Prefer `uv run` for HLE-with-tools so the benchmark environment stays isolated from the main MetaCoreBench environment.

Dry-run:

```bash
python scripts/run_hle_with_tools.py --dry-run --data-path data/modelscope/cais_hle/data/test-00000-of-00001.parquet
```

Run one example:

```bash
python scripts/run_hle_with_tools.py --data-path data/modelscope/cais_hle/data/test-00000-of-00001.parquet --model gpt-5.5 --base-url https://api.openai.com/v1 --num-tasks 1 --max-workers 2
```

Run all text-only examples with official checkpointing:

```bash
python scripts/run_hle_with_tools.py \
  --model openai/gpt-5.5 \
  --base-url https://evamux.alibaba-inc.com/v1 \
  --all-tasks \
  --max-workers 4 \
  --max-retries 3 \
  --process-retries 3 \
  --run-id hle_gpt55_full \
  --max-iterations 15 \
  --max-completion-tokens 40000 \
  --progress-interval 30 \
  --output-dir /home/wm496616/MetaCoreBench_wm/outputs/runs
```

Use the same `--run-id` to continue an interrupted run. The official runner writes `raw/official_run/hle_<model>.json.temp` incrementally, plus per-question logs under `raw/official_run/traces/trace_<question_id>.log`; the wrapper converts those traces into `benchmark_trajectories.jsonl`.

For unstable or rate-limited endpoints, start with `--max-workers 2`. The official runner asserts that workers must be at least 2.

The wrapper prints progress by counting completed predictions in the official temp/final JSON file. Lower `--progress-interval` if you want more frequent intermediate lines.

The official `scientific_search` tool needs `ACTIVELOOP_API_KEY`, `ACTIVELOOP_WORKSPACE`, and `ACTIVELOOP_BASE_URL`. Without those, tasks that call scientific search will record tool errors in the trace.

The wrapper does not send `temperature` by default, because some GPT-5.5-compatible endpoints reject that parameter. Use `--temperature` or `HLE_TEMPERATURE` only with models/endpoints that accept it.

The previous minimal local harness is still available as `scripts/run_hle_tools.py` for debugging simple JSON/JSONL/parquet questions, but official HLE-with-tools collection should use `scripts/run_hle_with_tools.py`.

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

Install Harbor with `uv tool install harbor`, then run `harbor --help`. Terminal-Bench 2.0 requires Docker or a supported Harbor sandbox provider.

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
