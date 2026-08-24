"""Run activeloopai/hle_with_tools and export MetaCoreBench trajectories."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_trajectories import summarize_trajectories
from scripts.runner_common import (
    ProgressPrinter,
    build_manifest,
    env_bool,
    env_int,
    load_dotenv_file,
    now_timestamp,
    prepare_subprocess_env,
    print_run_report,
    safe_run_part,
    shell_join,
    write_json,
)


BENCHMARK = "hle-with-tools"
TOOL_NAMES = ["code_interpreter", "web_browsing", "scientific_search"]


@dataclass
class HLEWithToolsConfig:
    run_id: str
    run_dir: Path
    repo_dir: Path
    benchmark: str
    dataset: str
    data_path: Path | None
    model: str
    base_url: str | None
    num_tasks: int | None
    max_workers: int
    max_retries: int
    process_retries: int
    max_completion_tokens: int
    max_iterations: int
    temperature: float | None
    text_only: bool
    disable_scientific_search: bool
    use_uv: bool
    progress_interval: float

    @property
    def dataset_arg(self) -> str:
        if self.data_path is not None:
            return str(self.data_path.resolve())
        return self.dataset

    @property
    def official_raw_dir(self) -> Path:
        return self.run_dir / "raw" / "official_run"

    @property
    def safe_model_id(self) -> str:
        return safe_official_id(self.model)

    @property
    def predictions_path(self) -> Path:
        return self.official_raw_dir / f"hle_{self.safe_model_id}.json"

    @property
    def temp_predictions_path(self) -> Path:
        return self.official_raw_dir / f"hle_{self.safe_model_id}.json.temp"

    @property
    def tool_names(self) -> list[str]:
        if self.disable_scientific_search:
            return [name for name in TOOL_NAMES if name != "scientific_search"]
        return list(TOOL_NAMES)


def build_hle_with_tools_config(env: dict[str, str] | None = None, now: str | None = None) -> HLEWithToolsConfig:
    env = env or os.environ
    timestamp = now or now_timestamp()
    benchmark = env.get("HLE_BENCHMARK", BENCHMARK)
    model = env.get("HLE_MODEL") or env.get("AGENT_MODEL") or env.get("MODEL_NAME") or "gpt-5.5"
    output_dir = Path(env.get("OUTPUT_DIR", "outputs/runs"))
    run_id = env.get("HLE_RUN_ID") or env.get("RUN_ID") or f"{timestamp}_{safe_run_part(model)}_{safe_run_part(benchmark)}"
    data_path = env.get("HLE_DATA_PATH")
    all_tasks = env_bool(env, "HLE_ALL_TASKS", False)
    num_tasks = None if all_tasks else env_int(env, "NUM_TASKS", env_int(env, "HLE_NUM_EXAMPLES", 1))
    temperature = optional_float(env.get("HLE_TEMPERATURE"))

    return HLEWithToolsConfig(
        run_id=run_id,
        run_dir=output_dir / run_id,
        repo_dir=Path(env.get("HLE_WITH_TOOLS_REPO_DIR", ".external/hle_with_tools")),
        benchmark=benchmark,
        dataset=env.get("HLE_DATASET", "cais/hle"),
        data_path=Path(data_path) if data_path else None,
        model=model,
        base_url=normalize_base_url(env.get("OPENAI_BASE_URL") or env.get("HLE_OPENAI_BASE_URL")),
        num_tasks=num_tasks,
        max_workers=max(2, env_int(env, "HLE_MAX_WORKERS", 2)),
        max_retries=max(0, env_int(env, "HLE_MAX_RETRIES", env_int(env, "OPENAI_MAX_RETRIES", 2))),
        process_retries=max(0, env_int(env, "HLE_PROCESS_RETRIES", 0)),
        max_completion_tokens=env_int(env, "HLE_MAX_COMPLETION_TOKENS", 40000),
        max_iterations=env_int(env, "HLE_MAX_ITERATIONS", env_int(env, "MAX_STEPS", 15)),
        temperature=temperature,
        text_only=env_bool(env, "HLE_WITH_TOOLS_TEXT_ONLY", not env_bool(env, "HLE_INCLUDE_MULTIMODAL", False)),
        disable_scientific_search=env_bool(env, "HLE_DISABLE_SCIENTIFIC_SEARCH", False),
        use_uv=env_bool(env, "HLE_WITH_TOOLS_USE_UV", True),
        progress_interval=float(env.get("HLE_PROGRESS_INTERVAL", "30")),
    )


def build_official_command(config: HLEWithToolsConfig) -> list[str]:
    runner = "hle_eval/run_agent_predictions.py"
    if config.use_uv and shutil.which("uv"):
        command = ["uv", "run", "--python", os.environ.get("HLE_WITH_TOOLS_UV_PYTHON", sys.executable), "python", runner]
    else:
        command = [sys.executable, runner]

    command.extend(
        [
            "--dataset",
            config.dataset_arg,
            "--model",
            config.model,
            "--max_completion_tokens",
            str(config.max_completion_tokens),
            "--num_workers",
            str(config.max_workers),
            "--max_iterations",
            str(config.max_iterations),
        ]
    )
    if config.num_tasks is not None:
        command.extend(["--max_samples", str(config.num_tasks)])
    if config.temperature is not None:
        command.extend(["--temperature", str(config.temperature)])
    return command


def prepare_official_env(config: HLEWithToolsConfig, base_env: dict[str, str] | os._Environ[str]) -> dict[str, str]:
    env = prepare_subprocess_env(base_env)
    if config.base_url:
        env["OPENAI_BASE_URL"] = config.base_url
    env["HLE_WITH_TOOLS_RUNS_DIR"] = str((config.run_dir / "raw").resolve())
    env["HLE_WITH_TOOLS_RUN_NAME"] = "official_run"
    env["HLE_WITH_TOOLS_TEXT_ONLY"] = "1" if config.text_only else "0"
    env["HLE_MAX_RETRIES"] = str(config.max_retries)
    env["OPENAI_MAX_RETRIES"] = str(config.max_retries)
    env["ENABLE_TRACING"] = "true"
    env["ENABLE_CONSOLE_TRACING"] = "false"
    if config.temperature is None:
        env["HLE_WITH_TOOLS_DISABLE_TEMPERATURE"] = "1"
    if config.base_url and "openrouter.ai" not in config.base_url:
        env.setdefault("HLE_WITH_TOOLS_DISABLE_EXTRA_BODY", "1")
    return env


def run_official_command(
    command: list[str],
    *,
    config: HLEWithToolsConfig,
    env: dict[str, str],
) -> int:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.official_raw_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = config.run_dir / "hle_with_tools.stdout.log"
    stderr_path = config.run_dir / "hle_with_tools.stderr.log"
    progress = ProgressPrinter(
        label="HLE-with-tools",
        total=config.num_tasks,
        interval_seconds=config.progress_interval,
    )

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=config.repo_dir,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        progress.update(current_prediction_count(config), extra=f"workers={config.max_workers}", force=True)
        while process.poll() is None:
            time.sleep(1)
            progress.update(current_prediction_count(config), extra=f"workers={config.max_workers}")
        progress.update(current_prediction_count(config), extra=f"workers={config.max_workers}", force=True)
        return int(process.returncode or 0)


def collect_hle_with_tools_outputs(config: HLEWithToolsConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = config.run_dir / "trajectories"
    benchmark_dir = config.run_dir / "benchmark_trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    clear_json_files(trajectories_dir)
    clear_json_files(benchmark_dir)

    predictions, prediction_file = load_predictions(config)
    trace_files = per_question_trace_files(config.official_raw_dir / "traces")
    task_ids = sorted(set(predictions) | set(trace_files))
    if not task_ids and (config.official_raw_dir / "traces" / "trace.log").exists():
        task_ids = ["unknown"]

    trajectories: list[dict[str, Any]] = []
    benchmark_records: list[dict[str, Any]] = []
    trajectories_jsonl = config.run_dir / "trajectories.jsonl"
    benchmark_jsonl = config.run_dir / "benchmark_trajectories.jsonl"

    with trajectories_jsonl.open("w", encoding="utf-8") as traj_file, benchmark_jsonl.open("w", encoding="utf-8") as bench_file:
        for index, task_id in enumerate(task_ids):
            normalized = normalize_official_record(
                task_id=task_id,
                prediction=predictions.get(task_id),
                prediction_file=prediction_file,
                config=config,
                trace_file=trace_files.get(task_id),
            )
            trajectories.append(normalized)
            trajectory_path = trajectories_dir / f"{safe_run_part(task_id) or f'task_{index:04d}'}.json"
            write_json(trajectory_path, normalized)
            traj_file.write(json.dumps(normalized, ensure_ascii=False) + "\n")

            benchmark_record = to_official_benchmark_record(normalized, config=config)
            benchmark_records.append(benchmark_record)
            benchmark_path = benchmark_dir / f"{safe_run_part(task_id) or f'task_{index:04d}'}.json"
            write_json(benchmark_path, benchmark_record)
            bench_file.write(json.dumps(benchmark_record, ensure_ascii=False) + "\n")

    summary = summarize_trajectories(trajectories)
    usage_totals = sum_usage(predictions.values())
    summary.update(
        {
            "benchmark_task_count": len(benchmark_records),
            "benchmark_trajectory_count": sum(len(item.get("trajectories", [])) for item in benchmark_records),
            "completed_prediction_count": len(predictions),
            "trace_file_count": len(trace_files),
            "tool_names": config.tool_names,
            "max_retries": config.max_retries,
            "process_retries": config.process_retries,
            "max_completion_tokens": config.max_completion_tokens,
            "max_iterations": config.max_iterations,
            "total_prompt_tokens": usage_totals.get("prompt_tokens", 0),
            "total_completion_tokens": usage_totals.get("completion_tokens", 0),
            "total_tokens": usage_totals.get("total_tokens", 0),
        }
    )
    write_json(config.run_dir / "summary.json", summary)
    return trajectories, benchmark_records, summary


def normalize_official_record(
    *,
    task_id: str,
    prediction: dict[str, Any] | None,
    prediction_file: Path | None,
    config: HLEWithToolsConfig,
    trace_file: Path | None,
) -> dict[str, Any]:
    trace_file = trace_file or fallback_trace_file(config.official_raw_dir / "traces")
    trace_text = read_text_or_empty(trace_file)
    code_file = config.official_raw_dir / "code" / f"{safe_official_id(task_id)}.py"
    code_text = read_text_or_empty(code_file)
    prediction = prediction or {}
    final_response = prediction.get("response")
    question = extract_question_from_trace(trace_text)
    events = [{"index": 0, "role": "user", "type": "message", "content": question}]
    if trace_text:
        events.append({"index": 1, "role": "assistant", "type": "trace", "content": trace_text})
    if final_response:
        events.append({"index": len(events), "role": "assistant", "type": "message", "content": final_response})

    raw_files = {
        "prediction_file": relative_path(prediction_file, config.run_dir) if prediction_file else None,
        "trace_file": relative_path(trace_file, config.run_dir) if trace_file else None,
        "code_file": relative_path(code_file, config.run_dir) if code_file.exists() else None,
    }
    return {
        "run_id": config.run_id,
        "domain": config.dataset_arg,
        "task_id": task_id,
        "trial_id": 0,
        "agent_model": prediction.get("model") or config.model,
        "user_model": None,
        "success": None,
        "score": None,
        "num_turns": count_iterations(trace_text),
        "num_tool_calls": count_tool_calls(trace_text, prediction),
        "task": {"instruction": question},
        "events": events,
        "raw": {
            "official_prediction": prediction,
            "trace_text": trace_text,
            "code": code_text,
            "official_files": raw_files,
            "completed_prediction": bool(prediction),
        },
        "raw_file": raw_files["prediction_file"],
    }


def to_official_benchmark_record(normalized: dict[str, Any], *, config: HLEWithToolsConfig) -> dict[str, Any]:
    task_id = str(normalized.get("task_id") or "unknown_task")
    source_model = str(normalized.get("agent_model") or config.model)
    raw = normalized.get("raw") if isinstance(normalized.get("raw"), dict) else {}
    prediction = raw.get("official_prediction") if isinstance(raw.get("official_prediction"), dict) else {}
    trace_text = raw.get("trace_text") if isinstance(raw.get("trace_text"), str) else ""
    trajectory_text = trace_text or str(prediction.get("response") or "")
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "task": {
            "instruction": (normalized.get("task") or {}).get("instruction"),
        },
        "trajectories": [
            {
                "trajectory_id": f"{safe_run_part(task_id)}_{safe_run_part(source_model)}_{safe_run_part(config.run_id)}",
                "source_model": source_model,
                "trajectory_text": trajectory_text,
                "metadata": {
                    "run_id": config.run_id,
                    "dataset": config.dataset_arg,
                    "official_repo": "activeloopai/hle_with_tools",
                    "agent_model": source_model,
                    "trial_id": normalized.get("trial_id"),
                    "success": normalized.get("success"),
                    "score": normalized.get("score"),
                    "num_turns": normalized.get("num_turns"),
                    "num_tool_calls": normalized.get("num_tool_calls"),
                    "usage": prediction.get("usage"),
                    "completed_prediction": bool(prediction),
                    "raw_file": normalized.get("raw_file"),
                    "trace_file": (raw.get("official_files") or {}).get("trace_file"),
                    "code_file": (raw.get("official_files") or {}).get("code_file"),
                    "tools": config.tool_names,
                    "max_completion_tokens": config.max_completion_tokens,
                    "max_iterations": config.max_iterations,
                    "text_only": config.text_only,
                    "tool_output_limits": {
                        "code_interpreter_timeout_seconds": 60,
                        "web_browsing_max_chars": 10000,
                        "scientific_search_max_chars": 15000,
                        "scientific_search_max_chars_per_result": 2000,
                    },
                },
            }
        ],
    }


def load_predictions(config: HLEWithToolsConfig) -> tuple[dict[str, dict[str, Any]], Path | None]:
    for path in (config.predictions_path, config.temp_predictions_path):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items() if isinstance(value, dict)}, path
    return {}, None


def current_prediction_count(config: HLEWithToolsConfig) -> int:
    predictions, _ = load_predictions(config)
    return len(predictions)


def per_question_trace_files(trace_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not trace_dir.exists():
        return files
    for path in sorted(trace_dir.glob("trace_*.log")):
        task_id = path.name[len("trace_") : -len(".log")]
        files[task_id] = path
    return files


def fallback_trace_file(trace_dir: Path) -> Path | None:
    path = trace_dir / "trace.log"
    return path if path.exists() else None


def extract_question_from_trace(trace_text: str) -> str | None:
    if not trace_text:
        return None
    patterns = [
        r"Question:\s*(.+?)(?:\n[^\n]*STEP|\n[-=]{4,}|\n\n|$)",
        r"QUESTION.*?\n.*?Question:\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, trace_text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return clean_trace_text(match.group(1))
    return None


def count_iterations(trace_text: str) -> int:
    if not trace_text:
        return 0
    return len(re.findall(r"\bSTEP\s+\d+/\d+", trace_text, flags=re.IGNORECASE))


def count_tool_calls(trace_text: str, prediction: dict[str, Any] | None = None) -> int:
    if trace_text:
        count = len(re.findall(r"TOOL CALL\s+([A-Za-z_][A-Za-z0-9_]*)", trace_text))
        if count:
            return count
    usage = (prediction or {}).get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("tool_calls"), int):
        return usage["tool_calls"]
    return 0


def sum_usage(records: Any) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = record.get("usage") if isinstance(record, dict) else None
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return totals


def clear_json_files(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.glob("*.json"):
        path.unlink()


def read_text_or_empty(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def relative_path(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def clean_trace_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_official_id(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return urlunparse(parsed._replace(path="/v1"))
    return value


def config_for_manifest(config: HLEWithToolsConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["run_dir"] = str(config.run_dir)
    payload["repo_dir"] = str(config.repo_dir)
    payload["data_path"] = str(config.data_path) if config.data_path else None
    payload["dataset_arg"] = config.dataset_arg
    payload["tools"] = config.tool_names
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official activeloopai/hle_with_tools and collect trajectories.")
    parser.add_argument("--dry-run", action="store_true", help="Print the official command without calling the model.")
    parser.add_argument("--collect-only", action="store_true", help="Only collect an existing official raw run in this run directory.")
    parser.add_argument("--repo-dir", type=Path, help="Path to activeloopai/hle_with_tools checkout.")
    parser.add_argument("--dataset", help="HF dataset name. Defaults to cais/hle.")
    parser.add_argument("--data-path", type=Path, help="Local HLE parquet/json/jsonl file. Overrides --dataset.")
    parser.add_argument("--model", help="Model name passed to the official runner.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Bare hosts get /v1 appended.")
    parser.add_argument("--num-tasks", type=int, help="Number of text-only HLE examples to run.")
    parser.add_argument("--all-tasks", action="store_true", help="Run all selected examples.")
    parser.add_argument("--max-workers", type=int, help="Official async worker count. Minimum is 2.")
    parser.add_argument("--max-retries", type=int, help="OpenAI client retry count for API calls.")
    parser.add_argument("--process-retries", type=int, help="Restart the official process this many times if it exits non-zero.")
    parser.add_argument("--max-completion-tokens", type=int, help="Max completion tokens. Official README recommends 40000.")
    parser.add_argument("--max-iterations", type=int, help="Max tool iterations. Official README recommends 15.")
    parser.add_argument("--max-steps", type=int, help="Alias for --max-iterations.")
    parser.add_argument("--temperature", type=float, help="Send temperature only when explicitly set.")
    parser.add_argument("--include-multimodal", action="store_true", help="Do not filter HLE image examples.")
    parser.add_argument("--run-id", help="Fixed run id, useful for resume.")
    parser.add_argument("--output-dir", type=Path, help="Base output directory.")
    parser.add_argument("--progress-interval", type=float, help="Seconds between progress updates.")
    parser.add_argument("--no-uv", action="store_true", help="Use current Python instead of uv run.")
    return parser.parse_args(argv)


def apply_cli_overrides(env: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    env = dict(env)
    updates = {
        "HLE_WITH_TOOLS_REPO_DIR": str(args.repo_dir) if args.repo_dir else None,
        "HLE_DATASET": args.dataset,
        "HLE_DATA_PATH": str(args.data_path) if args.data_path else None,
        "HLE_MODEL": args.model,
        "OPENAI_BASE_URL": normalize_base_url(args.base_url),
        "NUM_TASKS": args.num_tasks,
        "HLE_MAX_WORKERS": args.max_workers,
        "HLE_MAX_RETRIES": args.max_retries,
        "HLE_PROCESS_RETRIES": args.process_retries,
        "HLE_MAX_COMPLETION_TOKENS": args.max_completion_tokens,
        "HLE_MAX_ITERATIONS": args.max_iterations or args.max_steps,
        "HLE_TEMPERATURE": args.temperature,
        "HLE_RUN_ID": args.run_id,
        "OUTPUT_DIR": str(args.output_dir) if args.output_dir else None,
        "HLE_PROGRESS_INTERVAL": args.progress_interval,
    }
    for key, value in updates.items():
        if value is not None:
            env[key] = str(value)
    if args.all_tasks:
        env["HLE_ALL_TASKS"] = "1"
    if args.include_multimodal:
        env["HLE_WITH_TOOLS_TEXT_ONLY"] = "0"
        env["HLE_INCLUDE_MULTIMODAL"] = "1"
    if args.no_uv:
        env["HLE_WITH_TOOLS_USE_UV"] = "0"
    return env


def validate_real_run(config: HLEWithToolsConfig, env: dict[str, str]) -> str | None:
    if not config.repo_dir.exists():
        return f"official hle_with_tools repo not found: {config.repo_dir}"
    runner = config.repo_dir / "hle_eval" / "run_agent_predictions.py"
    if not runner.exists():
        return f"official runner not found: {runner}"
    if not env.get("OPENAI_API_KEY") and not env.get("OPENROUTER_API_KEY"):
        return "OPENAI_API_KEY or OPENROUTER_API_KEY is required for a real HLE with tools run"
    if config.data_path is not None and not config.data_path.exists():
        return f"HLE_DATA_PATH does not exist: {config.data_path}"
    return None


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file()
    args = parse_args(argv)
    env = apply_cli_overrides(dict(os.environ), args)
    config = build_hle_with_tools_config(env=env)
    command = build_official_command(config)
    official_env = prepare_official_env(config, env)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        run_id=config.run_id,
        benchmark=config.benchmark,
        command=command,
        config=config_for_manifest(config),
        dry_run=args.dry_run,
        executable_path=shutil.which(command[0]) or command[0],
    )
    manifest["official_repo"] = "activeloopai/hle_with_tools"
    manifest["official_cwd"] = str(config.repo_dir)
    write_json(config.run_dir / "manifest.json", manifest)

    if args.dry_run:
        print("Dry run: no API call will be made.")
        print("Official repo:", config.repo_dir)
        print("Output directory:", config.run_dir)
        print("Official raw directory:", config.official_raw_dir)
        print("Command:", shell_join(command))
        return 0

    validation_error = validate_real_run(config, env)
    if validation_error:
        print(f"ERROR: {validation_error}", file=sys.stderr)
        return 2

    returncode = 0
    if not args.collect_only:
        print("Running:", shell_join(command))
        print("Official cwd:", config.repo_dir)
        print("Output directory:", config.run_dir)
        total_attempts = config.process_retries + 1
        for attempt in range(1, total_attempts + 1):
            print(f"Official process attempt {attempt}/{total_attempts}")
            returncode = run_official_command(command, config=config, env=official_env)
            if returncode == 0:
                break
            if attempt < total_attempts:
                print(f"Official process exited with {returncode}; restarting with the same run id to resume completed predictions.")

    trajectories, _, summary = collect_hle_with_tools_outputs(config)
    manifest["returncode"] = returncode
    manifest["completed_at"] = now_timestamp()
    write_json(config.run_dir / "manifest.json", manifest)
    print_run_report(config.run_dir, command, len(list(config.official_raw_dir.rglob("*"))) if config.official_raw_dir.exists() else 0, len(trajectories), summary)
    if returncode != 0:
        print("ERROR: official hle_with_tools exited with a non-zero status. See hle_with_tools.stderr.log.", file=sys.stderr)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
