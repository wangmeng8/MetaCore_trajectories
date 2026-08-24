"""Run a minimal Humanity's Last Exam with tools harness and export trajectories."""

from __future__ import annotations

import argparse
import ast
import http.client
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_trajectories import collect_raw_files, summarize_trajectories, write_benchmark_records
from scripts.runner_common import (
    build_manifest,
    env_int,
    json_like_files,
    load_dotenv_file,
    now_timestamp,
    prepare_subprocess_env,
    print_run_report,
    ProgressPrinter,
    safe_run_part,
    shell_join,
    write_json,
)


@dataclass
class HLEConfig:
    run_id: str
    benchmark: str
    output_root: Path
    run_dir: Path
    data_path: Path
    model: str
    base_url: str
    num_tasks: int | None
    all_tasks: bool
    max_steps: int
    temperature: float | None
    include_multimodal: bool
    max_workers: int
    resume: bool
    request_timeout: float
    max_retries: int
    retry_sleep: float
    progress_interval: float


def build_hle_config(env: dict[str, str] | None = None, now: str | None = None) -> HLEConfig:
    env = env or os.environ
    timestamp = now or now_timestamp()
    benchmark = env.get("HLE_BENCHMARK", "hle-with-tools")
    model = env.get("AGENT_MODEL", "openai/gpt-5.4")
    output_root = Path(env.get("OUTPUT_DIR", "outputs/runs"))
    run_id = env.get("HLE_RUN_ID") or env.get("RUN_ID") or f"{timestamp}_{safe_run_part(model)}_{safe_run_part(benchmark)}"
    all_tasks = _env_bool(env, "HLE_ALL_TASKS", False)
    return HLEConfig(
        run_id=run_id,
        benchmark=benchmark,
        output_root=output_root,
        run_dir=output_root / run_id,
        data_path=Path(env.get("HLE_DATA_PATH", _default_hle_data_path())),
        model=model,
        base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        num_tasks=None if all_tasks else env_int(env, "NUM_TASKS", env_int(env, "HLE_NUM_EXAMPLES", 1)),
        all_tasks=all_tasks,
        max_steps=env_int(env, "MAX_STEPS", 8),
        temperature=_optional_float(env.get("HLE_TEMPERATURE")),
        include_multimodal=_env_bool(env, "HLE_INCLUDE_MULTIMODAL", False),
        max_workers=max(1, env_int(env, "HLE_MAX_WORKERS", 1)),
        resume=_env_bool(env, "HLE_RESUME", False),
        request_timeout=float(env.get("HLE_REQUEST_TIMEOUT", "180")),
        max_retries=max(0, env_int(env, "HLE_MAX_RETRIES", 2)),
        retry_sleep=float(env.get("HLE_RETRY_SLEEP", "2")),
        progress_interval=float(env.get("HLE_PROGRESS_INTERVAL", "30")),
    )


def load_hle_examples(
    path: Path,
    limit: int | None = None,
    include_multimodal: bool = False,
) -> list[dict[str, Any]]:
    payloads: list[Any] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                payloads.append(json.loads(stripped))
    elif path.suffix.lower() == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        payloads = frame.to_dict(orient="records")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payloads = payload
        elif isinstance(payload, dict):
            for key in ("examples", "questions", "data", "records"):
                if isinstance(payload.get(key), list):
                    payloads = payload[key]
                    break
            if not payloads:
                payloads = [payload]
        else:
            payloads = [payload]

    if not include_multimodal:
        payloads = [item for item in payloads if not is_multimodal_hle_item(item)]

    examples = [_normalize_hle_example(item, index) for index, item in enumerate(payloads)]
    return examples[:limit] if limit is not None else examples


def count_hle_examples(path: Path, include_multimodal: bool = False) -> int:
    return len(load_hle_examples(path, limit=None, include_multimodal=include_multimodal))


def is_multimodal_hle_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    for key in ("image", "image_preview", "rationale_image", "images", "image_url", "image_urls"):
        if _has_nonempty_multimodal_value(item.get(key)):
            return True
    return False


def _has_nonempty_multimodal_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value) > 0
    if isinstance(value, dict):
        return any(_has_nonempty_multimodal_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_nonempty_multimodal_value(item) for item in value)
    return True


def _default_hle_data_path() -> str:
    modelscope_path = Path("data/modelscope/cais_hle/data/test-00000-of-00001.parquet")
    if modelscope_path.exists():
        return str(modelscope_path)
    return "data/hle_sample.jsonl"


def run_calculator_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = str(arguments.get("expression", ""))
    try:
        result = _safe_eval_expression(expression)
    except Exception as exc:  # noqa: BLE001 - returned to the model as tool output
        return {"error": str(exc)}
    return {"result": result}


def to_hle_raw_record(
    *,
    example: dict[str, Any],
    messages: list[dict[str, Any]],
    model: str,
    raw_responses: list[dict[str, Any]],
) -> dict[str, Any]:
    final_answer = _last_assistant_content(messages)
    reference = example.get("answer")
    success = _grade_exact(final_answer, reference)
    score = None if success is None else (1.0 if success else 0.0)
    return {
        "benchmark": "hle-with-tools",
        "task_id": example.get("task_id"),
        "task": {"instruction": example.get("instruction")},
        "agent_model": model,
        "success": success,
        "score": score,
        "answer": final_answer,
        "reference_answer": reference,
        "messages": messages,
        "raw_responses": raw_responses,
        "raw_example": example.get("raw", example),
    }


def to_hle_checkpoint_raw_record(
    *,
    example: dict[str, Any],
    messages: list[dict[str, Any]],
    model: str,
    raw_responses: list[dict[str, Any]],
    status: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    record = {
        "benchmark": "hle-with-tools",
        "task_id": example.get("task_id"),
        "task": {"instruction": example.get("instruction")},
        "agent_model": model,
        "success": None,
        "score": None,
        "answer": _last_assistant_content(messages),
        "reference_answer": example.get("answer"),
        "messages": messages,
        "raw_responses": raw_responses,
        "status": status,
        "partial": True,
        "raw_example": example.get("raw", example),
    }
    if error is not None:
        record["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
        }
    return record


def initial_hle_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are solving Humanity's Last Exam questions. Use the calculator tool when useful. "
                "Return the final answer concisely."
            ),
        },
        {"role": "user", "content": str(example["instruction"])},
    ]


def to_hle_error_raw_record(
    *,
    example: dict[str, Any],
    model: str,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "benchmark": "hle-with-tools",
        "task_id": example.get("task_id"),
        "task": {"instruction": example.get("instruction")},
        "agent_model": model,
        "success": None,
        "score": None,
        "answer": None,
        "reference_answer": example.get("answer"),
        "messages": initial_hle_messages(example),
        "raw_responses": [],
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
        "raw_example": example.get("raw", example),
    }


def raw_path_for_example(raw_dir: Path, example: dict[str, Any], index: int) -> Path:
    task_id = example.get("task_id") or f"hle_{index:04d}"
    return raw_dir / f"{safe_run_part(str(task_id))}.json"


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file()
    parser = argparse.ArgumentParser(description="Run HLE-style questions with a minimal tool-calling harness.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned run and create manifest without calling the API.")
    parser.add_argument("--benchmark", help="Override BENCHMARK.")
    parser.add_argument("--data-path", type=Path, help="Path to HLE-style JSON/JSONL questions.")
    parser.add_argument("--model", help="Override AGENT_MODEL.")
    parser.add_argument("--base-url", help="Override OPENAI_BASE_URL.")
    parser.add_argument("--num-tasks", type=int, help="Override NUM_TASKS.")
    parser.add_argument("--all-tasks", action="store_true", help="Run every text-only HLE example after filtering.")
    parser.add_argument("--max-steps", type=int, help="Override MAX_STEPS.")
    parser.add_argument("--temperature", type=float, help="Override TEMPERATURE.")
    parser.add_argument("--include-multimodal", action="store_true", help="Include HLE examples that contain image fields.")
    parser.add_argument("--max-workers", type=int, help="Number of concurrent HLE examples to run.")
    parser.add_argument("--resume", action="store_true", help="Skip examples whose raw JSON already exists in this run directory.")
    parser.add_argument("--run-id", help="Use a fixed run id, useful with --resume.")
    parser.add_argument("--request-timeout", type=float, help="Per-request chat-completions timeout in seconds.")
    parser.add_argument("--max-retries", type=int, help="Retry count for transient chat-completions failures.")
    parser.add_argument("--retry-sleep", type=float, help="Base sleep seconds between transient request retries.")
    parser.add_argument("--progress-interval", type=float, help="Seconds between progress updates.")
    parser.add_argument("--output-dir", type=Path, help="Override OUTPUT_DIR.")
    args = parser.parse_args(argv)

    env = prepare_subprocess_env(os.environ)
    _apply_overrides(env, args)
    config = build_hle_config(env=env)
    raw_dir = config.run_dir / "raw"
    trajectories_dir = config.run_dir / "trajectories"
    benchmark_dir = config.run_dir / "benchmark_trajectories"
    for directory in (raw_dir, trajectories_dir, benchmark_dir):
        directory.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "scripts/run_hle_tools.py",
        "--data-path",
        str(config.data_path),
        "--model",
        config.model,
        "--base-url",
        config.base_url,
        "--max-steps",
        str(config.max_steps),
        "--max-workers",
        str(config.max_workers),
        "--request-timeout",
        str(config.request_timeout),
        "--max-retries",
        str(config.max_retries),
        "--retry-sleep",
        str(config.retry_sleep),
        "--progress-interval",
        str(config.progress_interval),
    ]
    if config.all_tasks:
        command.append("--all-tasks")
    else:
        command.extend(["--num-tasks", str(config.num_tasks)])
    if config.include_multimodal:
        command.append("--include-multimodal")
    if config.resume:
        command.append("--resume")
    command.extend(["--run-id", config.run_id])
    manifest = build_manifest(
        run_id=config.run_id,
        benchmark=config.benchmark,
        command=command,
        config=asdict(config),
        dry_run=args.dry_run,
        executable_path="OpenAI-compatible chat completions",
    )

    if args.dry_run:
        if not config.data_path.exists():
            print("Dry run: HLE_DATA_PATH does not exist yet. Provide a JSON, JSONL, or parquet file before a real run.")
        if "OPENAI_API_KEY" not in env:
            print("Dry run: OPENAI_API_KEY is not set. A real run will require it.")
        write_json(config.run_dir / "manifest.json", manifest)
        print_run_report(config.run_dir, command, raw_count=0, trajectory_count=0, summary={})
        return 0

    if "OPENAI_API_KEY" not in env:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    if not config.data_path.exists():
        print("ERROR: HLE_DATA_PATH does not exist.", file=sys.stderr)
        return 2

    print("Running:", shell_join(command))
    examples = load_hle_examples(
        config.data_path,
        limit=config.num_tasks,
        include_multimodal=config.include_multimodal,
    )
    manifest["available_examples_after_filter"] = count_hle_examples(
        config.data_path,
        include_multimodal=config.include_multimodal,
    )
    manifest["requested_examples"] = len(examples)
    if not examples:
        print("ERROR: No HLE examples available after filtering multimodal items.", file=sys.stderr)
        write_json(config.run_dir / "manifest.json", manifest)
        return 2

    def refresh_outputs(current_raw_files: list[Path]) -> None:
        collect_hle_outputs_live(
            config=config,
            raw_files=current_raw_files,
            trajectories_dir=trajectories_dir,
            benchmark_dir=benchmark_dir,
            manifest=manifest,
        )

    try:
        raw_files = run_hle_batch(
            examples,
            config=config,
            api_key=env["OPENAI_API_KEY"],
            raw_dir=raw_dir,
            on_raw_files_changed=refresh_outputs,
        )
    except KeyboardInterrupt:
        raw_files = json_like_files(raw_dir)
        manifest["interrupted"] = True
        summary = collect_hle_outputs_live(
            config=config,
            raw_files=raw_files,
            trajectories_dir=trajectories_dir,
            benchmark_dir=benchmark_dir,
            manifest=manifest,
        )
        print_run_report(config.run_dir, command, len(raw_files), summary.get("total_trajectories", 0), summary)
        return 130

    summary = collect_hle_outputs_live(
        config=config,
        raw_files=raw_files,
        trajectories_dir=trajectories_dir,
        benchmark_dir=benchmark_dir,
        manifest=manifest,
    )
    print_run_report(config.run_dir, command, len(raw_files), summary.get("total_trajectories", 0), summary)
    return 0


def run_one_hle_example(
    example: dict[str, Any],
    config: HLEConfig,
    api_key: str,
    *,
    checkpoint_path: Path | None = None,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chat_fn = chat_fn or call_chat_completions
    messages = initial_hle_messages(example)
    raw_responses: list[dict[str, Any]] = []
    _write_hle_checkpoint(checkpoint_path, example, messages, config.model, raw_responses, "started")
    try:
        for _ in range(config.max_steps):
            response = chat_fn(config=config, api_key=api_key, messages=messages)
            raw_responses.append(response)
            message = response["choices"][0]["message"]
            assistant_message = _assistant_message_for_storage(message)
            messages.append(assistant_message)
            _write_hle_checkpoint(checkpoint_path, example, messages, config.model, raw_responses, "in_progress")
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                break
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                arguments = _parse_json(function.get("arguments") or "{}")
                if function.get("name") == "calculator":
                    result = run_calculator_tool(arguments)
                else:
                    result = {"error": f"Unknown tool: {function.get('name')}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                _write_hle_checkpoint(checkpoint_path, example, messages, config.model, raw_responses, "in_progress")
    except BaseException as exc:
        _write_hle_checkpoint(checkpoint_path, example, messages, config.model, raw_responses, "error", error=exc)
        raise

    record = to_hle_raw_record(example=example, messages=messages, model=config.model, raw_responses=raw_responses)
    record["status"] = "completed"
    record["partial"] = False
    if checkpoint_path is not None:
        write_json(checkpoint_path, record)
    return record


def _write_hle_checkpoint(
    checkpoint_path: Path | None,
    example: dict[str, Any],
    messages: list[dict[str, Any]],
    model: str,
    raw_responses: list[dict[str, Any]],
    status: str,
    error: BaseException | None = None,
) -> None:
    if checkpoint_path is None:
        return
    write_json(
        checkpoint_path,
        to_hle_checkpoint_raw_record(
            example=example,
            messages=messages,
            model=model,
            raw_responses=raw_responses,
            status=status,
            error=error,
        ),
    )


def run_hle_batch(
    examples: list[dict[str, Any]],
    *,
    config: HLEConfig,
    api_key: str,
    raw_dir: Path,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
    on_raw_files_changed: Callable[[list[Path]], None] | None = None,
) -> list[Path]:
    progress = ProgressPrinter(label="HLE", total=len(examples), interval_seconds=config.progress_interval)
    progress.update(0, extra=f"workers={config.max_workers}", force=True)
    if config.max_workers == 1:
        raw_files: list[Path] = []
        for index, example in enumerate(examples):
            raw_file = _run_hle_example_to_raw(
                index,
                example,
                config=config,
                api_key=api_key,
                raw_dir=raw_dir,
                chat_fn=chat_fn,
            )
            raw_files.append(raw_file)
            if on_raw_files_changed is not None:
                on_raw_files_changed(list(raw_files))
            progress.update(
                len(raw_files),
                extra=f"last={raw_file.name}",
                force=len(raw_files) == len(examples),
            )
        return raw_files

    completed: list[tuple[int, Path]] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                _run_hle_example_to_raw,
                index,
                example,
                config=config,
                api_key=api_key,
                raw_dir=raw_dir,
                chat_fn=chat_fn,
            ): index
            for index, example in enumerate(examples)
        }
        for future in as_completed(futures):
            raw_file = future.result()
            completed.append((futures[future], raw_file))
            if on_raw_files_changed is not None:
                on_raw_files_changed([path for _, path in sorted(completed, key=lambda item: item[0])])
            progress.update(
                len(completed),
                extra=f"last={raw_file.name}",
                force=len(completed) == len(examples),
            )
    return [path for _, path in sorted(completed, key=lambda item: item[0])]


def _run_hle_example_to_raw(
    index: int,
    example: dict[str, Any],
    *,
    config: HLEConfig,
    api_key: str,
    raw_dir: Path,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> Path:
    output_path = raw_path_for_example(raw_dir, example, index)
    if config.resume and output_path.exists():
        return output_path
    try:
        record = run_one_hle_example(
            example,
            config=config,
            api_key=api_key,
            checkpoint_path=output_path,
            chat_fn=chat_fn,
        )
    except Exception as exc:  # noqa: BLE001 - each failed task is preserved for later inspection
        if output_path.exists():
            return output_path
        record = to_hle_error_raw_record(example=example, model=config.model, error=exc)
    write_json(output_path, record)
    return output_path


def call_chat_completions(config: HLEConfig, api_key: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "model": api_model_name(config.model),
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a safe arithmetic expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Arithmetic expression, e.g. '(2 + 3) * 4'.",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ],
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    for attempt in range(config.max_retries + 1):
        request = urllib.request.Request(
            f"{config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in _RETRYABLE_HTTP_STATUS_CODES or attempt >= config.max_retries:
                raise RuntimeError(f"Chat completions request failed: HTTP {exc.code}: {body}") from exc
            _sleep_before_retry(config, attempt)
        except _TRANSIENT_REQUEST_ERRORS as exc:
            if attempt >= config.max_retries:
                raise RuntimeError(
                    f"Chat completions request failed after {attempt + 1} attempts: "
                    f"{exc.__class__.__name__}: {exc}"
                ) from exc
            _sleep_before_retry(config, attempt)
    raise RuntimeError("Chat completions request failed without a response")


_RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_REQUEST_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    http.client.RemoteDisconnected,
    http.client.HTTPException,
    ConnectionResetError,
    ConnectionError,
)


def _sleep_before_retry(config: HLEConfig, attempt: int) -> None:
    if config.retry_sleep <= 0:
        return
    time.sleep(config.retry_sleep * (attempt + 1))


def api_model_name(model: str) -> str:
    """Convert LiteLLM-style OpenAI provider names for direct chat-completions calls."""
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _collect_outputs(
    config: HLEConfig,
    raw_files: list[Path],
    trajectories_dir: Path,
    benchmark_dir: Path,
) -> dict[str, Any]:
    if not raw_files:
        raw_files = json_like_files(config.run_dir / "raw")
    trajectories = collect_raw_files(
        raw_files=raw_files,
        trajectories_dir=trajectories_dir,
        jsonl_path=config.run_dir / "trajectories.jsonl",
        run_id=config.run_id,
        defaults={
            "domain": "hle",
            "agent_model": config.model,
            "user_model": None,
        },
        raw_root=config.run_dir,
    )
    benchmark_records = write_benchmark_records(
        trajectories=trajectories,
        output_dir=benchmark_dir,
        jsonl_path=config.run_dir / "benchmark_trajectories.jsonl",
        benchmark=config.benchmark,
    )
    summary = summarize_trajectories(trajectories)
    summary["benchmark_task_count"] = len(benchmark_records)
    summary["benchmark_trajectory_count"] = sum(len(record.get("trajectories", [])) for record in benchmark_records)
    return summary


def collect_hle_outputs_live(
    *,
    config: HLEConfig,
    raw_files: list[Path],
    trajectories_dir: Path,
    benchmark_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    raw_files = sorted(raw_files)
    summary = _collect_outputs(config, raw_files, trajectories_dir, benchmark_dir)
    manifest["raw_files"] = [str(path.relative_to(config.run_dir)) for path in raw_files]
    manifest["completed_or_error_raw_files"] = len(raw_files)
    write_json(config.run_dir / "summary.json", summary)
    write_json(config.run_dir / "manifest.json", manifest)
    return summary


def _normalize_hle_example(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        item = {"question": str(item)}
    task_id = item.get("task_id") or item.get("id") or item.get("question_id") or f"hle_{index:04d}"
    instruction = item.get("instruction") or item.get("question") or item.get("prompt") or item.get("problem")
    answer = item.get("answer") or item.get("target") or item.get("final_answer") or item.get("gold_answer")
    return {
        "task_id": str(task_id),
        "instruction": "" if instruction is None else str(instruction),
        "answer": None if answer is None else str(answer),
        "raw": item,
    }


def _safe_eval_expression(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    return _eval_ast_node(tree.body)


def _eval_ast_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    raise ValueError("Only arithmetic expressions are allowed")


_ALLOWED_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: math.pow(a, b),
}


def _assistant_message_for_storage(message: dict[str, Any]) -> dict[str, Any]:
    stored = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        stored["tool_calls"] = message["tool_calls"]
    return stored


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}


def _last_assistant_content(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message.get("content"))
    return None


def _grade_exact(answer: str | None, reference: Any) -> bool | None:
    if reference is None or answer is None:
        return None
    return str(answer).strip().lower() == str(reference).strip().lower()


def _apply_overrides(env: dict[str, str], args: argparse.Namespace) -> None:
    overrides = {
        "HLE_BENCHMARK": args.benchmark,
        "HLE_DATA_PATH": str(args.data_path) if args.data_path else None,
        "AGENT_MODEL": args.model,
        "OPENAI_BASE_URL": args.base_url,
        "NUM_TASKS": args.num_tasks,
        "HLE_ALL_TASKS": "1" if args.all_tasks else None,
        "MAX_STEPS": args.max_steps,
        "HLE_TEMPERATURE": args.temperature,
        "HLE_INCLUDE_MULTIMODAL": "1" if args.include_multimodal else None,
        "HLE_MAX_WORKERS": args.max_workers,
        "HLE_RESUME": "1" if args.resume else None,
        "HLE_RUN_ID": args.run_id,
        "HLE_REQUEST_TIMEOUT": args.request_timeout,
        "HLE_MAX_RETRIES": args.max_retries,
        "HLE_RETRY_SLEEP": args.retry_sleep,
        "HLE_PROGRESS_INTERVAL": args.progress_interval,
        "OUTPUT_DIR": str(args.output_dir) if args.output_dir else None,
    }
    for key, value in overrides.items():
        if value is not None:
            env[key] = str(value)


def _env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
