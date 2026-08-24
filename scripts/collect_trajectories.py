"""Collect Tau2 raw simulation files into normalized trajectory JSON."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


COMMON_TRAJECTORY_KEYS = ("messages", "conversation", "trajectory", "events", "turns", "steps")
TASK_ID_KEYS = ("task_id", "taskId", "task", "scenario_id", "scenarioId")
TRIAL_ID_KEYS = ("trial_id", "trialId", "trial", "seed")
SUCCESS_KEYS = ("success", "is_success", "isSuccess", "succeeded")
SCORE_KEYS = ("score", "reward", "final_reward", "finalReward")
INSTRUCTION_KEYS = ("instruction", "prompt", "question", "user_request", "task_instruction", "task_instructions")


def read_json_records(path: Path) -> list[Any]:
    return [record for record, _ in read_json_records_with_context(path)]


def read_json_records_with_context(path: Path) -> list[tuple[Any, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        records: list[tuple[Any, dict[str, Any]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append((json.loads(line), {}))
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [(item, {}) for item in payload]
    if isinstance(payload, dict):
        split_records = _records_from_container_with_context(payload)
        return split_records if split_records is not None else [(payload, {})]
    return [(payload, {})]


def collect_raw_files(
    raw_files: Iterable[Path],
    trajectories_dir: Path,
    jsonl_path: Path,
    run_id: str,
    defaults: dict[str, Any],
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    _clear_generated_json_files(trajectories_dir)

    trajectories: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for raw_path in raw_files:
            if raw_path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            try:
                records = read_json_records_with_context(raw_path)
            except (OSError, json.JSONDecodeError) as exc:
                records = [({"raw_parse_error": str(exc), "raw_file": str(raw_path)}, {})]

            for record, context in records:
                raw_file = _relative_path(raw_path, raw_root)
                merged_context = {**_context_from_raw_file(raw_path), **context}
                normalized = normalize_raw_object(
                    record,
                    run_id=run_id,
                    raw_file=raw_file,
                    defaults=defaults,
                    context=merged_context,
                )
                output_name = _trajectory_filename(normalized, len(trajectories))
                output_path = trajectories_dir / output_name
                output_path.write_text(
                    json.dumps(normalized, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                jsonl_file.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                trajectories.append(normalized)

    return trajectories


def normalize_raw_object(
    raw: Any,
    run_id: str,
    raw_file: str,
    defaults: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = defaults or {}
    context = context or {}
    task_id = _first_non_none(
        _record_task_id(raw),
        _first_value(raw, TASK_ID_KEYS),
        _context_task_id(context),
        _task_id_from_raw_file(raw_file),
    )
    trial_id = _first_non_none(
        _first_value(raw, TRIAL_ID_KEYS),
        _context_trial_id(context),
        _trial_id_from_raw_file(raw_file),
    )
    domain = _first_value(raw, ("domain", "environment")) or defaults.get("domain")
    success = _first_value(raw, SUCCESS_KEYS)
    score = _first_value(raw, SCORE_KEYS)
    if score is None:
        score = _context_score(context)
    if success is None:
        success = _context_success(context, score)
    events = extract_events(raw)
    task_instruction = _extract_task_instruction(raw, context)

    return {
        "run_id": run_id,
        "domain": domain,
        "task_id": _stringify_optional(task_id),
        "trial_id": _parse_int_or_original(trial_id),
        "agent_model": _first_value(raw, ("agent_model", "agentModel", "agent_llm", "agentLlm"))
        or _nested_value(raw, ("agent", "model_name"))
        or _context_agent_model(context)
        or defaults.get("agent_model"),
        "user_model": _first_value(raw, ("user_model", "userModel", "user_llm", "userLlm"))
        or defaults.get("user_model"),
        "success": _parse_bool_or_original(success),
        "score": _parse_float_or_original(score),
        "num_turns": sum(1 for event in events if event.get("type") in {"message", "tool_result"}),
        "num_tool_calls": sum(1 for event in events if event.get("type") == "tool_call"),
        "task": {"instruction": task_instruction},
        "events": events,
        "raw": raw,
        "raw_file": raw_file,
    }


def extract_events(raw: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    conversation = _find_conversation(raw)
    if isinstance(conversation, list):
        for item in conversation:
            _append_events_from_item(events, item)

    if isinstance(raw, dict):
        for call in _ensure_list(raw.get("tool_calls")):
            _append_tool_call(events, call)
        for result in _ensure_list(raw.get("observations") or raw.get("tool_results")):
            _append_tool_result(events, result)

    for index, event in enumerate(events):
        event["index"] = index
    return events


def summarize_trajectories(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(trajectories)
    known_success = [item.get("success") for item in trajectories if isinstance(item.get("success"), bool)]
    success_count = sum(1 for value in known_success if value)
    turn_counts = [item.get("num_turns") for item in trajectories if isinstance(item.get("num_turns"), int)]
    tool_counts = [item.get("num_tool_calls") for item in trajectories if isinstance(item.get("num_tool_calls"), int)]
    scores = [item.get("score") for item in trajectories if isinstance(item.get("score"), (int, float))]
    infrastructure_error_count = 0
    error_types: dict[str, int] = {}
    for item in trajectories:
        raw = item.get("raw")
        if not isinstance(raw, dict):
            continue
        if raw.get("termination_reason") == "infrastructure_error":
            infrastructure_error_count += 1
        info = raw.get("info")
        if isinstance(info, dict) and info.get("error_type"):
            error_type = str(info["error_type"])
            error_types[error_type] = error_types.get(error_type, 0) + 1

    return {
        "total_trajectories": total,
        "success_count": success_count if known_success else None,
        "success_rate": (success_count / len(known_success)) if known_success else None,
        "average_turns": _average(turn_counts),
        "average_tool_calls": _average(tool_counts),
        "average_score": _average(scores),
        "infrastructure_error_count": infrastructure_error_count,
        "error_types": error_types,
    }


def serialize_trajectory_text(normalized: dict[str, Any]) -> str:
    lines: list[str] = []
    for event in _ensure_list(normalized.get("events")):
        if not isinstance(event, dict):
            lines.append(f"UNKNOWN: {_serialize_text_value(event)}")
            continue

        event_type = event.get("type")
        role = str(event.get("role") or "unknown").upper()

        if event_type == "message":
            lines.append(f"{role}: {_serialize_text_value(event.get('content'))}")
        elif event_type == "tool_call":
            tool_name = event.get("tool_name") or "unknown_tool"
            lines.append(
                f"{role} TOOL_CALL {tool_name}: "
                f"{_serialize_text_value(event.get('arguments'))}"
            )
        elif event_type == "tool_result":
            tool_name = event.get("tool_name") or "unknown_tool"
            lines.append(f"TOOL RESULT {tool_name}: {_serialize_text_value(event.get('content'))}")
        else:
            lines.append(f"{role} {event_type or 'event'}: {_serialize_text_value(event)}")
    return "\n".join(lines)


def to_benchmark_record(normalized: dict[str, Any], benchmark: str) -> dict[str, Any]:
    task_id = _stringify_optional(normalized.get("task_id")) or "unknown_task"
    source_model = _stringify_optional(normalized.get("agent_model")) or "unknown_model"
    trajectory_id = _build_trajectory_id(normalized, task_id, source_model)
    task = normalized.get("task") if isinstance(normalized.get("task"), dict) else {}
    instruction = task.get("instruction") or _extract_task_instruction(normalized.get("raw"), {})

    return {
        "task_id": task_id,
        "benchmark": benchmark,
        "task": {
            "instruction": instruction,
        },
        "trajectories": [
            {
                "trajectory_id": trajectory_id,
                "source_model": source_model,
                "trajectory_text": serialize_trajectory_text(normalized),
                "metadata": {
                    "run_id": normalized.get("run_id"),
                    "domain": normalized.get("domain"),
                    "trial_id": normalized.get("trial_id"),
                    "agent_model": normalized.get("agent_model"),
                    "user_model": normalized.get("user_model"),
                    "success": normalized.get("success"),
                    "score": normalized.get("score"),
                    "num_turns": normalized.get("num_turns"),
                    "num_tool_calls": normalized.get("num_tool_calls"),
                    "raw_file": normalized.get("raw_file"),
                },
            }
        ],
    }


def write_benchmark_records(
    trajectories: list[dict[str, Any]],
    output_dir: Path,
    jsonl_path: Path,
    benchmark: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    _clear_generated_json_files(output_dir)

    records = build_benchmark_records(trajectories, benchmark=benchmark)
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for index, record in enumerate(records):
            output_path = output_dir / _benchmark_filename(record, index)
            output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def build_benchmark_records(trajectories: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for normalized in trajectories:
        single_record = to_benchmark_record(normalized, benchmark=benchmark)
        task_id = single_record["task_id"]
        if task_id not in grouped:
            grouped[task_id] = {
                "task_id": task_id,
                "benchmark": benchmark,
                "task": single_record["task"],
                "trajectories": [],
            }
            order.append(task_id)
        if not grouped[task_id]["task"].get("instruction") and single_record["task"].get("instruction"):
            grouped[task_id]["task"] = single_record["task"]
        grouped[task_id]["trajectories"].extend(single_record["trajectories"])
    return [grouped[task_id] for task_id in order]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize Tau2 raw simulation JSON into trajectory files.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing copied Tau2 raw files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--run-id", help="Run identifier to write into each trajectory. Defaults to the output directory name.")
    parser.add_argument("--benchmark", default="tau2-bench", help="Benchmark/dataset name for training export.")
    parser.add_argument("--domain", default="airline", help="Default Tau2 domain.")
    parser.add_argument("--agent-model", default="gpt-5.5", help="Default agent model.")
    parser.add_argument("--user-model", default="gpt-5.5", help="Default user simulator model. Use an empty value for benchmarks without a user simulator.")
    args = parser.parse_args(argv)

    run_id = args.run_id or infer_run_id(args.output_dir)
    raw_files = sorted(path for path in args.raw_dir.rglob("*") if path.is_file())
    trajectories_dir = args.output_dir / "trajectories"
    jsonl_path = args.output_dir / "trajectories.jsonl"
    trajectories = collect_raw_files(
        raw_files=raw_files,
        trajectories_dir=trajectories_dir,
        jsonl_path=jsonl_path,
        run_id=run_id,
        defaults={
            "domain": args.domain,
            "agent_model": args.agent_model,
            "user_model": args.user_model or None,
        },
        raw_root=args.output_dir,
    )
    benchmark_records = write_benchmark_records(
        trajectories=trajectories,
        output_dir=args.output_dir / "benchmark_trajectories",
        jsonl_path=args.output_dir / "benchmark_trajectories.jsonl",
        benchmark=args.benchmark,
    )
    summary = summarize_trajectories(trajectories)
    summary["benchmark_task_count"] = len(benchmark_records)
    summary["benchmark_trajectory_count"] = _count_benchmark_trajectories(benchmark_records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def infer_run_id(output_dir: Path) -> str:
    name = output_dir.name
    if name:
        return name
    return f"collect_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"


def _records_from_container(payload: dict[str, Any]) -> list[Any] | None:
    records_with_context = _records_from_container_with_context(payload)
    if records_with_context is None:
        return None
    return [record for record, _ in records_with_context]


def _records_from_container_with_context(payload: dict[str, Any]) -> list[tuple[Any, dict[str, Any]]] | None:
    key = _container_records_key(payload)
    if key is None:
        return None

    task_contexts = _build_task_context_index(payload.get("tasks"))
    records = payload.get(key)
    if not isinstance(records, list):
        return None
    return [(record, _context_for_record(record, task_contexts)) for record in records]


def _container_records_key(payload: dict[str, Any]) -> str | None:
    if any(key in payload for key in COMMON_TRAJECTORY_KEYS):
        return None
    for key in ("trajectories", "simulations", "results", "records"):
        value = payload.get(key)
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return key
    return None


def _build_task_context_index(tasks: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(tasks, list):
        return {}

    contexts: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = _task_id_from_task(task)
        instruction = _task_instruction_from_task(task)
        if task_id is not None:
            contexts[task_id] = {"task": {"instruction": instruction}}
    return contexts


def _context_for_record(record: Any, task_contexts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    task_id = _record_task_id(record)
    if task_id is None:
        return {}
    return dict(task_contexts.get(task_id, {}))


def _record_task_id(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    for key in ("task_id", "taskId", "scenario_id", "scenarioId"):
        if key in record:
            return _stringify_optional(record[key])
    task = record.get("task")
    if isinstance(task, dict):
        return _task_id_from_task(task)
    if task is not None:
        return _stringify_optional(task)
    return None


def _task_id_from_task(task: dict[str, Any]) -> str | None:
    for key in ("id", "task_id", "taskId", "scenario_id", "scenarioId"):
        if key in task:
            return _stringify_optional(task[key])
    return None


def _extract_task_instruction(raw: Any, context: dict[str, Any]) -> str | None:
    context_task = context.get("task") if isinstance(context, dict) else None
    if isinstance(context_task, dict) and "instruction" in context_task:
        return context_task.get("instruction")

    tau2_instruction = _task_instruction_from_task(raw)
    if tau2_instruction:
        return tau2_instruction

    atif_instruction = _task_instruction_from_atif(raw)
    if atif_instruction:
        return atif_instruction

    direct = _first_value(raw, INSTRUCTION_KEYS)
    return _text_value(direct)


def _task_instruction_from_task(task: Any) -> str | None:
    if not isinstance(task, dict):
        return None

    parts: list[str] = []
    _append_instruction_part(parts, "Purpose", _nested_value(task, ("description", "purpose")))

    instructions = _nested_value(task, ("user_scenario", "instructions"))
    if isinstance(instructions, dict):
        _append_instruction_part(parts, "Reason for call", instructions.get("reason_for_call"))
        _append_instruction_part(parts, "Known info", instructions.get("known_info"))
        _append_instruction_part(parts, "Task instructions", instructions.get("task_instructions"))
    elif instructions is not None:
        _append_instruction_part(parts, "Instructions", instructions)

    if parts:
        return "\n".join(parts)

    direct = _first_present(task, INSTRUCTION_KEYS)
    return _text_value(direct)


def _task_instruction_from_atif(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    steps = raw.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, dict):
            continue
        if _normalize_role(step.get("source")) != "user":
            continue
        message = step.get("message")
        if not isinstance(message, str):
            continue
        match = re.search(
            r"Task Description:\s*(?P<instruction>.*?)(?:\n\s*Current terminal state:|\Z)",
            message,
            flags=re.DOTALL,
        )
        if match:
            return match.group("instruction").strip()
    return None


def _append_instruction_part(parts: list[str], label: str, value: Any) -> None:
    text = _text_value(value)
    if text:
        parts.append(f"{label}: {text}")


def _nested_value(item: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = item
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _build_trajectory_id(normalized: dict[str, Any], task_id: str, source_model: str) -> str:
    run_id = _stringify_optional(normalized.get("run_id")) or "run"
    trial_id = normalized.get("trial_id")
    parts = [task_id, source_model, run_id]
    if trial_id is not None:
        parts.append(f"trial_{trial_id}")
    return "_".join(_safe_identifier(part) for part in parts if part)


def _count_benchmark_trajectories(records: list[dict[str, Any]]) -> int:
    return sum(len(record.get("trajectories", [])) for record in records)


def _clear_generated_json_files(output_dir: Path) -> None:
    for path in output_dir.glob("*.json"):
        if path.is_file():
            path.unlink()


def _benchmark_filename(record: dict[str, Any], index: int) -> str:
    task_id = _stringify_optional(record.get("task_id"))
    safe_id = _safe_identifier(task_id or f"benchmark_task_{index:04d}")
    return f"{safe_id[:160]}.json"


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _serialize_text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _find_conversation(raw: Any) -> Any:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return None
    for key in COMMON_TRAJECTORY_KEYS:
        value = raw.get(key)
        if isinstance(value, list):
            return value
    for value in raw.values():
        if isinstance(value, dict):
            nested = _find_conversation(value)
            if nested is not None:
                return nested
    return None


def _append_events_from_item(events: list[dict[str, Any]], item: Any) -> None:
    if isinstance(item, str):
        events.append({"role": "unknown", "type": "message", "content": item})
        return
    if not isinstance(item, dict):
        events.append({"role": "unknown", "type": "message", "content": item})
        return

    if _looks_like_atif_step(item):
        _append_atif_step(events, item)
        return

    explicit_type = str(item.get("type", "")).lower()
    if "tool_call" in explicit_type or explicit_type == "action":
        _append_tool_call(events, item)
        return
    if "tool_result" in explicit_type or explicit_type in {"observation", "tool"}:
        _append_tool_result(events, item)
        return

    role = _normalize_role(item.get("role") or item.get("speaker") or item.get("from"))
    content = _first_present(item, ("content", "text", "utterance", "message", "response"))
    if content is None:
        for key, guessed_role in (
            ("user", "user"),
            ("user_utterance", "user"),
            ("assistant", "assistant"),
            ("assistant_response", "assistant"),
            ("agent", "assistant"),
            ("agent_response", "assistant"),
        ):
            if key in item:
                role = guessed_role
                content = item[key]
                break

    if role == "tool":
        _append_tool_result(events, item)
    elif content is not None:
        events.append({"role": role or "unknown", "type": "message", "content": content})

    for call in _ensure_list(item.get("tool_calls") or item.get("tool_call") or item.get("actions")):
        _append_tool_call(events, call)
    for result in _ensure_list(item.get("tool_results") or item.get("tool_result") or item.get("observation")):
        _append_tool_result(events, result)


def _append_atif_step(events: list[dict[str, Any]], step: dict[str, Any]) -> None:
    role = _normalize_role(step.get("source")) or "unknown"
    if "message" in step and step.get("message") is not None:
        events.append(
            {
                "role": role,
                "type": "message",
                "content": step.get("message"),
            }
        )

    for call in _ensure_list(step.get("tool_calls")):
        _append_tool_call(events, call)

    observation = step.get("observation")
    if isinstance(observation, dict) and isinstance(observation.get("results"), list):
        for result in observation["results"]:
            _append_tool_result(events, result)
    elif observation is not None:
        _append_tool_result(events, observation)


def _looks_like_atif_step(item: dict[str, Any]) -> bool:
    if "step_id" in item and "source" in item:
        return True
    source = item.get("source")
    return source is not None and "message" in item and ("timestamp" in item or "observation" in item)


def _append_tool_call(events: list[dict[str, Any]], call: Any) -> None:
    if not isinstance(call, dict):
        events.append(
            {
                "role": "assistant",
                "type": "tool_call",
                "tool_name": None,
                "arguments": call,
            }
        )
        return

    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    tool_name = (
        call.get("tool_name")
        or call.get("name")
        or call.get("tool")
        or call.get("action")
        or call.get("function_name")
        or function.get("name")
    )
    arguments = (
        call.get("arguments")
        or call.get("args")
        or call.get("input")
        or call.get("parameters")
        or function.get("arguments")
    )
    events.append(
        {
            "role": "assistant",
            "type": "tool_call",
            "tool_name": _stringify_optional(tool_name),
            "arguments": _parse_json_if_possible(arguments),
        }
    )


def _append_tool_result(events: list[dict[str, Any]], result: Any) -> None:
    if not isinstance(result, dict):
        events.append({"role": "tool", "type": "tool_result", "tool_name": None, "content": result})
        return
    events.append(
        {
            "role": "tool",
            "type": "tool_result",
            "tool_name": _stringify_optional(
                result.get("tool_name") or result.get("name") or result.get("tool") or result.get("action")
            ),
            "content": _first_present(result, ("content", "result", "observation", "output", "response")),
        }
    )


def _first_value(raw: Any, keys: Iterable[str]) -> Any:
    if not isinstance(raw, dict):
        return None
    for key in keys:
        if key in raw:
            return raw[key]
    for value in raw.values():
        if isinstance(value, dict):
            nested = _first_value(value, keys)
            if nested is not None:
                return nested
    return None


def _first_present(item: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_role(role: Any) -> str | None:
    if role is None:
        return None
    role_text = str(role).lower()
    if role_text in {"agent", "assistant", "bot"}:
        return "assistant"
    if role_text in {"user", "customer", "human"}:
        return "user"
    if role_text in {"tool", "function", "observation"}:
        return "tool"
    return role_text


def _parse_json_if_possible(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _parse_bool_or_original(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes", "1", "success", "succeeded"}:
            return True
        if lowered in {"false", "no", "0", "failed", "failure"}:
            return False
    return value


def _parse_float_or_original(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _parse_int_or_original(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _stringify_optional(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _average(values: list[int | float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _relative_path(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _context_from_raw_file(raw_path: Path) -> dict[str, Any]:
    if raw_path.name != "trajectory.json" or raw_path.parent.name != "agent":
        return {}
    trial_dir = raw_path.parent.parent
    context: dict[str, Any] = {}
    for key, filename in (("terminal_result", "result.json"), ("terminal_config", "config.json")):
        path = trial_dir / filename
        if not path.exists():
            continue
        try:
            context[key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return context


def _context_task_id(context: dict[str, Any]) -> str | None:
    result = context.get("terminal_result")
    if isinstance(result, dict):
        task_name = result.get("task_name")
        if task_name:
            return _terminal_task_name_part(task_name)
        task_id = result.get("task_id")
        if isinstance(task_id, dict) and task_id.get("name"):
            return _stringify_optional(task_id["name"])
        if task_id is not None:
            return _stringify_optional(task_id)

    config = context.get("terminal_config")
    task = _nested_value(config, ("task",)) if isinstance(config, dict) else None
    if isinstance(task, dict) and task.get("name"):
        return _terminal_task_name_part(task["name"])
    return None


def _context_trial_id(context: dict[str, Any]) -> str | None:
    result = context.get("terminal_result")
    if isinstance(result, dict) and result.get("trial_name"):
        return _stringify_optional(result["trial_name"])
    config = context.get("terminal_config")
    if isinstance(config, dict) and config.get("trial_name"):
        return _stringify_optional(config["trial_name"])
    return None


def _context_score(context: dict[str, Any]) -> Any:
    result = context.get("terminal_result")
    rewards = _nested_value(result, ("verifier_result", "rewards")) if isinstance(result, dict) else None
    if isinstance(rewards, dict):
        if "reward" in rewards:
            return rewards["reward"]
        for value in rewards.values():
            if isinstance(value, (int, float)):
                return value
    return None


def _context_success(context: dict[str, Any], score: Any) -> bool | None:
    if isinstance(score, (int, float)):
        return score > 0
    return None


def _context_agent_model(context: dict[str, Any]) -> str | None:
    result = context.get("terminal_result")
    model_info = _nested_value(result, ("agent_info", "model_info")) if isinstance(result, dict) else None
    if isinstance(model_info, dict):
        provider = model_info.get("provider")
        name = model_info.get("name")
        if provider and name:
            return f"{provider}/{name}"
        if name:
            return _stringify_optional(name)
    config = context.get("terminal_config")
    model_name = _nested_value(config, ("agent", "model_name")) if isinstance(config, dict) else None
    return _stringify_optional(model_name)


def _task_id_from_raw_file(raw_file: str) -> str | None:
    trial_name = _trial_dir_name_from_raw_file(raw_file)
    if not trial_name:
        return None
    return _task_name_from_trial_name(trial_name)


def _trial_id_from_raw_file(raw_file: str) -> str | None:
    return _trial_dir_name_from_raw_file(raw_file)


def _trial_dir_name_from_raw_file(raw_file: str) -> str | None:
    parts = Path(raw_file).parts
    for index, part in enumerate(parts):
        if part == "agent" and index > 0:
            return parts[index - 1]
    return None


def _terminal_task_name_part(task_name: Any) -> str | None:
    text = _stringify_optional(task_name)
    if not text:
        return None
    return text.split("/")[-1]


def _task_name_from_trial_name(trial_name: str) -> str:
    if "__" in trial_name:
        return trial_name.split("__", 1)[0]
    return trial_name


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _trajectory_filename(normalized: dict[str, Any], index: int) -> str:
    task_id = normalized.get("task_id")
    trial_id = normalized.get("trial_id")
    if task_id is not None:
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id)).strip("_") or f"trajectory_{index:04d}"
        safe_trial = "unknown" if trial_id is None else re.sub(r"[^A-Za-z0-9_.-]+", "_", str(trial_id))
        return f"{safe_task}_trial_{safe_trial}.json"
    return f"trajectory_{index:04d}.json"


if __name__ == "__main__":
    raise SystemExit(main())
