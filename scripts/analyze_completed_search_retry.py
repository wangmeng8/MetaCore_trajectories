import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_IDS_PATH = (
    ROOT / "data/retry/hle_failed_web_browsing_20260727_450.parquet.ids.json"
)
OUTPUT_PATH = (
    ROOT
    / "outputs/runs/hle_retry_failed_web_browsing_20260727_gpt55"
    / "completed_357_quality_analysis.json"
)
RUNS = [
    (
        "mixed_stopped",
        ROOT / "outputs/runs/hle_retry_failed_search_tools_20260724_gpt55",
        "hle_gpt-5.5.json.temp",
    ),
    (
        "web_only",
        ROOT / "outputs/runs/hle_retry_failed_web_browsing_20260727_gpt55",
        "hle_gpt-5.5.json",
    ),
]
TOOLS = ("web_browsing", "scientific_search", "code_interpreter")

RESULT_RE = re.compile(
    r"(?m)^.*?([\u2713\u2717])\s+TOOL RESULT\s+"
    r"(web_browsing|scientific_search|code_interpreter)\s+\[[^\r\n]+$"
)
CALL_RE = re.compile(
    r"(?m)^.*?TOOL CALL\s+"
    r"(web_browsing|scientific_search|code_interpreter)\s+\[[^\r\n]+$"
)
API_RE = re.compile(
    r"(?im)^(?:(?!system:|user:|assistant:).)*"
    r"(LLM API ERROR|APIConnectionError|APITimeoutError|RateLimitError|"
    r"BadRequestError|InternalServerError|API call failed|"
    r"Connection error while calling (?:the )?API).*$"
)


def classify_failure(tool, block):
    text = block.lower()
    if tool == "web_browsing":
        if (
            "no results" in text
            or "no search results" in text
            or "returned 0 result" in text
        ):
            return "no_results"
        if "all results" in text and ("blocked" in text or "filtered" in text):
            return "all_results_blocked"
        if "timed out" in text or "timeout" in text:
            return "timeout"
        if any(
            phrase in text
            for phrase in (
                "cannot decrypt",
                "connection error",
                "cannot connect",
                "requesterror",
                "ssl:",
                "tls",
                "connection reset",
            )
        ):
            return "transport_or_tls_error"
        if "error searching for" in text:
            return "search_backend_error_other"
        return "other"
    if tool == "scientific_search":
        if "science-api.activeloop.ai" in text and (
            "cannot connect" in text or "connection" in text
        ):
            return "science_api_unreachable"
        if "timed out" in text or "timeout" in text:
            return "timeout"
        if "error searching scientific" in text:
            return "scientific_search_error_other"
        return "other"
    for exception_name in ("NameError", "SyntaxError", "ModuleNotFoundError"):
        if exception_name.lower() in text:
            return exception_name
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "traceback" in text or "error" in text or "exception" in text:
        return "execution_error_other"
    return "other"


def empty_tool_stats():
    return {
        tool: {
            "calls": 0,
            "results": 0,
            "success": 0,
            "failed": 0,
            "affected_failed_trajectories": set(),
            "failure_reasons": Counter(),
            "truncated_results": 0,
            "affected_truncated_trajectories": set(),
        }
        for tool in TOOLS
    }


def serialize_tool_stats(stats):
    serialized = {}
    for tool, values in stats.items():
        item = {}
        for key, value in values.items():
            if isinstance(value, set):
                item[key] = sorted(value)
            elif isinstance(value, Counter):
                item[key] = dict(value)
            else:
                item[key] = value
        item["failure_rate_pct"] = (
            round(100 * values["failed"] / values["results"], 2)
            if values["results"]
            else None
        )
        item["affected_failed_trajectory_count"] = len(
            values["affected_failed_trajectories"]
        )
        item.pop("affected_failed_trajectories")
        item["affected_truncated_trajectory_count"] = len(
            values["affected_truncated_trajectories"]
        )
        item.pop("affected_truncated_trajectories")
        serialized[tool] = item
    return serialized


def main():
    target_ids = set(json.loads(TARGET_IDS_PATH.read_text(encoding="utf-8")))
    assigned = {}
    records = {}
    source_counts = {}
    for name, run_dir, prediction_name in RUNS:
        predictions = json.loads(
            (run_dir / "raw/official_run" / prediction_name).read_text(
                encoding="utf-8"
            )
        )
        completed_ids = target_ids & set(predictions)
        source_counts[name] = len(completed_ids)
        for task_id in completed_ids:
            if task_id in assigned:
                raise RuntimeError(f"Duplicate completed task: {task_id}")
            assigned[task_id] = (name, run_dir)
            records[task_id] = predictions[task_id]

    overall = empty_tool_stats()
    by_source = {name: empty_tool_stats() for name, _, _ in RUNS}
    api_counts = Counter()
    api_ids = defaultdict(set)
    usage = Counter()
    missing_traces = []
    empty_responses = []
    zero_token_records = []
    call_result_mismatches = []
    loop_ids = set()
    max_iteration_ids = set()
    any_tool_failure_ids = set()
    any_tool_failure_ids_by_source = defaultdict(set)
    output_cap_ids = set()
    output_cap_ids_by_source = defaultdict(set)
    failure_examples = defaultdict(list)

    for task_id, (source_name, run_dir) in assigned.items():
        record = records[task_id]
        response = record.get("response")
        if not isinstance(response, str) or not response.strip():
            empty_responses.append(task_id)
        record_usage = record.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int(record_usage.get(key) or 0)
        if int(record_usage.get("total_tokens") or 0) == 0:
            zero_token_records.append(task_id)

        trace_path = (
            run_dir / "raw/official_run/traces" / f"trace_{task_id}.log"
        )
        if not trace_path.exists():
            missing_traces.append(task_id)
            continue
        text = trace_path.read_text(encoding="utf-8", errors="replace")

        for match in API_RE.finditer(text):
            error_name = match.group(1)
            api_counts[error_name] += 1
            api_ids[error_name].add(task_id)
        if "Potential tool call loop detected" in text:
            loop_ids.add(task_id)
        if (
            "[Results truncated due to length limit...]" in text
            or "[Content truncated due to length limit...]" in text
        ):
            output_cap_ids.add(task_id)
            output_cap_ids_by_source[source_name].add(task_id)
        if re.search(
            r"(?i)max(?:imum)? iterations? (?:reached|exceeded)|"
            r"reached max_iterations",
            text,
        ):
            max_iteration_ids.add(task_id)

        calls = Counter(CALL_RE.findall(text))
        results = Counter()
        result_matches = list(RESULT_RE.finditer(text))
        for index, match in enumerate(result_matches):
            status, tool = match.group(1), match.group(2)
            results[tool] += 1
            block_end = (
                result_matches[index + 1].start()
                if index + 1 < len(result_matches)
                else min(len(text), match.end() + 6000)
            )
            block = text[match.end() : block_end]
            block = re.split(
                r"(?m)^.*(?:Duration:|STEP \d+/|LLM API CALL).*$",
                block,
                maxsplit=1,
            )[0]
            for bucket in (overall[tool], by_source[source_name][tool]):
                bucket["results"] += 1
                if ord(status) == 0x2713:
                    bucket["success"] += 1
                else:
                    bucket["failed"] += 1
                    bucket["affected_failed_trajectories"].add(task_id)
                    bucket["failure_reasons"][
                        classify_failure(tool, block)
                    ] += 1

            if ord(status) == 0x2717:
                any_tool_failure_ids.add(task_id)
                any_tool_failure_ids_by_source[source_name].add(task_id)
                reason = classify_failure(tool, block)
                example_key = f"{tool}:{reason}"
                if len(failure_examples[example_key]) < 3:
                    clean_detail = " ".join(
                        line.strip(" |\u2502\u250c\u2500")
                        for line in block.splitlines()
                        if line.strip()
                    )
                    failure_examples[example_key].append(
                        {
                            "id": task_id,
                            "source": source_name,
                            "detail": clean_detail[:500],
                        }
                    )

        for tool in TOOLS:
            overall[tool]["calls"] += calls[tool]
            by_source[source_name][tool]["calls"] += calls[tool]
            if calls[tool] != results[tool]:
                call_result_mismatches.append(
                    {
                        "id": task_id,
                        "tool": tool,
                        "calls": calls[tool],
                        "results": results[tool],
                        "source": source_name,
                    }
                )

    all_api_ids = set().union(*api_ids.values()) if api_ids else set()
    report = {
        "scope": {
            "target_ids": len(target_ids),
            "completed_analyzed": len(assigned),
            "not_completed": len(target_ids - set(assigned)),
            "source_completed_counts": source_counts,
        },
        "prediction_integrity": {
            "missing_trace_files": missing_traces,
            "empty_responses": empty_responses,
            "zero_total_token_records": zero_token_records,
            "usage_totals": dict(usage),
        },
        "api_errors": {
            "event_counts": dict(api_counts),
            "affected_trajectory_counts": {
                key: len(value) for key, value in api_ids.items()
            },
            "affected_trajectory_count_any": len(all_api_ids),
        },
        "tool_results": serialize_tool_stats(overall),
        "tool_results_by_source": {
            name: serialize_tool_stats(stats)
            for name, stats in by_source.items()
        },
        "any_tool_failure": {
            "affected_trajectory_count": len(any_tool_failure_ids),
            "affected_trajectory_counts_by_source": {
                name: len(ids)
                for name, ids in any_tool_failure_ids_by_source.items()
            },
            "ids": sorted(any_tool_failure_ids),
        },
        "tool_call_result_mismatches": call_result_mismatches,
        "tool_call_loop_warning": {
            "affected_trajectory_count": len(loop_ids),
            "ids": sorted(loop_ids),
        },
        "max_iteration_markers": {
            "affected_trajectory_count": len(max_iteration_ids),
            "ids": sorted(max_iteration_ids),
        },
        "tool_output_length_cap": {
            "configured_character_limit": 10000,
            "affected_trajectory_count": len(output_cap_ids),
            "affected_trajectory_counts_by_source": {
                name: len(ids) for name, ids in output_cap_ids_by_source.items()
            },
            "ids": sorted(output_cap_ids),
            "meaning": (
                "The web tool intentionally capped returned text at 10,000 "
                "characters. This is not an LLM completion-token truncation."
            ),
        },
        "failure_examples": dict(failure_examples),
        "process_notes": {
            "web_only_first_child_exit_code": 3221225477,
            "hex": "0xC0000005",
            "wrapper_recovered_and_final_return_code": 0,
            "web_only_missing_predictions": 93,
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
