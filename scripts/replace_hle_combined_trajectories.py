import argparse
import copy
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl_atomic(path, records):
    temporary = path.with_name(path.name + ".replace.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def normalize_benchmark_record(record, combined_run_id):
    normalized = copy.deepcopy(record)
    task_id = str(normalized["task_id"])
    for trajectory in normalized.get("trajectories", []):
        source_model = trajectory.get("source_model", "gpt-5.5")
        trajectory["trajectory_id"] = (
            f"{task_id}_{source_model}_{combined_run_id}"
        )
        metadata = trajectory.setdefault("metadata", {})
        metadata["run_id"] = combined_run_id
        metadata["dataset"] = "cais/hle"
    return normalized


def normalize_standard_record(record, combined_run_id):
    normalized = copy.deepcopy(record)
    normalized["run_id"] = combined_run_id
    normalized["domain"] = "cais/hle"
    return normalized


def replace_records(existing, replacements, label):
    existing_ids = [str(record["task_id"]) for record in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError(f"Duplicate task IDs in existing {label}")
    replacement_ids = set(replacements)
    missing = replacement_ids - set(existing_ids)
    if missing:
        raise ValueError(
            f"{len(missing)} replacement IDs are absent from existing {label}"
        )
    return [
        replacements.get(str(record["task_id"]), record) for record in existing
    ]


def backup_replaced_artifacts(
    backup_dir,
    combined_dir,
    replacement_ids,
    benchmark_records,
    standard_records,
    raw_predictions,
    provenance,
):
    backup_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl_atomic(
        backup_dir / "benchmark_trajectories.replaced.jsonl",
        [
            record
            for record in benchmark_records
            if str(record["task_id"]) in replacement_ids
        ],
    )
    write_jsonl_atomic(
        backup_dir / "trajectories.replaced.jsonl",
        [
            record
            for record in standard_records
            if str(record["task_id"]) in replacement_ids
        ],
    )
    write_json(
        backup_dir / "raw_predictions.replaced.json",
        {
            task_id: raw_predictions[task_id]
            for task_id in sorted(replacement_ids)
        },
    )
    write_json(
        backup_dir / "provenance.replaced.json",
        {task_id: provenance[task_id] for task_id in sorted(replacement_ids)},
    )

    trace_backup = backup_dir / "traces"
    code_backup = backup_dir / "code"
    trace_backup.mkdir()
    code_backup.mkdir()
    combined_raw = combined_dir / "raw/official_run"
    for task_id in sorted(replacement_ids):
        trace_path = combined_raw / "traces" / f"trace_{task_id}.log"
        if trace_path.exists():
            shutil.copy2(trace_path, trace_backup / trace_path.name)
        for code_path in (combined_raw / "code").glob(f"{task_id}*"):
            if code_path.is_file():
                shutil.copy2(code_path, code_backup / code_path.name)


def recalculate_summary(
    old_summary,
    benchmark_records,
    standard_records,
    raw_predictions,
    provenance,
    replacement_run_id,
    replacement_count,
    replacement_missing_ids,
    replaced_source_counts,
    code_file_count,
):
    summary = copy.deepcopy(old_summary)
    trajectory_count = len(standard_records)
    summary["total_trajectories"] = trajectory_count
    summary["benchmark_task_count"] = len(benchmark_records)
    summary["benchmark_trajectory_count"] = sum(
        len(record.get("trajectories", [])) for record in benchmark_records
    )
    summary["completed_prediction_count"] = len(raw_predictions)
    summary["trace_file_count"] = trajectory_count
    summary["average_turns"] = (
        sum(int(record.get("num_turns") or 0) for record in standard_records)
        / trajectory_count
        if trajectory_count
        else 0.0
    )
    summary["average_tool_calls"] = (
        sum(
            int(record.get("num_tool_calls") or 0)
            for record in standard_records
        )
        / trajectory_count
        if trajectory_count
        else 0.0
    )

    usage_totals = Counter()
    for prediction in raw_predictions.values():
        usage = prediction.get("usage") or {}
        usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        usage_totals["completion_tokens"] += int(
            usage.get("completion_tokens") or 0
        )
        usage_totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    summary["total_prompt_tokens"] = usage_totals["prompt_tokens"]
    summary["total_completion_tokens"] = usage_totals["completion_tokens"]
    summary["total_tokens"] = usage_totals["total_tokens"]

    tool_names = []
    for record in benchmark_records:
        for trajectory in record.get("trajectories", []):
            for tool in trajectory.get("metadata", {}).get("tools", []):
                if tool not in tool_names:
                    tool_names.append(tool)
    summary["tool_names"] = tool_names
    summary["merged_official_prediction_records"] = len(raw_predictions)
    summary["source_prediction_counts"] = dict(Counter(provenance.values()))
    summary["copied_trace_count"] = trajectory_count
    summary["copied_code_file_count"] = code_file_count
    summary["latest_replacement"] = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run_id": replacement_run_id,
        "replaced_trajectory_count": replacement_count,
        "replacement_missing_count": len(replacement_missing_ids),
        "replacement_missing_ids": sorted(replacement_missing_ids),
        "replaced_previous_source_counts": dict(replaced_source_counts),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-dir", type=Path, required=True)
    parser.add_argument("--replacement-run-dir", type=Path, required=True)
    parser.add_argument(
        "--source-label",
        default="retry_failed_web_browsing_20260727",
    )
    args = parser.parse_args()

    combined_dir = args.combined_dir.resolve()
    replacement_dir = args.replacement_run_dir.resolve()
    combined_run_id = combined_dir.name
    replacement_run_id = replacement_dir.name

    combined_benchmark = load_jsonl(
        combined_dir / "benchmark_trajectories.jsonl"
    )
    combined_standard = load_jsonl(combined_dir / "trajectories.jsonl")
    replacement_benchmark_all = load_jsonl(
        replacement_dir / "benchmark_trajectories.jsonl"
    )
    replacement_standard_all = load_jsonl(
        replacement_dir / "trajectories.jsonl"
    )

    replacement_predictions = load_json(
        replacement_dir / "raw/official_run/hle_gpt-5.5.json"
    )
    replacement_ids = set(replacement_predictions)
    replacement_benchmark = {
        str(record["task_id"]): normalize_benchmark_record(
            record, combined_run_id
        )
        for record in replacement_benchmark_all
        if str(record["task_id"]) in replacement_ids
    }
    replacement_standard = {
        str(record["task_id"]): normalize_standard_record(
            record, combined_run_id
        )
        for record in replacement_standard_all
        if str(record["task_id"]) in replacement_ids
    }
    if set(replacement_benchmark) != replacement_ids:
        raise ValueError("Replacement benchmark records do not match predictions")
    if set(replacement_standard) != replacement_ids:
        raise ValueError("Replacement standard records do not match predictions")

    combined_predictions_path = (
        combined_dir / "raw/official_run/hle_gpt-5.5.json"
    )
    combined_predictions = load_json(combined_predictions_path)
    provenance_path = combined_dir / "provenance_by_id.json"
    provenance = load_json(provenance_path)
    if replacement_ids - set(combined_predictions):
        raise ValueError("Replacement IDs are absent from combined predictions")
    if replacement_ids - set(provenance):
        raise ValueError("Replacement IDs are absent from provenance")

    replaced_source_counts = Counter(
        provenance[task_id] for task_id in replacement_ids
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = combined_dir / "replacement_backups" / timestamp
    backup_replaced_artifacts(
        backup_dir,
        combined_dir,
        replacement_ids,
        combined_benchmark,
        combined_standard,
        combined_predictions,
        provenance,
    )

    updated_benchmark = replace_records(
        combined_benchmark, replacement_benchmark, "benchmark JSONL"
    )
    updated_standard = replace_records(
        combined_standard, replacement_standard, "standard JSONL"
    )
    combined_predictions.update(replacement_predictions)
    for task_id in replacement_ids:
        provenance[task_id] = args.source_label

    write_jsonl_atomic(
        combined_dir / "benchmark_trajectories.jsonl", updated_benchmark
    )
    write_jsonl_atomic(combined_dir / "trajectories.jsonl", updated_standard)
    write_json(combined_predictions_path, combined_predictions)
    write_json(provenance_path, provenance)

    benchmark_dir = combined_dir / "benchmark_trajectories"
    standard_dir = combined_dir / "trajectories"
    combined_raw = combined_dir / "raw/official_run"
    replacement_raw = replacement_dir / "raw/official_run"
    for task_id in sorted(replacement_ids):
        write_json(
            benchmark_dir / f"{task_id}.json",
            replacement_benchmark[task_id],
        )
        write_json(
            standard_dir / f"{task_id}.json",
            replacement_standard[task_id],
        )
        shutil.copy2(
            replacement_raw / "traces" / f"trace_{task_id}.log",
            combined_raw / "traces" / f"trace_{task_id}.log",
        )
        for old_code in (combined_raw / "code").glob(f"{task_id}*"):
            if old_code.is_file():
                old_code.unlink()
        for new_code in (replacement_raw / "code").glob(f"{task_id}*"):
            if new_code.is_file():
                shutil.copy2(new_code, combined_raw / "code" / new_code.name)

    replacement_dataset_ids = {
        str(record["task_id"]) for record in replacement_benchmark_all
    }
    replacement_missing_ids = replacement_dataset_ids - replacement_ids
    write_json(
        combined_dir / "latest_replacement_missing_ids.json",
        sorted(replacement_missing_ids),
    )

    old_summary = load_json(combined_dir / "summary.json")
    updated_summary = recalculate_summary(
        old_summary,
        updated_benchmark,
        updated_standard,
        combined_predictions,
        provenance,
        replacement_run_id,
        len(replacement_ids),
        replacement_missing_ids,
        replaced_source_counts,
        len(list((combined_raw / "code").glob("*"))),
    )
    write_json(combined_dir / "summary.json", updated_summary)

    print(
        json.dumps(
            {
                "combined_run_id": combined_run_id,
                "replacement_run_id": replacement_run_id,
                "replaced": len(replacement_ids),
                "not_replaced": len(replacement_missing_ids),
                "total": len(updated_benchmark),
                "backup_dir": str(backup_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
