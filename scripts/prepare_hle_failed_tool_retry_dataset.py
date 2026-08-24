"""Build an HLE retry parquet from failed tool results in benchmark trajectories."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TRAJECTORIES = Path(
    "outputs/runs/hle_official_combined_gpt55_1873/benchmark_trajectories.jsonl"
)
DEFAULT_SOURCE_DATA = Path("data/modelscope/cais_hle/data/test-00000-of-00001.parquet")
DEFAULT_OUTPUT = Path("data/retry/hle_failed_search_tools_retry.parquet")
DEFAULT_TOOLS = ("web_browsing", "scientific_search")
FINAL_ATTEMPT_MARKER = "QUESTION ID"
FAILED_RESULT_MARKERS = ("\u274c", "\u2717")


def main() -> int:
    args = parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas/pyarrow are required to read and write parquet files.")
        return 2

    for path in (args.trajectories, args.source_data):
        if not path.exists():
            print(f"ERROR: input not found: {path}")
            return 2

    tools = tuple(dict.fromkeys(args.tool or DEFAULT_TOOLS))
    failed_ids, analyzed_count = collect_failed_task_ids(args.trajectories, tools)
    selected_ids = set().union(*(failed_ids[tool] for tool in tools))

    frame = pd.read_parquet(args.source_data)
    if "id" not in frame.columns:
        print("ERROR: source data must contain an 'id' column.")
        return 2

    source_ids = frame["id"].astype(str)
    selected = frame[source_ids.isin(selected_ids)].copy()
    found_ids = set(selected["id"].astype(str))
    missing_ids = sorted(selected_ids - found_ids)

    overlap_count = 0
    if len(tools) == 2:
        overlap_count = len(failed_ids[tools[0]] & failed_ids[tools[1]])

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "trajectories": str(args.trajectories),
        "source_data": str(args.source_data),
        "output": str(args.output),
        "final_attempt_marker": FINAL_ATTEMPT_MARKER,
        "tools": list(tools),
        "trajectories_analyzed": analyzed_count,
        "failed_trajectory_count_by_tool": {
            tool: len(failed_ids[tool]) for tool in tools
        },
        "two_tool_overlap_count": overlap_count,
        "unique_failed_task_ids": len(selected_ids),
        "selected_source_rows": len(selected),
        "missing_source_ids": missing_ids,
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    args.output.with_suffix(args.output.suffix + ".ids.json").write_text(
        json.dumps(sorted(selected_ids), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not missing_ids else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select HLE rows whose final trajectory attempt has failed tool results."
    )
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tool",
        action="append",
        help="Tool to select. Repeat for multiple tools; defaults to web and scientific search.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_failed_task_ids(
    trajectories_path: Path, tools: tuple[str, ...]
) -> tuple[dict[str, set[str]], int]:
    failed_ids: dict[str, set[str]] = defaultdict(set)
    analyzed_count = 0

    with trajectories_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {trajectories_path}:{line_number}: {exc}"
                ) from exc

            task_id = str(record.get("task_id") or "")
            trajectory_text = extract_trajectory_text(record)
            final_attempt = extract_final_attempt(trajectory_text)
            for tool in tools:
                if has_failed_tool_result(final_attempt, tool):
                    failed_ids[tool].add(task_id)
            analyzed_count += 1

    return failed_ids, analyzed_count


def extract_trajectory_text(record: dict[str, Any]) -> str:
    trajectories = record.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        return ""
    trajectory = trajectories[0]
    if not isinstance(trajectory, dict):
        return ""
    value = trajectory.get("trajectory_text")
    return value if isinstance(value, str) else ""


def extract_final_attempt(trajectory_text: str) -> str:
    marker_position = trajectory_text.rfind(FINAL_ATTEMPT_MARKER)
    return trajectory_text[marker_position:] if marker_position >= 0 else trajectory_text


def has_failed_tool_result(final_attempt: str, tool: str) -> bool:
    tool_pattern = re.escape(tool)
    for line in final_attempt.splitlines():
        if f"TOOL RESULT {tool}" not in line:
            continue
        if any(marker in line for marker in FAILED_RESULT_MARKERS):
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
