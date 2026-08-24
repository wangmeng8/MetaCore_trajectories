"""Build a local HLE parquet containing examples not yet predicted.

This is intentionally small and file-based: it reads the source HLE parquet,
filters out multimodal rows by default, loads prediction IDs from one or more
MetaCoreBench HLE-with-tools run directories, and writes the remaining rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DATA = Path("data/modelscope/cais_hle/data/test-00000-of-00001.parquet")
DEFAULT_OUTPUT = Path("data/retry/hle_remaining_after_retry.parquet")
IMAGE_COLUMNS = ("image", "image_preview", "rationale_image")


def main() -> int:
    args = parse_args()
    source_data = args.source_data
    output_path = args.output

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas/pyarrow are required to read and write parquet files.")
        print("Install them in the active environment, e.g. python -m pip install pandas pyarrow")
        return 2

    if not source_data.exists():
        print(f"ERROR: source data not found: {source_data}")
        return 2

    frame = pd.read_parquet(source_data)
    if "id" not in frame.columns:
        print("ERROR: source data must contain an 'id' column.")
        return 2

    total_source = len(frame)
    if not args.include_multimodal:
        text_mask = frame.apply(is_text_only_row, axis=1)
        frame = frame[text_mask].copy()
    total_after_text_filter = len(frame)

    completed_ids: set[str] = set()
    prediction_sources: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        ids, detail = load_completed_prediction_ids(run_dir, include_zero_token=args.include_zero_token_predictions)
        completed_ids.update(ids)
        prediction_sources.append(detail)

    id_strings = frame["id"].astype(str)
    remaining = frame[~id_strings.isin(completed_ids)].copy()

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_data": str(source_data),
        "output": str(output_path),
        "include_multimodal": args.include_multimodal,
        "include_zero_token_predictions": args.include_zero_token_predictions,
        "source_rows": total_source,
        "rows_after_text_filter": total_after_text_filter,
        "completed_prediction_ids": len(completed_ids),
        "remaining_rows": len(remaining),
        "run_dirs": [str(path) for path in args.run_dir],
        "prediction_sources": prediction_sources,
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    remaining.to_parquet(output_path, index=False)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare text-only HLE rows that are not completed in existing runs.")
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA, help="Original local HLE parquet.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Existing HLE-with-tools MetaCoreBench run directory. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output parquet for remaining rows.")
    parser.add_argument("--include-multimodal", action="store_true", help="Keep image/multimodal rows.")
    parser.add_argument(
        "--include-zero-token-predictions",
        action="store_true",
        help="Treat zero-token predictions as completed. By default they are kept for rerun.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print counts; do not write parquet.")
    return parser.parse_args()


def load_completed_prediction_ids(run_dir: Path, *, include_zero_token: bool) -> tuple[set[str], dict[str, Any]]:
    raw_dir = run_dir / "raw" / "official_run"
    paths = sorted(raw_dir.glob("hle_*.json")) + sorted(raw_dir.glob("hle_*.json.temp"))
    completed: set[str] = set()
    loaded_files: list[dict[str, Any]] = []

    for path in paths:
        if path.name.endswith(".writing"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            loaded_files.append({"path": str(path), "loaded": False, "error": str(exc)})
            continue
        if not isinstance(payload, dict):
            loaded_files.append({"path": str(path), "loaded": False, "error": "prediction payload is not an object"})
            continue

        file_ids = 0
        skipped_zero = 0
        for task_id, prediction in payload.items():
            if not isinstance(prediction, dict):
                continue
            if prediction_is_completed(prediction, include_zero_token=include_zero_token):
                completed.add(str(task_id))
                file_ids += 1
            else:
                skipped_zero += 1
        loaded_files.append(
            {
                "path": str(path),
                "loaded": True,
                "prediction_count": len(payload),
                "completed_count": file_ids,
                "zero_or_empty_skipped": skipped_zero,
            }
        )

    return completed, {"run_dir": str(run_dir), "files": loaded_files, "completed_ids": len(completed)}


def prediction_is_completed(prediction: dict[str, Any], *, include_zero_token: bool) -> bool:
    response = prediction.get("response")
    if isinstance(response, str) and response.strip():
        return True
    usage = prediction.get("usage")
    if not isinstance(usage, dict):
        return False
    total_tokens = usage.get("total_tokens")
    if isinstance(total_tokens, int):
        return include_zero_token or total_tokens > 0
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    token_values = [value for value in (prompt_tokens, completion_tokens) if isinstance(value, int)]
    if not token_values:
        return False
    return include_zero_token or sum(token_values) > 0


def is_text_only_row(row: Any) -> bool:
    for column in IMAGE_COLUMNS:
        if column not in row:
            continue
        if has_content(row[column]):
            return False
    return True


def has_content(value: Any) -> bool:
    if value is None:
        return False
    try:
        import pandas as pd

        if pd.isna(value):
            return False
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value) > 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


if __name__ == "__main__":
    raise SystemExit(main())
