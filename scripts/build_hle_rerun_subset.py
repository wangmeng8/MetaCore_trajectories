from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HLE rerun subset from failed and timed-out progress IDs."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.source.read_text(encoding="utf-8"))
    progress = json.loads(args.progress.read_text(encoding="utf-8"))

    reason_ids = {
        "failed": sorted({int(value) for value in progress.get("failed_ids", [])}),
        "timed_out": sorted(
            {int(value) for value in progress.get("timed_out_ids", [])}
        ),
    }
    selected_ids = sorted(set(reason_ids["failed"]) | set(reason_ids["timed_out"]))
    records_by_id = {int(record["id"]): record for record in records}
    missing_ids = sorted(set(selected_ids) - records_by_id.keys())
    if missing_ids:
        raise ValueError(f"IDs missing from source dataset: {missing_ids}")

    selected = [records_by_id[task_id] for task_id in selected_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.name}.json"
    parquet_path = args.output_dir / f"{args.name}.parquet"
    ids_path = args.output_dir / f"{args.name}.ids.json"
    manifest_path = args.output_dir / f"{args.name}.manifest.json"
    zip_path = args.output_dir / f"{args.name}.zip"

    json_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(selected).to_parquet(parquet_path, index=False)
    ids_path.write_text(
        json.dumps(
            {"count": len(selected_ids), "ids": selected_ids, "reasons": reason_ids},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "source": str(args.source),
        "progress": str(args.progress),
        "count": len(selected),
        "ids": selected_ids,
        "reason_counts": {name: len(ids) for name, ids in reason_ids.items()},
        "categories": dict(
            sorted(Counter(record["category"] for record in selected).items())
        ),
        "files": {
            "json": json_path.name,
            "parquet": parquet_path.name,
            "ids": ids_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (json_path, parquet_path, ids_path, manifest_path):
            archive.write(path, arcname=path.name)

    print(json.dumps({**manifest, "zip": str(zip_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
