from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


EXCLUDED_IDS = {16, 18, 32, 40, 49, 53, 57, 62}

QWEN_TIMEOUT_IDS = {
    39,
    50,
    84,
    89,
    96,
    99,
    102,
    103,
    106,
    123,
    125,
    137,
    141,
    172,
    173,
    178,
    203,
    208,
    225,
    247,
    250,
    306,
    323,
    328,
    330,
    348,
    350,
    355,
    356,
    371,
    374,
    376,
    383,
    384,
    385,
    391,
    402,
    406,
    410,
    415,
    427,
    430,
    440,
    450,
    459,
    478,
    484,
    498,
}

GPT_OSS_SKIPPED_IDS = {3, 93, 94, 229, 236, 237, 366, 369, 376, 378, 380, 381}
GPT_OSS_FAILED_IDS = {115, 383}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HLE recovery subsets for incomplete runs.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/HLE/WebThinker_test_500_hle_with_tools.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/HLE/rerun"))
    return parser.parse_args()


def write_subset(
    name: str,
    ids: set[int],
    records_by_id: dict[int, dict],
    output_dir: Path,
) -> dict:
    ordered_ids = sorted(ids)
    records = [records_by_id[task_id] for task_id in ordered_ids]
    json_path = output_dir / f"{name}.json"
    parquet_path = output_dir / f"{name}.parquet"

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(records).to_parquet(parquet_path, index=False)

    return {
        "count": len(records),
        "ids": ordered_ids,
        "categories": dict(sorted(Counter(record["category"] for record in records).items())),
        "json": str(json_path),
        "parquet": str(parquet_path),
    }


def main() -> None:
    args = parse_args()
    records = json.loads(args.source.read_text(encoding="utf-8"))
    records_by_id = {int(record["id"]): record for record in records}
    if len(records_by_id) != 500:
        raise ValueError(f"Expected 500 unique source records, found {len(records_by_id)}")

    gpt_oss_unfinished_ids = GPT_OSS_SKIPPED_IDS | GPT_OSS_FAILED_IDS
    subsets = {
        "WebThinker_test_excluded8": EXCLUDED_IDS,
        "WebThinker_test_qwen_timeout48_plus_excluded8": QWEN_TIMEOUT_IDS | EXCLUDED_IDS,
        "WebThinker_test_gpt_oss_unfinished14_plus_excluded8": gpt_oss_unfinished_ids
        | EXCLUDED_IDS,
        "WebThinker_test_combined_recovery68": QWEN_TIMEOUT_IDS
        | gpt_oss_unfinished_ids
        | EXCLUDED_IDS,
    }

    missing_ids = sorted(set().union(*subsets.values()) - records_by_id.keys())
    if missing_ids:
        raise ValueError(f"IDs missing from source dataset: {missing_ids}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.source),
        "source_count": len(records_by_id),
        "reasons": {
            "excluded_from_both_original_runs": sorted(EXCLUDED_IDS),
            "qwen_timed_out": sorted(QWEN_TIMEOUT_IDS),
            "gpt_oss_timed_out": [],
            "gpt_oss_skipped_without_prediction": sorted(GPT_OSS_SKIPPED_IDS),
            "gpt_oss_failed_without_prediction": sorted(GPT_OSS_FAILED_IDS),
        },
        "subsets": {
            name: write_subset(name, ids, records_by_id, args.output_dir)
            for name, ids in subsets.items()
        },
    }
    manifest_path = args.output_dir / "recovery_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
