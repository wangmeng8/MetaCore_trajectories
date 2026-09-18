"""Build a self-contained retry package from one HLE run's generation failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.run_dir / "raw/official_run"
    prediction_files = list(raw_dir.glob("hle_*.json"))
    if len(prediction_files) != 1:
        raise ValueError(f"Expected one prediction JSON, found {len(prediction_files)}")
    prediction_path = prediction_files[0]
    progress_path = raw_dir / "progress_state.json"
    manifest_path = args.run_dir / "manifest.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    source = pd.read_parquet(args.source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["outcome_caa"]

    ids = sorted({int(value) for value in progress["failed_ids"] + progress["timed_out_ids"]})
    errors = {}
    for task_id in ids:
        record = predictions.get(str(task_id))
        if record is None or not record.get("error") or str(record.get("response", "")).strip():
            raise ValueError(f"Task {task_id} is not an empty-response generation failure")
        errors[str(task_id)] = record["error"]
    subset = source[source["id"].astype(int).isin(ids)].copy().sort_values("id")
    if subset["id"].astype(int).tolist() != ids:
        raise ValueError("Dataset subset does not exactly match failure IDs")

    package_dir = args.out_dir / args.name
    package_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = package_dir / f"{args.name}.parquet"
    json_path = package_dir / f"{args.name}.json"
    ids_path = package_dir / f"{args.name}.ids.json"
    package_manifest = package_dir / "manifest.json"
    readme_path = package_dir / "README.md"
    sums_path = package_dir / "SHA256SUMS.txt"

    subset.to_parquet(parquet_path, index=False)
    dump(json_path, json.loads(subset.to_json(orient="records", force_ascii=False)))
    dump(ids_path, {"count": len(ids), "ids": [str(value) for value in ids], "errors": errors})
    dump(
        package_manifest,
        {
            "count": len(ids),
            "source": str(args.source),
            "source_sha256": digest(args.source),
            "source_run_id": manifest["run_id"],
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": digest(manifest_path),
            "source_predictions": str(prediction_path),
            "source_predictions_sha256": digest(prediction_path),
            "reason_counts": dict(sorted(Counter(e["error_type"] for e in errors.values()).items())),
            "outcome_caa": {
                "model": identity["model"],
                "model_key": identity["model_key"],
                "layer": identity["layer"],
                "alpha": identity["alpha"],
                "tensor_key": identity["tensor_key"],
                "vector_sha256": identity["vector_sha256"],
            },
            "selection_policy": (
                "Only empty-response generation failures from the source run. "
                "Normally generated wrong answers and judge-call failures are excluded."
            ),
        },
    )
    readme_path.write_text(
        f"# {args.name}: {len(ids)} generation failures\n\n"
        f"Source run: `{manifest['run_id']}`. Use the same model and CAA identity: "
        f"layer {identity['layer']}, alpha {identity['alpha']}, vector SHA256 `{identity['vector_sha256']}`.\n\n"
        "After restoring the rest of the model-specific environment:\n\n"
        "```bash\n"
        f"export HLE_DATA_PATH=\"$PWD/{args.name}.parquet\"\n"
        f"export OUTCOME_CAA_VECTOR_LAYER={identity['layer']}\n"
        f"export OUTCOME_CAA_ALPHA={identity['alpha']}\n"
        f"export HLE_RUN_ID={args.name}\n"
        f"export OUTPUT_DIR=\"$PWD/outputs/{args.name}\"\n"
        "bash scripts/run_hle_outcome_caa.sh\n"
        "```\n\n"
        "Use a new output directory and return the complete run outputs for ID-based merging.\n",
        encoding="utf-8",
    )
    members = [package_manifest, ids_path, json_path, parquet_path, readme_path]
    sums_path.write_text("".join(f"{digest(path)}  {path.name}\n" for path in members), encoding="utf-8")
    members.append(sums_path)
    zip_path = args.out_dir / f"{args.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            archive.write(path, path.name)
    print(zip_path)


if __name__ == "__main__":
    main()
