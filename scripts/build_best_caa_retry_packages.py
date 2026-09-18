import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/HLE/WebThinker_test_500_hle_with_tools.parquet"
OUT = ROOT / "results/caa_tuning_20260916/rerun"

RUNS = [
    {
        "name": "qwen_caa_l40_a05_retry185",
        "model": "qwen",
        "run_dir": ROOT / "results/qwen/qwen_caa_40_05",
        "prediction": "hle_qwen3.6-27b.json",
        "expected_count": 185,
    },
    {
        "name": "gemma_caa_l37_a03_retry115",
        "model": "gemma",
        "run_dir": ROOT / "results/gemma/hle_gemma_caa_full_37_03",
        "prediction": "hle_gemma-4-31b.json",
        "expected_count": 115,
    },
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_package(spec: dict, source: pd.DataFrame) -> Path:
    name = spec["name"]
    run_dir = spec["run_dir"]
    raw_dir = run_dir / "raw/official_run"
    progress = json.loads((raw_dir / "progress_state.json").read_text(encoding="utf-8"))
    predictions_path = raw_dir / spec["prediction"]
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    original_manifest_path = run_dir / "manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))

    ids = sorted({int(value) for value in progress["failed_ids"] + progress["timed_out_ids"]})
    if len(ids) != spec["expected_count"]:
        raise ValueError(f"{name}: expected {spec['expected_count']} failures, found {len(ids)}")
    missing = sorted(set(ids) - set(source["id"].astype(int)))
    if missing:
        raise ValueError(f"{name}: source dataset is missing IDs {missing}")

    subset = source[source["id"].astype(int).isin(ids)].copy().sort_values("id")
    if subset["id"].astype(int).tolist() != ids:
        raise ValueError(f"{name}: subset ID/order mismatch")

    errors = {}
    for task_id in ids:
        record = predictions[str(task_id)]
        error = record.get("error")
        if not error:
            raise ValueError(f"{name}: task {task_id} has no generation error")
        errors[str(task_id)] = error
    reason_counts = dict(sorted(Counter(item["error_type"] for item in errors.values()).items()))

    package_dir = OUT / name
    package_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = package_dir / f"{name}.parquet"
    json_path = package_dir / f"{name}.json"
    ids_path = package_dir / f"{name}.ids.json"
    manifest_path = package_dir / "manifest.json"
    readme_path = package_dir / "README.md"

    subset.to_parquet(parquet_path, index=False)
    records = json.loads(subset.to_json(orient="records", force_ascii=False))
    dump_json(json_path, records)
    dump_json(ids_path, {"count": len(ids), "ids": [str(value) for value in ids], "errors": errors})

    identity = original_manifest["outcome_caa"]
    dump_json(
        manifest_path,
        {
            "model": spec["model"],
            "count": len(ids),
            "source": str(SOURCE.relative_to(ROOT)),
            "source_sha256": digest(SOURCE),
            "original_run_id": original_manifest["run_id"],
            "original_manifest": str(original_manifest_path.relative_to(ROOT)),
            "original_manifest_sha256": digest(original_manifest_path),
            "original_predictions": str(predictions_path.relative_to(ROOT)),
            "original_predictions_sha256": digest(predictions_path),
            "reason_counts": reason_counts,
            "outcome_caa": {
                "model": identity["model"],
                "model_key": identity["model_key"],
                "layer": identity["layer"],
                "alpha": identity["alpha"],
                "tensor_key": identity["tensor_key"],
                "vector_sha256": identity["vector_sha256"],
            },
            "selection_policy": (
                "Only generation failures from the highest-full-accuracy CAA run for this model. "
                "Normally generated wrong answers and judge-call failures are excluded. Original task IDs are preserved."
            ),
        },
    )

    readme_path.write_text(
        f"# {spec['model']} best CAA recovery: {len(ids)} questions\n\n"
        f"Original run: `{original_manifest['run_id']}`. Exact CAA identity: layer "
        f"{identity['layer']}, alpha {identity['alpha']}, vector SHA256 `{identity['vector_sha256']}`.\n\n"
        f"Use `{name}.parquet` as `HLE_DATA_PATH` with the same model, vector, layer, alpha, and service manifest. "
        "Use a new `HLE_RUN_ID` and output directory; do not overwrite the original run.\n\n"
        "Example after restoring the rest of the model-specific environment:\n\n"
        "```bash\n"
        f"export HLE_DATA_PATH=\"$PWD/{name}.parquet\"\n"
        f"export OUTCOME_CAA_VECTOR_LAYER={identity['layer']}\n"
        f"export OUTCOME_CAA_ALPHA={identity['alpha']}\n"
        f"export HLE_RUN_ID={name}\n"
        f"export OUTPUT_DIR=\"$PWD/outputs/{name}\"\n"
        "bash scripts/run_hle_outcome_caa.sh\n"
        "```\n\n"
        "Return the complete run outputs for ID-based merging and grading.\n",
        encoding="utf-8",
    )

    members = [manifest_path, ids_path, json_path, parquet_path, readme_path]
    sums_path = package_dir / "SHA256SUMS.txt"
    sums_path.write_text("".join(f"{digest(path)}  {path.name}\n" for path in members), encoding="utf-8")
    members.append(sums_path)

    zip_path = OUT / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            archive.write(path, path.name)
    return zip_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(SOURCE)
    packages = [build_package(spec, source) for spec in RUNS]

    combined_readme = OUT / "README.md"
    combined_readme.write_text(
        "# Best CAA recovery packages\n\n"
        "- Qwen3.6-27B: layer 40, alpha 0.5; 185 generation failures.\n"
        "- Gemma-4-31B-IT: layer 37, alpha 0.3; 115 generation failures.\n\n"
        "Each nested ZIP is self-contained. Extract and follow its README.\n",
        encoding="utf-8",
    )
    combined_sums = OUT / "SHA256SUMS.txt"
    combined_sums.write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in packages + [combined_readme]),
        encoding="utf-8",
    )
    combined = OUT / "qwen_gemma_best_caa_retry_300.zip"
    with zipfile.ZipFile(combined, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in packages + [combined_readme, combined_sums]:
            archive.write(path, path.name)
    print(*(str(path) for path in packages), str(combined), sep="\n")


if __name__ == "__main__":
    main()
