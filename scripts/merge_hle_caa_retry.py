"""Validate and merge an HLE CAA recovery run into its original 500-task run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_hle_predictions import build_summary


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prediction_file(run_dir: Path) -> Path:
    files = list((run_dir / "raw/official_run").glob("hle_*.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one prediction JSON in {run_dir}, found {len(files)}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--base-judged", type=Path)
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--base-summary", type=Path)
    args = parser.parse_args()

    original_manifest = load(args.original / "manifest.json")
    retry_manifest = load(args.retry / "manifest.json")
    original_identity = original_manifest["outcome_caa"]
    retry_identity = retry_manifest["outcome_caa"]
    identity_fields = ("model_key", "vector_sha256", "tensor_key", "layer", "alpha")
    mismatches = {
        key: (original_identity.get(key), retry_identity.get(key))
        for key in identity_fields
        if original_identity.get(key) != retry_identity.get(key)
    }
    if mismatches:
        raise ValueError(f"CAA identity mismatch: {mismatches}")

    base_judged_path = args.base_judged or args.original / "analysis/judged_gpt56luna.json"
    base_predictions_path = args.base_predictions or prediction_file(args.original)
    base_summary_path = args.base_summary or args.original / "analysis/summary_gpt56luna.json"
    original_judged = load(base_judged_path)
    retry_judged = load(args.retry / "analysis/judged_gpt56luna.json")
    original_failure_ids = {
        task_id
        for task_id, value in original_judged.items()
        if (value.get("judge_response") or {}).get("judge_model") == "fixed_incorrect"
    }
    if set(retry_judged) != original_failure_ids:
        raise ValueError(
            f"Retry IDs differ from original failures: retry={len(retry_judged)}, "
            f"original_failures={len(original_failure_ids)}"
        )

    original_predictions = load(base_predictions_path)
    retry_predictions = load(prediction_file(args.retry))
    merged_predictions = dict(original_predictions)
    merged_predictions.update(retry_predictions)
    merged_judged = dict(original_judged)
    merged_judged.update(retry_judged)
    dataset = load(args.dataset)
    merged_summary = build_summary(dataset, merged_judged)
    retry_summary = load(args.retry / "analysis/summary_gpt56luna.json")
    original_summary = load(base_summary_path)

    out = args.retry / "analysis"
    dump(out / "merged_predictions.json", merged_predictions)
    dump(out / "merged_judged_gpt56luna.json", merged_judged)
    dump(out / "merged_summary_gpt56luna.json", merged_summary)
    dump(
        out / "merge_manifest.json",
        {
            "original_run_id": original_manifest["run_id"],
            "retry_run_id": retry_manifest["run_id"],
            "base_judged": str(base_judged_path),
            "base_predictions": str(base_predictions_path),
            "base_summary": str(base_summary_path),
            "replaced_task_count": len(retry_judged),
            "identity": {key: original_identity[key] for key in identity_fields},
            "policy": "Replace exactly the original generation-failure task IDs with the validated retry results.",
        },
    )

    delta_correct = merged_summary["correct"] - original_summary["correct"]
    delta_accuracy = merged_summary["accuracy_percent"] - original_summary["accuracy_percent"]
    report = f"""# HLE CAA recovery result

CAA identity validated: layer **{original_identity['layer']}**, alpha **{original_identity['alpha']}**, vector SHA256 `{original_identity['vector_sha256']}`.

| Result | Correct | Accuracy | Generated | Generation failures | Accuracy on generated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base before this recovery | {original_summary['correct']}/500 | {original_summary['accuracy_percent']:.2f}% | {original_summary['answered_predictions']} | {original_summary['fixed_incorrect']} | {original_summary['answered_accuracy_percent']:.2f}% |
| Retry subset | {retry_summary['correct']}/{retry_summary['total']} | {retry_summary['accuracy_percent']:.2f}% | {retry_summary['answered_predictions']} | {retry_summary['fixed_incorrect']} | {retry_summary['answered_accuracy_percent']:.2f}% |
| Merged | {merged_summary['correct']}/500 | **{merged_summary['accuracy_percent']:.2f}%** | {merged_summary['answered_predictions']} | {merged_summary['fixed_incorrect']} | {merged_summary['answered_accuracy_percent']:.2f}% |

The recovery adds {delta_correct} correct answers and changes full-set accuracy by {delta_accuracy:+.2f} percentage points. Only the original generation-failure IDs were replaced; all originally generated answers were retained.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(merged_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
