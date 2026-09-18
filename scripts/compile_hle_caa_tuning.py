"""Compile Qwen and Gemma Outcome-CAA parameter sweeps into CSV/Markdown."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "caa_tuning_20260916"

RUNS = [
    ("Qwen3.6-27B", "qwen", "caa", "raw"),
    ("Qwen3.6-27B", "qwen", "caa_retry", "recovered"),
    ("Qwen3.6-27B", "qwen", "qwen_caa_40_03", "raw"),
    ("Qwen3.6-27B", "qwen", "qwen_caa_40_05", "raw"),
    ("Qwen3.6-27B", "qwen", "qwen_caa_l40_a05_retry185_d1", "recovered_run"),
    ("Qwen3.6-27B", "qwen", "qwen_caa_42_03", "raw"),
    ("Gemma-4-31B-IT", "gemma", "caa", "raw"),
    ("Gemma-4-31B-IT", "gemma", "caa_retry", "recovered"),
    ("Gemma-4-31B-IT", "gemma", "caa_30_05", "raw"),
    ("Gemma-4-31B-IT", "gemma", "hle_gemma_caa_full_30_03", "raw"),
    ("Gemma-4-31B-IT", "gemma", "hle_gemma_caa_full_37_03", "raw"),
    ("Gemma-4-31B-IT", "gemma", "hle_gemma_caa_l37_a03_retry54_d3", "recovered_run"),
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def value_after(args, key):
    try:
        return args[args.index(key) + 1]
    except (ValueError, IndexError):
        return None


def raw_resolved_summary(model_dir: Path):
    progress = load(model_dir / "caa/raw/official_run/progress_state.json")
    failures = set(map(str, progress["failed_ids"] + progress["timed_out_ids"]))
    judged = load(model_dir / "caa_retry/analysis/merged_judged_gpt56luna.json")
    correct = sum(
        qid not in failures and (record.get("judge_response") or {}).get("correct") == "yes"
        for qid, record in judged.items()
    )
    answered = 500 - len(failures)
    return {
        "total": 500,
        "correct": correct,
        "accuracy_percent": round(correct / 5, 2),
        "answered_predictions": answered,
        "answered_correct": correct,
        "answered_accuracy_percent": round(100 * correct / answered, 2),
        "fixed_incorrect": len(failures),
    }


def run_row(model_name, model_key, dirname, variant):
    model_dir = ROOT / "results" / model_key
    run_dir = model_dir / dirname
    if variant == "recovered_run":
        manifest = load(run_dir / "manifest.json")
        progress = load(run_dir / "raw/official_run/progress_state.json")
        summary_path = run_dir / "analysis/merged_summary_gpt56luna.json"
        summary = load(summary_path)
        generation_failed = summary["fixed_incorrect"]
        generated = summary["answered_predictions"]
        run_id = manifest["run_id"]
    elif variant == "recovered":
        manifest = load(model_dir / "caa/manifest.json")
        progress = load(model_dir / "caa_retry/raw/official_run/progress_state.json")
        summary_path = model_dir / "caa_retry/analysis/merged_summary_gpt56luna.json"
        summary = load(summary_path)
        generation_failed = summary["fixed_incorrect"]
        generated = summary["answered_predictions"]
        run_id = load(run_dir / "manifest.json")["run_id"]
    else:
        manifest = load(run_dir / "manifest.json")
        progress = load(run_dir / "raw/official_run/progress_state.json")
        if dirname == "caa":
            summary = raw_resolved_summary(model_dir)
            summary_path = model_dir / "caa_retry/analysis/merged_judged_gpt56luna.json"
        else:
            summary_path = run_dir / "analysis/summary_gpt56luna.json"
            summary = load(summary_path) if summary_path.exists() else None
        generation_failed = int(progress.get("failed", 0)) + int(progress.get("timed_out", 0))
        generated = int(progress.get("completed", 0))
        run_id = manifest["run_id"]
    identity = manifest["outcome_caa"]
    config = manifest["config"]
    service = identity.get("service", {})
    service_args = service.get("arguments", [])
    return {
        "model": model_name,
        "variant": "recovered" if variant.startswith("recovered") else variant,
        "layer": identity["layer"],
        "alpha": identity["alpha"],
        "run_id": run_id,
        "generated": generated,
        "generation_failed": generation_failed,
        "generation_failure_percent": round(100 * generation_failed / 500, 2),
        "correct": summary.get("correct") if summary else None,
        "accuracy_percent": summary.get("accuracy_percent") if summary else None,
        "answered_accuracy_percent": summary.get("answered_accuracy_percent") if summary else None,
        "score_status": "complete" if summary else "pending",
        "hle_workers": config.get("max_workers"),
        "vllm_max_num_seqs": value_after(service_args, "--max-num-seqs"),
        "api_timeout_seconds": config.get("api_timeout_seconds"),
        "question_timeout_seconds": config.get("question_timeout_seconds"),
        "max_retries": config.get("max_retries"),
        "vector_sha256": identity["vector_sha256"],
        "summary_path": str(summary_path.relative_to(ROOT)),
    }


def fmt(value):
    return "pending" if value is None else str(value)


def main():
    rows = [run_row(*spec) for spec in RUNS]
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "caa_parameter_sweep.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# HLE with Tools — Outcome-CAA parameter sweep",
        "",
        "Judge: `gpt-5.6-luna`; denominator: 500 questions. Generation failures count as incorrect.",
        "",
        "| Model | Layer | Alpha | Variant | Generated | Gen. failures | Failure rate | Correct | Accuracy | Accuracy on answered | HLE workers | Manifest max seqs |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['layer']} | {row['alpha']} | {row['variant']} | "
            f"{row['generated']} | {row['generation_failed']} | {row['generation_failure_percent']:.2f}% | "
            f"{fmt(row['correct'])} | {fmt(row['accuracy_percent'])}{'%' if row['accuracy_percent'] is not None else ''} | "
            f"{fmt(row['answered_accuracy_percent'])}{'%' if row['answered_accuracy_percent'] is not None else ''} | "
            f"{row['hle_workers']} | {row['vllm_max_num_seqs']} |"
        )
    lines += [
        "",
        "`raw` is the original 500-question run. `recovered` replaces only generation failures using the recorded retry run.",
        "The Qwen `caa_05` directory is excluded because its manifest and progress are an exact duplicate of `qwen/caa` (layer 23, alpha 1.0), rather than an alpha-0.5 run.",
        "Runs with different HLE concurrency, API timeout, question timeout, or retry counts are not controlled layer/alpha ablations.",
        "For the Gemma layer-37 runs, the stored service manifest records `--max-num-seqs 8`, while the operator confirmed the live vLLM service used 32. The table reports the manifest value; this metadata mismatch should be corrected before publication.",
        "",
        "## Findings",
        "",
        "- **Qwen:** after one recovery round, the best observed full-set score is layer 40, alpha 0.5 at 18.8%. This is +3.0 percentage points over the recovered layer-23, alpha-1.0 result (15.8%). Its answered-only accuracy is also slightly higher (26.26% vs. 26.07%), while its generation-failure rate is 11.0 points lower (28.4% vs. 39.4%). The gain is driven mainly by the higher completion rate.",
        "- **Qwen alpha 0.3:** layer 42 scores 12.8%, versus 10.6% at layer 40. The 2.2-point gain coincides with a much lower failure rate (42.2% vs. 55.8%); answered-only accuracy actually falls from 23.98% to 22.15%.",
        "- **Gemma:** after three recovery rounds, the best observed score is layer 37, alpha 0.3 at 21.8%. This is +5.0 points over layer 30, alpha 0.3 (16.8%) and +6.6 points over the recovered layer-30, alpha-1.0 result (15.2%). The recovery rounds add 15 correct answers and reduce generation failures from 115 to 37.",
        "- **Gemma layer 30:** raw full-set accuracy increases as alpha decreases: 13.6% at alpha 1.0, 14.6% at alpha 0.5, and 16.8% at alpha 0.3. Both answered-only accuracy and completion rate improve, but the alpha-0.3 run used different client concurrency and retry settings.",
        "",
        "These are single runs without uncertainty-controlled replication. Because HLE worker count, retry count, and timeouts differ across several settings, the observed differences cannot be attributed solely to layer or alpha.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "caa_parameter_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(csv_path)


if __name__ == "__main__":
    main()
