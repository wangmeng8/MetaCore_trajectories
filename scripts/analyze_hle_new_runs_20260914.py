from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis_20260914_hle_new_runs"
DATASET = ROOT / "data" / "HLE" / "WebThinker_test_500_hle_with_tools.json"

CATEGORY_ORDER = [
    "Math",
    "Physics",
    "Chemistry",
    "Biology/Medicine",
    "Computer Science/AI",
    "Engineering",
    "Humanities/Social Science",
    "Other",
]

JUDGED_PATHS = {
    "gpt_ours": ROOT / "results" / "gpt-oss" / "ours" / "analysis" / "judged_gpt56luna.json",
    "gpt_caa": ROOT / "results" / "gpt-oss" / "caa" / "analysis" / "judged_gpt56luna.json",
    "gemma_a05": ROOT / "results" / "gemma" / "caa_30_05" / "analysis" / "judged_gpt56luna.json",
    "gemma_a10": ROOT
    / "results"
    / "gemma"
    / "caa_retry"
    / "analysis"
    / "merged_judged_gpt56luna.json",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def state(value: dict[str, Any]) -> str:
    if value.get("error"):
        return "failure"
    return "correct" if value["judge_response"]["correct"] == "yes" else "incorrect"


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return (100 * (center - margin), 100 * (center + margin))


def exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = min(gains, losses)
    probability = sum(math.comb(discordant, i) for i in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


def paired_bootstrap_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    samples: int = 25_000,
) -> tuple[float, float]:
    delta = candidate.astype(float) - baseline.astype(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    batch_size = 500
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        boot[start:stop] = 100 * delta[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return (float(low), float(high))


def summarize(
    judged: dict[str, dict[str, Any]],
    dataset_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    states = Counter(state(value) for value in judged.values())
    correct = states["correct"]
    answered = states["correct"] + states["incorrect"]
    total = len(dataset_by_id)
    categories: dict[str, Any] = {}
    for category in CATEGORY_ORDER:
        ids = [qid for qid, row in dataset_by_id.items() if row["category"] == category]
        category_correct = sum(state(judged[qid]) == "correct" for qid in ids)
        categories[category] = {
            "total": len(ids),
            "correct": category_correct,
            "accuracy_percent": round(100 * category_correct / len(ids), 2),
        }
    error_types = Counter(
        str(value.get("error", {}).get("error_type"))
        for value in judged.values()
        if value.get("error", {}).get("error_type")
    )
    judge_models = Counter(
        str(value.get("judge_response", {}).get("judge_model")) for value in judged.values()
    )
    low, high = wilson_interval(correct, total)
    return {
        "total": total,
        "correct": correct,
        "accuracy_percent": round(100 * correct / total, 2),
        "wilson_95ci_percent": [round(low, 2), round(high, 2)],
        "answered_predictions": answered,
        "answered_correct": correct,
        "answered_accuracy_percent": round(100 * correct / answered, 2),
        "generation_failures": states["failure"],
        "generation_failure_rate_percent": round(100 * states["failure"] / total, 2),
        "error_types": dict(error_types),
        "judge_models": dict(judge_models),
        "categories": categories,
    }


def paired_comparison(
    name: str,
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    dataset_by_id: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ids = list(dataset_by_id)
    baseline_correct = np.array([state(baseline[qid]) == "correct" for qid in ids])
    candidate_correct = np.array([state(candidate[qid]) == "correct" for qid in ids])
    baseline_answered = np.array([state(baseline[qid]) != "failure" for qid in ids])
    candidate_answered = np.array([state(candidate[qid]) != "failure" for qid in ids])

    gains = int(np.sum(~baseline_correct & candidate_correct))
    losses = int(np.sum(baseline_correct & ~candidate_correct))
    common_mask = baseline_answered & candidate_answered
    common_gains = int(np.sum(~baseline_correct[common_mask] & candidate_correct[common_mask]))
    common_losses = int(np.sum(baseline_correct[common_mask] & ~candidate_correct[common_mask]))
    all_ci = paired_bootstrap_ci(baseline_correct, candidate_correct, seed=seed)
    common_ci = paired_bootstrap_ci(
        baseline_correct[common_mask], candidate_correct[common_mask], seed=seed + 1
    )

    transitions = Counter((state(baseline[qid]), state(candidate[qid])) for qid in ids)
    transition_rows = [
        {
            "comparison": name,
            "baseline_state": baseline_state,
            "candidate_state": candidate_state,
            "count": count,
        }
        for (baseline_state, candidate_state), count in sorted(transitions.items())
    ]
    item_rows = [
        {
            "comparison": name,
            "task_id": qid,
            "category": dataset_by_id[qid]["category"],
            "baseline_state": state(baseline[qid]),
            "candidate_state": state(candidate[qid]),
            "correctness_delta": int(state(candidate[qid]) == "correct")
            - int(state(baseline[qid]) == "correct"),
        }
        for qid in ids
    ]

    baseline_score = 100 * baseline_correct.mean()
    candidate_score = 100 * candidate_correct.mean()
    common_baseline_score = 100 * baseline_correct[common_mask].mean()
    common_candidate_score = 100 * candidate_correct[common_mask].mean()
    result = {
        "comparison": name,
        "baseline_accuracy_percent": round(baseline_score, 2),
        "candidate_accuracy_percent": round(candidate_score, 2),
        "delta_percentage_points": round(candidate_score - baseline_score, 2),
        "relative_change_percent": round(
            100 * (candidate_score - baseline_score) / baseline_score, 2
        ),
        "baseline_generation_failures": int(np.sum(~baseline_answered)),
        "candidate_generation_failures": int(np.sum(~candidate_answered)),
        "failure_delta": int(np.sum(~candidate_answered) - np.sum(~baseline_answered)),
        "gains": gains,
        "losses": losses,
        "mcnemar_exact_p": exact_mcnemar_p(gains, losses),
        "paired_bootstrap_95ci_delta_pp": [round(all_ci[0], 2), round(all_ci[1], 2)],
        "common_answered_tasks": int(np.sum(common_mask)),
        "common_baseline_accuracy_percent": round(common_baseline_score, 2),
        "common_candidate_accuracy_percent": round(common_candidate_score, 2),
        "common_delta_percentage_points": round(
            common_candidate_score - common_baseline_score, 2
        ),
        "common_gains": common_gains,
        "common_losses": common_losses,
        "common_mcnemar_exact_p": exact_mcnemar_p(common_gains, common_losses),
        "common_paired_bootstrap_95ci_delta_pp": [
            round(common_ci[0], 2),
            round(common_ci[1], 2),
        ],
        "transitions": {
            f"{baseline_state}->{candidate_state}": count
            for (baseline_state, candidate_state), count in sorted(transitions.items())
        },
    }
    return result, transition_rows, item_rows


def read_existing_table() -> dict[tuple[str, str], dict[str, str]]:
    path = ROOT / "results" / "hle_vanilla_prompt_reg_table_rows.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["model"], row["method"]): row for row in rows}


def plot_results(
    aggregate_rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    ax = axes[0]
    models = ["GPT-OSS-20B", "Gemma-4-31B-IT"]
    methods = {
        "GPT-OSS-20B": ["Vanilla", "Prompt-Reg.", "REVEAL-Base", "REVEAL (Ours)"],
        "Gemma-4-31B-IT": ["Vanilla", "Prompt-Reg.", "REVEAL-Base (a=1.0)", "REVEAL-Base (a=0.5)"],
    }
    colors = ["#a7b0ba", "#7f8c9a", "#4c78a8", "#e45756"]
    x_centers = np.arange(len(models)) * 5.2
    width = 0.82
    lookup = {(row["model"], row["method"]): row for row in aggregate_rows}
    for model_index, model in enumerate(models):
        for method_index, method in enumerate(methods[model]):
            x = x_centers[model_index] + (method_index - 1.5) * width
            value = float(lookup[(model, method)]["accuracy_percent"])
            ax.bar(x, value, width=width * 0.88, color=colors[method_index])
            ax.text(x, value + 0.45, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_centers, models)
    ax.set_ylabel("Accuracy over 500 tasks (%)")
    ax.set_title("A. Overall HLE-with-tools accuracy")
    ax.set_ylim(0, 26)
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(handles, ["Vanilla", "Prompt-Reg.", "REVEAL-Base", "New setting"], loc="upper left")

    ax = axes[1]
    settings = [
        ("GPT Base", summaries["gpt_caa"]),
        ("GPT Ours", summaries["gpt_ours"]),
        ("Gemma a=1.0", summaries["gemma_a10"]),
        ("Gemma a=0.5", summaries["gemma_a05"]),
    ]
    labels = [label for label, _ in settings]
    correct = np.array([summary["accuracy_percent"] for _, summary in settings])
    failure = np.array([summary["generation_failure_rate_percent"] for _, summary in settings])
    answered_wrong = 100 - correct - failure
    x = np.arange(len(settings))
    ax.bar(x, correct, color="#54a24b", label="Correct")
    ax.bar(x, answered_wrong, bottom=correct, color="#b9c0c8", label="Answered incorrect")
    ax.bar(x, failure, bottom=correct + answered_wrong, color="#e45756", label="Generation failure")
    for index, value in enumerate(failure):
        ax.text(index, 100 - value / 2, f"fail {value:.1f}%", ha="center", va="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Share of 500 tasks (%)")
    ax.set_title("B. Outcome composition")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")

    fig.savefig(OUT / "hle_new_runs_comparison.png", dpi=240)
    fig.savefig(OUT / "hle_new_runs_comparison.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_json(DATASET)
    dataset_by_id = {str(row["id"]): row for row in dataset}
    judged = {name: load_json(path) for name, path in JUDGED_PATHS.items()}

    expected_ids = set(dataset_by_id)
    for name, records in judged.items():
        if set(records) != expected_ids:
            raise RuntimeError(f"{name} task IDs do not match the 500-task dataset")

    summaries = {name: summarize(records, dataset_by_id) for name, records in judged.items()}
    for name in ("gpt_ours", "gemma_a05"):
        summary = summaries[name]
        if summary["judge_models"].get("gpt-5.6-luna") != summary["answered_predictions"]:
            raise RuntimeError(f"{name} has non-gpt-5.6-luna judged generated answers")
        if summary["judge_models"].get("fixed_incorrect") != summary["generation_failures"]:
            raise RuntimeError(f"{name} failure judgements are not fixed_incorrect")
        if "JudgeFailure" in summary["error_types"]:
            raise RuntimeError(f"{name} contains JudgeFailure records")

    comparisons: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for name, baseline_key, candidate_key, seed in [
        ("GPT-OSS: REVEAL (Ours) vs REVEAL-Base", "gpt_caa", "gpt_ours", 20260914),
        ("Gemma: alpha=0.5 vs alpha=1.0 (retry-merged)", "gemma_a10", "gemma_a05", 20260915),
    ]:
        comparison, transitions, items = paired_comparison(
            name,
            judged[baseline_key],
            judged[candidate_key],
            dataset_by_id,
            seed=seed,
        )
        comparisons.append(comparison)
        transition_rows.extend(transitions)
        item_rows.extend(items)

    old = read_existing_table()
    aggregate_rows: list[dict[str, Any]] = []
    for model, method in [
        ("GPT-OSS-20B", "Vanilla"),
        ("GPT-OSS-20B", "Prompt-Reg."),
        ("Gemma-4-31B-IT", "Vanilla"),
        ("Gemma-4-31B-IT", "Prompt-Reg."),
    ]:
        row = old[(model, method)]
        correct = int(row["correct"])
        total = int(row["total"])
        failures = int(row["fixed_incorrect"])
        aggregate_rows.append(
            {
                "model": model,
                "method": method,
                "correct": correct,
                "total": total,
                "accuracy_percent": float(row["All"]),
                "generation_failures": failures,
                "generation_failure_rate_percent": round(100 * failures / total, 2),
                "answered_accuracy_percent": round(100 * correct / (total - failures), 2),
            }
        )
    for model, method, key in [
        ("GPT-OSS-20B", "REVEAL-Base", "gpt_caa"),
        ("GPT-OSS-20B", "REVEAL (Ours)", "gpt_ours"),
        ("Gemma-4-31B-IT", "REVEAL-Base (a=1.0)", "gemma_a10"),
        ("Gemma-4-31B-IT", "REVEAL-Base (a=0.5)", "gemma_a05"),
    ]:
        summary = summaries[key]
        aggregate_rows.append(
            {
                "model": model,
                "method": method,
                "correct": summary["correct"],
                "total": summary["total"],
                "accuracy_percent": summary["accuracy_percent"],
                "generation_failures": summary["generation_failures"],
                "generation_failure_rate_percent": summary["generation_failure_rate_percent"],
                "answered_accuracy_percent": summary["answered_accuracy_percent"],
            }
        )

    category_rows: list[dict[str, Any]] = []
    for key, label in [
        ("gpt_caa", "GPT-OSS REVEAL-Base"),
        ("gpt_ours", "GPT-OSS REVEAL (Ours)"),
        ("gemma_a10", "Gemma REVEAL-Base alpha=1.0"),
        ("gemma_a05", "Gemma REVEAL-Base alpha=0.5"),
    ]:
        for category in CATEGORY_ORDER:
            value = summaries[key]["categories"][category]
            category_rows.append(
                {
                    "setting": label,
                    "category": category,
                    "correct": value["correct"],
                    "total": value["total"],
                    "accuracy_percent": value["accuracy_percent"],
                }
            )

    analysis = {
        "scoring": {
            "judge_model": "gpt-5.6-luna",
            "denominator": 500,
            "generation_failures_counted_incorrect": True,
            "judge_failures": 0,
        },
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "caveats": [
            "Each setting is represented by one stochastic run; confidence intervals and paired tests are descriptive.",
            "Gemma alpha=1.0 is the retry-merged result while alpha=0.5 has not been retried, so their generation-failure budgets differ.",
            "The GPT-OSS Ours vector metadata records dataset_split=false and extraction_checkpoint_verified=false; this run does not establish held-out generalization.",
            "Worker audit JSONL files were not included in the copied result directories; intervention identity is verified from each manifest.",
        ],
    }
    dump_json(OUT / "analysis.json", analysis)
    write_csv(
        OUT / "overall_results.csv",
        [
            "model",
            "method",
            "correct",
            "total",
            "accuracy_percent",
            "generation_failures",
            "generation_failure_rate_percent",
            "answered_accuracy_percent",
        ],
        aggregate_rows,
    )
    write_csv(
        OUT / "category_results.csv",
        ["setting", "category", "correct", "total", "accuracy_percent"],
        category_rows,
    )
    write_csv(
        OUT / "paired_transitions.csv",
        ["comparison", "baseline_state", "candidate_state", "count"],
        transition_rows,
    )
    write_csv(
        OUT / "paired_item_changes.csv",
        [
            "comparison",
            "task_id",
            "category",
            "baseline_state",
            "candidate_state",
            "correctness_delta",
        ],
        item_rows,
    )
    plot_results(aggregate_rows, summaries)

    gpt = summaries["gpt_ours"]
    gpt_base = summaries["gpt_caa"]
    gemma = summaries["gemma_a05"]
    gemma_base = summaries["gemma_a10"]
    gpt_cmp, gemma_cmp = comparisons
    report = f"""# HLE-with-tools 新实验评分与分析（2026-09-14）

评分模型为 `gpt-5.6-luna`。总分固定以 500 题为分母，实验生成失败按错误计；两组新结果均无 JudgeFailure。

## 总体结果

| 模型 | 设置 | 正确/500 | 总准确率 | 正常回答数 | 正常回答准确率 | 生成失败 |
|---|---|---:|---:|---:|---:|---:|
| GPT-OSS-20B | REVEAL-Base | {gpt_base['correct']}/500 | {gpt_base['accuracy_percent']:.2f}% | {gpt_base['answered_predictions']} | {gpt_base['answered_accuracy_percent']:.2f}% | {gpt_base['generation_failures']} ({gpt_base['generation_failure_rate_percent']:.1f}%) |
| GPT-OSS-20B | REVEAL (Ours) | {gpt['correct']}/500 | **{gpt['accuracy_percent']:.2f}%** | {gpt['answered_predictions']} | {gpt['answered_accuracy_percent']:.2f}% | {gpt['generation_failures']} ({gpt['generation_failure_rate_percent']:.1f}%) |
| Gemma-4-31B-IT | REVEAL-Base, alpha=1.0, retry 合并 | {gemma_base['correct']}/500 | {gemma_base['accuracy_percent']:.2f}% | {gemma_base['answered_predictions']} | {gemma_base['answered_accuracy_percent']:.2f}% | {gemma_base['generation_failures']} ({gemma_base['generation_failure_rate_percent']:.1f}%) |
| Gemma-4-31B-IT | REVEAL-Base, alpha=0.5 | {gemma['correct']}/500 | **{gemma['accuracy_percent']:.2f}%** | {gemma['answered_predictions']} | {gemma['answered_accuracy_percent']:.2f}% | {gemma['generation_failures']} ({gemma['generation_failure_rate_percent']:.1f}%) |

## GPT-OSS：Ours 向量

- Ours 为 **43/500 = 8.60%**。相对 REVEAL-Base 的 9.80% 下降 **1.20 个百分点**（相对变化 -12.24%）；相对 Prompt-Reg. 的 8.20% 提高 0.40 个百分点，相对 Vanilla 的 7.80% 提高 0.80 个百分点。
- Ours 的生成失败从 38 降至 28，覆盖率提高 2.0 个百分点，但正常回答准确率从 10.61% 降至 9.11%。因此相对 REVEAL-Base 的下降主要来自已正常完成题目的回答质量，而非生成失败。
- 在两边都正常生成的 {gpt_cmp['common_answered_tasks']} 题上，Ours 为 {gpt_cmp['common_candidate_accuracy_percent']:.2f}%，REVEAL-Base 为 {gpt_cmp['common_baseline_accuracy_percent']:.2f}%，差 {gpt_cmp['common_delta_percentage_points']:+.2f} 个百分点；正确翻转 {gpt_cmp['common_gains']} 题，错误翻转 {gpt_cmp['common_losses']} 题。McNemar 精确检验 p={gpt_cmp['common_mcnemar_exact_p']:.3f}，配对 bootstrap 95% CI 为 [{gpt_cmp['common_paired_bootstrap_95ci_delta_pp'][0]:.2f}, {gpt_cmp['common_paired_bootstrap_95ci_delta_pp'][1]:.2f}] 个百分点。
- 类别计数相对 REVEAL-Base：CS/AI +2、Engineering +1、Humanities/Social Science +1；Math -2、Physics -1、Chemistry -2、Biology/Medicine -3、Other -2。小类别波动较大，不应单独解释为稳定能力变化。
- 当前证据不支持 Ours 向量优于 REVEAL-Base。manifest 同时记录 `dataset_split=false` 和 `extraction_checkpoint_verified=false`，因此不能将该结果表述为独立留出集上的泛化提升。

## Gemma：alpha=0.5 消融

- alpha=0.5 为 **73/500 = 14.60%**；相对 retry 合并后的 alpha=1.0（15.20%）下降 **0.60 个百分点**（相对变化 -3.95%）。相对 Vanilla 22.40% 和 Prompt-Reg. 22.20% 仍分别低 7.80、7.60 个百分点。
- alpha=0.5 的正常回答准确率为 20.45%，高于 alpha=1.0 的 18.72%；但生成失败为 143，较 alpha=1.0 retry 合并结果的 94 多 49 条，失败率高 9.8 个百分点。这种 retry 预算不对称会压低 alpha=0.5 的总分。
- 在两边都正常生成的 {gemma_cmp['common_answered_tasks']} 题上，alpha=0.5 为 {gemma_cmp['common_candidate_accuracy_percent']:.2f}%，alpha=1.0 为 {gemma_cmp['common_baseline_accuracy_percent']:.2f}%，差 {gemma_cmp['common_delta_percentage_points']:+.2f} 个百分点；正确翻转 {gemma_cmp['common_gains']} 题，错误翻转 {gemma_cmp['common_losses']} 题。McNemar 精确检验 p={gemma_cmp['common_mcnemar_exact_p']:.3f}，配对 bootstrap 95% CI 为 [{gemma_cmp['common_paired_bootstrap_95ci_delta_pp'][0]:.2f}, {gemma_cmp['common_paired_bootstrap_95ci_delta_pp'][1]:.2f}] 个百分点。
- 类别计数相对 alpha=1.0：Biology/Medicine +3、Humanities/Social Science +1；Math -4、Chemistry -1、CS/AI -1、Other -1；Physics 与 Engineering 持平。
- alpha=0.5 显示出“成功生成时回答质量略高、但生成稳定性更差”的迹象。由于只有一次运行且 alpha=0.5 尚未按相同流程补跑失败题，当前不能据此确定 0.5 优于或劣于 1.0。

## 统计与复现实务

- GPT-OSS 全 500 题配对差值的 bootstrap 95% CI 为 [{gpt_cmp['paired_bootstrap_95ci_delta_pp'][0]:.2f}, {gpt_cmp['paired_bootstrap_95ci_delta_pp'][1]:.2f}] 个百分点，McNemar p={gpt_cmp['mcnemar_exact_p']:.3f}。
- Gemma 全 500 题配对差值的 bootstrap 95% CI 为 [{gemma_cmp['paired_bootstrap_95ci_delta_pp'][0]:.2f}, {gemma_cmp['paired_bootstrap_95ci_delta_pp'][1]:.2f}] 个百分点，McNemar p={gemma_cmp['mcnemar_exact_p']:.3f}。
- 这些区间都跨过 0，且每个设置只有一次随机运行；应把结论视为趋势。若要判断 alpha 效果，应先以相同 retry 规则补齐 alpha=0.5 的 143 个生成失败，再比较最终合并结果。
- 两个本地结果目录没有包含服务器端 worker audit JSONL；本次只能从 manifest 核验方法、向量哈希、层和 alpha，不能在本地复核逐请求的 `applied=true` 记录。
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "summaries": summaries, "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
