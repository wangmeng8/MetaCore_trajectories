from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must follow these criteria and be returned as one JSON object:

extracted_final_answer: The final exact answer extracted from [response]. Use "None" if there is no exact final answer.

[correct_answer]: {correct_answer}

reasoning: Explain only whether the extracted answer meaningfully matches [correct_answer]. Do not solve the problem again.

correct: "yes" if the answers match or are within a small margin of error for a numerical problem; otherwise "no".

confidence: The confidence percentage extracted from [response], or 100 if unavailable.

strict: true.
"""

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


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge response is not a JSON object")
    return value


def normalize_judgement(value: dict[str, Any], correct_answer: str) -> dict[str, Any]:
    correct = str(value.get("correct", "")).strip().lower()
    if correct not in {"yes", "no"}:
        raise ValueError(f"Invalid correct value: {correct!r}")
    try:
        confidence = int(round(float(value.get("confidence", 100))))
    except (TypeError, ValueError):
        confidence = 100
    return {
        "correct_answer": correct_answer,
        "model_answer": str(value.get("extracted_final_answer", "None")),
        "reasoning": str(value.get("reasoning", "")),
        "correct": correct,
        "confidence": max(0, min(100, confidence)),
        "judge_model": value.get("judge_model"),
    }


async def judge_one(
    client: AsyncOpenAI,
    *,
    judge_model: str,
    question: str,
    correct_answer: str,
    response: str,
    retries: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request: dict[str, Any] = {
                "model": judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_completion_tokens,
            }
            if attempt == 0:
                request["response_format"] = {"type": "json_object"}
            completion = await client.chat.completions.create(**request)
            message = completion.choices[0].message
            content = message.content or getattr(message, "reasoning_content", None) or ""
            if not content.strip():
                raise ValueError("Judge returned empty content")
            parsed = parse_json_object(content)
            parsed["judge_model"] = judge_model
            return normalize_judgement(parsed, correct_answer)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(min(30.0, (2**attempt) + random.random()))
    raise RuntimeError(f"Judge failed after {retries + 1} attempts: {last_error}")


def synthetic_incorrect(question: dict[str, Any], error_type: str, message: str) -> dict[str, Any]:
    return {
        "model": None,
        "response": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "termination_reason": "infrastructure_error",
        "error": {"error_type": error_type, "message": message},
        "judge_response": {
            "correct_answer": str(question["answer"]),
            "model_answer": "None",
            "reasoning": message,
            "correct": "no",
            "confidence": 100,
            "judge_model": "fixed_incorrect",
        },
    }


def build_summary(dataset: list[dict[str, Any]], judged: dict[str, Any]) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    total_correct = 0
    error_types: Counter[str] = Counter()
    for category in CATEGORY_ORDER:
        ids = [str(row["id"]) for row in dataset if row["category"] == category]
        correct = sum(judged[qid]["judge_response"]["correct"] == "yes" for qid in ids)
        total_correct += correct
        by_category[category] = {
            "total": len(ids),
            "correct": correct,
            "accuracy_percent": round(100 * correct / len(ids), 2) if ids else None,
        }
    for value in judged.values():
        error = value.get("error") or {}
        if error.get("error_type"):
            error_types[str(error["error_type"])] += 1
    n = len(dataset)
    accuracy = 100 * total_correct / n
    half_width = 1.96 * math.sqrt(accuracy * (100 - accuracy) / n)
    answered_ids = [qid for qid, value in judged.items() if not value.get("error")]
    answered_correct = sum(judged[qid]["judge_response"]["correct"] == "yes" for qid in answered_ids)
    return {
        "total": n,
        "correct": total_correct,
        "accuracy_percent": round(accuracy, 2),
        "wald_95ci_half_width_percent": round(half_width, 2),
        "answered_predictions": len(answered_ids),
        "answered_correct": answered_correct,
        "answered_accuracy_percent": round(100 * answered_correct / len(answered_ids), 2),
        "fixed_incorrect": n - len(answered_ids),
        "error_types": dict(error_types),
        "categories": by_category,
    }


async def run(args: argparse.Namespace) -> None:
    dataset = load_json(args.dataset)
    predictions = {str(key): value for key, value in load_json(args.predictions).items()}
    progress = load_json(args.progress) if args.progress and args.progress.exists() else {}
    timed_out_ids = {str(value) for value in progress.get("timed_out_ids", [])}
    judged = load_json(args.output) if args.output.exists() else {}
    judged = {str(key): value for key, value in judged.items()}

    dataset_by_id = {str(row["id"]): row for row in dataset}
    for qid, question in dataset_by_id.items():
        if qid in judged:
            continue
        if qid not in predictions:
            if qid in timed_out_ids:
                judged[qid] = synthetic_incorrect(
                    question,
                    "QuestionTimeout",
                    "Question exceeded the configured wall-clock timeout and was counted incorrect.",
                )
            else:
                judged[qid] = synthetic_incorrect(
                    question,
                    "HistoricalStuckQuestion",
                    "Question did not produce a prediction in the original run and was counted incorrect.",
                )
    atomic_json_dump(args.output, judged)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
        timeout=args.timeout,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.workers)
    write_lock = asyncio.Lock()
    completed_since_flush = 0
    failures: dict[str, str] = {}

    async def process(qid: str) -> None:
        nonlocal completed_since_flush
        question = dataset_by_id[qid]
        prediction = predictions[qid]
        async with semaphore:
            try:
                judgement = await judge_one(
                    client,
                    judge_model=args.judge_model,
                    question=str(question["question"]),
                    correct_answer=str(question["answer"]),
                    response=str(prediction.get("response", "")),
                    retries=args.retries,
                    max_completion_tokens=args.max_completion_tokens,
                )
            except Exception as exc:
                failures[qid] = str(exc)
                print(f"[{qid}] ERROR {exc}", flush=True)
                return
        value = dict(prediction)
        value["judge_response"] = judgement
        async with write_lock:
            judged[qid] = value
            completed_since_flush += 1
            if completed_since_flush >= args.flush_every:
                atomic_json_dump(args.output, judged)
                completed_since_flush = 0
            done = sum(
                qid in judged and "judge_response" in judged[qid]
                for qid in predictions
            )
            print(f"Judged {done}/{len(predictions)} answered predictions", flush=True)

    pending = [
        qid
        for qid in predictions
        if qid in dataset_by_id and "judge_response" not in judged.get(qid, {})
    ]
    print(
        f"Dataset={len(dataset)} predictions={len(predictions)} fixed_incorrect={len(dataset)-len(predictions)} "
        f"cached={len(predictions)-len(pending)} pending={len(pending)}",
        flush=True,
    )
    await asyncio.gather(*(process(qid) for qid in pending))
    atomic_json_dump(args.output, judged)
    if failures:
        failure_path = args.output.with_suffix(".failures.json")
        atomic_json_dump(failure_path, failures)
        raise RuntimeError(f"{len(failures)} judge calls failed; rerun to resume. See {failure_path}")
    summary = build_summary(dataset, judged)
    atomic_json_dump(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge HLE predictions and summarize official categories.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt-5.6-luna")
    parser.add_argument("--base-url")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--flush-every", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
