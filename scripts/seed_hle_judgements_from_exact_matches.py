"""Reuse prior judge results only for identical task-id and exact response text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(value):
    if isinstance(value, dict):
        return value.items()
    if isinstance(value, list):
        return ((str(x.get("task_id", x.get("id"))), x) for x in value)
    return ()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--search-root", type=Path, default=Path("results"))
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()

    predictions = load(args.predictions)
    output = {} if args.reset else (load(args.output) if args.output.exists() else {})
    cache = {}
    conflicts = set()
    source_files = []
    for path in args.search_root.rglob("*.json"):
        if "judged" not in path.name.lower() or path.resolve() == args.output.resolve():
            continue
        try:
            value = load(path)
        except Exception:
            continue
        used = False
        for qid, record in records(value):
            if not isinstance(record, dict):
                continue
            judge = record.get("judge_response")
            response = record.get("response")
            if not isinstance(judge, dict) or not isinstance(response, str):
                continue
            if record.get("error") or not response.strip():
                continue
            if judge.get("judge_model") != "gpt-5.6-luna":
                continue
            key = (str(qid), response)
            normalized = json.dumps(judge, ensure_ascii=False, sort_keys=True)
            if key in cache and json.dumps(cache[key], ensure_ascii=False, sort_keys=True) != normalized:
                conflicts.add(key)
            else:
                cache[key] = judge
                used = True
        if used:
            source_files.append(str(path))

    added = 0
    for qid, prediction in predictions.items():
        if not isinstance(prediction, dict):
            continue
        if prediction.get("error") or not str(prediction.get("response", "")).strip():
            continue
        if (output.get(qid, {}).get("judge_response") or {}).get("judge_model") == "gpt-5.6-luna":
            continue
        key = (str(qid), str(prediction.get("response", "")))
        if key in cache and key not in conflicts:
            value = dict(prediction)
            value["judge_response"] = cache[key]
            output[str(qid)] = value
            added += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"added": added, "cache_entries": len(cache), "conflicts": len(conflicts), "source_files": len(source_files)}))


if __name__ == "__main__":
    main()
