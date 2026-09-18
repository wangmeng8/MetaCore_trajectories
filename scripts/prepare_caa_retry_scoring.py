"""Merge CAA recovery runs without selecting answers by judge outcome."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from score_hle_predictions import atomic_json_dump, synthetic_incorrect


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    dataset = read(ROOT / "data/HLE/WebThinker_test_500_hle_with_tools.json")
    questions = {str(row["id"]): row for row in dataset}
    for model in ("gemma", "qwen"):
        base = ROOT / "results" / model
        original_path = next((base / "caa/raw/official_run").glob("hle_*.json"))
        retry_path = next((base / "caa_retry/raw/official_run").glob("hle_*.json"))
        original, retry = read(original_path), read(retry_path)
        assert set(original) == set(questions)
        assert set(retry) <= set(original)
        old_manifest = read(base / "caa/manifest.json")
        new_manifest = read(base / "caa_retry/manifest.json")
        for field in ("model_key", "layer", "alpha", "vector_sha256", "tensor_key"):
            assert old_manifest["outcome_caa"][field] == new_manifest["outcome_caa"][field], field
        previous = read(base / "caa/analysis/judged_gpt56luna.json")
        output_dir = base / "caa_retry/analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        judged_path = output_dir / "merged_judged_gpt56luna.json"
        cached = read(judged_path) if judged_path.exists() else {}
        merged, judged, origins = {}, {}, {}
        for qid, first in original.items():
            # Recovery is for generation failures; judge failures retain the original answer.
            use_retry = bool(first.get("error")) and qid in retry
            value = retry[qid] if use_retry else first
            merged[qid] = value
            origins[qid] = "caa_retry" if use_retry else "caa"
            if value.get("error"):
                error = value["error"]
                fixed = synthetic_incorrect(questions[qid], error["error_type"], error.get("message", "Generation failed"))
                judged[qid] = {**value, "judge_response": fixed["judge_response"]}
                continue
            for candidate in (cached.get(qid, {}), previous.get(qid, {}) if not use_retry else {}):
                jr = candidate.get("judge_response", {})
                if (not candidate.get("error") and candidate.get("response") == value.get("response")
                        and jr.get("judge_model") == "gpt-5.6-luna" and jr.get("correct") in ("yes", "no")):
                    judged[qid] = {**value, "judge_response": jr}
                    break
        atomic_json_dump(output_dir / "merged_predictions.json", merged)
        atomic_json_dump(judged_path, judged)
        manifest = {
            "model": model,
            "policy": "Keep original non-error predictions; replace original generation errors with retry records. Never select by correctness.",
            "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (original_path, retry_path)},
            "origins": origins,
            "total": len(merged),
            "retry_records": len(retry),
            "used_retry": sum(v == "caa_retry" for v in origins.values()),
            "generation_errors": dict(Counter(v["error"]["error_type"] for v in merged.values() if v.get("error"))),
            "pending_judge_ids": sorted(set(merged) - set(judged), key=int),
        }
        atomic_json_dump(output_dir / "merge_manifest.json", manifest)
        print(model, "used_retry", manifest["used_retry"], "errors", manifest["generation_errors"], "pending", len(manifest["pending_judge_ids"]), flush=True)


if __name__ == "__main__":
    main()
