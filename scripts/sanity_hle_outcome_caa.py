"""Explicit REMOTE API sanity test. Never invoked by prepare/import/unit tests."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.hle_outcome_caa import without_steering
from scripts.prepare_hle_outcome_caa import request_body


def check_audit(records, response_id, alpha, tp):
    events = [e for e in records if e.get("event") == "last_prefix" and
              (e["request_id"] == response_id or e["request_id"].startswith(response_id + "-"))]
    if alpha is None:
        if events:
            raise ValueError("Unconfigured request was steered")
        return
    if len(events) != tp or len({e["rank"] for e in events}) != tp:
        raise ValueError(f"Expected one final-prefix event per TP rank for {response_id}; got {events}")
    if any(e["alpha"] != alpha or e["token_position"] != e["prefix_length"] - 1 or e["applied"] != (alpha != 0) for e in events):
        raise ValueError("Wrong coefficient/position in worker audit")


async def run(args):
    from openai import AsyncOpenAI
    service = json.loads(args.service_manifest.read_text(encoding="utf-8"))
    if args.model != service["served_model_name"]:
        raise ValueError("Model differs from service manifest")
    messages = json.loads(args.messages_file.read_text(encoding="utf-8")) if args.messages_file else [
        {"role": "user", "content": "Reply with only the result of 17 + 25."}]
    body = request_body(args.vector, args.layer, 1., os.environ)
    if json.loads(body["vllm_xargs"]["steer"])["vector_sha256"] != service["vector_sha256"]:
        raise ValueError("Vector differs from service manifest")
    plain = without_steering(body)
    zero = request_body(args.vector, args.layer, 0., os.environ)
    results = []
    async with AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), max_retries=0, timeout=args.timeout) as client:
        async def request(label, extra, alpha):
            response = await client.chat.completions.create(model=args.model, messages=messages,
                temperature=0, seed=1729, max_completion_tokens=args.max_tokens, extra_body=extra)
            record = {"label": label, "alpha": alpha, "response_id": response.id,
                      "message": response.choices[0].message.model_dump(),
                      "finish_reason": response.choices[0].finish_reason}
            results.append(record)
            return record
        a = await request("plain", plain, None)
        b = await request("alpha_zero", zero, 0.)
        if a["message"] != b["message"] or a["finish_reason"] != b["finish_reason"]:
            raise ValueError("Deterministic plain and alpha=0 responses differ")
        await asyncio.gather(request("mixed_plain", plain, None), request("mixed_zero", zero, 0.), request("mixed_nonzero", body, 1.))
    events = []
    for path in Path(service["audit_dir"]).glob("worker-*.jsonl"):
        events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    for record in results:
        check_audit(events, record["response_id"], record["alpha"], service["tensor_parallel_size"])
    return {"passed": True, "service": service, "results": results,
            "scope": "plain/zero deterministic equality; concurrent request isolation; per-rank last-prefix audit",
            "remaining": "Run fixed HLE questions through the original tool loop, inspect tool-feedback/final-answer requests, then run full benchmark."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--vector", type=Path, required=True)
    p.add_argument("--service-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--messages-file", type=Path, help="Fixed JSON chat message array; use a long input to exercise chunked prefill")
    p.add_argument("--layer", type=int)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=300.)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new --output path; sanity evidence is not overwritten")
    from scripts.runner_common import load_dotenv_file
    load_dotenv_file()
    try:
        result = asyncio.run(run(args))
        code = 0
    except Exception as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS" if not code else "FAIL", args.output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
