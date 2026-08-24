"""Probe DDGS text-search backends with representative queries."""

from __future__ import annotations

import argparse
import json
import time

from ddgs import DDGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument(
        "--backends",
        default="auto,duckduckgo,google,bing,brave",
        help="Comma-separated DDGS backend names.",
    )
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max-results", type=int, default=8)
    args = parser.parse_args()

    for query in args.query:
        for backend in args.backends.split(","):
            started = time.perf_counter()
            record: dict[str, object] = {
                "query": query,
                "backend": backend,
            }
            try:
                results = DDGS(timeout=args.timeout).text(
                    query,
                    max_results=args.max_results,
                    backend=backend,
                )
                record.update(
                    ok=True,
                    count=len(results),
                    hosts=[
                        result.get("href", "").split("/")[2]
                        for result in results[:3]
                        if len(result.get("href", "").split("/")) > 2
                    ],
                )
            except Exception as exc:
                record.update(
                    ok=False,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            record["seconds"] = round(time.perf_counter() - started, 3)
            print(json.dumps(record, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
