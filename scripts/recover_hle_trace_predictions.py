from __future__ import annotations

import glob
import json
import os
import sys


def recover(raw_dir: str, output_path: str) -> None:
    main_path = os.path.join(raw_dir, "hle_gpt-5.6-sol.json")
    base = json.load(open(main_path, encoding="utf-8"))
    merged = dict(base)
    found = {}
    for path in glob.glob(os.path.join(raw_dir, "traces", "trace_*.log")):
        try:
            if os.path.getsize(path) >= 100 * 1024 * 1024:
                continue
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        marker_pos = text.rfind("QUESTION COMPLETE ID:")
        final_pos = text.rfind("FINAL ANSWER [", 0, marker_pos)
        if marker_pos < 0 or final_pos < 0:
            continue
        key_start = final_pos + len("FINAL ANSWER [")
        key_end = text.find("]", key_start)
        if key_end < 0:
            continue
        key = text[key_start:key_end]
        content = text.find("Content:", final_pos, marker_pos)
        if content < 0:
            continue
        lines = []
        for line in text[content + len("Content:") : marker_pos].splitlines():
            if line.lstrip().startswith("└"):
                break
            if line.startswith("│   "):
                line = line[4:]
            elif line.startswith("│ "):
                line = line[2:]
            lines.append(line.rstrip())
        response = "\n".join(lines).strip()
        usage_pos = text.find("Total tokens: prompt=", marker_pos)
        if usage_pos < 0 or not response:
            continue
        usage_text = text[usage_pos + len("Total tokens: prompt=") :].splitlines()[0]
        try:
            prompt_text, completion_text, total_text = usage_text.split(", ")
            prompt_tokens = int(prompt_text)
            completion_tokens = int(completion_text.split("=", 1)[1])
            total_tokens = int(total_text.split("=", 1)[1])
        except (ValueError, IndexError):
            continue
        if total_tokens <= 0:
            continue
        found[key] = {
            "model": "gpt-5.6-sol",
            "response": response,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
    merged.update(found)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=4)
    print(
        f"trace_successes={len(found)} merged_predictions={len(merged)} "
        f"new_keys={len(set(found) - set(base))} bytes={os.path.getsize(output_path)}"
    )


if __name__ == "__main__":
    recover(sys.argv[1], sys.argv[2])
