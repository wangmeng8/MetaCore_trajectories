from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(r"C:\Users\wm802\Documents\MetaCoreBench")
RUN = ROOT / "outputs/runs/hle_gpt56sol_full_web_only_r4"
JUDGED = RUN / "judged_predictions_gpt56_luna.json"
TASK_ID_RE = re.compile(r'"task_id"\s*:\s*"([^"]+)"')


def judged_score(task_id: str, judged: dict[str, object]) -> float | None:
    value = judged.get(task_id)
    if not isinstance(value, dict):
        return None
    response = value.get("judge_response")
    if not isinstance(response, dict):
        return None
    return 1.0 if str(response.get("correct", "")).lower() == "yes" else 0.0


def replace_once(text: str, field: str, value: str) -> str:
    return re.sub(
        rf'("{re.escape(field)}"\s*:\s*)null',
        rf'\g<1>{value}',
        text,
        count=1,
    )


def rewrite_individual(path: Path, judged: dict[str, object]) -> bool:
    with path.open(encoding="utf-8") as source:
        prefix = source.read(65536)
        task_match = TASK_ID_RE.search(prefix)
        if not task_match:
            return False
        score = judged_score(task_match.group(1), judged)
        if score is None:
            return False
        if not re.search(r'"score"\s*:\s*null', prefix):
            return False
        prefix = replace_once(prefix, "success", "true" if score else "false")
        prefix = replace_once(prefix, "score", str(score))
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as target:
            temporary = Path(target.name)
            target.write(prefix)
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
    for _ in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            time.sleep(0.25)
    else:
        temporary.unlink(missing_ok=True)
        raise
    return True


def rewrite_jsonl(path: Path, judged: dict[str, object], benchmark: bool = False) -> int:
    count = 0
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as target:
        temporary = Path(target.name)
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                task_match = TASK_ID_RE.search(line)
                score = judged_score(task_match.group(1), judged) if task_match else None
                if score is not None:
                    if benchmark:
                        metadata_at = line.find('"metadata"')
                        if metadata_at >= 0:
                            before = line[:metadata_at]
                            metadata = line[metadata_at:]
                            metadata = replace_once(metadata, "success", "true" if score else "false")
                            metadata = replace_once(metadata, "score", str(score))
                            line = before + metadata
                    else:
                        line = replace_once(line, "success", "true" if score else "false")
                        line = replace_once(line, "score", str(score))
                    count += 1
                target.write(line)
    os.replace(temporary, path)
    return count


def main() -> None:
    judged = json.loads(JUDGED.read_text(encoding="utf-8"))
    judged = {str(key): value for key, value in judged.items()}

    individual_count = 0
    for path in (RUN / "trajectories").glob("*.json"):
        individual_count += rewrite_individual(path, judged)

    trajectories_count = rewrite_jsonl(RUN / "trajectories.jsonl", judged)
    benchmark_count = rewrite_jsonl(RUN / "benchmark_trajectories.jsonl", judged, benchmark=True)
    print(
        json.dumps(
            {
                "judged_records": len(judged),
                "individual_trajectories_updated": individual_count,
                "trajectories_jsonl_records_updated": trajectories_count,
                "benchmark_trajectories_jsonl_records_updated": benchmark_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
