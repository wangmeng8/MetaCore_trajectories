"""Small shared helpers for benchmark runner scripts."""

from __future__ import annotations

import base64
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO


def load_dotenv_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def prepare_subprocess_env(base_env: dict[str, str] | os._Environ[str]) -> dict[str, str]:
    env = dict(base_env)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "0")
    return env


def env_int(env: dict[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or value == "":
        return default
    return int(value)


def env_optional_int(env: dict[str, str], key: str) -> int | None:
    value = env.get(key)
    if value is None or value == "":
        return None
    return int(value)


def env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_run_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def build_manifest(
    *,
    run_id: str,
    benchmark: str,
    command: list[str],
    config: dict[str, Any],
    dry_run: bool,
    executable_path: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "command": command,
        "command_string": shell_join(command),
        "config": _jsonable(config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "executable_path": executable_path,
        },
    }


def run_command(command: list[str], env: dict[str, str], run_dir: Path, log_prefix: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    (run_dir / f"{log_prefix}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / f"{log_prefix}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def snapshot_files(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    snapshot: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def changed_since(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[Path]:
    changed: list[Path] = []
    for relative, metadata in after.items():
        if before.get(relative) != metadata:
            changed.append(Path(relative))
    return sorted(changed)


def copy_changed_files(changed_files: Iterable[Path], source_root: Path, raw_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for relative in changed_files:
        source = source_root / relative
        if not source.is_file():
            continue
        target = raw_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def copy_tree_files(source_root: Path, raw_dir: Path) -> list[Path]:
    copied: list[Path] = []
    if not source_root.exists():
        return copied
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        target = raw_dir / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def json_like_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})


def print_run_report(run_dir: Path, command: list[str], raw_count: int, trajectory_count: int, summary: dict[str, Any]) -> None:
    print("Output directory:", run_dir)
    print("Command:", shell_join(command))
    print("Raw files:", raw_count)
    print("Trajectories:", trajectory_count)
    if summary:
        print("Summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def format_progress_bar(completed: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return f"{completed} completed"
    bounded_completed = max(0, min(completed, total))
    filled = int(width * bounded_completed / total)
    percent = 100.0 * bounded_completed / total
    return f"[{'#' * filled}{'-' * (width - filled)}] {bounded_completed}/{total} {percent:.1f}%"


def format_progress_line(
    label: str,
    *,
    completed: int,
    total: int | None,
    elapsed_seconds: float,
    extra: str | None = None,
) -> str:
    elapsed = _format_elapsed(elapsed_seconds)
    if total is None:
        progress = f"{completed} completed"
    else:
        progress = format_progress_bar(completed, total)
    line = f"{label} progress {progress} | elapsed {elapsed}"
    if extra:
        line = f"{line} | {extra}"
    return line


class ProgressPrinter:
    def __init__(
        self,
        *,
        label: str,
        total: int | None,
        interval_seconds: float = 30,
        stream: TextIO | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.label = label
        self.total = total
        self.interval_seconds = max(0.0, interval_seconds)
        self.stream = stream or sys.stdout
        self.clock = clock or time.monotonic
        self.started_at = self.clock()
        self.last_printed_at: float | None = None

    def update(self, completed: int, *, extra: str | None = None, force: bool = False) -> None:
        now = self.clock()
        if not force and self.last_printed_at is not None and now - self.last_printed_at < self.interval_seconds:
            return
        self.last_printed_at = now
        print(
            format_progress_line(
                self.label,
                completed=completed,
                total=self.total,
                elapsed_seconds=now - self.started_at,
                extra=extra,
            ),
            file=self.stream,
            flush=True,
        )


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def shell_join(command: list[str]) -> str:
    return " ".join(_quote_arg(part) for part in command)


def _quote_arg(part: str) -> str:
    if not part or any(char.isspace() for char in part):
        return '"' + part.replace('"', '\\"') + '"'
    return part


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "__type__": "bytes",
            "encoding": "base64",
            "length": len(raw),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return {"__type__": "float", "value": str(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    return str(value)
