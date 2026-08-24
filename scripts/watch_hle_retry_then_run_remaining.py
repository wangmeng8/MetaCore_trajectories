"""Wait for an HLE retry run to finish, then launch the remaining HLE rows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> int:
    args = parse_args()
    retry_run_dir = args.retry_run_dir
    remaining_run_dir = args.output_dir / args.remaining_run_id
    remaining_run_dir.mkdir(parents=True, exist_ok=True)

    log_path = remaining_run_dir / "handoff_watcher.log"
    with log_path.open("a", encoding="utf-8") as log:
        log_line(log, f"watcher started; retry_run_dir={retry_run_dir}; retry_pid={args.retry_pid}")
        status = wait_for_retry_completion(args, log)
        if status != 0:
            return status

        prepare_command = [
            sys.executable,
            "scripts/prepare_hle_remaining_dataset.py",
            "--source-data",
            str(args.source_data),
            "--run-dir",
            str(args.main_run_dir),
            "--run-dir",
            str(retry_run_dir),
            "--output",
            str(args.remaining_data),
        ]
        log_line(log, "preparing remaining parquet: " + shell_join(prepare_command))
        prepare_status = run_logged(prepare_command, log, cwd=Path.cwd())
        if prepare_status != 0:
            log_line(log, f"remaining parquet preparation failed with status {prepare_status}")
            return prepare_status

        remaining_count = count_remaining_from_manifest(args.remaining_data)
        log_line(log, f"remaining parquet ready: {args.remaining_data}; rows={remaining_count}")
        if remaining_count <= 0:
            log_line(log, "no remaining rows; nothing to launch")
            return 0

        run_command = [
            sys.executable,
            "scripts/run_hle_with_tools.py",
            "--data-path",
            str(args.remaining_data),
            "--model",
            args.model,
            "--base-url",
            args.base_url,
            "--all-tasks",
            "--max-workers",
            str(args.max_workers),
            "--max-retries",
            str(args.max_retries),
            "--process-retries",
            str(args.process_retries),
            "--max-completion-tokens",
            str(args.max_completion_tokens),
            "--max-iterations",
            str(args.max_iterations),
            "--run-id",
            args.remaining_run_id,
            "--output-dir",
            str(args.output_dir),
            "--progress-interval",
            str(args.progress_interval),
        ]
        log_line(log, "launching remaining run: " + shell_join(run_command))
        status = run_logged(
            run_command,
            log,
            cwd=Path.cwd(),
            stdout_path=remaining_run_dir / "background.stdout.log",
            stderr_path=remaining_run_dir / "background.stderr.log",
        )
        log_line(log, f"remaining run exited with status {status}")
        return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chain HLE zero-token retry into the remaining text-only run.")
    parser.add_argument("--retry-run-dir", type=Path, required=True)
    parser.add_argument("--retry-pid", type=int, required=True)
    parser.add_argument("--expected-retry-count", type=int, default=243)
    parser.add_argument("--source-data", type=Path, default=Path("data/modelscope/cais_hle/data/test-00000-of-00001.parquet"))
    parser.add_argument("--main-run-dir", type=Path, default=Path("outputs/runs/hle_with_tools_gpt55_full_newapi"))
    parser.add_argument("--remaining-data", type=Path, default=Path("data/retry/hle_remaining_after_retry_gpt55.parquet"))
    parser.add_argument("--remaining-run-id", default="hle_remaining_after_retry_gpt55")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="https://newapi.metamind.work/v1")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--process-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=40000)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--progress-interval", type=float, default=30)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser.parse_args()


def wait_for_retry_completion(args: argparse.Namespace, log: object) -> int:
    while True:
        count = prediction_count(args.retry_run_dir)
        summary_exists = (args.retry_run_dir / "summary.json").exists()
        alive = is_pid_alive(args.retry_pid)
        log_line(log, f"retry status: predictions={count}/{args.expected_retry_count}; alive={alive}; summary={summary_exists}")

        if summary_exists and count >= args.expected_retry_count:
            log_line(log, "retry run completed with summary")
            return 0
        if count >= args.expected_retry_count and not alive:
            log_line(log, "retry run reached expected count and process exited")
            return 0
        if not alive and count < args.expected_retry_count:
            log_line(log, "retry process exited before expected count; not launching remaining run")
            return 3
        time.sleep(max(1.0, args.poll_seconds))


def prediction_count(run_dir: Path) -> int:
    raw_dir = run_dir / "raw" / "official_run"
    for path in (raw_dir / "hle_gpt-5.5.json", raw_dir / "hle_gpt-5.5.json.temp"):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return len(payload)
    return 0


def count_remaining_from_manifest(data_path: Path) -> int:
    manifest_path = data_path.with_suffix(data_path.suffix + ".manifest.json")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return -1
    value = payload.get("remaining_rows")
    return value if isinstance(value, int) else -1


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def run_logged(
    command: list[str],
    log: object,
    *,
    cwd: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> int:
    if stdout_path is None or stderr_path is None:
        completed = subprocess.run(command, cwd=cwd, text=True, encoding="utf-8", errors="replace", capture_output=True)
        if completed.stdout:
            log_line(log, completed.stdout.rstrip())
        if completed.stderr:
            log_line(log, completed.stderr.rstrip())
        return int(completed.returncode)

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log_line(log, f"child pid={process.pid}; stdout={stdout_path}; stderr={stderr_path}")
        return int(process.wait())


def log_line(log: object, message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", file=log, flush=True)


def shell_join(command: list[str]) -> str:
    return " ".join(quote_arg(part) for part in command)


def quote_arg(part: object) -> str:
    value = str(part)
    if not value or any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
