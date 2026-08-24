"""Run Terminal-Bench 2.0 through Harbor and export trajectories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_trajectories import collect_raw_files, summarize_trajectories, write_benchmark_records
from scripts.runner_common import (
    build_manifest,
    changed_since,
    copy_changed_files,
    env_int,
    json_like_files,
    load_dotenv_file,
    now_timestamp,
    prepare_subprocess_env,
    print_run_report,
    ProgressPrinter,
    run_command,
    safe_run_part,
    shell_join,
    snapshot_files,
    write_json,
)


@dataclass
class TerminalBenchConfig:
    run_id: str
    benchmark: str
    output_root: Path
    run_dir: Path
    dataset: str
    agent: str
    model: str
    num_tasks: int
    all_tasks: bool
    num_trials: int
    max_workers: int | None
    jobs_dir: Path
    task_name: str | None
    harbor_job_name: str | None
    resume_missing_from_jobs: Path | None
    sandbox_env: str | None
    live_sync_interval: float


def build_terminal_bench_config(env: dict[str, str] | None = None, now: str | None = None) -> TerminalBenchConfig:
    env = env or os.environ
    timestamp = now or now_timestamp()
    benchmark = env.get("BENCHMARK", "terminal-bench-2.0")
    model = env.get("AGENT_MODEL", "openai/gpt-5.4")
    output_root = Path(env.get("OUTPUT_DIR", "outputs/runs"))
    run_id = env.get("TERMINAL_BENCH_RUN_ID") or env.get("RUN_ID") or f"{timestamp}_{safe_run_part(model)}_{safe_run_part(benchmark)}"
    return TerminalBenchConfig(
        run_id=run_id,
        benchmark=benchmark,
        output_root=output_root,
        run_dir=output_root / run_id,
        dataset=env.get("TERMINAL_BENCH_DATASET", "terminal-bench/terminal-bench-2"),
        agent=env.get("TERMINAL_BENCH_AGENT", "terminus-2"),
        model=model,
        num_tasks=env_int(env, "NUM_TASKS", 1),
        all_tasks=_env_bool(env, "TERMINAL_BENCH_ALL_TASKS"),
        num_trials=env_int(env, "NUM_TRIALS", 1),
        max_workers=_env_optional_int(env, "TERMINAL_BENCH_MAX_WORKERS"),
        jobs_dir=Path(env.get("HARBOR_JOBS_DIR", "jobs")),
        task_name=env.get("TERMINAL_BENCH_TASK_NAME") or None,
        harbor_job_name=env.get("HARBOR_JOB_NAME") or None,
        resume_missing_from_jobs=Path(env["TERMINAL_BENCH_RESUME_MISSING_FROM_JOBS"])
        if env.get("TERMINAL_BENCH_RESUME_MISSING_FROM_JOBS")
        else None,
        sandbox_env=env.get("HARBOR_ENV") or None,
        live_sync_interval=float(env.get("TERMINAL_BENCH_LIVE_SYNC_INTERVAL", "30")),
    )


def build_harbor_command(config: TerminalBenchConfig) -> list[str]:
    command = [
        "harbor",
        "run",
        "-d",
        config.dataset,
        "-a",
        config.agent,
        "-m",
        config.model,
        "-k",
        str(config.num_trials),
        "-o",
        str(config.jobs_dir),
    ]
    if config.harbor_job_name:
        command.extend(["--job-name", config.harbor_job_name])
    if config.max_workers is not None:
        command.extend(["-n", str(config.max_workers)])
    if config.task_name:
        command.extend(["--include-task-name", terminal_task_filter_name(config.task_name, config.dataset)])
    elif not config.all_tasks:
        command.extend(["-l", str(config.num_tasks)])
    if config.sandbox_env:
        command.extend(["--env", config.sandbox_env])
    return command


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file()
    parser = argparse.ArgumentParser(description="Run Terminal-Bench 2.0 with Harbor and collect trajectories.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and create manifest without running Harbor.")
    parser.add_argument("--benchmark", help="Override BENCHMARK.")
    parser.add_argument("--dataset", help="Override TERMINAL_BENCH_DATASET.")
    parser.add_argument("--agent", help="Override TERMINAL_BENCH_AGENT.")
    parser.add_argument("--model", help="Override AGENT_MODEL.")
    parser.add_argument("--num-tasks", type=int, help="Override NUM_TASKS, mapped to Harbor -l.")
    parser.add_argument("--all-tasks", action="store_true", help="Run all tasks by omitting Harbor -l.")
    parser.add_argument("--num-trials", type=int, help="Override NUM_TRIALS, mapped to Harbor -k.")
    parser.add_argument("--max-workers", type=int, help="Override Harbor -n concurrent trials.")
    parser.add_argument("--task-name", help="Run one Terminal-Bench task via --include-task-name.")
    parser.add_argument("--harbor-job-name", help="Override Harbor --job-name.")
    parser.add_argument(
        "--resume-missing-from-jobs",
        type=Path,
        help="Resume only tasks missing agent/trajectory.json, using a previous Harbor job directory as the task list source.",
    )
    parser.add_argument("--harbor-env", help="Optional Harbor sandbox provider, e.g. daytona.")
    parser.add_argument("--jobs-dir", type=Path, help="Harbor jobs directory to snapshot/copy from.")
    parser.add_argument("--output-dir", type=Path, help="Override OUTPUT_DIR.")
    parser.add_argument("--run-id", help="Use a fixed run id, useful for long background runs.")
    parser.add_argument("--live-sync-interval", type=float, help="Seconds between jobs-to-output live sync passes.")
    args = parser.parse_args(argv)

    env = prepare_subprocess_env(os.environ)
    _apply_overrides(env, args)
    config = build_terminal_bench_config(env=env)
    command = build_harbor_command(config)
    harbor_path = shutil.which("harbor")
    if harbor_path:
        command[0] = harbor_path

    raw_dir = config.run_dir / "raw"
    trajectories_dir = config.run_dir / "trajectories"
    benchmark_dir = config.run_dir / "benchmark_trajectories"
    for directory in (raw_dir, trajectories_dir, benchmark_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        run_id=config.run_id,
        benchmark=config.benchmark,
        command=command,
        config=asdict(config),
        dry_run=args.dry_run,
        executable_path=harbor_path,
    )
    before_snapshot = snapshot_files(config.jobs_dir)
    manifest["jobs_before_count"] = len(before_snapshot)

    if args.dry_run:
        if not harbor_path:
            print("Dry run: harbor command not found. Install with: uv tool install harbor")
        write_json(config.run_dir / "manifest.json", manifest)
        print_run_report(config.run_dir, command, raw_count=0, trajectory_count=0, summary={})
        return 0

    if not harbor_path:
        print("ERROR: harbor command not found. Install with: uv tool install harbor", file=sys.stderr)
        return 2
    if "OPENAI_API_KEY" not in env and "ANTHROPIC_API_KEY" not in env:
        print("ERROR: set the provider API key required by your Harbor agent/model.", file=sys.stderr)
        return 2

    if config.resume_missing_from_jobs:
        completed = run_missing_terminal_tasks(
            env=env,
            harbor_path=harbor_path,
            config=config,
            before_snapshot=before_snapshot,
            raw_dir=raw_dir,
            trajectories_dir=trajectories_dir,
            benchmark_dir=benchmark_dir,
            manifest=manifest,
        )
    else:
        print("Running:", shell_join(command))
        completed = run_harbor_command_with_live_sync(
            command,
            env=env,
            config=config,
            before_snapshot=before_snapshot,
            raw_dir=raw_dir,
            trajectories_dir=trajectories_dir,
            benchmark_dir=benchmark_dir,
            manifest=manifest,
        )
    manifest["returncode"] = completed.returncode

    after_snapshot = snapshot_files(config.jobs_dir)
    changed_files = changed_since(before_snapshot, after_snapshot)
    copied_raw_files = copy_changed_files(changed_files, source_root=config.jobs_dir, raw_dir=raw_dir)
    manifest["jobs_after_count"] = len(after_snapshot)
    manifest["raw_files"] = [str(path.relative_to(config.run_dir)) for path in copied_raw_files]

    # Final collection must rebuild from the full raw directory. Live sync may
    # have already copied older trajectories that are not in this final diff.
    summary = _collect_outputs(config, [], trajectories_dir, benchmark_dir)
    write_json(config.run_dir / "summary.json", summary)
    write_json(config.run_dir / "manifest.json", manifest)
    print_run_report(config.run_dir, command, len(copied_raw_files), summary.get("total_trajectories", 0), summary)
    if completed.returncode != 0:
        print("ERROR: Harbor exited with a non-zero status. See terminal_bench.stderr.log.", file=sys.stderr)
    return completed.returncode


def run_harbor_command_with_live_sync(
    command: list[str],
    *,
    env: dict[str, str],
    config: TerminalBenchConfig,
    before_snapshot: dict[str, tuple[int, int]],
    raw_dir: Path,
    trajectories_dir: Path,
    benchmark_dir: Path,
    manifest: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread = threading.Thread(target=_read_process_stream, args=(process.stdout, stdout_chunks), daemon=True)
    stderr_thread = threading.Thread(target=_read_process_stream, args=(process.stderr, stderr_chunks), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    total = None if config.all_tasks else max(0, config.num_tasks * config.num_trials)
    progress = ProgressPrinter(label="Terminal-Bench", total=total, interval_seconds=config.live_sync_interval)
    progress.update(0, extra="starting Harbor", force=True)
    poll_interval = 1.0 if config.live_sync_interval <= 0 else min(10.0, config.live_sync_interval)
    summary: dict[str, Any] = {}

    while process.poll() is None:
        time.sleep(poll_interval)
        summary = sync_terminal_outputs_live(
            config=config,
            raw_dir=raw_dir,
            trajectories_dir=trajectories_dir,
            benchmark_dir=benchmark_dir,
            manifest=manifest,
            before_snapshot=before_snapshot,
        )
        progress.update(
            int(summary.get("total_trajectories", 0) or 0),
            extra=f"raw files={manifest.get('raw_file_count', 0)}",
        )

    returncode = process.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    summary = sync_terminal_outputs_live(
        config=config,
        raw_dir=raw_dir,
        trajectories_dir=trajectories_dir,
        benchmark_dir=benchmark_dir,
        manifest=manifest,
        before_snapshot=before_snapshot,
    )
    progress.update(
        int(summary.get("total_trajectories", 0) or 0),
        extra=f"finished returncode={returncode}; raw files={manifest.get('raw_file_count', 0)}",
        force=True,
    )
    (config.run_dir / "terminal_bench.stdout.log").write_text("".join(stdout_chunks), encoding="utf-8")
    (config.run_dir / "terminal_bench.stderr.log").write_text("".join(stderr_chunks), encoding="utf-8")
    return subprocess.CompletedProcess(command, returncode, "".join(stdout_chunks), "".join(stderr_chunks))


def run_missing_terminal_tasks(
    *,
    env: dict[str, str],
    harbor_path: str | None,
    config: TerminalBenchConfig,
    before_snapshot: dict[str, tuple[int, int]],
    raw_dir: Path,
    trajectories_dir: Path,
    benchmark_dir: Path,
    manifest: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    source_job_dir = resolve_terminal_job_dir(config.resume_missing_from_jobs, config.jobs_dir)
    missing_tasks = missing_terminal_trajectory_task_names(source_job_dir, completed_root=raw_dir)
    max_workers = max(1, config.max_workers or 1)
    resume_batch_id = now_timestamp()
    manifest["resume_source_job"] = str(source_job_dir)
    manifest["resume_missing_task_names"] = missing_tasks
    manifest["resume_missing_task_count"] = len(missing_tasks)
    manifest["resume_max_workers"] = max_workers
    manifest["resume_batch_id"] = resume_batch_id

    if not missing_tasks:
        summary = sync_terminal_outputs_live(
            config=config,
            raw_dir=raw_dir,
            trajectories_dir=trajectories_dir,
            benchmark_dir=benchmark_dir,
            manifest=manifest,
            before_snapshot=before_snapshot,
        )
        print_run_report(config.run_dir, ["terminal-bench-resume-missing"], 0, summary.get("total_trajectories", 0), summary)
        return subprocess.CompletedProcess(["terminal-bench-resume-missing"], 0, "", "")

    print(f"Resuming {len(missing_tasks)} missing Terminal-Bench task(s) from {source_job_dir}")
    print(f"Wrapper concurrency: {max_workers}")
    progress = ProgressPrinter(label="Terminal-Bench resume", total=len(missing_tasks), interval_seconds=config.live_sync_interval)
    progress.update(0, extra="starting per-task Harbor jobs", force=True)
    poll_interval = 1.0 if config.live_sync_interval <= 0 else min(10.0, config.live_sync_interval)
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_one_terminal_resume_task,
                index=index,
                task_name=task_name,
                config=config,
                env=env,
                harbor_path=harbor_path,
                resume_batch_id=resume_batch_id,
            ): task_name
            for index, task_name in enumerate(missing_tasks, start=1)
        }
        while futures:
            done, _ = wait(futures, timeout=poll_interval, return_when=FIRST_COMPLETED)
            if done:
                for future in done:
                    task_name = futures.pop(future)
                    try:
                        results.append(future.result())
                    except Exception as exc:  # pragma: no cover - defensive process wrapper
                        results.append({"task_name": task_name, "returncode": 1, "error": repr(exc)})
            summary = sync_terminal_outputs_live(
                config=config,
                raw_dir=raw_dir,
                trajectories_dir=trajectories_dir,
                benchmark_dir=benchmark_dir,
                manifest=manifest,
                before_snapshot=before_snapshot,
            )
            manifest["resume_runs"] = results
            progress.update(
                len(results),
                extra=f"running={len(futures)}; trajectories={summary.get('total_trajectories', 0)}",
                force=not futures,
            )

    summary = sync_terminal_outputs_live(
        config=config,
        raw_dir=raw_dir,
        trajectories_dir=trajectories_dir,
        benchmark_dir=benchmark_dir,
        manifest=manifest,
        before_snapshot=before_snapshot,
    )
    manifest["resume_runs"] = results
    write_json(config.run_dir / "manifest.json", manifest)
    returncode = 0 if all(item.get("returncode") == 0 for item in results) else 1
    stdout = "\n".join(item.get("stdout_log", "") for item in results)
    stderr = "\n".join(item.get("stderr_log", "") for item in results if item.get("returncode") != 0)
    return subprocess.CompletedProcess(["terminal-bench-resume-missing"], returncode, stdout, stderr)


def run_one_terminal_resume_task(
    *,
    index: int,
    task_name: str,
    config: TerminalBenchConfig,
    env: dict[str, str],
    harbor_path: str | None,
    resume_batch_id: str,
) -> dict[str, Any]:
    safe_task = safe_run_part(task_name)[:48]
    task_config = replace(
        config,
        all_tasks=False,
        max_workers=None,
        task_name=task_name,
        harbor_job_name=f"resume_{resume_batch_id}_{index:03d}_{safe_task}",
        resume_missing_from_jobs=None,
    )
    command = build_harbor_command(task_config)
    if harbor_path:
        command[0] = harbor_path
    stdout_log = config.run_dir / f"terminal_bench_resume_{index:03d}_{safe_task}.stdout.log"
    stderr_log = config.run_dir / f"terminal_bench_resume_{index:03d}_{safe_task}.stderr.log"
    print(f"Starting resume task {index}: {shell_join(command)}")
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    stdout_log.write_text(completed.stdout or "", encoding="utf-8")
    stderr_log.write_text(completed.stderr or "", encoding="utf-8")
    return {
        "index": index,
        "task_name": task_name,
        "command": command,
        "returncode": completed.returncode,
        "stdout_log": str(stdout_log.relative_to(config.run_dir)),
        "stderr_log": str(stderr_log.relative_to(config.run_dir)),
    }


def _read_process_stream(stream: Any, chunks: list[str]) -> None:
    if stream is None:
        return
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        stream.close()


def resolve_terminal_job_dir(job_dir: Path | None, jobs_dir: Path) -> Path:
    if job_dir is None:
        raise ValueError("resume source job directory is required")
    candidates = [job_dir, jobs_dir / job_dir]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Terminal-Bench job directory not found: {job_dir}")


def missing_terminal_trajectory_task_names(job_dir: Path, completed_root: Path | None = None) -> list[str]:
    expected = task_names_from_terminal_lock(job_dir)
    completed = completed_terminal_trajectory_task_names(job_dir)
    if completed_root is not None and completed_root.exists():
        completed.update(completed_terminal_trajectory_task_names(completed_root))
    return [task_name for task_name in expected if task_name not in completed]


def task_names_from_terminal_lock(job_dir: Path) -> list[str]:
    lock_path = job_dir / "lock.json"
    if not lock_path.exists():
        return []
    payload = _read_json_file(lock_path)
    names: list[str] = []
    for trial in payload.get("trials", []) if isinstance(payload, dict) else []:
        task = trial.get("task", {}) if isinstance(trial, dict) else {}
        name = task.get("name") if isinstance(task, dict) else None
        short_name = terminal_task_short_name(name)
        if short_name and short_name not in names:
            names.append(short_name)
    return names


def completed_terminal_trajectory_task_names(root: Path) -> set[str]:
    completed: set[str] = set()
    if not root.exists():
        return completed
    for trajectory_path in root.rglob("agent/trajectory.json"):
        task_dir = trajectory_path.parent.parent
        result_path = task_dir / "result.json"
        short_name = None
        if result_path.exists():
            result = _read_json_file(result_path)
            if isinstance(result, dict):
                short_name = terminal_task_short_name(result.get("task_name"))
                if short_name is None:
                    task_id = result.get("task_id")
                    if isinstance(task_id, dict):
                        short_name = terminal_task_short_name(task_id.get("name"))
        if short_name is None:
            short_name = terminal_task_short_name(task_dir.name.split("__", 1)[0])
        if short_name:
            completed.add(short_name)
    return completed


def terminal_task_short_name(task_name: Any) -> str | None:
    if not isinstance(task_name, str) or not task_name.strip():
        return None
    name = task_name.strip().replace("\\", "/").rstrip("/")
    return name.rsplit("/", 1)[-1]


def terminal_task_filter_name(task_name: str, dataset: str) -> str:
    name = task_name.strip().replace("\\", "/").rstrip("/")
    if "/" in name:
        return name
    dataset_org = dataset.split("/", 1)[0] if "/" in dataset else "terminal-bench"
    return f"{dataset_org}/{name}"


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def sync_terminal_outputs_live(
    *,
    config: TerminalBenchConfig,
    raw_dir: Path,
    trajectories_dir: Path,
    benchmark_dir: Path,
    manifest: dict[str, Any],
    before_snapshot: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    if before_snapshot is None:
        relative_files = sorted(
            path.relative_to(config.jobs_dir)
            for path in config.jobs_dir.rglob("*")
            if path.is_file()
        ) if config.jobs_dir.exists() else []
    else:
        relative_files = changed_since(before_snapshot, snapshot_files(config.jobs_dir))
    copied_raw_files = copy_changed_files(relative_files, source_root=config.jobs_dir, raw_dir=raw_dir)
    raw_files = select_terminal_json_files(json_like_files(raw_dir))
    summary = _collect_outputs(config, raw_files, trajectories_dir, benchmark_dir)
    manifest["jobs_after_count"] = len(snapshot_files(config.jobs_dir))
    manifest["raw_files"] = [str(path.relative_to(config.run_dir)) for path in json_like_files(raw_dir)]
    manifest["raw_file_count"] = len(manifest["raw_files"])
    manifest["selected_raw_files"] = [str(path.relative_to(config.run_dir)) for path in raw_files]
    manifest["selected_raw_file_count"] = len(raw_files)
    write_json(config.run_dir / "summary.json", summary)
    write_json(config.run_dir / "manifest.json", manifest)
    return summary


def _collect_outputs(
    config: TerminalBenchConfig,
    copied_raw_files: list[Path],
    trajectories_dir: Path,
    benchmark_dir: Path,
) -> dict[str, Any]:
    raw_files = select_terminal_json_files([path for path in copied_raw_files if path.suffix.lower() in {".json", ".jsonl"}])
    if not raw_files:
        raw_files = select_terminal_json_files(json_like_files(config.run_dir / "raw"))
    trajectories = collect_raw_files(
        raw_files=raw_files,
        trajectories_dir=trajectories_dir,
        jsonl_path=config.run_dir / "trajectories.jsonl",
        run_id=config.run_id,
        defaults={
            "domain": config.dataset,
            "agent_model": config.model,
            "user_model": None,
        },
        raw_root=config.run_dir,
    )
    benchmark_records = write_benchmark_records(
        trajectories=trajectories,
        output_dir=benchmark_dir,
        jsonl_path=config.run_dir / "benchmark_trajectories.jsonl",
        benchmark=config.benchmark,
    )
    summary = summarize_trajectories(trajectories)
    summary["benchmark_task_count"] = len(benchmark_records)
    summary["benchmark_trajectory_count"] = sum(len(record.get("trajectories", [])) for record in benchmark_records)
    return summary


def select_terminal_json_files(raw_files: list[Path]) -> list[Path]:
    trajectories = [path for path in raw_files if path.name == "trajectory.json"]
    if trajectories:
        return sorted(trajectories)
    results = [path for path in raw_files if path.name == "result.json"]
    if results:
        return sorted(results)
    return sorted(raw_files)


def _apply_overrides(env: dict[str, str], args: argparse.Namespace) -> None:
    overrides = {
        "BENCHMARK": args.benchmark,
        "TERMINAL_BENCH_DATASET": args.dataset,
        "TERMINAL_BENCH_AGENT": args.agent,
        "AGENT_MODEL": args.model,
        "NUM_TASKS": args.num_tasks,
        "TERMINAL_BENCH_ALL_TASKS": "true" if args.all_tasks else None,
        "NUM_TRIALS": args.num_trials,
        "TERMINAL_BENCH_MAX_WORKERS": args.max_workers,
        "TERMINAL_BENCH_TASK_NAME": args.task_name,
        "HARBOR_JOB_NAME": args.harbor_job_name,
        "TERMINAL_BENCH_RESUME_MISSING_FROM_JOBS": str(args.resume_missing_from_jobs)
        if args.resume_missing_from_jobs
        else None,
        "HARBOR_ENV": args.harbor_env,
        "HARBOR_JOBS_DIR": str(args.jobs_dir) if args.jobs_dir else None,
        "OUTPUT_DIR": str(args.output_dir) if args.output_dir else None,
        "TERMINAL_BENCH_RUN_ID": args.run_id,
        "TERMINAL_BENCH_LIVE_SYNC_INTERVAL": args.live_sync_interval,
    }
    for key, value in overrides.items():
        if value is not None:
            env[key] = str(value)


def _env_bool(env: dict[str, str], key: str) -> bool:
    value = env.get(key)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_optional_int(env: dict[str, str], key: str) -> int | None:
    value = env.get(key)
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
