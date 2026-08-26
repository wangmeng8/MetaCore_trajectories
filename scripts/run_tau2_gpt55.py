"""Run a small Tau2-bench smoke test with GPT-5.5 and collect trajectories."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_trajectories import collect_raw_files, summarize_trajectories, write_benchmark_records
from scripts.runner_common import ProgressPrinter


DEFAULT_SIMULATIONS_DIR = Path("data") / "simulations"


@dataclass
class RunConfig:
    run_id: str
    output_root: Path
    run_dir: Path
    benchmark: str
    domain: str
    agent_model: str
    user_model: str
    agent_llm_args: dict[str, Any]
    user_llm_args: dict[str, Any]
    num_tasks: int
    all_tasks: bool
    task_ids: list[str] | None
    task_set_name: str | None
    task_split_name: str | None
    num_trials: int
    reasoning_effort: str
    temperature: str
    max_steps: int | None
    max_workers: int | None
    resume: bool
    one_by_one: bool
    save_to: str | None
    auto_resume: bool
    progress_interval: float
    simulations_dir: Path
    tau2_repo_dir: Path | None


@dataclass
class Tau2Invocation:
    command_prefix: list[str]
    cwd: Path | None
    available: bool
    source: str


def build_run_config(env: dict[str, str] | None = None, now: str | None = None) -> RunConfig:
    env = env or os.environ
    timestamp = now or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    domain = env.get("TAU2_DOMAIN", "airline")
    agent_model = env.get("AGENT_MODEL", "gpt-5.5")
    user_model = env.get("USER_MODEL", "gpt-5.5")
    output_root = Path(env.get("OUTPUT_DIR", "outputs/runs"))
    safe_model = _safe_run_part(agent_model)
    safe_domain = _safe_run_part(domain)
    run_id = env.get("TAU2_RUN_ID") or env.get("RUN_ID") or f"{timestamp}_{safe_model}_{safe_domain}"
    run_dir = output_root / run_id

    tau2_repo_dir_env = env.get("TAU2_REPO_DIR") or env.get("TAU2_BENCH_REPO_DIR")
    tau2_repo_dir = Path(tau2_repo_dir_env) if tau2_repo_dir_env else None

    simulations_dir_env = env.get("TAU2_SIMULATIONS_DIR")
    if simulations_dir_env:
        simulations_dir = Path(simulations_dir_env)
    else:
        tau2_data_dir = env.get("TAU2_DATA_DIR")
        if tau2_data_dir:
            simulations_dir = Path(tau2_data_dir) / "simulations"
        elif tau2_repo_dir:
            simulations_dir = tau2_repo_dir / "data" / "simulations"
        else:
            simulations_dir = DEFAULT_SIMULATIONS_DIR

    return RunConfig(
        run_id=run_id,
        output_root=output_root,
        run_dir=run_dir,
        benchmark=env.get("BENCHMARK", "tau2-bench"),
        domain=domain,
        agent_model=agent_model,
        user_model=user_model,
        agent_llm_args=_parse_json_object(env.get("AGENT_LLM_ARGS"), "AGENT_LLM_ARGS"),
        user_llm_args=_parse_json_object(env.get("USER_LLM_ARGS"), "USER_LLM_ARGS"),
        num_tasks=_env_int(env, "NUM_TASKS", 2),
        all_tasks=_env_bool(env, "TAU2_ALL_TASKS"),
        task_ids=_parse_task_ids(env.get("TAU2_TASK_IDS")),
        task_set_name=env.get("TAU2_TASK_SET_NAME") or None,
        task_split_name=env.get("TAU2_TASK_SPLIT_NAME") or None,
        num_trials=_env_int(env, "NUM_TRIALS", 1),
        reasoning_effort=env.get("REASONING_EFFORT", "medium"),
        temperature=env.get("TEMPERATURE", "0"),
        max_steps=_env_optional_int(env, "MAX_STEPS"),
        max_workers=_env_optional_int(env, "TAU2_MAX_WORKERS"),
        resume=_env_bool(env, "TAU2_RESUME"),
        one_by_one=_env_bool(env, "TAU2_ONE_BY_ONE"),
        save_to=env.get("TAU2_SAVE_TO") or None,
        auto_resume=_env_bool(env, "TAU2_AUTO_RESUME"),
        progress_interval=float(env.get("TAU2_PROGRESS_INTERVAL", "30")),
        simulations_dir=simulations_dir,
        tau2_repo_dir=tau2_repo_dir,
    )


def build_tau2_command(
    config: RunConfig,
    help_text: str | None = None,
    command_prefix: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    command = list(command_prefix or ["tau2"])
    command.append("run")
    ignored: list[str] = []

    _add_required_option(command, help_text, ("--domain",), config.domain, "TAU2_DOMAIN")
    _add_required_option(
        command,
        help_text,
        ("--agent-llm", "--agent_llm", "--agent-model", "--agent_model"),
        config.agent_model,
        "AGENT_MODEL",
    )
    _add_required_option(
        command,
        help_text,
        ("--user-llm", "--user_llm", "--user-model", "--user_model"),
        config.user_model,
        "USER_MODEL",
    )
    _add_optional_option(
        command,
        help_text,
        ("--agent-llm-args", "--agent_llm_args"),
        _compact_json(config.agent_llm_args),
        "AGENT_LLM_ARGS",
        ignored,
    )
    _add_optional_option(
        command,
        help_text,
        ("--user-llm-args", "--user_llm_args"),
        _compact_json(config.user_llm_args),
        "USER_LLM_ARGS",
        ignored,
    )
    _add_required_option(command, help_text, ("--num-trials", "--num_trials"), str(config.num_trials), "NUM_TRIALS")
    _add_optional_option(command, help_text, ("--task-set-name", "--task_set_name"), config.task_set_name, "TAU2_TASK_SET_NAME", ignored)
    _add_optional_option(command, help_text, ("--task-split-name", "--task_split_name"), config.task_split_name, "TAU2_TASK_SPLIT_NAME", ignored)
    if config.task_ids:
        _add_multi_value_option(command, help_text, ("--task-ids", "--task_ids"), config.task_ids, "TAU2_TASK_IDS", ignored)
    elif not config.all_tasks:
        _add_required_option(command, help_text, ("--num-tasks", "--num_tasks"), str(config.num_tasks), "NUM_TASKS")

    _add_optional_option(command, help_text, ("--reasoning-effort", "--reasoning_effort"), config.reasoning_effort, "REASONING_EFFORT", ignored)
    _add_optional_option(command, help_text, ("--temperature",), config.temperature, "TEMPERATURE", ignored)
    if config.max_steps is not None:
        _add_optional_option(command, help_text, ("--max-steps", "--max_steps"), str(config.max_steps), "MAX_STEPS", ignored)
    _add_optional_option(command, help_text, ("--save-to", "--save_to"), config.save_to, "TAU2_SAVE_TO", ignored)
    if config.auto_resume:
        _add_flag_option(command, help_text, ("--auto-resume", "--auto_resume"), "TAU2_AUTO_RESUME", ignored)
    if config.max_workers is not None:
        _add_optional_option(
            command,
            help_text,
            (
                "--max-concurrency",
                "--max_concurrency",
                "--max-workers",
                "--max_workers",
                "--num-workers",
                "--num_workers",
                "--concurrency",
            ),
            str(config.max_workers),
            "TAU2_MAX_WORKERS",
            ignored,
        )

    return command, ignored


def run_multiple_domains(args: argparse.Namespace, argv: list[str] | None) -> int:
    domains = []
    for value in args.domains or []:
        domains.extend(part.strip() for part in value.split(",") if part.strip())
    domains = list(dict.fromkeys(domains))
    if not domains:
        raise ValueError("--domains requires at least one domain")

    base_argv = _strip_multi_domain_args(list(sys.argv[1:] if argv is None else argv))
    base_run_id = (
        args.run_id
        or os.environ.get("TAU2_RUN_ID")
        or f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{_safe_run_part(args.agent_model or os.environ.get('AGENT_MODEL', 'gpt-5.5'))}"
    )
    return_codes: list[int] = []

    for domain in domains:
        domain_run_id = f"{base_run_id}_{_safe_run_part(domain)}"
        domain_argv = [*base_argv, "--domain", domain, "--run-id", domain_run_id]
        print(f"\n===== Tau2 domain {domain} ({len(return_codes) + 1}/{len(domains)}) =====")
        return_codes.append(main(domain_argv))

    return next((code for code in return_codes if code != 0), 0)


def _strip_multi_domain_args(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--domains":
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        if token.startswith("--domains="):
            index += 1
            continue
        if token == "--run-id":
            index += 2
            continue
        if token.startswith("--run-id="):
            index += 1
            continue
        stripped.append(token)
        index += 1
    return stripped


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file(Path(".env"))
    parser = argparse.ArgumentParser(description="Run Tau2-bench with GPT-5.5 and collect trajectories.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and create a dry-run manifest without running Tau2.")
    parser.add_argument("--domain", help="Override TAU2_DOMAIN.")
    parser.add_argument("--domains", nargs="+", help="Run multiple domains sequentially, e.g. airline retail telecom.")
    parser.add_argument("--benchmark", help="Override BENCHMARK.")
    parser.add_argument("--agent-model", help="Override AGENT_MODEL.")
    parser.add_argument("--user-model", help="Override USER_MODEL.")
    parser.add_argument("--agent-llm-args", help="JSON object passed to Tau2's agent LLM.")
    parser.add_argument("--user-llm-args", help="JSON object passed to Tau2's user LLM.")
    parser.add_argument("--num-tasks", type=int, help="Override NUM_TASKS.")
    parser.add_argument("--all-tasks", action="store_true", help="Run all tasks by omitting Tau2's --num-tasks option.")
    parser.add_argument("--task-ids", nargs="+", help="Run specific Tau2 task IDs.")
    parser.add_argument("--task-set-name", help="Override Tau2 --task-set-name.")
    parser.add_argument("--task-split-name", help="Override Tau2 --task-split-name.")
    parser.add_argument("--num-trials", "--num-rollouts", dest="num_trials", type=int, help="Rollouts per task; passed to Tau2 as --num-trials.")
    parser.add_argument("--output-dir", type=Path, help="Override OUTPUT_DIR.")
    parser.add_argument("--reasoning-effort", help="Override REASONING_EFFORT.")
    parser.add_argument("--temperature", help="Override TEMPERATURE.")
    parser.add_argument("--max-steps", type=int, help="Override MAX_STEPS.")
    parser.add_argument("--max-workers", type=int, help="Pass a Tau2 concurrency/worker option if supported by tau2 run --help.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing raw files in this run directory when collecting outputs.")
    parser.add_argument("--one-by-one", action="store_true", help="Run one Tau2 task ID per process and checkpoint after each task.")
    parser.add_argument("--save-to", help="Override Tau2 --save-to for a single-process run.")
    parser.add_argument("--auto-resume", action="store_true", help="Pass Tau2 --auto-resume when supported.")
    parser.add_argument("--run-id", help="Use a fixed run id, useful with --resume.")
    parser.add_argument("--progress-interval", type=float, help="Seconds between progress updates.")
    parser.add_argument("--simulations-dir", type=Path, help="Override TAU2_SIMULATIONS_DIR.")
    parser.add_argument("--tau2-command", help='Override TAU2_COMMAND, e.g. "uv run tau2".')
    parser.add_argument("--tau2-repo-dir", type=Path, help="Override TAU2_REPO_DIR for running tau2 from a checkout.")
    parser.add_argument("--tau2-use-uv", action="store_true", help='Run Tau2 as "uv run tau2" from TAU2_REPO_DIR.')
    args = parser.parse_args(argv)

    if args.domains:
        if args.domain:
            parser.error("use either --domain or --domains, not both")
        return run_multiple_domains(args, argv)

    env = prepare_tau2_env(os.environ)
    _apply_cli_overrides(env, args)
    config = build_run_config(env=env)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = config.run_dir / "raw"
    trajectories_dir = config.run_dir / "trajectories"
    benchmark_trajectories_dir = config.run_dir / "benchmark_trajectories"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    benchmark_trajectories_dir.mkdir(parents=True, exist_ok=True)

    tau2_invocation = resolve_tau2_invocation(env, config)
    help_text = get_tau2_run_help(tau2_invocation) if tau2_invocation.available else None
    command, ignored_options = build_tau2_command(
        config,
        help_text=help_text,
        command_prefix=tau2_invocation.command_prefix,
    )
    ignored_options = wrapper_ignored_options(config, ignored_options)

    manifest: dict[str, Any] = build_manifest(config, command, ignored_options, tau2_invocation, args.dry_run)
    existing_resume_raw_files = existing_raw_files_for_resume(config) if config.resume else []
    before_snapshot = snapshot_files(config.simulations_dir)
    manifest["simulations_before_count"] = len(before_snapshot)
    manifest["resume_raw_files_before_count"] = len(existing_resume_raw_files)

    if args.dry_run:
        if "OPENAI_API_KEY" not in env:
            print("Dry run: OPENAI_API_KEY is not set. A real run will require it.")
        if not tau2_invocation.available:
            print("Dry run: tau2 launcher is not available. Install tau2-bench or configure TAU2_COMMAND/TAU2_REPO_DIR.")
        write_json(config.run_dir / "manifest.json", manifest)
        print_run_report(config.run_dir, command, raw_count=0, trajectory_count=0, summary={})
        return 0

    if "OPENAI_API_KEY" not in env:
        print("ERROR: OPENAI_API_KEY is not set. Add it to your environment or .env file.", file=sys.stderr)
        return 2
    if not tau2_invocation.available:
        print(
            "ERROR: tau2 launcher is not available. Install tau2-bench or set TAU2_COMMAND='uv run tau2' "
            "and TAU2_REPO_DIR to your tau2-bench checkout.",
            file=sys.stderr,
        )
        return 2

    if config.one_by_one:
        return run_tau2_one_by_one(
            config=config,
            env=env,
            tau2_invocation=tau2_invocation,
            help_text=help_text,
            manifest=manifest,
            existing_resume_raw_files=existing_resume_raw_files,
            raw_dir=raw_dir,
            trajectories_dir=trajectories_dir,
            benchmark_trajectories_dir=benchmark_trajectories_dir,
        )

    print("Running:", _shell_join(command))
    completed = run_tau2_command_with_progress(
        command,
        env=env,
        cwd=str(tau2_invocation.cwd) if tau2_invocation.cwd else None,
        config=config,
        before_snapshot=before_snapshot,
    )
    (config.run_dir / "tau2.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (config.run_dir / "tau2.stderr.log").write_text(completed.stderr, encoding="utf-8")
    manifest["tau2_returncode"] = completed.returncode

    after_snapshot = snapshot_files(config.simulations_dir)
    changed_files = changed_since(before_snapshot, after_snapshot)
    copied_raw_files = copy_raw_files(changed_files, source_root=config.simulations_dir, raw_dir=raw_dir)
    if completed.returncode != 0:
        error_raw_path = raw_dir / "tau2_process_error.json"
        write_json(
            error_raw_path,
            tau2_error_raw_record(
                run_id=config.run_id,
                config=config,
                returncode=completed.returncode,
                stderr=completed.stderr,
                stdout=completed.stdout,
                command=command,
            ),
        )
        copied_raw_files.append(error_raw_path)
    raw_files = _dedupe_paths([*existing_resume_raw_files, *copied_raw_files])
    manifest["simulations_after_count"] = len(after_snapshot)
    manifest["raw_files"] = [str(path.relative_to(config.run_dir)) for path in raw_files]

    trajectories = collect_raw_files(
        raw_files=raw_files,
        trajectories_dir=trajectories_dir,
        jsonl_path=config.run_dir / "trajectories.jsonl",
        run_id=config.run_id,
        defaults={
            "domain": config.domain,
            "agent_model": config.agent_model,
            "user_model": config.user_model,
        },
        raw_root=config.run_dir,
    )
    benchmark_records = write_benchmark_records(
        trajectories=trajectories,
        output_dir=benchmark_trajectories_dir,
        jsonl_path=config.run_dir / "benchmark_trajectories.jsonl",
        benchmark=config.benchmark,
    )
    summary = summarize_trajectories(trajectories)
    summary["benchmark_task_count"] = len(benchmark_records)
    summary["benchmark_trajectory_count"] = sum(
        len(record.get("trajectories", [])) for record in benchmark_records
    )
    write_json(config.run_dir / "summary.json", summary)
    write_json(config.run_dir / "manifest.json", manifest)

    print_run_report(config.run_dir, command, len(raw_files), len(trajectories), summary)
    if completed.returncode != 0:
        print("ERROR: tau2 exited with a non-zero status. See tau2.stderr.log.", file=sys.stderr)
    return completed.returncode


def run_tau2_one_by_one(
    *,
    config: RunConfig,
    env: dict[str, str],
    tau2_invocation: Tau2Invocation,
    help_text: str | None,
    manifest: dict[str, Any],
    existing_resume_raw_files: list[Path],
    raw_dir: Path,
    trajectories_dir: Path,
    benchmark_trajectories_dir: Path,
) -> int:
    task_ids = config.task_ids or load_tau2_task_ids(tau2_invocation, config)
    if not task_ids:
        print("ERROR: no Tau2 task IDs found for one-by-one mode.", file=sys.stderr)
        write_json(config.run_dir / "manifest.json", manifest)
        return 2

    manifest["one_by_one_task_ids"] = task_ids
    manifest["one_by_one_total_tasks"] = len(task_ids)
    manifest["one_by_one_max_workers"] = one_by_one_worker_count(config)
    raw_files = _dedupe_paths(existing_resume_raw_files)
    run_records: list[dict[str, Any]] = []
    progress = ProgressPrinter(label="Tau2 tasks", total=len(task_ids), interval_seconds=config.progress_interval)
    progress.update(0, extra=f"starting one-by-one mode; workers={one_by_one_worker_count(config)}", force=True)
    final_returncode = 0
    completed_count = 0

    with ThreadPoolExecutor(max_workers=one_by_one_worker_count(config)) as executor:
        futures = {
            executor.submit(
                _run_one_tau2_task,
                index,
                len(task_ids),
                task_id,
                config=config,
                env=env,
                tau2_invocation=tau2_invocation,
                help_text=help_text,
                raw_dir=raw_dir,
            ): task_id
            for index, task_id in enumerate(task_ids, start=1)
        }
        for future in as_completed(futures):
            record = future.result()
            completed_count += 1
            copied = record.get("raw_file_paths", [])
            raw_files = _dedupe_paths([*raw_files, *copied])
            if record.get("returncode", 0) != 0 and final_returncode == 0:
                final_returncode = int(record.get("returncode") or 1)
            run_records.append(_serializable_one_by_one_record(record, config.run_dir))
            manifest["one_by_one_runs"] = run_records
            _write_tau2_outputs(config, raw_files, trajectories_dir, benchmark_trajectories_dir, manifest)
            status = "skipped" if record.get("skipped") else f"returncode={record.get('returncode')}"
            progress.update(
                completed_count,
                extra=f"task={record.get('task_id')}; {status}; raw files={len(raw_files)}",
                force=True,
            )

    manifest["one_by_one_runs"] = run_records
    manifest["raw_files"] = [str(path.relative_to(config.run_dir)) for path in raw_files]
    write_json(config.run_dir / "manifest.json", manifest)
    summary = _write_tau2_outputs(config, raw_files, trajectories_dir, benchmark_trajectories_dir, manifest)
    print_run_report(config.run_dir, manifest.get("command", []), len(raw_files), summary.get("total_trajectories", 0), summary)
    if final_returncode != 0:
        print("ERROR: one or more Tau2 task processes exited with a non-zero status. See tau2_task_*.stderr.log.", file=sys.stderr)
    return final_returncode


def one_by_one_worker_count(config: RunConfig) -> int:
    return max(1, config.max_workers or 1)


def build_one_by_one_task_config(config: RunConfig, task_id: str) -> RunConfig:
    return dataclasses.replace(
        config,
        all_tasks=False,
        task_ids=[task_id],
        num_tasks=1,
        save_to=one_by_one_save_to(config, task_id),
        auto_resume=True,
        max_workers=1,
    )


def wrapper_ignored_options(config: RunConfig, ignored_options: list[str]) -> list[str]:
    if not config.one_by_one:
        return ignored_options
    return [option for option in ignored_options if option != "TAU2_MAX_WORKERS"]


def _run_one_tau2_task(
    index: int,
    total: int,
    task_id: str,
    *,
    config: RunConfig,
    env: dict[str, str],
    tau2_invocation: Tau2Invocation,
    help_text: str | None,
    raw_dir: Path,
) -> dict[str, Any]:
    save_to = one_by_one_save_to(config, task_id)
    task_raw_root = raw_dir / save_to
    if config.resume and task_raw_root.exists() and any(task_raw_root.rglob("*.json")):
        existing = _json_like_files(task_raw_root)
        return {
            "task_id": task_id,
            "skipped": True,
            "save_to": save_to,
            "returncode": 0,
            "raw_file_paths": existing,
            "raw_files": [str(path.relative_to(config.run_dir)) for path in existing],
        }

    task_config = build_one_by_one_task_config(config, task_id)
    task_command, task_ignored = build_tau2_command(
        task_config,
        help_text=help_text,
        command_prefix=tau2_invocation.command_prefix,
    )
    print(f"Starting Tau2 task {index}/{total}:", _shell_join(task_command))
    completed = run_tau2_command_capture(
        task_command,
        env=env,
        cwd=str(tau2_invocation.cwd) if tau2_invocation.cwd else None,
    )
    safe_task = _safe_run_part(task_id)
    (config.run_dir / f"tau2_task_{safe_task}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (config.run_dir / f"tau2_task_{safe_task}.stderr.log").write_text(completed.stderr, encoding="utf-8")

    copied = copy_one_by_one_task_raw(config=config, task_id=task_id, raw_dir=raw_dir)
    if completed.returncode != 0:
        error_raw_path = raw_dir / f"tau2_process_error_{safe_task}.json"
        write_json(
            error_raw_path,
            tau2_error_raw_record(
                run_id=config.run_id,
                config=task_config,
                returncode=completed.returncode,
                stderr=completed.stderr,
                stdout=completed.stdout,
                command=task_command,
            ),
        )
        copied.append(error_raw_path)

    return {
        "task_id": task_id,
        "save_to": save_to,
        "command": task_command,
        "ignored_options": task_ignored,
        "returncode": completed.returncode,
        "raw_file_paths": copied,
        "raw_files": [str(path.relative_to(config.run_dir)) for path in copied],
    }


def _serializable_one_by_one_record(record: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    payload = {key: value for key, value in record.items() if key != "raw_file_paths"}
    if "command" in payload:
        payload["command"] = _redact_command(payload["command"])
    if "raw_files" not in payload:
        payload["raw_files"] = [str(path.relative_to(run_dir)) for path in record.get("raw_file_paths", [])]
    return payload


def run_tau2_command_capture(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: str | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=cwd,
    )


def copy_one_by_one_task_raw(*, config: RunConfig, task_id: str, raw_dir: Path) -> list[Path]:
    source_root = config.simulations_dir / one_by_one_save_to(config, task_id)
    target_root = raw_dir / one_by_one_save_to(config, task_id)
    if not source_root.exists():
        return []
    copied: list[Path] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _json_like_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})


def one_by_one_save_to(config: RunConfig, task_id: str) -> str:
    return f"{config.run_id}/{_safe_run_part(task_id)}"


def load_tau2_task_ids(invocation: Tau2Invocation, config: RunConfig) -> list[str]:
    command = [*tau2_python_command(invocation), "-c", _task_ids_probe_code(config)]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(invocation.cwd) if invocation.cwd else None,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not load Tau2 task IDs: {completed.stderr.strip() or completed.stdout.strip()}")
    return json.loads(completed.stdout.strip())


def tau2_python_command(invocation: Tau2Invocation) -> list[str]:
    prefix = invocation.command_prefix
    if len(prefix) >= 2 and prefix[0] == "uv" and prefix[-1] in {"tau2", "tau2.exe"}:
        return [*prefix[:-1], "python"]
    if len(prefix) == 1:
        executable = Path(prefix[0])
        if executable.name.lower() in {"tau2", "tau2.exe"}:
            sibling = executable.with_name("python.exe" if os.name == "nt" else "python")
            if sibling.exists():
                return [str(sibling)]
    return [sys.executable]


def _task_ids_probe_code(config: RunConfig) -> str:
    task_set_name = config.task_set_name or config.domain
    task_split_name = config.task_split_name or "base"
    num_tasks = "None" if config.all_tasks else repr(config.num_tasks)
    return (
        "import json\n"
        "from tau2.run import get_tasks\n"
        f"tasks = get_tasks({task_set_name!r}, task_split_name={task_split_name!r}, num_tasks={num_tasks})\n"
        "print(json.dumps([str(task.id) for task in tasks], ensure_ascii=False))\n"
    )


def _write_tau2_outputs(
    config: RunConfig,
    raw_files: list[Path],
    trajectories_dir: Path,
    benchmark_trajectories_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest["raw_files"] = [str(path.relative_to(config.run_dir)) for path in raw_files]
    trajectories = collect_raw_files(
        raw_files=raw_files,
        trajectories_dir=trajectories_dir,
        jsonl_path=config.run_dir / "trajectories.jsonl",
        run_id=config.run_id,
        defaults={
            "domain": config.domain,
            "agent_model": config.agent_model,
            "user_model": config.user_model,
        },
        raw_root=config.run_dir,
    )
    benchmark_records = write_benchmark_records(
        trajectories=trajectories,
        output_dir=benchmark_trajectories_dir,
        jsonl_path=config.run_dir / "benchmark_trajectories.jsonl",
        benchmark=config.benchmark,
    )
    summary = summarize_trajectories(trajectories)
    summary["benchmark_task_count"] = len(benchmark_records)
    summary["benchmark_trajectory_count"] = sum(
        len(record.get("trajectories", [])) for record in benchmark_records
    )
    write_json(config.run_dir / "summary.json", summary)
    write_json(config.run_dir / "manifest.json", manifest)
    return summary


def run_tau2_command_with_progress(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: str | None,
    config: RunConfig,
    before_snapshot: dict[str, tuple[int, int]],
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
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
    printer = ProgressPrinter(label="Tau2", total=total, interval_seconds=config.progress_interval)
    printer.update(0, extra="raw changed=0", force=True)
    poll_interval = 1.0 if config.progress_interval <= 0 else min(5.0, config.progress_interval)

    while process.poll() is None:
        time.sleep(poll_interval)
        changed_count = len(changed_since(before_snapshot, snapshot_files(config.simulations_dir)))
        completed_units = changed_count if total is None else min(changed_count, total)
        printer.update(completed_units, extra=f"raw changed={changed_count}")

    returncode = process.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    changed_count = len(changed_since(before_snapshot, snapshot_files(config.simulations_dir)))
    completed_units = changed_count if total is None else min(changed_count, total)
    printer.update(completed_units, extra=f"finished returncode={returncode}; raw changed={changed_count}", force=True)
    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


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


def existing_raw_files_for_resume(config: RunConfig) -> list[Path]:
    raw_dir = config.run_dir / "raw"
    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})


def tau2_error_raw_record(
    *,
    run_id: str,
    config: RunConfig,
    returncode: int,
    stderr: str,
    stdout: str,
    command: list[str],
) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "task_id": f"tau2_error_{run_id}",
        "domain": config.domain,
        "agent_model": config.agent_model,
        "user_model": config.user_model,
        "success": None,
        "score": None,
        "task": {
            "instruction": (
                f"Tau2 command failed while running domain={config.domain}, "
                f"agent_model={config.agent_model}, user_model={config.user_model}."
            )
        },
        "messages": [
            {
                "role": "system",
                "content": "Tau2 process error captured by MetaCoreBench runner.",
            },
            {
                "role": "user",
                "content": _shell_join(_redact_command(command)),
            },
            {
                "role": "assistant",
                "content": stdout,
            },
        ],
        "error": {
            "type": "Tau2ProcessError",
            "returncode": returncode,
            "stderr": stderr,
        },
        "raw": {
            "run_id": run_id,
            "command": _redact_command(command),
            "stdout": stdout,
            "stderr": stderr,
        },
    }


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def prepare_tau2_env(base_env: dict[str, str] | os._Environ[str]) -> dict[str, str]:
    env = dict(base_env)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "0")
    # Tau2 uses LiteLLM, whose OpenAI-compatible endpoint variable is
    # OPENAI_API_BASE; keep the OpenAI SDK variable working as well.
    if env.get("OPENAI_BASE_URL") and not env.get("OPENAI_API_BASE"):
        env["OPENAI_API_BASE"] = env["OPENAI_BASE_URL"]
    return env


def get_tau2_run_help(invocation: Tau2Invocation) -> str | None:
    try:
        completed = subprocess.run(
            [*invocation.command_prefix, "run", "--help"],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(invocation.cwd) if invocation.cwd else None,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return output or None


def resolve_tau2_invocation(env: dict[str, str], config: RunConfig) -> Tau2Invocation:
    repo_dir = config.tau2_repo_dir
    command_text = env.get("TAU2_COMMAND")
    use_uv = _env_bool(env, "TAU2_USE_UV") or command_text == "uv run tau2"

    if command_text:
        prefix = _split_command(command_text)
        return Tau2Invocation(
            command_prefix=prefix,
            cwd=repo_dir,
            available=_launcher_available(prefix, repo_dir),
            source="TAU2_COMMAND",
        )

    if use_uv:
        cwd = repo_dir or _default_tau2_repo_dir()
        prefix = ["uv", "run", "tau2"]
        return Tau2Invocation(
            command_prefix=prefix,
            cwd=cwd,
            available=_launcher_available(prefix, cwd),
            source="TAU2_USE_UV",
        )

    tau2_path = find_tau2_executable()
    if tau2_path:
        return Tau2Invocation(
            command_prefix=[tau2_path],
            cwd=None,
            available=True,
            source="PATH",
        )

    default_repo = repo_dir or _default_tau2_repo_dir()
    if default_repo and shutil.which("uv"):
        prefix = ["uv", "run", "tau2"]
        return Tau2Invocation(
            command_prefix=prefix,
            cwd=default_repo,
            available=_launcher_available(prefix, default_repo),
            source="auto-uv",
        )

    return Tau2Invocation(
        command_prefix=["tau2"],
        cwd=repo_dir,
        available=False,
        source="missing",
    )


def find_tau2_executable() -> str | None:
    path_tau2 = shutil.which("tau2")
    if path_tau2:
        return path_tau2
    local_tau2 = Path(".venv") / "Scripts" / "tau2.exe"
    if local_tau2.exists():
        return str(local_tau2)
    return None


def _default_tau2_repo_dir() -> Path | None:
    candidate = Path(".external") / "tau2-bench"
    return candidate if candidate.exists() else None


def _launcher_available(prefix: list[str], cwd: Path | None) -> bool:
    if not prefix:
        return False
    if cwd is not None and not cwd.exists():
        return False
    return shutil.which(prefix[0]) is not None or Path(prefix[0]).exists()


def _split_command(command_text: str) -> list[str]:
    parts = shlex.split(command_text, posix=os.name != "nt")
    if not parts:
        raise ValueError("TAU2_COMMAND cannot be empty.")
    return parts


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
    changed = []
    for relative, metadata in after.items():
        if before.get(relative) != metadata:
            changed.append(Path(relative))
    return sorted(changed)


def copy_raw_files(changed_files: list[Path], source_root: Path, raw_dir: Path) -> list[Path]:
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


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def build_manifest(
    config: RunConfig,
    command: list[str],
    ignored_options: list[str],
    tau2_invocation: Tau2Invocation,
    dry_run: bool,
) -> dict[str, Any]:
    safe_command = _redact_command(command)
    return {
        "run_id": config.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "command": safe_command,
        "command_string": _shell_join(safe_command),
        "ignored_options": ignored_options,
        "config": _jsonable_config(config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "tau2_launcher": {
                "command_prefix": tau2_invocation.command_prefix,
                "cwd": str(tau2_invocation.cwd) if tau2_invocation.cwd else None,
                "available": tau2_invocation.available,
                "source": tau2_invocation.source,
            },
        },
    }


def print_run_report(
    run_dir: Path,
    command: list[str],
    raw_count: int,
    trajectory_count: int,
    summary: dict[str, Any],
) -> None:
    print("Output directory:", run_dir)
    print("Tau2 command:", _shell_join(_redact_command(command)))
    print("Raw files:", raw_count)
    print("Trajectories:", trajectory_count)
    if summary:
        print("Summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _add_required_option(
    command: list[str],
    help_text: str | None,
    aliases: tuple[str, ...],
    value: str,
    env_name: str,
) -> None:
    flag = _choose_flag(help_text, aliases)
    if flag is None:
        raise ValueError(f"tau2 run --help did not show a supported option for {env_name}: {aliases}")
    command.extend([flag, value])


def _add_optional_option(
    command: list[str],
    help_text: str | None,
    aliases: tuple[str, ...],
    value: str | None,
    env_name: str,
    ignored: list[str],
) -> None:
    if value is None:
        return
    flag = _choose_flag(help_text, aliases, default_when_unknown=False)
    if flag is None:
        ignored.append(env_name)
        return
    command.extend([flag, value])


def _add_multi_value_option(
    command: list[str],
    help_text: str | None,
    aliases: tuple[str, ...],
    values: list[str] | None,
    env_name: str,
    ignored: list[str],
) -> None:
    if not values:
        return
    flag = _choose_flag(help_text, aliases, default_when_unknown=False)
    if flag is None:
        ignored.append(env_name)
        return
    command.append(flag)
    command.extend(values)


def _add_flag_option(
    command: list[str],
    help_text: str | None,
    aliases: tuple[str, ...],
    env_name: str,
    ignored: list[str],
) -> None:
    flag = _choose_flag(help_text, aliases, default_when_unknown=False)
    if flag is None:
        ignored.append(env_name)
        return
    command.append(flag)


def _choose_flag(
    help_text: str | None,
    aliases: tuple[str, ...],
    default_when_unknown: bool = True,
) -> str | None:
    if help_text is None:
        return aliases[0] if default_when_unknown else None
    for alias in aliases:
        if alias in help_text:
            return alias
    return None


def _apply_cli_overrides(env: dict[str, str], args: argparse.Namespace) -> None:
    overrides = {
        "TAU2_DOMAIN": args.domain,
        "BENCHMARK": args.benchmark,
        "AGENT_MODEL": args.agent_model,
        "USER_MODEL": args.user_model,
        "AGENT_LLM_ARGS": args.agent_llm_args,
        "USER_LLM_ARGS": args.user_llm_args,
        "NUM_TASKS": args.num_tasks,
        "TAU2_ALL_TASKS": "true" if args.all_tasks else None,
        "TAU2_TASK_IDS": " ".join(args.task_ids) if args.task_ids else None,
        "TAU2_TASK_SET_NAME": args.task_set_name,
        "TAU2_TASK_SPLIT_NAME": args.task_split_name,
        "NUM_TRIALS": args.num_trials,
        "OUTPUT_DIR": str(args.output_dir) if args.output_dir else None,
        "REASONING_EFFORT": args.reasoning_effort,
        "TEMPERATURE": args.temperature,
        "MAX_STEPS": args.max_steps,
        "TAU2_MAX_WORKERS": args.max_workers,
        "TAU2_RESUME": "true" if args.resume else None,
        "TAU2_ONE_BY_ONE": "true" if args.one_by_one else None,
        "TAU2_SAVE_TO": args.save_to,
        "TAU2_AUTO_RESUME": "true" if args.auto_resume else None,
        "TAU2_RUN_ID": args.run_id,
        "TAU2_PROGRESS_INTERVAL": args.progress_interval,
        "TAU2_SIMULATIONS_DIR": str(args.simulations_dir) if args.simulations_dir else None,
        "TAU2_COMMAND": args.tau2_command,
        "TAU2_REPO_DIR": str(args.tau2_repo_dir) if args.tau2_repo_dir else None,
        "TAU2_USE_UV": "true" if args.tau2_use_uv else None,
    }
    for key, value in overrides.items():
        if value is not None:
            env[key] = str(value)


def _jsonable_config(config: RunConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in ("output_root", "run_dir", "simulations_dir", "tau2_repo_dir"):
        if data[key] is not None:
            data[key] = str(data[key])
    data["agent_llm_args"] = _redact_llm_args(data["agent_llm_args"])
    data["user_llm_args"] = _redact_llm_args(data["user_llm_args"])
    return data


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _env_optional_int(env: dict[str, str], key: str) -> int | None:
    value = env.get(key)
    if value is None or value == "":
        return None
    return int(value)


def _env_bool(env: dict[str, str], key: str) -> bool:
    value = env.get(key)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_task_ids(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    task_ids = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    return task_ids or None


def _safe_run_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in ".-_" else "_" for char in value).strip("_") or "unknown"


def _parse_json_object(value: str | None, env_name: str) -> dict[str, Any]:
    if value is None or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{env_name} must be a JSON object")
    return parsed


def _compact_json(value: dict[str, Any]) -> str | None:
    if not value:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _redact_llm_args(value: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(value)
    for key in ("api_key", "api-key", "token", "access_token"):
        if key in redacted and redacted[key]:
            redacted[key] = "***REDACTED***"
    return redacted


def _redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, part in enumerate(redacted):
        if part in {"--agent-llm-args", "--agent_llm_args", "--user-llm-args", "--user_llm_args"}:
            if index + 1 >= len(redacted):
                continue
            try:
                args = json.loads(redacted[index + 1])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(args, dict):
                redacted[index + 1] = _compact_json(_redact_llm_args(args)) or "{}"
    return redacted


def _shell_join(command: list[str]) -> str:
    return " ".join(_quote_arg(part) for part in command)


def _quote_arg(part: str) -> str:
    if not part or any(char.isspace() for char in part):
        return '"' + part.replace('"', '\\"') + '"'
    return part


if __name__ == "__main__":
    raise SystemExit(main())
