"""Run OSWorld-Verified through the OSWorld repo and export trajectories."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_trajectories import collect_raw_files, summarize_trajectories, write_benchmark_records
from scripts.runner_common import (
    build_manifest,
    copy_tree_files,
    env_bool,
    env_int,
    env_optional_int,
    json_like_files,
    load_dotenv_file,
    now_timestamp,
    prepare_subprocess_env,
    print_run_report,
    run_command,
    safe_run_part,
    shell_join,
    write_json,
)


@dataclass
class OSWorldConfig:
    run_id: str
    benchmark: str
    output_root: Path
    run_dir: Path
    repo_dir: Path
    model: str
    provider_name: str
    observation_type: str
    client_password: str
    headless: bool
    max_steps: int
    sleep_after_execution: int
    result_dir: Path
    path_to_vm: str | None
    domain: str | None
    example_id: str | None
    test_all_meta_path: str | None
    use_multienv: bool
    num_envs: int


def build_osworld_config(env: dict[str, str] | None = None, now: str | None = None) -> OSWorldConfig:
    env = env or os.environ
    timestamp = now or now_timestamp()
    benchmark = env.get("BENCHMARK", "osworld-verified")
    model = env.get("AGENT_MODEL", "openai/gpt-5.4")
    output_root = Path(env.get("OUTPUT_DIR", "outputs/runs"))
    run_id = f"{timestamp}_{safe_run_part(model)}_{safe_run_part(benchmark)}"
    run_dir = output_root / run_id
    return OSWorldConfig(
        run_id=run_id,
        benchmark=benchmark,
        output_root=output_root,
        run_dir=run_dir,
        repo_dir=Path(env.get("OSWORLD_REPO_DIR", ".external/OSWorld")),
        model=model,
        provider_name=env.get("OSWORLD_PROVIDER_NAME", "docker"),
        observation_type=env.get("OSWORLD_OBSERVATION_TYPE", "screenshot"),
        client_password=env.get("OSWORLD_CLIENT_PASSWORD", "password"),
        headless=env_bool(env, "OSWORLD_HEADLESS", True),
        max_steps=env_optional_int(env, "MAX_STEPS") or 15,
        sleep_after_execution=env_int(env, "OSWORLD_SLEEP_AFTER_EXECUTION", 3),
        result_dir=run_dir / "osworld_results",
        path_to_vm=env.get("OSWORLD_PATH_TO_VM") or None,
        domain=env.get("OSWORLD_DOMAIN") or None,
        example_id=env.get("OSWORLD_EXAMPLE_ID") or None,
        test_all_meta_path=env.get("OSWORLD_TEST_ALL_META_PATH") or None,
        use_multienv=env_bool(env, "OSWORLD_USE_MULTIENV", False),
        num_envs=env_int(env, "OSWORLD_NUM_ENVS", 1),
    )


def build_osworld_command(config: OSWorldConfig, help_text: str | None = None) -> list[str]:
    script = config.repo_dir / ("scripts/python/run_multienv.py" if config.use_multienv else "run.py")
    command = [
        sys.executable,
        str(script),
        "--provider_name",
        config.provider_name,
        "--observation_type",
        config.observation_type,
        "--model",
        config.model,
        "--sleep_after_execution",
        str(config.sleep_after_execution),
        "--max_steps",
        str(config.max_steps),
        "--result_dir",
        str(config.result_dir),
    ]
    if _supports_flag(help_text, "--client_password"):
        command.extend(["--client_password", config.client_password])
    if config.headless:
        command.append("--headless")
    if config.path_to_vm:
        command.extend(["--path_to_vm", config.path_to_vm])
    if config.domain:
        command.extend(["--domain", config.domain])
    if config.example_id and _supports_flag(help_text, "--example_id"):
        command.extend(["--example_id", config.example_id])
    if config.test_all_meta_path:
        command.extend(["--test_all_meta_path", config.test_all_meta_path])
    if config.use_multienv:
        command.extend(["--num_envs", str(config.num_envs)])
    return command


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file()
    parser = argparse.ArgumentParser(description="Run OSWorld-Verified and collect trajectories.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and create manifest without running OSWorld.")
    parser.add_argument("--benchmark", help="Override BENCHMARK.")
    parser.add_argument("--repo-dir", type=Path, help="Override OSWORLD_REPO_DIR.")
    parser.add_argument("--model", help="Override AGENT_MODEL.")
    parser.add_argument("--provider-name", help="Override OSWORLD_PROVIDER_NAME.")
    parser.add_argument("--observation-type", help="Override OSWORLD_OBSERVATION_TYPE.")
    parser.add_argument("--client-password", help="Override OSWORLD_CLIENT_PASSWORD.")
    parser.add_argument("--max-steps", type=int, help="Override MAX_STEPS.")
    parser.add_argument("--domain", help="Optional OSWorld domain filter.")
    parser.add_argument("--example-id", help="Optional OSWorld example id filter.")
    parser.add_argument("--test-all-meta-path", help="Optional OSWorld meta JSON path.")
    parser.add_argument("--use-multienv", action="store_true", help="Use scripts/python/run_multienv.py.")
    parser.add_argument("--num-envs", type=int, help="Override OSWORLD_NUM_ENVS.")
    parser.add_argument("--path-to-vm", help="Optional VM path for vmware/virtualbox providers.")
    parser.add_argument("--output-dir", type=Path, help="Override OUTPUT_DIR.")
    args = parser.parse_args(argv)

    env = prepare_subprocess_env(os.environ)
    _apply_overrides(env, args)
    config = build_osworld_config(env=env)
    Path("logs").mkdir(exist_ok=True)
    help_text = get_osworld_help(config)
    command = build_osworld_command(config, help_text=help_text)

    raw_dir = config.run_dir / "raw"
    trajectories_dir = config.run_dir / "trajectories"
    benchmark_dir = config.run_dir / "benchmark_trajectories"
    for directory in (raw_dir, trajectories_dir, benchmark_dir, config.result_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        run_id=config.run_id,
        benchmark=config.benchmark,
        command=command,
        config=asdict(config),
        dry_run=args.dry_run,
        executable_path=str(config.repo_dir),
    )

    if args.dry_run:
        if not _osworld_script_exists(config):
            print("Dry run: OSWorld run script not found. Clone https://github.com/xlang-ai/OSWorld to OSWORLD_REPO_DIR.")
        write_json(config.run_dir / "manifest.json", manifest)
        print_run_report(config.run_dir, command, raw_count=0, trajectory_count=0, summary={})
        return 0

    if not _osworld_script_exists(config):
        print("ERROR: OSWorld run script not found. Set OSWORLD_REPO_DIR to a valid checkout.", file=sys.stderr)
        return 2
    if "OPENAI_API_KEY" not in env:
        print("ERROR: OPENAI_API_KEY is not set for OSWorld model calls.", file=sys.stderr)
        return 2

    print("Running:", shell_join(command))
    completed = run_command(command, env=env, run_dir=config.run_dir, log_prefix="osworld")
    manifest["returncode"] = completed.returncode

    copied_raw_files = copy_tree_files(config.result_dir, raw_dir)
    manifest["raw_files"] = [str(path.relative_to(config.run_dir)) for path in copied_raw_files]
    summary = _collect_outputs(config, copied_raw_files, trajectories_dir, benchmark_dir)
    write_json(config.run_dir / "summary.json", summary)
    write_json(config.run_dir / "manifest.json", manifest)
    print_run_report(config.run_dir, command, len(copied_raw_files), summary.get("total_trajectories", 0), summary)
    if completed.returncode != 0:
        print("ERROR: OSWorld exited with a non-zero status. See osworld.stderr.log.", file=sys.stderr)
    return completed.returncode


def _collect_outputs(
    config: OSWorldConfig,
    copied_raw_files: list[Path],
    trajectories_dir: Path,
    benchmark_dir: Path,
) -> dict[str, Any]:
    raw_files = [path for path in copied_raw_files if path.suffix.lower() in {".json", ".jsonl"}]
    if not raw_files:
        raw_files = json_like_files(config.run_dir / "raw")
    trajectories = collect_raw_files(
        raw_files=raw_files,
        trajectories_dir=trajectories_dir,
        jsonl_path=config.run_dir / "trajectories.jsonl",
        run_id=config.run_id,
        defaults={
            "domain": config.domain or "osworld",
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


def _osworld_script_exists(config: OSWorldConfig) -> bool:
    script = config.repo_dir / ("scripts/python/run_multienv.py" if config.use_multienv else "run.py")
    return script.exists()


def get_osworld_help(config: OSWorldConfig) -> str | None:
    if not _osworld_script_exists(config):
        return None
    script = config.repo_dir / ("scripts/python/run_multienv.py" if config.use_multienv else "run.py")
    try:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return output or None


def _supports_flag(help_text: str | None, flag: str) -> bool:
    return help_text is None or flag in help_text


def _apply_overrides(env: dict[str, str], args: argparse.Namespace) -> None:
    overrides = {
        "BENCHMARK": args.benchmark,
        "OSWORLD_REPO_DIR": str(args.repo_dir) if args.repo_dir else None,
        "AGENT_MODEL": args.model,
        "OSWORLD_PROVIDER_NAME": args.provider_name,
        "OSWORLD_OBSERVATION_TYPE": args.observation_type,
        "OSWORLD_CLIENT_PASSWORD": args.client_password,
        "MAX_STEPS": args.max_steps,
        "OSWORLD_DOMAIN": args.domain,
        "OSWORLD_EXAMPLE_ID": args.example_id,
        "OSWORLD_TEST_ALL_META_PATH": args.test_all_meta_path,
        "OSWORLD_USE_MULTIENV": "1" if args.use_multienv else None,
        "OSWORLD_NUM_ENVS": args.num_envs,
        "OSWORLD_PATH_TO_VM": args.path_to_vm,
        "OUTPUT_DIR": str(args.output_dir) if args.output_dir else None,
    }
    for key, value in overrides.items():
        if value is not None:
            env[key] = str(value)


if __name__ == "__main__":
    raise SystemExit(main())
