import json
import os
import tempfile
from pathlib import Path
import unittest

from scripts.run_tau2_gpt55 import (
    build_one_by_one_task_config,
    build_run_config,
    build_tau2_command,
    one_by_one_save_to,
    one_by_one_worker_count,
    prepare_tau2_env,
    resolve_tau2_invocation,
    tau2_error_raw_record,
    wrapper_ignored_options,
)


class RunTau2Gpt55Tests(unittest.TestCase):
    def test_builds_default_smoke_test_command_from_environment(self):
        env = {
            "AGENT_MODEL": "gpt-5.5",
            "USER_MODEL": "gpt-5.5",
            "TAU2_DOMAIN": "airline",
            "NUM_TASKS": "2",
            "NUM_TRIALS": "1",
            "REASONING_EFFORT": "medium",
            "TEMPERATURE": "0",
            "OUTPUT_DIR": "outputs/runs",
        }

        config = build_run_config(env=env, now="2026-06-24_120000")
        command, ignored = build_tau2_command(
            config,
            help_text="""
Usage: tau2 run [OPTIONS]
  --domain TEXT
  --agent-llm TEXT
  --user-llm TEXT
  --num-trials INTEGER
  --num-tasks INTEGER
  --reasoning-effort TEXT
  --temperature FLOAT
  --max-steps INTEGER
""",
        )

        self.assertEqual(config.run_id, "2026-06-24_120000_gpt-5.5_airline")
        self.assertEqual(config.benchmark, "tau2-bench")
        self.assertFalse(config.resume)
        self.assertIsNone(config.max_workers)
        self.assertEqual(config.progress_interval, 30.0)
        self.assertEqual(command[:2], ["tau2", "run"])
        self.assertIn("--domain", command)
        self.assertIn("airline", command)
        self.assertIn("--agent-llm", command)
        self.assertIn("gpt-5.5", command)
        self.assertIn("--user-llm", command)
        self.assertIn("--num-tasks", command)
        self.assertIn("2", command)
        self.assertEqual(ignored, [])

    def test_ignores_max_steps_when_cli_help_does_not_support_it(self):
        env = dict(os.environ)
        env.update({"MAX_STEPS": "7", "OUTPUT_DIR": "outputs/runs"})
        config = build_run_config(env=env, now="2026-06-24_120000")

        command, ignored = build_tau2_command(config, help_text="--domain\n--agent-llm\n--user-llm\n--num-trials\n--num-tasks")

        self.assertNotIn("--max-steps", command)
        self.assertIn("MAX_STEPS", ignored)

    def test_all_tasks_omits_num_tasks_option(self):
        env = {
            "TAU2_ALL_TASKS": "true",
            "NUM_TASKS": "2",
            "OUTPUT_DIR": "outputs/runs",
        }
        config = build_run_config(env=env, now="2026-06-24_120000")

        command, ignored = build_tau2_command(
            config,
            help_text="--domain\n--agent-llm\n--user-llm\n--num-trials\n--num-tasks",
        )

        self.assertTrue(config.all_tasks)
        self.assertNotIn("--num-tasks", command)
        self.assertNotIn("--num_tasks", command)
        self.assertNotIn("NUM_TASKS", ignored)

    def test_prepares_utf8_environment_for_tau2_subprocess(self):
        env = prepare_tau2_env(
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://proxy.example/v1",
            }
        )

        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONLEGACYWINDOWSSTDIO"], "0")
        self.assertEqual(env["OPENAI_API_BASE"], "https://proxy.example/v1")

    def test_preserves_explicit_litellm_api_base(self):
        env = prepare_tau2_env(
            {
                "OPENAI_BASE_URL": "https://sdk.example/v1",
                "OPENAI_API_BASE": "https://litellm.example/v1",
            }
        )

        self.assertEqual(env["OPENAI_API_BASE"], "https://litellm.example/v1")

    def test_forwards_separate_agent_and_user_llm_args_without_exposing_keys(self):
        env = {
            "AGENT_MODEL": "openai/qwen3.6-27b",
            "USER_MODEL": "openai/gpt-5.5",
            "AGENT_LLM_ARGS": json.dumps({"api_base": "http://127.0.0.1:8000/v1", "api_key": "local-key"}),
            "USER_LLM_ARGS": json.dumps({"api_base": "https://user.example/v1", "api_key": "remote-key"}),
            "OUTPUT_DIR": "outputs/runs",
        }
        config = build_run_config(env=env, now="2026-06-24_120000")
        command, ignored = build_tau2_command(
            config,
            help_text=(
                "--domain\n--agent-llm\n--user-llm\n--agent-llm-args\n"
                "--user-llm-args\n--num-trials\n--num-tasks\n"
                "--reasoning-effort\n--temperature"
            ),
        )

        self.assertEqual(ignored, [])
        self.assertIn("--agent-llm-args", command)
        self.assertIn("--user-llm-args", command)
        command_text = " ".join(command)
        self.assertIn('"api_base":"http://127.0.0.1:8000/v1"', command_text)
        self.assertIn('"api_base":"https://user.example/v1"', command_text)

        from scripts.run_tau2_gpt55 import _redact_command

        safe_command = json.dumps(_redact_command(command))
        self.assertNotIn("local-key", safe_command)
        self.assertNotIn("remote-key", safe_command)
        self.assertIn("***REDACTED***", safe_command)

    def test_builds_uv_run_tau2_command_prefix(self):
        env = {
            "TAU2_COMMAND": "uv run tau2",
            "TAU2_REPO_DIR": ".external/tau2-bench",
            "OUTPUT_DIR": "outputs/runs",
        }
        config = build_run_config(env=env, now="2026-06-24_120000")
        command, ignored = build_tau2_command(
            config,
            help_text="--domain\n--agent-llm\n--user-llm\n--num-trials\n--num-tasks",
            command_prefix=["uv", "run", "tau2"],
        )

        self.assertEqual(command[:4], ["uv", "run", "tau2", "run"])
        self.assertEqual(ignored, ["REASONING_EFFORT", "TEMPERATURE"])

    def test_config_supports_fixed_run_id_resume_and_max_workers(self):
        env = {
            "TAU2_RUN_ID": "tau2_airline_full",
            "TAU2_RESUME": "true",
            "TAU2_MAX_WORKERS": "4",
            "TAU2_PROGRESS_INTERVAL": "7.5",
            "OUTPUT_DIR": "outputs/runs",
        }

        config = build_run_config(env=env, now="2026-06-24_120000")

        self.assertEqual(config.run_id, "tau2_airline_full")
        self.assertEqual(config.run_dir, Path("outputs/runs") / "tau2_airline_full")
        self.assertTrue(config.resume)
        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.progress_interval, 7.5)

    def test_config_supports_one_by_one_task_ids_and_save_to(self):
        config = build_run_config(
            env={
                "TAU2_ONE_BY_ONE": "true",
                "TAU2_TASK_IDS": "0, 1 2",
                "TAU2_SAVE_TO": "metacore/custom",
                "TAU2_AUTO_RESUME": "true",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-24_120000",
        )

        self.assertTrue(config.one_by_one)
        self.assertEqual(config.task_ids, ["0", "1", "2"])
        self.assertEqual(config.save_to, "metacore/custom")
        self.assertTrue(config.auto_resume)

    def test_task_ids_omit_num_tasks_and_add_save_to_auto_resume(self):
        config = build_run_config(
            env={
                "TAU2_TASK_IDS": "0 1",
                "TAU2_SAVE_TO": "metacore/task_0",
                "TAU2_AUTO_RESUME": "true",
                "NUM_TASKS": "9",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-24_120000",
        )

        command, ignored = build_tau2_command(
            config,
            help_text=(
                "--domain\n--agent-llm\n--user-llm\n--num-trials\n--num-tasks\n"
                "--task-ids\n--save-to\n--auto-resume"
            ),
        )

        self.assertIn("--task-ids", command)
        self.assertIn("0", command)
        self.assertIn("1", command)
        self.assertNotIn("--num-tasks", command)
        self.assertIn("--save-to", command)
        self.assertIn("metacore/task_0", command)
        self.assertIn("--auto-resume", command)
        self.assertNotIn("TAU2_TASK_IDS", ignored)

    def test_one_by_one_save_to_is_stable_and_safe(self):
        config = build_run_config(
            env={
                "TAU2_RUN_ID": "tau2_airline_full",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-24_120000",
        )

        self.assertEqual(one_by_one_save_to(config, "task/001"), "tau2_airline_full/task_001")

    def test_one_by_one_worker_count_uses_max_workers_for_task_processes(self):
        config = build_run_config(
            env={
                "TAU2_ONE_BY_ONE": "true",
                "TAU2_MAX_WORKERS": "8",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-24_120000",
        )
        default_config = build_run_config(env={"OUTPUT_DIR": "outputs/runs"}, now="2026-06-24_120000")

        self.assertEqual(one_by_one_worker_count(config), 8)
        self.assertEqual(one_by_one_worker_count(default_config), 1)

    def test_one_by_one_task_config_forces_single_internal_worker(self):
        config = build_run_config(
            env={
                "TAU2_RUN_ID": "tau2_airline_full",
                "TAU2_ONE_BY_ONE": "true",
                "TAU2_MAX_WORKERS": "8",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-24_120000",
        )

        task_config = build_one_by_one_task_config(config, "task/001")

        self.assertEqual(task_config.task_ids, ["task/001"])
        self.assertEqual(task_config.save_to, "tau2_airline_full/task_001")
        self.assertEqual(task_config.num_tasks, 1)
        self.assertFalse(task_config.all_tasks)
        self.assertTrue(task_config.auto_resume)
        self.assertEqual(task_config.max_workers, 1)

    def test_one_by_one_wrapper_does_not_mark_task_workers_ignored(self):
        config = build_run_config(
            env={
                "TAU2_ONE_BY_ONE": "true",
                "TAU2_MAX_WORKERS": "8",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-24_120000",
        )

        ignored = wrapper_ignored_options(config, ["REASONING_EFFORT", "TAU2_MAX_WORKERS"])

        self.assertEqual(ignored, ["REASONING_EFFORT"])

    def test_adds_supported_tau2_worker_option(self):
        env = {
            "TAU2_MAX_WORKERS": "4",
            "OUTPUT_DIR": "outputs/runs",
        }
        config = build_run_config(env=env, now="2026-06-24_120000")

        command, ignored = build_tau2_command(
            config,
            help_text="--domain\n--agent-llm\n--user-llm\n--num-trials\n--num-tasks\n--max-concurrency",
        )

        self.assertIn("--max-concurrency", command)
        self.assertIn("4", command)
        self.assertNotIn("TAU2_MAX_WORKERS", ignored)

    def test_ignores_worker_option_when_tau2_help_does_not_support_it(self):
        env = {
            "TAU2_MAX_WORKERS": "4",
            "OUTPUT_DIR": "outputs/runs",
        }
        config = build_run_config(env=env, now="2026-06-24_120000")

        command, ignored = build_tau2_command(
            config,
            help_text="--domain\n--agent-llm\n--user-llm\n--num-trials\n--num-tasks",
        )

        self.assertNotIn("4", command)
        self.assertIn("TAU2_MAX_WORKERS", ignored)

    def test_tau2_error_raw_record_preserves_failure_context(self):
        record = tau2_error_raw_record(
            run_id="tau2_airline_full",
            config=build_run_config(env={"OUTPUT_DIR": "outputs/runs"}, now="2026-06-24_120000"),
            returncode=1,
            stderr="boom",
            stdout="partial output",
            command=["tau2", "run"],
        )

        self.assertEqual(record["benchmark"], "tau2-bench")
        self.assertEqual(record["task_id"], "tau2_error_tau2_airline_full")
        self.assertIsNone(record["success"])
        self.assertEqual(record["error"]["type"], "Tau2ProcessError")
        self.assertEqual(record["error"]["returncode"], 1)
        self.assertIn("boom", record["error"]["stderr"])
        self.assertEqual(record["messages"][0]["role"], "system")
        self.assertIn("tau2 run", record["messages"][1]["content"])

    def test_resume_collects_existing_raw_json_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "outputs" / "runs" / "tau2_airline_full"
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True)
            existing = raw_dir / "task_001.json"
            existing.write_text(json.dumps({"task_id": "task_001"}), encoding="utf-8")
            config = build_run_config(
                env={
                    "OUTPUT_DIR": str(Path(tmp) / "outputs" / "runs"),
                    "TAU2_RUN_ID": "tau2_airline_full",
                    "TAU2_RESUME": "true",
                },
                now="2026-06-24_120000",
            )

            from scripts.run_tau2_gpt55 import existing_raw_files_for_resume

            self.assertEqual(existing_raw_files_for_resume(config), [existing])

    def test_resolves_explicit_uv_launcher_with_repo_cwd(self):
        env = {
            "TAU2_COMMAND": "uv run tau2",
            "TAU2_REPO_DIR": ".external/tau2-bench",
            "OUTPUT_DIR": "outputs/runs",
        }
        config = build_run_config(env=env, now="2026-06-24_120000")

        invocation = resolve_tau2_invocation(env, config)

        self.assertEqual(invocation.command_prefix, ["uv", "run", "tau2"])
        self.assertEqual(invocation.cwd, Path(".external/tau2-bench"))
        self.assertEqual(invocation.source, "TAU2_COMMAND")


if __name__ == "__main__":
    unittest.main()
