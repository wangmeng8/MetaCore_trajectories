import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.run_terminal_bench import (
    build_harbor_command,
    build_terminal_bench_config,
    missing_terminal_trajectory_task_names,
    select_terminal_json_files,
    sync_terminal_outputs_live,
)


class RunTerminalBenchTests(unittest.TestCase):
    def test_builds_terminal_bench_harbor_command(self):
        env = {
            "AGENT_MODEL": "openai/gpt-5.4",
            "TERMINAL_BENCH_AGENT": "terminus-2",
            "TERMINAL_BENCH_DATASET": "terminal-bench/terminal-bench-2",
            "NUM_TASKS": "1",
            "NUM_TRIALS": "1",
            "OUTPUT_DIR": "outputs/runs",
        }

        config = build_terminal_bench_config(env=env, now="2026-06-25_131500")
        command = build_harbor_command(config)

        self.assertEqual(config.benchmark, "terminal-bench-2.0")
        self.assertEqual(config.run_id, "2026-06-25_131500_openai_gpt-5.4_terminal-bench-2.0")
        self.assertEqual(command[:2], ["harbor", "run"])
        self.assertIn("-d", command)
        self.assertIn("terminal-bench/terminal-bench-2", command)
        self.assertIn("-a", command)
        self.assertIn("terminus-2", command)
        self.assertIn("-m", command)
        self.assertIn("openai/gpt-5.4", command)
        self.assertIn("-l", command)
        self.assertIn("1", command)
        self.assertIn("-k", command)

    def test_all_tasks_omits_task_limit(self):
        env = {
            "AGENT_MODEL": "openai/gpt-5.5",
            "TERMINAL_BENCH_ALL_TASKS": "true",
            "NUM_TASKS": "89",
            "OUTPUT_DIR": "outputs/runs",
        }

        config = build_terminal_bench_config(env=env, now="2026-07-01_170000")
        command = build_harbor_command(config)

        self.assertTrue(config.all_tasks)
        self.assertNotIn("-l", command)
        self.assertNotIn("89", command)

    def test_adds_harbor_concurrency(self):
        env = {
            "AGENT_MODEL": "openai/gpt-5.5",
            "TERMINAL_BENCH_MAX_WORKERS": "8",
            "OUTPUT_DIR": "outputs/runs",
        }

        config = build_terminal_bench_config(env=env, now="2026-07-01_170000")
        command = build_harbor_command(config)

        self.assertEqual(config.max_workers, 8)
        self.assertIn("-n", command)
        self.assertIn("8", command)

    def test_supports_single_terminal_bench_task_name(self):
        env = dict(os.environ)
        env.update(
            {
                "AGENT_MODEL": "openai/gpt-5.4-mini",
                "TERMINAL_BENCH_TASK_NAME": "openssl-selfsigned-cert",
                "OUTPUT_DIR": "outputs/runs",
            }
        )
        config = build_terminal_bench_config(env=env, now="2026-06-25_131500")

        command = build_harbor_command(config)

        self.assertIn("--include-task-name", command)
        self.assertIn("terminal-bench/openssl-selfsigned-cert", command)

    def test_prefers_agent_trajectory_json_for_collection(self):
        raw_files = [
            Path("raw/job/result.json"),
            Path("raw/job/trial/config.json"),
            Path("raw/job/trial/agent/trajectory.json"),
            Path("raw/job/trial/verifier/ctrf.json"),
        ]

        selected = select_terminal_json_files(raw_files)

        self.assertEqual(selected, [Path("raw/job/trial/agent/trajectory.json")])

    def test_sync_terminal_outputs_live_copies_partial_job_and_refreshes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            jobs_dir = workspace / "jobs"
            trajectory_path = jobs_dir / "job-1" / "task-1" / "agent" / "trajectory.json"
            trajectory_path.parent.mkdir(parents=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1",
                        "agent": {"model": "openai/gpt-5.5"},
                        "steps": [
                            {
                                "type": "llm",
                                "messages": [
                                    {"role": "user", "content": "Task Description: say hi"},
                                    {"role": "assistant", "content": "hi"},
                                ],
                            }
                        ],
                        "final_metrics": {"total_cost_usd": 0.01},
                    }
                ),
                encoding="utf-8",
            )
            config = build_terminal_bench_config(
                env={
                    "OUTPUT_DIR": str(workspace / "outputs" / "runs"),
                    "HARBOR_JOBS_DIR": str(jobs_dir),
                    "TERMINAL_BENCH_RUN_ID": "terminal_live",
                    "AGENT_MODEL": "openai/gpt-5.5",
                },
                now="2026-07-01_170000",
            )
            raw_dir = config.run_dir / "raw"
            trajectories_dir = config.run_dir / "trajectories"
            benchmark_dir = config.run_dir / "benchmark_trajectories"
            manifest = {"run_id": config.run_id, "benchmark": config.benchmark}

            summary = sync_terminal_outputs_live(
                config=config,
                raw_dir=raw_dir,
                trajectories_dir=trajectories_dir,
                benchmark_dir=benchmark_dir,
                manifest=manifest,
            )

            self.assertEqual(summary["total_trajectories"], 1)
            self.assertTrue((raw_dir / "job-1" / "task-1" / "agent" / "trajectory.json").exists())
            self.assertTrue((config.run_dir / "benchmark_trajectories.jsonl").exists())
            self.assertEqual(manifest["raw_files"], ["raw\\job-1\\task-1\\agent\\trajectory.json"] if os.name == "nt" else ["raw/job-1/task-1/agent/trajectory.json"])

    def test_missing_terminal_trajectory_task_names_uses_lock_and_existing_agent_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            (job_dir / "lock.json").write_text(
                json.dumps(
                    {
                        "trials": [
                            {"task": {"name": "terminal-bench/task-a"}},
                            {"task": {"name": "terminal-bench/task-b"}},
                            {"task": {"name": "terminal-bench/task-c"}},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            task_a = job_dir / "task-a__abc"
            task_a.mkdir()
            (task_a / "result.json").write_text(json.dumps({"task_name": "terminal-bench/task-a"}), encoding="utf-8")
            (task_a / "agent").mkdir()
            (task_a / "agent" / "trajectory.json").write_text("{}", encoding="utf-8")
            task_b = job_dir / "task-b__def"
            task_b.mkdir()
            (task_b / "result.json").write_text(json.dumps({"task_name": "terminal-bench/task-b"}), encoding="utf-8")

            self.assertEqual(missing_terminal_trajectory_task_names(job_dir), ["task-b", "task-c"])


if __name__ == "__main__":
    unittest.main()
