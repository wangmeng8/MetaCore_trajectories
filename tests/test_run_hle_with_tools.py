import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import scripts.run_hle_with_tools as hle_runner

from scripts.run_hle_with_tools import (
    HLEWithToolsConfig,
    build_hle_with_tools_config,
    build_official_command,
    collect_hle_with_tools_outputs,
    current_prediction_count,
    normalize_base_url,
    apply_cli_overrides,
    parse_args,
)


class RunHLEWithToolsOfficialTests(unittest.TestCase):
    def test_builds_official_hle_defaults(self):
        config = build_hle_with_tools_config(env={"OUTPUT_DIR": "outputs/runs"}, now="2026-07-16_120000")

        self.assertEqual(config.benchmark, "hle-with-tools")
        self.assertEqual(config.dataset, "cais/hle")
        self.assertEqual(config.model, "gpt-5.5")
        self.assertEqual(config.max_completion_tokens, 40000)
        self.assertEqual(config.max_iterations, 15)
        self.assertEqual(config.question_timeout_seconds, 1800.0)
        self.assertEqual(config.max_workers, 2)
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.process_retries, 0)
        self.assertTrue(config.text_only)
        self.assertFalse(config.disable_scientific_search)
        self.assertEqual(config.run_id, "2026-07-16_120000_gpt-5.5_hle-with-tools")

    def test_can_disable_scientific_search_in_manifest_tool_list(self):
        config = build_hle_with_tools_config(
            env={
                "OUTPUT_DIR": "outputs/runs",
                "HLE_DISABLE_SCIENTIFIC_SEARCH": "1",
            },
            now="2026-07-16_120000",
        )

        self.assertTrue(config.disable_scientific_search)
        self.assertEqual(config.tool_names, ["code_interpreter", "web_browsing"])

    def test_builds_official_command_without_temperature_by_default(self):
        config = build_hle_with_tools_config(
            env={
                "OUTPUT_DIR": "outputs/runs",
                "HLE_MODEL": "gpt-5.5",
                "HLE_DATA_PATH": "data/modelscope/cais_hle/data/test-00000-of-00001.parquet",
                "NUM_TASKS": "2",
                "HLE_MAX_WORKERS": "8",
                "HLE_WITH_TOOLS_USE_UV": "0",
            },
            now="2026-07-16_120000",
        )

        command = build_official_command(config)

        self.assertIn("hle_eval/run_agent_predictions.py", command)
        self.assertIn("--dataset", command)
        self.assertIn(str(Path("data/modelscope/cais_hle/data/test-00000-of-00001.parquet").resolve()), command)
        self.assertIn("--max_samples", command)
        self.assertIn("2", command)
        self.assertIn("--num_workers", command)
        self.assertIn("8", command)
        self.assertIn("--max_completion_tokens", command)
        self.assertIn("40000", command)
        self.assertIn("--question_timeout_seconds", command)
        self.assertIn("1800.0", command)
        self.assertNotIn("--temperature", command)

    def test_collects_official_trace_as_benchmark_trajectory_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "hle_run"
            config = HLEWithToolsConfig(
                run_id="hle_run",
                run_dir=run_dir,
                repo_dir=Path(".external/hle_with_tools"),
                benchmark="hle-with-tools",
                dataset="cais/hle",
                data_path=None,
                model="gpt-5.5",
                base_url=None,
                num_tasks=1,
                num_rollouts=1,
                max_workers=2,
                max_retries=3,
                process_retries=3,
                max_completion_tokens=40000,
                max_iterations=15,
                temperature=None,
                text_only=True,
                disable_scientific_search=False,
                use_uv=False,
                progress_interval=30,
            )
            config.official_raw_dir.joinpath("traces").mkdir(parents=True)
            config.official_raw_dir.joinpath("code").mkdir(parents=True)
            config.predictions_path.write_text(
                json.dumps(
                    {
                        "q1": {
                            "model": "gpt-5.5",
                            "response": "Explanation: computed.\nExact Answer: 4\nConfidence: 99%",
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        }
                    }
                ),
                encoding="utf-8",
            )
            trace = "\n".join(
                [
                    "QUESTION ID: q1",
                    "Question: What is 2+2?",
                    "STEP 1/15",
                    "LLM API CALL Iteration 1",
                    "TOOL CALL code_interpreter",
                    "code: print(2+2)",
                    "TOOL RESULT code_interpreter",
                    "4",
                    "FINAL ANSWER",
                ]
            )
            config.official_raw_dir.joinpath("traces", "trace_q1.log").write_text(trace, encoding="utf-8")
            config.official_raw_dir.joinpath("code", "q1.py").write_text("print(2+2)\n", encoding="utf-8")

            trajectories, records, summary = collect_hle_with_tools_outputs(config)

            self.assertEqual(len(trajectories), 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["benchmark"], "hle-with-tools")
            self.assertIn("TOOL CALL code_interpreter", records[0]["trajectories"][0]["trajectory_text"])
            self.assertEqual(records[0]["trajectories"][0]["metadata"]["tools"][0], "code_interpreter")
            self.assertEqual(summary["total_tokens"], 15)
            self.assertTrue((run_dir / "benchmark_trajectories.jsonl").exists())

    def test_normalizes_bare_base_url(self):
        self.assertEqual(normalize_base_url("http://newapi.lab.arpa"), "http://newapi.lab.arpa/v1")
        self.assertEqual(normalize_base_url("https://api.example.com/v1"), "https://api.example.com/v1")

    def test_progress_count_uses_lightweight_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = build_hle_with_tools_config(
                env={"OUTPUT_DIR": tmp, "HLE_RUN_ID": "progress-test"},
                now="2026-07-16_120000",
            )
            config.official_raw_dir.mkdir(parents=True)
            config.official_raw_dir.joinpath("progress_state.json").write_text(
                json.dumps({"completed": 7, "running": 2, "failed": 1}),
                encoding="utf-8",
            )

            self.assertEqual(current_prediction_count(config), 7)

    def test_terminal_progress_includes_failures_timeouts_and_skips(self):
        self.assertTrue(hasattr(hle_runner, "terminal_progress_count"))
        self.assertEqual(
            hle_runner.terminal_progress_count(
                {
                    "completed": 492,
                    "failed": 2,
                    "timed_out": 3,
                    "skipped": 3,
                }
            ),
            500,
        )

    def test_stalled_official_process_is_terminated_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = build_hle_with_tools_config(
                env={
                    "OUTPUT_DIR": tmp,
                    "HLE_RUN_ID": "watchdog-test",
                    "HLE_WITH_TOOLS_REPO_DIR": tmp,
                    "HLE_PROGRESS_INTERVAL": "0.01",
                    "HLE_STALL_TIMEOUT_SECONDS": "0.05",
                },
                now="2026-07-16_120000",
            )
            self.assertTrue(hasattr(config, "stall_timeout_seconds"))

            started = time.monotonic()
            returncode = hle_runner.run_official_command(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                config=config,
                env=dict(os.environ),
            )
            elapsed = time.monotonic() - started

            self.assertEqual(returncode, hle_runner.WATCHDOG_EXIT_CODE)
            self.assertLess(elapsed, 3.0)

    def test_watchdog_terminates_official_child_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "orphan-child.txt"
            parent_ready = Path(tmp) / "parent-ready.txt"
            child_code = (
                "import time; from pathlib import Path; "
                f"time.sleep(1.0); Path({str(marker)!r}).write_text('orphan', encoding='utf-8')"
            )
            parent_code = (
                "import subprocess, sys, time; from pathlib import Path; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                f"Path({str(parent_ready)!r}).write_text('ready', encoding='utf-8'); "
                "time.sleep(10)"
            )
            config = build_hle_with_tools_config(
                env={
                    "OUTPUT_DIR": tmp,
                    "HLE_RUN_ID": "process-tree-test",
                    "HLE_WITH_TOOLS_REPO_DIR": tmp,
                    "HLE_STALL_TIMEOUT_SECONDS": "0.3",
                },
                now="2026-07-16_120000",
            )

            returncode = hle_runner.run_official_command(
                [sys.executable, "-c", parent_code],
                config=config,
                env=dict(os.environ),
            )
            time.sleep(1.2)

            self.assertEqual(returncode, hle_runner.WATCHDOG_EXIT_CODE)
            self.assertTrue(parent_ready.exists(), "test parent did not start its child")
            self.assertFalse(marker.exists(), "watchdog left an orphan child process running")

    def test_official_logs_are_appended_across_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = build_hle_with_tools_config(
                env={
                    "OUTPUT_DIR": tmp,
                    "HLE_RUN_ID": "append-log-test",
                    "HLE_WITH_TOOLS_REPO_DIR": tmp,
                    "HLE_STALL_TIMEOUT_SECONDS": "10",
                },
                now="2026-07-16_120000",
            )

            for marker in ("first-attempt", "second-attempt"):
                returncode = hle_runner.run_official_command(
                    [sys.executable, "-c", f"print({marker!r})"],
                    config=config,
                    env=dict(os.environ),
                )
                self.assertEqual(returncode, 0)

            output = (config.run_dir / "hle_with_tools.stdout.log").read_text(encoding="utf-8")
            self.assertIn("first-attempt", output)
            self.assertIn("second-attempt", output)

    def test_can_forward_skip_failed_on_resume(self):
        config = build_hle_with_tools_config(
            env={"OUTPUT_DIR": "outputs/runs", "HLE_SKIP_FAILED_ON_RESUME": "1"},
            now="2026-07-16_120000",
        )

        self.assertTrue(
            hasattr(config, "skip_failed_on_resume"),
            "config must expose skip_failed_on_resume",
        )
        self.assertTrue(config.skip_failed_on_resume)
        self.assertIn("--skip_failed_on_resume", build_official_command(config))

    def test_collects_marked_incorrect_prediction_as_zero_score_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "hle_error_run"
            config = HLEWithToolsConfig(
                run_id="hle_error_run",
                run_dir=run_dir,
                repo_dir=Path(".external/hle_with_tools"),
                benchmark="hle-with-tools",
                dataset="cais/hle",
                data_path=None,
                model="test-model",
                base_url=None,
                num_tasks=1,
                num_rollouts=1,
                max_workers=2,
                max_retries=3,
                process_retries=1,
                max_completion_tokens=40000,
                max_iterations=15,
                temperature=None,
                text_only=True,
                disable_scientific_search=False,
                use_uv=False,
                progress_interval=30,
            )
            config.official_raw_dir.mkdir(parents=True)
            config.predictions_path.write_text(
                json.dumps(
                    {
                        "slow": {
                            "model": "test-model",
                            "response": "",
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            "termination_reason": "infrastructure_error",
                            "error": {
                                "error_type": "QuestionTimeout",
                                "message": "Question timed out",
                            },
                            "judge_response": {
                                "correct": "no",
                                "confidence": 100,
                                "model_answer": "None",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            trajectories, records, summary = collect_hle_with_tools_outputs(config)

            self.assertFalse(trajectories[0]["success"])
            self.assertEqual(trajectories[0]["score"], 0.0)
            self.assertFalse(trajectories[0]["raw"]["completed_prediction"])
            self.assertEqual(summary["infrastructure_error_count"], 1)
            self.assertEqual(summary["error_types"], {"QuestionTimeout": 1})
            self.assertFalse(records[0]["trajectories"][0]["metadata"]["completed_prediction"])

    def test_cli_forwards_explicit_incorrect_ids(self):
        args = parse_args(["--mark-incorrect-ids", "16,18,32"])
        env = apply_cli_overrides({}, args)
        config = build_hle_with_tools_config(env=env, now="2026-07-16_120000")

        self.assertEqual(config.mark_incorrect_ids, ("16", "18", "32"))
        command = build_official_command(config)
        self.assertIn("--mark_incorrect_ids", command)
        self.assertIn("16,18,32", command)

    def test_cli_forwards_system_prompt_file_to_official_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "prompt_reg.txt"
            prompt_path.write_text("Feedback regulation policy.", encoding="utf-8")

            args = parse_args(["--system-prompt-file", str(prompt_path)])
            env = apply_cli_overrides({}, args)
            config = build_hle_with_tools_config(env=env, now="2026-07-16_120000")
            command = build_official_command(config)

            self.assertEqual(config.system_prompt_file, prompt_path)
            self.assertIn("--system_prompt_file", command)
            self.assertIn(str(prompt_path.resolve()), command)


if __name__ == "__main__":
    unittest.main()
