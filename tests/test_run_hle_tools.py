import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_hle_tools import (
    api_model_name,
    build_hle_config,
    collect_hle_outputs_live,
    is_multimodal_hle_item,
    load_hle_examples,
    raw_path_for_example,
    run_hle_batch,
    run_one_hle_example,
    run_calculator_tool,
    to_hle_checkpoint_raw_record,
    to_hle_error_raw_record,
    to_hle_raw_record,
)


class RunHLEToolsTests(unittest.TestCase):
    def test_loads_hle_jsonl_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hle.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "hle_001", "question": "What is 2+2?", "answer": "4"}),
                        json.dumps({"task_id": "hle_002", "prompt": "What is 3+5?", "target": "8"}),
                    ]
                ),
                encoding="utf-8",
            )

            examples = load_hle_examples(path, limit=2)

            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0]["task_id"], "hle_001")
            self.assertEqual(examples[0]["instruction"], "What is 2+2?")
            self.assertEqual(examples[0]["answer"], "4")
            self.assertEqual(examples[1]["task_id"], "hle_002")
            self.assertEqual(examples[1]["instruction"], "What is 3+5?")
            self.assertEqual(examples[1]["answer"], "8")

    def test_loads_all_hle_examples_when_limit_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hle.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"id": f"hle_{index}", "question": f"Q{index}", "answer": f"A{index}"})
                    for index in range(3)
                ),
                encoding="utf-8",
            )

            examples = load_hle_examples(path, limit=None)

            self.assertEqual([example["task_id"] for example in examples], ["hle_0", "hle_1", "hle_2"])

    def test_loads_hle_parquet_examples(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hle.parquet"
            pd.DataFrame(
                [
                    {
                        "id": "hle_parquet_001",
                        "question": "What is 10+7?",
                        "answer": "17",
                        "image": None,
                    }
                ]
            ).to_parquet(path)

            examples = load_hle_examples(path, limit=1)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["task_id"], "hle_parquet_001")
            self.assertEqual(examples[0]["instruction"], "What is 10+7?")
            self.assertEqual(examples[0]["answer"], "17")

    def test_filters_multimodal_examples_before_applying_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hle.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "image_question",
                                "question": "What is shown in this image?",
                                "answer": "a cat",
                                "image": "data:image/png;base64,AAAA",
                            }
                        ),
                        json.dumps({"id": "text_question", "question": "What is 2+2?", "answer": "4"}),
                    ]
                ),
                encoding="utf-8",
            )

            examples = load_hle_examples(path, limit=1)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["task_id"], "text_question")
            self.assertEqual(examples[0]["instruction"], "What is 2+2?")

    def test_detects_multimodal_hle_items(self):
        self.assertTrue(is_multimodal_hle_item({"image": "data:image/png;base64,AAAA"}))
        self.assertTrue(is_multimodal_hle_item({"image_preview": {"bytes": b"abc", "path": None}}))
        self.assertFalse(is_multimodal_hle_item({"image": None, "image_preview": None, "question": "Text only"}))

    def test_calculator_tool_evaluates_safe_arithmetic(self):
        self.assertEqual(run_calculator_tool({"expression": "2 + 3 * 4"})["result"], 14)
        self.assertIn("error", run_calculator_tool({"expression": "__import__('os').system('dir')" }))

    def test_builds_hle_raw_record_with_tool_events(self):
        example = {"task_id": "hle_001", "instruction": "Compute 2+2.", "answer": "4", "raw": {"id": "hle_001"}}
        messages = [
            {"role": "user", "content": "Compute 2+2."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": "{\"expression\":\"2+2\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "calculator", "content": "{\"result\":4}"},
            {"role": "assistant", "content": "4"},
        ]

        record = to_hle_raw_record(
            example=example,
            messages=messages,
            model="openai/gpt-5.4",
            raw_responses=[{"id": "chatcmpl_1"}],
        )

        self.assertEqual(record["benchmark"], "hle-with-tools")
        self.assertEqual(record["task_id"], "hle_001")
        self.assertEqual(record["task"]["instruction"], "Compute 2+2.")
        self.assertEqual(record["agent_model"], "openai/gpt-5.4")
        self.assertTrue(record["success"])
        self.assertEqual(record["score"], 1.0)
        self.assertEqual(len(record["messages"]), 4)

    def test_builds_hle_config_defaults(self):
        config = build_hle_config(
            env={
                "AGENT_MODEL": "openai/gpt-5.4",
                "HLE_DATA_PATH": "data/hle_sample.jsonl",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-25_131700",
        )

        self.assertEqual(config.benchmark, "hle-with-tools")
        self.assertEqual(config.run_id, "2026-06-25_131700_openai_gpt-5.4_hle-with-tools")
        self.assertEqual(config.num_tasks, 1)
        self.assertIsNone(config.temperature)
        self.assertFalse(config.all_tasks)
        self.assertEqual(config.max_workers, 1)
        self.assertFalse(config.resume)
        self.assertEqual(config.request_timeout, 180.0)
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.retry_sleep, 2.0)
        self.assertEqual(config.progress_interval, 30.0)

    def test_hle_config_supports_all_tasks_resume_and_worker_controls(self):
        config = build_hle_config(
            env={
                "AGENT_MODEL": "openai/gpt-5.5",
                "HLE_DATA_PATH": "data/hle_sample.jsonl",
                "OUTPUT_DIR": "outputs/runs",
                "HLE_ALL_TASKS": "true",
                "HLE_MAX_WORKERS": "4",
                "HLE_RESUME": "true",
                "HLE_REQUEST_TIMEOUT": "90",
                "HLE_MAX_RETRIES": "5",
                "HLE_RETRY_SLEEP": "1.5",
                "HLE_PROGRESS_INTERVAL": "5",
            },
            now="2026-06-29_170000",
        )

        self.assertTrue(config.all_tasks)
        self.assertIsNone(config.num_tasks)
        self.assertEqual(config.max_workers, 4)
        self.assertTrue(config.resume)
        self.assertEqual(config.request_timeout, 90.0)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.retry_sleep, 1.5)
        self.assertEqual(config.progress_interval, 5.0)

    def test_hle_run_id_can_be_fixed_for_resume(self):
        config = build_hle_config(
            env={
                "AGENT_MODEL": "openai/gpt-5.5",
                "HLE_DATA_PATH": "data/hle_sample.jsonl",
                "OUTPUT_DIR": "outputs/runs",
                "HLE_RUN_ID": "hle_gpt55_full",
            },
            now="2026-06-29_170000",
        )

        self.assertEqual(config.run_id, "hle_gpt55_full")
        self.assertEqual(config.run_dir, Path("outputs/runs") / "hle_gpt55_full")

    def test_hle_config_ignores_generic_benchmark_default(self):
        config = build_hle_config(
            env={
                "BENCHMARK": "tau2-bench",
                "AGENT_MODEL": "openai/gpt-5.5",
                "HLE_DATA_PATH": "data/hle_sample.jsonl",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-29_161842",
        )

        self.assertEqual(config.benchmark, "hle-with-tools")
        self.assertEqual(config.run_id, "2026-06-29_161842_openai_gpt-5.5_hle-with-tools")

    def test_hle_benchmark_can_be_overridden_explicitly(self):
        config = build_hle_config(
            env={
                "HLE_BENCHMARK": "custom-hle",
                "AGENT_MODEL": "openai/gpt-5.5",
                "HLE_DATA_PATH": "data/hle_sample.jsonl",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-29_161842",
        )

        self.assertEqual(config.benchmark, "custom-hle")
        self.assertEqual(config.run_id, "2026-06-29_161842_openai_gpt-5.5_custom-hle")

    def test_hle_ignores_generic_temperature_by_default(self):
        config = build_hle_config(
            env={
                "TEMPERATURE": "0",
                "AGENT_MODEL": "openai/gpt-5.5",
                "HLE_DATA_PATH": "data/hle_sample.jsonl",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-29_161842",
        )

        self.assertIsNone(config.temperature)

    def test_hle_temperature_can_be_overridden_explicitly(self):
        config = build_hle_config(
            env={
                "HLE_TEMPERATURE": "0",
                "AGENT_MODEL": "openai/gpt-5.4",
                "HLE_DATA_PATH": "data/hle_sample.jsonl",
                "OUTPUT_DIR": "outputs/runs",
            },
            now="2026-06-29_161842",
        )

        self.assertEqual(config.temperature, 0.0)

    def test_api_model_name_strips_openai_provider_prefix(self):
        self.assertEqual(api_model_name("openai/gpt-5.4"), "gpt-5.4")
        self.assertEqual(api_model_name("gpt-5.4"), "gpt-5.4")

    def test_raw_path_for_example_is_stable_and_safe(self):
        path = raw_path_for_example(Path("raw"), {"task_id": "hle/demo 001"}, 7)

        self.assertEqual(path, Path("raw") / "hle_demo_001.json")

    def test_hle_error_record_preserves_question_and_error(self):
        example = {"task_id": "hle_001", "instruction": "Compute 2+2.", "answer": "4", "raw": {"id": "hle_001"}}

        record = to_hle_error_raw_record(
            example=example,
            model="openai/gpt-5.5",
            error=RuntimeError("endpoint disconnected"),
        )

        self.assertEqual(record["task_id"], "hle_001")
        self.assertEqual(record["task"]["instruction"], "Compute 2+2.")
        self.assertEqual(record["agent_model"], "openai/gpt-5.5")
        self.assertIsNone(record["success"])
        self.assertIsNone(record["score"])
        self.assertEqual(record["error"]["type"], "RuntimeError")
        self.assertIn("endpoint disconnected", record["error"]["message"])
        self.assertEqual(record["messages"][1]["role"], "user")
        self.assertEqual(record["messages"][1]["content"], "Compute 2+2.")

    def test_hle_checkpoint_record_preserves_partial_messages(self):
        example = {"task_id": "hle_001", "instruction": "Compute 2+2.", "answer": "4", "raw": {"id": "hle_001"}}
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Compute 2+2."},
        ]

        record = to_hle_checkpoint_raw_record(
            example=example,
            messages=messages,
            model="openai/gpt-5.5",
            raw_responses=[],
            status="started",
        )

        self.assertEqual(record["task_id"], "hle_001")
        self.assertEqual(record["status"], "started")
        self.assertTrue(record["partial"])
        self.assertIsNone(record["success"])
        self.assertEqual(record["messages"][1]["content"], "Compute 2+2.")

    def test_run_one_hle_example_writes_initial_checkpoint_before_api_failure(self):
        example = {"task_id": "hle_001", "instruction": "Compute 2+2.", "answer": "4", "raw": {"id": "hle_001"}}
        config = build_hle_config(
            env={
                "AGENT_MODEL": "openai/gpt-5.5",
                "HLE_DATA_PATH": "data/hle_sample.jsonl",
                "OUTPUT_DIR": "outputs/runs",
                "MAX_STEPS": "1",
            },
            now="2026-06-29_170000",
        )

        def failing_chat(**_kwargs):
            raise RuntimeError("network interrupted")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "hle_001.json"

            with self.assertRaises(RuntimeError):
                run_one_hle_example(
                    example,
                    config=config,
                    api_key="test-key",
                    checkpoint_path=checkpoint_path,
                    chat_fn=failing_chat,
                )

            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["status"], "error")
            self.assertTrue(checkpoint["partial"])
            self.assertEqual(checkpoint["error"]["type"], "RuntimeError")
            self.assertEqual(checkpoint["messages"][1]["content"], "Compute 2+2.")

    def test_hle_batch_can_refresh_benchmark_outputs_after_each_raw_file(self):
        examples = [
            {"task_id": "hle_001", "instruction": "Compute 2+2.", "answer": "4", "raw": {"id": "hle_001"}},
            {"task_id": "hle_002", "instruction": "Compute 3+3.", "answer": "6", "raw": {"id": "hle_002"}},
        ]

        def chat_fn(**kwargs):
            user_message = kwargs["messages"][1]["content"]
            content = "4" if "2+2" in user_message else "6"
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}

        with tempfile.TemporaryDirectory() as tmp:
            config = build_hle_config(
                env={
                    "AGENT_MODEL": "openai/gpt-5.5",
                    "HLE_DATA_PATH": "data/hle_sample.jsonl",
                    "OUTPUT_DIR": str(Path(tmp) / "outputs" / "runs"),
                    "HLE_RUN_ID": "hle_live",
                    "HLE_MAX_WORKERS": "1",
                    "MAX_STEPS": "1",
                },
                now="2026-06-29_170000",
            )
            raw_dir = config.run_dir / "raw"
            trajectories_dir = config.run_dir / "trajectories"
            benchmark_dir = config.run_dir / "benchmark_trajectories"
            manifest = {"run_id": config.run_id, "benchmark": config.benchmark}
            jsonl_counts = []

            def on_raw_files_changed(raw_files):
                collect_hle_outputs_live(
                    config=config,
                    raw_files=raw_files,
                    trajectories_dir=trajectories_dir,
                    benchmark_dir=benchmark_dir,
                    manifest=manifest,
                )
                jsonl_path = config.run_dir / "benchmark_trajectories.jsonl"
                jsonl_counts.append(len(jsonl_path.read_text(encoding="utf-8").splitlines()))

            run_hle_batch(
                examples,
                config=config,
                api_key="test-key",
                raw_dir=raw_dir,
                chat_fn=chat_fn,
                on_raw_files_changed=on_raw_files_changed,
            )

            self.assertEqual(jsonl_counts, [1, 2])


if __name__ == "__main__":
    unittest.main()
