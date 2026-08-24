import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_hle_with_tools import (
    HLEWithToolsConfig,
    build_hle_with_tools_config,
    build_official_command,
    collect_hle_with_tools_outputs,
    normalize_base_url,
)


class RunHLEWithToolsOfficialTests(unittest.TestCase):
    def test_builds_official_hle_defaults(self):
        config = build_hle_with_tools_config(env={"OUTPUT_DIR": "outputs/runs"}, now="2026-07-16_120000")

        self.assertEqual(config.benchmark, "hle-with-tools")
        self.assertEqual(config.dataset, "cais/hle")
        self.assertEqual(config.model, "gpt-5.5")
        self.assertEqual(config.max_completion_tokens, 40000)
        self.assertEqual(config.max_iterations, 15)
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


if __name__ == "__main__":
    unittest.main()
