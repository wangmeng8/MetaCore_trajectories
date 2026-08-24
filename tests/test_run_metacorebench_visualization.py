import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_metacorebench_visualization import (
    build_pipeline_command,
    build_visualization_config,
    infer_visualization_benchmark,
    latest_run_with_benchmark_trajectories,
    write_visualization_config,
)


class RunMetaCoreBenchVisualizationTests(unittest.TestCase):
    def test_maps_collection_benchmarks_to_visualization_benchmarks(self):
        self.assertEqual(infer_visualization_benchmark("hle-with-tools", None), "hle_with_tools")
        self.assertEqual(infer_visualization_benchmark("terminal-bench-2.0", None), "terminal_bench_2")
        self.assertEqual(infer_visualization_benchmark("tau2-bench", "retail"), "tau2_retail")

    def test_builds_config_for_existing_benchmark_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "runs" / "demo"
            run_dir.mkdir(parents=True)
            input_path = run_dir / "benchmark_trajectories.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")

            config = build_visualization_config(
                run_dir=run_dir,
                benchmark="hle_with_tools",
                model_path="/models/Qwen",
                output_root=run_dir / "metacorebench_results",
                max_length=4096,
                gpu_numbers="0,1",
                backend="service",
                save_representations=True,
                save_activation_matrix=True,
                stage_to_ram=False,
                ram_cache_dir="/dev/shm/custom",
                service_port=8011,
            )

            self.assertEqual(config["run"]["name"], "demo_visualization")
            self.assertEqual(config["data"]["benchmark"], "hle_with_tools")
            self.assertEqual(config["data"]["input_path"], str(input_path.resolve()))
            self.assertEqual(config["model"]["model_name_or_path"], "/models/Qwen")
            self.assertEqual(config["model"]["gpu_numbers"], "0,1")
            self.assertEqual(config["model"]["backend"], "service")
            self.assertFalse(config["model"]["stage_to_ram"])
            self.assertEqual(config["model"]["service_port"], 8011)
            self.assertEqual(config["extraction"]["max_length"], 4096)
            self.assertTrue(config["extraction"]["save_representations"])
            self.assertTrue(config["visualization"]["save_activation_matrix"])

    def test_writes_json_yaml_compatible_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            write_visualization_config(path, {"run": {"name": "demo"}})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"run": {"name": "demo"}})

    def test_latest_run_requires_benchmark_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run = root / "old"
            new_run = root / "new"
            old_run.mkdir()
            new_run.mkdir()
            (old_run / "benchmark_trajectories.jsonl").write_text("{}\n", encoding="utf-8")
            (new_run / "summary.json").write_text("{}", encoding="utf-8")

            self.assertEqual(latest_run_with_benchmark_trajectories(root), old_run)

    def test_builds_cross_platform_python_pipeline_command(self):
        command = build_pipeline_command(
            external_repo=Path("vendor"),
            config_path=Path("run") / "config.yaml",
            python_bin="python",
        )

        self.assertEqual(command[:2], ["python", "scripts/step_01_extract_transformers.py"])
        self.assertEqual(command[-1], str(Path("run") / "config.yaml"))

    def test_script_can_run_by_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "benchmark_trajectories.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "task_001",
                        "benchmark": "hle-with-tools",
                        "task": {"instruction": "demo"},
                        "trajectories": [
                            {
                                "trajectory_id": "traj_001",
                                "source_model": "gpt-5.5",
                                "trajectory_text": "USER: hello",
                                "metadata": {},
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "visualization.yaml"
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_metacorebench_visualization.py"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--run-dir",
                    str(run_dir),
                    "--config-out",
                    str(config_path),
                    "--output-root",
                    str(root / "results"),
                ],
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(config_path.exists())
