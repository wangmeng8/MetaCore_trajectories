import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_trajectories import (
    build_benchmark_records,
    collect_raw_files,
    infer_run_id,
    normalize_raw_object,
    serialize_trajectory_text,
    summarize_trajectories,
    to_benchmark_record,
)


class CollectTrajectoriesTests(unittest.TestCase):
    def test_infers_run_id_from_output_directory_name(self):
        self.assertEqual(
            infer_run_id(Path("outputs/runs/2026-06-29_abc_airline")),
            "2026-06-29_abc_airline",
        )

    def test_normalizes_messages_tool_calls_and_tool_results(self):
        raw = {
            "task_id": "task_001",
            "trial_id": 0,
            "domain": "airline",
            "success": True,
            "score": 1.0,
            "messages": [
                {"role": "user", "content": "I need to change my flight."},
                {
                    "role": "assistant",
                    "content": "I can help with that.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "lookup_reservation",
                                "arguments": "{\"reservation_id\":\"ABC123\"}",
                            },
                        }
                    ],
                },
                {"role": "tool", "name": "lookup_reservation", "content": {"found": True}},
            ],
            "reward": 1,
        }

        normalized = normalize_raw_object(
            raw,
            run_id="run_demo",
            raw_file="raw/task_001.json",
            defaults={"agent_model": "gpt-5.5", "user_model": "gpt-5.5", "domain": "airline"},
        )

        self.assertEqual(normalized["task_id"], "task_001")
        self.assertEqual(normalized["trial_id"], 0)
        self.assertTrue(normalized["success"])
        self.assertEqual(normalized["score"], 1.0)
        self.assertEqual(normalized["num_turns"], 3)
        self.assertEqual(normalized["num_tool_calls"], 1)
        self.assertEqual(normalized["raw"], raw)
        self.assertEqual(
            [(event["type"], event["role"]) for event in normalized["events"]],
            [("message", "user"), ("message", "assistant"), ("tool_call", "assistant"), ("tool_result", "tool")],
        )
        self.assertEqual(normalized["events"][2]["tool_name"], "lookup_reservation")
        self.assertEqual(normalized["events"][2]["arguments"], {"reservation_id": "ABC123"})
        self.assertEqual(normalized["events"][3]["content"], {"found": True})

    def test_collects_any_json_shape_without_dropping_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            out_dir = Path(tmp) / "trajectories"
            raw_dir.mkdir()
            raw_path = raw_dir / "unknown_schema.json"
            raw_payload = {"unexpected": {"nested": ["kept"]}}
            raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")

            trajectories = collect_raw_files(
                raw_files=[raw_path],
                trajectories_dir=out_dir,
                jsonl_path=Path(tmp) / "trajectories.jsonl",
                run_id="run_unknown",
                defaults={"agent_model": "gpt-5.5", "user_model": "gpt-5.5", "domain": "airline"},
                raw_root=Path(tmp),
            )

            self.assertEqual(len(trajectories), 1)
            self.assertEqual(trajectories[0]["raw"], raw_payload)
            self.assertEqual(trajectories[0]["events"], [])
            self.assertTrue((out_dir / "trajectory_0000.json").exists())
            lines = (Path(tmp) / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["raw"], raw_payload)

    def test_summary_counts_infrastructure_errors(self):
        trajectories = [
            {
                "success": None,
                "num_turns": 0,
                "num_tool_calls": 0,
                "raw": {
                    "termination_reason": "infrastructure_error",
                    "info": {"error_type": "APIError"},
                },
            }
        ]

        summary = summarize_trajectories(trajectories)

        self.assertEqual(summary["infrastructure_error_count"], 1)
        self.assertEqual(summary["error_types"], {"APIError": 1})

    def test_serializes_trajectory_text_with_messages_and_tools(self):
        normalized = {
            "events": [
                {"index": 0, "role": "user", "type": "message", "content": "I need to change my flight."},
                {"index": 1, "role": "assistant", "type": "message", "content": "I can help."},
                {
                    "index": 2,
                    "role": "assistant",
                    "type": "tool_call",
                    "tool_name": "lookup_reservation",
                    "arguments": {"reservation_id": "ABC123"},
                },
                {
                    "index": 3,
                    "role": "tool",
                    "type": "tool_result",
                    "tool_name": "lookup_reservation",
                    "content": {"found": True},
                },
            ]
        }

        text = serialize_trajectory_text(normalized)

        self.assertIn("USER: I need to change my flight.", text)
        self.assertIn("ASSISTANT: I can help.", text)
        self.assertIn("ASSISTANT TOOL_CALL lookup_reservation", text)
        self.assertIn('"reservation_id": "ABC123"', text)
        self.assertIn("TOOL RESULT lookup_reservation", text)
        self.assertIn('"found": true', text)
        self.assertNotIn("[0]", text)
        self.assertNotIn("[1]", text)

    def test_keeps_tau2_timestamped_messages_with_null_source(self):
        raw = {
            "task_id": "task_001",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hi! How can I help you today?",
                    "tool_calls": None,
                    "timestamp": "2026-06-25T12:31:02",
                    "source": None,
                },
                {
                    "role": "user",
                    "content": "I need help with a reservation.",
                    "tool_calls": None,
                    "timestamp": "2026-06-25T12:31:05",
                    "source": None,
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"name": "get_user_details", "arguments": {"user_id": "u_123"}}],
                    "timestamp": "2026-06-25T12:31:07",
                    "source": None,
                },
            ],
        }

        normalized = normalize_raw_object(raw, run_id="run_demo", raw_file="raw/results.json")
        text = serialize_trajectory_text(normalized)

        self.assertIn("ASSISTANT: Hi! How can I help you today?", text)
        self.assertIn("USER: I need help with a reservation.", text)
        self.assertIn("ASSISTANT TOOL_CALL get_user_details", text)
        self.assertNotIn("[0]", text)

    def test_builds_benchmark_record_for_training_export(self):
        normalized = {
            "run_id": "2026-06-25_123050_openai_gpt-5.4_airline",
            "domain": "airline",
            "task_id": "0",
            "trial_id": 0,
            "agent_model": "openai/gpt-5.4",
            "user_model": "openai/gpt-5.4-mini",
            "success": True,
            "score": 1.0,
            "num_turns": 2,
            "num_tool_calls": 1,
            "task": {"instruction": "Original task instruction."},
            "events": [
                {"index": 0, "role": "user", "type": "message", "content": "Hello"},
                {
                    "index": 1,
                    "role": "assistant",
                    "type": "tool_call",
                    "tool_name": "get_flight",
                    "arguments": {"flight_id": "UA100"},
                },
            ],
            "raw_file": "raw/results.json",
        }

        record = to_benchmark_record(normalized, benchmark="tau2-bench")

        self.assertEqual(record["task_id"], "0")
        self.assertEqual(record["benchmark"], "tau2-bench")
        self.assertEqual(record["task"]["instruction"], "Original task instruction.")
        self.assertEqual(len(record["trajectories"]), 1)
        trajectory = record["trajectories"][0]
        self.assertEqual(trajectory["source_model"], "openai/gpt-5.4")
        self.assertIn("get_flight", trajectory["trajectory_text"])
        self.assertEqual(trajectory["metadata"]["user_model"], "openai/gpt-5.4-mini")
        self.assertEqual(trajectory["metadata"]["score"], 1.0)
        self.assertEqual(trajectory["metadata"]["raw_file"], "raw/results.json")

    def test_groups_benchmark_records_by_task_id(self):
        first = {
            "run_id": "run_1",
            "domain": "airline",
            "task_id": "0",
            "trial_id": 0,
            "agent_model": "openai/gpt-5.4",
            "user_model": "openai/gpt-5.4",
            "task": {"instruction": "Original instruction."},
            "events": [{"index": 0, "role": "user", "type": "message", "content": "A"}],
            "raw_file": "raw/results.json",
        }
        second = {
            **first,
            "run_id": "run_2",
            "trial_id": 1,
            "events": [{"index": 0, "role": "user", "type": "message", "content": "B"}],
        }

        records = build_benchmark_records([first, second], benchmark="tau2-bench")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["task_id"], "0")
        self.assertEqual(records[0]["task"]["instruction"], "Original instruction.")
        self.assertEqual(len(records[0]["trajectories"]), 2)
        self.assertIn("USER: A", records[0]["trajectories"][0]["trajectory_text"])
        self.assertIn("USER: B", records[0]["trajectories"][1]["trajectory_text"])
        self.assertNotIn("[0]", records[0]["trajectories"][0]["trajectory_text"])
        self.assertNotIn("[0]", records[0]["trajectories"][1]["trajectory_text"])

    def test_collects_tau2_task_instruction_from_parent_results_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            raw_path = raw_dir / "results.json"
            raw_payload = {
                "tasks": [
                    {
                        "id": "0",
                        "description": {"purpose": "Update an airline reservation."},
                        "user_scenario": {
                            "instructions": {
                                "reason_for_call": "Change my flight.",
                                "known_info": "Reservation code is ABC123.",
                                "task_instructions": "Move the traveler to an earlier flight.",
                            }
                        },
                    }
                ],
                "simulations": [
                    {
                        "task_id": "0",
                        "trial_id": 0,
                        "messages": [{"role": "user", "content": "I need an earlier flight."}],
                    }
                ],
            }
            raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")

            trajectories = collect_raw_files(
                raw_files=[raw_path],
                trajectories_dir=Path(tmp) / "trajectories",
                jsonl_path=Path(tmp) / "trajectories.jsonl",
                run_id="run_tau2",
                defaults={"agent_model": "gpt-5.5", "user_model": "gpt-5.5", "domain": "airline"},
                raw_root=Path(tmp),
            )

            self.assertEqual(len(trajectories), 1)
            self.assertEqual(trajectories[0]["task"]["instruction"], (
                "Purpose: Update an airline reservation.\n"
                "Reason for call: Change my flight.\n"
                "Known info: Reservation code is ABC123.\n"
                "Task instructions: Move the traveler to an earlier flight."
            ))

    def test_collects_terminal_bench_atif_steps_and_adjacent_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_dir = root / "raw" / "job" / "make-mips-interpreter__abc123"
            agent_dir = trial_dir / "agent"
            agent_dir.mkdir(parents=True)
            trajectory_path = agent_dir / "trajectory.json"
            trajectory_path.write_text(
                json.dumps(
                    {
                        "schema_version": "ATIF-v1.7",
                        "agent": {"name": "terminus-2", "model_name": "openai/gpt-5.4"},
                        "steps": [
                            {
                                "step_id": 1,
                                "source": "user",
                                "message": (
                                    "Task Description:\n"
                                    "Create /app/vm.js and boot Doom.\n\n"
                                    "Current terminal state:\nroot@container:/app#"
                                ),
                            },
                            {
                                "step_id": 2,
                                "source": "agent",
                                "message": "I will inspect the workspace.",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "bash_command",
                                        "arguments": {"keystrokes": "ls -la\n"},
                                    }
                                ],
                                "observation": {"results": [{"content": "doomgeneric_mips\n"}]},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": "terminal-bench/make-mips-interpreter",
                        "trial_name": "make-mips-interpreter__abc123",
                        "agent_info": {
                            "model_info": {"provider": "openai", "name": "gpt-5.4"},
                        },
                        "verifier_result": {"rewards": {"reward": 0.0}},
                    }
                ),
                encoding="utf-8",
            )

            trajectories = collect_raw_files(
                raw_files=[trajectory_path],
                trajectories_dir=root / "trajectories",
                jsonl_path=root / "trajectories.jsonl",
                run_id="run_terminal",
                defaults={"agent_model": "openai/gpt-5.4", "user_model": None, "domain": "terminal-bench/terminal-bench-2"},
                raw_root=root,
            )

            self.assertEqual(len(trajectories), 1)
            normalized = trajectories[0]
            self.assertEqual(normalized["task_id"], "make-mips-interpreter")
            self.assertEqual(normalized["trial_id"], "make-mips-interpreter__abc123")
            self.assertEqual(normalized["score"], 0.0)
            self.assertFalse(normalized["success"])
            self.assertEqual(normalized["task"]["instruction"], "Create /app/vm.js and boot Doom.")
            self.assertEqual(normalized["num_tool_calls"], 1)
            self.assertGreaterEqual(normalized["num_turns"], 3)

            text = serialize_trajectory_text(normalized)
            self.assertIn("Create /app/vm.js and boot Doom.", text)
            self.assertIn("ASSISTANT TOOL_CALL bash_command", text)
            self.assertIn("doomgeneric_mips", text)


if __name__ == "__main__":
    unittest.main()
