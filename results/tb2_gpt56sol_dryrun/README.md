# Terminal-Bench 2.0 Trajectory Run — tb2_gpt56sol_dryrun

## Run configuration

- **Dataset**: `terminal-bench/terminal-bench-2`
- **Agent**: `terminus-2`
- **Model**: `openai/gpt-5.6-sol`
- **Task**: `openssl-selfsigned-cert`
- **Endpoint**: `https://evamux.alibaba-inc.com/v1`
- **Run ID**: `tb2_gpt56sol_dryrun`
- **Date**: 2026-08-24

## Command

```bash
harbor run -d terminal-bench/terminal-bench-2 \
  -a terminus-2 \
  -m openai/gpt-5.6-sol \
  -k 1 \
  -o jobs \
  --include-task-name terminal-bench/openssl-selfsigned-cert
```

## Summary

```json
{
  "total_trajectories": 1,
  "success_count": 1,
  "success_rate": 1.0,
  "average_turns": 7.0,
  "average_tool_calls": 9.0,
  "average_score": 1.0,
  "infrastructure_error_count": 0,
  "benchmark_task_count": 1,
  "benchmark_trajectory_count": 1
}
```

## Files

- `manifest.json` — run metadata and Harbor command
- `summary.json` — aggregate statistics
- `trajectories.jsonl` — normalized per-trial trajectories
- `benchmark_trajectories.jsonl` — benchmark-style records grouped by task
- `trajectories/` — one JSON file per trajectory
- `benchmark_trajectories/` — one JSON file per benchmark task

## Notes

- The run was executed locally on macOS using Docker Desktop + Colima.
- The OpenAI-compatible endpoint was `https://evamux.alibaba-inc.com/v1`.
- No API key or credential is committed in this repository.
