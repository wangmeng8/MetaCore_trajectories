"""Prepare and optionally run MetaCoreBench representation visualizations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.runner_common import prepare_subprocess_env, shell_join


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_REPO = ROOT / ".external" / "MetaCoreBench-visualization"
DEFAULT_MODEL_PATH = "/data/oss_bucket_0/modelscope/hub/models/Qwen/Qwen3.6-27B"
ALLOWED_VISUALIZATION_BENCHMARKS = {
    "tau2_airline",
    "tau2_retail",
    "tau2_telecom",
    "hle_with_tools",
    "terminal_bench_2",
}
COLLECTION_BENCHMARK_MAP = {
    "hle-with-tools": "hle_with_tools",
    "hle_with_tools": "hle_with_tools",
    "terminal-bench-2.0": "terminal_bench_2",
    "terminal-bench-2": "terminal_bench_2",
    "terminal_bench_2": "terminal_bench_2",
}
TAU2_DOMAINS = {"airline", "retail", "telecom"}


def infer_visualization_benchmark(collection_benchmark: str | None, domain: str | None) -> str:
    """Map collection benchmark labels to the visualization repo's labels."""

    normalized = _normalize_label(collection_benchmark)
    if normalized in ALLOWED_VISUALIZATION_BENCHMARKS:
        return normalized
    if normalized in COLLECTION_BENCHMARK_MAP:
        return COLLECTION_BENCHMARK_MAP[normalized]
    if normalized in {"tau2-bench", "tau2_bench", "tau2"} or _normalize_label(domain) in TAU2_DOMAINS:
        tau2_domain = _normalize_label(domain) or "airline"
        if tau2_domain not in TAU2_DOMAINS:
            allowed = ", ".join(sorted(TAU2_DOMAINS))
            raise ValueError(f"Unsupported Tau2 domain for visualization: {domain!r}. Expected one of: {allowed}.")
        return f"tau2_{tau2_domain}"
    allowed = ", ".join(sorted(ALLOWED_VISUALIZATION_BENCHMARKS))
    raise ValueError(
        f"Cannot map collection benchmark {collection_benchmark!r} to MetaCoreBench visualization. "
        f"Pass --benchmark explicitly. Allowed values: {allowed}."
    )


def latest_run_with_benchmark_trajectories(runs_root: Path) -> Path:
    """Return the newest run directory containing benchmark_trajectories.jsonl."""

    root = Path(runs_root)
    candidates: list[Path] = []
    if (root / "benchmark_trajectories.jsonl").is_file():
        candidates.append(root)
    if root.exists():
        candidates.extend(path.parent for path in root.glob("*/benchmark_trajectories.jsonl") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No benchmark_trajectories.jsonl found under {root}.")
    return max(candidates, key=lambda path: ((path / "benchmark_trajectories.jsonl").stat().st_mtime_ns, path.name))


def build_visualization_config(
    *,
    run_dir: Path,
    benchmark: str,
    model_path: str,
    output_root: Path,
    max_length: int = 24576,
    gpu_numbers: str = "0",
    backend: str = "direct",
    save_representations: bool = False,
    save_activation_matrix: bool = False,
    stage_to_ram: bool = True,
    ram_cache_dir: str = "/dev/shm/metacorebench/models",
    service_port: int = 8010,
    resume: bool = True,
) -> dict[str, Any]:
    input_path = Path(run_dir) / "benchmark_trajectories.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing visualization input: {input_path}")
    if benchmark not in ALLOWED_VISUALIZATION_BENCHMARKS:
        allowed = ", ".join(sorted(ALLOWED_VISUALIZATION_BENCHMARKS))
        raise ValueError(f"Unsupported visualization benchmark: {benchmark}. Expected one of: {allowed}.")

    return {
        "run": {
            "name": f"{Path(run_dir).name}_visualization",
            "output_dir": str(Path(output_root).resolve()),
        },
        "model": {
            "model_name_or_path": model_path,
            "dtype": "bf16",
            "device_map": "auto",
            "trust_remote_code": True,
            "gpu_numbers": gpu_numbers,
            "backend": backend,
            "stage_to_ram": stage_to_ram,
            "ram_cache_dir": ram_cache_dir,
            "service_host": "127.0.0.1",
            "service_port": service_port,
        },
        "data": {
            "benchmark": benchmark,
            "input_dir": str(input_path.parent.resolve()),
            "input_path": str(input_path.resolve()),
        },
        "extraction": {
            "max_length": max_length,
            "resume": resume,
            "save_representations": save_representations,
        },
        "visualization": {
            "save_activation_matrix": save_activation_matrix,
        },
    }


def write_visualization_config(path: Path, config: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_pipeline_command(*, external_repo: Path, config_path: Path, python_bin: str) -> list[str]:
    _ = external_repo
    return [python_bin, "scripts/step_01_extract_transformers.py", str(config_path)]


def build_validate_command(*, bundle_dir: Path, python_bin: str) -> list[str]:
    return [python_bin, "scripts/step_02_validate_bundle.py", str(bundle_dir)]


def build_projection_command(*, bundle_dir: Path, python_bin: str) -> list[str]:
    return [python_bin, "scripts/step_05_plot_projection.py", str(bundle_dir), "--layer", "-1"]


def run_pipeline(
    *,
    external_repo: Path,
    config_path: Path,
    config: dict[str, Any],
    python_bin: str,
    run_extract: bool,
    install_deps: bool,
) -> None:
    repo = Path(external_repo)
    if not repo.is_dir():
        raise FileNotFoundError(f"Missing MetaCoreBench visualization repo: {repo}")

    env = prepare_subprocess_env(os.environ)
    env["PYTHONPATH"] = _prepend_pythonpath(repo, env.get("PYTHONPATH"))

    if install_deps:
        _run([python_bin, "-m", "pip", "install", "-e", "."], cwd=repo, env=env)

    bundle_dir = resolved_bundle_dir(config)
    if run_extract:
        _run(build_pipeline_command(external_repo=repo, config_path=config_path, python_bin=python_bin), cwd=repo, env=env)
    _run(build_validate_command(bundle_dir=bundle_dir, python_bin=python_bin), cwd=repo, env=env)

    if config["extraction"]["save_representations"] and _trajectory_count(bundle_dir) >= 2:
        _run(build_projection_command(bundle_dir=bundle_dir, python_bin=python_bin), cwd=repo, env=env)


def resolved_bundle_dir(config: dict[str, Any]) -> Path:
    return Path(config["run"]["output_dir"]) / config["data"]["benchmark"]


def load_run_hints(run_dir: Path) -> dict[str, str | None]:
    hints: dict[str, str | None] = {"benchmark": None, "domain": None}
    manifest = _read_json_if_exists(Path(run_dir) / "manifest.json")
    if isinstance(manifest, dict):
        hints["benchmark"] = _string_or_none(manifest.get("benchmark"))
        config = manifest.get("config")
        if isinstance(config, dict):
            hints["domain"] = _string_or_none(config.get("domain") or config.get("TAU2_DOMAIN"))

    first_record = _read_first_jsonl(Path(run_dir) / "benchmark_trajectories.jsonl")
    if isinstance(first_record, dict):
        hints["benchmark"] = hints["benchmark"] or _string_or_none(first_record.get("benchmark"))
        trajectories = first_record.get("trajectories")
        if isinstance(trajectories, list) and trajectories:
            metadata = trajectories[0].get("metadata") if isinstance(trajectories[0], dict) else None
            if isinstance(metadata, dict):
                hints["domain"] = hints["domain"] or _string_or_none(metadata.get("domain"))

    if not hints["domain"]:
        run_name = Path(run_dir).name.lower()
        for domain in sorted(TAU2_DOMAINS):
            if domain in run_name:
                hints["domain"] = domain
                break
    return hints


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate config and optionally run MetaCoreBench visualizations for collected trajectories."
    )
    parser.add_argument("--run-dir", type=Path, help="Run directory containing benchmark_trajectories.jsonl.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "outputs" / "runs",
        help="Directory scanned when --run-dir is omitted.",
    )
    parser.add_argument("--external-repo", type=Path, default=DEFAULT_EXTERNAL_REPO)
    parser.add_argument("--benchmark", choices=sorted(ALLOWED_VISUALIZATION_BENCHMARKS))
    parser.add_argument("--collection-benchmark", help="Original collection benchmark label, if auto-detection is wrong.")
    parser.add_argument("--domain", help="Original benchmark domain, for example airline, retail, or telecom.")
    parser.add_argument("--model-path", default=os.environ.get("METACOREBENCH_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--output-root", type=Path, help="Visualization output root. Defaults inside the selected run dir.")
    parser.add_argument("--config-out", type=Path, help="Generated config path. Defaults inside the selected run dir.")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--gpu-numbers", default="0")
    parser.add_argument("--backend", choices=("direct", "service"), default="direct")
    parser.add_argument("--service-port", type=int, default=8010)
    parser.add_argument("--ram-cache-dir", default="/dev/shm/metacorebench/models")
    parser.add_argument("--no-stage-to-ram", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-representations", action="store_true")
    parser.add_argument("--save-activation-matrix", action="store_true")
    parser.add_argument("--skip-extract", action="store_true", help="Only validate/plot an already extracted bundle.")
    parser.add_argument("--install-deps", action="store_true", help="Run python -m pip install -e . in the visualization repo.")
    parser.add_argument("--execute", action="store_true", help="Run the extraction/visualization commands after writing config.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = (args.run_dir or latest_run_with_benchmark_trajectories(args.runs_root)).resolve()
    hints = load_run_hints(run_dir)
    benchmark = args.benchmark or infer_visualization_benchmark(
        args.collection_benchmark or hints["benchmark"],
        args.domain or hints["domain"],
    )
    output_root = (args.output_root or (run_dir / "metacorebench_results")).resolve()
    config_path = (args.config_out or (run_dir / "metacorebench_visualization.yaml")).resolve()
    config = build_visualization_config(
        run_dir=run_dir,
        benchmark=benchmark,
        model_path=args.model_path,
        output_root=output_root,
        max_length=args.max_length,
        gpu_numbers=args.gpu_numbers,
        backend=args.backend,
        save_representations=args.save_representations,
        save_activation_matrix=args.save_activation_matrix,
        stage_to_ram=not args.no_stage_to_ram,
        ram_cache_dir=args.ram_cache_dir,
        service_port=args.service_port,
        resume=not args.no_resume,
    )
    write_visualization_config(config_path, config)

    bundle_dir = resolved_bundle_dir(config)
    print(f"Run directory: {run_dir}")
    print(f"Visualization repo: {args.external_repo.resolve()}")
    print(f"Config written: {config_path}")
    print(f"Bundle output: {bundle_dir}")

    if not args.execute:
        command = build_pipeline_command(external_repo=args.external_repo, config_path=config_path, python_bin=args.python_bin)
        print("Dry run. Add --execute to run:")
        print(f"  cd {args.external_repo.resolve()}")
        print(f"  {shell_join(command)}")
        print(f"  {shell_join(build_validate_command(bundle_dir=bundle_dir, python_bin=args.python_bin))}")
        return 0

    run_pipeline(
        external_repo=args.external_repo.resolve(),
        config_path=config_path,
        config=config,
        python_bin=args.python_bin,
        run_extract=not args.skip_extract,
        install_deps=args.install_deps,
    )
    print(f"Done. Outputs are under: {bundle_dir}")
    return 0


def _normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _read_json_if_exists(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_first_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    return json.loads(stripped)
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _prepend_pythonpath(path: Path, existing: str | None) -> str:
    parts = [str(path.resolve())]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _trajectory_count(bundle_dir: Path) -> int:
    inputs = Path(bundle_dir) / "inputs.jsonl"
    if not inputs.is_file():
        return 0
    return sum(1 for line in inputs.read_text(encoding="utf-8").splitlines() if line.strip())


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(shell_join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
