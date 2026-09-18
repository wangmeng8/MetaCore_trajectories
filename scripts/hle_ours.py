"""Fourth-row server/evaluator entry point. Import/plan are offline; serve/run are explicit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.hle_outcome_caa import MODELS, sha256, vector_metadata, without_steering, selected_dataset
from scripts.prepare_hle_outcome_caa import request_body
from scripts.runner_common import load_dotenv_file


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError(f"Existing file differs; choose a new run/service name: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def profile(key):
    p = read(ROOT / "scripts/ours_profiles" / f"{key}.json")
    if p["model_key"] != key:
        raise ValueError("Profile model mismatch")
    return p


def vector_path(key):
    return ROOT / "data/HLE/ours_steering_vectors" / key / "benchmark_global/steering_vector.pt"


def service_path(key, name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("service-name must contain letters, digits, underscores or hyphens")
    return ROOT / "outputs/ours_service" / f"{key}_{name}" / "service.json"


def serving_plan(key, name, model_path=None):
    p = profile(key)
    meta, _, _ = vector_metadata(vector_path(key), key)
    if meta.get("method") != "trajre_mvp":
        raise ValueError("Ours requires a TrajRE vector")
    env = dict(os.environ)
    path = service_path(key, name)
    env.update(OUTCOME_CAA_MODEL_KEY=key, OUTCOME_CAA_VECTOR_PATH=str(vector_path(key)),
               OUTCOME_CAA_VECTOR_ROOT=str(ROOT / "data/HLE/ours_steering_vectors"),
               OUTCOME_CAA_SERVICE_MANIFEST=str(path), OUTCOME_CAA_AUDIT_DIR=str(path.parent / "worker_audit"),
               MODEL_PATH=str(model_path or p["model_path"]), SERVED_MODEL_NAME=p["model"],
               TENSOR_PARALLEL_SIZE=str(p["tp"]), HOST=p["host"], PORT=str(p["port"]),
               VLLM_VANILLA_ARGS_FILE=str(ROOT / "scripts/ours_profiles" / f"{key}_serve_args.json"),
               VLLM_HOOK_PYTHON=sys.executable, VLLM_BIN=str(Path(sys.executable).with_name("vllm")))
    return ["bash", str(ROOT / "scripts/serve_hle_outcome_caa.sh")], env


def parameter_tag(value):
    """Compact filesystem-safe tag used by the existing l30/a1 run naming scheme."""
    return format(value, ".12g").replace("-", "m").replace(".", "")


def evaluation_plan(key, name, mode, run_id=None, data_path=None, layer=None, alpha=1.0):
    p = profile(key)
    env = dict(os.environ)
    # Prevent prior smoke/retry/Prompt-Reg variables or .env values from changing row four.
    env.update(HLE_SYSTEM_PROMPT_FILE="", HLE_MARK_INCORRECT_IDS="", HLE_NUM_TASKS="",
               HLE_TEMPERATURE="" if p["budget"]["temperature"] is None else str(p["budget"]["temperature"]),
               HLE_SKIP_FAILED_ON_RESUME="0", HLE_WITH_TOOLS_TEXT_ONLY="1",
               HLE_DISABLE_SCIENTIFIC_SEARCH="1" if p["budget"]["disable_scientific_search"] else "0",
               HLE_WITH_TOOLS_EXTRA_BODY_JSON="{}", HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON=json.dumps(p["extra_body"]),
               HLE_WITH_TOOLS_AUXILIARY_EXTRA_BODY_JSON=json.dumps(p["extra_body"]),
               OUTCOME_CAA_MODEL_KEY=key, OUTCOME_CAA_SERVICE_MANIFEST=str(service_path(key, name)),
               HLE_ALL_TASKS="1" if mode == "full" else "0")
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    selected_layer = MODELS[key][0] if layer is None else layer
    body = request_body(vector_path(key), selected_layer, alpha, env)
    dataset = Path(data_path).resolve() if data_path else ROOT / "data/HLE/WebThinker_test_500_hle_with_tools.parquet"
    if not data_path and sha256(dataset) != p["dataset_sha256"]:
        raise ValueError("Dataset differs from the third-row full500 run")
    rid = run_id or (
        f"hle_{key}_ours_l{selected_layer}_a{parameter_tag(alpha)}_"
        f"{'smoke3' if mode == 'smoke' else 'full500'}_{name}"
    )
    if Path(rid).name != rid or '/' in rid or '\\' in rid or rid in ('.', '..'):
        raise ValueError("run-id must be a directory name")
    command = [sys.executable, str(ROOT / "scripts/run_hle_with_tools.py"),
               "--repo-dir", str(ROOT / ".external/hle_with_tools"), "--data-path", str(dataset),
               "--model", p["model"], "--base-url", p["base_url"], "--run-id", rid,
               "--output-dir", str(ROOT / "outputs/runs"), "--no-uv", "--agent-extra-body-json", json.dumps(body)]
    for k,v in p["budget"].items():
        if k in ("temperature", "disable_scientific_search"):
            continue
        command.extend(["--"+k.replace("_", "-"), str(v)])
    if p["budget"]["temperature"] is not None:
        command.extend(["--temperature", str(p["budget"]["temperature"])])
    command.extend(["--num-tasks", "3"] if mode == "smoke" else ["--all-tasks"])
    # Validate actual selected dataset before any model request.
    from types import SimpleNamespace
    selected = selected_dataset(SimpleNamespace(data_path=dataset, text_only=True, num_tasks=3 if mode == "smoke" else None))
    if not data_path and selected["task_count"] != (3 if mode == "smoke" else 500):
        raise ValueError("Unexpected selected question count")
    return command, env, rid


def audit_smoke(run_dir, service, before):
    """Check fresh worker events during this isolated smoke; no extra model calls."""
    mf = read(run_dir / "manifest.json")
    identity = mf["outcome_caa"]
    if identity["table_method"] != "REVEAL (Ours)" or identity["dataset"]["task_count"] != 3:
        raise ValueError("Expected an Ours three-question smoke")
    pred = read(next((run_dir / "raw/official_run").glob("hle_*.json")))
    if set(pred) != set(identity["dataset"]["task_ids"]) or any(v.get("error") or not v.get("response", "").strip() for v in pred.values()):
        raise ValueError("Smoke has missing/failed/empty predictions; inspect run logs")
    groups = {}
    for path in Path(service["audit_dir"]).glob("worker-*.jsonl"):
        with path.open("rb") as f:
            f.seek(before.get(path.name, 0))
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("event") != "last_prefix":
                    continue
                if (e.get("vector_sha256") != identity["vector_sha256"] or e.get("layer") != identity["layer"]
                        or e.get("alpha") != identity["alpha"] or e.get("applied") is not True
                        or e.get("token_position") != e.get("prefix_length", 0)-1):
                    raise ValueError("Fresh audit contains wrong vector/layer/alpha/position")
                groups.setdefault(e["request_id"], set()).add(e["rank"])
    ranks = set(range(service["tensor_parallel_size"]))
    if len(groups) < 3 or any(r != ranks for r in groups.values()):
        raise ValueError("Missing full TP-rank audit evidence")
    result = {"passed": True, "table_method": identity["table_method"], "vector_sha256": identity["vector_sha256"],
              "layer": identity["layer"], "alpha": identity["alpha"], "audited_requests": len(groups), "questions": 3,
              "scope": "Fresh events during an isolated smoke; run no other evaluator against this service during smoke"}
    (run_dir / "ours_smoke_audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    imp = sub.add_parser("import-vectors")
    imp.add_argument("--archive", type=Path, required=True)
    for action in ("serve", "smoke", "full"):
        p = sub.add_parser(action)
        p.add_argument("--model-key", choices=MODELS, required=True)
        p.add_argument("--service-name", default="v1")
        p.add_argument("--plan", action="store_true", help="Offline preview only; never launch server/evaluator")
        if action == "serve":
            p.add_argument("--model-path", type=Path)
        else:
            p.add_argument("--run-id")
            p.add_argument("--data-path", type=Path, help="Explicit recovery dataset; requires --run-id")
            p.add_argument("--layer", type=int, help="Zero-based decoder block; defaults to the original controlled setting")
            p.add_argument("--alpha", type=float, default=1.0, help="Steering coefficient")
    args = parser.parse_args()
    if args.action == "import-vectors":
        from scripts.hle_ours_vectors import import_vectors
        import_vectors(args.archive, ROOT / "data/HLE/ours_steering_vectors", ROOT / "data/HLE/outcome_caa_vectors")
        return 0
    load_dotenv_file(ROOT / ".env")
    if args.action == "serve":
        cmd, env = serving_plan(args.model_key, args.service_name, args.model_path)
        if args.plan:
            print(json.dumps({"command": cmd, "service":env["OUTCOME_CAA_SERVICE_MANIFEST"], "model_path":env["MODEL_PATH"], "args":read(env["VLLM_VANILLA_ARGS_FILE"])}, indent=2))
            return 0
    else:
        if args.data_path and not args.run_id:
            parser.error("--data-path requires a new explicit --run-id")
        cmd, env, rid = evaluation_plan(
            args.model_key, args.service_name, args.action, args.run_id, args.data_path,
            args.layer, args.alpha,
        )
        if args.plan:
            print(json.dumps({"command": cmd, "service":env["OUTCOME_CAA_SERVICE_MANIFEST"],
                              "layer": args.layer if args.layer is not None else MODELS[args.model_key][0],
                              "alpha": args.alpha}, indent=2))
            return 0
        if args.action == "smoke":
            run = ROOT / "outputs/runs" / rid
            if run.exists():
                raise ValueError("Smoke evidence must be fresh; choose a new --run-id")
            service = read(env["OUTCOME_CAA_SERVICE_MANIFEST"])
            before = {p.name:p.stat().st_size for p in Path(service["audit_dir"]).glob("worker-*.jsonl")}
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    if args.action == "smoke" and rc == 0:
        audit_smoke(run, service, before)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
