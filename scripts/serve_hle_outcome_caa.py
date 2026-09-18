"""Container-side launch validation. Called only by serve_hle_outcome_caa.sh."""
from __future__ import annotations

import importlib.metadata
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.hle_outcome_caa import HOOK_COMMIT, sha256, vector_metadata, verify_checkpoint


def serving_args(vanilla):
    if not isinstance(vanilla, list) or not all(isinstance(x, str) for x in vanilla):
        raise ValueError("VLLM_VANILLA_ARGS_FILE must contain a JSON string array")
    forbidden = {"--enable-prefix-caching", "--no-enable-prefix-caching", "--async-scheduling", "--no-async-scheduling",
                 "--speculative-config", "--worker-extension-cls", "--worker-cls", "--pipeline-parallel-size", "-pp",
                 "--tensor-parallel-size", "-tp", "--served-model-name", "--model", "--host", "--port",
                 "--enable-dbo", "--decode-context-parallel-size", "-dcp", "--prefill-context-parallel-size", "-pcp",
                 "--enforce-eager", "--no-enforce-eager"}
    for item in vanilla:
        flag = item.split("=", 1)[0]
        if flag in forbidden:
            raise ValueError(f"Reserved/unsupported serving option in Vanilla args: {flag}")
    if not any(x == "--max-model-len" or x.startswith("--max-model-len=") for x in vanilla):
        raise ValueError("Explicit Vanilla --max-model-len required; do not guess a cross-model context budget")
    return list(vanilla)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true",
                        help="Validate and write service.json without starting vLLM")
    parser.add_argument("--vanilla-args", nargs=argparse.REMAINDER,
                        help="Explicit serving options instead of VLLM_VANILLA_ARGS_FILE; put last")
    options = parser.parse_args(argv)
    env = os.environ
    # The CAA worker reads V1 input_batch/requests/query_start_loc. vLLM 0.28
    # may select the incompatible V2 model runner by default for some models.
    # Set this before importing vLLM/plugins and inherit it in the exec'd server.
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    vanilla = serving_args(options.vanilla_args if options.vanilla_args is not None else
                           json.loads(Path(env["VLLM_VANILLA_ARGS_FILE"]).read_text(encoding="utf-8")))
    vector = Path(env["OUTCOME_CAA_VECTOR_PATH"]).resolve(strict=True)
    root = Path(env["OUTCOME_CAA_VECTOR_ROOT"]).resolve(strict=True)
    if not vector.is_relative_to(root):
        raise ValueError("Vector path is outside read-only vector root")
    metadata, digest, metadata_digest = vector_metadata(vector, env.get("OUTCOME_CAA_MODEL_KEY"))
    checkpoint = verify_checkpoint(env["MODEL_PATH"], metadata)
    from vllm_hook_plugins.workers import steer_activation_worker as worker
    if getattr(worker, "OUTCOME_CAA_VERSION", None) != "hle-outcome-caa-v2":
        raise ValueError("Extended worker missing; run prepare_hle_vllm_hook.sh in serving environment")
    installed_sha = sha256(worker.__file__)
    if installed_sha != sha256(ROOT / "patches/vllm_hook/steer_activation_worker.py"):
        raise ValueError("Installed worker differs from this project's patch")
    versions = {name: importlib.metadata.version(name) for name in ("vllm", "torch", "vllm-hook-plugins", "safetensors", "zstandard")}
    import vllm
    source_root = Path(vllm.__file__).parent
    source_files = ["model_executor/models/qwen3_5.py", "model_executor/models/qwen3_next.py",
                    "model_executor/models/gpt_oss.py", "model_executor/models/gemma4.py",
                    "model_executor/layers/layernorm.py", "v1/worker/gpu_model_runner.py",
                    "v1/worker/gpu_input_batch.py"]
    source_hashes = {f: sha256(source_root / f) for f in source_files if (source_root / f).is_file()}
    if versions["zstandard"] != "0.23.0":
        raise ValueError("Require zstandard==0.23.0 in serving environment")
    tp = int(env.get("TENSOR_PARALLEL_SIZE", "2"))
    if tp < 1:
        raise ValueError("TP must be positive")
    args = ["serve", env["MODEL_PATH"], "--served-model-name", env["SERVED_MODEL_NAME"],
            "--tensor-parallel-size", str(tp), "--host", env.get("HOST", "127.0.0.1"),
            "--port", env.get("PORT", "8101"), *vanilla,
            "--enforce-eager", "--no-enable-prefix-caching", "--no-async-scheduling"]
    manifest = {"worker_version": worker.OUTCOME_CAA_VERSION, "worker_sha256": installed_sha,
                "hook_commit": HOOK_COMMIT, "worker_patch_sha256": sha256(ROOT / "patches/vllm_hook/0001-hle-outcome-caa-worker.patch"),
                "versions": versions, "vllm_source_sha256": source_hashes,
                "model_runner": {"VLLM_USE_V2_MODEL_RUNNER": "0",
                                 "class": "vllm.v1.worker.gpu_model_runner.GPUModelRunner"},
                "model_key": metadata["model_key"], "model_path": env["MODEL_PATH"],
                "served_model_name": env["SERVED_MODEL_NAME"], "checkpoint_verification": checkpoint,
                "vector_sha256": digest, "prefix_caching": False, "async_scheduling": False,
                "tensor_parallel_size": tp, "arguments": args,
                "image_reference": env.get("OUTCOME_CAA_IMAGE_REFERENCE", "not supplied; use recorded package versions"),
                "audit_dir": env["OUTCOME_CAA_AUDIT_DIR"]}
    path = Path(env["OUTCOME_CAA_SERVICE_MANIFEST"])
    if metadata.get("method") == "trajre_mvp":
        manifest.update(method="TrajRE-MVP", table_method="REVEAL (Ours)", tensor_key="steering_vector",
                        vector_metadata_sha256=metadata_digest,
                        deployment_reference_sha256=metadata["deployment_reference_sha256"],
                        extraction_checkpoint_verified=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Service manifest exists with different settings; choose a new path")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Service identity:", path, flush=True)
    print("CAA model runner: V1 (VLLM_USE_V2_MODEL_RUNNER=0)", flush=True)
    if options.manifest_only:
        print("Manifest prepared; vLLM has not been started.", flush=True)
        return
    print("Launching:", shlex.join([env.get("VLLM_BIN", "vllm"), *args]), flush=True)
    os.execvp(env.get("VLLM_BIN", "vllm"), [env.get("VLLM_BIN", "vllm"), *args])


if __name__ == "__main__":
    main()
