"""Offline provenance and resume guards; this module never calls a model API."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

HOOK_COMMIT = "e3d6885899264c81ac61c403c8b56efaf5a02dab"
MODELS = {"qwen36_27b": (23, [64, 5120]), "gemma4_31b_it": (30, [60, 5376]),
          "gptoss_20b": (12, [24, 2880])}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(value):
    obj = json.loads(value) if isinstance(value, str) else value
    if not isinstance(obj, dict):
        raise ValueError("extra_body must be a JSON object")
    return obj


def deep_merge(base, overlay):
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        out[key] = deep_merge(out[key], value) if isinstance(out.get(key), dict) and isinstance(value, dict) else copy.deepcopy(value)
    return out


def merged_agent_json(env):
    base = env.get("HLE_WITH_TOOLS_EXTRA_BODY_JSON")
    agent = env.get("HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON")
    if base and agent:
        return json.dumps(deep_merge(json_object(base), json_object(agent)))
    return agent or base


def without_steering(body):
    out = copy.deepcopy(body)
    if isinstance(out.get("vllm_xargs"), dict):
        out["vllm_xargs"].pop("steer", None)
        if not out["vllm_xargs"]:
            del out["vllm_xargs"]
    return out


def vector_metadata(vector_path, model_key=None):
    path = Path(vector_path).resolve(strict=True)
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("method") == "trajre_mvp":
        from scripts.hle_ours_vectors import validate_ours_vector
        return validate_ours_vector(path, metadata, model_key)
    if metadata.get("benchmark") != "HLE with Tools" or metadata.get("kind") != "benchmark_global":
        raise ValueError("Require the target model's HLE benchmark_global vector")
    key = metadata.get("model_key")
    if key not in MODELS or (model_key and key != model_key):
        raise ValueError("Vector model_key does not match target model")
    if metadata.get("layer_index_base") != 0 or metadata.get("shape") != MODELS[key][1]:
        raise ValueError("Unexpected vector dimensions or layer indexing")
    if not metadata.get("capture_site") or not metadata.get("model_fingerprint", {}).get("file_hashes"):
        raise ValueError("Vector lacks capture site or model/tokenizer fingerprint")
    # Validate supplied sidecars too, not merely a cached digest in a manifest.
    sums = path.with_name("SHA256SUMS.txt")
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            expected, filename = line.split(maxsplit=1)
            target = (path.parent / filename.lstrip("*")).resolve()
            if not target.is_relative_to(path.parent) or sha256(target) != expected:
                raise ValueError(f"Vector bundle checksum mismatch: {filename}")
    return metadata, sha256(path), sha256(metadata_path)


def verify_checkpoint(model_path, metadata):
    """No weights loaded. Check config/tokenizer/index hashes and shard byte sizes.

    Source archives do not contain weight-content hashes. Do not pretend that
    copied mtime values provide full weight verification.
    """
    root = Path(model_path).resolve(strict=True)
    fingerprint = metadata.get("model_fingerprint") or metadata["deployment_reference"]["model_fingerprint"]
    for name, expected in fingerprint["file_hashes"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Model/tokenizer fingerprint mismatch: {name}")
    for shard in fingerprint.get("weight_shards", []):
        path = (root / shard["name"]).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != shard["bytes"]:
            raise ValueError(f"Model shard size mismatch: {shard['name']}")
    result = {"file_hashes": fingerprint["file_hashes"], "weight_shards": fingerprint.get("weight_shards", []),
              "verification_scope": "config/tokenizer/index SHA256 and shard sizes; weight contents not hashed"}
    if metadata.get("method") == "trajre_mvp":
        result.update(reference="previous Outcome-CAA checkpoint; NOT the new vector extraction fingerprint",
                      extraction_checkpoint_verified=False,
                      deployment_reference_sha256=metadata["deployment_reference_sha256"])
    return result


def selected_dataset(config):
    if config.data_path is None or config.data_path.suffix.lower() not in (".json", ".jsonl", ".parquet"):
        raise ValueError("Outcome-CAA requires a local Parquet/JSON/JSONL dataset for exact ID auditing")
    suffix = config.data_path.suffix.lower()
    if suffix == ".parquet":
        # Use the same loader as hle_eval.run_agent_predictions, preserving row order
        # and image representations before applying its text-only/task-limit rules.
        from datasets import load_dataset

        records = list(load_dataset("parquet", data_files=str(config.data_path), split="train"))
    else:
        text = config.data_path.read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()] if suffix == ".jsonl" else json.loads(text)
    if not isinstance(records, list):
        raise ValueError("Outcome-CAA dataset must be a JSON array or JSONL records")
    if config.text_only:
        records = [r for r in records if all(r.get(k) in (None, "", [], {}) for k in ("image", "image_preview", "rationale_image"))]
    if config.num_tasks is not None:
        records = records[:config.num_tasks]
    ids = [str(r["id"]) for r in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selected dataset is empty or contains duplicate IDs")
    return {"sha256": sha256(config.data_path), "task_ids": ids, "task_count": len(ids), "text_only": config.text_only}


def build_identity(config, env):
    body = json_object(config.agent_extra_body_json) if config.agent_extra_body_json else {}
    steer = body.get("vllm_xargs", {}).get("steer")
    if steer is None:
        return None
    if config.system_prompt_file or env.get("HLE_SYSTEM_PROMPT_FILE"):
        raise ValueError("Outcome-CAA cannot be combined with Prompt-Reg/system prompt overrides")
    steer = json_object(steer)
    layer, alpha = steer.get("optimal_layer"), steer.get("coefficient")
    if type(layer) is not int or layer != steer.get("vector_layer") or layer < 0:
        raise ValueError("Equal zero-based optimal_layer/vector_layer required")
    if type(alpha) not in (float, int) or not math.isfinite(alpha):
        raise ValueError("Finite coefficient required")
    if steer.get("method") != "add_vector" or steer.get("position") != "last_prefix_token" or steer.get("apply_at_all_positions") is not False:
        raise ValueError("Require add_vector at last_prefix_token with apply_at_all_positions=false")
    metadata, vector_sha, metadata_sha = vector_metadata(steer["vector_path"])
    ours = metadata.get("method") == "trajre_mvp"
    if steer.get("tensor_key") != ("steering_vector" if ours else "mean_difference"):
        raise ValueError("tensor_key differs from vector format")
    if layer >= metadata["shape"][0]:
        raise ValueError("Steering layer outside target model")
    if steer.get("vector_sha256") != vector_sha:
        raise ValueError("steer.vector_sha256 required and must match vector bytes")
    service_path = env.get("OUTCOME_CAA_SERVICE_MANIFEST")
    if not service_path:
        raise ValueError("OUTCOME_CAA_SERVICE_MANIFEST required; use the serving script's JSON record")
    service = json.loads(Path(service_path).read_text(encoding="utf-8"))
    if service.get("served_model_name") != config.model or service.get("model_key") != metadata["model_key"]:
        raise ValueError("Serving model and vector checkpoint do not match evaluator")
    if service.get("vector_sha256") != vector_sha or service.get("prefix_caching") is not False:
        raise ValueError("Serving vector/cache settings differ from evaluator")
    if ours and (service.get("vector_metadata_sha256") != metadata_sha or
                 service.get("deployment_reference_sha256") != metadata["deployment_reference_sha256"] or
                 service.get("table_method") != "REVEAL (Ours)"):
        raise ValueError("Ours serving provenance differs from evaluator")
    if service.get("hook_commit") != HOOK_COMMIT or service.get("worker_version") != "hle-outcome-caa-v2":
        raise ValueError("Serving worker version does not match Outcome-CAA implementation")
    worker_path = Path(__file__).resolve().parents[1] / "patches/vllm_hook/steer_activation_worker.py"
    if service.get("worker_sha256") != sha256(worker_path):
        raise ValueError("Serving worker SHA256 differs from evaluator code; prepare a matching service")
    auxiliary = json_object(env.get("HLE_WITH_TOOLS_AUXILIARY_EXTRA_BODY_JSON", "{}"))
    if auxiliary != without_steering(auxiliary):
        raise ValueError("Auxiliary tool requests must not contain steering")
    identity = {"method": "TrajRE-MVP" if ours else "Outcome-CAA", "table_method": "REVEAL (Ours)" if ours else "REVEAL-Base", "model": config.model, "base_url": config.base_url,
            "model_key": metadata["model_key"], "vector_sha256": vector_sha,
            "vector_metadata_sha256": metadata_sha, "tensor_key": steer["tensor_key"],
            "layer": layer, "alpha": alpha, "collection_site": metadata["capture_site"],
            "intervention_site": "decoder block complete residual at last full chat-template prefix token",
            "model_fingerprint": metadata.get("model_fingerprint"), "agent_extra_body": body,
            "service": service, "dataset": selected_dataset(config),
            "budget": {k: getattr(config, k) for k in ("num_rollouts", "max_completion_tokens", "max_iterations", "max_workers", "max_retries", "process_retries", "question_timeout_seconds", "api_timeout_seconds", "temperature", "disable_scientific_search")}}
    if ours:
        identity.update(source_method="trajre_mvp", source_layers=metadata["source_layers"],
                        dataset_split=metadata["dataset_split"],
                        deployment_reference_sha256=metadata["deployment_reference_sha256"],
                        extraction_checkpoint_verified=False,
                        layer_semantics="source_layers used as declared model block indices; exporter code not supplied")
    return identity


def guard_resume(run_dir, identity):
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8")).get("outcome_caa")
        if old != identity:
            raise ValueError("Experiment identity differs; use a new run-id (existing manifest preserved)")
    elif identity is not None and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Nonempty run directory has no identity; use a new run-id")
