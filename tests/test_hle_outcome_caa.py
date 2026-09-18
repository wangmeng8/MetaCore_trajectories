import importlib.util
import json
from pathlib import Path

import pytest

from scripts import run_hle_with_tools as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".parquet", ".PARQUET"])
@pytest.mark.parametrize("text_only,limit,expected", [
    (True, 2, ["9", "2"]),
    (True, None, ["9", "2", "7"]),
    (False, 2, ["9", "4"]),
])
def test_dataset_formats_preserve_selection(tmp_path, monkeypatch, suffix, text_only, limit, expected):
    from types import SimpleNamespace
    from scripts.hle_outcome_caa import selected_dataset, sha256

    records = [
        {"id": 9, "image": None, "image_preview": "", "rationale_image": None},
        {"id": 4, "image": "image.png", "image_preview": "", "rationale_image": None},
        {"id": 5, "image": None, "image_preview": "preview.png", "rationale_image": None},
        {"id": 6, "image": None, "image_preview": "", "rationale_image": "rationale.png"},
        {"id": 2, "image": "", "image_preview": "", "rationale_image": None},
        {"id": 7, "image": None, "image_preview": "", "rationale_image": None},
    ]
    path = tmp_path / ("data" + suffix)
    if suffix.lower() == ".parquet":
        import datasets
        import pyarrow as pa
        import pyarrow.parquet as pq

        monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path.parent / "hf")
        pq.write_table(pa.Table.from_pylist(records), path)
    else:
        path.write_text(json.dumps(records) if suffix == ".json" else "\n".join(map(json.dumps, records)), encoding="utf-8")
    actual = selected_dataset(SimpleNamespace(data_path=path, text_only=text_only, num_tasks=limit))
    assert actual == {"sha256": sha256(path), "task_ids": expected,
                      "task_count": len(expected), "text_only": text_only}


@pytest.mark.parametrize("records", [[{"id": "same"}, {"id": "same"}], [{"id": "image-only", "image": "image.png"}]])
def test_parquet_rejects_duplicate_or_empty_selection(tmp_path, monkeypatch, records):
    from types import SimpleNamespace
    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.hle_outcome_caa import selected_dataset

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path.parent / "hf")
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    with pytest.raises(ValueError, match="empty or contains duplicate IDs"):
        selected_dataset(SimpleNamespace(data_path=path, text_only=True, num_tasks=None))


def test_main_body_preserves_nested_vanilla_fields():
    cfg = runner.build_hle_with_tools_config({
        "HLE_WITH_TOOLS_EXTRA_BODY_JSON": json.dumps({"chat_template_kwargs": {"enable_thinking": True}, "vllm_xargs": {"other": 3}}),
        "HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON": json.dumps({"vllm_xargs": {"steer": "{}"}}),
    })
    assert json.loads(cfg.agent_extra_body_json) == {
        "chat_template_kwargs": {"enable_thinking": True}, "vllm_xargs": {"other": 3, "steer": "{}"}}


def test_cli_steering_preserves_agent_defaults_loaded_from_dotenv():
    env = {"HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON": '{"chat_template_kwargs":{"enable_thinking":true}}'}
    args = runner.parse_args(["--agent-extra-body-json", '{"vllm_xargs":{"steer":"{}"}}'])
    actual = runner.build_hle_with_tools_config(runner.apply_cli_overrides(env, args))
    assert json.loads(actual.agent_extra_body_json)["chat_template_kwargs"]["enable_thinking"] is True


def test_auxiliary_body_retains_vanilla_without_steering():
    path = ROOT / ".external/hle_with_tools/hle_eval/agent/extra_body.py"
    assert path.exists(), "Missing main/auxiliary request isolation helper"
    spec = importlib.util.spec_from_file_location("isolated_bodies", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    main, aux = mod.request_bodies({
        "HLE_WITH_TOOLS_EXTRA_BODY_JSON": '{"chat_template_kwargs":{"enable_thinking":true},"vllm_xargs":{"steer":"{}","other":1}}',
        "HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON": json.dumps({"vllm_xargs": {"steer": json.dumps({"a": 1})}}),
    }, {})
    assert "steer" in main["vllm_xargs"]
    assert aux == {"chat_template_kwargs": {"enable_thinking": True}, "vllm_xargs": {"other": 1}}
    main["chat_template_kwargs"]["enable_thinking"] = False
    assert aux["chat_template_kwargs"]["enable_thinking"] is True


def test_identity_refuses_changed_or_unidentified_resume(tmp_path):
    from scripts.hle_outcome_caa import guard_resume
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"outcome_caa": {"alpha": 1}}))
    guard_resume(tmp_path, {"alpha": 1})
    with pytest.raises(ValueError, match="run"):
        guard_resume(tmp_path, {"alpha": 2})
    with pytest.raises(ValueError):
        guard_resume(tmp_path, None)
    manifest.write_text('{}')
    with pytest.raises(ValueError):
        guard_resume(tmp_path, {"alpha": 1})


def test_outcome_rejects_prompt_reg_before_launch(tmp_path):
    from scripts.hle_outcome_caa import build_identity
    cfg = runner.build_hle_with_tools_config({"HLE_SYSTEM_PROMPT_FILE": "prompt.txt",
        "HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON": '{"vllm_xargs":{"steer":"{}"}}'})
    with pytest.raises(ValueError, match="Prompt-Reg"):
        build_identity(cfg, {})


def test_migration_contains_patches_and_docs():
    from scripts.create_migration_package import DEFAULT_ITEMS
    assert "patches" in DEFAULT_ITEMS
    assert "docs" in DEFAULT_ITEMS


@pytest.mark.parametrize("flag", ["--enable-prefix-caching", "--enable-prefix-caching=true", "--async-scheduling", "--worker-extension-cls", "--speculative-config", "-pp"])
def test_unsafe_serving_overrides_rejected(flag):
    from scripts.serve_hle_outcome_caa import serving_args
    with pytest.raises(ValueError):
        serving_args(["--max-model-len", "4096", flag])


def test_serving_args_preserve_json_quotes_and_require_context():
    from scripts.serve_hle_outcome_caa import serving_args
    args = ["--max-model-len", "4096", "--chat-template-kwargs", '{"enable_thinking": true}']
    assert serving_args(args) == args
    with pytest.raises(ValueError):
        serving_args(["--tool-call-parser", "some-parser"])


@pytest.mark.parametrize("inherited", [None, "1"])
def test_launch_pins_compatible_runner_and_records_it(tmp_path, monkeypatch, inherited):
    import os
    import sys
    import types
    from scripts import serve_hle_outcome_caa as serving

    if inherited is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", inherited)
    args_file = tmp_path / "args.json"
    args_file.write_text('["--max-model-len", "131072"]')
    vector = tmp_path / "vector.pt"
    vector.touch()
    manifest = tmp_path / "service.json"
    for key, value in {
        "VLLM_VANILLA_ARGS_FILE": args_file, "OUTCOME_CAA_VECTOR_PATH": vector,
        "OUTCOME_CAA_VECTOR_ROOT": tmp_path, "MODEL_PATH": tmp_path,
        "SERVED_MODEL_NAME": "gemma-4-31b", "TENSOR_PARALLEL_SIZE": "4",
        "OUTCOME_CAA_SERVICE_MANIFEST": manifest, "OUTCOME_CAA_AUDIT_DIR": tmp_path / "audit",
    }.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(serving, "vector_metadata", lambda *a: ({"model_key": "gemma4_31b_it"}, "digest", "metadata"))
    monkeypatch.setattr(serving, "verify_checkpoint", lambda *a: {})
    monkeypatch.setattr(serving.importlib.metadata, "version", lambda name: "0.23.0" if name == "zstandard" else "test")
    workers = types.ModuleType("vllm_hook_plugins.workers")
    workers.steer_activation_worker = types.SimpleNamespace(
        OUTCOME_CAA_VERSION="hle-outcome-caa-v2",
        __file__=str(ROOT / "patches/vllm_hook/steer_activation_worker.py"))
    monkeypatch.setitem(sys.modules, "vllm_hook_plugins.workers", workers)
    vllm = types.ModuleType("vllm")
    vllm.__file__ = str(tmp_path / "__init__.py")
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    class LaunchCaptured(Exception):
        pass

    def capture_launch(binary, argv):
        assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"
        assert "--enforce-eager" in argv
        assert json.loads(manifest.read_text())["model_runner"] == {
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "class": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        }
        raise LaunchCaptured

    monkeypatch.setattr(serving.os, "execvp", capture_launch)
    with pytest.raises(LaunchCaptured):
        serving.main([])


def test_supplied_vectors_match_metadata_and_request_hashes():
    import torch
    from scripts.prepare_hle_outcome_caa import request_body
    from scripts.hle_outcome_caa import MODELS, vector_metadata
    for key, (layer, shape) in MODELS.items():
        path = ROOT / f"data/HLE/outcome_caa_vectors/{key}/benchmark_global/steering_vector.pt"
        metadata, digest, _ = vector_metadata(path, key)
        body = request_body(path, None, 1., {"OUTCOME_CAA_MODEL_KEY": key})
        cfg = json.loads(body["vllm_xargs"]["steer"])
        assert cfg["optimal_layer"] == layer and cfg["vector_sha256"] == digest
        assert metadata["shape"] == shape
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        direction = tensors["mean_difference"]
        assert list(direction.shape) == shape and direction.dtype == torch.float32
        assert torch.isfinite(direction).all()
        assert torch.equal(direction, tensors["mean_positive"] - tensors["mean_negative"])


def test_full_identity_and_changed_budget_guard(tmp_path):
    from scripts.hle_outcome_caa import build_identity, guard_resume, vector_metadata, HOOK_COMMIT, sha256
    from scripts.prepare_hle_outcome_caa import request_body
    vector = ROOT / "data/HLE/outcome_caa_vectors/qwen36_27b/benchmark_global/steering_vector.pt"
    _, digest, _ = vector_metadata(vector)
    service = tmp_path / "service.json"
    service.write_text(json.dumps({"served_model_name": "qwen", "model_key": "qwen36_27b", "vector_sha256": digest,
        "prefix_caching": False, "hook_commit": HOOK_COMMIT, "worker_version": "hle-outcome-caa-v2",
        "worker_sha256": sha256(ROOT / "patches/vllm_hook/steer_activation_worker.py")}))
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"id": "a", "image": ""}, {"id": "b", "image": "img"}, {"id": "c"}]))
    env = {"HLE_MODEL": "qwen", "HLE_DATA_PATH": str(data), "HLE_ALL_TASKS": "1",
           "HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON": json.dumps(request_body(vector, 23, 1., {})),
           "OUTCOME_CAA_SERVICE_MANIFEST": str(service)}
    config = runner.build_hle_with_tools_config(env)
    identity = build_identity(config, env)
    assert identity["dataset"]["task_ids"] == ["a", "c"]
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"outcome_caa": identity}))
    guard_resume(run, build_identity(config, env))
    config.max_completion_tokens += 1
    with pytest.raises(ValueError):
        guard_resume(run, build_identity(config, env))


def test_sanity_audit_requires_all_tp_ranks_and_correct_token():
    from scripts.sanity_hle_outcome_caa import check_audit
    events = [dict(event="last_prefix", request_id="chatcmpl-x-0", rank=r, alpha=1., token_position=7, prefix_length=8, applied=True) for r in range(2)]
    check_audit(events, "chatcmpl-x", 1., 2)
    with pytest.raises(ValueError):
        check_audit(events[:1], "chatcmpl-x", 1., 2)
    with pytest.raises(ValueError):
        check_audit(events, "chatcmpl-x", None, 2)


def test_hle_delta_patch_roundtrip_without_git_directory(tmp_path):
    import os
    import shutil
    import subprocess
    target = tmp_path / "restored"
    folder = target / "hle_eval/agent"
    folder.mkdir(parents=True)
    for name in ("config.py", "naive.py", "tools.py", "extra_body.py"):
        shutil.copy2(ROOT / ".external/hle_with_tools/hle_eval/agent" / name, folder / name)
    before = {p.name: p.read_text(encoding="utf-8") for p in folder.iterdir()}
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tmp_path))
    patch_file = str(ROOT / "patches/hle_with_tools/outcome_caa_agent_extra_body.patch")
    for extra in (["--reverse", "--check"], ["--reverse"], ["--check"], []):
        subprocess.run(["git", "-C", str(target), "apply", *extra, patch_file], env=env, check=True, capture_output=True)
    # Git for Windows may honor core.autocrlf; compare normalized source text.
    assert {p.name: p.read_text(encoding="utf-8") for p in folder.iterdir()} == before


def test_all_existing_agent_api_sites_use_isolated_body():
    import ast
    for name in ("naive.py", "tools.py"):
        text = (ROOT / ".external/hle_with_tools/hle_eval/agent" / name).read_text(encoding="utf-8")
        tree = ast.parse(text)
        assert "extra_body=EXTRA_BODY" not in text and '"extra_body": EXTRA_BODY' not in text
        assert "AGENT_EXTRA_BODY" in text
        # Inspect actual named keyword and dictionary call construction sites.
        bodies = [kw.value for n in ast.walk(tree) if isinstance(n, ast.Call) for kw in n.keywords if kw.arg == "extra_body"]
        bodies += [v for n in ast.walk(tree) if isinstance(n, ast.Dict) for k, v in zip(n.keys, n.values)
                   if isinstance(k, ast.Constant) and k.value == "extra_body"]
        assert len(bodies) == (1 if name == "naive.py" else 2)
        assert all(isinstance(n, ast.Name) and n.id == "AGENT_EXTRA_BODY" for n in bodies)
