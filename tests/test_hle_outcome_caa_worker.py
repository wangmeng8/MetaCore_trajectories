"""CPU only: synthetic tensors, no model weights, no inference or API calls."""
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class DeferredDecoderFixture:
    def forward(self, hidden_states, residual):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return hidden_states, residual


class GptDecoderFixture:
    def forward(self, hidden_states, residual):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        output, residual = self.post_attention_layernorm(hidden_states, residual)
        return output, residual


def make_decoder(model, name, norm_name, forward, post_name=None, norm_module=None):
    norm_module = norm_module or "vllm.model_executor.layers.layernorm"
    norm = type(norm_name, (), {"__module__": norm_module})
    post = type(post_name or norm_name, (), {"__module__": norm_module})
    cls = type(name, (), {"__module__": "vllm.model_executor.models." + model,
                         "forward": forward})
    decoder = cls()
    decoder.input_layernorm = norm()
    decoder.post_attention_layernorm = post()
    return decoder


@pytest.mark.parametrize("model,name", [("qwen3_5", "Qwen3_5DecoderLayer"),
                                       ("qwen3_next", "Qwen3NextDecoderLayer")])
def test_qwen_gemma_rmsnorm_residual_contract(worker_module, model, name):
    decoder = make_decoder(model, name, "GemmaRMSNorm", DeferredDecoderFixture.forward)
    assert worker_module.decoder_semantics(decoder)[0] == "deferred_sum"


def test_gpt_rmsnorm_residual_contract(worker_module):
    decoder = make_decoder("gpt_oss", "TransformerBlock", "RMSNorm", GptDecoderFixture.forward)
    assert worker_module.decoder_semantics(decoder)[0] == "deferred_sum"


@pytest.mark.parametrize("norm,post,module", [
    ("RMSNorm", None, None),
    ("GemmaRMSNorm", "UnknownNorm", None),
    ("GemmaRMSNorm", None, "unreviewed.layernorm"),
])
def test_qwen_unknown_norms_still_rejected(worker_module, norm, post, module):
    decoder = make_decoder("qwen3_5", "Qwen3_5DecoderLayer", norm,
                           DeferredDecoderFixture.forward, post, module)
    with pytest.raises(RuntimeError, match="residual norm"):
        worker_module.decoder_semantics(decoder)


@pytest.fixture
def worker_module():
    context = types.ModuleType("vllm.forward_context")
    context.get_forward_context = lambda: types.SimpleNamespace(attn_metadata=None)
    common = types.ModuleType("vllm_hook_plugins.workers._common")
    common.get_query_metadata = lambda m: (m.query_start_loc, m.seq_lens)
    common.match_layer = lambda n: int(n.rsplit(".", 1)[1]) if n.startswith("model.layers.") else None
    common.iter_matched_modules = lambda m, f: ((n, v, f(n)) for n, v in m.named_modules() if f(n) is not None)
    with patch.dict(sys.modules, {"vllm.forward_context": context,
                                 "vllm_hook_plugins.workers._common": common}):
        spec = importlib.util.spec_from_file_location("caa_worker_test", ROOT / "patches/vllm_hook/steer_activation_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


def config(path, **updates):
    return dict(method="add_vector", position="last_prefix_token", apply_at_all_positions=False,
                vector_path=str(path), tensor_key="mean_difference", vector_layer=0,
                optimal_layer=0, coefficient=1.0, **updates)


def test_last_chunk_and_mixed_batch_positions(worker_module):
    # Non-final chunk, final chunk, decode, single-token prompt.
    assert worker_module.last_prefix_rows([0, 3, 5, 6, 7], [0, 6, 8, 0], [8, 8, 8, 1]) == [None, 4, None, 6]


def test_prefill_crossing_into_decode_fails(worker_module):
    with pytest.raises(ValueError):
        worker_module.last_prefix_rows([0, 4], [6], [8])


def test_split_output_adds_to_complete_residual(worker_module):
    h = torch.tensor([[1., 2.], [3., 4.]])
    r = torch.tensor([[5., 6.], [7., 8.]])
    out = worker_module.add_at_row((h, r), "deferred_sum", 1, torch.tensor([2., -1.]), 1.)
    assert torch.equal(out[0][0], h[0]) and torch.equal(out[1][0], r[0])
    assert torch.equal(out[0][1] + out[1][1], h[1] + r[1] + torch.tensor([2., -1.]))
    assert torch.equal(h, torch.tensor([[1., 2.], [3., 4.]]))


def test_gemma_complete_output_and_zero_identity(worker_module):
    original = (torch.ones(3, 2), None)
    assert worker_module.add_at_row(original, "complete_tuple", 1, torch.ones(2), 0.) is original
    out = worker_module.add_at_row(original, "complete_tuple", 1, torch.ones(2), 2.)
    assert out[1] is None
    assert torch.equal(out[0], torch.tensor([[1., 1.], [3., 3.], [1., 1.]]))


def test_tensor_only_load_never_falls_back(worker_module, tmp_path):
    path = tmp_path / "vector.pt"
    path.touch()
    with patch.object(torch, "load", side_effect=TypeError("weights_only unsupported")) as mocked:
        with pytest.raises(TypeError):
            worker_module._load_tensor_file(str(path), None)
    assert mocked.call_count == 1
    assert mocked.call_args.kwargs["weights_only"] is True


def test_vector_shapes_and_keys(worker_module, tmp_path):
    path = tmp_path / "vector.pt"
    torch.save({"mean_difference": torch.arange(6.).reshape(3, 2), "mean_positive": torch.ones(3, 2)}, path)
    assert torch.equal(worker_module._prepare_vector(str(path), "mean_difference", 2, 2), torch.tensor([4., 5.]))
    with pytest.raises(ValueError):
        worker_module._prepare_vector(str(path), None, 0, 2)
    with pytest.raises(ValueError):
        worker_module._prepare_vector(str(path), "mean_difference", 3, 2)
    with pytest.raises(ValueError):
        worker_module._prepare_vector(str(path), "mean_difference", 0, 3)
    torch.save(torch.tensor([float("nan"), 1.]), path)
    with pytest.raises(ValueError):
        worker_module._prepare_vector(str(path), None, 0, 2)


@pytest.mark.parametrize("update", [{"optimal_layer": 99}, {"vector_layer": 1},
                                  {"coefficient": float("nan")}, {"position": "decode"},
                                  {"apply_at_all_positions": "false"}, {"optimal_layer": 0.5}])
def test_invalid_request_config_fails_even_for_zero(worker_module, tmp_path, update):
    cfg = config(tmp_path / "x.pt")
    cfg.update(coefficient=0.)
    cfg.update(update)
    with pytest.raises((ValueError, TypeError)):
        worker_module.validate_config(cfg, {0})


def make_worker(mod, tmp_path, configs, starts, computed, prefixes):
    worker = mod.SteerHookActWorker()
    ids = [f"request-{i}" for i in range(len(configs))]
    states = {rid: types.SimpleNamespace(num_prompt_tokens=p, sampling_params=types.SimpleNamespace(
        extra_args={} if c is None else {"steer": json.dumps(c)})) for rid, c, p in zip(ids, configs, prefixes)}
    worker.model_runner = types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=ids,
        num_computed_tokens_cpu=computed), requests=states,
        query_start_loc=types.SimpleNamespace(cpu=torch.tensor(starts)))
    worker._layers = {0}
    worker._vector_root = tmp_path.resolve()
    worker._vector_cache, worker._device_cache = {}, {}
    mod.get_forward_context = lambda: types.SimpleNamespace(attn_metadata={"present": True})
    return worker


def test_real_hook_mixed_final_chunk_zero_and_decode(worker_module, tmp_path):
    path = tmp_path / "v.pt"
    torch.save({"mean_difference": torch.tensor([[2., -1.]])}, path)
    cfg = config(path)
    zero = dict(cfg, coefficient=0.)
    worker = make_worker(worker_module, tmp_path, [None, zero, cfg, cfg], [0, 2, 4, 6, 7], [0, 0, 6, 8], [2, 2, 8, 8])
    original = (torch.ones(7, 2), None)
    out = worker._steer(original, 0, "complete_tuple")
    expected = torch.ones(7, 2)
    expected[5] = torch.tensor([3., 0.])
    assert torch.equal(out[0], expected)
    # Exactly one disk load across zero and nonzero requests using the same file.
    assert len(worker._vector_cache) == 1 and len(worker._device_cache) == 1
    # Next decode iteration is an actual identity operation.
    worker.model_runner.input_batch.num_computed_tokens_cpu = [2, 2, 8, 8]
    assert worker._steer(out, 0, "complete_tuple") is out


def test_hook_fails_on_missing_metadata_but_plain_passes(worker_module, tmp_path):
    cfg = config(tmp_path / "x.pt")
    worker = make_worker(worker_module, tmp_path, [cfg], [0, 1], [0], [1])
    worker_module.get_forward_context = lambda: types.SimpleNamespace(attn_metadata=None)
    with pytest.raises(RuntimeError, match="metadata"):
        worker._steer((torch.ones(1, 2), None), 0, "complete_tuple")
    worker.model_runner.requests["request-0"].sampling_params.extra_args = {}
    original = object()
    assert worker._steer(original, 0, "complete_tuple") is original


def test_vector_path_escape_and_wrong_digest_fail(worker_module, tmp_path):
    path = tmp_path / "x.pt"
    torch.save(torch.ones(2), path)
    worker = make_worker(worker_module, tmp_path, [], [0], [], [])
    cfg = dict(config(path), tensor_key=None, vector_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        worker._vector(cfg, torch.ones(1, 2))
    worker._vector_root = tmp_path / "allowed"
    with pytest.raises(ValueError, match="escapes"):
        worker._vector(cfg, torch.ones(1, 2))


def test_prefix_cache_refused_at_install(worker_module):
    worker = worker_module.SteerHookActWorker()
    worker.model_runner = types.SimpleNamespace(model=object(), vllm_config=types.SimpleNamespace(
        cache_config=types.SimpleNamespace(enable_prefix_caching=True)))
    with pytest.raises(RuntimeError, match="prefix-caching"):
        worker.install_hooks()
    assert worker._hooks_installed is False


def test_safetensors_key_and_direct_pt(worker_module, tmp_path):
    from safetensors.torch import save_file
    path = tmp_path / "v.safetensors"
    save_file({"a": torch.ones(2), "b": torch.zeros(2)}, str(path))
    with pytest.raises(ValueError):
        worker_module._load_tensor_file(path, None)
    assert torch.equal(worker_module._load_tensor_file(path, "b"), torch.zeros(2))
    path = tmp_path / "direct.pt"
    torch.save(torch.ones(2), path)
    assert torch.equal(worker_module._load_tensor_file(path, None), torch.ones(2))
    with pytest.raises(ValueError):
        worker_module._load_tensor_file(path, "mean_difference")
