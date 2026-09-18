"""HLE Outcome-CAA for IBM/vLLM-Hook e3d6885: V1 eager, synchronous scheduling.

Only complete decoder residuals at the final prefix token are modified.
TP supports replicated block outputs. Unsupported execution paths fail closed.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import textwrap

import torch
from vllm.forward_context import get_forward_context
from vllm_hook_plugins.workers._common import iter_matched_modules, match_layer

OUTCOME_CAA_VERSION = "hle-outcome-caa-v2"


def _load_tensor_file(vector_path, tensor_key):
    path = Path(vector_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if tensor_key is None:
                if len(keys) != 1:
                    raise ValueError("tensor_key required for multiple tensors")
                tensor_key = keys[0]
            return f.get_tensor(tensor_key)
    if path.suffix.lower() != ".pt":
        raise ValueError("Only .pt and .safetensors vectors are supported")
    # Never fall back to pickle loading, including on older torch versions.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, torch.Tensor):
        if tensor_key is not None:
            raise ValueError("Direct tensor files do not have a tensor_key")
        return raw
    if not isinstance(raw, dict):
        raise TypeError("Expected a tensor or tensor dictionary")
    tensors = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
    if tensor_key is None:
        if len(tensors) != 1:
            raise ValueError("tensor_key required for multiple tensors")
        tensor_key = next(iter(tensors))
    return tensors[tensor_key]


def _prepare_vector(vector_path, tensor_key, vector_layer, hidden_size):
    vector = _load_tensor_file(vector_path, tensor_key)
    if not vector.is_floating_point():
        raise ValueError("Vector must have floating point dtype")
    if vector.ndim == 2:
        if not 0 <= vector_layer < vector.shape[0]:
            raise ValueError("vector_layer outside vector shape")
        vector = vector[vector_layer]
    if vector.ndim != 1 or vector.shape[0] != hidden_size:
        raise ValueError(f"Expected vector [{hidden_size}], got {tuple(vector.shape)}")
    if not torch.isfinite(vector).all().item():
        raise ValueError("Vector contains non-finite values")
    return vector.detach().contiguous()


def _decode_config(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("steer must be a JSON object")
    return value


def validate_config(cfg, layers):
    if cfg.get("method") != "add_vector" or cfg.get("position") != "last_prefix_token":
        raise ValueError("Require add_vector at last_prefix_token")
    if cfg.get("apply_at_all_positions") is not False:
        raise ValueError("apply_at_all_positions must be JSON false")
    layer, row = cfg.get("optimal_layer"), cfg.get("vector_layer")
    if type(layer) is not int or type(row) is not int or layer != row or layer not in layers:
        raise ValueError("optimal_layer/vector_layer must be equal valid zero-based layer indices")
    alpha = cfg.get("coefficient")
    if type(alpha) not in (float, int) or not math.isfinite(alpha):
        raise ValueError("coefficient must be finite")
    if not isinstance(cfg.get("vector_path"), str) or not cfg["vector_path"]:
        raise ValueError("vector_path required")
    if cfg.get("tensor_key") is not None and not isinstance(cfg["tensor_key"], str):
        raise ValueError("tensor_key must be a string or null")
    return cfg


def last_prefix_rows(starts, computed, prompt_lengths):
    """Use V1 scheduled offsets and pre-forward CPU counts, including final chunks.

    Preemption discards KV: the final prefix must be steered on recomputation too.
    """
    if len(starts) != len(computed) + 1 or len(computed) != len(prompt_lengths):
        raise ValueError("Inconsistent batch metadata lengths")
    starts = [int(x) for x in starts]
    if starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError("Invalid scheduled token offsets")
    rows = []
    for i, (done, prefix) in enumerate(zip(computed, prompt_lengths)):
        done, prefix = int(done), int(prefix)
        if done < 0 or prefix <= 0:
            raise ValueError("Invalid computed/prompt length")
        end = done + starts[i + 1] - starts[i]
        if done < prefix < end:
            raise ValueError("Chunk crosses prefill/decode boundary; speculative path unsupported")
        rows.append(starts[i + 1] - 1 if done < prefix and end == prefix else None)
    return rows


def add_at_row(output, semantics, row, vector, alpha):
    if alpha == 0:
        return output  # Preserve objects and bits, including deferred residuals.
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError("Reviewed decoder contract requires a two-element tuple")
    hidden, residual = output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or not 0 <= row < hidden.shape[0]:
        raise RuntimeError("Invalid block output shape/token offset")
    delta = vector.to(device=hidden.device, dtype=hidden.dtype) * alpha
    if not torch.isfinite(delta).all().item():
        raise ValueError("Vector/coefficient overflows activation dtype")
    hidden_out = hidden.clone()
    if semantics == "complete_tuple":
        if residual is not None:
            raise RuntimeError("Gemma4 contract changed: expected (complete_hidden_states, None)")
        hidden_out[row] = hidden[row] + delta
        return hidden_out, None
    if semantics != "deferred_sum" or not isinstance(residual, torch.Tensor) or residual.shape != hidden.shape:
        raise RuntimeError("Unsupported residual return semantics")
    # Qwen/GPT-OSS defer the final addition to the next fused RMSNorm.
    # Materialize the complete block output ONLY at this row, then add CAA.
    # Returning (0, complete+delta) preserves that result in the next fused add.
    residual_out = residual.clone()
    residual_out[row] = (hidden[row].float() + residual[row].float()).to(hidden.dtype) + delta
    hidden_out[row] = 0
    return hidden_out, residual_out


def decoder_semantics(module):
    """Explicit source-checked adapters; never infer semantics from tuple indices."""
    cls = type(module)
    qualified = cls.__module__ + "." + cls.__name__
    source = textwrap.dedent(inspect.getsource(cls.forward))
    tree = ast.parse(source)
    returns = [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Return)]
    allowed = {
        "vllm.model_executor.models.qwen3_5.Qwen3_5DecoderLayer": "hidden_states",
        "vllm.model_executor.models.qwen3_next.Qwen3NextDecoderLayer": "hidden_states",
        "vllm.model_executor.models.gpt_oss.TransformerBlock": "output",
    }
    if qualified == "vllm.model_executor.models.gemma4.Gemma4DecoderLayer":
        if returns != ["(hidden_states, None)"]:
            raise RuntimeError("Unrecognized Gemma4 residual contract")
        semantics = "complete_tuple"
    elif qualified in allowed:
        if returns != [f"({allowed[qualified]}, residual)"]:
            raise RuntimeError("Unrecognized deferred residual contract")
        calls = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if "self.input_layernorm(hidden_states, residual)" not in calls or "self.post_attention_layernorm(hidden_states, residual)" not in calls:
            raise RuntimeError("Expected fused residual RMSNorm calls missing")
        # Qwen3_5/Next aliases GemmaRMSNorm; GPT-OSS uses RMSNorm. Both add
        # the residual before normalization, but their affine weights differ.
        # Preserve the actual norm implementation and check both fused add sites.
        norm_name = "RMSNorm" if qualified == "vllm.model_executor.models.gpt_oss.TransformerBlock" else "GemmaRMSNorm"
        expected = "vllm.model_executor.layers.layernorm." + norm_name
        for attr in ("input_layernorm", "post_attention_layernorm"):
            norm_cls = type(getattr(module, attr))
            actual = norm_cls.__module__ + "." + norm_cls.__name__
            if actual != expected:
                raise RuntimeError(f"Unrecognized residual norm implementation: {qualified}.{attr} is {actual}; expected {expected}")
        semantics = "deferred_sum"
    else:
        raise RuntimeError(f"No reviewed Outcome-CAA residual adapter for {qualified}")
    return semantics, {"class": qualified, "semantics": semantics,
                       "forward_sha256": hashlib.sha256(source.encode()).hexdigest()}


class SteerHookActWorker:
    _hooks_installed = False

    def install_hooks(self):
        if self._hooks_installed:
            return
        self._install_hooks()
        self._hooks_installed = True

    def _install_hooks(self):
        runner = self.model_runner
        model = runner.model
        config = runner.vllm_config
        if config.cache_config.enable_prefix_caching is not False:
            raise RuntimeError("Outcome-CAA requires --no-enable-prefix-caching")
        if not config.model_config.enforce_eager:
            raise RuntimeError("Outcome-CAA requires --enforce-eager")
        if getattr(config.scheduler_config, "async_scheduling", False):
            raise RuntimeError("Outcome-CAA requires --no-async-scheduling")
        if config.parallel_config.pipeline_parallel_size != 1 or getattr(config, "speculative_config", None):
            raise RuntimeError("PP/speculative decoding unsupported for Outcome-CAA")
        for _, module in model.named_modules():
            if getattr(module, "use_sequence_parallel", False) or getattr(module, "use_attn_reduce_scatter_for_moe", False):
                raise RuntimeError("Sequence-sharded block residuals unsupported; disable sequence parallel")
        self._vector_root = Path(os.environ["OUTCOME_CAA_VECTOR_ROOT"]).resolve(strict=True)
        if not self._vector_root.is_dir():
            raise ValueError("OUTCOME_CAA_VECTOR_ROOT must be a directory")
        self._vector_cache, self._device_cache, self._hooks = {}, {}, []
        matches = list(iter_matched_modules(model, match_layer))
        self._layers = {layer for _, _, layer in matches}
        if not matches or len(self._layers) != len(matches):
            raise RuntimeError("No unique decoder layers found")
        contracts = [(n, m, l, *decoder_semantics(m)) for n, m, l in matches]
        for name, module, layer, semantics, record in contracts:
            self._hooks.append(module.register_forward_hook(
                lambda mod, inp, out, ln=layer, sm=semantics: self._steer(out, ln, sm)))
            self._audit(dict(event="install", layer=layer, module=name, **record))

    def _audit(self, record):
        record = dict(version=OUTCOME_CAA_VERSION, pid=os.getpid(), rank=getattr(self, "rank", 0), **record)
        directory = os.environ.get("OUTCOME_CAA_AUDIT_DIR")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"worker-{os.getpid()}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def _vector(self, cfg, hidden):
        path = Path(cfg["vector_path"]).expanduser().resolve(strict=True)
        if not path.is_relative_to(self._vector_root):
            raise ValueError("vector_path escapes OUTCOME_CAA_VECTOR_ROOT")
        key = (str(path), cfg.get("tensor_key"), cfg["vector_layer"], hidden.shape[-1])
        if key not in self._vector_cache:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if cfg.get("vector_sha256") and digest != cfg["vector_sha256"]:
                raise ValueError("Vector SHA256 differs from evaluator identity")
            vector = _prepare_vector(str(path), cfg.get("tensor_key"), cfg["vector_layer"], hidden.shape[-1])
            self._vector_cache[key] = (vector, digest)
            self._audit(dict(event="vector_loaded", path=str(path), sha256=digest, layer=cfg["vector_layer"]))
        vector, digest = self._vector_cache[key]
        if cfg.get("vector_sha256") and digest != cfg["vector_sha256"]:
            raise ValueError("Cached vector SHA256 differs from request")
        device_key = (key, hidden.device, hidden.dtype)
        if device_key not in self._device_cache:
            converted = vector.to(device=hidden.device, dtype=hidden.dtype)
            if not torch.isfinite(converted).all().item():
                raise ValueError("Vector overflows activation dtype")
            self._device_cache[device_key] = converted
        return self._device_cache[device_key], digest

    def _steer(self, output, layer, semantics):
        runner = self.model_runner
        batch = runner.input_batch
        ids = list(batch.req_ids)
        configs = []
        for req_id in ids:
            if req_id not in runner.requests:
                raise RuntimeError("Active request missing from V1 request map")
            params = runner.requests[req_id].sampling_params
            cfg = _decode_config((getattr(params, "extra_args", None) or {}).get("steer"))
            configs.append(validate_config(cfg, self._layers) if cfg is not None else None)
        if not any(cfg is not None for cfg in configs):
            return output
        if getattr(get_forward_context(), "attn_metadata", None) is None:
            raise RuntimeError("Missing attention metadata on a configured request")
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CUDA graph capture cannot execute Outcome-CAA")
        # Authoritative CPU buffers avoid sliding-window seq_lens and async output counts.
        try:
            starts = runner.query_start_loc.cpu[:len(ids) + 1].tolist()
            computed = batch.num_computed_tokens_cpu[:len(ids)]
            lengths = [runner.requests[r].num_prompt_tokens for r in ids]
        except AttributeError as exc:
            raise RuntimeError("Unsupported V1 batch metadata layout") from exc
        rows = last_prefix_rows(starts, computed, lengths)
        for req_id, cfg, row, prefix in zip(ids, configs, rows, lengths):
            if cfg is None or cfg["optimal_layer"] != layer or row is None:
                continue
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[0] < starts[-1]:
                raise RuntimeError("Output is sharded or token offsets exceed block shape")
            vector, digest = self._vector(cfg, hidden)
            output = add_at_row(output, semantics, row, vector, cfg["coefficient"])
            self._audit(dict(event="last_prefix", request_id=req_id, layer=layer,
                             alpha=cfg["coefficient"], prefix_length=int(prefix),
                             token_position=int(prefix)-1, batch_row=row,
                             vector_sha256=digest, applied=cfg["coefficient"] != 0))
        return output
