"""Validate and import original TrajRE artifacts, without rewriting tensors."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

from scripts.hle_outcome_caa import MODELS, sha256

ARCHIVES = {"Qwen36_27B_HLE_TrajRE_MVP_benchmark_global": "qwen36_27b",
            "Gemma4_31B_IT_HLE_TrajRE_MVP_benchmark_global": "gemma4_31b_it",
            "GPTOSS_20B_HLE_TrajRE_MVP_benchmark_global": "gptoss_20b"}
FILES = {"steering_vector.pt", "steering_vector.npy", "metadata.json", "validation.json",
         "selected_pairs.jsonl.gz", "SHA256SUMS"}


def validate_ours_vector(path, metadata, model_key=None, *, validate_pt=False):
    import numpy as np

    key = metadata.get("model_key")
    if key not in MODELS or (model_key and key != model_key):
        raise ValueError("Vector model_key mismatch")
    if metadata.get("benchmark") != "hle_with_tools" or metadata.get("method") != "trajre_mvp":
        raise ValueError("Unexpected Ours method/benchmark")
    n, width = MODELS[key][1]
    layers = metadata.get("source_layers")
    # Current worker requires vector row == model block index. Reject sparse or shifted maps.
    if layers != list(range(n - 1)):
        raise ValueError("Unsupported source_layers mapping; never pad or shift rows")
    checked = set()
    for line in path.with_name("SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, filename = line.split(maxsplit=1)
        filename = filename.lstrip("*")
        if filename not in FILES - {"SHA256SUMS"} or filename in checked:
            raise ValueError("Unexpected or duplicate checksum member")
        target = path.with_name(filename)
        if sha256(target) != expected:
            raise ValueError(f"Vector bundle checksum mismatch: {filename}")
        checked.add(filename)
    if checked != FILES - {"SHA256SUMS"}:
        raise ValueError("Incomplete vector checksums")
    array = np.load(path.with_suffix(".npy"), allow_pickle=False)
    if list(array.shape) != [n-1, width] or array.dtype != np.float32:
        raise ValueError("Unexpected Ours tensor shape/dtype")
    if not np.isfinite(array).all() or not (np.linalg.norm(array, axis=1) > 0).all():
        raise ValueError("Non-finite or zero vector row")
    # Import verifies PT on CPU; evaluator uses checked sidecars without needing torch.
    # The serving worker also independently checks the actual PT row before use.
    if validate_pt:
        import torch
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict) or raw.get("layers") != layers or raw.get("model_key") != key:
            raise ValueError("PT layer/model identity differs from metadata")
        if raw.get("method") != "trajre_mvp" or raw.get("benchmark") != "hle_with_tools":
            raise ValueError("PT method/benchmark mismatch")
        vector = raw.get("steering_vector")
        if not isinstance(vector, torch.Tensor) or vector.dtype != torch.float32 or not np.array_equal(vector.numpy(), array):
            raise ValueError("PT/NPY differ")
    if metadata.get("hidden_size") != width:
        raise ValueError("Metadata hidden_size mismatch")
    reference_path = path.with_name("deployment_reference.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("model_key") != key or not reference.get("model_fingerprint", {}).get("file_hashes"):
        raise ValueError("Missing previous deployment checkpoint reference")
    normalized = dict(metadata)
    normalized.update(shape=[n-1, width], layer_index_base=0,
                      capture_site=metadata["local_response"], model_fingerprint=None,
                      deployment_reference=reference, deployment_reference_sha256=sha256(reference_path))
    return normalized, sha256(path), sha256(path.with_name("metadata.json"))


def import_vectors(archive, destination, reference_root):
    destination, reference_root = Path(destination).resolve(), Path(reference_root).resolve()
    planned = {}
    with zipfile.ZipFile(archive) as z:
        seen = set()
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            if len(parts) != 2 or parts[0] not in ARCHIVES or parts[1] not in FILES or info.filename in seen:
                raise ValueError(f"Unexpected archive member: {info.filename}")
            seen.add(info.filename)
            key = ARCHIVES[parts[0]]
            planned[destination / key / "benchmark_global" / parts[1]] = z.read(info)
        if len(seen) != len(ARCHIVES) * len(FILES):
            raise ValueError("Incomplete Ours archive")
    for key in ARCHIVES.values():
        path = reference_root / key / "benchmark_global/metadata.json"
        ref = json.loads(path.read_text(encoding="utf-8"))
        if ref.get("model_key") != key:
            raise ValueError("Wrong deployment reference model")
        record = {"model_key": key, "reference_metadata_sha256": sha256(path),
                  "scope": "previous Outcome-CAA deployment reference; not extraction provenance of this vector",
                  "model_fingerprint": ref["model_fingerprint"]}
        planned[destination / key / "benchmark_global/deployment_reference.json"] = (json.dumps(record, indent=2)+"\n").encode()
    for target, content in planned.items():
        if not target.resolve().is_relative_to(destination):
            raise ValueError("Unsafe destination")
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different artifact: {target}")
    for target, content in planned.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for key in ARCHIVES.values():
        path = destination / key / "benchmark_global/steering_vector.pt"
        meta = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
        _, digest, _ = validate_ours_vector(path, meta, key, validate_pt=True)
        print(key, digest, flush=True)
