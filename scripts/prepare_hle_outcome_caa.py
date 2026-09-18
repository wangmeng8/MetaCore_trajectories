"""Offline vector import, request generation and server provenance (no inference)."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.hle_outcome_caa import MODELS, deep_merge, json_object, sha256, vector_metadata

ARCHIVES = {"Qwen36_27B_HLE_CAA_benchmark_global.zip": "qwen36_27b",
            "Gemma4_31B_IT_HLE_CAA_benchmark_global.zip": "gemma4_31b_it",
            "GPTOSS_20B_HLE_CAA_benchmark_global.zip": "gptoss_20b"}


def import_vectors(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as outer:
        for name, key in ARCHIVES.items():
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                root = destination / key
                files = {}
                for entry in inner.infolist():
                    if entry.is_dir():
                        continue
                    relative = Path(entry.filename)
                    target = (root / relative).resolve()
                    if not target.is_relative_to(root) or relative.parts[0] != "benchmark_global":
                        raise ValueError("Unsafe vector archive member")
                    content = inner.read(entry)
                    if target.exists() and target.read_bytes() != content:
                        raise ValueError(f"Refusing to overwrite different vector artifact: {target}")
                    files[target] = content
                for target, content in files.items():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                _, digest, _ = vector_metadata(root / "benchmark_global/steering_vector.pt", key)
                print(json.dumps({"model_key": key, "sha256": digest, "path": str(root)}))


def request_body(vector_path, layer, alpha, env):
    metadata, digest, _ = vector_metadata(vector_path, env.get("OUTCOME_CAA_MODEL_KEY"))
    layer = MODELS[metadata["model_key"]][0] if layer is None else layer
    if not 0 <= layer < metadata["shape"][0]:
        raise ValueError("Invalid layer for this vector")
    import math
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    body = json_object(env.get("HLE_WITH_TOOLS_EXTRA_BODY_JSON", "{}"))
    body = deep_merge(body, json_object(env.get("HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON", "{}")))
    steer = {"method": "add_vector", "position": "last_prefix_token", "apply_at_all_positions": False,
             "vector_path": str(Path(vector_path).resolve()), "vector_sha256": digest,
             "tensor_key": "steering_vector" if metadata.get("method") == "trajre_mvp" else "mean_difference", "vector_layer": layer, "optimal_layer": layer, "coefficient": alpha}
    return deep_merge(body, {"vllm_xargs": {"steer": json.dumps(steer, separators=(",", ":"))}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("import-vectors")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--destination", type=Path, default=ROOT / "data/HLE/outcome_caa_vectors")
    p = sub.add_parser("request-body")
    p.add_argument("--vector", type=Path, required=True)
    p.add_argument("--layer", type=int)
    p.add_argument("--alpha", type=float, default=1.)
    args = parser.parse_args(argv)
    if args.command == "import-vectors":
        import_vectors(args.archive, args.destination)
    else:
        from scripts.runner_common import load_dotenv_file
        load_dotenv_file()
        print(json.dumps(request_body(args.vector, args.layer, args.alpha, os.environ)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
