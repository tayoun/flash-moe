#!/usr/bin/env python3
"""Build a Gemma expert index from safetensors metadata only.

The output index is consumed by repack_experts.py and validate_expert_metadata.py.
"""

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path


def parse_safetensors_header(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def load_config(model_path):
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)

    fields = {
        "num_hidden_layers": int(text_cfg["num_hidden_layers"]),
        "num_experts": int(text_cfg["num_experts"]),
        "hidden_size": int(text_cfg["hidden_size"]),
        "moe_intermediate_size": int(text_cfg["moe_intermediate_size"]),
        "top_k_experts": int(text_cfg.get("top_k_experts", text_cfg.get("num_experts_per_tok", 0))),
        "layer_types": text_cfg.get("layer_types", []),
        "model_type": text_cfg.get("model_type", cfg.get("model_type", "gemma4_text")),
    }
    return cfg, fields


def component_tensor_name(layer_idx, component):
    return f"model.language_model.layers.{layer_idx}.experts.{component}"


def main():
    parser = argparse.ArgumentParser(description="Build Gemma expert index from safetensors metadata")
    parser.add_argument("--model", required=True, help="Path to Gemma checkpoint directory")
    parser.add_argument("--output", default="expert_index.gemma4.json", help="Output JSON path")
    parser.add_argument("--strict", action="store_true", help="Fail on missing layer/components")
    args = parser.parse_args()

    model_path = Path(args.model)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")

    _, cfg = load_config(model_path)

    with open(index_path) as f:
        index_payload = json.load(f)
    weight_map = index_payload["weight_map"]

    num_layers = cfg["num_hidden_layers"]
    num_experts = cfg["num_experts"]
    hidden = cfg["hidden_size"]
    moe_intermediate = cfg["moe_intermediate_size"]

    components = ["gate_up_proj", "down_proj"]

    by_file = defaultdict(list)
    missing = []
    for layer in range(num_layers):
        for comp in components:
            tname = component_tensor_name(layer, comp)
            shard = weight_map.get(tname)
            if shard is None:
                missing.append(tname)
                continue
            by_file[shard].append(tname)

    if missing and args.strict:
        raise RuntimeError(f"missing {len(missing)} required expert tensors in strict mode")

    header_cache = {}
    for shard in sorted(by_file.keys()):
        header_cache[shard] = parse_safetensors_header(model_path / shard)

    expert_reads = {}
    observed_sizes = defaultdict(set)

    for layer in range(num_layers):
        layer_key = str(layer)
        layer_info = {}
        for comp in components:
            tname = component_tensor_name(layer, comp)
            shard = weight_map.get(tname)
            if shard is None:
                continue
            header, data_start = header_cache[shard]
            if tname not in header:
                raise RuntimeError(f"{tname} missing from shard header {shard}")

            meta = header[tname]
            offsets = meta["data_offsets"]
            total_size = offsets[1] - offsets[0]
            if total_size % num_experts != 0:
                raise RuntimeError(
                    f"{tname}: total_size {total_size} not divisible by num_experts {num_experts}"
                )

            expert_size = total_size // num_experts
            shape = meta.get("shape", [])

            entry = {
                "file": shard,
                "tensor": tname,
                "abs_offset": data_start + offsets[0],
                "expert_stride": expert_size,
                "expert_size": expert_size,
                "total_size": total_size,
                "shape": shape,
                "dtype": meta.get("dtype", "UNKNOWN"),
            }
            layer_info[comp] = entry
            observed_sizes[comp].add(expert_size)
        if layer_info:
            expert_reads[layer_key] = layer_info

    gate_sizes = observed_sizes["gate_up_proj"]
    down_sizes = observed_sizes["down_proj"]

    if len(gate_sizes) != 1 or len(down_sizes) != 1:
        raise RuntimeError(
            f"inconsistent expert component sizes across layers: gate_up={sorted(gate_sizes)} down={sorted(down_sizes)}"
        )

    gate_size = next(iter(gate_sizes))
    down_size = next(iter(down_sizes))

    layout = {
        "format_version": 1,
        "architecture": "gemma4_text_moe_bf16",
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_size": gate_size + down_size,
        "components": [
            {"name": "gate_up_proj", "offset": 0, "size": gate_size, "dtype": "BF16"},
            {"name": "down_proj", "offset": gate_size, "size": down_size, "dtype": "BF16"},
        ],
    }

    out = {
        "schema_version": 1,
        "model_path": str(model_path),
        "model_family": "gemma4",
        "source_model_type": cfg.get("model_type"),
        "model_config": {
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
            "top_k_experts": cfg.get("top_k_experts"),
            "hidden_size": hidden,
            "moe_intermediate_size": moe_intermediate,
            "layer_types": cfg.get("layer_types", []),
        },
        "layout": layout,
        "expert_reads": expert_reads,
        "missing_tensors": missing,
        "notes": [
            "metadata_only_index=true",
            "routed experts only (gate_up_proj/down_proj)",
            "vision tensors excluded",
        ],
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Model path: {model_path}")
    print(f"Layers expected: {num_layers}, indexed: {len(expert_reads)}")
    print(f"Experts/layer: {num_experts}")
    print(f"Component sizes: gate_up_proj={gate_size} bytes, down_proj={down_size} bytes")
    print(f"Output: {out_path}")
    if missing:
        print(f"Missing tensors: {len(missing)}")


if __name__ == "__main__":
    main()
