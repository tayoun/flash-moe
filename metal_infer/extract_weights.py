#!/usr/bin/env python3
"""
extract_weights.py — Extract non-expert weights from Qwen3.5-397B or Gemma4-26B
into a single binary file that the C inference engine can mmap.

SUPPORTED MODELS:
  - Qwen3.5 MoE (4-bit packed source, extracted as native)
  - Gemma4 26B (BF16 source — NOTE: bring-up / reference only;
    BF16 weights will OOM on 16GB Mac Mini; 4-bit path required for deployment)

The extraction script is model-family-aware and skips vision tensors for Gemma v1.

Outputs:
  - model_weights.bin: binary blob containing all non-expert weight tensors
  - model_weights.json: manifest describing each tensor's location, shape, dtype

The binary format is simple:
  - Tensors are packed contiguously, 64-byte aligned
  - Each tensor is stored in its native format (U32 packed, BF16 as uint16, F32)
  - The JSON manifest maps tensor names to {offset, size, shape, dtype}

Usage:
    python extract_weights.py [--model PATH] [--output DIR]
"""

import json
import struct
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
import re


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def load_source_config(model_path):
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return json.load(f)


def detect_model_family(source_config):
    if not source_config:
        return "qwen"
    model_type = str(source_config.get("model_type", "")).lower()
    text_cfg = source_config.get("text_config", {})
    text_model_type = str(text_cfg.get("model_type", "")).lower()
    architectures = [str(a).lower() for a in source_config.get("architectures", []) if isinstance(a, str)]
    corpus = " ".join([model_type, text_model_type] + architectures)
    if "gemma" in corpus:
        return "gemma"
    return "qwen"


def default_qwen_layer_types(num_layers=60, interval=4):
    layer_types = []
    for i in range(num_layers):
        if (i + 1) % interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")
    return layer_types


def build_manifest_config(source_config, model_family):
    # Preserve existing Qwen defaults when source config is missing or incomplete.
    cfg = {
        "architecture": "qwen3_moe",
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "full_attention_interval": 4,
        "linear_num_value_heads": 64,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000.0,
    }

    cfg["layer_types"] = default_qwen_layer_types()

    if not source_config:
        return cfg

    text_cfg = source_config.get("text_config", source_config)
    model_type = source_config.get("model_type") or text_cfg.get("model_type")
    architectures = source_config.get("architectures")
    if model_type:
        cfg["model_type"] = model_type
    if model_family == "gemma":
        cfg["architecture"] = "gemma4_text_moe"
    elif model_type:
        cfg["architecture"] = model_type
    if isinstance(architectures, list) and architectures:
        cfg["architectures"] = architectures

    scalar_fields = [
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "vocab_size", "rms_norm_eps", "num_experts", "moe_intermediate_size",
        "shared_expert_intermediate_size", "intermediate_size", "sliding_window", "global_head_dim",
        "rope_theta"
    ]
    for key in scalar_fields:
        if key in text_cfg:
            cfg[key] = text_cfg[key]

    # rope_theta_full: RoPE base for full-attention layers in Gemma (separate from sliding)
    # Gemma uses 10000.0 for sliding attention, 1000000.0 for full attention
    # This is a Gemma-specific runtime value; source config may not have it.
    if model_family == "gemma":
        cfg["rope_theta_full"] = 1000000.0  # Gemma full-attention RoPE base
    elif "rope_theta_full" in text_cfg:
        cfg["rope_theta_full"] = text_cfg["rope_theta_full"]
    else:
        cfg["rope_theta_full"] = cfg.get("rope_theta", 10000000.0)  # Qwen: same as regular

    # Gemma-specific: num_global_key_value_heads (for full attention layers)
    if "num_global_key_value_heads" in text_cfg:
        cfg["num_global_key_value_heads"] = text_cfg["num_global_key_value_heads"]

    # Gemma-specific: attention_k_eq_v flag (K and V share weights on full attention)
    if "attention_k_eq_v" in text_cfg:
        cfg["attention_k_eq_v"] = 1 if text_cfg["attention_k_eq_v"] else 0

    # Gemma-specific: final_logit_softcapping
    if "final_logit_softcapping" in text_cfg:
        cfg["final_logit_softcapping"] = text_cfg["final_logit_softcapping"]

    if "top_k_experts" in text_cfg:
        cfg["num_experts_per_tok"] = text_cfg["top_k_experts"]
    elif "num_experts_per_tok" in text_cfg:
        cfg["num_experts_per_tok"] = text_cfg["num_experts_per_tok"]

    if isinstance(text_cfg.get("layer_types"), list) and text_cfg["layer_types"]:
        cfg["layer_types"] = text_cfg["layer_types"]
    elif model_family == "qwen":
        cfg["layer_types"] = default_qwen_layer_types(
            int(cfg.get("num_hidden_layers", 60)),
            int(cfg.get("full_attention_interval", 4)),
        )

    return cfg


def classify_tensor(name, model_family, include_experts):
    qwen_expert_pattern = re.compile(
        r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$'
    )
    # Gemma 4 26B-A4B: MoE experts are layers.N.experts.gate_up_proj / down_proj
    # Dense FFN uses mlp.gate_proj / up_proj / down_proj (extract these)
    gemma_expert_pattern = re.compile(r'\.experts\.(gate_up_proj|down_proj)$')

    # Skip vision / audio / multimodal tensors
    if name.startswith(("vision_tower", "model.visual", "model.vision_tower",
                        "model.embed_vision", "model.audio_", "model.conformer")):
        return "skip_vision"

    if model_family == "gemma":
        # Gemma 4 tensor names are like: model.language_model.layers.N.xxx or model.language_model.embed_tokens.weight
        # Strip the model.language_model. prefix to get the local name
        local_name = name
        if name.startswith("model.language_model."):
            local_name = name[len("model.language_model."):]
        is_layer = local_name.startswith("layers.")
        is_top_level = local_name in ("embed_tokens.weight", "final_norm.weight",
                                       "lm_head.weight", "norm.weight")
        is_gemma = is_layer or is_top_level

        if not is_gemma:
            return "skip_non_text"
        if not include_experts and gemma_expert_pattern.search(local_name):
            return "skip_expert"
        return "extract"

    if not include_experts and qwen_expert_pattern.search(name):
        return "skip_expert"
    return "extract"


def sanitize_name(name, model_family):
    if model_family == "gemma":
        # Strip model.language_model. prefix for Gemma
        if name.startswith("model.language_model."):
            return name[len("model.language_model."):]
        return name
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def plan_tensor_layout(tensors_to_extract, header_cache, model_family, align):
    all_tensors = []  # (sanitized_name, original_name, filename)
    for name in sorted(tensors_to_extract.keys()):
        all_tensors.append((sanitize_name(name, model_family), name, tensors_to_extract[name]))

    planned_size = 0
    for _, orig_name, filename in all_tensors:
        header, _ = header_cache[filename]
        if orig_name not in header:
            continue
        if planned_size % align != 0:
            planned_size += align - (planned_size % align)
        tensor_offsets = header[orig_name]["data_offsets"]
        planned_size += tensor_offsets[1] - tensor_offsets[0]

    return all_tensors, planned_size


def ensure_disk_space(output_dir, planned_size, extra_headroom_gb):
    stat = os.statvfs(output_dir)
    free_bytes = stat.f_bavail * stat.f_frsize
    headroom_bytes = int(extra_headroom_gb * (1024 ** 3))
    needed = planned_size + headroom_bytes
    if free_bytes < needed:
        free_gb = free_bytes / (1024 ** 3)
        need_gb = needed / (1024 ** 3)
        payload_gb = planned_size / (1024 ** 3)
        raise RuntimeError(
            f"insufficient free disk space: free={free_gb:.2f} GiB, "
            f"required={need_gb:.2f} GiB (payload={payload_gb:.2f} GiB + "
            f"headroom={extra_headroom_gb:.2f} GiB)"
        )


def main():
    parser = argparse.ArgumentParser(description='Extract non-expert weights to binary')
    parser.add_argument('--model', type=str,
                        default=os.path.expanduser(
                            '~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit'
                            '/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3'),
                        help='Path to model directory')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for model_weights.bin and .json')
    parser.add_argument('--include-experts', action='store_true',
                        help='Also extract expert weights (huge, not recommended)')
    parser.add_argument('--skip-disk-check', action='store_true',
                        help='Skip output-volume free-space preflight check')
    parser.add_argument('--disk-headroom-gb', type=float, default=8.0,
                        help='Additional free-space headroom required beyond planned payload size')
    parser.add_argument('--dry-run', action='store_true',
                        help='Plan extraction and write only manifest (no model_weights.bin)')
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the weight index
    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)
    source_config = load_source_config(model_path)
    model_family = detect_model_family(source_config)

    weight_map = idx['weight_map']

    tensors_to_extract = {}  # name -> filename
    skipped_expert = 0
    skipped_vision = 0
    skipped_non_text = 0

    for name, filename in weight_map.items():
        decision = classify_tensor(name, model_family, args.include_experts)
        if decision == "skip_vision":
            skipped_vision += 1
            continue
        if decision == "skip_non_text":
            skipped_non_text += 1
            continue
        if decision == "skip_expert":
            skipped_expert += 1
            continue
        tensors_to_extract[name] = filename

    print(f"Model: {model_path}")
    print(f"Model family: {model_family}")
    print(f"Total weights in index: {len(weight_map)}")
    print(f"Skipped vision: {skipped_vision}")
    if model_family == "gemma":
        print(f"Skipped non-text tensors: {skipped_non_text}")
    print(f"Skipped expert: {skipped_expert}")
    print(f"Extracting: {len(tensors_to_extract)} tensors")

    # Group by shard file for sequential I/O
    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    # Parse headers and plan layout
    print("\nParsing safetensors headers...")
    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))

    ALIGN = 64  # 64-byte alignment for Metal buffers
    all_tensors, planned_size = plan_tensor_layout(tensors_to_extract, header_cache, model_family, ALIGN)

    if not args.skip_disk_check and not args.dry_run:
        print("\nRunning disk preflight...")
        try:
            ensure_disk_space(output_dir, planned_size, args.disk_headroom_gb)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print(
            f"Disk preflight OK: planned payload={planned_size / (1024 ** 3):.2f} GiB, "
            f"headroom={args.disk_headroom_gb:.2f} GiB"
        )

    # Build skip records for manifest (which tensors skipped and why)
    skip_records = {}
    for name, filename in weight_map.items():
        decision = classify_tensor(name, model_family, args.include_experts)
        if decision != "extract":
            skip_records[name] = {
                "decision": decision,
                "file": filename,
            }

    # Write binary file
    bin_path = output_dir / 'model_weights.bin'
    manifest = {
        "model": str(model_path),
        "model_family": model_family,
        "num_tensors": len(all_tensors),
        "tensors": {},
        "config": build_manifest_config(source_config, model_family),
        "skipped_tensors": skip_records,
        "extraction_note": (
            "NOTE: Gemma weights are BF16/native precision. The dense weights here "
            "are suitable for bring-up / reference validation. "
            "Running Gemma 26B BF16 on a 16GB Mac Mini will OOM — a 4-bit quantized "
            "or Mac-Intel-4bit artifact is required for practical deployment."
        ),
    }

    offset = 0
    total_bytes = 0

    if args.dry_run:
        print("\nDry run: skipping model_weights.bin write")
        for san_name, orig_name, filename in all_tensors:
            header, _ = header_cache[filename]
            if orig_name not in header:
                continue
            meta = header[orig_name]
            tensor_offsets = meta['data_offsets']
            byte_len = tensor_offsets[1] - tensor_offsets[0]
            if offset % ALIGN != 0:
                offset += ALIGN - (offset % ALIGN)
            manifest["tensors"][san_name] = {
                "offset": offset,
                "size": byte_len,
                "shape": meta["shape"],
                "dtype": meta["dtype"],
            }
            offset += byte_len
            total_bytes += byte_len
    else:
        print(f"\nWriting {bin_path}...")
        t0 = time.time()
        with open(bin_path, 'wb') as out_f:
            for i, (san_name, orig_name, filename) in enumerate(all_tensors):
                filepath = model_path / filename
                header, data_start = header_cache[filename]

                if orig_name not in header:
                    print(f"  WARNING: {orig_name} not found in {filename}, skipping")
                    continue

                meta = header[orig_name]
                tensor_offsets = meta['data_offsets']
                byte_len = tensor_offsets[1] - tensor_offsets[0]
                shape = meta['shape']
                dtype = meta['dtype']

                # Align offset
                if offset % ALIGN != 0:
                    pad = ALIGN - (offset % ALIGN)
                    out_f.write(b'\x00' * pad)
                    offset += pad

                # Read tensor data from safetensors
                with open(filepath, 'rb') as sf:
                    sf.seek(data_start + tensor_offsets[0])
                    data = sf.read(byte_len)

                out_f.write(data)

                manifest["tensors"][san_name] = {
                    "offset": offset,
                    "size": byte_len,
                    "shape": shape,
                    "dtype": dtype,
                }

                offset += byte_len
                total_bytes += byte_len

                if (i + 1) % 100 == 0 or i == len(all_tensors) - 1:
                    print(f"  [{i+1}/{len(all_tensors)}] {total_bytes / 1e9:.2f} GB written")

        elapsed = time.time() - t0
        throughput = total_bytes / elapsed / 1e9
        print(f"\nDone: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s ({throughput:.1f} GB/s)")
        print(f"Binary: {bin_path} ({os.path.getsize(bin_path) / 1e9:.2f} GB)")

    # Write manifest
    json_path = output_dir / 'model_weights.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {json_path}")

    # Print summary by category
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for san_name, info in manifest["tensors"].items():
        if "embed_tokens" in san_name:
            cat = "embedding"
        elif "norm.weight" in san_name and "layers." not in san_name:
            cat = "final_norm"
        elif "lm_head" in san_name:
            cat = "lm_head"
        elif "input_layernorm" in san_name or "post_attention_layernorm" in san_name:
            cat = "layer_norms"
        elif "linear_attn" in san_name:
            cat = "linear_attention"
        elif "self_attn" in san_name:
            cat = "full_attention"
        elif "mlp.gate." in san_name:
            cat = "routing_gate"
        elif "shared_expert." in san_name:
            cat = "shared_expert"
        elif "shared_expert_gate" in san_name:
            cat = "shared_expert_gate"
        elif "switch_mlp" in san_name:
            cat = "routed_experts"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories.keys()):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes']/1e6:8.1f} MB")


if __name__ == '__main__':
    main()
