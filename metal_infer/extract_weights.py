#!/usr/bin/env python3
"""
extract_weights.py — Extract Gemma 4 26B-A4B text-only dense weights.

Outputs:
  model_weights.bin  — flat binary blob of all non-expert, non-vision weight tensors
  model_weights.json  — manifest with tensor names, shapes, offsets, dtypes

MoE expert weights (experts.gate_up_proj, experts.down_proj per layer) are NOT
included — those go to the slot-bank expert packer (Task 12).

Usage:
  python extract_weights.py \
    --model /Volumes/Seagate\ Backup\ Plus\ Drive/gemma-4-26B-A4B-it/ \
    --out-dir /Volumes/Seagate\ Backup\ Plus\ Drive/gemma-4-26B-A4B-it/out
"""

import argparse
import json
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# SafeTensor reading helpers
# ---------------------------------------------------------------------------

def parse_safetensors_index(model_dir: Path):
    idx_path = model_dir / "model.safetensors.index.json"
    with open(idx_path) as f:
        idx = json.load(f)
    return idx["weight_map"]


def parse_safetensors_header(path: Path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


# ---------------------------------------------------------------------------
# Manifest / I/O
# ---------------------------------------------------------------------------

ALIGN = 64  # 64-byte alignment for Metal buffers

DTYPE_MAP = {
    "BF16": "BF16",
    "F16":  "F16",
    "F32":  "F32",
    "U16":  "U16",
    "U32":  "U32",
    "I32":  "I32",
}

# Skip these substrings — vision / audio towers
SKIP_SUBSTRINGS = ("vision_tower", "embed_vision", "audio_")

# MoE per-expert weights — these go to the slot-bank expert packer (Task 12),
# NOT to model_weights.bin
MOE_EXPERT_TENSORS = ("experts.gate_up_proj", "experts.down_proj")


def strip_prefix(name: str) -> str:
    """Strip the model.language_model. prefix so C engine sees clean names.
    Also rename the final norm to final_norm.weight for clarity."""
    PREFIX = "model.language_model."
    if name.startswith(PREFIX):
        stripped = name[len(PREFIX):]
        if stripped == "norm.weight":
            return "final_norm.weight"
        return stripped
    return name


def natural_key(name: str):
    out = []
    cur = ""
    is_digit = None
    for ch in name:
        d = ch.isdigit()
        if is_digit is None:
            cur = ch
            is_digit = d
        elif d == is_digit:
            cur += ch
        else:
            out.append(int(cur) if is_digit else cur)
            cur = ch
            is_digit = d
    if cur:
        out.append(int(cur) if is_digit else cur)
    return out


def category_of(name: str) -> str:
    if "embed_tokens" in name:
        return "embed_tokens"
    if name == "final_norm.weight":
        return "final_norm"
    if "lm_head" in name:
        return "lm_head"
    if "_layernorm" in name or "layer_scalar" in name:
        return "layer_norms"
    if "self_attn" in name:
        return "self_attn"
    if "mlp." in name:
        return "mlp"
    if "router" in name:
        return "router"
    if "experts" in name:
        return "experts"  # should not reach here normally
    return "other"


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="/Volumes/Seagate Backup Plus Drive/gemma-4-26B-A4B-it/",
    )
    ap.add_argument(
        "--out-dir",
        default="/Volumes/Seagate Backup Plus Drive/gemma-4-26B-A4B-it/out",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser()
    out_dir   = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    tc = config["text_config"]

    num_layers   = tc["num_hidden_layers"]
    layer_types  = tc["layer_types"]   # list of "sliding_attention" | "full_attention"
    full_attn_layers = {i for i, lt in enumerate(layer_types) if lt == "full_attention"}
    rope_params  = tc.get("rope_parameters", {})

    print(f"Model:          {model_dir}")
    print(f"Num layers:     {num_layers}")
    print(f"Full-attn:      {sorted(full_attn_layers)}")
    print(f"Sliding window: {tc.get('sliding_window', '?')}")

    # ------------------------------------------------------------------
    # 2. Load weight map
    # ------------------------------------------------------------------
    weight_map = parse_safetensors_index(model_dir)

    # Group by shard for sequential I/O
    by_shard = defaultdict(dict)
    for tensor_name, shard_name in weight_map.items():
        by_shard[shard_name][tensor_name] = True

    # Parse all shard headers once
    shard_headers = {}
    for shard_name in by_shard:
        path = model_dir / shard_name
        header, data_start = parse_safetensors_header(path)
        shard_headers[shard_name] = (header, data_start)

    # ------------------------------------------------------------------
    # 3. Select tensors: skip vision_tower, embed_vision, audio_,
    #    and MoE expert weights (experts.gate_up_proj, experts.down_proj)
    # ------------------------------------------------------------------
    selected = []   # list of (stripped_name, original_name, shard_name)

    for orig_name in sorted(weight_map.keys(), key=natural_key):
        # Skip vision / audio
        if any(tok in orig_name for tok in SKIP_SUBSTRINGS):
            continue

        # Skip MoE expert tensors (go to slot-bank packer)
        if any(tok in orig_name for tok in MOE_EXPERT_TENSORS):
            continue

        shard_name = weight_map[orig_name]
        stripped = strip_prefix(orig_name)
        selected.append((stripped, orig_name, shard_name))

    print(f"\nSelected tensors: {len(selected)}")

    if args.dry_run:
        for stripped, orig, shard in selected[:60]:
            header, _ = shard_headers[shard]
            meta = header[orig]
            start, end = meta["data_offsets"]
            print(f"  {stripped:60s} {str(meta['shape']):40s} {meta['dtype']:6s} {end-start:12d}  [{shard}]")
        return

    # ------------------------------------------------------------------
    # 4. Write binary + manifest
    # ------------------------------------------------------------------
    bin_path      = out_dir / "model_weights.bin"
    manifest_path = out_dir / "model_weights.json"

    manifest = {
        "model": str(model_dir),
        "config": {
            "hidden_size":              tc["hidden_size"],
            "num_hidden_layers":        num_layers,
            "num_attention_heads":      tc["num_attention_heads"],
            "num_key_value_heads":      tc["num_key_value_heads"],
            "head_dim":                 tc["head_dim"],
            "vocab_size":               tc["vocab_size"],
            "rms_norm_eps":             tc.get("rms_norm_eps", 1e-6),
            "intermediate_size":        tc.get("intermediate_size", 0),
            "moe_intermediate_size":    tc.get("moe_intermediate_size", 0),
            "num_experts":              tc.get("num_experts", 0),
            "num_experts_per_tok":      tc.get("top_k_experts", 0),
            "sliding_window":           tc.get("sliding_window", 0),
            "full_attention_layers":    sorted(full_attn_layers),
            "rope_theta_sliding":       rope_params.get("sliding_attention", {}).get("rope_theta", 10000),
            "rope_theta_full":          rope_params.get("full_attention", {}).get("rope_theta", 1_000_000),
            "partial_rotary_factor":    rope_params.get("full_attention", {}).get("partial_rotary_factor", 0.25),
            "hidden_activation":        tc.get("hidden_activation", "gelu_pytorch_tanh"),
            "layer_types":              layer_types,
        },
        "tensors": {},
    }

    t0 = time.time()
    offset = 0
    total_bytes = 0

    with open(bin_path, "wb") as out_f:
        for idx, (stripped, orig, shard_name) in enumerate(selected, 1):
            header, data_start = shard_headers[shard_name]

            if orig not in header:
                print(f"  WARNING: {orig} not found in {shard_name}, skipping")
                continue

            meta = header[orig]
            start_off, end_off = meta["data_offsets"]
            byte_len = end_off - start_off
            shape    = meta["shape"]
            dtype    = DTYPE_MAP.get(meta["dtype"], meta["dtype"])

            # 64-byte alignment
            if offset % ALIGN != 0:
                pad = ALIGN - (offset % ALIGN)
                out_f.write(b"\x00" * pad)
                offset += pad

            # Read tensor bytes from safetensors
            with open(model_dir / shard_name, "rb") as sf:
                sf.seek(data_start + start_off)
                data = sf.read(byte_len)

            out_f.write(data)

            manifest["tensors"][stripped] = {
                "offset": offset,
                "size":   byte_len,
                "shape":  shape,
                "dtype":  dtype,
            }

            offset += byte_len
            total_bytes += byte_len

            if idx % 100 == 0 or idx == len(selected):
                elapsed = time.time() - t0
                rate = total_bytes / elapsed / 1e9 if elapsed > 0 else 0
                print(f"  [{idx:3d}/{len(selected)}] {total_bytes/1e9:.3f} GB  ({rate:.2f} GB/s)")

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # 5. Write manifest
    # ------------------------------------------------------------------
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    print(f"\nDone: {total_bytes/1e9:.3f} GB in {elapsed:.1f}s")
    print(f"  Binary:   {bin_path}  ({bin_path.stat().st_size/1e9:.3f} GB)")
    print(f"  Manifest: {manifest_path}")

    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for name, info in manifest["tensors"].items():
        cat = category_of(name)
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in ["embed_tokens", "final_norm", "lm_head", "layer_norms",
                "self_attn", "mlp", "router", "layer_scalar", "experts", "other"]:
        if cat in categories:
            info = categories[cat]
            print(f"  {cat:20s}: {info['count']:4d} tensors,  {info['bytes']/1e6:8.1f} MB")


if __name__ == "__main__":
    main()
