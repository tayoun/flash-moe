#!/usr/bin/env python3
"""Build expert index for Gemma 4 26B-A4B MoE experts.

Reads safetensors index, extracts expert tensor metadata, and produces:
1. expert-index.json — metadata for runtime expert loading via pread
2. Creates packed_experts/gemma4/ directory structure

Binary layout of packed_experts/gemma4/layer_XX.bin:
  Header (64 bytes):
    magic: "GEMM" (4 bytes)
    version: uint32 = 1
    num_experts: uint32 = 128
    expert_hidden: uint32 = 704
    hidden_size: uint32 = 2816
    gate_up_bytes: uint64 (per expert = 7,963,648)
    down_bytes: uint64 (per expert = 3,964,928)
    reserved: 32 bytes

  Per expert e (0-127):
    gate_proj:  [704, 2816] BF16 = 3,981,824 bytes
    up_proj:    [704, 2816] BF16 = 3,981,824 bytes
    down_proj:  [2816, 704] BF16 = 7,512,064 bytes
    Per expert total: 15,475,712 bytes
    All experts: 128 × 15,475,712 = 1,980,891,136 bytes ≈ 1.85 GB per layer

Usage:
  python tools/gemma4/build_expert_index.py --model /path/to/gemma/checkpoint --output /path/to/output
"""

import json
import struct
import argparse
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants — Gemma 4 26B-A4B
# ---------------------------------------------------------------------------
GEMMA4_HEADER_BYTES = 64
GEMMA4_NUM_EXPERTS = 128
GEMMA4_EXPERT_HIDDEN = 704
GEMMA4_HIDDEN_SIZE = 2816
GEMMA4_GATE_UP_BYTES_PER_EXPERT = GEMMA4_EXPERT_HIDDEN * GEMMA4_HIDDEN_SIZE * 2 * 2  # 7,963,648
GEMMA4_DOWN_BYTES_PER_EXPERT = GEMMA4_HIDDEN_SIZE * GEMMA4_EXPERT_HIDDEN * 2        # 3,964,928
GEMMA4_PER_EXPERT_TOTAL = GEMMA4_GATE_UP_BYTES_PER_EXPERT + GEMMA4_DOWN_BYTES_PER_EXPERT  # 11,928,576
GEMMA4_TOTAL_BYTES_PER_LAYER = GEMMA4_NUM_EXPERTS * GEMMA4_PER_EXPERT_TOTAL          # 1,526,857,728

# In the safetensors, gate_up_proj is stored as [128, 1408, 2816] BF16
# So each expert's gate+up occupies: 1408 * 2816 * 2 = 7,963,648 bytes
GEMMA4_GATE_UP_STRIDE = 1408 * GEMMA4_HIDDEN_SIZE * 2   # 7,963,648
GEMMA4_DOWN_STRIDE = GEMMA4_HIDDEN_SIZE * GEMMA4_EXPERT_HIDDEN * 2  # 3,964,928


def parse_safetensors_header(filepath: str):
    """Return (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def layer_tensor_name(layer_idx: int, component: str) -> str:
    return f"model.language_model.layers.{layer_idx}.experts.{component}"


def main():
    parser = argparse.ArgumentParser(description="Build Gemma 4 expert index from safetensors metadata")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to Gemma 4 checkpoint directory (contains model.safetensors.index.json)")
    parser.add_argument('--output', type=str, default='.',
                        help="Output directory for expert-index.json and packed_experts/gemma4/")
    parser.add_argument('--pack', action='store_true',
                        help="Also generate packed expert binaries")
    parser.add_argument('--strict', action='store_true',
                        help="Fail on missing expert tensors")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)

    # -------------------------------------------------------------------------
    # Load safetensors index
    # -------------------------------------------------------------------------
    index_file = model_path / 'model.safetensors.index.json'
    if not index_file.exists():
        raise FileNotFoundError(f"Safetensors index not found: {index_file}")

    with open(index_file) as f:
        idx = json.load(f)
    weight_map = idx['weight_map']

    # -------------------------------------------------------------------------
    # Detect number of layers from expert tensor names
    # -------------------------------------------------------------------------
    num_layers = 0
    for name in weight_map:
        if 'experts.gate_up_proj' in name:
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                        num_layers = max(num_layers, layer_num + 1)
                    except ValueError:
                        pass

    if num_layers == 0:
        raise RuntimeError("Could not detect number of layers from expert tensor names")

    print(f"Detected {num_layers} layers with expert tensors")

    # -------------------------------------------------------------------------
    # Load all safetensors headers (cache)
    # -------------------------------------------------------------------------
    files = sorted(set(weight_map.values()))
    headers = {}
    for fname in files:
        filepath = model_path / fname
        if not filepath.exists():
            raise FileNotFoundError(f"Safetensors file not found: {filepath}")
        header, data_start = parse_safetensors_header(str(filepath))
        headers[fname] = (header, data_start)

    # -------------------------------------------------------------------------
    # Build expert index
    # -------------------------------------------------------------------------
    expert_index = {
        'version': 1,
        'num_experts': GEMMA4_NUM_EXPERTS,
        'expert_hidden': GEMMA4_EXPERT_HIDDEN,
        'hidden_size': GEMMA4_HIDDEN_SIZE,
        'gate_up_bytes_per_expert': GEMMA4_GATE_UP_BYTES_PER_EXPERT,
        'down_bytes_per_expert': GEMMA4_DOWN_BYTES_PER_EXPERT,
        'per_expert_total_bytes': GEMMA4_PER_EXPERT_TOTAL,
        'total_bytes_per_layer': GEMMA4_TOTAL_BYTES_PER_LAYER,
        'header_bytes': GEMMA4_HEADER_BYTES,
        'layers': []
    }

    missing = []
    layers_sorted = []

    for layer_idx in range(num_layers):
        gate_name = layer_tensor_name(layer_idx, 'gate_up_proj')
        down_name = layer_tensor_name(layer_idx, 'down_proj')

        gate_shard = weight_map.get(gate_name)
        down_shard = weight_map.get(down_name)

        if gate_shard is None or down_shard is None:
            missing.append(layer_idx)
            if args.strict:
                raise RuntimeError(
                    f"Layer {layer_idx}: gate_up_proj={gate_shard}, down_proj={down_shard}"
                )
            continue

        gate_header, gate_data_start = headers[gate_shard]
        down_header, down_data_start = headers[down_shard]

        gate_meta = gate_header.get(gate_name)
        down_meta = down_header.get(down_name)

        if gate_meta is None:
            raise RuntimeError(f"Layer {layer_idx}: tensor {gate_name} not in shard header {gate_shard}")
        if down_meta is None:
            raise RuntimeError(f"Layer {layer_idx}: tensor {down_name} not in shard header {down_shard}")

        gate_offsets = gate_meta['data_offsets']
        down_offsets = down_meta['data_offsets']

        # gate_up_proj in safetensors: [128, 1408, 2816] BF16
        # expert e: gate at offset + e * 1408 * 2816 * 2
        #           up at offset + (e * 1408 + 704) * 2816 * 2
        # down_proj in safetensors: [128, 2816, 704] BF16
        # expert e: at offset + e * 2816 * 704 * 2

        gate_abs_offset = gate_data_start + gate_offsets[0]
        down_abs_offset = down_data_start + down_offsets[0]

        # Offsets within the packed binary
        # Header (64 bytes), then all gate_up for all 128 experts, then all down for all 128 experts
        gate_up_packed_offset = GEMMA4_HEADER_BYTES
        down_packed_offset = GEMMA4_HEADER_BYTES + GEMMA4_NUM_EXPERTS * GEMMA4_GATE_UP_BYTES_PER_EXPERT

        layer_entry = {
            'layer': layer_idx,
            'gate_up_proj': {
                'filename': gate_shard,
                'offset': gate_abs_offset,
                'shape': gate_meta['shape'],        # [128, 1408, 2816]
                'dtype': gate_meta.get('dtype', 'BF16'),
                'expert_stride': GEMMA4_GATE_UP_STRIDE,  # 7,963,648
            },
            'down_proj': {
                'filename': down_shard,
                'offset': down_abs_offset,
                'shape': down_meta['shape'],        # [128, 2816, 704]
                'dtype': down_meta.get('dtype', 'BF16'),
                'expert_stride': GEMMA4_DOWN_STRIDE,   # 3,964,928
            },
            'packed_file': f'packed_experts/gemma4/layer_{layer_idx:02d}.bin',
            'gate_up_offset': gate_up_packed_offset,
            'down_offset': down_packed_offset,
        }
        layers_sorted.append(layer_entry)

    # Sort by layer index
    layers_sorted.sort(key=lambda x: x['layer'])
    expert_index['layers'] = layers_sorted

    # -------------------------------------------------------------------------
    # Write expert-index.json
    # -------------------------------------------------------------------------
    output_path.mkdir(parents=True, exist_ok=True)
    packed_dir = output_path / 'packed_experts' / 'gemma4'
    packed_dir.mkdir(parents=True, exist_ok=True)

    index_file_out = output_path / 'expert-index.json'
    with open(index_file_out, 'w') as f:
        json.dump(expert_index, f, indent=2)

    print(f"Expert index written to: {index_file_out}")
    print(f"  version:            {expert_index['version']}")
    print(f"  num_experts:        {expert_index['num_experts']}")
    print(f"  expert_hidden:      {expert_index['expert_hidden']}")
    print(f"  hidden_size:        {expert_index['hidden_size']}")
    print(f"  per_expert_bytes:   {expert_index['per_expert_total_bytes']:,}")
    print(f"  total_bytes/layer: {expert_index['total_bytes_per_layer']:,}")
    print(f"  layers indexed:     {len(layers_sorted)}")
    if missing:
        print(f"  MISSING layers:     {missing} (strict={args.strict})")

    # -------------------------------------------------------------------------
    # Optionally pack expert binaries
    # -------------------------------------------------------------------------
    if args.pack:
        print("\nPacking expert binaries...")
        _pack_all_layers(expert_index, model_path, packed_dir)


def _pack_all_layers(expert_index, model_path, output_dir):
    """Pack all layers into per-layer .bin files with 64-byte header."""
    layers = expert_index['layers']
    num_experts = expert_index['num_experts']
    total_size = expert_index['total_bytes_per_layer']

    # Open all source safetensors files
    needed_files = {}
    for layer in layers:
        for key in ('gate_up_proj', 'down_proj'):
            fpath = layer[key]['filename']
            if fpath not in needed_files:
                fd = os.open(model_path / fpath, os.O_RDONLY)
                needed_files[fpath] = fd

    def make_header():
        return struct.pack(
            '<4s III QQ 32s',
            b'GEMM',          # magic
            1,                # version
            num_experts,      # num_experts
            GEMMA4_EXPERT_HIDDEN,  # expert_hidden
            GEMMA4_HIDDEN_SIZE,    # hidden_size
            GEMMA4_GATE_UP_BYTES_PER_EXPERT,
            GEMMA4_DOWN_BYTES_PER_EXPERT,
            b'\x00' * 32,    # reserved
        )

    try:
        for layer in layers:
            layer_idx = layer['layer']
            out_path = output_dir / f'layer_{layer_idx:02d}.bin'

            fd_out = os.open(str(out_path), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd_out, GEMMA4_HEADER_BYTES + total_size)

            # Write header
            os.pwrite(fd_out, make_header(), 0)

            # Read gate_up_proj for all experts
            gate_info = layer['gate_up_proj']
            gate_up_size = num_experts * GEMMA4_GATE_UP_BYTES_PER_EXPERT
            gate_src_fd = needed_files[gate_info['filename']]
            gate_data = os.pread(gate_src_fd, gate_up_size, gate_info['offset'])
            os.pwrite(fd_out, gate_data, GEMMA4_HEADER_BYTES)

            # Read down_proj for all experts
            down_info = layer['down_proj']
            down_size = num_experts * GEMMA4_DOWN_BYTES_PER_EXPERT
            down_src_fd = needed_files[down_info['filename']]
            down_data = os.pread(down_src_fd, down_size, down_info['offset'])
            down_offset = GEMMA4_HEADER_BYTES + num_experts * GEMMA4_GATE_UP_BYTES_PER_EXPERT
            os.pwrite(fd_out, down_data, down_offset)

            os.close(fd_out)
            print(f"  Packed layer {layer_idx:2d}: {out_path}")

    finally:
        for fd in needed_files.values():
            os.close(fd)


if __name__ == '__main__':
    main()
