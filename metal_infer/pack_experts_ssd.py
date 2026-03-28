#!/usr/bin/env python3
"""
Pack all expert files from packed_experts/ into a single sequential SSD file.

Layout:
  [header: 16 bytes]
    - num_layers:    uint32 (big-endian)
    - num_experts:  uint32 (big-endian)
    - expert_size:  uint64 (big-endian)
  [data: num_layers * num_experts * expert_size bytes]

Expert offset in packed file:
  offset = 16 + (layer * num_experts + expert) * expert_size

Usage:
  python3 pack_experts_ssd.py [--out packed_experts_ssd.bin]
"""

import struct
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Pack expert files into sequential SSD layout")
    parser.add_argument("--packed-dir", default="out_122b/packed_experts",
                        help="Directory containing layer_XX.bin files")
    parser.add_argument("--out", default="out_122b/packed_experts_ssd.bin",
                        help="Output packed SSD file")
    parser.add_argument("--expert-size", type=int, default=5308416,
                        help="Bytes per expert (default: 5308416)")
    args = parser.parse_args()

    packed_dir = args.packed_dir
    output_path = args.out
    expert_size = args.expert_size

    # Find all layer files
    layer_files = []
    for fname in sorted(os.listdir(packed_dir)):
        if fname.startswith("layer_") and fname.endswith(".bin"):
            layer_files.append(os.path.join(packed_dir, fname))

    if not layer_files:
        print(f"ERROR: No layer_XX.bin files found in {packed_dir}", file=sys.stderr)
        sys.exit(1)

    num_layers = len(layer_files)
    num_experts = 256  # 256 experts per layer (from layout.json)

    print(f"Packing {num_layers} layers x {num_experts} experts = {num_layers * num_experts} total experts")
    print(f"Expert size: {expert_size:,} bytes ({expert_size / 1024 / 1024:.2f} MB)")
    total_size = 16 + num_layers * num_experts * expert_size
    print(f"Total packed size: {total_size:,} bytes ({total_size / 1024**3:.2f} GB)")
    print(f"Output: {output_path}")

    # Create output file
    with open(output_path, "wb") as fout:
        # Write header
        header = struct.pack(">IIQ", num_layers, num_experts, expert_size)
        fout.write(header)

        # Read each layer file and append
        for i, lf_path in enumerate(layer_files):
            fname = os.path.basename(lf_path)
            layer_size = os.path.getsize(lf_path)
            expected = num_experts * expert_size

            if layer_size != expected:
                print(f"WARNING: {fname} size {layer_size:,} != expected {expected:,}", file=sys.stderr)
                # Pad or truncate
                if layer_size < expected:
                    print(f"  Padding {expected - layer_size:,} bytes with zeros", file=sys.stderr)
                else:
                    print(f"  Truncating to {expected:,} bytes", file=sys.stderr)

            with open(lf_path, "rb") as lf:
                # Read in chunks to handle large files
                remaining = layer_size
                chunk_size = 64 * 1024 * 1024  # 64MB chunks
                while remaining > 0:
                    to_read = min(chunk_size, remaining)
                    data = lf.read(to_read)
                    if not data:
                        break
                    fout.write(data)
                    remaining -= len(data)

                # Pad if file was smaller than expected
                if remaining > 0:
                    fout.write(b"\x00" * remaining)

            print(f"  [{i+1:2d}/{num_layers}] {fname} -> appended ({layer_size:,} bytes)")

    print(f"\nDone! Packed file: {output_path}")
    actual = os.path.getsize(output_path)
    print(f"Actual size: {actual:,} bytes ({actual / 1024**3:.2f} GB)")

if __name__ == "__main__":
    main()
