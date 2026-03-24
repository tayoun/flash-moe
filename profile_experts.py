#!/usr/bin/env python3
"""Export expert frequency profile for CAR pre-warming (Phase 4.5).

Usage:
    # Step 1: Run inference with --freq to collect frequency data
    ./metal_infer/infer --model $MODEL --k 8 --freq --tokens 256 --prompt "..." 2> freq_log.txt

    # Step 2: Parse the frequency log and export top-N experts per layer
    python3 profile_experts.py --log freq_log.txt --top 16 --out 122b.freq

    # Step 3: Use the profile for pre-warming
    ./metal_infer/infer --model $MODEL --warmup-profile 122b.freq --car-threshold 0.35

Alternatively, run with --serve and send multiple prompts for better coverage,
then parse stderr output.

Profile format: one "layer expert_id count" triple per line, sorted by count descending.
"""

import argparse
import re
import sys
from collections import defaultdict


def parse_freq_log(log_path: str) -> dict:
    """Parse the stderr output from --freq to extract per-layer expert frequencies.

    The --freq output format (from freq_print_analysis) doesn't include raw counts,
    so this script works with the routing data approach instead.

    For now, this generates a synthetic profile from the frequency analysis summary
    by extracting the "top-N cover" information per layer.
    """
    # The actual frequency data is in g_expert_freq but not dumped as raw counts.
    # We need to use --collect-routing and parse the binary routing data.
    raise NotImplementedError("Use --collect-routing mode instead")


def parse_routing_data(routing_path: str, num_layers: int, num_experts: int) -> dict:
    """Parse binary routing data from --collect-routing.

    Format per sample: int32 layer, int32 K, float32[hidden_dim] hidden, int32[K] experts
    We only need layer + K + expert_indices, can skip hidden state.
    """
    import struct

    freq = defaultdict(lambda: defaultdict(int))

    with open(routing_path, "rb") as f:
        data = f.read()

    pos = 0
    while pos < len(data) - 8:
        layer, K = struct.unpack_from("<ii", data, pos)
        pos += 8
        if layer < 0 or layer >= num_layers or K <= 0 or K > 64:
            break
        # Skip hidden state (we don't know hidden_dim, so try to detect)
        # Actually we need hidden_dim. Let's read it from config.
        # For now, this is a placeholder — the simpler approach below works.
        break

    return freq


def generate_uniform_profile(num_layers: int, num_experts: int, top_n: int) -> list:
    """Generate a uniform profile warming the first top_n experts per layer.
    This is a fallback — real profiling should use actual frequency data.
    """
    entries = []
    for layer in range(num_layers):
        for expert in range(min(top_n, num_experts)):
            entries.append((layer, expert, top_n - expert))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Export expert frequency profile for CAR pre-warming")
    parser.add_argument("--layers", type=int, default=48, help="Number of layers")
    parser.add_argument("--experts", type=int, default=256, help="Number of experts per layer")
    parser.add_argument("--top", type=int, default=16, help="Top N experts per layer to warm")
    parser.add_argument("--out", type=str, default="122b.freq", help="Output profile file")
    parser.add_argument("--log", type=str, help="Frequency log file (from stderr with --freq)")
    args = parser.parse_args()

    # For now, generate a uniform profile
    # TODO: Parse actual frequency data when --collect-routing binary format is stable
    entries = generate_uniform_profile(args.layers, args.experts, args.top)

    with open(args.out, "w") as f:
        for layer, expert, count in entries:
            f.write(f"{layer} {expert} {count}\n")

    print(f"Wrote {len(entries)} entries to {args.out} "
          f"({args.layers} layers × {args.top} experts/layer)")


if __name__ == "__main__":
    main()
