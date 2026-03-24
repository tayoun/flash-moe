#!/usr/bin/env python3
"""Cluster experts by co-occurrence and repack layer files.

Reads co-occurrence data from infer --cooccur output, clusters experts so
frequently co-occurring ones are adjacent in the layer file, then repacks.

Usage:
    # Step 1: Collect co-occurrence data (run ~500 tokens)
    ./metal_infer/infer --model $MODEL --weights ... --cooccur cooccur.json \
        --prompt "Explain why mixture-of-experts models improve compute efficiency." \
        --tokens 500

    # Step 2: Cluster and repack
    python3 cluster_experts.py \
        --cooccur cooccur.json \
        --packed-dir metal_infer/out_122b/packed_experts \
        --output-dir metal_infer/out_122b/packed_experts_clustered

    # Step 3: Benchmark with clustered layout
    # (symlink or copy clustered dir to packed_experts)
"""

import argparse
import json
import os
import shutil
import sys
import time


def load_cooccurrence(path):
    """Load co-occurrence JSON from infer --cooccur output."""
    with open(path) as f:
        data = json.load(f)
    return data


def greedy_cluster(cooccur_matrix, num_experts):
    """Greedy nearest-neighbor clustering: start from the most active expert,
    then always pick the unvisited expert with highest co-occurrence to the
    current one. This produces an ordering where co-occurring experts are adjacent.
    """
    # Sum co-occurrence to find the most connected expert as starting point
    total_cooccur = [sum(cooccur_matrix[e]) for e in range(num_experts)]
    start = max(range(num_experts), key=lambda e: total_cooccur[e])

    visited = set()
    order = []
    current = start

    for _ in range(num_experts):
        order.append(current)
        visited.add(current)
        # Find unvisited expert with highest co-occurrence to current
        best_next = -1
        best_score = -1
        for e in range(num_experts):
            if e not in visited and cooccur_matrix[current][e] > best_score:
                best_score = cooccur_matrix[current][e]
                best_next = e
        if best_next == -1:
            # All remaining experts have zero co-occurrence; add them in order
            for e in range(num_experts):
                if e not in visited:
                    best_next = e
                    break
        current = best_next

    return order


def repack_layer(layer_idx, order, packed_dir, output_dir, expert_size, in_place=False):
    """Repack a single layer file using the new expert ordering.

    order[new_position] = old_expert_id
    So we read old expert at order[i] and write to position i.

    If in_place=True, reads entire layer into memory, reorders, writes back.
    """
    src_path = os.path.join(packed_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(src_path):
        print(f"  Layer {layer_idx}: source file not found, skipping")
        return 0

    num_experts = len(order)
    layer_size = num_experts * expert_size

    if in_place:
        # Read entire layer into memory, reorder, write back
        with open(src_path, 'rb') as f:
            layer_data = f.read(layer_size)
        if len(layer_data) != layer_size:
            raise IOError(f"Short read: layer {layer_idx}, expected {layer_size}, got {len(layer_data)}")

        # Build reordered buffer
        reordered = bytearray(layer_size)
        for new_pos, old_id in enumerate(order):
            src_start = old_id * expert_size
            dst_start = new_pos * expert_size
            reordered[dst_start:dst_start + expert_size] = \
                layer_data[src_start:src_start + expert_size]

        # Write back
        with open(src_path, 'wb') as f:
            f.write(reordered)
        del layer_data, reordered
        return layer_size
    else:
        dst_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
        src_fd = os.open(src_path, os.O_RDONLY)
        dst_fd = os.open(dst_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(dst_fd, layer_size)

        for new_pos, old_id in enumerate(order):
            data = os.pread(src_fd, expert_size, old_id * expert_size)
            if len(data) != expert_size:
                raise IOError(f"Short read: layer {layer_idx}, expert {old_id}")
            os.pwrite(dst_fd, data, new_pos * expert_size)

        os.close(src_fd)
        os.close(dst_fd)
        return layer_size


def main():
    parser = argparse.ArgumentParser(
        description="Cluster experts by co-occurrence and repack layer files")
    parser.add_argument('--cooccur', required=True,
                        help='Co-occurrence JSON from infer --cooccur')
    parser.add_argument('--packed-dir', required=True,
                        help='Source packed_experts directory')
    parser.add_argument('--output-dir', default=None,
                        help='Destination directory for clustered layer files')
    parser.add_argument('--in-place', action='store_true',
                        help='Reorder layer files in-place (reads entire layer into RAM)')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show clustering stats without repacking')
    args = parser.parse_args()

    # Load layout from source dir
    layout_path = os.path.join(args.packed_dir, "layout.json")
    if not os.path.exists(layout_path):
        print(f"ERROR: layout.json not found in {args.packed_dir}")
        sys.exit(1)
    with open(layout_path) as f:
        layout = json.load(f)
    expert_size = layout['expert_size']
    num_experts = layout['num_experts']
    num_layers = layout['num_layers']

    # Load co-occurrence
    print(f"Loading co-occurrence data from {args.cooccur}...")
    cooccur_data = load_cooccurrence(args.cooccur)
    cooccur_ne = cooccur_data['num_experts']
    cooccur_nl = cooccur_data['num_layers']
    tokens = cooccur_data['tokens']
    print(f"  {cooccur_nl} layers, {cooccur_ne} experts, {tokens} tokens")

    if cooccur_ne != num_experts or cooccur_nl != num_layers:
        print(f"ERROR: co-occurrence dimensions ({cooccur_nl}L x {cooccur_ne}E) "
              f"don't match layout ({num_layers}L x {num_experts}E)")
        sys.exit(1)

    # Parse layers
    if args.layers is None or args.layers == 'all':
        layers = list(range(num_layers))
    else:
        layers = []
        for part in args.layers.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-', 1)
                layers.extend(range(int(a), int(b) + 1))
            else:
                layers.append(int(part))
        layers = sorted(set(layers))

    if not args.in_place and not args.output_dir:
        print("ERROR: --output-dir is required unless --in-place is used")
        sys.exit(1)
    output_dir = args.packed_dir if args.in_place else args.output_dir
    if not args.in_place:
        os.makedirs(output_dir, exist_ok=True)

    # Cluster each layer and compute adjacency improvement stats
    permutations = {}
    total_adjacency_before = 0
    total_adjacency_after = 0

    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in cooccur_data['layers']:
            print(f"  Layer {layer_idx}: no co-occurrence data, using identity ordering")
            permutations[layer_idx] = list(range(num_experts))
            continue

        matrix = cooccur_data['layers'][layer_key]
        order = greedy_cluster(matrix, num_experts)
        permutations[layer_idx] = order

        # Build inverse map: old_id -> new_position
        inverse = [0] * num_experts
        for new_pos, old_id in enumerate(order):
            inverse[old_id] = new_pos

        # Compute adjacency score: sum of co-occurrence between adjacent experts
        # Before: experts 0,1,2,... are adjacent (identity ordering)
        adj_before = sum(matrix[e][e+1] for e in range(num_experts - 1))
        # After: experts in new order
        adj_after = sum(matrix[order[i]][order[i+1]] for i in range(num_experts - 1))

        total_adjacency_before += adj_before
        total_adjacency_after += adj_after

        # Count how many of the K=8 routed experts land within a "cluster" of 8 adjacent positions
        # This is the key metric for read coalescing
        print(f"  Layer {layer_idx:2d}: adjacency score {adj_before} -> {adj_after} "
              f"({adj_after / max(adj_before, 1):.1f}x)")

    print(f"\nTotal adjacency: {total_adjacency_before} -> {total_adjacency_after} "
          f"({total_adjacency_after / max(total_adjacency_before, 1):.1f}x improvement)")

    if args.dry_run:
        print("\nDRY RUN: no files written")
        # Save permutation map anyway for inspection
        perm_path = os.path.join(output_dir, "permutation.json")
        with open(perm_path, 'w') as f:
            json.dump({"permutations": {str(k): v for k, v in permutations.items()}}, f)
        print(f"Saved permutation map to {perm_path}")
        return

    # Copy layout.json to output (skip if in-place)
    if not args.in_place:
        shutil.copy2(layout_path, os.path.join(output_dir, "layout.json"))

    # Repack each layer
    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        t0 = time.monotonic()
        order = permutations[layer_idx]
        bytes_written = repack_layer(layer_idx, order, args.packed_dir,
                                     output_dir, expert_size,
                                     in_place=args.in_place)
        total_written += bytes_written
        elapsed = time.monotonic() - t0
        throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else 0
        print(f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
              f"({throughput:.1f} GB/s)")

    # Save permutation map (needed for inference: expert_id -> physical_position)
    perm_path = os.path.join(output_dir, "permutation.json")
    with open(perm_path, 'w') as f:
        json.dump({
            "description": "permutation[layer][new_position] = old_expert_id",
            "inverse_description": "To find expert E, look at position inverse[layer][E]",
            "permutations": {str(k): v for k, v in permutations.items()},
            "inverse": {
                str(k): [0] * num_experts for k in permutations
            }
        }, f, indent=2)
    # Fill in inverse maps
    with open(perm_path) as f:
        perm_data = json.load(f)
    for layer_key, order in perm_data['permutations'].items():
        inv = [0] * num_experts
        for new_pos, old_id in enumerate(order):
            inv[old_id] = new_pos
        perm_data['inverse'][layer_key] = inv
    with open(perm_path, 'w') as f:
        json.dump(perm_data, f, indent=2)

    total_elapsed = time.monotonic() - t_start
    print(f"\nDONE: {total_written/1024**3:.1f} GB written in {total_elapsed:.0f}s")
    print(f"Permutation map: {perm_path}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
