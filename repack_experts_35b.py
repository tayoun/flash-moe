#!/usr/bin/env python3
"""Repack expert weights from scattered safetensors into contiguous per-layer binary files.

Creates one binary file per layer: packed_experts/layer_XX.bin
Each file = NUM_EXPERTS experts x EXPERT_SIZE bytes
Expert E starts at byte offset E * EXPERT_SIZE

Within each expert block, 9 components packed in fixed order:
  gate_proj.weight, gate_proj.scales, gate_proj.biases,
  up_proj.weight,   up_proj.scales,   up_proj.biases,
  down_proj.weight,  down_proj.scales,  down_proj.biases

All sizes are derived from the expert index (no hardcoded model dimensions).

Usage:
    python repack_experts_35b.py --index expert_index.json          # repack all layers
    python repack_experts_35b.py --index expert_index.json --layers 0-4
    python repack_experts_35b.py --index expert_index.json --dry-run
    python repack_experts_35b.py --index expert_index.json --delete-consumed-shards
"""

import argparse
import json
import os
import time
import sys

# Component names in packing order
COMPONENT_NAMES = [
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
    "down_proj.weight",  "down_proj.scales",  "down_proj.biases",
]


def derive_layout(expert_reads):
    """Derive COMPONENTS, EXPERT_SIZE, NUM_EXPERTS, NUM_LAYERS from the expert index."""
    num_layers = len(expert_reads)
    # Use first layer to derive sizes
    sample_layer = sorted(expert_reads.keys(), key=int)[0]
    layer_info = expert_reads[sample_layer]

    # Derive num_experts from first component's shape (dim 0)
    first_comp = layer_info[COMPONENT_NAMES[0]]
    num_experts = first_comp['shape'][0]

    components = []
    offset = 0
    for name in COMPONENT_NAMES:
        info = layer_info[name]
        size = info['expert_size']
        components.append({
            "name": name,
            "offset": offset,
            "size": size,
            "dtype": info['dtype'],
            "shape": info['shape'][1:],  # strip expert dim
        })
        offset += size

    expert_size = offset
    layer_size = num_experts * expert_size

    return components, expert_size, num_experts, num_layers, layer_size


def parse_layers(spec, num_layers):
    """Parse layer specification like '0-4' or '0,5,10' or 'all'."""
    if spec is None or spec == 'all':
        return list(range(num_layers))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_index(index_path):
    """Load expert index JSON and return expert_reads dict + model_path."""
    with open(index_path) as f:
        idx = json.load(f)
    return idx['expert_reads'], idx['model_path']


def verify_component_sizes(expert_reads, components):
    """Verify that all layers have consistent component sizes."""
    expected = {c['name']: c['size'] for c in components}
    for layer_key, comps in expert_reads.items():
        for comp_name, info in comps.items():
            if comp_name not in expected:
                print(f"WARNING: unknown component {comp_name} in layer {layer_key}")
                continue
            if info['expert_size'] != expected[comp_name]:
                print(f"MISMATCH: layer {layer_key}, {comp_name}: "
                      f"index says {info['expert_size']}, expected {expected[comp_name]}")
                return False
    print("Component sizes verified: all layers consistent")
    return True


def open_source_files(expert_reads, model_path, layers):
    """Open all needed safetensors files, return {filename: fd}."""
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"WARNING: layer {layer_idx} not found in expert_reads")
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info['file'])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source safetensors files")
    return fds


def get_layers_per_shard(expert_reads):
    """Build mapping: shard_filename -> set of layer indices that use it."""
    shard_layers = {}
    for layer_key, comps in expert_reads.items():
        layer_idx = int(layer_key)
        for info in comps.values():
            shard_layers.setdefault(info['file'], set()).add(layer_idx)
    return shard_layers


def repack_layer_read(layer_idx, expert_reads, fds, components,
                      expert_size, num_experts):
    """Read all expert data for one layer into memory.

    Returns list of (dst_offset, data) tuples, or None if layer not found.
    """
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        return None

    layer_info = expert_reads[layer_key]
    missing = [c['name'] for c in components if c['name'] not in layer_info]
    if missing:
        raise KeyError(f"Layer {layer_idx} is missing components in expert index: {missing}")

    read_plan = []
    for expert_idx in range(num_experts):
        for comp in components:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']
            read_plan.append((src_fd, src_offset, dst_offset, comp['size']))

    read_plan.sort(key=lambda x: (x[0], x[1]))

    write_plan = []
    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} "
                          f"at offset {src_offset}")
        write_plan.append((dst_offset, data))

    return write_plan


def repack_layer_write(layer_idx, write_plan, output_dir, layer_size):
    """Write buffered expert data to a contiguous binary file.

    Returns bytes_written.
    """
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)

    bytes_written = 0
    for dst_offset, data in write_plan:
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += len(data)

    os.close(fd_out)
    return bytes_written


def repack_layer_dry_run(layer_idx, expert_reads, components,
                         expert_size, num_experts, layer_size, output_dir):
    """Validate offsets for one layer without writing."""
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return layer_size

    layer_info = expert_reads[layer_key]
    missing = [c['name'] for c in components if c['name'] not in layer_info]
    if missing:
        raise KeyError(f"Layer {layer_idx} is missing components in expert index: {missing}")

    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
    for expert_idx in range(num_experts):
        for comp in components:
            info = layer_info[comp['name']]
            if info['expert_size'] != comp['size']:
                raise ValueError(
                    f"Layer {layer_idx}, {comp['name']}: index expert_size {info['expert_size']} != expected {comp['size']}"
                )
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']
            if src_offset < 0 or dst_offset < 0:
                raise ValueError(
                    f"Negative offset detected for layer {layer_idx}, expert {expert_idx}, component {comp['name']}"
                )
    print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {layer_size:,} bytes to {out_path}")
    return layer_size


def verify_layer(layer_idx, expert_reads, fds, output_dir, components,
                 expert_size, num_experts):
    """Read back several experts from packed file and compare to originals.

    If source shards have been deleted, skips verification (returns True).
    """
    layer_key = str(layer_idx)
    layer_info = expert_reads[layer_key]
    missing = [c['name'] for c in components if c['name'] not in layer_info]
    if missing:
        print(f"  Layer {layer_idx}: verification skipped, missing components: {missing}")
        return False

    # Check if all source shards are still available
    needed_files = set(info['file'] for info in layer_info.values())
    missing_shards = needed_files - set(fds.keys())
    if missing_shards:
        print(f"  Layer {layer_idx}: verification skipped (source shards deleted: {missing_shards})")
        return True

    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)

    spot_checks = [0, 1, num_experts // 2, num_experts - 1]
    mismatches = 0
    for expert_idx in spot_checks:
        for comp in components:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']

            original = os.pread(src_fd, comp['size'], src_offset)
            packed = os.pread(fd_packed, comp['size'], dst_offset)

            if original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)

    if mismatches == 0:
        print(f"  Layer {layer_idx}: verification PASSED (experts {spot_checks})")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")

    return mismatches == 0


def write_layout(output_dir, expert_size, num_layers, num_experts, components):
    """Write layout.json describing the packed format."""
    layout = {
        "expert_size": expert_size,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "components": components,
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Repack expert weights into contiguous per-layer binary files")
    parser.add_argument('--index', required=True,
                        help='Path to expert index JSON (e.g., expert_index_35b.json)')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify offsets without writing')
    parser.add_argument('--verify-only', type=int, default=None, metavar='LAYER',
                        help='Verify a specific layer against originals')
    parser.add_argument('--model-path', default=None,
                        help='Override model_path from the expert index (e.g., for 122B)')
    parser.add_argument('--output-dir', default=None,
                        help='Override output directory (default: <model_path>/packed_experts)')
    parser.add_argument('--delete-consumed-shards', action='store_true',
                        help='Delete safetensors shards after all their layers are repacked (saves disk)')
    args = parser.parse_args()

    print("Loading expert index...")
    expert_reads, model_path = load_index(args.index)
    if args.model_path:
        model_path = args.model_path
    print(f"Model path: {model_path}")
    print(f"Layers in index: {len(expert_reads)}")

    # Derive layout from index
    components, expert_size, num_experts, num_layers, layer_size = derive_layout(expert_reads)

    print(f"Expert size: {expert_size:,} bytes ({expert_size / 1e6:.2f} MB)")
    print(f"Experts per layer: {num_experts}")
    print(f"Layer file size: {layer_size:,} bytes ({layer_size / 1e9:.2f} GB)")

    # Verify all layers are consistent
    if not verify_component_sizes(expert_reads, components):
        print("ABORTING: component size mismatch")
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else os.path.join(model_path, "packed_experts")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine which layers to process
    if args.verify_only is not None:
        layers = [args.verify_only]
    else:
        layers = parse_layers(args.layers, num_layers)

    if not layers:
        print("No layers selected")
        sys.exit(1)

    print(f"Layers to process: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    # Shard deletion tracking
    shard_layers = get_layers_per_shard(expert_reads) if args.delete_consumed_shards else {}
    processed_layers = set()
    deleted_shards = set()

    if not args.dry_run and args.verify_only is None:
        total_bytes = len(layers) * layer_size
        print(f"Total data to write: {total_bytes / (1024**3):.1f} GB")

        # Check free disk space
        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = total_bytes / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < total_bytes and not args.delete_consumed_shards:
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB but only {free_gb:.1f} GB free.")
            approx_layer_gb = layer_size / (1024**3)
            max_layers_fit = max(0, int(free_gb / approx_layer_gb) - 1)
            print(f"Hint: use --layers to process a subset, e.g. --layers 0-{max_layers_fit}")
            print(f"Hint: or use --delete-consumed-shards to free space incrementally")
            sys.exit(1)
        elif free_bytes < total_bytes:
            print(f"NOTE: Not enough space for all layers at once, but --delete-consumed-shards "
                  f"will free space incrementally")

    # Open source files
    fds = open_source_files(expert_reads, model_path, layers)

    if args.verify_only is not None:
        verify_layer(args.verify_only, expert_reads, fds, output_dir,
                     components, expert_size, num_experts)
        for fd in fds.values():
            os.close(fd)
        return

    # Write layout.json
    write_layout(output_dir, expert_size, num_layers, num_experts, components)

    # Repack each layer
    t_start = time.monotonic()
    total_written = 0

    def try_delete_consumed_shards():
        """Delete any shards whose layers have all been read."""
        if not args.delete_consumed_shards:
            return
        for shard_name, shard_layer_set in shard_layers.items():
            if shard_name in deleted_shards:
                continue
            if shard_layer_set.issubset(read_layers):
                if shard_name in fds:
                    os.close(fds[shard_name])
                    del fds[shard_name]
                shard_path = os.path.join(model_path, shard_name)
                shard_size = os.path.getsize(shard_path)
                os.unlink(shard_path)
                deleted_shards.add(shard_name)
                print(f"  DELETED consumed shard: {shard_name} "
                      f"(freed {shard_size / 1e9:.2f} GB)")

    read_layers = set()  # layers whose data has been read into memory

    for i, layer_idx in enumerate(layers):
        if args.dry_run:
            bytes_written = repack_layer_dry_run(
                layer_idx, expert_reads, components,
                expert_size, num_experts, layer_size, output_dir
            )
            total_written += bytes_written
            processed_layers.add(layer_idx)
            continue

        t0 = time.monotonic()

        # Phase 1: read all data into memory
        write_plan = repack_layer_read(
            layer_idx, expert_reads, fds, components,
            expert_size, num_experts
        )
        if write_plan is None:
            print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
            continue

        read_layers.add(layer_idx)

        # Phase 2: delete shards that are fully read (frees disk BEFORE writing)
        try_delete_consumed_shards()

        # Phase 3: write to disk
        bytes_written = repack_layer_write(layer_idx, write_plan, output_dir, layer_size)
        del write_plan

        total_written += bytes_written
        processed_layers.add(layer_idx)

        elapsed = time.monotonic() - t0
        if bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float('inf')
            overall_elapsed = time.monotonic() - t_start
            overall_throughput = total_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
            eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
            print(f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                  f"({throughput:.1f} GB/s) | "
                  f"Total: {total_written/1024**3:.1f}/{len(layers)*layer_size/1024**3:.1f} GB "
                  f"({overall_throughput:.1f} GB/s avg) | "
                  f"ETA: {eta:.0f}s")

            # Verify this layer immediately
            if not verify_layer(layer_idx, expert_reads, fds, output_dir,
                                components, expert_size, num_experts):
                print(f"ABORTING: verification failed for layer {layer_idx}")
                sys.exit(1)

    # Close remaining source files
    for fd in fds.values():
        os.close(fd)

    # Final summary
    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\n{'='*40}")
        print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
        print(f"Time: {total_elapsed:.1f}s")
        print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
        print(f"Output: {output_dir}")
        if deleted_shards:
            print(f"Deleted {len(deleted_shards)} consumed shards")
    elif args.dry_run:
        print(f"\nDRY RUN complete: {len(layers)} layers validated")


if __name__ == '__main__':
    main()
