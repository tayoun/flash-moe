#!/usr/bin/env python3
"""Repack routed expert weights into contiguous per-layer binary files.

Supports legacy Qwen indexes and architecture-specific layout metadata (e.g. Gemma).
"""

import argparse
import json
import os
import time
import sys

# Legacy Qwen 4-bit component order and expected sizes.
LEGACY_QWEN_COMPONENTS = [
    {"name": "gate_proj.weight", "offset": 0, "size": 2097152, "dtype": "U32", "shape": [1024, 512]},
    {"name": "gate_proj.scales", "offset": 2097152, "size": 131072, "dtype": "BF16", "shape": [1024, 64]},
    {"name": "gate_proj.biases", "offset": 2228224, "size": 131072, "dtype": "BF16", "shape": [1024, 64]},
    {"name": "up_proj.weight", "offset": 2359296, "size": 2097152, "dtype": "U32", "shape": [1024, 512]},
    {"name": "up_proj.scales", "offset": 4456448, "size": 131072, "dtype": "BF16", "shape": [1024, 64]},
    {"name": "up_proj.biases", "offset": 4587520, "size": 131072, "dtype": "BF16", "shape": [1024, 64]},
    {"name": "down_proj.weight", "offset": 4718592, "size": 2097152, "dtype": "U32", "shape": [4096, 128]},
    {"name": "down_proj.scales", "offset": 6815744, "size": 131072, "dtype": "BF16", "shape": [4096, 16]},
    {"name": "down_proj.biases", "offset": 6946816, "size": 131072, "dtype": "BF16", "shape": [4096, 16]},
]
LEGACY_QWEN_EXPERT_SIZE = 7077888
LEGACY_QWEN_NUM_EXPERTS = 512
LEGACY_QWEN_NUM_LAYERS = 60

# Gemma 4 26B BF16 expert layout
# Each expert = fused gate_up_proj ([2*704, 2816] BF16) + down_proj ([2816, 704] BF16)
# expert_size = 2 * 704 * 2816 * 2 + 2816 * 704 * 2 = 7,929,856 + 3,964,928 = 11,894,784 bytes
GEMMA4_COMPONENTS = [
    {"name": "gate_up_proj", "offset": 0, "size": 7929856, "dtype": "BF16", "shape": [128, 1408, 2816]},
    {"name": "down_proj",    "offset": 7929856, "size": 3964928, "dtype": "BF16", "shape": [128, 2816, 704]},
]
GEMMA4_EXPERT_SIZE = 11894784  # 7,929,856 + 3,964,928 bytes
GEMMA4_NUM_EXPERTS = 128
GEMMA4_NUM_LAYERS = 30


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
    for l in layers:
        if l < 0 or l >= num_layers:
            raise ValueError(f"Layer index out of range: {l} (valid 0..{num_layers - 1})")
    return sorted(set(layers))


def load_index(index_path):
    with open(index_path) as f:
        payload = json.load(f)
    if "expert_reads" not in payload or "model_path" not in payload:
        raise ValueError("index missing required keys: expert_reads/model_path")
    return payload


def normalize_layout(index_payload, layout_mode):
    """Return normalized layout dict with components + shape constants."""
    if layout_mode == "qwen_legacy":
        return {
            "format_version": 1,
            "architecture": "qwen3_moe",
            "num_experts": LEGACY_QWEN_NUM_EXPERTS,
            "num_layers": LEGACY_QWEN_NUM_LAYERS,
            "expert_size": LEGACY_QWEN_EXPERT_SIZE,
            "components": LEGACY_QWEN_COMPONENTS,
        }

    if layout_mode == "gemma4":
        return {
            "format_version": 1,
            "architecture": "gemma4_text_moe_bf16",
            "num_experts": GEMMA4_NUM_EXPERTS,
            "num_layers": GEMMA4_NUM_LAYERS,
            "expert_size": GEMMA4_EXPERT_SIZE,
            "components": GEMMA4_COMPONENTS,
        }

    layout = index_payload.get("layout")
    if not layout:
        # Backward-compatible auto path for old Qwen expert_index.json files.
        return {
            "format_version": 1,
            "architecture": "qwen3_moe",
            "num_experts": LEGACY_QWEN_NUM_EXPERTS,
            "num_layers": LEGACY_QWEN_NUM_LAYERS,
            "expert_size": LEGACY_QWEN_EXPERT_SIZE,
            "components": LEGACY_QWEN_COMPONENTS,
        }

    components = []
    cursor = 0
    for comp in layout.get("components", []):
        name = comp["name"]
        size = int(comp["size"])
        offset = int(comp.get("offset", cursor))
        dtype = comp.get("dtype", "UNKNOWN")
        cursor = max(cursor, offset + size)
        components.append({
            "name": name,
            "offset": offset,
            "size": size,
            "dtype": dtype,
            "shape": comp.get("shape"),
        })

    if not components:
        raise ValueError("layout.components must be non-empty")

    expert_size = int(layout.get("expert_size", max(c["offset"] + c["size"] for c in components)))

    return {
        "format_version": int(layout.get("format_version", 1)),
        "architecture": layout.get("architecture", "unknown"),
        "num_experts": int(layout.get("num_experts", index_payload.get("num_experts", LEGACY_QWEN_NUM_EXPERTS))),
        "num_layers": int(layout.get("num_layers", index_payload.get("num_layers", len(index_payload["expert_reads"])))),
        "expert_size": expert_size,
        "components": sorted(components, key=lambda c: c["offset"]),
    }


def validate_metadata(index_payload, layout, strict_layer_count=True):
    """Metadata-only validation against layout and index consistency."""
    expert_reads = index_payload["expert_reads"]
    errors = []
    expected_comp = {c["name"]: c for c in layout["components"]}
    expected_layers = layout["num_layers"]
    expected_experts = layout["num_experts"]

    if strict_layer_count and len(expert_reads) != expected_layers:
        errors.append(f"layer count mismatch: index has {len(expert_reads)} layers, expected {expected_layers}")

    for layer_idx in range(expected_layers):
        layer_key = str(layer_idx)
        layer_info = expert_reads.get(layer_key)
        if layer_info is None:
            errors.append(f"missing layer {layer_idx} in expert_reads")
            continue

        for comp_name, comp in expected_comp.items():
            info = layer_info.get(comp_name)
            if info is None:
                errors.append(f"layer {layer_idx}: missing component {comp_name}")
                continue

            got_size = int(info.get("expert_size", -1))
            if got_size != int(comp["size"]):
                errors.append(
                    f"layer {layer_idx} {comp_name}: expert_size={got_size}, expected={comp['size']}"
                )

            stride = int(info.get("expert_stride", got_size))
            if stride < got_size:
                errors.append(f"layer {layer_idx} {comp_name}: expert_stride={stride} < expert_size={got_size}")

            if "total_size" in info:
                total_size = int(info["total_size"])
                min_total = stride * expected_experts
                if total_size < min_total:
                    errors.append(
                        f"layer {layer_idx} {comp_name}: total_size={total_size} < stride*num_experts={min_total}"
                    )

            if "shape" in info and isinstance(info["shape"], list) and info["shape"]:
                if int(info["shape"][0]) != expected_experts:
                    errors.append(
                        f"layer {layer_idx} {comp_name}: shape[0]={info['shape'][0]} expected {expected_experts}"
                    )

            if "dtype" in info and comp.get("dtype") and info["dtype"] != comp["dtype"]:
                errors.append(
                    f"layer {layer_idx} {comp_name}: dtype={info['dtype']} expected {comp['dtype']}"
                )

    return errors


def open_source_files(expert_reads, model_path, layers, components):
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            continue
        for comp in components:
            name = comp["name"]
            info = expert_reads[layer_key].get(name)
            if info:
                needed_files.add(info['file'])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source safetensors files")
    return fds


def repack_layer(layer_idx, expert_reads, fds, output_dir, layout, dry_run=False):
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return 0, 0.0

    components = layout["components"]
    num_experts = layout["num_experts"]
    expert_size = layout["expert_size"]
    layer_size = num_experts * expert_size
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if dry_run:
        for expert_idx in range(num_experts):
            for comp in components:
                info = layer_info[comp['name']]
                _ = info['abs_offset'] + expert_idx * info['expert_stride']
                _ = expert_idx * expert_size + comp['offset']
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {layer_size:,} bytes to {out_path}")
        return layer_size, 0.0

    t0 = time.monotonic()
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)

    bytes_written = 0
    read_plan = []
    for expert_idx in range(num_experts):
        for comp in components:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']
            read_plan.append((src_fd, src_offset, dst_offset, comp['size']))

    read_plan.sort(key=lambda x: (x[0], x[1]))

    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} at offset {src_offset}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    elapsed = time.monotonic() - t0
    return bytes_written, elapsed


def verify_layer(layer_idx, expert_reads, fds, output_dir, layout):
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: missing in index")
        return False

    components = layout["components"]
    num_experts = layout["num_experts"]
    expert_size = layout["expert_size"]
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    spot = sorted(set([0, 1, max(0, num_experts // 2), max(0, num_experts - 1)]))

    fd_packed = os.open(out_path, os.O_RDONLY)
    mismatches = 0

    for expert_idx in spot:
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
        print(f"  Layer {layer_idx}: verification PASSED (experts {', '.join(str(x) for x in spot)})")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")

    return mismatches == 0


def write_layout(output_dir, layout):
    payload = {
        "format_version": layout["format_version"],
        "architecture": layout["architecture"],
        "expert_size": layout["expert_size"],
        "num_layers": layout["num_layers"],
        "num_experts": layout["num_experts"],
        "components": layout["components"],
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Repack expert weights into contiguous per-layer binary files")
    parser.add_argument('--index', default='expert_index.json', help='Path to expert index JSON')
    parser.add_argument('--layout', default='auto', choices=['auto', 'qwen_legacy', 'gemma4'],
                        help='Layout source: auto (index metadata or legacy fallback), qwen_legacy')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--dry-run', action='store_true', help='Verify offsets without writing')
    parser.add_argument('--output', default=None,
                        help='Output directory for packed experts (default: <model_path>/packed_experts)')
    parser.add_argument('--verify-only', type=int, default=None, metavar='LAYER',
                        help='Verify a specific packed layer against originals')
    parser.add_argument('--metadata-only', action='store_true',
                        help='Run metadata validation only, then exit')
    parser.add_argument('--skip-metadata-validation', action='store_true',
                        help='Skip metadata validation before writing')
    parser.add_argument('--allow-layer-count-mismatch', action='store_true',
                        help='Allow index layer count to differ from layout expectation')
    args = parser.parse_args()

    print("Loading expert index...")
    index_payload = load_index(args.index)
    expert_reads = index_payload['expert_reads']
    model_path = index_payload['model_path']
    layout = normalize_layout(index_payload, args.layout)

    print(f"Model path: {model_path}")
    print(f"Architecture: {layout['architecture']}")
    print(f"Layers in index: {len(expert_reads)}")
    print(f"Layout layers: {layout['num_layers']}, experts/layer: {layout['num_experts']}, expert size: {layout['expert_size']:,}")

    if not args.skip_metadata_validation:
        errors = validate_metadata(
            index_payload,
            layout,
            strict_layer_count=not args.allow_layer_count_mismatch,
        )
        if errors:
            print("Metadata validation FAILED:")
            for err in errors[:50]:
                print(f"  - {err}")
            if len(errors) > 50:
                print(f"  ... and {len(errors) - 50} more")
            sys.exit(1)
        print("Metadata validation passed")

    if args.metadata_only:
        print("Metadata-only check complete")
        return

    output_dir = args.output if args.output else os.path.join(model_path, "packed_experts")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.verify_only is not None:
        layers = [args.verify_only]
    else:
        layers = parse_layers(args.layers, layout["num_layers"])

    print(f"Layers to process: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    layer_size = layout["num_experts"] * layout["expert_size"]

    if not args.dry_run and args.verify_only is None:
        total_bytes = len(layers) * layer_size
        print(f"Total data to write: {total_bytes / (1024**3):.1f} GB")

        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = total_bytes / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < total_bytes:
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB but only {free_gb:.1f} GB free.")
            sys.exit(1)

    fds = {}
    if not args.dry_run:
        fds = open_source_files(expert_reads, model_path, layers, layout["components"])

    try:
        if args.verify_only is not None:
            if not verify_layer(args.verify_only, expert_reads, fds, output_dir, layout):
                sys.exit(1)
            return

        write_layout(output_dir, layout)

        t_start = time.monotonic()
        total_written = 0

        for i, layer_idx in enumerate(layers):
            bytes_written, elapsed = repack_layer(
                layer_idx, expert_reads, fds, output_dir, layout, dry_run=args.dry_run
            )
            total_written += bytes_written

            if not args.dry_run and bytes_written > 0:
                throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float('inf')
                overall_elapsed = time.monotonic() - t_start
                overall_throughput = total_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
                eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
                print(
                    f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                    f"({throughput:.1f} GB/s) | Total: {total_written/1024**3:.1f}/"
                    f"{len(layers)*layer_size/1024**3:.1f} GB ({overall_throughput:.1f} GB/s avg) | ETA: {eta:.0f}s"
                )

                if not verify_layer(layer_idx, expert_reads, fds, output_dir, layout):
                    print(f"ABORTING: verification failed for layer {layer_idx}")
                    sys.exit(1)

        total_elapsed = time.monotonic() - t_start
        if not args.dry_run and total_written > 0:
            print(f"\n{'='*60}")
            print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
            print(f"Time: {total_elapsed:.1f}s")
            print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
            print(f"Output: {output_dir}")
        elif args.dry_run:
            print(f"\nDRY RUN complete: {len(layers)} layers validated")
    finally:
        for fd in fds.values():
            os.close(fd)


if __name__ == '__main__':
    main()
