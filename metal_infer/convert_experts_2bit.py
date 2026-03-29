#!/usr/bin/env python3
"""
convert_experts_2bit.py — Fast NumPy-vectorized 4-bit → 2-bit conversion for 122B MoE.
Processes all 256 experts per layer simultaneously. ~40min for all 48 layers.
"""

import struct, os, sys, argparse, time
import numpy as np
from pathlib import Path

NUM_LAYERS = 48
NUM_EXPERTS = 256
EXPERT_SIZE_4BIT = 5_308_416
EXPERT_SIZE_2BIT = 2_654_208

# 4-bit offsets
GATE_W, GATE_S, GATE_B = 0, 1_572_864, 1_671_168
UP_W,   UP_S,   UP_B   = 1_769_472, 3_342_336, 3_440_640
DOWN_W, DOWN_S, DOWN_B = 3_538_944, 5_111_808, 5_210_112


def quantize_proj(weight4, scale4, bias4, out_w, out_s, out_b, *,
                  R, C, groups_per_row):
    """
    Convert one projection type from 4-bit to 2-bit.
    weight4: (E, R, C) uint32 4-bit weights (C = groups_per_row * 8)
    scale4:  (E, R, groups_per_row) float16
    bias4:   (E, R, groups_per_row) float16
    out_w:   preallocated (E, R, C//2) uint32 output weights
    out_s:   preallocated (E, R, groups_per_row) float16 output scales
    out_b:   preallocated (E, R, groups_per_row) float16 output biases
    """
    E = NUM_EXPERTS
    vals_per_group = 64  # 8 uint32 × 8 vals each

    # Reshape weight to (E, R, groups_per_row, 8) — 8 vals per uint32
    w4 = weight4.reshape(E, R, groups_per_row, 8)

    # Extract 4-bit nibbles: (E, R, groups_per_row, 8, 8)
    nib = np.zeros((E, R, groups_per_row, 8, 8), dtype=np.uint8)
    for bit in range(8):
        nib[..., bit] = (w4 >> (bit * 4)) & 0xF

    # Scale/bias to (E, R, groups_per_row, 1, 1) for broadcasting
    sc = scale4[:, :, :, None, None].astype(np.float32)
    bi = bias4[:, :, :, None, None].astype(np.float32)

    # Dequantize: (E, R, groups_per_row, 8, 8) × scale + bias
    dequant = nib.astype(np.float32) * sc + bi

    # Requantize to 2-bit affine per group
    # Reduce over all 64 values per group → (E, R, groups_per_row)
    mn = dequant.reshape(E, R, groups_per_row, -1).min(axis=-1)
    mx = dequant.reshape(E, R, groups_per_row, -1).max(axis=-1)
    sc2 = (mx - mn) / 3.0
    bi2 = mn
    del mn, mx

    # Quantize: q = round((v - bias) / scale), clip to [0, 3]
    q = ((dequant - bi2[..., None, None]) / sc2[..., None, None]).round()
    np.clip(q, 0, 3, out=q)
    q = q.astype(np.uint8)
    del dequant

    # Reshape to (E, R, groups_per_row, 4, 16) — pack 16 vals per uint32
    q = q.reshape(E, R, groups_per_row, 4, 16)
    packed = (q[..., 0].astype(np.uint32) |
              (q[..., 1].astype(np.uint32) << 2) |
              (q[..., 2].astype(np.uint32) << 4) |
              (q[..., 3].astype(np.uint32) << 6))
    out_w[:] = packed.reshape(E, R, C // 2)
    out_s[:] = sc2.astype(np.float16)
    out_b[:] = bi2.astype(np.float16)


def convert_layer_fast(data: bytes) -> bytes:
    """Convert 256 experts from 4-bit to 2-bit. Returns ~680 MB bytes."""
    E = NUM_EXPERTS

    # ── Parse all experts ──────────────────────────────────────────────
    # gate_proj: [1024, 384] 4-bit → [1024, 192] 2-bit, 48 groups/row
    gate_w4 = np.frombuffer(data[GATE_W:GATE_W + E*1_572_864], dtype=np.uint32).reshape(E, 1024, 384)
    gate_s4 = np.frombuffer(data[GATE_S:GATE_S + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    gate_b4 = np.frombuffer(data[GATE_B:GATE_B + E*98_304], dtype=np.float16).reshape(E, 1024, 48)

    # up_proj: same structure as gate
    up_w4   = np.frombuffer(data[UP_W:UP_W   + E*1_572_864], dtype=np.uint32).reshape(E, 1024, 384)
    up_s4   = np.frombuffer(data[UP_S:UP_S   + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    up_b4   = np.frombuffer(data[UP_B:UP_B   + E*98_304], dtype=np.float16).reshape(E, 1024, 48)

    # down_proj: [3072, 128] 4-bit → [3072, 64] 2-bit, 16 groups/row
    # Stored as row-major [3072, 128]: 128 uint32 per row, 8 vals per uint32 = 1024 vals/row = 16 groups
    down_w4 = np.frombuffer(data[DOWN_W:DOWN_W + E*1_572_864], dtype=np.uint32).reshape(E, 3072, 128)
    down_s4 = np.frombuffer(data[DOWN_S:DOWN_S + E*98_304], dtype=np.float16).reshape(E, 3072, 16)
    down_b4 = np.frombuffer(data[DOWN_B:DOWN_B + E*98_304], dtype=np.float16).reshape(E, 3072, 16)

    # ── Pre-allocate output buffers ─────────────────────────────────
    gate_w2 = np.zeros((E, 1024, 192), dtype=np.uint32)
    gate_s2 = np.zeros((E, 1024, 48), dtype=np.float16)
    gate_b2 = np.zeros((E, 1024, 48), dtype=np.float16)
    up_w2   = np.zeros((E, 1024, 192), dtype=np.uint32)
    up_s2   = np.zeros((E, 1024, 48), dtype=np.float16)
    up_b2   = np.zeros((E, 1024, 48), dtype=np.float16)
    down_w2 = np.zeros((E, 3072, 64), dtype=np.uint32)
    down_s2 = np.zeros((E, 3072, 16), dtype=np.float16)
    down_b2 = np.zeros((E, 3072, 16), dtype=np.float16)

    # ── Convert each projection ─────────────────────────────────────
    quantize_proj(gate_w4, gate_s4, gate_b4, gate_w2, gate_s2, gate_b2,
                  R=1024, C=384, groups_per_row=48)
    quantize_proj(up_w4, up_s4, up_b4, up_w2, up_s2, up_b2,
                  R=1024, C=384, groups_per_row=48)
    quantize_proj(down_w4, down_s4, down_b4, down_w2, down_s2, down_b2,
                  R=3072, C=128, groups_per_row=16)

    # ── Assemble ─────────────────────────────────────────────────────
    parts = []
    for ei in range(E):
        parts.append(gate_w2[ei].tobytes());  parts.append(gate_s2[ei].tobytes());  parts.append(gate_b2[ei].tobytes())
        parts.append(up_w2[ei].tobytes());    parts.append(up_s2[ei].tobytes());    parts.append(up_b2[ei].tobytes())
        parts.append(down_w2[ei].tobytes());  parts.append(down_s2[ei].tobytes());  parts.append(down_b2[ei].tobytes())
    return b"".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/models/flash-moe/Qwen3.5-122B-A10B-4bit")
    parser.add_argument("--out", default="/Volumes/Seagate Backup Plus Drive/flash-moe-2bit/packed_experts_2bit_ssd.bin")
    parser.add_argument("--layer", type=int, default=None, help="Convert single layer (testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from last incomplete layer")
    args = parser.parse_args()

    packed_dir = Path(os.path.expanduser(args.model)) / "packed_experts"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = NUM_LAYERS * NUM_EXPERTS
    print(f"Converting {NUM_LAYERS}L × {NUM_EXPERTS}E = {total} experts")
    print(f"Output: {out_path} ({NUM_LAYERS * NUM_EXPERTS * EXPERT_SIZE_2BIT / 1024**3:.1f} GB)")

    import shutil
    free_space = shutil.disk_usage(out_path.parent).free
    required = NUM_LAYERS * NUM_EXPERTS * EXPERT_SIZE_2BIT
    print(f"Required: {required/1024**3:.1f} GB, Free: {free_space/1024**3:.1f} GB")
    if free_space < required:
        print("ERROR: Not enough space!"); sys.exit(1)

    # Determine starting layer and file mode BEFORE opening
    if args.layer is not None:
        layers = [args.layer]
        fout_open_mode = 'wb'
        if out_path.exists():
            out_path.unlink()
    elif args.resume and out_path.exists():
        file_size = out_path.stat().st_size
        header = 16
        layer_bytes = NUM_EXPERTS * EXPERT_SIZE_2BIT
        last_complete = (file_size - header) // layer_bytes
        start_layer = min(int(last_complete) + 1, NUM_LAYERS)
        print(f'Resuming from layer {start_layer} ({int(last_complete)} layers already complete)')
        layers = range(start_layer, NUM_LAYERS)
        fout_open_mode = 'ab'
    else:
        layers = range(NUM_LAYERS)
        fout_open_mode = 'wb'
        if out_path.exists():
            out_path.unlink()

    start = time.time()
    with open(out_path, fout_open_mode) as fout:
        if fout_open_mode == 'wb':
            fout.write(struct.pack(">IIQ", NUM_LAYERS, NUM_EXPERTS, EXPERT_SIZE_2BIT))
        for li in layers:
            t0 = time.time()
            lf = packed_dir / f"layer_{li:02d}.bin"
            data = open(lf, "rb").read()
            layer_2bit = convert_layer_fast(data)
            fout.write(layer_2bit)
            t1 = time.time()
            done = (li + 1) * NUM_EXPERTS
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done)
            print(f"  L{li:02d}: {done}/{total} ({100*done/total:.0f}%) "
                  f"layer={t1-t0:.1f}s ETA={eta/60:.0f}min")

    elapsed = time.time() - start
    size = out_path.stat().st_size
    print(f"Done in {elapsed/60:.0f}min — {size/1024**3:.1f} GB ({size/elapsed/1024**2:.0f} MB/s)")


if __name__ == "__main__":
    main()
