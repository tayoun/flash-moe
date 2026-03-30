#!/usr/bin/env python3
"""
Convert Qwen3.5-MoE 4-bit experts to 2-bit.
Reads from packed_experts_ssd.bin (4-bit), converts to 2-bit format.
Writes to packed_experts_2bit_ssd.bin.

VERIFIED 4-bit layout (from empirical analysis of packed_experts_ssd.bin):
  gate_w: bytes 0-1,572,864    [1024, 384] uint32 (4-bit, 3072 vals/row)
  gate_s: bytes 1,572,864-1,671,168  [1024, 48] bf16 (scales)
  gate_b: bytes 1,671,168-1,769,472  [1024, 48] bf16 (biases, sparse)
  up_w:   bytes 1,769,472-3,342,336  [1024, 384] uint32 (4-bit)
  up_s:   bytes 3,342,336-3,440,640  [1024, 48] bf16 (scales)
  up_b:   bytes 3,440,640-3,538,944  [1024, 48] bf16 (biases, sparse)
  down_w: bytes 3,538,944-5,111,808  [3072, 128] uint32 (4-bit)
  down_s: bytes 5,111,808-5,406,720  [3072, 16] bf16 (scales)
  down_b: bytes 5,406,720-5,701,632  [3072, 16] bf16 (biases)
  TOTAL: 5,701,632 bytes/expert (but file only stores 5,308,416 — down_b is all zeros)

2-bit layout (kernel formula with vals_per_u32=16):
  gate_w:  0 - 786,432   (1024*192*4)
  gate_s:  786,432 - 884,736  (1024*48*2)
  gate_b:  884,736 - 983,040  (1024*48*2)
  up_w:    983,040 - 1,769,472 (1024*192*4)
  up_s:    1,769,472 - 1,867,776 (1024*48*2)
  up_b:    1,867,776 - 1,966,080 (1024*48*2)
  down_w:  1,966,080 - 2,752,512 (3072*64*4)
  down_s:  2,752,512 - 2,851,416 (3072*16*2)
  down_b:  2,851,416 - 2,950,320 (3072*16*2)
  TOTAL: 2,950,320 bytes/expert

Model: hidden_dim=3072, moe_intermediate=1024, group_size=64
"""

import struct, os, sys, argparse, time
import numpy as np
from pathlib import Path

NUM_LAYERS = 48
NUM_EXPERTS = 256

# Verified 4-bit file layout
EXPERT_SIZE_4BIT = 5_308_416
GATE_W_OFF  = 0;           GATE_W_SZ  = 1_572_864   # [1024, 384] uint32
GATE_S_OFF  = 1_572_864;   GATE_S_SZ  = 98_304       # [1024, 48] bf16
GATE_B_OFF  = 1_671_168;   GATE_B_SZ  = 98_304       # [1024, 48] bf16
UP_W_OFF    = 1_769_472;   UP_W_SZ    = 1_572_864    # [1024, 384] uint32
UP_S_OFF    = 3_342_336;   UP_S_SZ    = 98_304
UP_B_OFF    = 3_440_640;   UP_B_SZ    = 98_304
DOWN_W_OFF  = 3_538_944;   DOWN_W_SZ  = 1_572_864    # [3072, 128] uint32
DOWN_S_OFF  = 5_111_808;   DOWN_S_SZ  = 98_304       # [3072, 16] bf16
DOWN_B_OFF  = 5_406_720;   DOWN_B_SZ  = 98_304       # [3072, 16] bf16

# 2-bit offsets (kernel formula, vals_per_u32=16)
GATE_W_2  = 0;             GATE_S_2  = 786_432;     GATE_B_2  = 884_736
UP_W_2    = 983_040;       UP_S_2    = 1_769_472;    UP_B_2    = 1_867_776
DOWN_W_2  = 1_966_080;     DOWN_S_2  = 2_752_512;    DOWN_B_2  = 3_047_424
EXPERT_SIZE_2BIT = 3_342_336

print(f"4-bit expert size: {EXPERT_SIZE_4BIT:,} = {EXPERT_SIZE_4BIT/1e6:.3f}MB")
print(f"2-bit expert size: {EXPERT_SIZE_2BIT:,} = {EXPERT_SIZE_2BIT/1e6:.3f}MB")
print(f"Expected compression: {EXPERT_SIZE_4BIT/EXPERT_SIZE_2BIT:.2f}x")


def dequant_4bit_to_f32(weight, scales, biases, R, vals_per_group=64):
    """Dequantize 4-bit affine to float32.
    weight: (R, packed_cols) uint32 — each uint32 has 8 values at 4-bit
    For gate/up (hid=3072): packed_cols=384, 384/8=48 groups of 8 uint32s
    For down (hid=1024): packed_cols=128, 128/8=16 groups of 8 uint32s
    """
    num_groups = weight.shape[1]  # 384 for gate/up, 128 for down
    groups_per_u32 = 8

    w = weight.reshape(R, num_groups // groups_per_u32, groups_per_u32)
    # w: (R, 48, 8) or (R, 16, 8)

    # Extract 4-bit nibbles and reconstruct float values
    # nibbles per uint32: 8 values * 4 bits = 32 bits ✓
    nib = np.zeros((*w.shape[:2], 64), dtype=np.float32)
    for bit in range(8):
        nib += ((w >> (bit * 4)) & 0xF).astype(np.float32) * (2.0 ** (bit * 4))
    nib = nib.reshape(R, -1)  # (R, vals)

    # Broadcast scales/biases: (R, num_groups) → (R, num_groups, vals_per_group)
    n_g = nib.shape[1] // vals_per_group
    sc = scales[:, :n_g, None].astype(np.float32)
    bi = biases[:, :n_g, None].astype(np.float32)
    return (nib * sc + bi).reshape(R, -1)


def quant_f32_to_2bit(fvals, R, vals_per_group=64, num_groups=None):
    """Quantize float32 to 2-bit affine (merge=1: one scale/bias per group).
    fvals: (R, vals) float32 — vals must be divisible by vals_per_group
    Returns: (packed_u32, scales_f16, biases_f16)
    """
    if num_groups is None:
        num_groups = fvals.shape[1] // vals_per_group

    # Reshape: (R, num_groups, vals_per_group)
    fvals = fvals.reshape(R, num_groups, vals_per_group)

    # Per-group: min/max → scale/bias
    mn = fvals.min(axis=2)   # (R, num_groups)
    mx = fvals.max(axis=2)
    diff = np.clip(mx - mn, 1e-8, 65504.0)
    scales = (diff / 3.0).astype(np.float16)
    biases = np.clip(mn, -65504.0, 65504.0).astype(np.float16)

    # Quantize: q = round((v - mn) / scale * 3), clip to [0, 3]
    scale_vec = (mx - mn)[:, :, None]  # (R, num_groups, 1)
    scale_vec = np.where(scale_vec < 1e-8, 1.0, scale_vec)
    q = ((fvals - mn[:, :, None]) / scale_vec * 3.0).round()
    np.clip(q, 0, 3, out=q)

    # Pack 16 vals per uint32: (R, num_groups, 4, 4) → (R, num_groups, 4, 4)
    n_u32 = vals_per_group // 16
    q = q.reshape(R, num_groups, n_u32, 16)
    packed = (q[..., 0].astype(np.uint32) |
              (q[..., 1].astype(np.uint32) << 2) |
              (q[..., 2].astype(np.uint32) << 4) |
              (q[..., 3].astype(np.uint32) << 6))
    packed = packed.reshape(R, num_groups * n_u32)  # (R, num_groups * vals_per_group / 16)
    return packed, scales, biases


def convert_expert_2bit(expert_4bit):
    """Convert one expert from 4-bit to 2-bit. Returns 2,950,320 bytes."""
    R1, R2 = 1024, 3072
    vals_per_group = 64

    # Read 4-bit sections
    gw4 = np.frombuffer(expert_4bit[GATE_W_OFF:GATE_W_OFF + GATE_W_SZ], dtype=np.uint32).reshape(R1, 384)
    gs4 = np.frombuffer(expert_4bit[GATE_S_OFF:GATE_S_OFF + GATE_S_SZ], dtype=np.float16).reshape(R1, 48)
    gb4 = np.frombuffer(expert_4bit[GATE_B_OFF:GATE_B_OFF + GATE_B_SZ], dtype=np.float16).reshape(R1, 48)
    uw4 = np.frombuffer(expert_4bit[UP_W_OFF:UP_W_OFF + UP_W_SZ], dtype=np.uint32).reshape(R1, 384)
    us4 = np.frombuffer(expert_4bit[UP_S_OFF:UP_S_OFF + UP_S_SZ], dtype=np.float16).reshape(R1, 48)
    ub4 = np.frombuffer(expert_4bit[UP_B_OFF:UP_B_OFF + UP_B_SZ], dtype=np.float16).reshape(R1, 48)
    dw4 = np.frombuffer(expert_4bit[DOWN_W_OFF:DOWN_W_OFF + DOWN_W_SZ], dtype=np.uint32).reshape(R2, 128)
    ds4 = np.frombuffer(expert_4bit[DOWN_S_OFF:DOWN_S_OFF + DOWN_S_SZ], dtype=np.float16).reshape(R2, 16)
    # down_b not stored in 4-bit file (empirically all zeros) — zero-initialize
    db4 = np.zeros((R2, 16), dtype=np.float16)

    # Dequantize 4-bit → float32
    gw_f = dequant_4bit_to_f32(gw4, gs4, gb4, R1, vals_per_group)  # (1024, 3072)
    uw_f = dequant_4bit_to_f32(uw4, us4, ub4, R1, vals_per_group)  # (1024, 3072)
    dw_f = dequant_4bit_to_f32(dw4, ds4, db4, R2, vals_per_group)  # (3072, 1024)

    # Quantize float32 → 2-bit
    # gate/up: 3072 vals → 48 groups (64 vals/group) → 48*4 = 192 uint32s/row
    gw2, gs2, gb2 = quant_f32_to_2bit(gw_f, R1, vals_per_group, 48)  # (1024, 192), (1024, 48), (1024, 48)
    uw2, us2, ub2 = quant_f32_to_2bit(uw_f, R1, vals_per_group, 48)
    # down: 1024 vals → 16 groups (64 vals/group) → 16*4 = 64 uint32s/row
    dw2, ds2, db2 = quant_f32_to_2bit(dw_f, R2, vals_per_group, 16)  # (3072, 64), (3072, 16), (3072, 16)

    # Assemble 2-bit expert
    buf = bytearray(EXPERT_SIZE_2BIT)
    def put(arr, off):
        n = arr.nbytes
        buf[off:off+n] = arr.tobytes()
    put(gw2, GATE_W_2)
    put(gs2, GATE_S_2)
    put(gb2, GATE_B_2)
    put(uw2, UP_W_2)
    put(us2, UP_S_2)
    put(ub2, UP_B_2)
    put(dw2, DOWN_W_2)
    put(ds2, DOWN_S_2)
    put(db2, DOWN_B_2)
    return bytes(buf)


def convert_expert_2bit_fast(expert_4bit):
    """Convert one expert from 4-bit to 2-bit using per-group dequant/quant."""
    R1, R2 = 1024, 3072
    vals_per_group = 64
    gate_groups = 48   # 3072/64
    down_groups = 16   # 1024/64

    # Read sections
    gw4 = np.frombuffer(expert_4bit[GATE_W_OFF:GATE_W_OFF + GATE_W_SZ], dtype=np.uint32).reshape(R1, 384)
    gs4 = np.frombuffer(expert_4bit[GATE_S_OFF:GATE_S_OFF + GATE_S_SZ], dtype=np.float16).reshape(R1, 48)
    gb4 = np.frombuffer(expert_4bit[GATE_B_OFF:GATE_B_OFF + GATE_B_SZ], dtype=np.float16).reshape(R1, 48)
    uw4 = np.frombuffer(expert_4bit[UP_W_OFF:UP_W_OFF + UP_W_SZ], dtype=np.uint32).reshape(R1, 384)
    us4 = np.frombuffer(expert_4bit[UP_S_OFF:UP_S_OFF + UP_S_SZ], dtype=np.float16).reshape(R1, 48)
    ub4 = np.frombuffer(expert_4bit[UP_B_OFF:UP_B_OFF + UP_B_SZ], dtype=np.float16).reshape(R1, 48)
    dw4 = np.frombuffer(expert_4bit[DOWN_W_OFF:DOWN_W_OFF + DOWN_W_SZ], dtype=np.uint32).reshape(R2, 128)
    ds4 = np.frombuffer(expert_4bit[DOWN_S_OFF:DOWN_S_OFF + DOWN_S_SZ], dtype=np.float16).reshape(R2, 16)
    # down_b not stored in 4-bit file (empirically all zeros) — zero-initialize
    db4 = np.zeros((R2, 16), dtype=np.float16)

    def convert_proj(W4, S4, B4, R, packed_cols, n_g):
        """Convert one projection from 4-bit to 2-bit."""
        # W4: (R, packed_cols), S4: (R, n_g), B4: (R, n_g)
        # 4-bit: each uint32 has 8 nibbles, each nibble = 1 value at 4-bit
        # Each group has vals_per_group values = 8 uint32s (8 nibbles each = 64 values)
        # W4.reshape: (R, packed_cols) = (R, n_g * 8) → (R, n_g, 8)
        W = W4.reshape(R, packed_cols // 8, 8)  # (R, n_g, 8)

        # Build float32 reconstruction: each uint32 has 8 nibbles, each nibble = 1 value
        # vals_per_group = 64 values = 8 uint32s * 8 nibbles each
        W_f = np.zeros((R, n_g, vals_per_group), dtype=np.float32)
        for bit in range(8):
            nib = ((W >> (bit * 4)) & 0xF).astype(np.float32)  # (R, n_g, 8)
            # Repeat each nibble 8 times to fill the 64-value group
            W_f += np.repeat(nib, vals_per_group // 8, axis=2) * (2.0 ** (bit * 4))
        # W_f: (R, n_g, vals_per_group) = (R, n_g, 64)

        # Dequantize: W_deq = W_f * scales + biases (per group)
        S4_f = S4.astype(np.float32)  # (R, n_g)
        B4_f = B4.astype(np.float32)  # (R, n_g)
        W_deq = W_f * S4_f[:, :, None] + B4_f[:, :, None]  # (R, n_g, 64)

        # Quantize to 2-bit per group
        W_deq = W_deq.reshape(R, n_g, vals_per_group)
        mn = W_deq.min(axis=2)
        mx = W_deq.max(axis=2)
        # Compute scale in float32, clamp to prevent float16 overflow, then convert
        diff = np.clip(mx - mn, 1e-8, 65504.0)  # max float16
        sc = (diff / 3.0).astype(np.float16)
        bi = np.clip(mn, -65504.0, 65504.0).astype(np.float16)

        scale_vec = (mx - mn)[:, :, None]
        scale_vec = np.where(scale_vec < 1e-8, 1.0, scale_vec)
        q = ((W_deq - mn[:, :, None]) / scale_vec * 3.0).round()
        np.clip(q, 0, 3, out=q)

        # Pack 16 vals per uint32
        n_u32 = vals_per_group // 16
        q = q.reshape(R, n_g, n_u32, 16)
        packed = (q[..., 0].astype(np.uint32) |
                  (q[..., 1].astype(np.uint32) << 2) |
                  (q[..., 2].astype(np.uint32) << 4) |
                  (q[..., 3].astype(np.uint32) << 6))
        packed = packed.reshape(R, n_g * n_u32)
        return packed, sc, bi

    gw2, gs2, gb2 = convert_proj(gw4, gs4, gb4, R1, 384, gate_groups)
    uw2, us2, ub2 = convert_proj(uw4, us4, ub4, R1, 384, gate_groups)
    dw2, ds2, db2 = convert_proj(dw4, ds4, db4, R2, 128, down_groups)

    buf = bytearray(EXPERT_SIZE_2BIT)
    def put(arr, off):
        n = arr.nbytes
        buf[off:off+n] = arr.tobytes()
    put(gw2, GATE_W_2)
    put(gs2, GATE_S_2)
    put(gb2, GATE_B_2)
    put(uw2, UP_W_2)
    put(us2, UP_S_2)
    put(ub2, UP_B_2)
    put(dw2, DOWN_W_2)
    put(ds2, DOWN_S_2)
    put(db2, DOWN_B_2)
    return bytes(buf)


def test_single_expert():
    """Test conversion on expert 0 from the 4-bit file."""
    src = "/Volumes/Seagate Backup Plus Drive/packed_experts_ssd.bin"
    print(f"\n=== Testing on expert 0 from {src} ===")

    with open(src, 'rb') as f:
        header = f.read(16)
        nl, ne, es = struct.unpack('>IIQ', header)
        print(f"Header: layers={nl}, experts={ne}, expert_size={es:,}")

        f.seek(16)  # expert 0
        expert_4bit = f.read(EXPERT_SIZE_4BIT)
        print(f"Read expert 0: {len(expert_4bit):,} bytes")

    # Convert
    expert_2bit = convert_expert_2bit_fast(expert_4bit)
    print(f"2-bit expert: {len(expert_2bit):,} bytes")
    print(f"Expected: {EXPERT_SIZE_2BIT:,}")
    print(f"Match: {len(expert_2bit) == EXPERT_SIZE_2BIT}")

    # Verify offsets
    print("\n--- Offset verification ---")
    for name, off, sz in [
        ('gate_w', GATE_W_2, 786432), ('gate_s', GATE_S_2, 98304),
        ('gate_b', GATE_B_2, 98304), ('up_w', UP_W_2, 786432),
        ('up_s', UP_S_2, 98304), ('up_b', UP_B_2, 98304),
        ('down_w', DOWN_W_2, 786432), ('down_s', DOWN_S_2, 98304),
        ('down_b', DOWN_B_2, 98304),
    ]:
        data = expert_2bit[off:off+sz]
        print(f"  {name:8s}: off={off:>10,} sz={sz:>7,} null_pad={all(b == 0 for b in data[:4])}")

    # Check non-zero count for down_w (should be all non-zero since 4-bit down_w is dense)
    dw = np.frombuffer(expert_2bit[GATE_W_2:GATE_W_2+786432], dtype=np.uint32)
    print(f"\n  gate_w 2bit: {len(dw)} uint32s, non-zero: {(dw != 0).sum()}")
    dw2 = np.frombuffer(expert_2bit[DOWN_W_2:DOWN_W_2+786432], dtype=np.uint32)
    print(f"  down_w 2bit: {len(dw2)} uint32s, non-zero: {(dw2 != 0).sum()}")

    return expert_2bit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert 4-bit experts to 2-bit')
    parser.add_argument('--src', default='/Volumes/Seagate Backup Plus Drive/packed_experts_ssd.bin')
    parser.add_argument('--dst', default='/Volumes/Seagate Backup Plus Drive/packed_experts_2bit_ssd.bin')
    parser.add_argument('--test', action='store_true', help='Test on single expert')
    parser.add_argument('--layer', type=int, help='Convert specific layer (0-47)')
    parser.add_argument('--expert', type=int, help='Convert specific expert (0-255)')
    args = parser.parse_args()

    if args.test or args.layer is not None or args.expert is not None:
        test_single_expert()
    else:
        print(f"Conversion: {args.src} → {args.dst}")
        print(f"Requires ~{NUM_LAYERS * NUM_EXPERTS * EXPERT_SIZE_2BIT / 1e12:.2f}TB output space")
        print("Run with --test first to verify conversion on expert 0")
