#!/usr/bin/env python3
"""
convert_experts_2bit_fixed.py — Convert 4-bit experts to 2-bit for Qwen3.5-MoE.
Reads from packed_experts_ssd.bin (4-bit), converts to 2-bit expert format.
Writes to packed_experts_2bit_ssd.bin.

Uses the SAME layout as infer.m compute_expert_offsets():
  gate/up: [1024, 3072] float32 → 4-bit: [1024, 384] uint32 → 2-bit: [1024, 192] uint32
  down:   [3072, 1024] float32 → 4-bit: [3072, 128] uint32 → 2-bit: [3072, 64] uint32
  2-bit expert size: 2,949,120 bytes

Weight dims from model config:
  hidden_dim = 3072, moe_intermediate = 1024, group_size = 64

4-bit file format (per expert, 5,308,416 bytes):
  gate_w [1024, 3072] in 4-bit:   [1024, 384] uint32 (384 = 3072/8)
  gate_s [1024, 48] bf16:         [1024, 48] bf16 (48 = 3072/64)
  gate_b [1024, 48] bf16
  up_w   [1024, 3072] in 4-bit:   [1024, 384] uint32
  up_s   [1024, 48] bf16
  up_b   [1024, 48] bf16
  down_w [3072, 1024] in 4-bit:   [3072, 128] uint32 (128 = 1024/8)
  down_s [3072, 16] bf16:         [3072, 16] bf16 (16 = 1024/64)
  down_b [3072, 16] bf16
"""

import struct, os, sys, argparse, time
import numpy as np
from pathlib import Path

NUM_LAYERS = 48
NUM_EXPERTS = 256

# 4-bit file expert offsets (verified against actual 4-bit SSD file size)
EXPERT_SIZE_4BIT = 5_308_416
GATE_W_SZ  = 1024 * 384 * 4      # 1,572,864
GATE_S_SZ  = 1024 * 48 * 2       # 98,304
UP_W_SZ    = 1024 * 384 * 4      # 1,769,472 (same packing ratio)
DOWN_W_SZ  = 3072 * 128 * 4     # 4,718,592

# 2-bit expert offsets (matching infer.m formula with vals_per_u32=16)
EXPERT_SIZE_2BIT = 2_949_120
GATE_W_2  = 0;               GATE_S_2  = 786_432;   GATE_B_2  = 884_736
UP_W_2    = 983_040;         UP_S_2    = 1_769_472; UP_B_2    = 1_867_776
DOWN_W_2  = 2_064_000;       DOWN_S_2  = 2_850_432; DOWN_B_2  = 2_948_736


def convert_expert_2bit(expert_4bit):
    """Convert one expert from 4-bit to 2-bit. Returns 2,949,120 bytes."""
    E = 1
    R1 = 1024; I1 = 3072  # gate/up: [1024, 3072]
    R2 = 3072; I2 = 1024  # down: [3072, 1024]
    gs = 64  # group_size
    
    # --- Gate ---
    # 4-bit gate: [1024, 384] uint32 stored → [1024, 3072] float32 values
    gw4 = np.frombuffer(expert_4bit[GATE_W_SZ:GATE_W_SZ + GATE_W_SZ], dtype=np.uint32).reshape(R1, 384)
    gs4 = np.frombuffer(expert_4bit[GATE_S_SZ:GATE_S_SZ + GATE_S_SZ], dtype=np.float16).reshape(R1, 48)
    gb4 = np.frombuffer(expert_4bit[GATE_S_SZ * 2:GATE_S_SZ * 2 + GATE_S_SZ], dtype=np.float16).reshape(R1, 48)
    
    # --- Up ---
    up_off = GATE_W_SZ * 2 + GATE_S_SZ * 2  # 3,440,640
    uw4 = np.frombuffer(expert_4bit[up_off:up_off + UP_W_SZ], dtype=np.uint32).reshape(R1, 384)
    us4 = np.frombuffer(expert_4bit[up_off + UP_W_SZ:up_off + UP_W_SZ + GATE_S_SZ], dtype=np.float16).reshape(R1, 48)
    ub4 = np.frombuffer(expert_4bit[up_off + UP_W_SZ + GATE_S_SZ:up_off + UP_W_SZ + GATE_S_SZ * 2], dtype=np.float16).reshape(R1, 48)
    
    # --- Down ---
    down_off = GATE_W_SZ * 2 + GATE_S_SZ * 2 + UP_W_SZ + GATE_S_SZ * 2  # 5,210,112
    dw4 = np.frombuffer(expert_4bit[down_off:down_off + DOWN_W_SZ], dtype=np.uint32).reshape(R2, 128)
    ds4 = np.frombuffer(expert_4bit[down_off + DOWN_W_SZ:down_off + DOWN_W_SZ + R2 * 16 * 2], dtype=np.float16).reshape(R2, 16)
    db4 = np.frombuffer(expert_4bit[down_off + DOWN_W_SZ * 2:down_off + DOWN_W_SZ * 2 + R2 * 16 * 2], dtype=np.float16).reshape(R2, 16)
    
    def dequant_4bit_to_f32(weight, scales, biases, R, vals_per_group=64):
        """Dequantize 4-bit affine to float32.
        weight: (R, packed_cols) uint32 — each uint32 has vals_per_group/8 * 8 values
        For 4-bit: vals_per_group=64, so packed_cols = vals/8 per row.
        """
        # Reshape: (R, packed_cols) → (R, num_groups, 8) where num_groups = packed_cols
        # Each uint32 = 8 4-bit values in SAME group
        # Actually for 4-bit affine: each group has vals_per_group=64 values
        # and 8 uint32s per group
        num_groups = weight.shape[1]  # 384 for gate/up, 128 for down
        groups_per_u32 = 8  # 8 groups per uint32 (each group has 8 values at 4-bit)
        assert num_groups % groups_per_u32 == 0
        
        w = weight.reshape(R, num_groups // groups_per_u32, groups_per_u32)
        # w: (R, 48, 8) for gate/up, (R, 16, 8) for down
        
        # Extract 4-bit nibbles: (packed >> (n*4)) & 0xF, giving 8 values per uint32
        nib = np.zeros((R, num_groups // groups_per_u32, 64), dtype=np.float32)
        for bit in range(8):
            nib += ((w >> (bit * 4)) & 0xF).astype(np.float32) * (2.0 ** (bit * 4))
        # nib: (R, 48, 64) for gate/up — reshape to (R, 3072)
        nib = nib.reshape(R, -1)
        
        # Scales/biases: (R, num_groups) — broadcast to each group's 64 values
        sc = scales[:, :, None].astype(np.float32)  # (R, 48, 1)
        bi = biases[:, :, None].astype(np.float32)   # (R, 48, 1)
        return (nib * sc + bi).reshape(R, -1)  # (R, 3072) for gate/up
    
    def quant_f32_to_2bit(fvals, R, vals_per_group=64, num_groups=None):
        """Quantize float32 to 2-bit affine (merge=1: one scale/bias per group).
        fvals: (R, vals) float32 — vals must be divisible by vals_per_group
        Returns: (packed_u32, scales_f16, biases_f16)
        """
        if num_groups is None:
            num_groups = fvals.shape[1] // vals_per_group
        merge = 1  # NO merging
        
        # Reshape: (R, num_groups, vals_per_group)
        fvals = fvals.reshape(R, num_groups, vals_per_group)
        
        # Per-group: min/max → scale/bias
        mn = fvals.min(axis=2)   # (R, num_groups)
        mx = fvals.max(axis=2)
        scales = ((mx - mn) / 3.0).astype(np.float16)
        biases = mn.astype(np.float16)
        
        # Quantize: q = round((v - mn) / scale), clip to [0, 3]
        q = ((fvals - mn[:, :, None]) / (mx - mn)[:, :, None] * 3.0).round()
        np.clip(q, 0, 3, out=q).astype(np.uint8)
        
        # Pack 16 vals per uint32: (R, num_groups, vals_per_group) → (R, num_groups, 4, 4)
        assert vals_per_group % 16 == 0
        n_u32 = vals_per_group // 16
        q = q.reshape(R, num_groups, n_u32, 16)
        packed = (q[..., 0].astype(np.uint32) |
                  (q[..., 1].astype(np.uint32) << 2) |
                  (q[..., 2].astype(np.uint32) << 4) |
                  (q[..., 3].astype(np.uint32) << 6))
        packed = packed.reshape(R, num_groups * n_u32)  # (R, num_groups * vals_per_group / 16)
        return packed, scales, biases
    
    # Dequantize gate/up/down from 4-bit
    gw_f = dequant_4bit_to_f32(gw4, gs4, gb4, R1)  # (1024, 3072)
    uw_f = dequant_4bit_to_f32(uw4, us4, ub4, R1)  # (1024, 3072)
    dw_f = dequant_4bit_to_f32(dw4, ds4, db4, R2)  # (3072, 1024)
    
    # Quantize to 2-bit
    # gate/up: 3072 vals → 48 groups (64 vals/group) → 48 * 64 / 16 = 192 uint32s
    gw2, gs2, gb2 = quant_f32_to_2bit(gw_f, R1, 64, 48)  # (1024, 192), (1024, 48), (1024, 48)
    uw2, us2, ub2 = quant_f32_to_2bit(uw_f, R1, 64, 48)
    # down: 1024 vals → 16 groups (64 vals/group) → 16 * 64 / 16 = 64 uint32s
    dw2, ds2, db2 = quant_f32_to_2bit(dw_f, R2, 64, 16)  # (3072, 64), (3072, 16), (3072, 16)
    
    # Assemble into 2,949,120 byte buffer
    out = bytearray(EXPERT_SIZE_2BIT)
    off = 0
    
    def put(arr, offset):
        arr.tobytes().__buffer__(out, offset)
    
    put(gw2, GATE_W_2)
    put(gs2, GATE_S_2)
    put(gb2, GATE_B_2)
    put(uw2, UP_W_2)
    put(us2, UP_S_2)
    put(ub2, UP_B_2)
    put(dw2, DOWN_W_2)
    put(ds2, DOWN_S_2)
    put(db2, DOWN_B_2)
    
    assert out.__len__() == EXPERT_SIZE_2BIT
    return bytes(out)


def convert_layer_vectorized(layer_4bit, num_experts=NUM_EXPERTS):
    """Convert all 256 experts in one layer. Vectorized for speed."""
    E = num_experts
    R1, R2 = 1024, 3072
    
    # Gate: (E*1024, 384) uint32 → (E, 1024, 384)
    gw4 = np.frombuffer(layer_4bit[GATE_W_SZ * E:GATE_W_SZ * E + E * GATE_W_SZ],
                         dtype=np.uint32).reshape(E, R1, 384)
    gs4 = np.frombuffer(layer_4bit[GATE_S_SZ * E:GATE_S_SZ * E + E * GATE_S_SZ],
                         dtype=np.float16).reshape(E, R1, 48)
    gb4 = np.frombuffer(layer_4bit[GATE_S_SZ * 2 * E:GATE_S_SZ * 2 * E + E * GATE_S_SZ],
                         dtype=np.float16).reshape(E, R1, 48)
    
    up_start = GATE_W_SZ * E * 2 + GATE_S_SZ * E * 2
    uw4 = np.frombuffer(layer_4bit[up_start:up_start + E * UP_W_SZ],
                         dtype=np.uint32).reshape(E, R1, 384)
    us4 = np.frombuffer(layer_4bit[up_start + E * UP_W_SZ:up_start + E * UP_W_SZ + E * GATE_S_SZ],
                         dtype=np.float16).reshape(E, R1, 48)
    ub4 = np.frombuffer(layer_4bit[up_start + E * UP_W_SZ + E * GATE_S_SZ:up_start + E * UP_W_SZ + E * GATE_S_SZ * 2],
                         dtype=np.float16).reshape(E, R1, 48)
    
    down_start = up_start + UP_W_SZ * E + GATE_S_SZ * E * 2
    dw4 = np.frombuffer(layer_4bit[down_start:down_start + E * DOWN_W_SZ],
                         dtype=np.uint32).reshape(E, R2, 128)
    ds4 = np.frombuffer(layer_4bit[down_start + E * DOWN_W_SZ:down_start + E * DOWN_W_SZ + E * R2 * 16 * 2],
                         dtype=np.float16).reshape(E, R2, 16)
    db4 = np.frombuffer(layer_4bit[down_start + E * DOWN_W_SZ * 2:down_start + E * DOWN_W_SZ * 2 + E * R2 * 16 * 2],
                         dtype=np.float16).reshape(E, R2, 16)
    
    def dequant_vec(weight, scales, biases, R):
        """weight: (E, R, packed_cols). scales/biases: (E, R, num_groups)."""
        num_groups = weight.shape[2]  # 384 for gate, 128 for down
        groups_per_u32 = 8
        assert num_groups % groups_per_u32 == 0
        W = weight.reshape(E * R, num_groups // groups_per_u32, groups_per_u32)
        nib = np.zeros((E * R, num_groups // groups_per_u32, 64), dtype=np.float32)
        for bit in range(8):
            nib += ((W >> (bit * 4)) & 0xF).astype(np.float32) * (2.0 ** (bit * 4))
        nib = nib.reshape(E * R, -1)
        sc = scales.reshape(E * R, -1)[:, None, :].astype(np.float32)
        bi = biases.reshape(E * R, -1)[:, None, :].astype(np.float32)
        return (nib * sc + bi).reshape(E, R, -1)
    
    def quant_vec(fvals, R, vals_g=64, n_g=None):
        """fvals: (E, R, vals). Returns (E, R, packed_u32), (E, R, n_g), (E, R, n_g)."""
        if n_g is None:
            n_g = fvals.shape[2] // vals_g
        merge = 1
        fvals = fvals.reshape(E * R, n_g, vals_g)
        mn = fvals.min(axis=2)[:, :, None].astype(np.float16)
        mx = fvals.max(axis=2)[:, :, None].astype(np.float16)
        sc = ((mx - mn) / 3.0).astype(np.float16)
        bi = mn.astype(np.float16)
        q = ((fvals - mn) / (mx - mn) * 3.0).round().clip(0, 3).astype(np.uint8)
        n_u32 = vals_g // 16
        q = q.reshape(E * R, n_g, n_u32, 16)
        packed = (q[..., 0].astype(np.uint32) |
                  (q[..., 1].astype(np.uint32) << 2) |
                  (q[..., 2].astype(np.uint32) << 4) |
                  (q[..., 3].astype(np.uint32) << 6))
        packed = packed.reshape(E * R, n_g * n_u32)
        return (packed.reshape(E, R, -1),
                sc.reshape(E, R, -1),
                bi.reshape(E, R, -1))
    
    print(f"    Dequantizing gate/up/down for {E} experts...")
    gw_f = dequant_vec(gw4, gs4, gb4, R1)  # (E, 1024, 3072)
    uw_f = dequant_vec(uw4, us4, ub4, R1)  # (E, 1024, 3072)
    dw_f = dequant_vec(dw4, ds4, db4, R2)  # (E, 3072, 1024)
    
    print(f"    Quantizing to 2-bit...")
    gw2, gs2, gb2 = quant_vec(gw_f, R1, 64, 48)
    uw2, us2, ub2 = quant_vec(uw_f, R1, 64, 48)
    dw2, ds2, db2 = quant_vec(dw_f, R2, 64, 16)
    
    print(f"    Assembling layer...")
    layer_out = bytearray(E * EXPERT_SIZE_2BIT)
    for ei in range(E):
        off = ei * EXPERT_SIZE_2BIT
        e = ei
        gw2[e].tobytes().__buffer__(layer_out, off + GATE_W_2)
        gs2[e].tobytes().__buffer__(layer_out, off + GATE_S_2)
        gb2[e].tobytes().__buffer__(layer_out, off + GATE_B_2)
        uw2[e].tobytes().__buffer__(layer_out, off + UP_W_2)
        us2[e].tobytes().__buffer__(layer_out, off + UP_S_2)
        ub2[e].tobytes().__buffer__(layer_out, off + UP_B_2)
        dw2[e].tobytes().__buffer__(layer_out, off + DOWN_W_2)
        ds2[e].tobytes().__buffer__(layer_out, off + DOWN_S_2)
        db2[e].tobytes().__buffer__(layer_out, off + DOWN_B_2)
    
    return bytes(layer_out)


def main():
    import argparse
    p = argparse.ArgumentParser(description='Convert 4-bit SSD experts to 2-bit for Qwen3.5-MoE')
    p.add_argument('--input', '-i', default='metal_infer/out_122b/packed_experts_ssd.bin')
    p.add_argument('--output', '-o', default='metal_infer/out_122b/packed_experts_2bit_ssd.bin')
    p.add_argument('--resume', '-r', action='store_true')
    p.add_argument('--expert-size', '-e', type=int, default=EXPERT_SIZE_4BIT,
                   help='4-bit expert size (default: 5,308,416)')
    args = p.parse_args()
    
    inp = Path(args.input).expanduser()
    out = Path(args.output).expanduser()
    ESZ4 = args.expert_size
    
    if not inp.exists():
        print(f"ERROR: {inp} not found"); sys.exit(1)
    
    inp_size = inp.stat().st_size
    total_exp = NUM_LAYERS * NUM_EXPERTS
    inp_exps = inp_size // ESZ4
    inp_lays = inp_exps // NUM_EXPERTS
    
    print(f"Input:  {inp} ({inp_size/1e9:.2f} GB, {inp_lays} layers × {NUM_EXPERTS} experts)")
    print(f"Output: {out}")
    print(f"Expert: {ESZ4:,} (4-bit) → {EXPERT_SIZE_2BIT:,} (2-bit)")
    print(f"Ratio:  {EXPERT_SIZE_2BIT/ESZ4:.3f}x compression")
    
    mode = 'r+b' if args.resume and out.exists() else 'wb'
    start_layer = 0
    
    if args.resume and out.exists():
        out_size = out.stat().st_size
        start_layer = out_size // (NUM_EXPERTS * EXPERT_SIZE_2BIT)
        inp_size_verified = inp_lays * NUM_EXPERTS * ESZ4
        if inp_size < inp_size_verified:
            print(f"WARNING: Input file changed. Starting from layer 0.")
            start_layer = 0; mode = 'wb'
        else:
            print(f"Resuming from layer {start_layer}...")
    
    t0 = time.time()
    with open(inp, 'rb') as fin, open(out, mode) as fout:
        if mode == 'wb':
            # Write header: magic(4) + ver(4) + nlayers(4) + nexperts(4) + expert_size(8)
            header = struct.pack('>IIIIQ', 0x53445046, 2, NUM_LAYERS, NUM_EXPERTS, EXPERT_SIZE_2BIT)
            fout.write(header)
            print(f"Header: {NUM_LAYERS}L × {NUM_EXPERTS}E × {EXPERT_SIZE_2BIT:,}B = {NUM_LAYERS*NUM_EXPERTS*EXPERT_SIZE_2BIT/1e9:.2f} GB")
        
        for li in range(start_layer, NUM_LAYERS):
            t1 = time.time()
            off = li * NUM_EXPERTS * ESZ4
            fin.seek(off)
            data = fin.read(NUM_EXPERTS * ESZ4)
            layer_out = convert_layer_vectorized(data, NUM_EXPERTS)
            fout.write(layer_out)
            
            elapsed = time.time() - t0
            eta = elapsed / (li - start_layer + 1) * (NUM_LAYERS - li - 1)
            spd = (li - start_layer + 1) * NUM_EXPERTS * ESZ4 / elapsed / 1e9
            print(f"  Layer {li}: {layer_out.__len__()/1e6:.0f} MB, "
                  f"{spd:.1f} GB/s overall, ETA: {eta:.0f}s")
    
    total = time.time() - t0
    out_sz = out.stat().st_size
    print(f"\nDone! {out_sz/1e9:.2f} GB in {total:.0f}s ({out_sz/total/1e9:.1f} GB/s)")


if __name__ == '__main__':
    main()
