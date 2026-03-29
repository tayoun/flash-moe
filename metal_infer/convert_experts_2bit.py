#!/usr/bin/env python3
"""
convert_experts_2bit.py — Convert 4-bit experts to 2-bit for 122B MoE.
Output layout matches Metal kernel dequant_matvec_2bit expectations:
  gate/up: weight [out=1024, in=384] → 2-bit [1024, 384/16=24]
  down:   weight [out=3072, in=128]  → 2-bit [3072, 128/16=8]
Per-expert 2-bit size: 2,949,120 bytes (matches infer.m layout).
"""

import struct, os, sys, argparse, time
import numpy as np
from pathlib import Path

NUM_LAYERS = 48
NUM_EXPERTS = 256
EXPERT_SIZE_4BIT = 5_308_416
EXPERT_SIZE_2BIT = 2_949_120

# 4-bit offsets
GATE_W, GATE_S, GATE_B = 0, 1_572_864, 1_671_168
UP_W,   UP_S,   UP_B   = 1_769_472, 3_342_336, 3_440_640
DOWN_W, DOWN_S, DOWN_B = 3_538_944, 5_111_808, 5_210_112


def dequantize_4bit_to_f32(weight_u32, scales, biases, *, R, vals_per_group):
    """Dequantize 4-bit weights to float32. Returns (R, num_groups, vals_per_group) per expert."""
    # weight_u32: (R, num_u32) = (R, vals_per_group * num_groups / 8)
    num_groups = weight_u32.shape[1] * 8 // vals_per_group  # e.g. 384*8/64 = 48 for gate
    # Reshape to (R, num_groups, 8)
    w = weight_u32.reshape(R, num_groups, vals_per_group // 8, 8)
    # Extract 4-bit nibbles: (R, num_groups, vals_per_group)
    nib = np.zeros((R, num_groups, vals_per_group), dtype=np.float32)
    for bit in range(8):
        nib += ((w >> (bit * 4)) & 0xF).astype(np.float32) * (2 ** (bit * 4))
    # Scale and bias
    # scales/biases: (R, num_groups) → reshape for broadcast
    sc = scales[:, :, None].astype(np.float32)  # (R, num_groups, 1)
    bi = biases[:, :, None].astype(np.float32)
    return nib * sc + bi  # (R, num_groups, vals_per_group)


def quantize_to_2bit(fvals, *, R, vals_per_group, num_groups):
    """Convert float32 values to 2-bit. Returns (packed_uint32, scales, biases)."""
    # fvals: (R, num_groups, vals_per_group)
    # Merge groups: merge every 8 groups of vals_per_group into 1 group of vals_per_group
    # Result: (R, num_groups/8, vals_per_group*8) — but num_groups must be divisible by 8
    merge = vals_per_group  # merge within each group
    assert num_groups % merge == 0
    n_out_groups = num_groups // merge
    # Reshape: (R, n_out_groups, merge, vals_per_group) → (R, n_out_groups, merge*vals_per_group)
    fvals = fvals.reshape(R, n_out_groups, merge, vals_per_group)
    # Per merged group: min/max → scale, bias
    mn = fvals.min(axis=(2, 3))  # (R, n_out_groups)
    mx = fvals.max(axis=(2, 3))
    sc = ((mx - mn) / 3.0).astype(np.float16)
    bi = mn.astype(np.float16)
    # Quantize: q = round((v - mn) / scale), clip to [0, 3]
    q = ((fvals - mn[:, :, None, None]) / sc[:, :, None, None]).round()
    np.clip(q, 0, 3, out=q)
    # Pack 16 vals per uint32: reshape (R, n_out_groups, merge, vals_per_group) → pack into (R, n_out_groups, merge, vals_per_group/4)
    # vals_per_group must be divisible by 16 for this to work
    assert vals_per_group % 16 == 0
    n_u32 = vals_per_group // 16
    q = q.reshape(R, n_out_groups, merge, n_u32, 16)
    packed = (q[..., 0].astype(np.uint32) |
              (q[..., 1].astype(np.uint32) << 2) |
              (q[..., 2].astype(np.uint32) << 4) |
              (q[..., 3].astype(np.uint32) << 6))
    return packed, sc, bi


def convert_expert(expert_data):
    """Convert one 4-bit expert to 2-bit. Returns 2,949,120 bytes."""
    E = 1
    # Parse 4-bit expert
    gw  = np.frombuffer(expert_data[GATE_W:GATE_W+1_572_864], dtype=np.uint32).reshape(1, 1024, 384)
    gs  = np.frombuffer(expert_data[GATE_S:GATE_S+98_304], dtype=np.float16).reshape(1, 1024, 48)
    gb  = np.frombuffer(expert_data[GATE_B:GATE_B+98_304], dtype=np.float16).reshape(1, 1024, 48)
    uw  = np.frombuffer(expert_data[UP_W:UP_W+1_572_864], dtype=np.uint32).reshape(1, 1024, 384)
    us  = np.frombuffer(expert_data[UP_S:UP_S+98_304], dtype=np.float16).reshape(1, 1024, 48)
    ub  = np.frombuffer(expert_data[UP_B:UP_B+98_304], dtype=np.float16).reshape(1, 1024, 48)
    dw  = np.frombuffer(expert_data[DOWN_W:DOWN_W+1_572_864], dtype=np.uint32).reshape(1, 3072, 128)
    ds  = np.frombuffer(expert_data[DOWN_S:DOWN_S+98_304], dtype=np.float16).reshape(1, 3072, 16)
    db  = np.frombuffer(expert_data[DOWN_B:DOWN_B+98_304], dtype=np.float16).reshape(1, 3072, 16)

    # gate/up: 4-bit [1024, 384] → 2-bit [1024, 24]
    # down: 4-bit [3072, 128] → 2-bit [3072, 8]

    # Gate
    gw_f = dequantize_4bit_to_f32(gw[0], gs[0], gb[0], R=1024, vals_per_group=64)
    gw2, gs2, gb2 = quantize_to_2bit(gw_f, R=1024, vals_per_group=64, num_groups=48)
    # up
    uw_f = dequantize_4bit_to_f32(uw[0], us[0], ub[0], R=1024, vals_per_group=64)
    uw2, us2, ub2 = quantize_to_2bit(uw_f, R=1024, vals_per_group=64, num_groups=48)
    # down: 4-bit [3072, 128], 16 groups/row
    dw_f = dequantize_4bit_to_f32(dw[0], ds[0], db[0], R=3072, vals_per_group=64)
    dw2, ds2, db2 = quantize_to_2bit(dw_f, R=3072, vals_per_group=64, num_groups=16)

    # Verify shapes
    assert gw2.shape == (1024, 24), f"gate weight: {gw2.shape}"
    assert gs2.shape == (1024, 6), f"gate scales: {gs2.shape}"
    assert uw2.shape == (1024, 24), f"up weight: {uw2.shape}"
    assert dw2.shape == (3072, 8), f"down weight: {dw2.shape}"
    assert ds2.shape == (3072, 2), f"down scales: {ds2.shape}"

    # Assemble
    out = bytearray(EXPERT_SIZE_2BIT)
    off = 0
    gw2.tobytes().__buffer__(out, off);  off += 98_304
    gs2.tobytes().__buffer__(out, off); off += 12_288
    gb2.tobytes().__buffer__(out, off); off += 12_288
    uw2.tobytes().__buffer__(out, off); off += 98_304
    us2.tobytes().__buffer__(out, off); off += 12_288
    ub2.tobytes().__buffer__(out, off); off += 12_288
    dw2.tobytes().__buffer__(out, off); off += 294_912
    ds2.tobytes().__buffer__(out, off); off += 12_288
    db2.tobytes().__buffer__(out, off); off += 12_288
    assert off == EXPERT_SIZE_2BIT, f"off={off} != {EXPERT_SIZE_2BIT}"
    return bytes(out)


def convert_layer_fast(data):
    """Convert all 256 experts in one layer using vectorized operations."""
    E = NUM_EXPERTS
    ESZ4 = EXPERT_SIZE_4BIT

    # Parse entire layer
    gw = np.frombuffer(data[GATE_W:GATE_W + E*1_572_864], dtype=np.uint32).reshape(E, 1024, 384)
    gs = np.frombuffer(data[GATE_S:GATE_S + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    gb = np.frombuffer(data[GATE_B:GATE_B + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    uw = np.frombuffer(data[UP_W:UP_W + E*1_572_864], dtype=np.uint32).reshape(E, 1024, 384)
    us = np.frombuffer(data[UP_S:UP_S + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    ub = np.frombuffer(data[UP_B:UP_B + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    dw = np.frombuffer(data[DOWN_W:DOWN_W + E*1_572_864], dtype=np.uint32).reshape(E, 3072, 128)
    ds = np.frombuffer(data[DOWN_S:DOWN_S + E*98_304], dtype=np.float16).reshape(E, 3072, 16)
    db = np.frombuffer(data[DOWN_B:DOWN_B + E*98_304], dtype=np.float16).reshape(E, 3072, 16)

    def proj_to_2bit(weight, scales, biases, R, vals_g, num_g):
        """weight: (E, R, num_g*vals_g/8) uint32. scales/biases: (E, R, num_g) bf16."""
        # Reshape to (E, R, num_g, vals_g/8, 8)
        vp8 = vals_g // 8
        w = weight.reshape(E, R, num_g, vp8, 8)
        # Extract 4-bit nibbles: (E, R, num_g, vals_g) float32
        nib = np.zeros((E, R, num_g, vals_g), dtype=np.float32)
        for bit in range(8):
            nib += ((w >> (bit * 4)) & 0xF).astype(np.float32) * (2 ** (bit * 4))
        del w
        # Scale/bias: (E, R, num_g, 1) broadcast
        sc = scales[:, :, :, None].astype(np.float32)
        bi = biases[:, :, :, None].astype(np.float32)
        fvals = nib * sc + bi  # (E, R, num_g, vals_g)
        del nib, sc, bi

        # Merge groups: every vals_g vals in same group → new group count = num_g
        # reshape: (E, R, num_g, vals_g) → (E, R, 1, vals_g) then tile to (E, R, num_g, vals_g) — NO
        # We merge: group[i] = vals[i*vals_g:(i+1)*vals_g] for each of num_g groups
        # Result: (E, R, num_g, vals_g) — same shape but values are per-merged-group
        # For 4-bit→2-bit: merge 8 consecutive 4-bit groups into 1 2-bit group
        merge = 8  # 8 groups × 64 vals = 512 vals → merge to 1 group × 512 vals
        assert num_g % merge == 0
        n_out = num_g // merge  # 48→6 for gate, 16→2 for down
        # Reshape: (E, R, num_g, vals_g) → (E, R, n_out, merge, vals_g)
        fvals = fvals.reshape(E, R, n_out, merge, vals_g)
        mn = fvals.min(axis=(3, 4))  # (E, R, n_out)
        mx = fvals.max(axis=(3, 4))
        sc2 = ((mx - mn) / 3.0).astype(np.float16)
        bi2 = mn.astype(np.float16)
        del mx

        q = ((fvals - mn[:, :, :, None, None]) / sc2[:, :, :, None, None]).round()
        np.clip(q, 0, 3, out=q); q = q.astype(np.uint8)
        del fvals

        # Pack: vals_g/16 = 4 uint32s per group
        assert vals_g % 16 == 0
        n_u32 = vals_g // 16
        q = q.reshape(E, R, n_out, merge, n_u32, 16)
        packed = (q[..., 0].astype(np.uint32) |
                  (q[..., 1].astype(np.uint32) << 2) |
                  (q[..., 2].astype(np.uint32) << 4) |
                  (q[..., 3].astype(np.uint32) << 6))
        # packed: (E, R, n_out, merge, n_u32) — flatten last 3 dims to (E, R, n_out*n_u32*merge)
        # kernel expects [out_dim, n_out*n_u32*merge] = [R, n_out*n_u32*merge]
        packed = packed.reshape(E, R, n_out * n_u32 * merge)
        return packed, sc2, bi2

    gw2, gs2, gb2 = proj_to_2bit(gw, gs, gb, 1024, 64, 48)  # [E, 1024, 24]
    uw2, us2, ub2 = proj_to_2bit(uw, us, ub, 1024, 64, 48)   # [E, 1024, 24]
    dw2, ds2, db2 = proj_to_2bit(dw, ds, db, 3072, 64, 16)   # [E, 3072, 8]

    # Verify shapes
    assert gw2.shape == (E, 1024, 24), f"gate: {gw2.shape}"
    assert gs2.shape == (E, 1024, 6),  f"gate sc: {gs2.shape}"
    assert dw2.shape == (E, 3072, 8),   f"down: {dw2.shape}"
    assert ds2.shape == (E, 3072, 2),   f"down sc: {ds2.shape}"

    # Assemble
    parts = []
    for ei in range(E):
        parts.append(gw2[ei].tobytes())   # 1024*24*4 = 98,304
        parts.append(gs2[ei].tobytes())   # 1024*6*2 = 12,288
        parts.append(gb2[ei].tobytes())  # 12,288
        parts.append(uw2[ei].tobytes())   # 98,304
        parts.append(us2[ei].tobytes())   # 12,288
        parts.append(ub2[ei].tobytes())  # 12,288
        parts.append(dw2[ei].tobytes())  # 3072*8*4 = 98,304
        parts.append(ds2[ei].tobytes())  # 3072*2*2 = 12,288
        parts.append(db2[ei].tobytes())  # 12,288
    return b"".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/Volumes/Seagate Backup Plus Drive/flash-moe-2bit")
    parser.add_argument("--out", default="/Volumes/Seagate Backup Plus Drive/flash-moe-2bit/packed_experts_2bit_ssd.bin")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    packed_dir = Path(args.model) / "packed_experts"
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

    if args.layer is not None:
        layers = [args.layer]; fout_mode = 'wb'
        if out_path.exists(): out_path.unlink()
    elif args.resume and out_path.exists():
        file_size = out_path.stat().st_size
        done = (file_size - 16) // (NUM_EXPERTS * EXPERT_SIZE_2BIT)
        start_l = min(int(done) + 1, NUM_LAYERS)
        print(f"Resuming from layer {start_l} ({int(done)} complete)")
        layers = range(start_l, NUM_LAYERS); fout_mode = 'ab'
    else:
        layers = range(NUM_LAYERS); fout_mode = 'wb'
        if out_path.exists(): out_path.unlink()

    start = time.time()
    with open(out_path, fout_mode) as fout:
        if fout_mode == 'wb':
            fout.write(struct.pack(">IIQ", NUM_LAYERS, NUM_EXPERTS, EXPERT_SIZE_2BIT))
        for li in layers:
            t0 = time.time()
            data = open(packed_dir / f"layer_{li:02d}.bin", "rb").read()
            fout.write(convert_layer_fast(data))
            done = (li + 1) * NUM_EXPERTS
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done)
            print(f"  L{li:02d}: {done}/{total} ({100*done/total:.0f}%) "
                  f"layer={time.time()-t0:.1f}s ETA={eta/60:.0f}min")

    sz = out_path.stat().st_size
    print(f"Done in {(time.time()-start)/60:.0f}min — {sz/1024**3:.1f} GB")


if __name__ == "__main__":
    main()
