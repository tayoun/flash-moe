#!/usr/bin/env python3
"""
Convert Qwen3.5-MoE 4-bit experts to 2-bit.
Reads 4-bit from: /Volumes/Seagate Backup Plus Drive/packed_experts_ssd.bin
Writes 2-bit per-layer files to: /Volumes/Seagate Backup Plus Drive/flash-moe-2bit-new/packed_experts/

Format: per-layer files (256 experts each), no header, expert_size=3,342,336 bytes.
Total output: ~38.3 GB (48 layers x 256 experts x 3,342,336 bytes)
"""
import struct, os, sys, time, multiprocessing
import numpy as np

NUM_LAYERS = 48
NUM_EXPERTS = 256
EXPERT_SIZE_4BIT = 5_308_416
EXPERT_SIZE_2BIT = 3_342_336

SRC = "/Volumes/Seagate Backup Plus Drive/packed_experts_ssd.bin"
DST_DIR = "/Volumes/Seagate Backup Plus Drive/flash-moe-2bit-new/packed_experts"
os.makedirs(DST_DIR, exist_ok=True)

# 4-bit file layout (verified empirically)
GATE_W_OFF = 0;           GATE_W_SZ = 1_572_864
GATE_S_OFF = 1_572_864;   GATE_S_SZ = 98_304
GATE_B_OFF = 1_671_168;   GATE_B_SZ = 98_304
UP_W_OFF   = 1_769_472;   UP_W_SZ   = 1_572_864
UP_S_OFF   = 3_342_336;   UP_S_SZ   = 98_304
UP_B_OFF   = 3_440_640;   UP_B_SZ   = 98_304
DOWN_W_OFF = 3_538_944;   DOWN_W_SZ = 1_572_864
DOWN_S_OFF = 5_111_808;   DOWN_S_SZ = 98_304
DOWN_B_OFF = 5_406_720;   DOWN_B_SZ = 98_304

# 2-bit offsets (kernel formula)
GATE_W_2 = 0;             GATE_S_2 = 786_432;     GATE_B_2 = 884_736
UP_W_2   = 983_040;       UP_S_2   = 1_769_472;    UP_B_2   = 1_867_776
DOWN_W_2 = 1_966_080;     DOWN_S_2 = 2_752_512;    DOWN_B_2 = 3_047_424


def convert_proj(W4, S4, B4, R, packed_cols, n_g):
    """Convert one projection from 4-bit to 2-bit."""
    vals_per_group = 64
    # W4: (R, packed_cols), reshape to groups of 8 uint32s
    W = W4.reshape(R, packed_cols // 8, 8)  # (R, n_g, 8)
    # Extract 4-bit nibbles, build float32
    W_f = np.zeros((R, n_g, vals_per_group), dtype=np.float32)
    for bit in range(8):
        nib = ((W >> (bit * 4)) & 0xF).astype(np.float32)  # (R, n_g, 8)
        W_f += np.repeat(nib, vals_per_group // 8, axis=2) * (2.0 ** (bit * 4))
    W_f = W_f.reshape(R, -1)  # (R, n_g * vals_per_group)

    # Dequantize: W_deq = W_f * scales + biases
    S4_f = S4.astype(np.float32)  # (R, n_g)
    B4_f = B4.astype(np.float32)  # (R, n_g)
    W_deq = W_f * S4_f + B4_f  # broadcasts to (R, n_g*vals_per_group)

    # Quantize to 2-bit per group
    W_deq = W_deq.reshape(R, n_g, vals_per_group)
    mn = W_deq.min(axis=2)   # (R, n_g)
    mx = W_deq.max(axis=2)   # (R, n_g)
    diff = np.clip(mx - mn, 1e-8, 65504.0)
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
    return packed.reshape(R, n_g * n_u32), sc, bi


def convert_expert(expert_4bit):
    """Convert one 4-bit expert to 2-bit. Returns bytes of length EXPERT_SIZE_2BIT."""
    R1, R2 = 1024, 3072
    gate_groups, down_groups = 48, 16

    gw4 = np.frombuffer(expert_4bit[GATE_W_OFF:GATE_W_OFF + GATE_W_SZ], dtype=np.uint32).reshape(R1, 384)
    gs4 = np.frombuffer(expert_4bit[GATE_S_OFF:GATE_S_OFF + GATE_S_SZ], dtype=np.float16).reshape(R1, 48)
    gb4 = np.frombuffer(expert_4bit[GATE_B_OFF:GATE_B_OFF + GATE_B_SZ], dtype=np.float16).reshape(R1, 48)
    uw4 = np.frombuffer(expert_4bit[UP_W_OFF:UP_W_OFF + UP_W_SZ], dtype=np.uint32).reshape(R1, 384)
    us4 = np.frombuffer(expert_4bit[UP_S_OFF:UP_S_OFF + UP_S_SZ], dtype=np.float16).reshape(R1, 48)
    ub4 = np.frombuffer(expert_4bit[UP_B_OFF:UP_B_OFF + UP_B_SZ], dtype=np.float16).reshape(R1, 48)
    dw4 = np.frombuffer(expert_4bit[DOWN_W_OFF:DOWN_W_OFF + DOWN_W_SZ], dtype=np.uint32).reshape(R2, 128)
    ds4 = np.frombuffer(expert_4bit[DOWN_S_OFF:DOWN_S_OFF + DOWN_S_SZ], dtype=np.float16).reshape(R2, 16)
    db4 = np.zeros((R2, 16), dtype=np.float16)  # not stored in 4-bit file

    gw2, gs2, gb2 = convert_proj(gw4, gs4, gb4, R1, 384, gate_groups)
    uw2, us2, ub2 = convert_proj(uw4, us4, ub4, R1, 384, gate_groups)
    dw2, ds2, db2 = convert_proj(dw4, ds4, db4, R2, 128, down_groups)

    buf = bytearray(EXPERT_SIZE_2BIT)
    def put(arr, off):
        buf[off:off + arr.nbytes] = arr.tobytes()
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


def convert_layer(args):
    """Convert one layer (256 experts) and write per-layer file."""
    layer_idx, src_path, dst_dir = args
    t0 = time.time()

    with open(src_path, 'rb') as f:
        f.seek(16 + layer_idx * NUM_EXPERTS * EXPERT_SIZE_4BIT)
        layer_data = f.read(NUM_EXPERTS * EXPERT_SIZE_4BIT)

    experts = []
    for e in range(NUM_EXPERTS):
        expert_4bit = layer_data[e * EXPERT_SIZE_4BIT:(e + 1) * EXPERT_SIZE_4BIT]
        expert_2bit = convert_expert(expert_4bit)
        experts.append(expert_2bit)

    dst_path = os.path.join(dst_dir, f"layer_{layer_idx:02d}.bin")
    with open(dst_path, 'wb') as fout:
        for expert_2bit in experts:
            fout.write(expert_2bit)

    elapsed = time.time() - t0
    return layer_idx, elapsed


def main():
    num_workers = min(4, multiprocessing.cpu_count())
    print(f"Converting {NUM_LAYERS} layers with {num_workers} workers...")
    print(f"Source: {SRC}")
    print(f"Dest: {DST_DIR}")
    print(f"Per layer: {NUM_EXPERTS * EXPERT_SIZE_2BIT / 1e6:.1f}MB")
    print(f"Total output: {NUM_LAYERS * NUM_EXPERTS * EXPERT_SIZE_2BIT / 1e9:.1f}GB")
    print(f"Per expert: {EXPERT_SIZE_2BIT / 1e6:.3f}MB (kernel formula)")
    print()

    t0 = time.time()
    args_list = [(l, SRC, DST_DIR) for l in range(NUM_LAYERS)]

    completed = 0
    with multiprocessing.Pool(num_workers) as pool:
        for result in pool.imap_unordered(convert_layer, args_list):
            completed += 1
            elapsed_total = time.time() - t0
            rate = completed / elapsed_total
            eta = (NUM_LAYERS - completed) / rate / 60
            print(f"Layer {result[0]:02d}: {result[1]:.1f}s | {completed}/{NUM_LAYERS} | "
                  f"Total: {elapsed_total/60:.1f}min | ETA: {eta:.1f}min")

    total = time.time() - t0
    print(f"\n=== COMPLETE in {total/60:.1f} minutes ===")

    # Verify
    total_size = sum(
        os.path.getsize(os.path.join(DST_DIR, f"layer_{l:02d}.bin"))
        for l in range(NUM_LAYERS)
    )
    expected = NUM_LAYERS * NUM_EXPERTS * EXPERT_SIZE_2BIT
    print(f"Output: {total_size / 1e9:.2f}GB | Expected: {expected / 1e9:.2f}GB | Match: {total_size == expected}")


if __name__ == '__main__':
    main()
