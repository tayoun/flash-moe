#!/usr/bin/env python3
"""Resume the 2-bit conversion from where it left off."""
import struct, time
import numpy as np
from pathlib import Path

NUM_EXPERTS = 256
EXPERT_SIZE_2BIT = 2_654_208
GATE_W, GATE_S, GATE_B = 0, 1_572_864, 1_671_168
UP_W,   UP_S,   UP_B   = 1_769_472, 3_342_336, 3_440_640
DOWN_W, DOWN_S, DOWN_B = 3_538_944, 5_111_808, 5_210_112

def quantize_proj(weight4, scale4, bias4, out_w, out_s, out_b, *, R, C, groups_per_row):
    E = NUM_EXPERTS
    w4 = weight4.reshape(E, R, groups_per_row, 8)
    nib = np.zeros((E, R, groups_per_row, 8, 8), dtype=np.uint8)
    for bit in range(8):
        nib[..., bit] = (w4 >> (bit * 4)) & 0xF
    sc = scale4[:, :, :, None, None].astype(np.float32)
    bi = bias4[:, :, :, None, None].astype(np.float32)
    dequant = nib.astype(np.float32) * sc + bi
    mn = dequant.reshape(E, R, groups_per_row, -1).min(axis=-1)
    mx = dequant.reshape(E, R, groups_per_row, -1).max(axis=-1)
    sc2 = (mx - mn) / 3.0
    bi2 = mn
    q = ((dequant - bi2[..., None, None]) / sc2[..., None, None]).round()
    np.clip(q, 0, 3, out=q); q = q.astype(np.uint8)
    del dequant
    q = q.reshape(E, R, groups_per_row, 4, 16)
    packed = (q[..., 0].astype(np.uint32) |
              (q[..., 1].astype(np.uint32) << 2) |
              (q[..., 2].astype(np.uint32) << 4) |
              (q[..., 3].astype(np.uint32) << 6))
    out_w[:] = packed.reshape(E, R, C // 2)
    out_s[:] = sc2.astype(np.float16)
    out_b[:] = bi2.astype(np.float16)

def convert_layer(data):
    E = NUM_EXPERTS
    gate_w4 = np.frombuffer(data[GATE_W:GATE_W + E*1_572_864], dtype=np.uint32).reshape(E, 1024, 384)
    gate_s4 = np.frombuffer(data[GATE_S:GATE_S + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    gate_b4 = np.frombuffer(data[GATE_B:GATE_B + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    up_w4   = np.frombuffer(data[UP_W:UP_W   + E*1_572_864], dtype=np.uint32).reshape(E, 1024, 384)
    up_s4   = np.frombuffer(data[UP_S:UP_S   + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    up_b4   = np.frombuffer(data[UP_B:UP_B   + E*98_304], dtype=np.float16).reshape(E, 1024, 48)
    down_w4 = np.frombuffer(data[DOWN_W:DOWN_W + E*1_572_864], dtype=np.uint32).reshape(E, 3072, 128)
    down_s4 = np.frombuffer(data[DOWN_S:DOWN_S + E*98_304], dtype=np.float16).reshape(E, 3072, 16)
    down_b4 = np.frombuffer(data[DOWN_B:DOWN_B + E*98_304], dtype=np.float16).reshape(E, 3072, 16)
    gate_w2 = np.zeros((E, 1024, 192), dtype=np.uint32); gate_s2 = np.zeros((E, 1024, 48), dtype=np.float16); gate_b2 = np.zeros((E, 1024, 48), dtype=np.float16)
    up_w2   = np.zeros((E, 1024, 192), dtype=np.uint32); up_s2   = np.zeros((E, 1024, 48), dtype=np.float16); up_b2   = np.zeros((E, 1024, 48), dtype=np.float16)
    down_w2 = np.zeros((E, 3072, 64), dtype=np.uint32); down_s2 = np.zeros((E, 3072, 16), dtype=np.float16); down_b2 = np.zeros((E, 3072, 16), dtype=np.float16)
    quantize_proj(gate_w4, gate_s4, gate_b4, gate_w2, gate_s2, gate_b2, R=1024, C=384, groups_per_row=48)
    quantize_proj(up_w4, up_s4, up_b4, up_w2, up_s2, up_b2, R=1024, C=384, groups_per_row=48)
    quantize_proj(down_w4, down_s4, down_b4, down_w2, down_s2, down_b2, R=3072, C=128, groups_per_row=16)
    parts = []
    for ei in range(E):
        parts.append(gate_w2[ei].tobytes()); parts.append(gate_s2[ei].tobytes()); parts.append(gate_b2[ei].tobytes())
        parts.append(up_w2[ei].tobytes());   parts.append(up_s2[ei].tobytes());   parts.append(up_b2[ei].tobytes())
        parts.append(down_w2[ei].tobytes()); parts.append(down_s2[ei].tobytes()); parts.append(down_b2[ei].tobytes())
    return b"".join(parts)

model_dir = Path("~/models/flash-moe/Qwen3.5-122B-A10B-4bit").expanduser() / "packed_experts"
out_path = Path("/Volumes/Seagate Backup Plus Drive/flash-moe-2bit/packed_experts_2bit_ssd.bin")

# Find how many layers already done
size = out_path.stat().st_size
layer_size = NUM_EXPERTS * EXPERT_SIZE_2BIT  # 679,477,248
layers_done = (size - 16) // layer_size
start_layer = layers_done + 1 if (size - 16) % layer_size != 0 else layers_done
remaining = 48 - start_layer
print(f"Resume: {start_layer}/48 layers done, {remaining} remaining")
print(f"File: {size/1024**3:.1f}GB, Expected: {30.5:.1f}GB")

start = time.time()
with open(out_path, "ab") as fout:
    for li in range(start_layer, 48):
        t0 = time.time()
        data = open(model_dir / f"layer_{li:02d}.bin", "rb").read()
        fout.write(convert_layer(data))
        t1 = time.time()
        done = li + 1
        elapsed = time.time() - start
        total_done = done * NUM_EXPERTS
        eta = (elapsed / total_done) * (48 * NUM_EXPERTS - total_done)
        print(f"  L{li:02d}: {done}/48 ({100*done/48:.0f}%) {t1-t0:.1f}s ETA={eta/60:.0f}min")

size = out_path.stat().st_size
print(f"Done! {size/1024**3:.1f}GB in {(time.time()-start)/60:.0f}min")
