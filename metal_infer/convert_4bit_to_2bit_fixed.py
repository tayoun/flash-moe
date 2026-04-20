#!/usr/bin/env python3
"""Convert Qwen3.5-MoE packed 4-bit experts to 2-bit experts for dequant_matvec_2bit_qwen.

Source file format:
- Header is big-endian: >IIII = (num_layers, num_experts, reserved, expert_size)
- Payload expert data is little-endian (u32 weights + bf16 scales/biases)

Per-expert 4-bit layout (bytes):
- gate_w  [1024, 384] u32
- gate_s  [1024, 48]  bf16
- gate_b  [1024, 48]  bf16
- up_w    [1024, 384] u32
- up_s    [1024, 48]  bf16
- up_b    [1024, 48]  bf16
- down_w  [3072, 128] u32
- down_s  [3072, 16]  bf16
- down_b  [3072, 16]  bf16

Per-expert 2-bit layout expected by infer.m + dequant_matvec_2bit_qwen:
- gate_w  [1024, 192] u32
- gate_s  [1024, 48]  bf16
- gate_b  [1024, 48]  bf16
- up_w    [1024, 192] u32
- up_s    [1024, 48]  bf16
- up_b    [1024, 48]  bf16
- down_w  [3072, 64]  u32
- down_s  [3072, 16]  bf16
- down_b  [3072, 16]  bf16

Output expert size: 2,949,120 bytes.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

R_GATE = 1024
R_DOWN = 3072
GROUP = 64

PACKED4_GATE = 384  # 3072 / 8
PACKED4_DOWN = 128  # 1024 / 8
PACKED2_GATE = 192  # 3072 / 16
PACKED2_DOWN = 64   # 1024 / 16

OFF_GATE_W = 0
SZ_GATE_W = R_GATE * PACKED4_GATE * 4
OFF_GATE_S = OFF_GATE_W + SZ_GATE_W
SZ_GATE_S = R_GATE * 48 * 2
OFF_GATE_B = OFF_GATE_S + SZ_GATE_S
SZ_GATE_B = R_GATE * 48 * 2

OFF_UP_W = OFF_GATE_B + SZ_GATE_B
SZ_UP_W = R_GATE * PACKED4_GATE * 4
OFF_UP_S = OFF_UP_W + SZ_UP_W
SZ_UP_S = R_GATE * 48 * 2
OFF_UP_B = OFF_UP_S + SZ_UP_S
SZ_UP_B = R_GATE * 48 * 2

OFF_DOWN_W = OFF_UP_B + SZ_UP_B
SZ_DOWN_W = R_DOWN * PACKED4_DOWN * 4
OFF_DOWN_S = OFF_DOWN_W + SZ_DOWN_W
SZ_DOWN_S = R_DOWN * 16 * 2
OFF_DOWN_B = OFF_DOWN_S + SZ_DOWN_S
SZ_DOWN_B = R_DOWN * 16 * 2

EXPERT_SIZE_4BIT = OFF_DOWN_B + SZ_DOWN_B  # 5,308,416

OFF2_GATE_W = 0
SZ2_GATE_W = R_GATE * PACKED2_GATE * 4
OFF2_GATE_S = OFF2_GATE_W + SZ2_GATE_W
SZ2_GATE_S = R_GATE * 48 * 2
OFF2_GATE_B = OFF2_GATE_S + SZ2_GATE_S
SZ2_GATE_B = R_GATE * 48 * 2

OFF2_UP_W = OFF2_GATE_B + SZ2_GATE_B
SZ2_UP_W = R_GATE * PACKED2_GATE * 4
OFF2_UP_S = OFF2_UP_W + SZ2_UP_W
SZ2_UP_S = R_GATE * 48 * 2
OFF2_UP_B = OFF2_UP_S + SZ2_UP_S
SZ2_UP_B = R_GATE * 48 * 2

OFF2_DOWN_W = OFF2_UP_B + SZ2_UP_B
SZ2_DOWN_W = R_DOWN * PACKED2_DOWN * 4
OFF2_DOWN_S = OFF2_DOWN_W + SZ2_DOWN_W
SZ2_DOWN_S = R_DOWN * 16 * 2
OFF2_DOWN_B = OFF2_DOWN_S + SZ2_DOWN_S
SZ2_DOWN_B = R_DOWN * 16 * 2

EXPERT_SIZE_2BIT = OFF2_DOWN_B + SZ2_DOWN_B  # 2,949,120


@dataclass
class SrcHeader:
    num_layers: int
    num_experts: int
    reserved: int
    expert_size: int


def parse_header(path: Path) -> SrcHeader:
    with path.open("rb") as f:
        h = f.read(16)
    if len(h) != 16:
        raise ValueError("Source file too small for 16-byte header")
    a, b, c, d = struct.unpack(">IIII", h)
    return SrcHeader(num_layers=a, num_experts=b, reserved=c, expert_size=d)


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    bits = (u16.astype(np.uint32) << 16)
    return bits.view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return (bits >> 16).astype(np.uint16)


def load_u32_le(blob: bytes, off: int, count: int) -> np.ndarray:
    return np.frombuffer(blob[off:off + count * 4], dtype="<u4").astype(np.uint32)


def load_bf16_le(blob: bytes, off: int, count: int) -> np.ndarray:
    raw = np.frombuffer(blob[off:off + count * 2], dtype="<u2").astype(np.uint16)
    return bf16_bits_to_f32(raw)


def unpack_4bit_words(words_u32: np.ndarray) -> np.ndarray:
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    q = ((words_u32[..., None] >> shifts) & 0xF).astype(np.float32)
    return q.reshape(words_u32.shape[0], words_u32.shape[1] * 8)


def quantize_to_2bit(values: np.ndarray, num_groups: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = values.shape
    assert cols == num_groups * GROUP
    vg = values.reshape(rows, num_groups, GROUP)

    mn = vg.min(axis=2)
    mx = vg.max(axis=2)
    span = mx - mn

    scale = np.where(span > 0.0, span / 3.0, 0.0).astype(np.float32)
    safe_scale = np.where(span > 0.0, scale, 1.0).astype(np.float32)

    q = np.rint((vg - mn[..., None]) / safe_scale[..., None]).clip(0, 3).astype(np.uint32)

    q16 = q.reshape(rows, num_groups, GROUP // 16, 16)
    shifts = (np.arange(16, dtype=np.uint32) * 2)[None, None, None, :]
    packed = np.bitwise_or.reduce(q16 << shifts, axis=3).reshape(rows, num_groups * (GROUP // 16))

    scale_bf16 = f32_to_bf16_bits(scale)
    bias_bf16 = f32_to_bf16_bits(mn.astype(np.float32))
    return packed, scale_bf16, bias_bf16


def convert_proj_4_to_2(
    w4_u32: np.ndarray,
    s4_f32: np.ndarray,
    b4_f32: np.ndarray,
    rows: int,
    packed4_cols: int,
    num_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w4_u32 = w4_u32.reshape(rows, packed4_cols)
    q4 = unpack_4bit_words(w4_u32).reshape(rows, num_groups, GROUP)

    vals = q4 * s4_f32.reshape(rows, num_groups, 1) + b4_f32.reshape(rows, num_groups, 1)
    vals = vals.reshape(rows, num_groups * GROUP)
    return quantize_to_2bit(vals, num_groups=num_groups)


def convert_expert(expert_blob: bytes) -> bytes:
    if len(expert_blob) != EXPERT_SIZE_4BIT:
        raise ValueError(f"Expected {EXPERT_SIZE_4BIT} bytes, got {len(expert_blob)}")

    gw = load_u32_le(expert_blob, OFF_GATE_W, R_GATE * PACKED4_GATE).reshape(R_GATE, PACKED4_GATE)
    gs = load_bf16_le(expert_blob, OFF_GATE_S, R_GATE * 48).reshape(R_GATE, 48)
    gb = load_bf16_le(expert_blob, OFF_GATE_B, R_GATE * 48).reshape(R_GATE, 48)

    uw = load_u32_le(expert_blob, OFF_UP_W, R_GATE * PACKED4_GATE).reshape(R_GATE, PACKED4_GATE)
    us = load_bf16_le(expert_blob, OFF_UP_S, R_GATE * 48).reshape(R_GATE, 48)
    ub = load_bf16_le(expert_blob, OFF_UP_B, R_GATE * 48).reshape(R_GATE, 48)

    dw = load_u32_le(expert_blob, OFF_DOWN_W, R_DOWN * PACKED4_DOWN).reshape(R_DOWN, PACKED4_DOWN)
    ds = load_bf16_le(expert_blob, OFF_DOWN_S, R_DOWN * 16).reshape(R_DOWN, 16)
    db = load_bf16_le(expert_blob, OFF_DOWN_B, R_DOWN * 16).reshape(R_DOWN, 16)

    gw2, gs2, gb2 = convert_proj_4_to_2(gw, gs, gb, R_GATE, PACKED4_GATE, 48)
    uw2, us2, ub2 = convert_proj_4_to_2(uw, us, ub, R_GATE, PACKED4_GATE, 48)
    dw2, ds2, db2 = convert_proj_4_to_2(dw, ds, db, R_DOWN, PACKED4_DOWN, 16)

    out = bytearray(EXPERT_SIZE_2BIT)

    def put_u32(arr: np.ndarray, off: int) -> None:
        out[off:off + arr.size * 4] = arr.astype("<u4", copy=False).tobytes()

    def put_u16(arr: np.ndarray, off: int) -> None:
        out[off:off + arr.size * 2] = arr.astype("<u2", copy=False).tobytes()

    put_u32(gw2, OFF2_GATE_W)
    put_u16(gs2, OFF2_GATE_S)
    put_u16(gb2, OFF2_GATE_B)

    put_u32(uw2, OFF2_UP_W)
    put_u16(us2, OFF2_UP_S)
    put_u16(ub2, OFF2_UP_B)

    put_u32(dw2, OFF2_DOWN_W)
    put_u16(ds2, OFF2_DOWN_S)
    put_u16(db2, OFF2_DOWN_B)

    if len(out) != EXPERT_SIZE_2BIT:
        raise AssertionError("Unexpected output size")
    return bytes(out)


def read_expert(path: Path, layer: int, expert: int, h: SrcHeader) -> bytes:
    idx = layer * h.num_experts + expert
    off = 16 + idx * h.expert_size
    with path.open("rb") as f:
        f.seek(off)
        blob = f.read(h.expert_size)
    if len(blob) != h.expert_size:
        raise ValueError(f"Short read: expected {h.expert_size}, got {len(blob)}")
    return blob


def write_packed_header(out_path: Path, num_layers: int, num_experts: int) -> None:
    with out_path.open("wb") as f:
        f.write(struct.pack(">IIQ", num_layers, num_experts, EXPERT_SIZE_2BIT))


def append_expert(out_path: Path, expert_2bit: bytes) -> None:
    with out_path.open("ab") as f:
        f.write(expert_2bit)


def main() -> int:
    p = argparse.ArgumentParser(description="Convert Flash-MOE Qwen 4-bit expert(s) to 2-bit format")
    p.add_argument("--src", default="/Volumes/Seagate Backup Plus Drive/packed_experts_ssd.bin")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--count", type=int, default=1, help="Number of consecutive experts to convert")
    p.add_argument("--out", default="/Users/tayoun/projects-external/flash-moe/metal_infer/one_expert_2bit.bin")
    p.add_argument("--packed", action="store_true", help="Write packed file with >IIQ header, appending converted experts")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    h = parse_header(src)
    print(f"[src] {src}")
    print(f"[src] header >IIII: layers={h.num_layers} experts={h.num_experts} reserved={h.reserved} expert_size={h.expert_size}")

    if h.expert_size != EXPERT_SIZE_4BIT:
        raise ValueError(f"Unexpected 4-bit expert_size={h.expert_size}, expected {EXPERT_SIZE_4BIT}")

    if not (0 <= args.layer < h.num_layers):
        raise ValueError("layer out of range")
    if not (0 <= args.expert < h.num_experts):
        raise ValueError("expert out of range")
    if args.count < 1:
        raise ValueError("count must be >=1")

    end_expert = args.expert + args.count - 1
    if end_expert >= h.num_experts:
        raise ValueError("count exceeds experts in layer")

    if out.exists() and not args.overwrite and not args.packed:
        raise FileExistsError(f"Output exists: {out}. Use --overwrite to replace")

    if args.packed:
        if out.exists() and args.overwrite:
            out.unlink()
        if not out.exists():
            write_packed_header(out, 1, args.count)
    else:
        if args.count != 1:
            raise ValueError("Non-packed mode supports exactly one expert; use --packed for multiple")

    for i in range(args.count):
        e = args.expert + i
        blob4 = read_expert(src, args.layer, e, h)
        blob2 = convert_expert(blob4)
        if len(blob2) != EXPERT_SIZE_2BIT:
            raise AssertionError("Converted expert size mismatch")

        if args.packed:
            append_expert(out, blob2)
        else:
            out.write_bytes(blob2)

        print(f"[ok] layer={args.layer} expert={e} -> {len(blob2)} bytes")

    if args.packed:
        size = out.stat().st_size
        print(f"[out] {out} size={size} bytes (header + {args.count} experts)")
    else:
        size = out.stat().st_size
        print(f"[out] {out} size={size} bytes")

    print(f"[expect] expert_size_2bit={EXPERT_SIZE_2BIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
