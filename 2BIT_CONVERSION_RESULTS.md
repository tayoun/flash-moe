# Flash-MOE 2-bit Conversion Results (2026-03-30)

## 1) Expected 2-bit layout from `infer.m` / `dequant_matvec_2bit_qwen`

From `compute_expert_offsets()` in `metal_infer/infer.m` (Qwen3.5-MoE dims: `hidden_dim=3072`, `moe_intermediate=1024`, `group_size=64`, `bits=2`):

- `gate_w`: `1024 x (3072/16) x 4 = 786,432`
- `gate_s`: `1024 x (3072/64) x 2 = 98,304`
- `gate_b`: `98,304`
- `up_w`: `786,432`
- `up_s`: `98,304`
- `up_b`: `98,304`
- `down_w`: `3072 x (1024/16) x 4 = 786,432`
- `down_s`: `3072 x (1024/64) x 2 = 98,304`
- `down_b`: `98,304`

**Expected per-expert 2-bit size: `2,949,120` bytes**.

2-bit offsets used by the fixed converter:
- `gate_w=0`
- `gate_s=786,432`
- `gate_b=884,736`
- `up_w=983,040`
- `up_s=1,769,472`
- `up_b=1,867,776`
- `down_w=1,966,080`
- `down_s=2,752,512`
- `down_b=2,850,816`

## 2) 4-bit source file format verification

Source: `/Volumes/Seagate Backup Plus Drive/packed_experts_ssd.bin`

Header bytes parse as:
- `>IIII = (48, 256, 0, 5308416)`
- Equivalent `>IIQ = (48, 256, 5308416)`

So the file is `48 layers x 256 experts x 5,308,416 bytes/expert`, with 16-byte header.

Important endianness detail found during testing:
- Header is **big-endian**.
- Expert payload values are interpreted correctly as **little-endian** words (`u32` weights, `bf16` scales/biases), matching how `infer.m`/Metal consume them in memory.

## 3) Fixed converter implemented

Created:
- `/Users/tayoun/projects-external/flash-moe/metal_infer/convert_4bit_to_2bit_fixed.py`

What it does:
1. Reads source header (`>IIII`).
2. Reads one or more experts from layer/expert index.
3. Dequantizes 4-bit affine (`q*scale+bias`) using 64-value groups.
4. Re-quantizes to 2-bit per 64-value group.
5. Writes 2-bit expert layout matching `dequant_matvec_2bit_qwen` offsets.
6. Emits `bf16` scales/biases and little-endian `u32` packed weights.

## 4) One-expert conversion test

Command:

```bash
/Users/tayoun/projects-external/flash-moe/metal_infer/convert_4bit_to_2bit_fixed.py \
  --layer 0 --expert 0 --count 1 \
  --out /Users/tayoun/projects-external/flash-moe/metal_infer/one_expert_2bit.bin \
  --overwrite
```

Observed output:
- Source header recognized: `layers=48 experts=256 reserved=0 expert_size=5308416`
- Converted expert size: **`2,949,120` bytes**
- Output file: `/Users/tayoun/projects-external/flash-moe/metal_infer/one_expert_2bit.bin`

## 5) Can old Danveloper 2-bit file be fixed by header-only patch?

Checked file:
- `/Volumes/Seagate Backup Plus Drive/flash-moe-2bit/packed_experts_2bit_ssd.bin`
- Header: `(48, 256, 2,654,208)`
- Actual size: `33,218,887,696`

Consistency checks:
- Header-implied size should be `16 + 48*256*2,654,208 = 32,614,907,920` (does **not** match actual).
- Payload (`size-16`) is exactly divisible by `2,949,120`: `11,264 experts = 44 layers x 256`.

Conclusion:
- This file is **not** fixable by changing only `expert_size` in the header.
- It is structurally inconsistent with its own header and appears to contain only 44 layers worth of 2,949,120-byte experts.
- Re-conversion from the 4-bit source is required for a correct full 48-layer artifact.

## 6) Remaining work

- Run full conversion over all `48 x 256` experts to produce a complete packed 2-bit SSD file with header `>IIQ = (48, 256, 2,949,120)`.
- Then run end-to-end inference quality/speed validation with `--2bit`.
