# Phase 0 — 122B Validation Report

## 1. Architecture Verification

**Config:** `/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit/config.json`

| Parameter | Expected | Actual | Match |
|---|---|---|---|
| num_hidden_layers | 48 | 48 | YES |
| num_experts | 256 | 256 | YES |
| hidden_size | 3072 | 3072 | YES |
| moe_intermediate_size | 1024 | 1024 | YES |
| num_experts_per_tok | 8 | 8 | YES |
| num_attention_heads | 32 | 32 | YES |
| num_key_value_heads | 2 | 2 | YES |
| head_dim | 256 | 256 | YES |
| shared_expert_intermediate_size | 1024 | 1024 | YES |
| linear_num_value_heads | — | 64 | NEW (35B has 32) |
| linear_num_key_heads | — | 16 | SAME |
| vocab_size | 248320 | 248320 | YES |
| full_attention_interval | 4 | 4 | YES (from layer_types) |
| quantization | 4-bit, gs=64 | 4-bit, gs=64 | YES |

**Key differences from 35B:** hidden_size (2048→3072), num_hidden_layers (40→48),
num_attention_heads (16→32), moe_intermediate_size (512→1024),
shared_expert_intermediate_size (512→1024), linear_num_value_heads (32→64),
num_experts_per_tok (8 vs 35B's 8 — same).

## 2. build_expert_index_35b.py Compatibility

**Result: WORKS UNMODIFIED on 122B.**

- Primary regex patterns (`language_model.model.layers.X.mlp.switch_mlp.*`) match all 432 expert tensor keys
- All 48 layers detected, range 0–47
- 9 components per layer × 48 layers = 432 matches (100%)
- No fallback patterns needed

## 3. Per-Expert Byte Size (from safetensors metadata)

122B expert tensor shapes (per layer, packed with 256 experts on dim 0):

| Component | Shape | dtype | Total size | Per-expert |
|---|---|---|---|---|
| gate_proj.weight | [256, 1024, 384] | U32 | 402,653,184 | 1,572,864 |
| gate_proj.scales | [256, 1024, 48] | BF16 | 25,165,824 | 98,304 |
| gate_proj.biases | [256, 1024, 48] | BF16 | 25,165,824 | 98,304 |
| up_proj.weight | [256, 1024, 384] | U32 | 402,653,184 | 1,572,864 |
| up_proj.scales | [256, 1024, 48] | BF16 | 25,165,824 | 98,304 |
| up_proj.biases | [256, 1024, 48] | BF16 | 25,165,824 | 98,304 |
| down_proj.weight | [256, 3072, 128] | U32 | 402,653,184 | 1,572,864 |
| down_proj.scales | [256, 3072, 16] | BF16 | 25,165,824 | 98,304 |
| down_proj.biases | [256, 3072, 16] | BF16 | 25,165,824 | 98,304 |
| **TOTAL** | | | **1,358,954,496** | **5,308,416** |

Per-expert: **5,308,416 bytes (5.06 MB)** — 3× larger than 35B's 1,769,472 bytes (1.69 MB).

Cross-check: `compute_expert_offsets()` in infer.m will derive the same value from
mid=1024, hid=3072, gs=64, bits=4: 3×1,572,864 + 6×98,304 = 5,308,416. ✓

## 4. Hard-Coded 35B Assumptions — Status

**All scripts are already generalized** (fixes applied in prior commits):

### extract_weights_35b.py — ALREADY GENERIC
- Reads all config from model's `config.json` (no hardcoded dimensions)
- Layer types read from `config.json` `layer_types` array
- No layer count validation — derives from config

### repack_experts_35b.py — ALREADY GENERIC
- Uses `derive_layout()` which reads all sizes from expert index JSON
- No hardcoded expert sizes, layer counts, or expert counts

### infer.m — ALREADY GENERIC
- `load_model_config()` reads everything from `config.json`
- K defaults to `-1` → `config.num_experts_per_tok` (dynamic)
- Default path search tries `out_122b` before `out_35b` (auto-detect)
- All buffer allocations use `cfg.*` dimensions

### build_expert_index_35b.py — ALREADY GENERIC
- Regex patterns match 122B tensors unmodified

## 5. Disk Space Budget

| Item | Size |
|---|---|
| packed_experts/ (48 layers × 1.36 GB) | **65.23 GB** |
| model_weights.bin (non-expert tensors) | **3.46 GB** |
| Total output needed | **~68.7 GB** |
| Current safetensor shards (input) | 69.59 GB |
| Free disk space | ~6.1 GB |

## 6. Incremental Repacking Feasibility

**FEASIBLE with `--delete-consumed-shards`, but tight.**

Expert layer distribution across shards (14 shards):
- Shard 01: layers 0, 1, 2
- Shard 02: layers 2, 3, 4, 5, 6
- Shard 03: layers 6, 7, 8, 9
- ...pattern continues with 1-layer overlaps at shard boundaries...
- Shard 14: layers 46, 47

**Strategy:** Process layers in order. After all layers from a shard are packed, delete
that shard. Adjacent shards share 1 overlapping layer (e.g., layer 2 has components in
both shard 1 and shard 2), so a shard can only be deleted after the overlapping layer
is fully processed.

**Budget walkthrough (first 2 shards):**
- Start: 6.1 GB free, all 14 shards on disk
- Pack layers 0, 1 → -2.72 GB → 3.38 GB free
- Pack layer 2 (needs shards 1+2) → -1.36 GB → 2.02 GB free
- Delete shard 1 → +5.12 GB → **7.14 GB free**
- Pack layers 3, 4, 5 → -4.08 GB → 3.06 GB free
- Pack layer 6 (needs shards 2+3) → -1.36 GB → 1.70 GB free
- Delete shard 2 → +5.20 GB → **6.90 GB free**

Pattern repeats. **Minimum free space: ~1.7 GB** at worst point. Feasible but requires:
1. `--delete-consumed-shards` flag in repack script
2. Processing in strict layer order
3. Tracking which shards are fully consumed
4. Running `model_weights.bin` extraction FIRST (before deleting any shards), since
   non-expert tensors span all shards

**Recommended order:**
1. Extract `model_weights.bin` first (3.46 GB) — needs all shards, but 6.1 GB free is enough
2. Then incremental repack with shard deletion

## 7. Phase 1 — Build & Verify Results

### Build
- `make infer` succeeds (10 warnings, 0 errors)
- No shader changes needed — existing Metal shaders handle variable dimensions

### First-token generation
```
./infer --model .../Qwen3.5-122B-A10B-4bit/ --weights out_122b/model_weights.bin \
  --manifest out_122b/model_weights.json --vocab vocab_122b.bin --k 8 --tokens 10 \
  --prompt "What is the capital of France?"
```

**Output:** "The capital of France is Paris." — factually correct.

| Metric | Value |
|---|---|
| TTFT (cold) | 10,294 ms |
| Sustained decode | 1.16 tok/s |
| Expert layers loaded | 48/48 |
| Expert size (4-bit) | 5,308,416 bytes |
| Non-expert weights | 3.46 GB |
| Config-driven dims | hidden=3072, heads=32, kv=2, experts=256, K=8 |

### Notes
- 35B model not available on this machine for regression test, but code change is
  trivial (comment-only in infer.m, docstring in extract_weights_35b.py)
- All allocations are config-driven; no 35B-specific code paths remain
