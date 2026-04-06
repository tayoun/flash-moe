# Gemma 4 Expert Layout

## Source Layout (from safetensors)

Gemma 4 uses a **routed MoE** with 128 experts per layer. Each expert consists of:

| Component | Shape | Dtype | Bytes/Expert | Description |
|-----------|-------|-------|-------------|-------------|
| `gate_up_proj` | [128, **1408**, 2816] (per-expert) | BF16 | 7,929,856 | Fused gate+up: `[2×moe_inter, hidden]`. Gate uses GELU activation. |
| `down_proj` | [128, 2816, **704**] (per-expert) | BF16 | 3,964,928 | Down projection: `[hidden, moe_inter]` |

**Per-expert sizes:**
- `gate_up_proj`: 2 × 704 × 2816 × 2 = **7,929,856 bytes**
- `down_proj`: 2816 × 704 × 2 = **3,964,928 bytes**
- **Total per expert: 11,894,784 bytes**

**Total per layer (128 experts):** 128 × 11,894,784 = **1,522,052,352 bytes ≈ 1.42 GB**

**Router (shared, not per-expert, not repacked):**
- `router.proj.weight`: [128, 2816] BF16 — shared across all experts (matvec + softmax for routing)
- `router.per_expert_scale`: [128] BF16 — per-expert scaling
- `router.scale`: [2816] BF16 — output RMSNorm

## Packed Binary Layout

```
┌──────────────────────────────────────┐
│         Layer N binary file           │
├──────────────┬───────────────────────┤
│ gate_up_proj │      down_proj        │
│  [128 ×      │  [128 ×              │
│   1408×2816] │   2816×704]           │
├──────────────┴───────────────────────┤
│ Expert 0: 0 … 11,894,783            │
│ Expert 1: 11,894,784 … 23,789,567   │
│ …                                    │
│ Expert 127: 1,510,157,568 … 1,522,052,351 │
└──────────────────────────────────────┘
```

- **expert_size** = 11,894,784 bytes (constant, all 128 experts identical size)
- **expert_stride** = 11,894,784 (contiguous, no padding between experts)
- **layer_size** = 128 × 11,894,784 = 1,522,052,352 bytes

## Component Offsets (within each expert)

| Component | Offset | Size |
|-----------|--------|------|
| `gate_up_proj` | 0 | 7,929,856 |
| `down_proj` | 7,929,856 | 3,964,928 |

## Comparison: Gemma vs Qwen Expert Layout

| Property | Gemma 4 | Qwen 3.5 MoE |
|----------|---------|---------------|
| Experts/layer | 128 | 512 |
| Expert size | 11,894,784 B | 7,077,888 B |
| Layer size | 1.42 GB | 3.44 GB |
| Components | gate_up (fused) + down | gate + up + down (all separate) |
| Routing | matvec + softmax | top-k gating |
| Storage dtype | BF16 | U4 (4-bit quantized) |

**Note:** Qwen uses 4-bit quantization (U32 packed) for experts; Gemma uses BF16 directly.

## Runtime Reader (infer.m)

The runtime reads packed experts using:
```c
// Per-expert offsets within a layer binary:
size_t gate_offset = 0;                           // gate_up_proj start
size_t down_offset = GEMMA_EXPERT_GATE_SIZE;    // down_proj start

// For expert e:
float *expert_gate = layer_bin + (size_t)e * expert_size + gate_offset;
float *expert_down = layer_bin + (size_t)e * expert_size + down_offset;
```

**Config values for Gemma 4:**
- `NUM_EXPERTS = 128`
- `NUM_EXPERTS_PER_TOK = 8`
- `MOE_INTERMEDIATE = 704`
- `GEMMA_EXPERT_GATE_SIZE = 7929856`
- `GEMMA_EXPERT_DOWN_SIZE = 3964928`
- `GEMMA_EXPERT_SIZE = 11894784`
