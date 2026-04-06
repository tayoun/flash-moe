# Expert Streaming Gemma 4 26B-A4B — Implementation Plan

**Goal:** Implement SSD-streamed expert inference for Gemma 4 26B-A4B in the `flash-moe` engine, borrowing the native slot-bank pattern from the Anemll llama.cpp fork.
**Spec:** Architecture at `g4.si5.pl` + slot-bank porting guide from `anemll/anemll-flash-llama.cpp`
**Tech Stack:** Objective-C/C + Metal, Python safetensors tooling
**Tasks:** 20 tasks across 4 waves
**Effort:** human: ~2-3h / agent: ~300-400 min
**Diagrams:** component, data flow, state, sequence

---

## Architecture

### Gemma 4 26B-A4B — Key Numbers

| Parameter | Value |
|-----------|-------|
| Total params | ~26B |
| Active params | ~4B |
| Hidden size | 2,816 |
| Layers | 30 (5 sliding + 1 full pattern, last layer always full) |
| Sliding attention | head_dim=256, 16 Q heads, 8 KV heads, theta=10K, full rotation |
| Full attention | head_dim=512, 16 Q heads, 2 KV heads, theta=1M, p-RoPE=0.25, K=V sharing |
| Dense FFN | GeGLU, hidden=2,112 (all layers, always-on) |
| MoE | 128 experts, top-8 routing, expert hidden=704 |
| Router | RMSNorm(no scale) → scale×1/√hidden → Linear→num_experts → softmax → top-k → per-expert scale |
| Output combine | 1/√2 × (dense_out + moe_out), then post-norm |
| Vocab | 262,144 |
| Context | 256K |
| Logit cap | tanh(x/30)×30 |

### Expert Storage Layout (per layer)

Each expert lives in a packed binary at `packed_experts/gemma4/layer_XX.bin`:
```
gate_up_proj: 2,816 × 704 BF16  (gate + up concatenated, 2,816×704 each)
down_proj:    704 × 2,816 BF16
```
128 experts × (2,816×704×2 BF16 + header) ≈ 1.3 GB per layer on disk.
30 layers × 1.3 GB ≈ 39 GB packed on SSD.

### Parallel FFN + MoE Data Flow

```
pre-FFN residual (RMSNorm)
    │
    ├──▶ DENSE PATH ───────────────────────────────────────────────────────┐
    │    gate_proj (2816×2112 BF16) → GeGLU                               │
    │    up_proj   (2816×2112 BF16) ──→ element-wise × ──▶ down_proj      │
    │                                                          │          │
    │                                                     post_norm       │
    │                                                          │          │
    └────────────────────────────────────────────────────────────┴──────────┤
                                                                         ▼
pre-FFN residual ──▶ ROUTER ──▶ top-k expert ids + weights                 │
                       │                                                   │
                       ▼                                                   │
              MoE PARALLEL PATH ──────────────────────────────────────────┤
              (only top-8 experts loaded)                                 │
              for each selected expert e:                                  │
                  gate_up_w[e] → GeGLU ──▶ down_w[e]                      │
                  weighted by top-k weight                               │
                                                     post_norm (shared)   │
                                                          │               │
                             FINAL: residual + post_norm(dense_out + moe_out)
```

### Native Slot-Bank Pattern (borrowed from llama.cpp)

The key insight from the llama.cpp porting guide: instead of routing to global expert IDs (0-127),
the runtime maps selected expert IDs → **slot IDs** within a fixed-size bank.

```
top-k expert ids:       [e5, e42, e17, e87, e3, e99, e22, e55]  (true expert IDs)
slot ids:               [0, 1, 2, 3, 4, 5, 6, 7]              (local to bank)
```

**Why this matters:** The matmul consumer only sees slot IDs (consecutive 0-N).
No gaps, no global expert lookup. Miss handling (Oracle replay / temporal prefetch)
is slot-id-based, not expert-id-based.

**Key runtime questions (from llama.cpp guide):**
- Where are routed top-k expert IDs produced? → `gemma_layer_forward`, router forward
- Which routed tensors consume those IDs? → expert matvecs (gate_up, down)
- Which consumers need true expert IDs (not slot IDs)? → gating-weight lookup, per-expert scale
- Are gate/up/down aligned to same expert-id contract? → YES for Gemma (gate_up_proj is one fused tensor)

---

## Waves

### Wave 1 — Foundation (model config + layer classification + Python export)
*Independent — all parallel*

**Task 1: Gemma ModelConfig struct and accessors**
- Add `ModelConfigGemma` with all Gemma 26B-A4B constants
- Layer type classification: `is_full_attention_layer(i)`, `is_sliding_attention_layer(i)`
- Per-layer attention geometry: `head_dim_for_layer(i)`, `kv_heads_for_layer(i)`, `rope_theta_for_layer(i)`, `pRoPE_factor_for_layer(i)`
- MoE constants: `num_experts=128`, `top_k=8`, `expert_hidden=704`, `dense_ffn_hidden=2112`
- `cfg_is_gemma()`

**Task 2: Dual-path RMSNorm — Q/K norm vs V-norm**
- Implement `cpu_rms_norm_with_scale(float *x, const uint16_t *w_bf16, float *out, int dim, float eps)` — Q/K norms (learned scale)
- Implement `cpu_rms_norm_no_scale(float *x, float *out, int dim, float eps)` — V-norm (no learned params)
- Both BF16 weight variants; V-norm is pure F32

**Task 3: Gemma weight extraction in Python**
- Read `config.json` from Gemma 4 26B-A4B checkpoint
- Extract `model.language_model.*` text tensors (not vision_tower)
- Dense weights: q_proj, k_proj, v_proj (or K=V), o_proj, gate_up_proj, down_proj
- Layer norm weights: input_norm, post_attn_norm, pre_feedforward_layernorm, post_feedforward_layernorm, post_feedforward_layernorm_1, post_feedforward_layernorm_2
- Q/K/V norm weights: q_norm, k_norm, v_norm
- Router weights: gate (2816×128), per_expert_scale (128)
- Per-layer scalars
- Output: `model_weights.bin/json` for runtime loading

**Task 4: Gemma tokenizer + chat template export**
- Export Gemma tokenizer with correct special token IDs
- Implement Gemma chat template (turn markers, bos/eos handling)
- Export vocab bin for runtime

### Wave 2 — Attention path + dense FFN (CPU bring-up)
*Depends on Wave 1*

**Task 5: Gemma attention — Q/K projection + QK norm**
- Q projection: BF16 matvec, then per-head RMS norm with learned weight (q_norm)
- K projection: BF16 matvec, then per-head RMS norm with learned weight (k_norm)
- K=V sharing for full attention layers: skip V projection, clone K as V
- V norm (no scale): per-head RMS norm without learned weight

**Task 6: Gemma RoPE — standard + p-RoPE**
- Sliding layers: standard RoPE, theta=10K, full rotation (all dims)
- Full layers: p-RoPE, theta=1M, partial=0.25 — only top 25% of dims rotated, rest cos=1, sin=0
- Configurable per-layer via accessor functions

**Task 7: KV cache — per-layer geometry + store**
- Per-layer KV cache sizing based on layer type
- Store K and V in cache (V = K when K=V sharing)
- GQA grouping: each Q head group attends to one KV head

**Task 8: CPU attention forward + O projection**
- Attention scores: Q @ K^T (no QK scaling — Gemma uses QK norm)
- Softmax, weighted V sum, O projection (BF16 matvec)
- Sliding window support (if pos > window, start from pos-window)

**Task 9: Dense FFN — GeGLU path**
- gate_proj BF16 matvec → up_proj BF16 matvec
- GeGLU: `out = gelu(gate) * up`, where gelu ≈ `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³)))`
- down_proj BF16 matvec
- Post-norm on FFN output

**Task 10: Full gemma_layer_forward (dense + dense parallel)**
- Wire: input RMS norm → Q/K/V → QK norm + V norm → RoPE → attention → O proj → residual add → post-attn RMS norm
- → pre-FFN RMS norm → dense GeGLU path
- → residual + 1/√2 × (dense_out + moe_out) → post-norm
- Layer scalar applied at end
- Dispatch between sliding and full attention geometry per layer

### Wave 3 — MoE + Expert Streaming
*Depends on Wave 2*

**Task 11: Gemma router implementation**
- RMSNorm (no learned scale) on pre-FFN hidden
- Scale × 1/√hidden_size
- matvec: W_gate (2816×128, BF16) → expert_scores (128)
- softmax over expert_scores → probs
- top-k (k=8): indices + weights
- Per-expert scale multiply: `weights[e] *= per_expert_scale[e]`
- Normalize: `weights /= weights.sum()`
- Returns: `top_k_ids[8]`, `top_k_weights[8]`

**Task 12: Expert binary layout + per-layer expert index**
- Define `packed_experts/gemma4/layer_XX.bin` layout per spec above
- Build expert index: for each layer, byte offsets for each expert's gate_up and down
- `tools/gemma4/build_expert_index.py` — scan safetensors, produce `expert-index.json`
- Expert index fields: `layer`, `expert_idx`, `gate_up_offset`, `gate_up_size`, `down_offset`, `down_size`

**Task 13: Expert loading via pread (slot-bank aligned)**
- `load_expert_slot(layer_idx, slot_id, expert_idx)` — reads one expert's gate_up + down from packed bin
- Uses `pread()` for SSD-friendly offset reads
- Bank size: 8 (max active experts) — preallocate slot storage
- Miss handling: if expert not in bank, load from packed source

**Task 14: MoE forward (top-8 expert routing + GeGLU + weighted accum)**
- Given top_k_ids[8], top_k_weights[8], pre-FFN residual
- For each slot s in 0..7:
    - Load expert e = top_k_ids[s] into slot s via pread
    - gate_up = gate_up_w[s] (2816×704 BF16)
    - GeGLU(gate_up @ x) → intermediate
    - down_w[s] (704×2816 BF16) @ intermediate → expert_out
    - weighted: expert_out × top_k_weights[s], accumulate into moe_out[2816]
- Apply post-norm, combine with dense FFN output via 1/√2

**Task 15: gemma_moe_expert_forward — slot-bank integration**
- Takes layer_idx, pre-FFN residual, top_k_ids, top_k_weights
- Manages slot bank state: which expert is in which slot (LRU or direct map)
- Returns moe_out[2816] (already weighted and accumulated)

### Wave 4 — Integration + Validation
*Depends on Wave 3*

**Task 16: Wire gemma_layer_forward into main decode loop**
- In main decode loop, detect `cfg_is_gemma()` and call `gemma_layer_forward` instead of `fused_layer_forward`
- Pass KVCache pointer, position, hidden pointer
- Verify one-token decode produces finite logits

**Task 17: Final logit softcapping**
- After LM head projection: `logits = tanh(logits / 30.0) * 30.0`
- Gemma uses this instead of raw logits

**Task 18: One-layer CPU reference (Python vs C diff)**
- Implement `tools/gemma4/check_one_layer.py` reference: pure Python one-layer decode using safetensors
- Extract layer 0 weights, run single forward pass
- Compare logits against `infer` one-layer C output
- Must match within 1e-3 at this stage (before MoE)

**Task 19: Flash-MOE slot-bank miss handling + oracle**
- Implement miss detection: expert not in slot bank
- Miss install: load from packed expert binary via pread
- Oracle replay: record + replay top-k sequences for benchmark reproducibility
- Prefetch: next token's top-k experts prefetched in background

**Task 20: End-to-end validation — logits + throughput**
- Run `infer` on standard prompts ( Alpaca eval format)
- Verify output tokens are valid (no NaN/Inf, vocab in range)
- Benchmark: tokens/sec, first-token latency, decode speed
- Compare against reference (transformers pipeline or llama.cpp branch)

---

## Expert-Loading State Machine

```
INITIAL          → bank empty, no experts loaded
TOKEN n FORWARD  → router produces top_k_ids[8]
                  for each expert:
                    if expert in bank:
                      USE_SLOT(slot)
                    else:
                      MISS → LOAD_FROM_SSD → USE_SLOT(new_slot)
                  run GeGLU + down_proj for each slot
                  weighted accumulate → moe_out
                  combine with dense FFN → residual add → next layer
```

## Failure Modes Registry

| Task | Failure Mode | Handled? | Test? | User Sees |
|------|-------------|----------|-------|-----------|
| 3 | Missing Gemma tensor names | Partial | Yes | null bind / preflight |
| 5 | K=V sharing wrong path | Partial | Yes | attention regression |
| 6 | p-RoPE dimensions wrong | Partial | Yes | garbage full-attention output |
| 11 | Router softmax instability | No | Yes | all experts equal weight |
| 13 | pread offset error | Partial | Yes | garbage expert weights |
| 14 | Top-k weight normalization | Partial | Yes | MoE output too large |
| 18 | Python/C logits mismatch | N/A | Yes | can't validate |

## Test Plan
See: `docs/specs/gemma4-port/test-plan.md`

## READINESS GATE
═══════════════
Spec traceability:     PASS (all 20 tasks trace to g4.si5.pl architecture)
Constitution:          N/A (no CONSTITUTION.md in this repo)
Pattern consistency:   PASS (follows existing flash-moe patterns)
Ambiguity:             0 unresolved markers
Spec completeness:     PASS (parallel FFN+MoE, slot-bank, p-RoPE, K=V, logit cap all covered)

Verdict: READY
