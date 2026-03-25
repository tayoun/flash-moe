# Batch 5 Experiment Plan — Prefill Optimization Suite

## Motivation

Cold TTFT on 122B is **16.2s** (K=8). Prefill processes 14 tokens at ~1s each, sequentially.
Each token triggers full expert routing: 8 experts × 48 layers × 5.3MB = ~2GB of SSD reads.
For a 14-token prompt, that's ~28GB of SSD reads just for prefill.

Decode-side optimizations (Batches 1–4) improve sustained tok/s.
This batch targets TTFT independently, exploiting the insight that **prefill quality
requirements are different from decode quality requirements.** The model builds its
internal state (KV cache + delta-net state) during prefill — this state just needs to
be "good enough" to prime decode. We can trade prefill fidelity for speed.

## Architecture Reference

122B has two layer types with fundamentally different memory roles:
- **36 linear attention layers** (delta-net): recurrent state, no KV cache
- **12 full attention layers**: KV cache — this is where prompt information gets
  written most critically into long-range memory

This asymmetry is exploitable: linear layers may tolerate aggressive expert reduction
during prefill while full attention layers need higher fidelity.

---

## Current Prefill Behavior

```
Prompt: 14 tokens
Prefill time: 14,761 ms (first: 4,813 ms, rest avg: 829 ms)
TTFT: 16,188 ms (prefill + lm_head)

Per-token prefill:
  1. Embed token
  2. For each of 48 layers:
     a. Run attention (linear or full)
     b. Route experts (softmax + topK)
     c. pread K=8 experts from SSD (~42 MB per layer)
     d. GPU expert forward pass
  3. Repeat for next token
```

Each prefill token is processed identically to a decode token. No batching, no
expert policy differentiation, no chunk processing.

---

## Experiment 1: Shared-Expert-Only Prefill (K=0 Bound)

**Priority: FIRST — establishes theoretical TTFT floor**

### What
Skip ALL routed experts during prefill. Only the shared expert (always active,
already in memory) runs. This eliminates 100% of SSD reads during prefill.

### Implementation
- Add `--prefill-k 0` flag
- During prefill phase: skip topK routing, skip expert pread, run shared expert only
- At first decode token: switch to full K=8 routing
- Delta-net state and KV cache are still populated (from attention + shared expert)

### Expected
- TTFT: ~2-4s (down from 16s) — eliminates the ~14s of SSD expert reads
- Decode quality: unknown — may degrade if KV cache state is too impoverished
- This is a BOUND, not necessarily a shippable config

### Metric
- TTFT reduction
- Decode quality on 10-prompt suite (compare first 64 tokens of decode)
- If decode quality is acceptable: this alone is a massive win

---

## Experiment 2: Prefill-K Flag (--prefill-k N)

**Priority: SECOND — finds the quality/speed sweet spot**

### What
Separate K for prefill vs decode. Keep decode at K=8 (full quality),
reduce prefill to K=N where N < 8.

### Implementation
- Add `--prefill-k N` flag (default: same as --k)
- During prefill: use prefill-K for expert routing
- At first decode token: switch to --k

### Test Matrix
| prefill-k | Decode K | Expected TTFT | SSD Reduction |
|-----------|----------|--------------|---------------|
| 0 | 8 | ~2-4s | 100% |
| 2 | 8 | ~6-8s | 75% |
| 4 | 8 | ~8-10s | 50% |
| 6 | 8 | ~10-12s | 25% |
| 8 | 8 | 16.2s (baseline) | 0% |

### Quality Gate
For each prefill-K: generate 64 decode tokens, compare against prefill-K=8 baseline.
Focus on first-sentence coherence and factual accuracy.

---

## Experiment 3: Full-Attention-Only Experts During Prefill

**Priority: THIRD — architecturally motivated, likely best quality/speed ratio**

### What
During prefill, only run routed experts on the 12 full-attention layers.
The 36 linear-attention layers use shared expert only (K=0).

### Rationale
Full-attention layers write to KV cache — the persistent memory the model uses
for long-range context. These layers need accurate expert computation to build
a good KV cache.

Linear-attention layers use delta-net (recurrent state). The recurrent state is
less sensitive to per-token expert accuracy because:
- It's a running average (decayed), so individual token errors get diluted
- Linear attention is inherently lower-resolution than full attention
- The shared expert provides a reasonable baseline signal

### Implementation
- Add `--prefill-policy full-attn-only` flag
- During prefill: check `cfg.is_full_attn[layer]`
  - Full attention layer: route K=8 experts (normal)
  - Linear attention layer: skip routing, use shared expert only
- During decode: all layers use K=8 (normal)

### Expected
- SSD reads during prefill: reduced by 75% (only 12/48 layers load experts)
- TTFT: ~4-6s (estimated)
- Quality: likely the best among reduced-prefill experiments
  (preserving the most important expert computations)

### Variants
- `full-attn-only`: K=8 on full, K=0 on linear
- `full-attn-priority`: K=8 on full, K=2 on linear (compromise)

---

## Experiment 4: Batched Prefill

**Priority: FOURTH — changes execution model, higher effort**

### What
Stop processing prefill tokens one at a time. Batch them into chunks and
process each chunk as a unit.

### Current Behavior
```
for token in prompt:
    embed(token)
    for layer in layers:
        attention(token)         # sequential
        route_experts(token)     # one routing decision
        pread_experts(token)     # one SSD read set
        expert_forward(token)    # one GPU dispatch
```

### Batched Behavior
```
for chunk in prompt.chunks(chunk_size):
    embed_batch(chunk)                    # batch embed
    for layer in layers:
        attention_batch(chunk)            # batch attention
        route_experts_batch(chunk)        # N routing decisions
        deduplicate_experts(routings)     # key optimization
        pread_unique_experts(deduped)     # fewer SSD reads!
        expert_forward_batch(chunk)       # batch GPU dispatch
```

### Key Optimization: Expert Deduplication
In a chunk of 16 tokens, many will route to the same experts. If 10/16 tokens
use expert #42 in layer 5, we pread expert #42 ONCE instead of 10 times.

With K=8, 256 experts, and chunk_size=16:
- Total expert slots: 16 × 8 = 128
- Unique experts (estimated): ~40-60 (many collisions)
- SSD read reduction: ~50-70% within each chunk

### Implementation
- Add `--prefill-batch N` flag (chunk size)
- Requires batch attention implementation (may already exist for encode path?)
- Expert deduplication: sort selected experts, merge duplicates, track which
  tokens need which experts for the combine step

### Effort
This is the highest-effort experiment — requires reworking the prefill loop.
But the payoff is large: fewer SSD reads AND better GPU utilization (batch matmuls).

---

## Experiment 5: Prefill Chunk Size Knob

**Priority: FIFTH — pairs with Experiment 4**

### What
Once batched prefill exists, expose chunk size as a tunable parameter.

### Test Matrix
| Chunk Size | Tokens/Chunk | Expected Dedup | Metal Efficiency |
|-----------|-------------|---------------|-----------------|
| 1 | 1 | 0% (current) | Low |
| 8 | 8 | ~30% | Medium |
| 16 | 16 | ~50% | Good |
| 32 | 32 | ~60% | Good |
| 64 | 64 | ~65% | Best? |
| 128 | 128 | ~70% | Memory pressure? |

### Considerations
- Larger chunks = more expert dedup = fewer SSD reads
- Larger chunks = bigger intermediate buffers = memory pressure
- Apple Silicon GPU has good batch matmul efficiency up to ~64 tokens
- Beyond 64-128, memory pressure may offset dedup gains

---

## Experiment 6: Layer-Class-Aware Expert Policies

**Priority: SIXTH — generalizes everything above**

### What
The most flexible experiment: specify different K values for different layer
groups, independently for prefill and decode.

### CLI Design
```bash
# Simple: different K for linear vs full-attention layers
--prefill-k "linear:0,full:8"

# Granular: per-layer-range K
--prefill-k "0-11:4,12-35:0,36-47:8"

# Production profile: fast prefill, full decode
--prefill-k "linear:2,full:8" --k 8
```

### Implementation
- Parse policy string into per-layer K array
- During routing: look up K for current layer
- Apply independently for prefill and decode phases

### Test Matrix (selected interesting policies)
| Policy | Description | Expected TTFT |
|--------|------------|--------------|
| `linear:0,full:8` | Exp 3 (full-attn only) | ~4-6s |
| `linear:2,full:8` | Compromise | ~6-8s |
| `linear:0,full:4` | Aggressive | ~3-4s |
| `0-11:8,12-35:0,36-47:8` | First+last layers full | ~5-7s |
| `all:0` | K=0 bound (Exp 1) | ~2-4s |

---

## Execution Order

| Phase | Experiment | What It Tells Us |
|-------|-----------|-----------------|
| 1 | K=0 prefill bound (#1) | Theoretical TTFT floor |
| 2 | --prefill-k sweep (#2) | Quality vs TTFT curve |
| 3 | Full-attn-only (#3) | Best quality/speed ratio |
| 4 | Batched prefill (#4) | Execution model improvement |
| 5 | Chunk size sweep (#5) | Optimal batch size |
| 6 | Layer-class policies (#6) | Production tuning |

Experiments 1-3 are quick (flag changes, low code effort).
Experiment 4 is the big investment (new execution mode).
Experiments 5-6 build on 4.

## Success Criteria
- TTFT < 8s at K=8 decode quality (50% reduction)
- Stretch: TTFT < 4s with acceptable quality
- All configs must pass decode quality gate (10-prompt suite)
- Prefill quality metric: first 64 decode tokens coherent and factually correct

## Dependencies
- Requires the model quality bug to be fixed first (currently under investigation)
- Experiments 1-3 are independent of Batch 4
- Experiment 4 can run in parallel with Batch 4 cache experiments
