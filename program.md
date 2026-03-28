# autoresearch — Flash-MoE 122B Optimization

This is an autonomous research loop for optimizing the Flash-MoE 122B inference engine on Apple Silicon (M4 Mac mini, 16GB RAM).

## Overview

Flash-MoE is a pure C/Metal inference runtime for Qwen MoE models, streaming experts from SSD through unified memory. The 122B model (Qwen3.5-122B-A10B-4bit) has 64GB of expert data that must flow through 16GB RAM — making it **severely I/O-bound** (SSD reads account for ~70-74% of per-layer latency). The 35B model is already well-optimized at 11.5 tok/s; this campaign focuses exclusively on 122B.

## Setup

To set up a new experiment campaign, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files** for full context:
   - `program.md` — this file (agent instructions + optimization plans)
   - `metal_infer/infer.m` — the inference engine (~8900 lines, C/Objective-C + Metal). **This is the primary file you modify.**
   - `metal_infer/shaders.metal` — GPU kernel shaders (~1400 lines). **Secondary modification target.**
   - `bench.sh` — automated benchmark script (starts server, runs inference, reports RESULT line)
   - `quality_gate.sh` — 10-prompt A/B quality comparison suite
   - `results.tsv` — experiment log (tab-separated)
   - `.agent-context.md` — project context and key paths
   - `CLAUDE.md` — technical overview
4. **Verify model files exist**: Check that these paths are valid:
   - Model config: `~/models/flash-moe/Qwen3.5-122B-A10B-4bit/config.json`
   - Weights: `metal_infer/out_122b/model_weights.bin` (3.46 GB)
   - Packed experts: `metal_infer/out_122b/packed_experts/` (~61 GB)
   - Manifest: `metal_infer/out_122b/model_weights.json`
   - Vocab: `metal_infer/vocab_122b.bin`
5. **Verify build**: `cd metal_infer && make infer` must succeed.
6. **Run baseline**: If the latest baseline in `results.tsv` is stale (different commit), run a fresh baseline first.
7. **Confirm and go**: Confirm setup looks good with the user, then begin the loop.

## Architecture Reference

### 122B Model (Qwen3.5-122B-A10B-4bit)
- 48 layers: **36 linear-attention (delta-net)** + **12 full-attention (KV cache)**
- 256 routed experts per layer, K=8 selected per token
- Per-expert size: ~5.3 MB (4-bit quantized)
- Per-token SSD reads: K × 48 layers × 5.3 MB ≈ **2 GB**
- Shared expert: always active, already in memory (no SSD read)
- Total expert data on disk: ~61 GB
- Non-expert weights (model_weights.bin): 3.46 GB (mmap'd, fits in RAM)

### Current Performance (baseline, commit `12330e5`)
| Metric | Value |
|---|---|
| Sustained decode | 1.18 tok/s (K=8) |
| Cold TTFT | 20-28s |
| I/O fraction | ~70-74% of per-layer time |
| OS page cache hit rate | ~5-10% (61 GB working set, ~12 GB available) |

### Hardware
- Apple M4 Mac mini, 16GB unified memory
- Internal NVMe SSD: ~5.5 GB/s sequential read, ~3-4 GB/s random
- 48 GB free disk

### Key Constraint: Unified Memory Architecture
SSD DMA and GPU compute share the same memory controller. Optimizations that look good in isolation can cancel each other when both pathways are active simultaneously. **Always test compound effects after isolated wins.**

## What You CAN Modify

- `metal_infer/infer.m` — the inference engine. Everything is fair game: I/O strategy, caching, expert routing, buffer management, pipeline structure, prefill logic, memory layout.
- `metal_infer/shaders.metal` — GPU compute kernels. Kernel fusion, new kernels, thread group tuning.
- `bench.sh` — only to add new measurement fields (don't break the RESULT line format).
- Python helper scripts (e.g. `simulate.py`, `profile_experts.py`, `cluster_experts.py`) — for offline analysis, profiling, or data preparation.
- `expert_index_122b.json`, `cooccur_122b.json`, `car_table_122b.json` — precomputed data files.
- Expert packing layout (`repack_experts_35b.py`) — when testing new disk layouts or compression schemes.

## What You CANNOT Modify

- `quality_gate.sh` — the evaluation harness is ground truth.
- Model weight files (`model_weights.bin`) — the dense/shared weights are read-only.
- `prepare.py`, `build_expert_index_35b.py` — data prep scripts.
- The 10-prompt quality suite (defined in `quality_gate.sh`).

## Goals

**Primary metric: sustained tok/s at K=8** (lower K is a separate experiment track, not the primary goal).

| Target | Value | Status |
|---|---|---|
| Baseline | 1.18 tok/s, ~20s TTFT | Current |
| Minimum | 2.0 tok/s at K=8 | Required |
| Stretch | 3.0+ tok/s at K=8 | Aspirational |
| TTFT | < 8s (50% reduction) | Required |
| TTFT stretch | < 4s | Aspirational |

**Secondary metrics:**
- TTFT (time to first token) — especially relevant for prefill experiments
- Quality — must pass the 10-prompt quality gate (≤2/10 degraded)
- Memory pressure — should not cause system-wide memory pressure or compressor thrash

## Build & Benchmark Protocol

### Build
```bash
cd metal_infer && make infer
```

### Benchmark (automated)
```bash
MODEL_DIR=~/models/flash-moe/Qwen3.5-122B-A10B-4bit ./bench.sh
```
This starts a server, runs a streaming chat completion (256 tokens), and prints:
```
RESULT: tok_s=X.XX ttft_s=X.XX crashes=0 quality=pass tokens=256
```

### Quality Gate (when testing routing/expert changes)
```bash
MODEL_DIR=~/models/flash-moe/Qwen3.5-122B-A10B-4bit ./quality_gate.sh
```
Runs 10 prompts at temperature=0 and compares outputs. Gate passes if ≤2/10 show degradation.

### Quick Functional Test (fast sanity check)
```bash
./metal_infer/infer \
  --model ~/models/flash-moe/Qwen3.5-122B-A10B-4bit \
  --weights metal_infer/out_122b/model_weights.bin \
  --manifest metal_infer/out_122b/model_weights.json \
  --vocab metal_infer/vocab_122b.bin \
  --prompt-tokens metal_infer/tokenizer_122b.bin \
  --prompt "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nWhat is the capital of Lebanon?\n<|im_end|>\n<|im_start|>assistant\n" \
  --tokens 30 --k 8
```

## Logging Results

Log every experiment to `results.tsv` (tab-separated). The TSV has a header row and columns:

```
experiment	tok_s	ttft_s	tokens	notes
```

1. **experiment** — short descriptive name (e.g. `arc-cache-4gb`, `prefill-k0`, `batch-prefill-16`)
2. **tok_s** — sustained decode tok/s (use 0.00 for crashes)
3. **ttft_s** — time to first token in seconds (use 0.00 for crashes)
4. **tokens** — number of tokens generated in the benchmark
5. **notes** — short description of what this experiment tried + outcome

Example:
```
experiment	tok_s	ttft_s	tokens	notes
baseline-warm	1.18	20.42	256	baseline after weight fix (commit 12330e5)
arc-cache-4gb	1.85	18.10	256	4GB ARC cache (750 experts) — +57% decode
prefill-k0	1.18	3.20	256	K=0 during prefill — TTFT 84% reduction, decode unchanged
batch-prefill-16	1.18	6.50	256	16-token chunk prefill with expert dedup — TTFT 68% reduction
crash-lz4-expert	0.00	0.00	0	LZ4 per-expert compression — segfault in decompress path
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25`).

**LOOP FOREVER:**

1. **Look at the git state**: the current branch/commit we're on.
2. **Pick the next experiment** from the optimization plans below (or devise a new one based on what you've learned). Prioritize by expected impact and dependencies.
3. **Implement the change** in `infer.m` and/or `shaders.metal`.
4. **Build**: `cd metal_infer && make infer 2>&1 | tail -5`. If build fails, fix and retry (up to 3 attempts).
5. **Benchmark**: `MODEL_DIR=~/models/flash-moe/Qwen3.5-122B-A10B-4bit ./bench.sh > run.log 2>&1`
6. **Read results**: `grep "^RESULT:" run.log`
   - If empty or crashes: `tail -50 run.log` to read the error.
7. **Record** the results in `results.tsv`.
8. **Decide**:
   - If tok_s improved (higher) OR ttft_s improved (lower) without tok_s regression: **keep**. Commit and advance.
   - If neutral or worse: **discard**. `git checkout -- metal_infer/` to revert.
   - If crash and fixable (typo, off-by-one): fix and re-run. If fundamentally broken after 3 attempts: discard and move on.
9. **Quality gate** (required for any experiment that changes expert routing, K values, or the forward pass logic): Run `quality_gate.sh` before committing a keep. If gate fails (>2/10 degraded), discard.
10. **Commit**: `git add -A && git commit -m "122b: <short description>"`. Include tok_s and delta in the commit message.

**Timeout**: Each benchmark takes ~5-10 minutes (server startup + 256 tokens at ~1 tok/s). If a run exceeds 20 minutes, kill it and treat it as a failure.

**Compound testing**: After every 3-5 individual keeps, run a compound benchmark to verify wins stack. If compound is worse than sum of parts, bisect to find the interference.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. They may be asleep or away. Run indefinitely until manually stopped. If you run out of planned experiments, think harder — re-read infer.m for new angles, try combining near-misses, try more radical approaches. The loop runs until the human interrupts you.

---

## Optimization Plans

The experiments below are organized into categories. Within each category, they're ordered by priority. You don't have to follow this order rigidly — adapt based on what you learn from each experiment.

### Anti-Results (DO NOT REPEAT)

These have already been tested and failed on 122B. Do not re-run them unless you have a fundamentally different approach:

| Experiment | Result | Why It Failed |
|---|---|---|
| CAR with full mincore | 0.18 tok/s | mincore() syscall overhead catastrophic at 384 calls/token |
| CAR with popularity table | 0.19 tok/s | mincore still dominates; popularity table adds nothing |
| F_NOCACHE (all modes) | neutral | OS page cache already thrashing; nocache doesn't help |
| madvise (sequential/willneed/random) | neutral | No effect when working set >> RAM |
| Pin shared weights (mlock) | neutral | Shared weights already resident; pinning doesn't free cache for experts |
| Temporal expert prediction (F_RDADVISE) | -18% (35B) | 25% hit rate, NVMe command contention |
| LZ4 DRAM cache (35B) | -13% | Decompress overhead > cache savings (but see note for 122B) |
| Custom Metal LRU cache (35B) | worse | Metal buffers wire memory → reduces OS page cache |
| dispatch_io | +1-2% (noise) | Marginal, within measurement noise |

---

### Category A: Decode-Side I/O Optimization (Primary Bottleneck)

These target the core bottleneck: 70-74% of per-layer time is SSD reads.

#### A1. Trace-Driven Offline Simulator & Autotuner
**Priority: FIRST — accelerates everything else**

Build an offline tool to replay routing traces and evaluate cache/layout/prefetch strategies in seconds instead of 5-10 minute real benchmarks.

- **Trace capture**: Add `--trace <path>` flag to infer. Dump a file of (token_id, layer, expert_indices[K], read_time_us, cache_hit) tuples per token. Also log cache events (hit/miss/eviction), read timings, and memory stats (vm_stat snapshots).
- **Simulator** (`simulate.py`): Read trace + experiment config → replay with configurable parameters:
  - Cache policies (ARC, LRU, frequency, co-occurrence-weighted)
  - Cache sizes (1-8 GB)
  - Expert disk layouts (current, clustered, compressed)
  - Prefetch strategies (co-occurrence, temporal, hybrid)
  - Warm-cache preloading strategies
- **Autotuner mode**: Sweep parameter space automatically (cache size × policy × prefetch budget) and report Pareto-optimal configs.
- **Validation**: Run simulator predictions vs actual benchmarks for K=8 baseline. Simulator must predict within ±10% of actual tok/s to be useful.
- **Deliverable**: `--trace` flag, `simulate.py`, validation report.

#### A2. Budget-Aware Hot Expert Cache
**Priority: HIGH — biggest potential for 122B specifically**

Unlike 35B where the OS page cache handled everything, 122B has ~5-10% page cache hit rate. A smart userspace cache has massive headroom.

- **Memory budget**: 4 GB malloc'd (leaves ~6 GB for OS page cache + overhead). Configurable via `--cache-mb`.
- **Capacity**: ~750 experts (out of 12,288 total = 256 × 48 layers)
- **Cache policy** — go beyond simple recency. Implement a composite scoring function:
  - **Recency**: when was this expert last accessed?
  - **Frequency**: how often has this expert been accessed over the last N tokens?
  - **Co-occurrence affinity**: if expert X is in cache and expert Y frequently co-occurs with X in the same or adjacent layers, boost Y's score.
  - Track per-layer expert hit rates and expert-pair co-occurrence at runtime.
  - Add separate quotas for **globally hot** experts (high frequency across all prompts) vs **locally hot** experts (high frequency in current context window).
- **Implementation rules** (avoid 35B mistakes):
  - Use `malloc`, NOT Metal buffers (avoid wiring pages)
  - Cap at exact budget (never exceed `--cache-mb`)
  - Copy expert data into cache on miss (memcpy from pread buffer)
  - Expert reads still go through pread → cache on miss
  - No mlock on cache memory (let OS swap if needed)
  - Emit telemetry: cache hit rate, eviction rate, per-layer hit distribution
- **Test matrix**: 2 GB (~375 experts), 4 GB (~750), 6 GB (~1125). Policies: ARC, LRU, frequency-only, composite (recency+frequency+co-occurrence).
- **Use simulator first** (A1) to predict which cache size/policy is optimal before real benchmarks.
- **Expected**: 30-60% decode improvement if cache hit rate reaches 20-30%

#### A3. Expert Repacking for Sequential SSD Access
**Priority: MEDIUM — builds on proven +14% baseline**

Current greedy clustering gave +14%. Better algorithms and disk-layout awareness could improve further.

- **Better clustering algorithms**:
  - Simulated annealing: start from greedy solution, randomly swap expert positions, accept if co-occurrence locality improves
  - Spectral clustering: treat co-occurrence matrix as graph, use spectral methods to find communities
  - Multi-prompt profiling: current clustering uses one prompt. Profile with 5-10 diverse prompts (2000+ tokens) for more robust co-occurrence data
  - Cross-layer awareness: if experts A,B,C often fire together across layers 5-8, pack them at similar offsets for SSD read pattern locality
- **SSD-aware packing**:
  - Align expert blobs to NVMe page boundaries (4KB or 16KB) to eliminate partial page reads
  - Pack co-occurring experts contiguously so K scattered reads become fewer, larger sequential reads
  - Benchmark different alignment sizes and measure syscall count + read pattern
- **Expert read coalescing** (runtime): when multiple selected experts are adjacent in the layer file after clustering, issue a single large pread instead of K separate preads. Scan offsets for adjacency and merge.
- **Use simulator** (A1) to evaluate layouts before repacking (repacking takes ~3 minutes)
- **Expected**: 5-20% additional improvement over current clustering

#### A4. Predictive Expert Prefetch
**Priority: MEDIUM — moderate risk, moderate reward**

Previous temporal prediction failed (25% hit rate, NVMe contention). This is a fundamentally different approach: structured prefetch queue with co-occurrence prediction, overlapping SSD I/O with current-token compute.

- **Two-class prefetch system**:
  1. **Exact prefetch** (current token): after routing for layer N, immediately start async pread for layer N's experts while GPU processes the previous layer's CMD3. This overlaps I/O with GPU compute.
  2. **Speculative prefetch** (next token/layer): use co-occurrence transition tables ("if expert 42 in layer N → experts {78, 156, 203} likely in layer N+1") to pre-stage likely experts into scratch buffers during CPU-idle phases.
- **Prefetch queue**: maintain a priority queue of prefetch requests. Process during identified CPU-idle windows:
  - After GPU attention completes, before next layer starts
  - During routing softmax + topK computation
  - During post-expert combine + norm
- **Double-buffer expert data** (BUF_A / BUF_B): flip each layer so current-token reads and speculative-next reads don't conflict.
- **Metrics to track**: prefetch hit rate, wasted reads (bytes fetched but unused), bytes read/token, SSD utilization overlap, decode tok/s impact.
- **Guard rails**: max prefetch budget per layer (e.g. 4 experts), only during identified CPU windows, kill speculative reads if they contend with exact reads.
- **Expected**: 15-30% decode improvement if co-occurrence prediction accuracy is 40%+
- **Risk**: SSD bandwidth contention between prefetch and actual reads. Monitor carefully.

#### A5. Lossless On-Disk Expert Compression
**Priority: MEDIUM — quick to validate, potentially large payoff**

Failed on 35B but the math is completely different on 122B:
- 35B: I/O 56%, page cache 71% → compression competes with fast cache
- 122B: I/O 74%, page cache ~5% → almost every read is from SSD

- **Step 1 — Validate ratio FIRST**: Measure actual LZ4 compression ratio on a sample of 4-bit quantized expert blobs. If ratio < 1.3x → skip this experiment entirely. 4-bit quantized data may not compress well.
- **Step 2 — Incremental rollout**: Don't repack all 61GB at once. Start with cold experts only (bottom 50% by frequency) or bandwidth-heavy tensors. This validates the concept with much less effort and risk.
- **Step 3 — Full rollout** (if Step 2 succeeds): Compress all expert blobs with LZ4 (per-expert, not per-layer). Store compressed experts with offset index in layer files.
- **Step 4 — Runtime path**: pread compressed blob → LZ4 decompress into existing expert buffer/staging buffer before execution.
- **Math**: LZ4 decompress at ~40 GB/s. Decompress 5.3 MB expert: 0.13 ms vs ~1 ms pread. If compression ratio 1.5x: effective SSD throughput = 5.5 × 1.5 = 8.25 GB/s.
- **Measure**: SSD bytes saved, CPU decompression overhead, net tok/s change.
- **Expected**: 20-40% decode improvement if compression ratio ≥ 1.5x
- **Risk**: 4-bit quantized data may not compress well. That's why Step 1 comes first.

#### A6. Per-Expert Precision Tiers
**Priority: LOW — high effort, needs careful quality validation**

Not all experts are equal. Hot experts fire 10-20x more than cold ones.

- Keep shared expert and hottest routed experts (top 50% by frequency) at full 4-bit precision
- Quantize colder experts (bottom 50%) more aggressively to 2-bit (50% less SSD read per cold expert)
- At runtime: check expert precision tier, use appropriate dequant kernel
- **Requires**: mixed-precision dequant support in shaders.metal, careful quality gate
- **Expected**: 15-25% I/O reduction, quality risk on cold experts
- **Quality gate**: especially important here — run full 10-prompt suite at each precision split

---

### Category B: Prefill / TTFT Optimization

These target TTFT independently. Prefill quality requirements differ from decode — the model just needs "good enough" internal state to prime decode. Treat prefill as its own execution mode, not "decode repeated many times."

#### B1. Selective Expert Policies During Prefill
**Priority: FIRST in this category — unified framework for all prefill expert experiments**

Add a unified `--prefill-policy` system that controls expert routing during prefill independently from decode. Decode always stays at full quality (K=8) by default.

**Specific configurations to test (in order):**

1. **K=0 bound** (`--prefill-k 0`): Skip ALL routed experts during prefill. Only shared expert runs. Eliminates 100% of SSD reads during prefill. Establishes theoretical TTFT floor. Expected TTFT: ~2-4s (down from 20s). This is a BOUND, not necessarily shippable.

2. **Prefill-K sweep** (`--prefill-k N` for N=2,4,6):
   | prefill-k | Expected TTFT | SSD Reduction |
   |---|---|---|
   | 0 | ~2-4s | 100% |
   | 2 | ~6-8s | 75% |
   | 4 | ~8-10s | 50% |
   | 6 | ~10-12s | 25% |

3. **Full-attention-only** (`--prefill-policy full-attn-only`): During prefill, run routed experts ONLY on the 12 full-attention layers. The 36 linear-attention (delta-net) layers use shared expert only. Rationale: full-attention layers write to KV cache (persistent long-range memory); delta-net layers are less sensitive to per-token expert accuracy. SSD reads reduced by 75%. Expected: TTFT ~4-6s with best quality preservation. Variants: K=8 on full + K=0 on linear, K=8 on full + K=2 on linear.

4. **Layer-class-aware policies** (`--prefill-k "linear:0,full:8"`): Generalize into per-layer or per-layer-group expert policies. Parse policy string into per-layer K array. Apply independently for prefill and decode.
   ```bash
   --prefill-k "linear:0,full:8"          # full-attn only
   --prefill-k "0-11:4,12-35:0,36-47:8"  # first+last layers full
   --prefill-k "linear:2,full:8" --k 8    # compromise
   ```

**Quality gate**: Required for each configuration. Compare first 64 decode tokens against full-K baseline.

#### B2. Batched Prefill as a Dedicated Execution Mode
**Priority: SECOND — highest effort, highest potential**

Stop processing prefill tokens one at a time. Treat prefill as a fundamentally different execution mode where projections and weights are reused across many prompt tokens.

- **Batch prompt chunks** so that attention projections, expert routing, and expert forward passes process multiple tokens simultaneously. This enables weight reuse across tokens (each projection weight matrix is loaded once per chunk instead of once per token).
- **Expert deduplication within chunks**: In a chunk of 16 tokens with K=8: 128 expert slots but many collisions → ~40-60 unique experts. Pread each unique expert ONCE instead of per-token → 50-70% SSD read reduction within each chunk.
- **Expose prefill chunk size as a first-class runtime knob**: `--prefill-chunk N` (default: 1 = current behavior).
- **Benchmark TTFT and prefill tok/s separately** from decode tok/s. Add prefill-specific metrics to bench.sh output.
- **Test matrix**: chunk sizes 8, 16, 32, 64, 128, 256.
- **Effort**: requires reworking the prefill loop, batch attention implementation, batch expert routing + deduplication.
- **Expected**: TTFT 40-60% reduction + better GPU utilization from batch matmuls.
- **Depends on**: C3 (phase-specific kernel selection) for optimal GPU utilization at different chunk sizes.

---

### Category C: GPU Compute Optimization

These target the non-I/O portion of per-layer time (~26-30%).

#### C1. Port 35B Shader Wins to 122B Dimensions
**Priority: HIGH — proven wins, just needs dimension adaptation**

The 35B autoresearch (branch `autoresearch/mar23b`) produced several kernel improvements. These need to be verified/adapted for 122B dimensions (hidden=3072, heads=32, etc.):

- `matvec_v3` for CMD2 o_proj
- `tg128` matvec for expert gate/up projections
- Fused `residual_add + rms_norm_sum` kernel
- Coalesced batched matvec encoders
- `v3/tg128` kernels for batched CMD1/CMD2 projections

Some of these may already apply if the kernels are dimension-generic. Others may need thread group size or tile size adjustments for the different projection dimensions.

#### C2. GPU Private Buffer Compression
**Priority: MEDIUM — hardware-transparent optimization**

- StorageModePrivate buffers let the GPU's memory controller apply lossless compression (transparent to shaders)
- 4-bit quantized weights with 2.4-3.7 bits entropy → highly compressible → effective bandwidth could double
- **Implementation**: blit shared→private (~0.02ms per expert), then matvec from private buffer
- **Risk**: blit cost may exceed compression benefit. Test in isolation first.

#### C3. Phase-Specific Kernel Selection
**Priority: HIGH — prerequisite for efficient batched prefill**

Add separate kernel-selection policies for prefill and decode. Do not use one global fast-path toggle for both phases.

- **Prefill kernels**: Implement a prefill-only fast tensor-matmul path optimized for batch processing (M > 1). When processing a chunk of 32/64/128 tokens, use large-tile kernels designed for matrix-matrix multiply (not matrix-vector).
- **Decode kernels**: Keep the current M=1 low-latency matvec path (optimized for single-token decode). LM head and other decode-sensitive ops stay on the current path unless proven otherwise.
- **Chunk-size-aware kernel routing**: Automatically select kernel based on M/chunk size:
  - M=1 (decode, single token) → current low-latency matvec kernels
  - M=8-32 (small batch) → intermediate tile kernels
  - M=64-256 (large batch/prefill) → large-tile matrix-matrix kernels
- **Benchmark chunk sizes**: 32, 64, 128, 256 tokens — measure GPU throughput (TFLOPS) and latency per chunk.
- **Expected**: significant prefill speedup when combined with B2 (batched prefill). May also help decode if batched decode is ever explored.
- **Implementation**: add kernel dispatch table indexed by (phase, M_size) → kernel_id. Phase is "prefill" or "decode". M_size is bucketed.

---

### Category D: Memory Management & Efficiency

These experiments manage the 16GB unified memory budget explicitly to prevent interference between subsystems.

#### D1. Three-Pool Memory Budgeting
**Priority: HIGH — infrastructure for compound experiments**

Make memory budgets explicit and configurable. Without this, multiple experiments (cache, prefetch, batched prefill) will compete for the same 16GB unpredictably.

- **Four pools with explicit budgets**:
  1. **Dense/shared weights** (~3.5 GB): model_weights.bin, mmap'd, effectively pinned by OS
  2. **Hot expert cache** (configurable, default 4 GB): A2's cache
  3. **Prefetch staging buffers** (configurable, default 0.5-1 GB): A4's prefetch queue
  4. **KV cache + delta-net state** (~0.2 GB for single-batch, grows with context): attention state
  - Remainder (~6-7 GB): left for OS page cache + system overhead
- **Configurable via CLI**: `--cache-mb 4096 --prefetch-mb 512 --kv-reserve-mb 256`
- **Adaptive resizing**: when context length grows (longer prompts → larger KV cache), automatically shrink expert cache or prefetch pool. When chunk size changes (batched prefill → larger intermediate buffers), rebalance.
- **Telemetry**: emit periodic stats showing per-pool usage, when one pool forces evictions in another, OS memory pressure (vm_stat).
- **Expected**: prevents compound experiments from interfering. No direct perf gain, but enables A2 + A4 + B2 to coexist without thrashing.

#### D2. TurboQuant KV-Cache Compression (PolarQuant + Walsh-Hadamard)
**Priority: HIGH — proven 4.6x compression, frees memory for expert cache, enables longer context**

Port Google's TurboQuant (ICLR 2026) KV cache compression into our Metal engine. Reference implementation: <https://github.com/TheTom/turboquant_plus> (Python + C/Metal, Apache 2.0).

**Core algorithm (per KV vector x ∈ R^d):**
1. Extract norm: γ = ||x||, x̂ = x/γ
2. Walsh-Hadamard rotation → transforms high-kurtosis KV tensors into near-Gaussian (kurtosis 900 → 2.9)
3. PolarQuant: optimal scalar quantization on the rotated coordinates (3-bit = 4.9x compression)
4. QJL residual: 1-bit sign correction for unbiased inner product recovery
5. Total: 3.25 bits/value → **4.6x compression** vs f16

**Why this matters for 122B specifically:**
- Only 12 full-attention layers have real KV cache (36 are delta-net with recurrent state)
- At short context, KV is small (~0.2 GB) — frees memory for expert cache
- At longer context (8K+), KV grows significantly — TurboQuant keeps it bounded
- Memory freed goes directly to expert cache (A2), improving decode tok/s
- Validated results: PPL 5.460 vs q8_0's 5.414 (+0.8%), prefill at q8_0 speed parity

**Implementation plan:**
1. **Port Python reference first** — validate compression/decompression on our actual Qwen 122B KV tensors. Measure kurtosis before/after WHT rotation, compression ratios, cosine similarity.
2. **Metal kernels** — port the C/Metal kernels from turboquant_plus's llama.cpp fork:
   - `ggml-turbo-quant.c` → adapt quantize/dequantize for our buffer layout
   - `ggml-metal.metal` → extract WHT + PolarQuant + dequant shaders, adapt to our kernel dispatch
   - Key optimization: graph-side WHT rotation (rotate queries instead of keys at decode time)
   - Block-32 storage layout for Metal-friendly memory access
3. **Architecture-aware precision tiers:**
   - Full-attention KV (12 layers): turbo3 (3.25 bits, 4.6x) — these are the critical layers
   - Delta-net recurrent state (36 layers): turbo3 or even turbo2 — recurrent averaging is more tolerant
4. **Recency-aware option**: recent KV entries (last N tokens) at full precision, older entries compressed
5. **CLI**: `--kv-compression turbo3` (default: none/f16)

**Reference files to study:**
- `turboquant_plus/turboquant/rotation.py` — WHT implementation
- `turboquant_plus/turboquant/polar_quant.py` — PolarQuant algorithm
- `turboquant_plus/turboquant/turboquant.py` — full pipeline
- `turboquant_plus/` C port files in their llama.cpp fork (Metal kernels)

**Validation:**
- Cosine similarity ≥ 0.99 on our KV tensors
- Quality gate must pass (≤2/10 degraded)
- Measure memory freed → expert cache capacity increase → decode tok/s impact

**Expected**: Free 50-80% of KV memory. At longer contexts, prevents KV from crowding out expert cache. Combined with A2 (hot expert cache), this is a significant compound win. Decode speed benefit is indirect (more cache room) but real.

#### C4. Metal Kernel Study & Port from llama.cpp
**Priority: HIGH — years of community optimization to mine**

llama.cpp has the most battle-tested Metal compute kernels in the open-source LLM ecosystem. Study their latest `ggml-metal.metal` shaders for optimizations applicable to our kernels.

**Areas to investigate:**
1. **Attention kernels** — llama.cpp's flash attention Metal implementation. Our attention is split across CPU and GPU; their fully-fused Metal attention may be significantly faster. Key files: `ggml-metal.metal` flash attention kernels.
2. **Matrix-vector multiply** — compare their SIMD reduction, thread group sizing, and memory access patterns against our `dequant_matvec_4bit_v3`. They support multiple quantization formats (q4_0, q4_1, q5_0, q8_0, IQ formats) with format-specific kernel optimizations.
3. **RMSNorm and SwiGLU** — their fused kernels may have tighter thread group utilization than ours.
4. **Quantization format dequant paths** — study IQ2_M, IQ3_XXS, and other importance-matrix quant dequant kernels. These use lookup tables and SIMD tricks for sub-4-bit that could inform our own lower-bit expert experiments (A6).
5. **Memory access patterns** — how they handle bfloat16, half4 vectorized loads, threadgroup memory usage, and Metal buffer binding strategies.
6. **Command buffer structure** — their dispatch pattern (how many encoders per pass, commit strategy, async compute) vs our 3-cmd-buffer-per-layer approach.

**Implementation approach:**
- Clone llama.cpp (or just fetch `ggml-metal.metal` and related headers)
- Document applicable optimizations in a comparison table
- Port the most impactful patterns into our `shaders.metal`, adapted for our buffer layout and quantization format
- Benchmark each ported optimization individually before compounding

**Also study mac-code specifically:**
- Their `--flash-attn on` flag enables fused flash attention — understand what this activates in the Metal backend
- Their KV cache quantization (`--cache-type-k q4_0 --cache-type-v q4_0`) — how is this implemented in Metal?
- Thread count tuning (`-t 4`) — they run 4 CPU threads; understand the interaction with Metal dispatch

**Expected**: 10-30% compute speedup on the non-I/O portion of per-layer time. Since I/O is 70-74%, this translates to ~3-9% end-to-end. However, any compute speedup also opens more CPU-idle time for prefetch (A4) and cache management (A2), so the compound effect is larger.

---

### Category E: Compound & Validation Experiments

#### E1. K Reduction with Quality Validation
**Priority: MEDIUM — simple but needs quality gate**

Already tested: K=6 gives 1.60 tok/s (+30%), K=5 gives 1.92 tok/s (+56%). But quality gate hasn't been run.

- Run quality_gate.sh at K=6 and K=5
- If quality passes, these become viable production configs
- Compound with other optimizations (e.g. cache + K=6)

#### E2. Adaptive K per Layer
**Priority: MEDIUM — architecturally motivated**

Not all layers are equal. Early layers (embedding refinement) and late layers (output projection) may need higher K than middle layers.

- Profile per-layer expert importance (measure output magnitude change when dropping experts)
- Assign per-layer K values based on sensitivity
- **Expected**: match K=6 throughput with K=8-level quality

---

### Category F: Second-Wave / Optional Experiments

These are lower priority or depend on specific conditions being met. Pursue after Categories A-E have been explored.

#### F1. Multi-Request Cache-Affinity Scheduling
**Only relevant if testing concurrent serving.**

When serving multiple concurrent requests, route requests to maximize expert cache hits. If request A just used experts {42, 78} in layer 5, and request B needs {78, 156}, schedule B immediately after A → expert 78 is still hot.

- Requires request queuing and routing-aware scheduling
- Skip until multi-user serving is the focus

#### F2. Startup / Prompt Warm-Cache
**Priority: MEDIUM — pairs with A2 (cache)**

- **Startup pre-warming**: At server start, preload top historically hot experts per layer (from frequency profile data) into the expert cache before any requests arrive.
- **Prompt-aware warming**: use prompt/domain heuristics to predict which experts will be needed and pre-stage them. E.g., code prompts may activate different experts than factual Q&A.
- **CLI**: `--warmup-profile <freq_file>` to specify frequency data. `--warmup-tokens N` to run N tokens with full K before enabling any K-reduction optimizations.
- **Expected**: reduces cold-start penalty, improves first-request TTFT and decode speed.

---

### Category G: Paper-Inspired Experiments

These experiments are directly inspired by peer-reviewed papers and open-source systems. Each includes the source, specific implementation approach for Flash-MOE's codebase, and expected impact.

---

#### G1. Apple Row-Column Bundling for Sequential SSD Reads
**Paper: "LLM in a Flash" (Alizadeh et al., arxiv:2312.11514)**

**Insight:** Flash SSDs achieve throughput via parallelism — large sequential reads are dramatically faster than many small random reads. Apple showed 20-25x GPU speedup by bundling related weight matrices.

**What to implement:** Change the SSD expert packing layout from per-expert to per-layer bundles. Instead of storing each expert's (gate, up, down) projections as separate 5.3 MB blobs, pack them contiguously: all gate projections for 8 selected experts together, all up projections together, all down projections together. This transforms K=8 separate 5.3 MB reads into 3 larger sequential reads (gate_batch, up_batch, down_batch) per layer.

**Implementation:** Modify `pack_experts_ssd.py` to bundle projection types. Update `expert_file_offset()` and `expert_pick_fd()` in `infer.m` to compute offsets within the bundled layout. Requires repacking the expert files (~61 GB, one-time cost).

**Expected impact:** 20-40% decode improvement from SSD parallelism. Combined with the existing sequential .bin file, this gives the SSD controller maximum read-ahead opportunity.

**Risk:** LOW — purely a layout change; correctness verified by quality gate.

---

#### G2. Apple Windowing — Expert Output Caching Across Tokens
**Paper: "LLM in a Flash" (Alizadeh et al.)**

**Insight:** Adjacent tokens in decode reuse many of the same experts. Instead of re-executing already-computed expert outputs, cache them and apply a lightweight "residual" correction when reuse isn't perfect.

**What to implement:** Cache the OUTPUT of expert computations (not just the weights). For each (layer, expert) pair, store the computed output tensor. On the next token, if the same experts fire, reuse cached outputs instead of recomputing. Use a small learned residual to handle token-level drift.

**Implementation:** Add `expert_output_cache[layer][expert]` — malloc'd buffer per expert output size. On each layer forward, check if selected experts match previous token's experts. If match >50%, reuse cached output with 50% weight and recompute with 50% weight as correction. Add `--expert-output-cache <mb>` flag.

**Expected impact:** 15-30% decode improvement if expert reuse rate >50%. Particularly strong during coherent generation (e.g., long responses).

**Risk:** MEDIUM — requires careful memory budgeting (D1) to avoid thrashing. May cause quality regressions if residual is too aggressive.

---

#### G3. HOBBIT — Mixed-Precision Expert Offloading
**Paper: "HOBBIT: Mixed Precision for MoE Offloading" (arxiv:2411.01433)**

**Insight:** Not all experts are equally important. Under memory pressure, load "cold" experts (bottom 50% by frequency) at 2-bit precision, keeping "hot" experts (top 50%) at full 4-bit. The paper showed 9.93x speedup over full-precision MoE offloading.

**What to implement:** Add a `--mixed-precision-offload` flag. At init, profile expert frequencies and classify experts as hot/cold. Store experts at 2-bit using a simple fixed-point quantization scheme (e.g., fp4 simulant via per-tensor scales). At runtime, use different dequant kernels for hot vs cold experts.

**Implementation:** Add `expert_precision_tier[layer][expert]` (HOT=4bit, COLD=2bit). Modify `expert_offload_copy()` to use appropriate dequant path. Add `--cold-expert-bitwidth 2` flag. Requires Metal kernel for fast 2-bit → f16 dequantization.

**Expected impact:** 2-3x reduction in cold expert SSD read volume. Combined with hot expert cache, could double effective cache hit rate.

**Risk:** HIGH — requires new Metal kernel for 2-bit dequantization. Quality gate mandatory.

---

#### G4. SpecMoEOff — Speculative Expert Prefetch
**Paper: "SpecMoEOff: Speculative Decoding for MoE Offloading" (arxiv:2508.21706)**

**Insight:** Use speculative decoding principles to prefetch experts ahead of their actual need — while the GPU is computing with experts for token N, prefetch experts for token N+1.

**What to implement:** Combine the existing `--predict` (temporal prediction) with a verification layer. After each token, the predictor speculates which experts will fire on the next token. Submit those as async preads while the current token's expert outputs are being used. If prediction is wrong, discard and load the correct expert synchronously.

**Implementation:** In `async_pread_start()`, issue two batches: (1) current token's predicted experts, (2) next token's speculated experts. In `async_pread_wait()`, check if speculation was correct. Track speculative hit rate separately. Add `--spec-moe-window N` for max speculative window size.

**Expected impact:** 15-25% decode improvement if speculative accuracy >70%. Overlaps I/O with compute effectively.

**Risk:** MEDIUM — requires careful async queue management. Wrong speculation wastes SSD bandwidth.

---

#### G5. Hypura — Layer-Phase-Aware Expert Prefetch
**Paper: "Hypura: Storage-Tier-Aware LLM Inference" (github.com/t8/hypura)**

**Insight:** Transformer layers have deterministic access patterns — attention layers follow a predictable phase order. Hypura exploits this with a specialized prefetch scheduler that knows WHEN in the layer pipeline to issue SSD reads.

**What to implement:** Implement a phase-aware prefetch trigger. For each layer, identify the "CPU idle window" (after attention softmax, before expert routing) and the "GPU busy window" (expert matvec). Issue preads only during CPU idle windows, never competing with CPU cache management.

**Implementation:** In `fused_layer_forward()`, add explicit prefetch trigger points at each layer boundary. After the attention RMSNorm completes and before expert routing starts — this is when the CPU has free cycles. The existing `infer_prefetch` path already does something similar; formalize it as a state machine with PHASE_PREP, PHASE_ATTEN, PHASE_ROUTE, PHASE_EXPERT, PHASE_NORM.

**Expected impact:** 10-15% decode improvement from eliminating I/O contention with CPU cache management.

**Risk:** LOW — refactors existing prefetch logic into a cleaner state machine.

---

#### G6. LZ4 Per-Expert Compression with On-Demand Decompress
**Paper: Derived from "LLM in a Flash" compression insights + llama.cpp LZ4 implementation**

**Insight:** 4-bit quantized data has residual entropy. LZ4 compression on top can achieve 1.3-2x compression ratios on 4-bit expert blobs, reducing SSD read volume proportionally.

**What to implement:** Compress each expert blob with LZ4 and store alongside the uncompressed data in the layer files. At runtime, pread compressed blob → decompress into expert buffer → execute. LZ4 decompresss at ~40 GB/s; decompressing a 5.3 MB expert takes ~0.13 ms vs ~1 ms raw SSD read.

**Implementation:** Add `--lz4-expert-compress` flag. In `expert_offload_copy()`, check if expert is compressed (bit in header), decompress if needed. Use the existing LZ4 library already in the codebase (from llama_turboquant_ref). Validate compression ratio first — if <1.3x, skip this experiment.

**Expected impact:** 20-40% decode improvement if compression ratio ≥1.5x. Zero quality impact (lossless).

**Risk:** MEDIUM — decompression CPU overhead must be measured. Test on cold experts first (incremental rollout).

---

#### G7. Co-occurrence Graph Partitioning for SSD Layout
**Paper: "MoE Inference Optimization Survey" (arxiv:2412.14219) + spectral clustering theory**

**Insight:** Expert co-occurrence forms a graph — experts that frequently fire together should be physically adjacent on SSD. The current greedy clustering may leave performance on the table. Spectral clustering finds globally optimal graph partitions.

**What to implement:** Build the co-occurrence matrix (already done: `cooccur_122b.json`). Apply spectral clustering to partition 256 experts into N spatial regions (e.g., 16 regions of 16 experts each). Repack experts by cluster region. Benchmark against current greedy layout.

**Implementation:** Add `--cluster-method <greedy|spectral|metis>` to `cluster_experts.py`. Implement spectral clustering using scipy's `spectral_clustering`. Run on co-occurrence matrix, produce new expert ordering. Benchmark both layouts with `bench.sh`.

**Expected impact:** 5-15% decode improvement over greedy clustering if spectral finds better global partition.

**Risk:** LOW — offline layout experiment, no runtime code changes.

---

#### G8. KV Cache TurboQuant — Walsh-Hadamard Rotation + PolarQuant
**Paper: Google's TurboQuant (ICLR 2026) — port from python/turboquant_plus**

**Insight:** KV cache tensors have extreme kurtosis (900+) making them poorly suited for uniform quantization. Walsh-Hadamard rotation transforms them to near-Gaussian (kurtosis ~3), enabling 3-bit PolarQuant with minimal quality loss — 4.6x compression.

**What to implement:** Port TurboQuant's Metal kernels from the reference implementation. Apply WHT rotation + PolarQuant to KV vectors at write time; dequantize + inverse WHT at read time. Target the 12 full-attention layers specifically.

**Implementation:** Study `turboquant_plus/turboquant/rotation.py` and C/Metal port files. Extract WHT + PolarQuant shaders. Add `--kv-turboquant` flag. Integrate into KV cache write path in `infer.m`. Validate with cosine similarity ≥0.99 vs f16.

**Expected impact:** Frees 50-80% of KV cache memory. Memory goes to expert cache → 20-40% decode improvement at longer contexts.

**Risk:** HIGH — requires porting novel kernels from external reference. Quality gate mandatory.

---

#### G9. Expert Frequency Tiering — Hot/Cold Separate Caches
**Paper: HOBBIT + general cache theory**

**Insight:** The expert access distribution is extremely skewed — top 20% of experts account for 80% of accesses. Give hot experts their own dedicated cache partition with high priority.

**What to implement:** Split the expert cache into two pools: hot pool (e.g., 2 GB for top 100 frequent experts) and cold pool (e.g., 2 GB for remaining experts). Use different eviction policies: hot pool uses frequency-aware LRU, cold pool uses pure recency LRU. Add `--hot-cache-mb N` and `--cold-cache-mb N` flags.

**Implementation:** Extend the malloc cache (D1) to support pool-aware eviction. Track per-expert frequency counter. On eviction from hot pool, demote to cold pool. On access in cold pool, promote to hot pool if frequency threshold met.

**Expected impact:** 15-25% decode improvement from giving hot experts preferential cache treatment.

**Risk:** LOW — purely a cache policy change; no quality impact.

---

#### G10. Prefill-Only Expert Prefetch (Batched)
**Paper: SpecMoEOff + general prefetch theory**

**Insight:** During prefill, expert routing is known BEFORE computation (no dependency chain). This enables full batch prefetch — know all K×48 expert accesses, deduplicate, pread all unique experts in one shot before any GPU compute starts.

**What to implement:** In prefill mode, collect all expert indices for all prompt tokens upfront. Deduplicate → produce unique expert list → submit all as single batched pread → wait for all → execute all layers. This amortizes SSD read latency completely during prefill.

**Implementation:** In `prefill_forward()`, before the token loop, build `all_expert_indices[K * prompt_len]`, deduplicate, issue batch pread. Only start GPU compute after all preads complete. Add `--batch-prefill` flag.

**Expected impact:** 40-60% TTFT reduction. Prefill becomes purely compute-bound (no I/O stalls).

**Risk:** MEDIUM — requires enough RAM to hold all unique expert weights simultaneously. Memory budget (D1) needed.

---

#### G11. Async I/O Queue Depth Tuning
**Paper: Hypura (I/O scheduling) + general systems theory**

**Insight:** macOS's `dispatch_io` and the POSIX AIO interface have tunable queue depth. The default queue may under- or over-subscribe the NVMe controller. Finding the optimal depth maximizes SSD throughput without causing command contention.

**What to implement:** Add `--aio-depth N` flag. Experiment with queue depths from 1 to 32. For each depth, measure: (a) SSD bytes/second utilization, (b) decode tok/s, (c) CPU idle time during I/O. Find the Pareto-optimal depth.

**Implementation:** In `metal_io_pool.c`, add `io_queue_depth` parameter. Expose via CLI. Benchmark with `bench.sh` at depths 1, 2, 4, 8, 16, 32. Plot tok/s vs depth.

**Expected impact:** 5-15% decode improvement from optimal queue depth. Typically 4-8 for NVMe.

**Risk:** LOW — configuration tuning, no code changes needed beyond the flag.

---

#### G12. Expert Load Deduplication Within Layer
**Paper: General MoE routing theory**

**Insight:** When K=8 experts are selected per token, there can be duplicates in some routing scenarios (e.g., near-identical scores). Deduplicating before loading saves redundant SSD reads.

**What to implement:** Before issuing preads for K experts, sort and scan for duplicates. If expert_indices[i] == expert_indices[j], skip loading expert j — just reuse expert i's weight buffer. Rarer than expected but zero-cost to check.

**Implementation:** In `async_pread_start()`, add a deduplication pass: `qsort(indices, K)`, scan for `indices[i]==indices[i-1]`, set `valid_tmp[dup_idx]=1` immediately. Reduce actual pread count. Track "dedup rate" metric.

**Expected impact:** 1-5% decode improvement. Mostly noise except for specific prompt patterns.

**Risk:** LOW — correctness-only change.

---

#### G13. Expert Eviction Priority — Co-occurrence Boost
**Paper: Apple "LLM in a Flash" cache insights**

**Insight:** When evicting an expert from cache, consider co-occurring experts — evicting expert X but keeping co-occurring expert Y is suboptimal if Y will need X soon.

**What to implement:** Add co-occurrence-aware eviction scoring. When choosing a victim for eviction, boost the score of experts whose co-occurring partners are currently cached. This keeps expert "teams" together in cache.

**Implementation:** In the cache eviction path, read `cooccur_122b.json` to get per-expert co-occurrence list. Compute eviction_score = recency_score + β * cooccur_cached_count. Evict the expert with lowest combined score. Add `--cooccur-eviction-beta <float>` flag.

**Expected impact:** 5-10% decode improvement in cache hit rate. Complements A2 (hot expert cache).

**Risk:** LOW — eviction policy change only.

---

#### G14. Metal Buffer Strategy — Private vs Shared
**Paper: Apple's Metal memory management insights**

**Insight:** Metal buffers can be created as `StorageModeShared` (CPU/GPU shared, periodically synced) or `StorageModePrivate` (GPU-only, requires explicit copy). The current expert buffers may be using the wrong mode, causing unnecessary synchronization overhead.

**What to implement:** Benchmark both buffer modes for expert weight storage. `StorageModeShared`: pread directly into the buffer, GPU reads it — simple but may have sync overhead. `StorageModePrivate`: pread into CPU buffer, blit to GPU buffer, GPU reads from private — extra copy but no sync stalls. Add `--metal-buffer-mode <shared|private>` flag.

**Implementation:** In `createMetalBuffer()`, try both modes. Add telemetry for buffer allocation time and GPU read time. Benchmark tok/s with each mode.

**Expected impact:** 5-15% decode improvement if private mode eliminates sync stalls.

**Risk:** LOW — measurement study first.

---

#### G15. CUDA/MPS-Style Priority Streams for I/O vs Compute
**Paper: General GPU I/O overlap theory (similar to CUDA streams)**

**Insight:** Metal command buffers are submitted serially by default. Using separate command buffer streams for I/O acknowledgment and compute allows maximum overlap — while GPU executes layer N's expert matvec, CPU processes layer N-1's I/O completion and submits layer N+1's preads.

**What to implement:** Restructure the per-layer loop into two interleaved streams: STREAM_A handles pread submission and I/O wait for layer L; STREAM_B handles GPU compute for layer L. Use `MTLCommandBuffer` completion handlers to coordinate stream synchronization. Effectively double-buffers the layer pipeline.

**Implementation:** In `fused_layer_forward()`, split into `submit_layer(L)` and `wait_compute(L)`. Use `dispatch_queue` for CPU-side work and `MTLCommandBuffer.completionBlock` for GPU-CPU handoff. Track pipeline bubble count.

**Expected impact:** 10-20% decode improvement from better I/O-compute overlap. Most impactful when I/O and compute durations are similar (which they are at ~50/50 split).

**Risk:** HIGH — significant code restructuring. Can cause subtle pipeline ordering bugs.

---

## Simplicity Criterion

All else being equal, simpler is better. A small improvement (+0.01 tok/s) that adds 200 lines of complex code is probably not worth it. A small improvement from DELETING code is always worth it. When evaluating whether to keep a change, weigh complexity cost against improvement magnitude.

## Experiment Strategy

**Recommended execution order:**

| Phase | Experiments | Rationale |
|---|---|---|
| 1 | **D2** (TurboQuant KV compression) | **TOP PRIORITY** — proven 4.6x KV compression, frees memory for expert cache, clear reference implementation to port |
| 2 | **C4** (Metal kernel study from llama.cpp) | **TOP PRIORITY** — mine years of community Metal optimizations, inform all kernel work |
| 3 | **A1** (simulator) | Accelerates everything — screen experiments offline |
| 4 | **B1.1** (K=0 prefill bound) | Quick, reveals theoretical TTFT floor |
| 5 | **D1** (memory budgeting) | Infrastructure — prevents compound interference |
| 6 | **A2** (hot expert cache) | Biggest decode potential, use simulator to guide |
| 7 | **B1.3** (full-attn-only prefill) | Best quality/speed TTFT ratio |
| 8 | **C1** (port 35B shader wins) | Proven improvements, dimension adaptation |
| 9 | **C3** (phase-specific kernels) | Prerequisite for efficient batched prefill |
| 10 | **A3** (expert repacking) | Low risk, builds on +14% |
| 11 | **B1.2** (prefill-K sweep) | Map the quality/TTFT curve |
| 12 | **A5** (LZ4 compression) | Validate ratio first — quick go/no-go |
| 13 | **A4** (predictive prefetch) | Moderate risk/reward |
| 14 | **E1** (K reduction quality gate) | Validate existing data |
| 15 | **B2** (batched prefill) | Highest effort, save for after C3 |
| 16 | **Everything else** | As time/results dictate |

Adapt based on results. If the cache (A2) works well, prioritize cache-related follow-ups (A4, F2). If prefill reduction (B1) shows huge TTFT wins, explore that path deeper (B2, C3). If LZ4 ratio is good (A5), pursue compression aggressively.

## Key Files Reference

| File | Purpose |
|---|---|
| `metal_infer/infer.m` | Main inference engine (~8900 lines) |
| `metal_infer/shaders.metal` | GPU compute kernels (~1400 lines) |
| `bench.sh` | Automated benchmark (starts server, runs inference, reports RESULT) |
| `quality_gate.sh` | 10-prompt quality comparison suite |
| `results.tsv` | Experiment log |
| `expert_index_122b.json` | Expert-to-layer mapping |
| `cooccur_122b.json` | Expert co-occurrence matrix |
| `car_table_122b.json` | CAR substitution table |
| `simulate.py` | Offline routing trace simulator (to be built in A1) |
| `cluster_experts.py` | Expert clustering script |
| `profile_experts.py` | Expert frequency profiling |
| `metal_infer/out_122b/` | 122B engine artifacts (weights + packed experts) |
| `~/models/flash-moe/Qwen3.5-122B-A10B-4bit/` | HuggingFace model config |
