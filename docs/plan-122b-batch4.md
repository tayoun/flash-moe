# Batch 4 Experiment Plan — 122B Advanced Optimizations

## Why Revisit 35B Anti-Results

The 35B model (3.4GB experts) fits in 16GB RAM. The OS page cache handles everything.
Custom caches, prefetch, compression — all competed with a system that was already working.

The 122B model (61GB experts) is **18x** page cache capacity. The system is NOT working —
73.7% of per-layer time is I/O. Every optimization that failed on 35B because "the OS
handles it better" deserves re-evaluation, because the OS demonstrably *can't* handle 122B.

### Key differences: 35B vs 122B on M4 16GB

| Metric | 35B | 122B |
|--------|-----|------|
| Expert data size | 3.4 GB | 61 GB |
| Fits in RAM? | Yes (47%) | No (381%) |
| Page cache hit rate | ~71% | ~5-10% (estimated) |
| Per-token expert reads | ~28 MB | ~2 GB |
| I/O fraction of latency | ~56% | ~74% |
| Sustained tok/s | 11.5 | 1.28 |
| I/O-bound? | Moderate | Severely |

---

## Current Best Results (for reference)

| Config | tok/s | Notes |
|--------|-------|-------|
| Baseline (K=8) | 1.12 | No optimizations |
| Clustering (K=8) | 1.28 | +14% |
| K=6 | 1.60 | +30%, needs quality gate |
| K=5 | 1.92 | +56%, needs quality gate |
| K=4 | 4.04 | +228%, quality degraded |

---

## Experiment 1: Trace-Driven Offline Simulator

**Priority: FIRST — accelerates everything else**

### Why
Each benchmark takes 5-10 min (server startup + inference). Testing cache policies,
disk layouts, and prefetch strategies takes hours. A simulator runs in seconds.

### Implementation
1. Record routing traces: for each token, log which experts were selected per layer
   (the router already computes this — just dump to file)
2. Build `simulate.py`:
   - Input: routing trace + experiment config (cache size, policy, disk layout)
   - Simulate: per-token cache hits/misses, SSD reads, prefetch hits
   - Output: estimated tok/s, cache hit rate, SSD read volume per token
3. Validate: run simulator predictions vs actual benchmarks for K=8 and K=6
4. Use simulator to screen all subsequent experiments before running real benchmarks

### Deliverable
- `--trace <path>` flag on infer to dump routing decisions
- `simulate.py` with cache/layout/prefetch simulation
- Validation report: simulator vs actual for known configs

---

## Experiment 2: Budget-Aware Hot Expert Cache

**Revisiting the biggest 35B anti-result with discipline**

### 35B anti-result
Custom caches (500-3000 entries, 3.5-21 GB Metal LRU) all made things worse:
- Metal buffers wire memory → reduces OS page cache
- Compressor thrash: 60-130K decompressions/sec with cache active
- "Trust the OS" won at 5.74 tok/s

### Why 122B is different
On 122B, the OS page cache has ~5-10% hit rate (61GB working set, ~12GB available).
"Trust the OS" gives us 1.12 tok/s. There's nothing to trust — the OS is thrashing.

### Design (key constraint: explicit memory budget)
```
Total RAM:       16 GB
Kernel/system:   ~2 GB
Shared weights:  3.5 GB (model_weights.bin, mmap'd)
KV cache:        ~0.2 GB (small for 122B single-batch)
Budget for cache: 4 GB (leaves ~6 GB for OS page cache + overhead)
Expert size:     5.3 MB each
Cache capacity:  ~750 experts (out of 12,288 total = 256 experts × 48 layers)
```

### Policy: ARC (Adaptive Replacement Cache)
Not simple LRU. Use ARC (same family as the OS page cache's CLOCK-Pro):
- Tracks both recency and frequency
- Adapts the balance between them based on workload
- Well-studied, O(1) operations, no tuning parameters

### Implementation rules (avoid 35B mistakes)
- Use `malloc`, NOT Metal buffers (avoid wiring pages)
- Cap at exactly 4 GB (configurable via `--cache-mb`)
- Copy expert data into cache on hit for next use (memcpy, not pointer reuse)
- Expert reads still go through pread → cache on miss
- No mlock on cache memory (let OS swap if needed)
- Track: cache hit rate, memory pressure (vm_stat), per-token cache stats

### Test matrix
- 2 GB cache (~375 experts)
- 4 GB cache (~750 experts) — recommended
- 6 GB cache (~1125 experts) — aggressive
- Each with: ARC, LRU, frequency-only

### Validation with simulator first
Run experiment 1's simulator to predict which cache size/policy is optimal
before burning 30 min on real benchmarks.

---

## Experiment 3: Predictive Prefetch (CPU-Phase Only)

**Revisiting with the memory controller constraint in mind**

### 35B anti-result
Temporal expert prediction: 25% hit rate, -18% performance.
"SSD DMA and GPU compute share the same memory controller — cannot be profitably overlapped."

### Why revisit
The insight: don't prefetch during GPU compute. Prefetch during CPU-only phases:
- After GPU attention completes, before next layer starts: ~0.01ms CPU flush
- During routing softmax + topK: ~0.003ms CPU
- During post-expert combine + norm: CPU orchestration time

These windows are small. But if we can predict even 2-3 experts correctly per layer
and start the pread during a CPU phase, those reads complete before the GPU needs them.

### Better prediction
- Don't use temporal (previous token's experts). Use co-occurrence:
  "if expert 42 was selected in layer N, experts {78, 156, 203} are likely in layer N+1"
- Co-occurrence data already exists from clustering
- Expected hit rate: 40-60% (vs 25% temporal)

### Implementation
1. Build co-occurrence predictor from existing profiling data
2. After each layer's routing, predict next layer's experts
3. Issue pread for predicted experts during CPU phases only
4. If expert arrives before GPU needs it → cache hit (0ms)
5. If not → normal pread path (no worse than baseline)

### Guard rails
- Prefetch budget: max 4 experts per layer (avoid SSD contention)
- Only prefetch during explicitly identified CPU windows
- Monitor: GPU stall rate, prefetch hit rate, SSD utilization overlap

---

## Experiment 4: Per-Expert Precision Tiers

**Novel — not previously tested**

### Concept
Not all experts are equal. Some are called frequently (10-20x average), others
are rare (1-2 tokens in a 256-token run). Keep hot experts at full 4-bit quality,
store cold experts at 2-bit.

### Benefit
- Cold experts at 2-bit = 50% less SSD read per cold expert
- Hot experts stay at 4-bit = quality preserved for the most impactful weights
- Net I/O reduction depends on hot/cold ratio

### Implementation
1. Profile expert frequency across diverse prompts (1000+ tokens)
2. Rank experts by frequency per layer
3. Top 50% → keep at 4-bit (already in packed_experts/)
4. Bottom 50% → quantize to 2-bit (using existing 2-bit support)
5. At runtime: check expert precision tier, use appropriate dequant kernel
6. Pack files: interleave 4-bit and 2-bit experts (sort by frequency for locality)

### Risk
- Quality degradation from 2-bit on cold experts
- Complexity: mixed-precision dequant in the same layer
- Need careful quality gate

---

## Experiment 5: Lossless On-Disk Compression

**Revisiting LZ4 with 122B math**

### 35B anti-result
LZ4 cache: -13% (decompress overhead > warm cache savings)

### Why 122B is different
- 35B: I/O was 56% of latency, page cache hit 71% → compression competes with fast cache
- 122B: I/O is 74% of latency, page cache hit ~5% → almost every read is from SSD
- LZ4 decompression: ~40 GB/s throughput
- M4 SSD sequential: ~5.5 GB/s cold
- If compression ratio is 1.5x: effective SSD throughput = 5.5 × 1.5 = 8.25 GB/s
- Decompress 5.3 MB expert: 0.13 ms (vs ~1 ms pread)

### Implementation
1. Compress each expert blob with LZ4 (per-expert, not per-layer)
2. Store compressed experts in layer files with an offset index
3. At runtime: pread compressed blob → LZ4 decompress into existing buffer
4. Measure: compression ratio on 4-bit quantized expert data
5. Benchmark: compare pread raw vs pread compressed + decompress

### Prerequisites
- Measure actual compression ratio first (4-bit quantized data may not compress well)
- If ratio < 1.3x → skip (overhead not worth it)

---

## Experiment 6: Improved Expert Clustering

**Optimize beyond greedy nearest-neighbor**

### Current result
Greedy clustering: 11.4x adjacency improvement, +14% performance.

### Better approaches
1. **Simulated annealing**: start from greedy solution, randomly swap expert positions,
   accept if co-occurrence locality improves (or with probability if SA temperature allows)
2. **Spectral clustering**: treat co-occurrence matrix as graph, use spectral methods
   to find communities of frequently co-activated experts
3. **Multi-prompt profiling**: current clustering uses 541 tokens from one prompt.
   Profile with 5-10 diverse prompts (2000+ tokens) for more robust co-occurrence data
4. **Cross-layer awareness**: if experts A,B,C often fire together across layers 5-8,
   pack them at similar offsets across those layer files for SSD read pattern locality

### Validation: use simulator (experiment 1) to evaluate layouts before repacking

---

## Experiment 7: Multi-Request Cache Affinity (Optional)

**Only relevant for production multi-user serving**

When serving multiple concurrent requests:
- Route requests to maximize expert cache hits
- If request A just used experts {42, 78} in layer 5, and request B needs {78, 156},
  schedule B immediately after A on the same thread → expert 78 is still hot
- Requires request queuing and routing-aware scheduling

### Skip for now — only matters post-optimization when deploying as a service.

---

## Execution Order

| Phase | Experiments | Rationale |
|-------|------------|-----------|
| 1 | Trace simulator (#1) | Accelerates everything |
| 2 | Simulate cache + layout options | Screen before building |
| 3 | Budget-aware cache (#2) | Biggest potential if math works |
| 4 | Improved clustering (#6) | Low risk, builds on proven +14% |
| 5 | Compression test (#5) | Quick to measure ratio |
| 6 | Predictive prefetch (#3) | Moderate risk, moderate reward |
| 7 | Precision tiers (#4) | High effort, high potential |

## Success Criteria
- Any single experiment that beats K=8 clustering (1.28 tok/s) is a keeper
- Compound target: 2.0+ tok/s at K=8, or 3.0+ tok/s at K=6
- All keepers must pass the 10-prompt quality gate
