# Batch 3 Experiment Plan — 122B Optimizations

Inspired by @danpacary's Kimi-K2 (1T param) thread + Flash-MoE 397B results.

## Current State
- Best: 1.28 tok/s (clustering only)
- Baseline: 1.12 tok/s
- System: 73.7% I/O bound (11.54ms/layer expert_io)
- Hardware: M4 Mac mini, 16GB, K=8, 48 layers, 256 experts

## Critical Context: Why 122B ≠ 35B for Page Cache

35B experts: 3.4GB total → fits in 16GB → page cache works → F_NOCACHE hurts
122B experts: 61GB total → 3.8x RAM → page cache thrashes → F_NOCACHE should help

Per-token expert data read: 8 experts × 48 layers × 5.3MB = **~2 GB per token**
Shared weights (must stay hot): 3.46 GB (model_weights.bin)
Available page cache: ~12GB (16GB - kernel - shared weights)

The page cache cannot hold even ONE full pass of expert reads (2GB).
Every token pollutes 2GB of page cache with experts that won't be needed next token
(different routing = different experts). This EVICTS the shared weights.

---

## Experiment 1: F_NOCACHE on Expert Files (HIGHEST PRIORITY)
**Expected: +30-50%** (tweet reported +46% on similar setup)

Re-enable F_NOCACHE on expert layer fds. The 35B anti-result doesn't apply here
because 122B experts vastly exceed RAM.

Implementation:
- Re-activate `layer_fds_cold[i]` with `fcntl(fd, F_NOCACHE, 1)`
- Route ALL expert reads through the F_NOCACHE fd (not just first reads)
- This prevents expert reads from polluting page cache
- Shared weights (mmap'd model_weights.bin) stay hot in page cache

### Variant 1a: Full F_NOCACHE (all expert reads bypass cache)
### Variant 1b: Tiered — cold fd for first read, warm fd for repeats (original design)
### Variant 1c: F_NOCACHE + clustering compound

---

## Experiment 2: FMA Kernel Verification
**Expected: already done (verify)**

The tweet mentions +12% from FMA rearrangement:
`(nibble * scale + bias) * x` → `fma(nibble, scale*x, bias*x)`

Our shaders.metal already has this optimization (lines 310-315 in tg128 kernel).
Verify the NON-tg128 kernels also use FMA. If the fallback kernel (lines 83-84, 133-140)
still uses the naive `float(nibble) * scale + bias` pattern, port FMA there.

---

## Experiment 3: Deferred GPU Expert Compute
**Expected: +5-15%**

From the tweet's pipeline: "CMD3 is submitted without waiting. GPU executes while CPU
prepares next layer."

Check if our 122B pipeline defers expert GPU work or waits synchronously.
If synchronous: convert to deferred/pipelined execution.

---

## Experiment 4: K Reduction (K=6 vs K=8)
**Expected: +25-50% (if quality holds)**

The tweet/FOMOE found K=4 was viable for 397B (designed for K=10).
Our 122B uses K=8 (model default). Test:
- K=7: 12.5% fewer expert reads
- K=6: 25% fewer expert reads (35B production setting)
- K=5: 37.5% fewer
- K=4: 50% fewer

Each K reduction directly reduces SSD reads proportionally.
Must pass quality gate (10-prompt suite at temperature=0).

---

## Experiment 5: Disable F_RDAHEAD (Already Done — Verify)
The code already calls `fcntl(layer_fds[i], F_RDAHEAD, 0)`.
Verify this is actually effective. The tweet mentions readahead as wasteful
for random expert access patterns.

---

## Experiment 6: Shared Weight Pinning
**Expected: +5-10%**

Use `mlock()` or `madvise(MADV_WILLNEED)` on the 3.46GB model_weights.bin mmap
to ensure shared weights are NEVER evicted from page cache by expert reads.
This is complementary to F_NOCACHE but helps even without it.

---

## Experiment 7: Reduce Expert Size via 2-bit Quantization
**Expected: +50-100% (if quality holds)**

2-bit experts are half the size of 4-bit. Same architecture, half the I/O.
The 35B repo already supports 2-bit. Test on 122B if 2-bit weights exist.
Quality concern: 2-bit broke JSON/tool calling on 35B.

---

## Experiment 8: Compound Winners
Stack all winning experiments:
- F_NOCACHE + clustering + K reduction + FMA + shared weight pinning
- Measure compound effect

---

## Execution Priority

| # | Experiment | Expected | Effort | Priority |
|---|-----------|----------|--------|----------|
| 1 | F_NOCACHE on experts | +30-50% | Low | **1st** |
| 4 | K reduction (6→4) | +25-50% | Low | **2nd** |
| 6 | Shared weight mlock | +5-10% | Low | **3rd** |
| 2 | FMA verify non-tg128 | +0-12% | Low | **4th** |
| 3 | Deferred GPU compute | +5-15% | Medium | **5th** |
| 7 | 2-bit experts | +50-100% | Medium | **6th** |
| 8 | Compound stack | Sum | Low | **Last** |

## Theoretical Ceiling (optimistic)
Starting: 1.28 tok/s (clustering)
+ F_NOCACHE (+40%): ~1.79 tok/s
+ K=6 (+25%): ~2.24 tok/s
+ Shared pinning (+7%): ~2.40 tok/s
+ FMA fix (+5%): ~2.52 tok/s
+ Deferred GPU (+10%): ~2.77 tok/s

Realistic target: **2.0-2.5 tok/s** with compound wins.
3 tok/s likely requires K reduction to 5 or lower + quality holding.
