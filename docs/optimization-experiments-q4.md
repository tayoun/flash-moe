# Q4 Optimization Experiments (M3 Baseline + M4 Port)

## Summary

This document now tracks two phases:

1. Original M3 Max optimization pass (48GB), culminating at 4.4 tok/s.
2. M4 16GB port + optimization pass, culminating at 11.5 tok/s and 2.5s TTFT.

## Results Comparison

| Phase | Hardware | K | Sustained tok/s | TTFT | Outcome |
|---|---|---:|---:|---:|---|
| Original baseline | M3 Max 48GB | 4 | 4.4 | ~5.6s | First production-quality release |
| Current best | M4 16GB | 6 | **11.5** | **2.5s** | 2.6x faster on lower-cost hardware |

## M3 Max Track (Historical)

The original optimization series established:

- Stable 4-bit quality with tool calling.
- SSD streaming as the practical path for routed experts.
- Baseline pipeline behavior and first-wave kernel tuning.

Representative kept/discarded experiments from that phase remain relevant for context, including cache strategy and prefetch behavior under Apple unified memory.

## M4 16GB Port: What Changed

The M4 pass focused on end-to-end token latency rather than isolated microbench gains.

### 1. `tg128` Matvec Kernels

- Adopted `tg128` kernel variants in hot matvec paths.
- Increased effective threadgroup utilization on M4.
- Reduced matvec-dominant per-layer latency in routed expert compute.

### 2. Encoder Coalescing

- Coalesced encode/prefill work to reduce synchronization and launch overhead.
- Lowered non-matvec overhead in prompt processing and token-step setup.
- Major contributor to TTFT reduction.

### 3. Kernel Fusion

- Fused latency-critical adjacent operations in hot paths.
- Reduced intermediate memory traffic and CPU-GPU round trips.
- Improved sustained decode throughput in steady state.

### 4. K Became Runtime-Configurable

- K was previously tuned around fixed M3 assumptions.
- Runtime now supports configurable `--k`; current M4 production profile uses `K=6`.
- This enables hardware-specific tradeoff tuning without code changes.

## Current M4 Production Profile

- Model: Qwen3.5-35B-A3B-4bit
- Hardware: Mac mini M4, 16GB unified memory
- Routing: `K=6`
- Performance: 11.5 tok/s sustained, 2.5s TTFT
- Stability: zero crashes in benchmark and tool-calling use

## Lessons

- M-series generation changes can overturn previous kernel assumptions.
- Launch/sync overhead matters at this scale; coalescing and fusion compound.
- K must be a runtime knob to preserve portability across Apple Silicon tiers.
- End-to-end throughput improvements can significantly exceed per-kernel microbench deltas when multiple bottlenecks are addressed together.
