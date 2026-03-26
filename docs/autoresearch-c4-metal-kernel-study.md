# C4: Metal Kernel Study from llama.cpp

## Summary

Analyzed llama.cpp's `ggml-metal.metal` (~11K lines) for optimization patterns applicable to flash-moe's 122B inference engine.

## Key Findings

### 1. Fused RMSNorm Kernel (HIGH IMPACT)

**llama.cpp pattern:** `kernel_rms_norm_fuse_impl<T, F>` with template parameter F:
- F=1: RMSNorm only
- F=2: RMSNorm + element-wise multiply (with weights)
- F=3: RMSNorm + multiply + add

**Our current approach:** Two separate kernels:
1. `rms_norm_sum_sq` - compute sum of squares
2. `rms_norm_apply` - normalize and multiply by weights

**Opportunity:** Fuse into single kernel. Saves one kernel dispatch + one memory pass over the buffer.

**Expected gain:** 5-10% reduction in RMSNorm latency (48 layers × 2 norms/layer = 96 norm operations per token)

### 2. Float4 Vectorized Operations

**llama.cpp pattern:** `kernel_soft_max_4`, `kernel_ssm_conv_f32_f32_4`
- Process 4 elements per thread using float4
- Reduces thread count, improves memory coalescing
- Uses `dot(float4, float4)` for efficient 4-wide reduction

**Our current approach:** Most kernels use scalar float operations

**Opportunity:** Add float4 variants for:
- Softmax routing
- Element-wise operations
- Residual add

### 3. Multi-Row Processing in MatVec

**llama.cpp pattern:** NR0 template parameter processes multiple output rows per threadgroup
- `N_R0_Q4_0 = 2` or higher
- Multiple `ax[row]` pointers processed in same threadgroup
- Better weight reuse, fewer threadgroups

**Our current approach:** `ROWS_PER_TG = 8` in v3 kernel (already similar)

**Status:** We already implement this pattern.

### 4. SIMD Group Matrix Operations (Flash Attention)

**llama.cpp pattern:** Uses `simdgroup_half8x8`, `simdgroup_float8x8` for flash attention
- Hardware-accelerated 8x8 matrix operations
- Enables fully fused flash attention in single kernel

**Our current approach:** Split attention: separate score computation and value accumulation

**Opportunity:** Implement fused flash attention with SIMD group matrices. However, this is a large undertaking better suited for a dedicated experiment.

### 5. Quantized Flash Attention

**llama.cpp pattern:** `kernel_flash_attn_ext_q4_0_dk*`
- Flash attention directly on quantized KV cache
- Dequantize inline during attention computation
- Supports turbo3 (our TurboQuant format!)

**Our current approach:** Dequantize KV before attention

**Opportunity:** Inline dequantization in attention kernel. Would eliminate KV dequant buffer traffic.

### 6. FOR_UNROLL Pragma

**llama.cpp pattern:**
```metal
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
FOR_UNROLL (short i = 0; i < 8; i++) { ... }
```

**Our current approach:** Manual unrolling or compiler-default

**Opportunity:** Add `#pragma clang loop unroll(full)` hints to critical inner loops.

## Recommended Ports (Priority Order)

| Priority | Optimization | Effort | Expected Gain |
|----------|-------------|--------|---------------|
| 1 | Fused RMSNorm (sum_sq + apply) | LOW | 5-10% norm time |
| 2 | FOR_UNROLL pragma hints | LOW | 2-5% kernel time |
| 3 | Float4 softmax routing | MEDIUM | 3-5% routing time |
| 4 | Inline KV dequant in attention | HIGH | 10-15% attention time |
| 5 | Full flash attention fusion | VERY HIGH | 30%+ attention time |

## Implementation Plan

### Port 1: Fused RMSNorm (this session)

Replace:
```metal
kernel void rms_norm_sum_sq(...) { /* compute sum_sq */ }
kernel void rms_norm_apply(...) { /* normalize */ }
```

With single fused kernel:
```metal
kernel void rms_norm_fused(
    device const float* x,
    device const uint16_t* weight,  // bf16
    device float* out,
    constant uint& dim,
    constant float& eps,
    ...
) {
    // Step 1: Parallel sum of squares
    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }
    float sum_sq = simd_reduce(acc);  // SIMD reduction

    // Step 2: Compute scale and apply with weight
    float rms = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = x[i] * rms * bf16_to_f32(weight[i]);
    }
}
```

This eliminates one command buffer encode + GPU round trip per norm operation.

## Impact Analysis for 122B

**Critical finding:** The 122B model is **I/O bound** (70-74% of per-layer time is SSD reads). GPU compute optimizations have limited end-to-end impact:

- GPU compute time: ~26-30% of total
- Best-case kernel speedup: 20-30%
- End-to-end impact: 26% × 30% = **~8% max**

For the 122B optimization campaign, prioritize I/O experiments (A1-A5, B1-B2) over GPU kernel work (C1-C4).

## Conclusion

The llama.cpp kernel study identified several applicable optimizations. However, for the I/O-bound 122B model, these GPU-side improvements would yield <10% end-to-end gain.

**Recommendation:** Defer kernel ports (fused RMSNorm, flash attention) until I/O bottleneck is addressed. Focus on A1 (simulator), B1.1 (K=0 prefill), D1 (memory budgeting), and A2 (hot expert cache).

## Reference Files

- `llama_turboquant_ref/ggml/src/ggml-metal/ggml-metal.metal` (10922 lines)
- `llama_turboquant_ref/ggml/src/ggml-metal/ggml-metal-impl.h` (kernel args structs)
