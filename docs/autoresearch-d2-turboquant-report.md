# D2: TurboQuant KV Cache Compression Report

## Summary

Experiment D2 implements TurboQuant (ICLR 2026) KV cache compression for the 122B model. The algorithm uses PolarQuant 3-bit quantization with Walsh-Hadamard rotation to achieve ~4.6x compression vs fp16 with >98% cosine similarity.

## Status: Infrastructure Complete, Integration Partial

### Completed
- [x] Python validation script (`validate_turboquant.py`)
- [x] Metal kernels (`turbo3_quantize_kv_256`, `turbo3_dequantize_kv_256`)
- [x] CLI flag `--kv-compression turbo3`
- [x] Compressed buffer allocation (38.5 MB vs 201 MB fp32)
- [x] Quantize dispatch in KV write path

### Pending
- [ ] Dequantize dispatch before attention reads
- [ ] Replace fp32 KV buffers with compressed-only storage
- [ ] Full memory savings validation

## Validation Results

### Kurtosis Reduction (Walsh-Hadamard Transform)
| Dimension | Raw Kurtosis | Rotated Kurtosis | Reduction |
|-----------|-------------|------------------|-----------|
| 256 (122B head_dim) | 255.8 | 0.96 | 267x |
| 128 (reference) | 392.9 | 3.04 | 129x |

The WHT successfully Gaussianizes the KV tensor distribution (target: kurtosis ~0).

### Compression Quality
| Bit Width | Cosine Similarity | MSE | Compression vs fp16 |
|-----------|------------------|-----|---------------------|
| 3-bit | **98.29%** | 0.0077 | **5.12x** |
| 4-bit | 99.54% | 0.0020 | 3.88x |

### Memory Savings (122B @ 8K context)
| Storage | Size | Compression |
|---------|------|-------------|
| fp32 (current) | 403 MB | 1.0x |
| fp16 (baseline) | 201 MB | 2.0x |
| **TurboQuant 3-bit** | **44 MB** | **9.1x** |
| Memory freed vs fp16 | 157 MB | — |

The 157 MB freed can be used for expert cache (A2), improving decode throughput.

## Implementation Details

### Metal Kernels (shaders.metal)
```metal
// Fast Walsh-Hadamard Transform for 256 elements
// O(n log n) = 2048 ops vs O(n^2) = 65536 for dense matvec
inline void turbo_fwht_256(thread float *x);

// Quantize: norm extraction -> WHT rotation -> 3-bit centroid lookup
kernel void turbo3_quantize_kv_256(
    device const float* kv_fp32,
    device block_turbo3_256* kv_compressed,
    constant uint& n_heads,
    uint head_idx [[thread_position_in_grid]]
);

// Dequantize: centroid lookup -> inverse WHT -> norm rescale
kernel void turbo3_dequantize_kv_256(...);
```

### Storage Format (98 bytes per 256-element head)
- `half norm` (2 bytes): original ||x||_2
- `uint8_t qs[96]` (96 bytes): 256 * 3-bit = 768 bits packed

### CLI Usage
```bash
./infer --model ... --kv-compression turbo3
```

## Architecture Notes

The 122B model has:
- **12 full-attention layers** (with real KV cache)
- **36 linear-attention/delta-net layers** (no KV cache)
- `head_dim = 256`, `num_kv_heads = 2`
- KV dim per token per layer = 2 * 256 = 512 floats

TurboQuant applies only to the 12 full-attention layers.

## Reference Implementation

Ported from:
- Python: `turboquant_ref/turboquant/` (rotation.py, polar_quant.py, turboquant.py)
- C/Metal: `llama_turboquant_ref/ggml/src/ggml-turbo-quant.c`, `turbo-wht.h`

## Next Steps

1. **Dequantize Integration**: Add GPU dispatch before `attn_scores_batched` reads KV
2. **Compressed-Only Storage**: Remove fp32 KV buffers when TurboQuant enabled
3. **Quality Gate**: Run `quality_gate.sh` to validate output quality
4. **Expert Cache Synergy**: Test with A2 (hot expert cache) to measure compound benefit

## Commits

- `b4423c0`: Add TurboQuant KV cache compression infrastructure
- `d95716f`: Add TurboQuant quantize dispatch in KV cache write path
