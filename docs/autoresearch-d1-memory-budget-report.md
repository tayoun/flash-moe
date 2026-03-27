# D1: Three-Pool Memory Budgeting

## Summary

Added explicit memory pool configuration and runtime memory budget reporting. The three pools are:

1. **Expert cache** - Malloc-based LRU cache for frequently accessed experts
2. **KV cache** - GPU attention buffers for full-attention layers
3. **Delta-net state** - Persistent linear attention state per layer

## Implementation

### New CLI Flag: `--cache-mb N`

Specifies expert cache size in MB instead of entry count. Automatically converts to entries based on expert size:

```
./infer --cache-mb 4096  # 4GB expert cache
```

Output:
```
[config] Cache budget: 4096 MB → 809 entries (4294.5 MB actual)
```

### Memory Budget Summary

At startup, displays breakdown of all runtime memory pools:

```
--- Memory Budget (D1) ---
  Expert cache:   4295 MB (809 entries)
  KV cache:       403 MB (12 full-attn layers × 8192 seq)
  Delta-net:      151 MB (36 linear layers)
  Total runtime:  4849 MB
--------------------------
```

With TurboQuant KV compression:
```
  KV cache:       89 MB (12 full-attn layers × 8192 seq, TurboQuant)
```

## Memory Pool Details

### Expert Cache (`--cache-mb` or `--malloc-cache`)
- Each entry holds one expert (5.3 MB for 4-bit, 2.65 MB for 2-bit)
- LRU eviction policy
- Zero-copy Metal buffer wrappers for GPU dispatch

### KV Cache (fixed at GPU_KV_SEQ=8192)
- 12 full-attention layers × 2 (K and V) × 512 dim × 8192 seq × 4 bytes = 403 MB
- With TurboQuant: ~89 MB (4.5x compression)

### Delta-net State
- 36 linear layers × persistent state buffers
- ~4.2 MB per layer = 151 MB total

## Example Configurations

| Config | Expert Cache | KV Cache | Delta-net | Total |
|--------|-------------|----------|-----------|-------|
| Minimal | 0 MB | 403 MB | 151 MB | 554 MB |
| 4GB cache | 4295 MB | 403 MB | 151 MB | 4849 MB |
| 8GB cache + TurboQuant | 8590 MB | 89 MB | 151 MB | 8830 MB |

## Files Changed

- `metal_infer/infer.m`:
  - Added `--cache-mb` CLI flag (option 277)
  - Added MB-to-entries conversion after config load
  - Added memory budget summary printout at startup
