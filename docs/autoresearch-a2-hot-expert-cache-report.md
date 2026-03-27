# A2: Hot Expert Cache with Composite Scoring

## Summary

Enhanced the malloc-based expert cache with composite scoring that combines recency (LRU) and frequency for smarter eviction decisions.

## Implementation

### New CLI Flag: `--cache-composite`

Enables composite eviction policy instead of pure LRU:

```bash
./infer --cache-mb 4096 --cache-composite ...
```

### Composite Scoring Formula

When `--cache-composite` is enabled, the eviction policy uses:

```
score = freq_count / age
```

Where:
- `freq_count`: number of times this cached expert has been accessed
- `age`: `current_counter - last_used + 1`

Higher score = keep, lower score = evict.

This approximates LRFU (Least Recently/Frequently Used) behavior.

### Code Changes

Added to `MallocExpertCache` struct:
- `uint32_t *freq_count`: per-entry access frequency counter

Modified functions:
- `malloc_cache_init`: allocate freq_count array
- `malloc_cache_lookup`: increment freq_count on hit
- `malloc_cache_insert`: reset freq_count=1 on new entry; composite eviction logic
- `malloc_cache_free`: free freq_count array

## Usage Examples

```bash
# Pure LRU (default)
./infer --cache-mb 4096 --k 6 ...

# Composite scoring (LRU + frequency)
./infer --cache-mb 4096 --cache-composite --k 6 ...
```

## Expected Behavior

Composite scoring should improve hit rates for workloads where:
- Some experts are accessed repeatedly (high frequency)
- Pure LRU would evict frequently-used but not recently-used experts

For uniform access patterns, composite scoring may not improve over LRU.

## Files Changed

- `metal_infer/infer.m`:
  - Added `g_cache_composite_scoring` flag
  - Added `freq_count` tracking to MallocExpertCache
  - Added `--cache-composite` CLI option (option 278)
  - Modified eviction to use composite score when enabled
