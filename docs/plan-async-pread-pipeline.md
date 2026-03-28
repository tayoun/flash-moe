# Async Pread Pipeline — Implementation Notes

**Status:** Implemented (2026-03-25 to 2026-03-27)

## Architecture

```
Token T prefill → async pre-pread all 48 layers' predicted experts
                        ↓
Token T decode (layer 0) → wait for T's preads
                        ↓
Token T decode (layer 1) → wait for T's preads
                        ↓
...
Token T decode (layer 47) → wait for T's preads
                        ↓
Token T+1 prefill → async pre-pread all 48 layers' predicted experts
```

## Key Implementation Details

### Serialization (Critical)
**Multiple dispatches can be in flight simultaneously** — prefill starts new preads for T+1 while T's decode layers are still processing. All 48 layers share one `g_async_pread` state (`entries[]`, `generation`, `dispatch_group_t`).

**Solution:** `async_pread_start()` explicitly waits for any pending dispatch before starting a new one:
```c
if (g_async_pread.active > 0) {
    async_pread_wait();  // serialize: wait for previous dispatch
}
g_async_pread.active = 1;
g_async_pread.generation++;
```

### Generation Tagging
Each dispatch gets a unique `generation` number. Entries are tagged with the generation they belong to:
```c
// async_pread_start: tag entries with current generation
g_async_pread.entries[k].gen = gen;
g_async_pread.entries[k].done = 0;

// async_pread_wait: only update entries for the generation we waited for
for (int k = 0; k < g_async_pread.num_experts; k++) {
    if (g_async_pread.entries[k].gen == waited_gen) {
        g_async_pread.entries[k].done = valid_tmp[k];
    }
}
```

Matching loop validates by generation:
```c
if (g_async_pread.entries[p].done &&
    g_async_pread.entries[p].gen == g_async_pread.generation) {
    // Valid hit — this entry belongs to current generation
}
```

### Prediction Path (per layer during decode)
1. Wait for predicted preads: `async_pread_wait()`
2. Match predictions against actual routing (K experts selected by router)
3. For each actual expert:
   - **Hit:** Pre-loaded in `buf_B[p]` → use directly
   - **Miss:** Sync pread into `buf_multi_expert_data[k]` → 0.5–2ms per expert
4. GPU encode with expert buffers
5. Dispatch GPU command buffer (non-blocking)

### Misspread: Sync Pread (Not Thread Pool)
Misspread uses **synchronous pread** instead of the io_pool thread workers:
```c
// Miss experts are small (~5MB); sync pread is fast enough (<2ms)
for (int m = 0; m < miss_count; m++) {
    int k = miss_k_slots[m];
    int fd = expert_pick_fd(layer_idx, miss_ei[m], packed_fd);
    void *dst = [g_metal->buf_multi_expert_data[k] contents];
    ssize_t r = pread(fd, dst, esz, off);
    valid[k] = (r == (ssize_t)esz);
}
```
**Why:** The io_pool thread workers caused stack corruption when accessing main thread stack variables (`miss_k_slots[]`, `miss_ei[]`).

### io_pool_dispatch Fix
Workers stride by `g_num_io_threads`. If `num_tasks < g_num_io_threads`, workers would access out-of-bounds:
```c
// WRONG: workers 0..7 all check tasks[0], tasks[8], tasks[16]...
while (g_io_pool.tasks_completed < g_num_io_threads)  // DEADLOCK/HANG if miss_count < 8

// CORRECT: wait for actual number of dispatched tasks
while (g_io_pool.tasks_completed < num_tasks)
```

## Prediction Accuracy
- 122B model, K=8: typically **1 hit, 7 misses** per layer
- Higher K models (K=16, K=32) would have better hit rates
- Hit rate depends on routing stability across tokens

## Performance (122B, K=8, --predict --kv-compression turbo3)
| Metric | Without prediction | With prediction | Improvement |
|--------|-------------------|----------------|-------------|
| TTFT | 17.79s | 15.61s | 2.18s faster |
| Decode tok/s | 2.7 | 4.96 | 1.84× faster |
| Total (80 tokens) | ~29s | ~21s | ~30% faster |

## Bugs Fixed During Implementation
1. **io_pool_dispatch deadlock** — waited for wrong count
2. **Concurrent dispatch race** — explicit serialization added
3. **Generation collision** — tagged entries, only update matching gen
4. **Thread pool stack corruption** — switched to sync pread

## Files Changed
- `metal_infer/infer.m` — async pread, prediction matching, generation tagging
