# Plan: 122B Model Adaptation + CAR Integration

## Goal
Run Qwen3.5-122B-A10B at usable speeds (target: 3+ tok/s) on M4 Mac mini (16GB) by:
1. Adapting Flash-MoE engine for the 122B architecture
2. Integrating FOMOE's Cache-Aware Routing (CAR) to reduce SSD reads
3. Running autoresearch loop to optimize

## Why CAR Is Critical for 122B on 16GB

**The math without CAR:**
- Expert size at 4-bit: ~6.3MB each (moe_intermediate=1024, group_size=64)
- Per-token I/O at K=8: 48 layers × 8 experts × 6.3MB = **2.4GB from SSD**
- M4 SSD: ~4 GB/s sustained → 600ms per token = **1.7 tok/s max**
- Non-expert weights: ~5GB mmap'd → only ~9GB left for page cache
- 60GB+ expert data with 9GB cache = ~15% hit rate → nearly all reads go to SSD

**With CAR (estimated):**
- CAR substitution at threshold=0.35 avoids ~40-50% of SSD reads (based on FOMOE data)
- Effective I/O drops to ~1.2-1.4GB per token
- Target: **3-5 tok/s** (2-3x improvement over naive streaming)

## Architecture Comparison

| | 35B-A3B (current) | 122B-A10B (target) |
|---|---|---|
| Layers | 40 (30 linear + 10 full) | 48 (36 linear + 12 full) |
| Hidden size | 2048 | 3072 |
| Attn heads | 16 | 32 |
| KV heads | 2 | 2 |
| Head dim | 256 | 256 |
| Experts | 256, K=8 default | 256, K=8 default |
| MoE intermediate | 512 | 1024 |
| Shared expert intermediate | 512 | 1024 |
| Expert size (4-bit) | ~1.7MB | ~6.3MB |
| Full attn interval | every 4th layer | every 4th layer |
| Linear attn key heads | 4 | 16 |
| Linear attn value heads | 16 | 64 |
| Linear key head dim | 128 | 128 |
| Linear value head dim | 128 | 128 |
| Conv kernel dim | 4 | 4 |
| Non-expert weights | ~1.4GB | ~5GB (est.) |

**Key differences that require engine changes:**
- Larger hidden_size (2048→3072): all projection matrices change dimensions
- More attention heads (16→32): attention buffer sizes change
- Larger MoE intermediate (512→1024): expert kernel dispatch sizes change
- More linear attention heads (4→16 key, 16→64 value): delta-net state matrices larger
- 48 layers vs 40: buffer allocation, loop bounds

## Phases

### Phase 1: 122B Engine Adaptation (prerequisite)

**1.1 — Update weight extraction scripts**
- Modify `extract_weights_35b.py` → `extract_weights_122b.py`
- Handle new tensor dimensions (hidden=3072, heads=32, etc.)
- Update tensor manifest format for new shapes
- Output: `metal_infer/out_122b/model_weights.bin` + `.json`

**1.2 — Update expert repacking**
- Modify `repack_experts_35b.py` → `repack_experts_122b.py`
- Update `build_expert_index_35b.py` → `build_expert_index_122b.py`
- Expert size changes: moe_intermediate=1024 → new packed size
- Output: `packed_experts/` with 48 layer files

**1.3 — Update inference engine (infer.m)**
- Config parsing: read new dimensions from config.json automatically
- Already reads most params dynamically (`num_experts`, `num_experts_per_tok`, etc.)
- Need to verify: hidden_size, num_attention_heads, linear attention params
- Metal buffer allocation: scale for hidden=3072, heads=32
- Delta-net state: 64 value heads × 128 dim = 8192-dim state (vs 2048 for 35B)
- GPU attention buffers: larger KV caches for 32 heads
- Projection matrices: all matvec dimensions change
- Test: build, load weights, generate 1 token without crash

**1.4 — Update Metal shaders (if needed)**
- Matvec kernels: should be dimension-agnostic (parameterized)
- tg128 heuristics: may need retuning for new dimensions
- Expert kernels: intermediate=1024 changes dispatch sizes

**1.5 — Validate baseline**
- Build and run bench.sh with 122B
- Record baseline tok/s and TTFT
- Verify output quality (coherent text, not garbage)
- Expected: 1-2 tok/s without CAR

### Phase 2: CAR Integration (the big win)

**2.1 — Implement explicit expert cache**
Currently Flash-MoE uses "trust the OS" page cache. For CAR we need an explicit cache layer so we can query "is expert X cached?" before routing.

Port from FOMOE (`expert_cache.c`):
- `ram_cache_t`: LRU cache with per-layer partitioning
  - `map[layer * n_experts + expert_id]` → slot index or -1
  - `slot_ts[]` for LRU eviction
  - `slots_per_layer` = total_cache_slots / n_layers
- Cache sizing: with ~9GB free RAM, allocate ~8GB for expert cache
  - 8GB / 6.3MB per expert = ~1,270 slots
  - 1,270 / 48 layers = ~26 slots per layer (out of 256 experts)
  - ~10% cache coverage per layer — CAR substitution fills the gap

**2.2 — Implement CAR algorithm**
Port from FOMOE (`car.c`), simplified for single-GPU / no VRAM tier:
- After router selects K experts, check each against cache
- For uncached experts: find highest-scoring cached alternative
- If score_ratio >= threshold, substitute
- Dampening mode: scale substitute weight by ratio
- Renormalize routing weights after substitution
- Configurable: `--car-threshold F` (0.35 recommended, 1.0 = disabled)

**2.3 — Implement background backfill**
Port from FOMOE (`prefetch.c`):
- Background thread loads substituted experts during idle phases
- Priority: highest router score first
- Idle window: during attention compute (GPU busy, SSD idle)
- Result: substituted expert becomes cached for next token
- Reduces CAR's accuracy penalty over time

**2.4 — Implement frequency profiling**
Port from FOMOE (`freq_profile.c`):
- Offline pass: run N prompts, record per-layer expert frequency
- Output: `122b.freq` file
- At startup: seed cache with most frequent experts
- Warm cache = fewer substitutions in early tokens

**2.5 — Implement warmup phase**
- `--car-warmup N`: force no substitutions for first N tokens
- All experts load from SSD during warmup, seeding cache
- After warmup, CAR kicks in with a populated cache

### Phase 3: Autoresearch Loop

Once Phase 1+2 are working, create `autoresearch/122b` branch and run the experiment loop with:

**Immutable:**
- K=8 (default for 122B, may test K=6 later)
- Compiler flags: -O3 -flto -ffast-math -fno-math-errno

**Tunable parameters (for autoresearch):**
- CAR threshold (0.0–1.0)
- CAR dampening on/off
- Cache size allocation
- Backfill batch size
- Warmup token count
- Frequency profile seeding vs cold start
- Expert cache slot partitioning strategy
- I/O thread count

**Code-level experiments:**
- All M4 shader optimizations from 35B (tg128, encoder coalescing, kernel fusion)
- CAR-aware I/O scheduling (skip pread for substituted experts)
- Adaptive CAR threshold (tighter on early/late layers, looser on middle)
- Expert read coalescing for remaining NVMe reads
- Metal kernel tuning for new dimensions (hidden=3072)

## Execution Order

1. **Wait for 122B download to complete** (~69.6GB)
2. **Phase 1.1-1.2**: Adapt scripts (Codex task, ~30 min)
3. **Phase 1.3-1.4**: Adapt engine (Codex task, ~1-2 hours)
4. **Phase 1.5**: Validate baseline (manual bench, ~10 min)
5. **Phase 2.1-2.3**: CAR integration (Codex task, ~2-3 hours)
6. **Phase 2.4-2.5**: Profiling + warmup (Codex task, ~1 hour)
7. **Phase 3**: Autoresearch loop (autonomous Codex, ongoing)

## Success Criteria

- 122B model loads and generates coherent text
- CAR reduces SSD reads by 40%+ (measurable via cache hit stats)
- Sustained decode: **3+ tok/s** (stretch: 5+ tok/s)
- TTFT: under 10s
- Quality: coherent output, tool calling works
- Zero crashes in stability test

## References

- FOMOE repo: `/Users/tayoun/projects-external/fomoe/`
  - `src/car.c` — CAR algorithm (~240 lines)
  - `src/expert_cache.c` — LRU cache with per-layer partitioning (~220 lines)
  - `src/prefetch.c` — background backfill (~380 lines)
  - `src/freq_profile.c` — frequency profiling
  - `include/car.h` — CAR state struct and API
- Flash-MoE engine: `/Users/tayoun/projects-external/flash-moe/metal_infer/infer.m`
- 122B config: `/tmp/qwen122b/config.json`
- 122B model (downloading): `/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit/`
