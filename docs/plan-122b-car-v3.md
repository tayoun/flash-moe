# Plan v3: 122B Model Adaptation + CAR Integration (FINAL)

*Incorporates both Codex reviews. Ready for execution.*

## Goal
Run Qwen3.5-122B-A10B at usable speeds (target: 3+ tok/s) on M4 Mac mini (16GB).

---

## Phase 0 — Validate Assumptions
*Before editing any code.*

1. Confirm 122B download is complete: `du -sh /Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit/`
2. Verify `config.json` is present and matches expected architecture (48 layers, 256 experts, K=8, hidden=3072)
3. Test `build_expert_index_35b.py` against 122B tensors — check if regex patterns match. Only fork if they don't.
4. Compute actual per-expert byte size: `moe_intermediate=1024`, `group_size=64`, `bits=4` → expected ~6.3MB per expert. Verify from safetensors.
5. Confirm tokenizer files are compatible (same vocab size 248320, same EOS tokens)

---

## Phase 1 — Minimal 122B Bring-Up
*Goal: 122B loads and generates 1 token. No CAR, no custom cache, no shader changes.*

### 1.1 — Generalize extraction scripts

**`extract_weights_35b.py`:**
- Remove hard-coded config block (lines 123-143) → read from model's config.json
- Remove 40-layer validation (lines 155-158) → derive from config
- Parameterize output dir (support `out_122b` alongside `out_35b`)

**`repack_experts_35b.py`:**
- Remove hard-coded expert shapes/sizes (lines 27-45) → derive from config
- Remove 40-layer assumption (lines 42-45) → derive from config

**`build_expert_index_35b.py`:**
- Test on 122B first. If regex patterns match, reuse as-is with `--model-path` pointing to 122B.

### 1.2 — Remove 35B assumptions from infer.m

Target specific hard-codes (not a rewrite):
- Remove explicit 35B checks: lines 7252-7265
- Generalize default paths from `out_35b`: lines 7150-7152, 7293-7301
- Fix K default mismatch: help text says 4 (line 7155), code says 6 (line 7179) → set default to model's `num_experts_per_tok` from config.json
- Verify config.json-driven dims propagate correctly for hidden=3072, heads=32, linear attn (16 key heads, 64 value heads)
- Do NOT touch Metal shaders — defer until profiling shows need

### 1.3 — Build and verify
- `cd metal_infer && make infer`
- Run with 122B: `./infer --model $MODEL_DIR --k 8 --serve 8100`
- Verify: weights load, tokenizer works, 1-token generation, short HTTP chat completes
- If crash: check buffer sizes for hidden=3072 (attention buffers, delta-net state)

---

## Phase 2 — Baseline Measurement
*Goal: Know exactly how fast 122B runs before any optimization.*

### Benchmark protocol (locked)
```
CLI:           --k 8 (match model's num_experts_per_tok)
Cold TTFT:     first request after server start
Warm TTFT:     second request (page cache partially seeded)
Decode test:   128+ generated tokens, report sustained tok/s
Runs:          3 consecutive for consistency
Prompt:        "Explain why mixture-of-experts models improve compute efficiency."
Max tokens:    256
Record in:     results.tsv
```

Do NOT start CAR work before this baseline exists.

---

## Phase 2.5 — Residency Oracle
*Goal: Give CAR a way to ask "is this expert in the OS page cache?" without building a custom cache.*

### Why this is needed
With "trust the OS" (no custom cache), there is currently no per-expert cached/not-cached signal. The custom cache hooks (lines 5657-5763) only exist when `--cache` is enabled. The default path uses mmap'd layer files with no app-level residency table.

### Implementation: `mincore()` over mmap'd expert byte ranges
- Expert layer files are already mmap'd (lines 7453-7469)
- For a given (layer, expert_id), compute the byte range in the layer file
- Call `mincore(addr, expert_size, vec)` → returns per-page residency bitmap
- Expert is "cached" if all (or most) pages are resident
- Wrap in: `int expert_is_resident(int layer, int expert_id)` — returns 0 or 1
- Cost: ~1 syscall per expert check. At K=8 × 48 layers = 384 checks per token. Negligible vs SSD I/O.

### Why mincore() is the right choice
- Directly queries the real cache (OS page cache) — no duplication
- Zero memory overhead — no app-level LRU structures
- Aligned with repo's "trust the OS" principle
- Approximate but honest — exactly what CAR needs

---

## Phase 3 — Dry-Run CAR (measure opportunity, no routing changes)
*Goal: Know if CAR has enough headroom before risking quality.*

### Implementation
After router selects K=8 experts per layer:
1. For each expert, call `expert_is_resident(layer, expert_id)`
2. For each non-resident expert, scan all 256 experts' router scores to find the highest-scoring resident alternative not already selected
3. Compute score_ratio = alternative_score / original_score
4. Log per-token stats: `uncached_count`, `substitutable_count` (at various thresholds), `avg_score_ratio`
5. Do NOT substitute — just measure and log

### Output
```
[car-dry] token=42 layer=15 uncached=5/8 substitutable=3 @0.35 avg_ratio=0.72
[car-dry] SUMMARY: tokens=128 avg_uncached=4.2/8 avg_substitutable=2.8 @0.35 potential_ssd_reduction=35%
```

### Decision gate
- If potential SSD reduction < 20% at threshold=0.35 → CAR not worth it on 16GB, stop here
- If ≥ 20% → proceed to Phase 4

---

## Phase 4 — CAR Substitution
*Only if Phase 3 shows ≥20% SSD reduction potential.*

### 4.1 — Thresholded substitution
- Integrate into routing path after topK selection
- CLI: `--car-threshold F` (1.0 = disabled, 0.35 = recommended start)
- For each uncached expert: if best resident alternative has score_ratio ≥ threshold, substitute
- Score-dampened mode (`--car-dampen`): scale substitute weight by ratio to reduce hidden state drift
- Renormalize routing weights after substitution (unless dampening)
- Track and log: substitution rate, SSD reads avoided, score ratios

### 4.2 — Quality gate (A/B prompt suite)
Compare outputs: CAR=disabled (threshold=1.0) vs CAR=0.35

**Prompt suite (10 prompts):**
1. "What is the capital of Lebanon?" (factual)
2. "Explain quantum entanglement to a 10-year-old." (explanation)
3. "Write a Python function to find the longest palindromic substring." (code)
4. "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?" (logic)
5. "Summarize the key differences between TCP and UDP." (technical)
6. "Write a haiku about machine learning." (creative)
7. "What were the main causes of World War I?" (history)
8. "Given: f(x) = 3x² + 2x - 1. Find f'(x) and f(3)." (math)
9. "Translate to French: The weather is beautiful today and I would like to go for a walk." (translation)
10. "Compare the economic models of capitalism and socialism." (analysis)

**Settings:** temperature=0, max_tokens=256, --k 8
**Scoring:** Manual comparison — reject if >2/10 prompts show factual errors or reasoning degradation that don't appear in baseline.

### 4.3 — Background backfill (OS page-cache warming)
*Only if substitution shows promise.*

When CAR substitutes an expert:
- Record the substituted expert_id + layer
- During the next idle phase (attention compute, GPU busy, SSD idle):
  - Async `pread()` the substituted expert's byte range from the layer file
  - This warms the OS page cache — kernel retains the pages
  - Next token: `mincore()` returns resident, CAR doesn't need to substitute
- Priority: highest router score first
- Budget: limit to N experts per token to avoid SSD contention with real reads
- This is NOT a persistent app cache — it's just page-cache warming

### 4.4 — Warmup phase
- CLI: `--car-warmup N` (default: 0 = disabled)
- For first N tokens: force threshold=1.0 (no substitutions)
- All experts load from SSD, warming page cache
- After N tokens: CAR activates with a more populated page cache

### 4.5 — Frequency-based pre-warming (optional, only if it beats cold-start)
- Use existing frequency tracking code (lines 6126-6165) to profile expert usage
- Export to file: `python3 profile_experts.py` → `122b.freq`
- At startup: `pread()` the most frequent experts per layer to warm page cache
- CLI: `--warmup-profile 122b.freq`
- Only add if benchmarks show improvement over Phase 4.4 warmup alone

---

## Phase 5 — Autoresearch Loop
*Only after: 122B stable, baseline recorded, CAR has measurable upside, quality suite passes.*

### Branch: `autoresearch/122b`

### Immutable
- K=8 (model default)
- Compiler flags: -O3 -flto -ffast-math -fno-math-errno
- Quality: must pass A/B prompt suite

### Experiment categories
- CAR threshold tuning (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
- CAR dampening vs renormalization
- Backfill batch size and timing
- Warmup token count (0, 32, 64, 128, 256)
- Metal kernel tuning for hidden=3072 dimensions
- tg128 heuristics for new projection sizes
- Encoder coalescing (port from 35B wins)
- Adaptive CAR threshold by layer position (tighter early/late, looser middle)
- Expert read coalescing for remaining NVMe reads
- I/O thread count tuning

### Benchmark per experiment
- Record: tok/s, TTFT, crashes, quality, CAR substitution rate, SSD reads avoided
- Keep/discard criteria same as 35B autoresearch (see program.md)

---

## Success Criteria

**Correctness:**
- Loads without crash
- Generates coherent output on all 10 prompts
- HTTP API works (streaming SSE)

**Performance:**
- Cold TTFT < 15s
- Sustained decode ≥ 3 tok/s (stretch: 5+)
- CAR reduces SSD reads by ≥30%

**Quality:**
- A/B prompt suite: ≤2/10 prompts show degradation vs baseline
- Substitution rate diagnostics stable across runs

---

## References
- FOMOE CAR source: `/Users/tayoun/projects-external/fomoe/src/car.c` (~240 lines)
- FOMOE expert cache: `/Users/tayoun/projects-external/fomoe/src/expert_cache.c` (~220 lines)
- FOMOE backfill: `/Users/tayoun/projects-external/fomoe/src/prefetch.c` (~380 lines)
- Flash-MoE engine: `/Users/tayoun/projects-external/flash-moe/metal_infer/infer.m`
- Existing cache hooks: `infer.m:3737-4185` (custom caches, disabled by default)
- Existing prefetch thread: `infer.m:4059-4185`
- Existing freq tracking: `infer.m:6126-6165`
- Custom cache anti-results: `docs/io-and-gpu-exploration.md:92-110`, `paper/flash_moe.tex:278-282,495-507`
- 122B config: `/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit/config.json`
- 122B model: `/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit/`
