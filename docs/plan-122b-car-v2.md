# Plan v2: 122B Model Adaptation + CAR Integration

*Revised per Codex review. Key changes: simpler bring-up (remove 35B hard-codes, not rewrite), no custom cache (trust the OS), dry-run CAR before real substitution, proper quality gate.*

## Goal
Run Qwen3.5-122B-A10B at usable speeds (target: 3+ tok/s) on M4 Mac mini (16GB).

## Phase 0 — Validate Assumptions (before editing code)
1. Confirm 122B tensor naming matches `build_expert_index_35b.py` regexes
2. Confirm expert tensor layout and actual per-expert byte size at 4-bit
3. Confirm 122B download is complete and `config.json` is final
4. Check if `build_expert_index_35b.py` already works on 122B without changes

## Phase 1 — Minimal 122B Bring-Up
**Goal: 122B loads and generates 1 token. No CAR, no custom cache.**

1.1 — Generalize `extract_weights_35b.py`
- Remove hard-coded config block (lines 123-143)
- Remove 40-layer validation (lines 155-158)
- Read all dims from model's config.json

1.2 — Generalize `repack_experts_35b.py`
- Remove hard-coded expert shapes/sizes (lines 27-45)
- Remove 40-layer assumption (lines 42-45)
- Derive from config.json

1.3 — Test `build_expert_index_35b.py` on 122B first; only fork if needed

1.4 — Remove 35B assumptions from `infer.m`
- Remove explicit 35B checks (lines 7252-7265)
- Generalize default paths from `out_35b` (lines 7150-7152, 7293-7301)
- Ensure config.json-driven dims propagate correctly for hidden=3072
- Do NOT touch shaders yet — defer until profiling shows need

1.5 — Verify
- Weights load without crash
- Tokenizer path works
- 1-token generation produces output
- Short HTTP chat completes

## Phase 2 — Baseline Measurement
**Goal: Know exactly how fast 122B runs before any optimization.**

Benchmark protocol (locked):
- CLI: explicit `--k 8` (match model's num_experts_per_tok)
- Cold TTFT: first request after server start
- Warm TTFT: second request (cache seeded)
- Sustained decode: 128+ generated tokens, report tok/s
- 3 consecutive runs for consistency
- Record in `results.tsv`

Do NOT start CAR work before this baseline exists.

## Phase 3 — Dry-Run CAR (measure opportunity without changing routing)
**Goal: Know if CAR has enough headroom before risking quality.**

Implement a read-only CAR evaluator:
- After router selects K experts, check existing cache hooks (`infer.m:3737-4185`) for residency
- For each uncached expert: find highest-scoring cached alternative
- Log per-token: how many substitutions WOULD have been made, at what score ratios
- Log per-layer: substitution opportunity distribution
- Compute: theoretical SSD reads avoided at various thresholds (0.2, 0.35, 0.5, 0.7)
- Do NOT actually substitute — just measure

**Key decision point:** If dry-run shows <20% SSD reads avoidable at threshold=0.35, CAR may not be worth the complexity on 16GB. Proceed to Phase 4 only if numbers justify it.

**Cache residency approach:** Use existing cache hooks in `infer.m`, NOT a new 8GB custom cache. Repo history shows custom caches hurt on Apple unified memory (see `docs/io-and-gpu-exploration.md:92-110`, paper sections 278-282, 495-507). CAR should query residency, not own a cache.

## Phase 4 — CAR Substitution (if Phase 3 justifies it)
**Goal: Real substitution, carefully measured.**

4.1 — Thresholded substitution
- Integrate into routing path after topK selection
- `--car-threshold F` (1.0 = disabled, 0.35 = recommended)
- Score-dampened mode: scale substitute weight by ratio (FOMOE's `skip_renorm`)
- Renormalize routing weights after substitution

4.2 — Quality gate (A/B prompt suite)
- Fixed set of 10 prompts: factual Q&A, reasoning, code, tool calling
- Compare CAR=disabled vs CAR=0.35 outputs
- Track per run:
  - substitution rate by layer
  - average score ratio
  - output agreement/degradation
- Reject thresholds that degrade factual accuracy even if fluency is fine

4.3 — Background backfill (only if substitution shows promise)
- Reuse existing prefetch thread (`infer.m:4059-4185`)
- Load substituted experts during attention idle phase
- Priority: highest router score first
- Next token gets DRAM hit instead of another substitution

4.4 — Warmup phase
- `--car-warmup N`: no substitutions for first N tokens
- Seeds cache for subsequent CAR decisions

4.5 — Frequency profiling (optional, only if it beats cold-start)
- Use existing frequency tracking (`infer.m:6126-6165`)
- Generate 122b.freq from representative prompts
- Seed cache at startup with `--freq-profile`

## Phase 5 — Autoresearch Loop
**Only after: 122B is stable, baseline recorded, CAR has measurable upside, quality suite passes.**

Branch: `autoresearch/122b`

Immutable:
- K=8 (model default)
- Compiler flags: -O3 -flto -ffast-math -fno-math-errno

Experiment categories:
- CAR threshold tuning
- CAR dampening vs renormalization
- Cache-aware I/O scheduling
- Metal kernel tuning for hidden=3072
- tg128 heuristics for new dimensions
- Backfill batch size and timing
- Warmup token count
- All 35B shader wins (port and test)

## Success Criteria

**Correctness:** loads, no crash, coherent output on 10-prompt suite
**Performance:** 
- Cold TTFT < 15s
- Sustained decode ≥ 3 tok/s (stretch: 5+)
- CAR reduces SSD reads by 30%+
**Quality:**
- A/B prompt suite shows <5% factual degradation
- Substitution rate diagnostics stable across runs

## References
- FOMOE CAR algorithm: `/Users/tayoun/projects-external/fomoe/src/car.c`
- Existing Flash-MoE caches: `metal_infer/infer.m:3737-4185`
- Existing prefetch: `metal_infer/infer.m:4059-4185`
- Existing freq tracking: `metal_infer/infer.m:6126-6165`
- Custom cache anti-pattern: `docs/io-and-gpu-exploration.md:92-110`
- 122B config: `/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit/config.json`
