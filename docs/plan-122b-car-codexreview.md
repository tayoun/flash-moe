# Codex review — `plan-122b-car.md`

## Verdict
Good direction. Bad decomposition.

The plan correctly identifies 122B as primarily an I/O problem and CAR as a plausible lever. But it overstates how much brand-new engine work is required for 122B bring-up, and it understates how much existing repo evidence argues **against** adding a large application-managed cache on Apple unified memory.

## What looks right
- Phase split is sensible: **bring-up first, optimization second, autoresearch last**.
- The architecture diff is directionally correct: the biggest 122B deltas are hidden size, layer count, MoE width, and linear-attention state.
- CAR is worth testing because decode is likely SSD-bound at K=8.

## Critical comments

### 1) Phase 1 should be framed as **removing 35B hard-codes**, not “adapting the whole engine”
`infer.m` already reads most model dimensions from `config.json` dynamically:
- config load: `metal_infer/infer.m:241-256`
- derived dims / layer maps: `metal_infer/infer.m:280-345`

The main 122B blockers in the engine are not broad architectural rewrites. They are the remaining **35B-specific assertions and defaults**:
- explicit 35B checks: `metal_infer/infer.m:7252-7265`
- default paths still point to `out_35b`: `metal_infer/infer.m:7150-7152`, `metal_infer/infer.m:7293-7301`

Recommendation:
- Rename Phase 1.3 to **“generalize remaining 35B assumptions in infer.m”**.
- Treat shader changes as conditional, only after a successful 122B load/profile shows a real issue.

### 2) Phase 2.1 duplicates infrastructure that already exists
The plan says Flash-MoE currently “trusts the OS” and therefore first needs an explicit cache layer. That is only partially true.

The repo already contains:
- Metal LRU expert cache: `metal_infer/infer.m:3737-3902`
- malloc-backed zero-copy cache: `metal_infer/infer.m:3905-4055`
- background prefetch thread: `metal_infer/infer.m:4059-4185`
- frequency tracking / analysis: `metal_infer/infer.m:6126-6165`
- CLI switches for cache / freq: `metal_infer/infer.m:7156-7164`

So CAR integration should **reuse or deliberately replace** existing hooks. Porting FOMOE’s cache/prefetch/freq code verbatim into a parallel subsystem will create duplicate concepts and more failure modes.

### 3) Biggest strategic risk: the plan conflicts with the repo’s strongest existing result
The plan proposes allocating ~8 GB to an explicit expert cache (`plan-122b-car.md:98-101`). That directly conflicts with prior measurements in this repo:
- custom Metal/malloc caches regressed throughput: `docs/io-and-gpu-exploration.md:92-110`
- best result was **no custom cache, trust OS**: `docs/io-and-gpu-exploration.md:100-107`
- paper confirms OS-only beat Metal LRU: `paper/flash_moe.tex:278-282`, `paper/flash_moe.tex:495-507`, `paper/flash_moe.tex:675-678`

This matters because 122B on 16 GB is *more* memory-constrained than the 35B experiments, not less. A duplicated application cache may make pressure/compression worse.

Recommendation:
1. Establish a **122B OS-page-cache baseline first**.
2. Only add an app-level cache if CAR cannot be implemented without duplicating residency.
3. Better option: make CAR aware of **OS residency** instead of owning an 8 GB duplicate cache. A lightweight residency signal is more aligned with the repo’s “trust the OS” result than reintroducing a large LRU.

### 4) The scripts are not all equally 35B-specific
The plan treats the three scripts as equally in need of 122B rewrites. They are not.

#### Must be generalized
- `metal_infer/extract_weights_35b.py`
  - hard-coded config block: `metal_infer/extract_weights_35b.py:123-143`
  - hard-coded 40-layer validation: `metal_infer/extract_weights_35b.py:155-158`

- `repack_experts_35b.py`
  - hard-coded expert component shapes/sizes: `repack_experts_35b.py:27-45`
  - fixed 40-layer assumptions: `repack_experts_35b.py:42-45`

#### Likely reusable with minor edits
- `build_expert_index_35b.py`
  - already shape/stride driven: `build_expert_index_35b.py:135-160`
  - not obviously 35B-specific except output naming and regex assumptions

Recommendation:
- Don’t spend time cloning all three scripts first.
- First validate whether `build_expert_index_35b.py` already works on 122B tensor names.
- Focus effort on extractor + repacker, where the hard-codes clearly are.

### 5) “Update tensor manifest format” is probably low priority
The plan calls out manifest updates (`plan-122b-car.md:55-59`), but `infer.m` loads the model’s `config.json` directly and uses the manifest mainly for tensor lookup:
- config source: `metal_infer/infer.m:235-359`
- manifest loader: `metal_infer/infer.m:650-721`

So the primary source of truth for dims is **HF config.json**, not the manifest config blob written by `extract_weights_35b.py`.

Recommendation:
- Keep the manifest simple.
- Only change manifest schema if some actual consumer needs it.

### 6) Benchmark protocol is underspecified
The success criteria are fine directionally, but the measurement protocol is missing. That makes autoresearch noisy.

Missing definitions:
- cold vs warm cache behavior
- prompt length for TTFT
- decode length for sustained tok/s
- number of repeated trials
- exact K used at runtime

There is already a parameter mismatch risk:
- plan assumes K=8: `plan-122b-car.md:137`
- help text says default K=4: `metal_infer/infer.m:7155`
- main initializes `K = 6`: `metal_infer/infer.m:7179-7181`
- model config says `num_experts_per_tok = 8`

Recommendation:
- Lock benchmark runs to an explicit CLI `--k`.
- Report at minimum:
  - cold TTFT
  - warm TTFT
  - sustained decode tok/s over 64+ generated tokens
  - substitution rate
  - SSD reads avoided
  - quality regression results

### 7) Quality gate for CAR is too weak
“Coherent output” is not enough for a routing-substitution feature. CAR can preserve fluency while damaging factuality or reasoning.

Recommendation:
- Add an A/B prompt suite before Phase 3.
- Track:
  - substitution rate by layer
  - average score ratio of substitutions
  - output agreement / degradation on a fixed prompt set
- Move “tool calling works” out of the core success criteria. Tool calling is downstream and can fail for chat-template reasons unrelated to CAR.

### 8) Shader work should not be assumed up front
The plan puts shader work directly in Phase 1.4. That may be premature.

From the current code, much of the engine is already parameterized off config-derived dimensions. The first likely failures are more mundane:
- tensor naming mismatches
- expert size / repack layout mismatches
- hard-coded 35B assertions
- default path assumptions

Recommendation:
- Defer shader changes until after:
  1. 122B loads,
  2. 1-token generation works,
  3. profiling shows a real kernel bottleneck or dimension bug.

## Revised execution order

### Phase 0 — Validate assumptions before editing code
1. Confirm 122B tensor naming matches current index-builder regexes.
2. Confirm expert tensor layout and actual per-expert byte size.
3. Confirm 122B download is complete and `config.json` is final.

### Phase 1 — Minimal 122B bring-up
1. Generalize `extract_weights_35b.py`.
2. Generalize `repack_experts_35b.py`.
3. Reuse `build_expert_index_35b.py` if it already matches 122B names; only fork if needed.
4. Remove 35B assertions/defaults from `infer.m`.
5. Bring up 122B **with no CAR and no custom cache** first.
6. Verify:
   - weights load
   - tokenizer path works
   - 1-token generation works
   - short HTTP chat works

### Phase 2 — Baseline measurement
Measure 122B on explicit settings:
- exact `--k`
- cold and warm runs
- fixed prompt length
- fixed decode length
- TTFT + sustained tok/s

Do not start CAR work before this baseline exists.

### Phase 3 — CAR instrumentation before CAR substitution
Implement a **dry-run CAR evaluator** first:
- compute how often a valid substitute would exist
- compute avoided-read potential
- log score ratios by layer
- do **not** change routing yet

This tells you whether CAR has enough headroom on 122B before risking quality.

### Phase 4 — CAR substitution
1. Start with thresholded substitution + metrics.
2. Add dampening.
3. Add background backfill only if substitution shows promise.
4. Add startup seeding only if it beats cold-start behavior.

### Phase 5 — Autoresearch
Only after:
- 122B is stable,
- baseline is recorded,
- CAR has measurable upside,
- quality regression suite exists.

## Concrete edits I would make to the plan
- Replace “implement explicit expert cache” with **“decide whether CAR uses existing cache hooks, OS residency, or a small logical cache layer”**.
- Add an explicit warning that repo history shows **custom caches hurt on Apple unified memory**.
- Split success metrics into:
  - **correctness**: loads, no crash, coherent output
  - **performance**: TTFT, decode tok/s, SSD-read reduction
  - **quality**: A/B prompt suite, substitution-rate diagnostics
- Add a Phase 0 assumption-validation step.
- Move shader tuning behind profiling evidence.

## Bottom line
Keep the goal. Rewrite the plan around this sequence:

**generalize 35B hard-codes → bring up 122B with OS-only baseline → measure → dry-run CAR opportunity → enable CAR carefully → autoresearch**

That path is lower-risk, more consistent with the repo’s actual findings, and much more likely to produce a trustworthy result.