# Lessons Learned

## 2026-03-22: String-scanning chat bodies broke OpenAI compatibility
**Tags:** [bug] [api] [compatibility]
**What happened:** `POST /v1/chat/completions` only extracted the last `"content"` string by scanning raw JSON text.
**Root cause:** Request handling used ad-hoc string parsing and session continuation assumptions instead of parsing the OpenAI-style message structure.
**Resolution:** Switched to JSON parsing, assembled prompts from ordered `messages[]`, validated supported roles/content formats, and returned `400` for unsupported content shapes.
**Lesson:** For structured API payloads, parse the JSON schema directly; string scanning is fragile and misses valid request variants.
**Files affected:** `metal_infer/infer.m`

## 2026-03-22: Hidden server-side prompt defaults reduced client compatibility
**Tags:** [design] [api]
**What happened:** Server mode depended on `~/.flash-moe/system.md`/implicit system prefill.
**Root cause:** API mode reused local chat assumptions, introducing hidden context not represented in request payloads.
**Resolution:** Server baseline snapshot now starts empty; request-provided `system`/`developer` messages define behavior.
**Lesson:** API/server mode should avoid implicit local state when stateless client compatibility is a goal.
**Files affected:** `metal_infer/infer.m`, `docs/DECISIONS.md`

## 2026-03-24: M4 port required end-to-end pipeline retuning
**Tags:** [perf] [architecture] [hardware]
**What happened:** M3-tuned settings and kernels did not automatically deliver best performance on M4 16GB.
**Root cause:** The bottleneck mix shifted across kernel occupancy, launch overhead, and fusion opportunities.
**Resolution:** Added `tg128` matvec kernels, encoder coalescing, and kernel fusion tuned for M4; moved to runtime-configurable K and selected `K=6` for the production profile.
**Lesson:** Treat each Apple Silicon generation as a new optimization target; preserve tuning knobs (`--k`) in runtime, not code constants.
**Files affected:** `README.md`, `CLAUDE.md`, `docs/optimization-experiments-q4.md`

## 2026-04-01: Prefill dropped deferred MoE completion and corrupted prompt conditioning
**Tags:** [bug] [inference] [moe] [prefill]
**What happened:** Prompt prefill for intermediate tokens called `discard_deferred_experts()` after running all layers, which waited for GPU safety but threw away deferred routed/shared expert completion.
**Root cause:** The prefill path assumed the last-layer hidden state for intermediate prompt tokens could be discarded because the next token embedding overwrites `hidden`. That assumption is invalid for the deferred MoE pipeline because expert completion is part of the token's prompt-conditioned state evolution before generation begins.
**Resolution:** Switched intermediate prefill tokens to `complete_deferred_experts()` so routed/shared expert outputs are finalized during prompt conditioning. Applied the same fix to other prefill branches (system-prompt cache prefill and serve-mode request prefill) to keep prompt processing consistent everywhere. Verified the deferred finalize path now runs across prefill layers.
**Lesson:** Deferred compute in prompt prefill cannot be discarded just because the immediate hidden buffer will be overwritten; if the deferred work contributes to model state/conditioning, prefill must complete it before advancing.
**Files affected:** `metal_infer/infer.m`

## 2026-04-01: Routed experts silently died in the no-cache path because async validity never completed
**Tags:** [bug] [inference] [moe] [io] [async]
**What happened:** With `--cache-entries 0`, routed experts were selected with nonzero weights but never contributed at runtime. Debug logs showed `valid[k]=0`, `done[k]=0`, and `moe_out_rms=0`, while the shared expert path remained active.
**Root cause:** The no-cache / no-prediction / no-LZ4 path relied on `async_pread_start(...)` + `entries[k].done` to mark expert buffers valid. In this configuration the async path left `done=0`, so selected experts were treated as invalid and skipped by the live routed-expert execution/combine path.
**Resolution:** Replaced the affected no-cache branch with direct synchronous `pread` into `buf_multi_expert_data[k]` and set `valid[k]` from the actual read result. Verified routed experts became live again (`v0=1`, `v1=1`, expert outputs nonzero, `moe_out_rms > 0`).
**Lesson:** When debugging fused MoE paths, always log both router weights and final validity/contribution signals (`valid[k]`, `done[k]`, `moe_out_rms`). A model can appear to route correctly while selected experts are silently dropped by I/O bookkeeping.
**Files affected:** `metal_infer/infer.m`

## 2026-04-01: Partial reference probes are the right oracle on memory-constrained hardware
**Tags:** [debugging] [reference] [moe] [tooling]
**What happened:** Full MLX reference comparison was impractical on the 16 GB machine, so a small NumPy-based probe was used instead to run the first token through layer 0 directly from checkpoint weights.
**Resolution:** Added `metal_infer/probe_layer0_qwen122b.py` to inspect embedding, linear attention, router, shared expert, and one routed expert path using the actual 122B checkpoint. The probe showed layer 0 was numerically sane and helped eliminate several false leads before the live-path MoE bugs were isolated.
**Lesson:** For huge models on constrained hardware, build a partial reference probe first. It is often enough to validate formulas, narrow the bad runtime branch, and avoid wasting time chasing unrelated full-model issues.
**Files affected:** `metal_infer/probe_layer0_qwen122b.py`
