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
