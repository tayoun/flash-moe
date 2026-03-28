# Codex review — `plan-122b-car-v2.md`

## Verdict
Much better. This version fixes the major problems in v1:
- bring-up is now correctly framed as removing 35B assumptions
- baseline comes before optimization
- CAR is gated behind a dry-run opportunity check
- the plan no longer reintroduces a big custom cache up front

I’d call this **close to executable**, but there are still **2 real blockers** and **3 cleanup items**.

## What improved
- Phase 0 / 1 decomposition is materially better (`plan-122b-car-v2.md:8-39`)
- Baseline measurement is now explicit and happens before CAR (`plan-122b-car-v2.md:41-52`)
- The plan correctly respects repo history that custom caches hurt on Apple unified memory (`plan-122b-car-v2.md:67`)
- Shader work is now deferred until profiling (`plan-122b-car-v2.md:29-33`)

## Remaining blockers

### 1) Phase 3 has no actual residency oracle yet
The plan says:
- “check existing cache hooks … for residency” (`plan-122b-car-v2.md:57-67`)

That is the main remaining logic gap.

In the current engine, the “existing cache hooks” are the **custom caches**:
- custom cache lookup paths: `metal_infer/infer.m:5657-5763`
- cache objects only exist when enabled: `metal_infer/infer.m:7335-7343`

But the default path is now **OS page cache only**:
- always use warm fd / trust OS page cache: `metal_infer/infer.m:476-482`
- no-cache async pread path: `metal_infer/infer.m:5838-5855`
- packed expert files are opened and mmap’d with no app-level cache residency table: `metal_infer/infer.m:7453-7469`

So with “no custom cache,” there is currently **no per-expert cached/not-cached signal** for CAR to query.

### Recommendation
Insert a new step before Phase 3:

**Phase 2.5 — implement a residency oracle**
Possible options:
1. **`mincore()` over the mmap’d layer file pages** for the expert byte range
   - best fit with the current OS-page-cache strategy
   - approximate but directly aligned with the real cache you care about
2. lightweight latency-based proxy
   - less precise
   - easier to implement, but noisier
3. explicit “seen recently” heuristic
   - cheapest, but weakest

Best option is probably **`mincore()`**, since the layer files are already mmap’d.

Without this, Phase 3 is conceptually right but not implementable as written.

---

### 2) Phase 4.3 backfill is underspecified relative to the current prefetch thread
The plan says:
- reuse existing prefetch thread (`plan-122b-car-v2.md:87-91`)
- “next token gets DRAM hit instead of another substitution” (`plan-122b-car-v2.md:91`)

But the current prefetch thread is built for **immediate-use destination buffers**, not a persistent per-expert residency system:
- prefetch thread + plan: `metal_infer/infer.m:4059-4162`

That means “reuse existing prefetch thread” is not wrong, but it is incomplete. As written, it sounds like backfill will populate a persistent cache. The current implementation does not do that.

### Recommendation
Reword Phase 4.3 to one of these explicit models:
- **OS-page-cache backfill:** async `pread` / page-touch substituted experts so the *kernel* likely retains them for the next token
- **persistent expert buffer backfill:** introduce a small dedicated residency structure for backfilled experts

Given the repo’s prior results, the first framing is the safer one.

## Cleanup items

### 3) Phase 4.5 references functionality that does not exist yet
The plan says:
- use existing frequency tracking (`plan-122b-car-v2.md:97-100`)
- seed cache at startup with `--freq-profile` (`plan-122b-car-v2.md:100`)

Current state:
- `--freq` exists: `metal_infer/infer.m:7160`
- frequency analysis exists and prints diagnostics: `metal_infer/infer.m:6126-6165`
- `--freq-profile` does **not** exist in CLI flags: `metal_infer/infer.m:7150-7207`

Also, with the new “no custom cache” stance, “seed cache at startup” needs a precise meaning:
- pre-touch pages into OS page cache?
- populate a new logical residency table?
- seed prediction hints only?

### Recommendation
Reword 4.5 as:
- “export and persist frequency profile” first
- “optional startup warmup / pre-touch based on profile” second
- only introduce a CLI flag after the mechanism exists

---

### 4) The quality gate is directionally good but still not operationalized
The plan now includes an A/B prompt suite (`plan-122b-car-v2.md:78-85`), which is the right move.

Still missing:
- exact 10 prompts
- decoding settings (temperature, max tokens, think budget)
- pass/fail rubric for “<5% factual degradation” (`plan-122b-car-v2.md:129`)

### Recommendation
Add a small appendix or separate file with:
- prompt list
- exact runtime flags
- scoring rubric
- who/what judges the outputs

Otherwise the acceptance test will drift.

---

### 5) Minor but worth fixing: `--k` help/default mismatch still exists in code
You already lock benchmarking to `--k 8` (`plan-122b-car-v2.md:45`), which is good.

But while touching Phase 1.4, also clean up the current mismatch:
- help text says default `4`: `metal_infer/infer.m:7155`
- code initializes default `K = 6`: `metal_infer/infer.m:7179-7180`

This is small, but it’s exactly the kind of thing that causes benchmark confusion later.

## Suggested revised sequence
1. Phase 0 — validate assumptions
2. Phase 1 — minimal 122B bring-up
3. Phase 2 — baseline measurement
4. **Phase 2.5 — add residency oracle for OS page cache**
5. Phase 3 — dry-run CAR opportunity measurement
6. Phase 4 — real substitution
7. Phase 5 — autoresearch

## Bottom line
v2 is substantially better and mostly on the right track.

I would **green-light it after one more edit pass** that:
1. adds an explicit residency-oracle step,
2. clarifies backfill as OS-page-cache warming vs persistent cache population,
3. removes or defers the nonexistent `--freq-profile` mechanism.
