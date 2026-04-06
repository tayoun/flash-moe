# Gemma 4 Port Implementation Plan

**Goal:** Adapt `flash-moe` into a text-only Gemma 4 26B A4B inference path on Apple Silicon using SSD-streamed experts.
**Spec:** `docs/gemma4/PORTING_PLAN.md`, `docs/gemma4/GAP_ANALYSIS.md`
**Tech Stack:** Objective-C/C + Metal, Python safetensors tooling, HuggingFace tokenizer/config assets
**Tasks:** 10 tasks across 5 waves
**Effort:** human: ~1-2 review sessions / agent: ~180-260 min
**Diagrams:** component, data flow, state, sequence

## Architecture

### Component Diagram

```text
┌────────────────────────────┐
│ Original Gemma checkpoint  │
│ config.json                │
│ tokenizer.json             │
│ safetensors shards         │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Python conversion layer    │
│ - extract_weights.py       │
│ - repack_experts.py        │
│ - export_tokenizer.py      │
│ - export_vocab.py          │
└──────────────┬─────────────┘
               │ produces
               ▼
┌────────────────────────────┐
│ Runtime model directory    │
│ model_weights.bin/json     │
│ packed_experts/            │
│ tokenizer.bin              │
│ vocab.bin                  │
└──────────────┬─────────────┘
               │ loaded by
               ▼
┌────────────────────────────┐
│ Metal runtime              │
│ metal_infer/infer.m        │
│ - config loader            │
│ - layer binder             │
│ - attention path           │
│ - MoE routing              │
│ - SSD expert streaming     │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Validation harness         │
│ smoke decode / logits      │
│ one-layer checks           │
│ prompt-template checks     │
└────────────────────────────┘
```

### Data Flow Diagram

```text
INPUT checkpoint
  └─▶ VALIDATION
       - config present?
       - tokenizer present?
       - shard index present?
       - text-only tensors identifiable?
         └─▶ TRANSFORM
              - dense tensor extract
              - routed expert packing
              - tokenizer/vocab export
                └─▶ PERSIST
                     - model_weights.bin/json
                     - packed_experts/
                     - tokenizer.bin
                     - vocab.bin
                       └─▶ OUTPUT
                            - infer loads Gemma config
                            - one-token forward works
                            - greedy decode works
```

### State Diagram

```text
DISCOVERED
  │
  ▼
MAPPED
  │  (tensor names + config understood)
  ▼
EXTRACTABLE
  │  (conversion scripts emit valid runtime artifacts)
  ▼
LOADABLE
  │  (runtime binds tensors without missing names)
  ▼
CORRECT
  │  (one-layer / one-token outputs reasonable)
  ▼
DECODING
  │  (end-to-end prompt works)
  ▼
OPTIMIZED
```

Invalid transitions:
- `DISCOVERED -> DECODING` blocked by missing runtime artifacts
- `EXTRACTABLE -> OPTIMIZED` blocked until correctness exists

### Hero Flow Sequence

```text
User/Developer       Conversion Scripts         Runtime               Model Dir
     │                      │                     │                      │
     │ choose Gemma model   │                     │                      │
     │─────────────────────▶│                     │                      │
     │                      │ extract dense       │                      │
     │                      │ pack experts        │─────────────────────▶│
     │                      │ export tokenizer    │                      │
     │                      │                     │ load config/artifacts│
     │ run infer            │                     │─────────────────────▶│
     │───────────────────────────────────────────▶│                      │
     │                      │                     │ decode prompt        │
     │◀───────────────────────────────────────────│                      │
```

## Waves

## Wave 1 — Discovery + runtime generalization scaffolding

### Task 1: Gemma checkpoint reconnaissance artifact

**Goal:** Document the exact Gemma tensor/config surface needed by the port.

**Files:**
- Create: `docs/specs/gemma4-port/checkpoint-notes.md`
- Modify: `docs/gemma4/GAP_ANALYSIS.md`

**Steps:**
1. Record Gemma text config values from `config.json`.
2. Summarize tensor families from `model.safetensors.index.json`.
3. Identify which vision/multimodal tensors are out of scope for text-only v1.
4. Update gap analysis with concrete Gemma-vs-Qwen deltas.

**Context:** Verified from the downloaded checkpoint: Gemma text config has **30 hidden layers** (`text_config.num_hidden_layers = 30`), **128 experts**, **top-k 8**, **hidden size 2816**, a **25 sliding-attention / 5 full-attention** layer schedule, and text tensors under `model.language_model.*`. If another source reports 42 layers, that appears to refer to a different Gemma variant, not this exact checkpoint.
**Depends on:** none
**Verify:** `test -f docs/specs/gemma4-port/checkpoint-notes.md`
**Failure mode:** tensor names misread from index. **Handled?:** yes, by using raw index file. **Test?:** yes, manual file check.

### Task 2: Introduce manifest-driven runtime config skeleton

**Goal:** Stop treating Qwen constants as the only architecture.

**Files:**
- Modify: `metal_infer/infer.m`
- Modify: `metal_infer/extract_weights.py`

**Steps:**
1. Define a runtime config struct in `infer.m` loaded from manifest JSON.
2. Extend manifest output in `extract_weights.py` to carry architecture + Gemma-friendly fields.
3. Thread config object to code paths that currently derive layer type/count from constants.
4. Preserve Qwen behavior as default path.

**Context:** do not fully de-hardcode kernels yet; create the load/bind seam first.
**Depends on:** Task 1
**Verify:** build passes: `make -C metal_infer infer`
**Failure mode:** config read path breaks Qwen. **Handled?:** yes, preserve default fallback. **Test?:** yes, compile check.

## Wave 2 — Conversion pipeline for Gemma

### Task 3: Gemma-aware dense extractor

**Goal:** Extract text-side non-expert tensors from Gemma safetensors into runtime format.

**Files:**
- Modify: `metal_infer/extract_weights.py`
- Create: `docs/specs/gemma4-port/extractor-map.md`
- Create: `scripts/check_disk_space.sh`

**Steps:**
1. Add model-type detection from `config.json`.
2. Add a disk-space pre-check before extraction to estimate required output size and fail early if the target volume lacks headroom.
3. Filter to `model.language_model.*` tensors and skip vision tensors for v1.
4. Emit Gemma manifest config values from source config instead of hard-coded Qwen values.
5. Record the emitted tensor-name map and skipped tensors.
6. Explicitly document the memory ceiling risk: this source checkpoint is BF16/native and is not expected to fit the 16GB M4 runtime directly; the later 4-bit path is required for practical deployment on the Mac mini.

**Context:** Gemma weights are BF16/native, not MLX 4-bit packed tensors like the current Qwen path. Extraction from BF16 is a bring-up step; deployment on the 16GB Mac mini will require a quantized/runtime-reduced artifact, and that transition should be called out explicitly when we introduce the 4-bit path.
**Depends on:** Task 2
**Verify:** run extractor in dry/small validation mode or header-only mode if implemented; otherwise verify manifest generation path with a limited sample and confirm disk pre-check output.
**Failure mode:** accidentally skip required text tensors or start extraction without enough free space. **Handled?:** partially, via name map doc and disk pre-check. **Test?:** yes, manifest inspection and preflight check.

### Task 4: Gemma expert index + packer design

**Goal:** Adapt `repack_experts.py` from Qwen fixed-layout packing to Gemma routed-expert packing.

**Files:**
- Modify: `repack_experts.py`
- Create: `docs/specs/gemma4-port/expert-layout.md`
- Create: `tools/gemma4/build_expert_index.py`
- Create: `tools/gemma4/validate_expert_metadata.py`

**Steps:**
1. Define Gemma expert tensor layout from raw tensor names:
   - `experts.gate_up_proj`
   - `experts.down_proj`
2. Determine packed blob format for streamed routed experts.
3. Build a Gemma expert index generator from safetensors metadata.
4. Add a **metadata-only validation step before packing** that verifies expert tensor shapes, strides, dtypes, and per-layer consistency against the runtime reader’s expected layout.
5. Update repacker to support an architecture-specific expert layout spec.

**Context:** Gemma appears to fuse gate+up in one tensor family; do not force Qwen’s 9-component layout. Metadata-only validation is required here so shape/layout mismatches are discovered before writing large packed artifacts.
**Depends on:** Task 1
**Verify:** `python3 tools/gemma4/build_expert_index.py --help` and metadata sanity output; `python3 tools/gemma4/validate_expert_metadata.py ...` reports pass before packing
**Failure mode:** packed layout mismatches runtime reader. **Handled?:** partially, via metadata-only validation before pack. **Test?:** yes, metadata validation in this task and runtime validation in Wave 4.

## Wave 3 — Runtime binding + prompt path

### Task 5: Gemma layer-cache binder

**Goal:** Bind Gemma tensor names into runtime layer structures without disturbing Qwen.

**Files:**
- Modify: `metal_infer/infer.m`

**Steps:**
1. Split `build_layer_cache()` into architecture-specific binders.
2. Add Gemma bindings for:
   - norms
   - attention q/k/v/o + q_norm/k_norm
   - router tensors
   - dense MLP tensors if needed
3. Handle Gemma layer-type schedule from config (`sliding_attention` vs `full_attention`).
4. Keep Qwen binder intact.

**Context:** this is the critical runtime seam.
**Depends on:** Task 2, Task 3
**Verify:** runtime logs can enumerate all required tensor pointers without null failures in a validation mode
**Failure mode:** missing tensor names crash at runtime. **Handled?:** yes, add explicit validation mode. **Test?:** yes.

### Task 6: Gemma tokenizer + chat template path

**Goal:** Make prompt encoding/serving produce valid Gemma conversations.

**Files:**
- Modify: `metal_infer/export_tokenizer.py`
- Modify: `metal_infer/export_vocab.py`
- Modify: `metal_infer/infer.m`
- Create: `docs/specs/gemma4-port/prompt-format.md`

**Steps:**
1. Confirm tokenizer.json compatibility with current exporter.
2. Add Gemma special-token handling as needed.
3. Replace Qwen chat prompt construction with architecture-specific formatter.
4. Use `chat_template.jinja` / tokenizer config as source of truth for v1 formatting.

**Context:** current runtime hardcodes Qwen `<|im_start|>` format, which will be wrong for Gemma.
**Depends on:** Task 1
**Verify:** tokenize/decode smoke test with Gemma special tokens
**Failure mode:** valid model, invalid prompt format. **Handled?:** yes. **Test?:** yes.

## Wave 4 — Correctness bring-up

### Task 7: One-layer CPU reference path for Gemma MoE

**Goal:** Validate routed/shared expert math before full decode.

**Files:**
- Modify: `metal_infer/infer.m`
- Create: `tools/gemma4/check_one_layer.py`

**Steps:**
1. Add a debug/validation path for one Gemma layer.
2. Validate router top-k and expert output shape assumptions.
3. Compare runtime intermediate outputs against a **PyTorch** reference implementation for one token.
4. Record acceptable numerical tolerances.
5. Add a lightweight coherence metric beyond “non-empty text” for bring-up quality, such as:
   - repeated-token rate ceiling over a short decode window,
   - EOS-within-range behavior,
   - and/or perplexity/cross-entropy on a tiny fixed prompt continuation slice if feasible.

**Context:** correctness before speed. PyTorch is the reference stack for v1 because it can read the original safetensors/config path directly with fewer translation assumptions than a NumPy-only reference.
**Depends on:** Task 4, Task 5
**Verify:** one-layer validation command returns pass/fail with tolerances and reports the chosen coherence metric
**Failure mode:** MoE math wrong but hidden by full decode noise. **Handled?:** yes. **Test?:** yes.

### Task 8: End-to-end single-token / short-prompt decode

**Goal:** Get Gemma text-only decode working through the main infer path.

**Files:**
- Modify: `metal_infer/infer.m`
- Modify: `metal_infer/Makefile`
- Create: `docs/specs/gemma4-port/bringup-log.md`

**Steps:**
1. Load Gemma-extracted model directory.
2. Run prefill + first decode token.
3. Fix missing paths in norms, attention, lm_head, embeddings, or expert loading.
4. Extend to a short greedy decode smoke run.

**Context:** sliding attention can initially share the existing full-attention implementation if needed, even if not yet optimized.
**Depends on:** Task 5, Task 6, Task 7
**Verify:** short prompt returns text without runtime failure and meets the Wave 4 coherence threshold defined in Task 7 (not just non-empty output)
**Failure mode:** decodes garbage due to prompt/template mismatch or attention bug. **Handled?:** partially. **Test?:** yes.

## Wave 5 — Hardening + handoff

### Task 9: Validation and regression harness

**Goal:** Make future iteration safe.

**Files:**
- Create: `docs/specs/gemma4-port/test-matrix.md`
- Create: `scripts/gemma4_smoke.sh`
- Modify: `metal_infer/Makefile`

**Steps:**
1. Add repeatable smoke commands for extraction, binding, one-layer check, short decode.
2. Create a small regression matrix for prompt, tokenization, tensor binding, and decode.
3. Add make targets where useful.

**Context:** this port will need many iterations; regression hygiene matters.
**Depends on:** Task 8
**Verify:** `bash scripts/gemma4_smoke.sh`
**Failure mode:** future refactor silently breaks bring-up. **Handled?:** yes. **Test?:** yes.

### Task 10: Optimization backlog + next-step brief

**Goal:** Separate correctness completion from performance work.

**Files:**
- Create: `docs/specs/gemma4-port/optimization-backlog.md`
- Modify: `docs/gemma4/PORTING_PLAN.md`

**Steps:**
1. List all deferred optimizations.
2. Mark which can only begin after correctness.
3. Define likely bottlenecks for M4 16GB.
4. Create next-step brief for implementation sessions.

**Context:** avoid mixing porting and optimization prematurely.
**Depends on:** Task 8
**Verify:** docs complete and reviewed
**Failure mode:** team starts premature optimization. **Handled?:** yes. **Test?:** doc review.

## Effort Summary

- Total tasks: 10 across 5 waves
- Parallelizable now:
  - Wave 1 Task 1 and partial Task 2 can overlap
  - Wave 2 Tasks 3 and 4 can overlap after reconnaissance
  - Wave 5 Tasks 9 and 10 can overlap
- Critical path:
  - Task 1 → Task 2 → Task 3 → Task 5 → Task 6 → Task 7 → Task 8

## Failure Modes Registry

| Task | Failure Mode | Handled? | Test? | User Sees |
|------|--------------|----------|-------|-----------|
| 2 | Qwen runtime regresses | Yes | Yes | build/load failure |
| 3 | Missing required Gemma tensors / insufficient disk | Partial | Yes | null tensor bind / preflight failure |
| 4 | Wrong expert layout | Partial | Yes | metadata validation or expert forward mismatch |
| 5 | Binder misses names | Yes | Yes | validation failure |
| 6 | Bad chat template | Yes | Yes | garbage / empty output |
| 7 | Router/expert math wrong | Yes | Yes | one-layer diff failure |
| 8 | Decode unstable | Partial | Yes | wrong text / crash |

## Readiness Gate

READINESS GATE
══════════════
Spec traceability:     PASS
Constitution:          PASS
Pattern consistency:   PASS with intentional deviation (new `tools/gemma4/` helpers)
Ambiguity:             PASS for planning; implementation still depends on validating exact tensor shapes from the checkpoint
Spec completeness:     PASS

Verdict: READY
