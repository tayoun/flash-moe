# Gemma 4 Porting Plan — file/function map

## Goal
Adapt `Anemll/flash-moe` from the current Qwen3.5-397B-A17B-specific implementation to a Gemma 4 text-only MoE path suitable for Apple Silicon SSD-streamed inference.

## Strategy
- Keep `metal_infer/infer.m` as the runtime base.
- Treat current Qwen support as reference implementation.
- Add Gemma-aware config, tensor mapping, tokenizer export, weight extraction, expert packing, and architecture dispatch.
- Defer multimodal support; target text-only first.

---

## 1. Runtime core: `metal_infer/infer.m`

### A. Model-shape/config generalization
**Current issue:** hard-coded Qwen constants and assumptions are spread throughout the file.

**Areas to change**
- global constants/macros for:
  - hidden dim
  - vocab size
  - num layers
  - num experts
  - experts-per-token
  - head counts
  - RoPE params
  - attention cadence / layer typing
  - linear-attention dimensions
- `load_manifest()`
- `open_weights()`
- `build_layer_cache()`
- `init_layer_scratch()`
- any code using `NUM_LAYERS`, `HIDDEN_DIM`, `NUM_EXPERTS`, `VOCAB_SIZE`, `FULL_ATTN_INTERVAL`

**Required refactor**
- Introduce a runtime `ModelConfig` / `ArchConfig` struct populated from `model_weights.json`
- Replace hard-coded Qwen globals in hot paths with config-backed values where architecture differs
- Keep compile-time fast paths only where dimensions are truly fixed by current kernels

### B. Architecture dispatch
**Current issue:** runtime is specialized to Qwen’s mixed architecture:
- full attention every N layers
- GatedDeltaNet / linear attention elsewhere

**Functions directly affected**
- `fused_layer_forward(...)`
- `full_attention_forward(...)`
- `linear_attention_forward(...)`
- `build_layer_cache(...)`
- KV/linear state allocators and setup

**Porting plan**
- Add per-layer `layer_types` from manifest instead of deriving via `FULL_ATTN_INTERVAL`
- If Gemma 4 is standard attention-only MoE, bypass/remove Qwen linear-attention path for Gemma
- Ensure runtime can allocate only KV caches when linear-attention state is unused

### C. Tensor naming + layer cache binding
**Current issue:** `build_layer_cache()` binds exact Qwen tensor names.

**Functions**
- `build_layer_cache(...)`
- `get_tensor_ptr(...)`
- `get_tensor_info(...)`
- GGUF overlay attach functions if reused

**Porting plan**
- Add tensor-name mapping layer for Gemma
- Separate:
  - routing gate tensor names
  - routed expert tensor names
  - shared expert tensor names
  - attention tensor names
  - norms / embeddings / lm_head
- Create architecture-specific binder helpers:
  - `bind_qwen_layer_cache(...)`
  - `bind_gemma_layer_cache(...)`

### D. MoE forward path
**Current issue:** `moe_forward(...)` and fused path assume current routed expert layout and shared-expert semantics.

**Functions**
- `moe_forward(...)`
- deferred expert pipeline code around:
  - `gpu_encode_experts_batched(...)`
  - `gpu_encode_expert_forward(...)`
  - `finalize_deferred_experts()`
  - `complete_deferred_experts()`
  - `cpu_forward_expert_blob(...)`
  - `expert_layout_for_kind(...)`

**Porting plan**
- Verify Gemma expert topology:
  - routed experts count
  - top-k
  - shared expert count/shape
  - gate/up/down ordering
- If Gemma uses different expert component sizes/layouts, define new `ExpertLayout`
- Keep streaming abstraction, change packed blob format as needed

### E. Attention / positional encoding
**Functions**
- `apply_rotary_emb(...)`
- `full_attention_forward(...)`
- maybe `embed_lookup(...)`
- any q/k norm usage in `build_layer_cache(...)`

**Porting plan**
- Verify Gemma RoPE theta/scaling and q/k norm behavior
- Verify whether Gemma uses pre/post norms compatible with current RMSNorm implementation
- If Gemma attention is conventional, reuse full-attention path and skip Qwen linear-attn code

### F. Tokenization / chat formatting
**Functions**
- `init_tokenizer()`
- `encode_prompt_text_to_tokens(...)`
- `tokenize_user_turn(...)`
- `tokenize_continuation_turn(...)`
- `tokenize_chat_message(...)`
- `load_system_prompt()`
- `load_vocab(...)`
- `decode_token(...)`

**Porting plan**
- Replace Qwen chat template tokens (`<|im_start|>...`) with Gemma template
- Validate tokenizer.json compatibility with exporter format
- Confirm BOS/EOS/special token handling for Gemma

---

## 2. Weight extraction: `metal_infer/extract_weights.py`

**Current issue:** explicitly coded for Qwen3.5 tensor names and config.

**Functions/areas**
- tensor filtering by regex
- `sanitize_name()`
- manifest `config` payload
- layer type generation
- category summary naming

**Porting plan**
- Add Gemma extractor mode:
  - architecture detection from `config.json`
  - Gemma tensor whitelist / skip rules
  - text-only filtering if source checkpoint contains multimodal tensors
- Emit manifest config from model config, not hard-coded literals
- Generate layer types based on actual architecture

---

## 3. Expert packing: `repack_experts.py`

**Current issue:** assumes exact Qwen expert component ordering and byte sizes.

**Functions/areas**
- `COMPONENTS`
- `EXPERT_SIZE`
- `verify_component_sizes()`
- `repack_layer(...)`
- `verify_layer(...)`
- index loading from `expert_index.json`

**Porting plan**
- Build a Gemma-specific expert index generator or generalized index format
- Replace hard-coded component list with architecture-specific spec
- Add packed-format versioning in metadata
- Verify whether shared experts stay resident and only routed experts stream

---

## 4. Tokenizer export scripts

### `metal_infer/export_tokenizer.py`
### `metal_infer/export_vocab.py`

**Porting plan**
- Validate Gemma tokenizer format:
  - BPE vs SentencePiece-like serialization inside tokenizer.json
  - added tokens / special tokens
- If Gemma tokenizer is not compatible with current BPE assumptions, either:
  - extend binary tokenizer runtime, or
  - add Gemma-specific tokenizer exporter + runtime loader

---

## 5. GGUF / hybrid tooling

### Files
- `autoresearch/extract_gguf_embedding.py`
- `autoresearch/extract_gguf_full_attn_overlay.py`
- `autoresearch/extract_gguf_linear_overlay.py`
- `autoresearch/extract_gguf_lm_head.py`
- `autoresearch/extract_gguf_qkv_overlay.py`
- `autoresearch/repack_experts_q3.py`
- `docs/gguf-*`

**Porting plan**
- These are Qwen-specific and optional for first bring-up
- For Gemma phase 1:
  - ignore GGUF overlays
  - focus on one clean source path: original Gemma checkpoint → extracted dense weights + packed experts
- After correctness:
  - add GGUF ingestion only if it materially improves disk footprint or quality

---

## 6. Build/test/docs surface

### `metal_infer/Makefile`
- no major changes needed initially
- later add:
  - `infer-gemma` target or runtime `--arch gemma4`

### docs to add
- `docs/gemma4/PORTING_PLAN.md`
- `docs/gemma4/GAP_ANALYSIS.md`
- `docs/gemma4/IMPLEMENTATION_PLAN.md`

---

## Proposed implementation order
1. Detect Gemma architecture/config and confirm text-only tensor set
2. Generalize manifest/config loading in `extract_weights.py`
3. Refactor `infer.m` to support runtime arch/config selection
4. Implement Gemma layer binding in `build_layer_cache()`
5. Implement Gemma prompt template + tokenizer compatibility
6. Implement Gemma routed expert packing/index generation
7. Bring up CPU correctness for 1 layer / 1 token
8. Bring up full decode path
9. Optimize Metal kernels only after parity
