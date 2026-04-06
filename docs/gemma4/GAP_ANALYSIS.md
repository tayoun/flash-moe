# Gemma 4 Gap Analysis vs current `flash-moe`

## Current engine assumptions
The repo is not a generic MoE runtime yet. It is a highly optimized Qwen3.5-specific engine with these baked-in assumptions:

1. **Qwen tensor names** throughout extraction and runtime binding
2. **Qwen layer schedule**: 45 linear-attention + 15 full-attention layers
3. **Qwen dimension constants** baked into runtime and kernels
4. **Qwen tokenizer/chat template** baked into serving path
5. **Qwen expert packing layout** hard-coded in repacker and runtime
6. **Qwen conversion pipeline** for MLX + optional GGUF overlays

## Checkpoint-verified Gemma 4 facts (26B-A4B-it)
- `text_config.num_hidden_layers = 30`
- `text_config.hidden_size = 2816`
- `text_config.num_experts = 128`
- `text_config.top_k_experts = 8`
- Layer schedule from `text_config.layer_types`: 25 `sliding_attention` + 5 `full_attention` (layers `5, 11, 17, 23, 29`)
- Text tensors are under `model.language_model.*`
- Routed expert tensor families are `experts.gate_up_proj` + `experts.down_proj` (not Qwen `switch_mlp.*`)
- Checkpoint includes vision/multimodal tensors (`model.vision_tower.*`, `model.embed_vision.*`), explicitly out of scope for text-only v1

## Exact implementation surface

### Critical blockers
1. **Architecture mismatch**
   - Current runtime centers on Qwen GatedDeltaNet + periodic full attention
   - Gemma 4 may not match that layer structure
   - Impact: `fused_layer_forward`, state allocation, layer cache, manifest config

2. **Hard-coded tensor namespace**
   - Current runtime expects names like:
     - `model.layers.%d.self_attn.*`
     - `model.layers.%d.linear_attn.*`
     - `model.layers.%d.mlp.shared_expert.*`
   - Gemma likely differs
   - Impact: extraction + runtime binding both fail immediately

3. **Hard-coded expert blob layout**
   - `repack_experts.py` is fixed to 9 component blobs and exact byte sizes
   - If Gemma expert tensors differ even slightly, expert streaming breaks

4. **Tokenizer incompatibility risk**
   - Current binary tokenizer path assumes a HF tokenizer.json layout compatible with existing BPE exporter
   - Gemma may require exporter/runtime updates

5. **Prompt template mismatch**
   - Current serve/chat path uses Qwen-style `<|im_start|>` messages
   - Gemma instruction format differs

### Medium-risk gaps
6. **RoPE / norm semantics**
   - current runtime may assume q/k norm or RoPE details specific to Qwen
7. **Shared expert semantics**
   - current engine assumes one shared expert path with dedicated gate
8. **GGUF path mismatch**
   - existing GGUF extraction scripts are Qwen-specific

### Lower-risk gaps
9. **Build system**
   - straightforward
10. **Docs / run scripts**
   - straightforward

## Minimal viable path
To get Gemma 4 running with lowest risk, the minimum implementation surface is:

1. A **Gemma-aware extraction script**
2. A **Gemma-aware tensor binding layer** in `infer.m`
3. A **Gemma expert packer/indexer**
4. A **Gemma prompt/tokenizer path**
5. A runtime mode that uses:
   - Gemma layer config
   - Gemma attention path
   - Gemma MoE routing semantics

## Recommended non-goals for v1
- multimodal support
- GGUF overlay path
- NAX-specific optimization changes
- speculative routing improvements
- new cache experiments

## Recommended validation staircase
1. manifest builds from original Gemma checkpoint
2. tokenizer export/import round-trip works
3. tensor binding finds all required tensors
4. one-layer CPU reference for routed/shared expert matches source framework
5. full forward for one token matches source logits closely
6. end-to-end greedy decode works
7. only then optimize Metal hot paths
