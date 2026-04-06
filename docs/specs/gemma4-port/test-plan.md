# Test Plan: gemma4-port

Generated on 2026-04-02
Spec: `docs/gemma4/PORTING_PLAN.md`
Plan: `docs/specs/gemma4-port/plan.md`

## Affected Modules
- `metal_infer/infer.m` — runtime config, layer binding, prompt formatting, decode path
- `metal_infer/extract_weights.py` — checkpoint extraction
- `repack_experts.py` — routed expert packing
- `metal_infer/export_tokenizer.py` / `export_vocab.py` — tokenizer export

## Key Interactions to Verify
- Gemma config loads and produces sane runtime values
- Gemma text tensors bind without null lookups
- Gemma tokenizer exports and reloads correctly
- Routed expert blobs are readable by runtime
- Short prompt decode returns coherent output, not merely non-empty output

## Hero Flow Test
1. Point extractor at `/Volumes/Seagate Backup Plus Drive/gemma-4-26B-A4B-it/`
2. Run disk-space preflight and confirm sufficient headroom before extraction
3. Generate runtime model directory artifacts
4. Run metadata-only expert validation before packing
5. Launch `metal_infer/infer` against those artifacts
6. Run short prompt decode
7. Confirm no missing tensor errors, no tokenizer failures, no expert read failures, and coherence threshold satisfied

## Edge Cases
- Missing vision tensors should not fail text-only extraction
- Empty/short prompts should still tokenize correctly
- EOS handling should stop decode cleanly
- Sliding-attention layers should not require Qwen linear-attn state
- BF16/native source artifacts should be treated as bring-up-only on the 16GB Mac mini until a practical quantized path exists

## Failure Scenarios
- Invalid tensor-name map — runtime should report exact missing tensor
- Wrong expert pack layout — metadata validation or one-layer validation should fail loudly
- Bad chat template — smoke prompt should produce malformed/empty output
- Large-model path on external drive — runtime should tolerate slow cold reads
- Insufficient free space on target volume — extraction should fail before heavy writes begin

## Critical Paths (must not break)
- Qwen existing runtime still compiles
- Gemma extraction emits manifest + artifacts
- Disk-space preflight runs before extraction
- Metadata-only expert validation runs before packing
- Gemma runtime binds all required tensors
- One-layer validation path works using the PyTorch reference stack
- End-to-end short decode works and satisfies the defined coherence metric
