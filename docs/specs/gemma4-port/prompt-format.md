# Gemma 4 Prompt Format

## Gemma vs Qwen Chat Format

| Aspect | Gemma 4 | Qwen 3.5 MoE |
|--------|---------|---------------|
| **Turn markers** | `<start_of_turn>`, `<end_of_turn>` | `<\|im_start\|>`, `<\|im_end\|>` |
| **Role tags** | `user`, `model`, `system` (no wrapper) | `user`, `system`, `assistant` (in `<\|im_start\|>role\n`) |
| **BOS token** | `<bos>` at input start | Not required (handled separately) |
| **EOS between turns** | `<end_of_turn>` | `<\|im_end\|>` |
| **System prompt** | Inline in `<start_of_turn>system\n...\n<end_of_turn>` | `<\|im_start\|>system\n...\n<\|im_end\|>` |

## Gemma Conversation Format

### First message (with system prompt)
```
<start_of_turn>system
{system_prompt}
<end_of_turn>
<start_of_turn>user
{user_message}
<end_of_turn>
<start_of_turn>model
```
*(model generates response)*

### Continuation turn (user replies to model)
```
<end_of_turn>
<start_of_turn>user
{user_message}
<end_of_turn>
<start_of_turn>model
```
*(Note: `<end_of_turn>` closes the previous model response)*

### Special Tokens

| Token | Gemma Tokenizer | Tokenizer ID |
|-------|-----------------|--------------|
| `<bos>` | Beginning of sequence | 2 |
| `<eos>` | End of sequence | 1 |
| `<pad>` | Padding | 0 |
| `<start_of_turn>` | New turn start | from vocab |
| `<end_of_turn>` | Turn end | from vocab |
| `user` | User role | from vocab |
| `model` | Model role | from vocab |

### Important Notes

- **Gemma uses tied weights** — `embed_tokens` is also used as `lm_head` (output projection shares weights with the embedding table).
- **BOS is NOT explicitly prepended** in the prompt format above — the model's own embedding lookup handles the sequence start state. The tokenizer encodes `<start_of_turn>` which implicitly carries the BOS signal for Gemma.
- **EOS behavior**: Gemma generates `<eos>` (token ID 1) at end of generation. The serving code should detect this and stop.
- **No `<bos>` in prompt**: Unlike some models, Gemma does not require an explicit `<bos>` token in the prompt string — the model architecture handles this internally through the embedding layer.

## Qwen Conversation Format

### First message (with system prompt)
```
<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
```
*(model generates response)*

### Continuation turn
```
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
```

## Implementation Notes

**`infer.m`** — Architecture-aware prompt construction:

- `tokenize_chat_message(user_content)` — First message, builds system + user prompt
- `tokenize_user_turn(user_content)` — Subsequent turns (no system prompt)
- `tokenize_continuation_turn(user_content)` — Turns with prior conversation state

Both functions switch format based on `cfg_is_gemma()` at runtime.

**`export_tokenizer.py`** — Gemma tokenizer detection via `is_gemma_tokenizer()`:
- Detects Gemma tokenizer class or `name_or_path` containing "gemma"
- Preserves `<start_of_turn>`, `<end_of_turn>` special tokens
- Exports as `tokenizer.bin` (BPET format)
