#!/usr/bin/env python3
"""Export HuggingFace tokenizer.json to a compact binary format for C.

Usage:
  python export_tokenizer.py <tokenizer.json> [output.bin]
  python export_tokenizer.py <tokenizer.json> [output.bin] [tokenizer_config.json]

Binary format:
  Header:
    magic: "BPET" (4 bytes)
    version: uint32
    vocab_size: uint32
    num_merges: uint32
    num_added: uint32
  Vocab section (sorted by token_id):
    uint32 token_id, uint16 str_len, char[str_len]
  Merges section (ordered by priority):
    uint16 len_a, char[len_a], uint16 len_b, char[len_b]
  Added tokens section:
    uint32 token_id, uint16 str_len, char[str_len]
  [Gemma 4 only] Gemma special-token footer:
    uint32 version (=2)
    uint32 bos_token_id
    uint32 eos_token_id
    uint32 pad_token_id
    uint32 num_gemma_added (extra Gemma-specific special tokens)
    Then for each Gemma special token:
      uint32 token_id, uint16 str_len, char[str_len]
"""

import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any


def normalize_merges(merges):
    normalized = []
    for pair in merges or []:
        if isinstance(pair, list) and len(pair) == 2:
            normalized.append((pair[0], pair[1]))
            continue
        if isinstance(pair, str):
            parts = pair.split(" ", 1)
            if len(parts) == 2:
                normalized.append((parts[0], parts[1]))
                continue
        raise ValueError(f"Unsupported merge rule format: {pair!r}")
    return normalized


def load_tokenizer_config(tok_path, explicit_cfg_path=None):
    if explicit_cfg_path:
        cfg_path = Path(explicit_cfg_path)
    else:
        cfg_path = Path(tok_path).with_name("tokenizer_config.json")
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_gemma_tokenizer(tokenizer_config):
    """Detect Gemma-family tokenizers.

    Matches if tokenizer_class or name_or_path contains "gemma", or if
    model_type is "gemma3" (which covers Gemma 3 and Gemma 4 tokenizers
    that report that model type in their config).
    """
    if not tokenizer_config:
        return False
    klass = str(tokenizer_config.get("tokenizer_class", "")).lower()
    name_or_path = str(tokenizer_config.get("name_or_path", "")).lower()
    model_type = str(tokenizer_config.get("model_type", "")).lower()
    return "gemma" in klass or "gemma" in name_or_path or model_type == "gemma3"


def is_gemma4_tokenizer(tokenizer_config, added_tokens=None):
    """Gemma 4 tokenizers use <|turn> / <turn|> turn markers.

    Gemma 3 tokenizers use <start_of_turn> / <end_of_turn>.
    We detect Gemma 4 by the presence of <|turn> (open turn, id 105) and
    <turn|> (close turn, id 106) in the added tokens list.
    """
    if tokenizer_config:
        # Check tokenizer_config for Gemma 4-specific tokens
        for key in ["bos_token", "eos_token", "pad_token"]:
            val = str(tokenizer_config.get(key, ""))
            if "<|turn>" in val or "<turn|>" in val:
                return True
        # Also check extra_special_tokens for Gemma 4 markers
        extra = tokenizer_config.get("extra_special_tokens", [])
        for t in extra:
            if isinstance(t, str) and ("<|turn>" in t or "<turn|>" in t):
                return True
    # Check added_tokens list directly (most reliable for Gemma 4)
    # Gemma 4 uses: <|turn> (open, no trailing |) and <turn|> (close)
    if added_tokens:
        added_content = {tok.get("content", "") for tok in added_tokens}
        if "<|turn>" in added_content and "<turn|>" in added_content:
            return True
    return False


def get_gemma_special_tokens(
    tokenizer_config: Optional[Dict],
    added_tokens: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Extract Gemma-specific special token IDs.

    Returns a dict with:
      bos_token_id, eos_token_id, pad_token_id, unk_token_id
      gemma_added: list of (token_id, token_str) for Gemma-specific tokens

    IDs are first looked up from tokenizer_config. When not present there
    (common for Gemma 4), they are resolved from the added_tokens list.
    """
    result = {
        "bos_token_id": None,
        "eos_token_id": None,
        "pad_token_id": None,
        "unk_token_id": None,
        "gemma_added": [],  # [(id, content), ...]
    }
    if not tokenizer_config and not added_tokens:
        return result

    # First try tokenizer_config (works for most tokenizers)
    if tokenizer_config:
        result["bos_token_id"] = tokenizer_config.get("bos_token_id")
        result["eos_token_id"] = tokenizer_config.get("eos_token_id")
        result["pad_token_id"] = tokenizer_config.get("pad_token_id")
        result["unk_token_id"] = tokenizer_config.get("unk_token_id")

    # Build content -> id lookup from added_tokens (Gemma 4 style)
    if added_tokens:
        by_content = {tok.get("content", ""): tok.get("id") for tok in added_tokens}
        # Fill in any missing IDs from the added_tokens lookup
        if result["bos_token_id"] is None:
            result["bos_token_id"] = by_content.get("<bos>")
        if result["eos_token_id"] is None:
            result["eos_token_id"] = by_content.get("<eos>")
        if result["pad_token_id"] is None:
            result["pad_token_id"] = by_content.get("<pad>")
        if result["unk_token_id"] is None:
            result["unk_token_id"] = by_content.get("<unk>")

        # Collect Gemma-specific extra tokens from tokenizer_config.extra_special_tokens
        if tokenizer_config:
            extra = tokenizer_config.get("extra_special_tokens", [])
            for tok in extra:
                if isinstance(tok, str):
                    result["gemma_added"].append(tok)

    return result


def get_gemma_added_from_tokenizer_json(added_tokens: List[Dict]) -> List[tuple]:
    """Extract Gemma-specific added tokens from tokenizer.json added_tokens list.

    Returns list of (token_id, token_content) for Gemma-specific tokens
    like <|turn>, <turn|>, <|channel|>, <channel|>, tool tokens, etc.
    Skips standard tokens (bos, eos, pad, unk, mask) that are already in
    tokenizer_config.
    """
    STANDARD_TOKEN_STRINGS = {
        "<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
        "[multimodal]",  # Gemma 4 multimodal marker
    }
    gemma_specific = []
    for tok in added_tokens:
        content = tok.get("content", "")
        tok_id = tok.get("id")
        if content in STANDARD_TOKEN_STRINGS:
            continue
        if tok_id is not None:
            gemma_specific.append((tok_id, content))
    return gemma_specific


def format_gemma_chat(
    messages: List[Dict[str, str]],
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """Format a list of chat messages into Gemma 4 prompt format.

    Gemma 4 uses <bos> + <|turn>role\\n content <turn|>\\n markers.

    Example output:
        <bos><|turn>system
        You are a helpful assistant.<turn|>
        <|turn>user
        Hello!<turn|>
        <|turn>model
        Hi there!<turn|>

    Args:
        messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
                  messages. The first message may have role="system" for a
                  system prompt. Subsequent system messages are not supported.
        enable_thinking: If True, prepends <|think|> after <bos> in the system
                         block (Gemma 4 thinking mode).
        add_generation_prompt: If True, appends <|turn>model\\n (or the last
                               model turn opener) to prompt the model to respond.

    Returns:
        A formatted string ready for tokenization by the Gemma tokenizer.
    """
    if not messages:
        return ""

    # BOS token — Gemma 4 always starts with <bos>
    parts = ["<bos>"]

    # Separate system prompt (first message with role=system or developer)
    # from the rest of the conversation
    system_content = None
    loop_messages = messages

    first = messages[0]
    if first.get("role") in ("system", "developer"):
        system_content = first.get("content", "")
        loop_messages = messages[1:]

    # System / tools / thinking block
    # Emit a system turn when:
    # - there is an actual system message, OR
    # - enable_thinking is True (to emit <|think|> even without system content)
    has_system_block = (system_content is not None) or enable_thinking

    if has_system_block:
        parts.append("<|turn>system\n")

        if enable_thinking:
            parts.append("<|think|>")

        if system_content:
            parts.append(system_content.strip())

        # Tool definitions would go here if provided
        # (tools are passed separately in the full template; omitted here
        # for simplicity — callers can inject them via system_content)

        parts.append("<turn|>\n")

    # Conversation turns
    prev_message_had_tool_response = False
    for i, message in enumerate(loop_messages):
        role = message.get("role", "user")
        content = message.get("content", "")

        # Map assistant -> model (Gemma uses "model")
        if role == "assistant":
            role = "model"

        # Skip empty content (but still emit the turn marker for model)
        content_str = str(content).strip() if content else ""

        # For model turns, strip any <|channel|>...<channel|> thinking blocks
        # from the content (the template uses strip_thinking for this)
        if role == "model" and "<|channel|>" in content_str:
            # Remove content between <|channel|> and <channel|>
            import re
            content_str = re.sub(r"<\|channel\|>.*?<channel\|>", "", content_str, flags=re.DOTALL)
            content_str = content_str.strip()

        parts.append(f"<|turn>{role}\n")
        parts.append(content_str)
        parts.append("<turn|>\n")
        prev_message_had_tool_response = False

    # Generation prompt
    if add_generation_prompt:
        # Only add model turn opener if the last message wasn't a tool_response
        if loop_messages:
            last_role = loop_messages[-1].get("role", "user")
            if last_role == "assistant":
                last_role = "model"
            # Don't add another prompt if last was already a model turn ending
            # (the loop already emitted the model's <turn|> — add model opener)
            if last_role != "tool_response":
                parts.append("<|turn>model\n")
                if not enable_thinking:
                    parts.append("<|channel|>thought\n<channel|>")
        else:
            # No conversation messages — just system/thinking block
            parts.append("<|turn>model\n")
            if not enable_thinking:
                parts.append("<|channel|>thought\n<channel|>")

    return "".join(parts)


def export_tokenizer_binary(
    tok_json_path: str,
    out_bin_path: str,
    cfg_path: Optional[str] = None,
    include_gemma_footer: bool = True,
) -> Dict[str, Any]:
    """Export tokenizer.json to binary format with optional Gemma 4 footer.

    Args:
        tok_json_path: Path to tokenizer.json
        out_bin_path: Output path for .bin file
        cfg_path: Optional explicit path to tokenizer_config.json
        include_gemma_footer: If True and Gemma 4 is detected, append the
                              Gemma special-token footer (bos/eos/pad IDs +
                              Gemma-specific added tokens).

    Returns:
        Dict with export stats: vocab_size, num_merges, num_added, is_gemma,
        is_gemma4, gemma_special_tokens, file_size.
    """
    with open(tok_json_path, "r", encoding="utf-8") as f:
        t = json.load(f)

    model = t.get("model", {})
    vocab = model.get("vocab", {})
    merges = normalize_merges(model.get("merges", []))
    added = t.get("added_tokens", [])

    if not vocab:
        raise ValueError("tokenizer.json is missing model.vocab")

    tokenizer_config = load_tokenizer_config(tok_json_path, cfg_path)
    gemma = is_gemma_tokenizer(tokenizer_config)
    gemma4 = gemma and is_gemma4_tokenizer(tokenizer_config, added)

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(out_bin_path, "wb") as f:
        f.write(b"BPET")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(sorted_vocab)))
        f.write(struct.pack("<I", len(merges)))
        f.write(struct.pack("<I", len(added)))

        for token_str, token_id in sorted_vocab:
            b = token_str.encode("utf-8")
            f.write(struct.pack("<I", token_id))
            f.write(struct.pack("<H", len(b)))
            f.write(b)

        for a, b in merges:
            ab = a.encode("utf-8")
            bb = b.encode("utf-8")
            f.write(struct.pack("<H", len(ab)))
            f.write(ab)
            f.write(struct.pack("<H", len(bb)))
            f.write(bb)

        for tok in added:
            b = tok["content"].encode("utf-8")
            f.write(struct.pack("<I", tok["id"]))
            f.write(struct.pack("<H", len(b)))
            f.write(b)

        # Gemma 4 footer: version=2, bos/eos/pad IDs, Gemma-specific added tokens
        if include_gemma_footer and gemma4:
            f.write(struct.pack("<I", 2))  # footer version = 2

            spec = get_gemma_special_tokens(tokenizer_config, added)
            f.write(struct.pack("<I", spec["bos_token_id"] or 0))
            f.write(struct.pack("<I", spec["eos_token_id"] or 0))
            f.write(struct.pack("<I", spec["pad_token_id"] or 0))

            gemma_added = get_gemma_added_from_tokenizer_json(added)
            f.write(struct.pack("<I", len(gemma_added)))
            for tok_id, tok_content in gemma_added:
                b = tok_content.encode("utf-8")
                f.write(struct.pack("<I", tok_id))
                f.write(struct.pack("<H", len(b)))
                f.write(b)

    sz = os.path.getsize(out_bin_path)
    stats = {
        "vocab_size": len(sorted_vocab),
        "num_merges": len(merges),
        "num_added": len(added),
        "is_gemma": gemma,
        "is_gemma4": gemma4,
        "file_size": sz,
        "file_size_mb": sz / 1024 / 1024,
    }

    if gemma:
        added_content = {tok.get("content") for tok in added}
        has_turn = "<|turn>" in added_content and "<turn|>" in added_content
        has_start_end_turn = "<start_of_turn>" in added_content and "<end_of_turn>" in added_content
        stats["turn_tokens"] = "<|turn>/<turn|>" if has_turn else (
            "<start_of_turn>/<end_of_turn>" if has_start_end_turn else "none"
        )
        spec = get_gemma_special_tokens(tokenizer_config, added)
        stats["bos_token_id"] = spec["bos_token_id"]
        stats["eos_token_id"] = spec["eos_token_id"]
        stats["pad_token_id"] = spec["pad_token_id"]
        if gemma4:
            gemma_added = get_gemma_added_from_tokenizer_json(added)
            stats["gemma_added_tokens"] = [(tid, content) for tid, content in gemma_added]

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_tokenizer.py <tokenizer.json> [output.bin] [tokenizer_config.json]", file=sys.stderr)
        sys.exit(1)

    tok_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "tokenizer.bin"
    cfg_path = sys.argv[3] if len(sys.argv) > 3 else None

    stats = export_tokenizer_binary(tok_path, out_path, cfg_path)

    print(f"Exported to {out_path}:")
    print(f"  Vocab: {stats['vocab_size']} entries")
    print(f"  Merges: {stats['num_merges']} rules")
    print(f"  Added tokens: {stats['num_added']} entries")
    if stats["is_gemma"]:
        print("  Model family: Gemma tokenizer detected")
        print(f"  Gemma 4: {'yes' if stats['is_gemma4'] else 'no'}")
        print(f"  Turn tokens: {stats.get('turn_tokens', 'unknown')}")
        print(f"  bos_token_id: {stats.get('bos_token_id')}")
        print(f"  eos_token_id: {stats.get('eos_token_id')}")
        print(f"  pad_token_id: {stats.get('pad_token_id')}")
        if stats.get("gemma_added_tokens"):
            print(f"  Gemma-specific added tokens ({len(stats['gemma_added_tokens'])}):")
            for tid, content in stats["gemma_added_tokens"]:
                print(f"    {tid}: {repr(content)}")
    print(f"  File size: {stats['file_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
