#!/usr/bin/env python3
"""Export vocab.bin from tokenizer.json for the C inference engine.

Binary format (matches load_vocab in infer.m):
  uint32 num_entries
  uint32 max_id (unused, set to num_entries)
  repeated num_entries times:
    uint16 byte_len
    char[byte_len] UTF-8 string

Usage:
  python export_vocab.py <tokenizer.json> [output.bin] [tokenizer_config.json]
"""

import json
import struct
import sys
from pathlib import Path


def build_byte_decoder():
    """Inverse of GPT-2 bytes_to_unicode mapping."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def decode_bpe_token(token_str, byte_decoder):
    """Convert a GPT-2-style BPE token string to decoded UTF-8 text."""
    try:
        raw_bytes = bytes([byte_decoder[c] for c in token_str])
        return raw_bytes.decode("utf-8", errors="replace")
    except (KeyError, UnicodeDecodeError):
        return token_str


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
    if not tokenizer_config:
        return False
    klass = str(tokenizer_config.get("tokenizer_class", "")).lower()
    name_or_path = str(tokenizer_config.get("name_or_path", "")).lower()
    return "gemma" in klass or "gemma" in name_or_path


def should_decode_token(token_str, model_type):
    if token_str.startswith("<") and token_str.endswith(">"):
        return False
    return model_type == "BPE"


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_vocab.py <tokenizer.json> [output.bin] [tokenizer_config.json]", file=sys.stderr)
        sys.exit(1)

    tok_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "vocab.bin"
    cfg_path = sys.argv[3] if len(sys.argv) > 3 else None

    with open(tok_path, "r", encoding="utf-8") as f:
        tok = json.load(f)

    tokenizer_config = load_tokenizer_config(tok_path, cfg_path)
    gemma = is_gemma_tokenizer(tokenizer_config)
    model_type = tok.get("model", {}).get("type", "")

    vocab = {}
    if "model" in tok and "vocab" in tok["model"]:
        for token_str, token_id in tok["model"]["vocab"].items():
            vocab[token_id] = token_str

    if "added_tokens" in tok:
        for entry in tok["added_tokens"]:
            vocab[entry["id"]] = entry["content"]

    num_entries = max(vocab.keys()) + 1 if vocab else 0
    print(f"Vocab size: {len(vocab)} tokens, max_id: {num_entries - 1}")

    byte_decoder = build_byte_decoder()

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", num_entries))
        f.write(struct.pack("<I", num_entries))

        for i in range(num_entries):
            s = vocab.get(i, "")
            if s and should_decode_token(s, model_type):
                s = decode_bpe_token(s, byte_decoder)
            b = s.encode("utf-8") if s else b""
            f.write(struct.pack("<H", len(b)))
            if b:
                f.write(b)

    print(f"Wrote {out_path} ({num_entries} entries)")
    if gemma:
        print("Gemma tokenizer metadata detected; preserved turn/control special tokens.")


if __name__ == "__main__":
    main()
