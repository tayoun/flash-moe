#!/usr/bin/env python3
import json
import struct
import sys
from pathlib import Path

from tokenizers import Tokenizer


def main():
    tok_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "vocab.bin"

    tok_path = str(Path(tok_path).expanduser())
    out_path = str(Path(out_path).expanduser())

    with open(tok_path, "r", encoding="utf-8") as f:
        t = json.load(f)

    vocab = t["model"]["vocab"]          # str -> int
    added = t.get("added_tokens", [])    # list of {id, content, ...}

    tokenizer = Tokenizer.from_file(tok_path)

    id_to_text = {}

    # Decode regular vocab ids using the tokenizer itself so byte-level tokens
    # become proper UTF-8 text instead of raw tokenizer-internal symbols.
    max_vocab_id = max(vocab.values()) if vocab else -1
    for token_id in range(max_vocab_id + 1):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            decoded = ""
        id_to_text[token_id] = decoded

    # Added tokens should decode to their literal content.
    for tok in added:
        id_to_text[tok["id"]] = tok["content"]

    max_id = max(id_to_text.keys())
    num_entries = max_id + 1

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", num_entries))
        f.write(struct.pack("<I", max_id))

        for token_id in range(num_entries):
            text = id_to_text.get(token_id, "")
            b = text.encode("utf-8")
            if len(b) > 65535:
                raise ValueError(f"Token {token_id} too long: {len(b)} bytes")
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    print(f"Exported legacy vocab to {out_path}")
    print(f"  num_entries: {num_entries}")
    print(f"  max_id: {max_id}")


if __name__ == "__main__":
    main()
