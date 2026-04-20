#!/usr/bin/env python3
import json
import struct
import sys
from pathlib import Path

# Dependency-free vocab exporter.
# Writes the legacy vocab.bin format used by infer.m:
#   uint32 num_entries
#   uint32 max_id
#   repeated: uint16 utf8_len + utf8 bytes
# We intentionally preserve tokenizer piece strings from tokenizer.json rather than
# using external decoder libraries, because infer.m performs GPT-style byte decoding.


def main():
    tok_path = str(Path(sys.argv[1] if len(sys.argv) > 1 else 'tokenizer.json').expanduser())
    out_path = str(Path(sys.argv[2] if len(sys.argv) > 2 else 'vocab.bin').expanduser())

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    vocab = t['model']['vocab']
    added = t.get('added_tokens', [])

    id_to_text = {}
    for token_str, token_id in vocab.items():
        id_to_text[int(token_id)] = token_str
    for tok in added:
        id_to_text[int(tok['id'])] = tok['content']

    max_id = max(id_to_text.keys()) if id_to_text else -1
    num_entries = max_id + 1

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', num_entries))
        f.write(struct.pack('<I', max_id))
        for token_id in range(num_entries):
            text = id_to_text.get(token_id, '')
            b = text.encode('utf-8')
            if len(b) > 65535:
                raise ValueError(f'Token {token_id} too long: {len(b)} bytes')
            f.write(struct.pack('<H', len(b)))
            f.write(b)

    print(f'Exported vocab to {out_path}')
    print(f'  num_entries: {num_entries}')
    print(f'  max_id: {max_id}')


if __name__ == '__main__':
    main()
