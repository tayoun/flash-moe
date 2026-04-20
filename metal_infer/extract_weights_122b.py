#!/usr/bin/env python3
import argparse
import json
import mmap
import os
import struct
from pathlib import Path

# Extract all non-expert model weights expected by infer.m from Qwen3.5-122B-A10B-4bit
# Output format matches infer.m manifest expectations:
# {
#   "tensors": {
#      name: {offset,size,shape,dtype,source_shard}
#   }
# }
# Binary payload is a flat concat of raw tensor bytes exactly as stored in safetensors.

SKIP_SUBSTRINGS = (
    'switch_mlp.',  # experts already exported separately
    'vision_tower.',
)

DTYPE_MAP = {
    'BF16': 'BF16',
    'F16': 'F16',
    'F32': 'F32',
    'U16': 'U16',
    'U32': 'U32',
    'I32': 'I32',
}


def natural_key(name: str):
    out = []
    cur = ''
    is_digit = None
    for ch in name:
        d = ch.isdigit()
        if is_digit is None:
            cur = ch
            is_digit = d
        elif d == is_digit:
            cur += ch
        else:
            out.append(int(cur) if is_digit else cur)
            cur = ch
            is_digit = d
    if cur:
        out.append(int(cur) if is_digit else cur)
    return out


class Shard:
    def __init__(self, path: Path):
        self.path = path
        self.f = path.open('rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.header_len = struct.unpack('<Q', self.mm[:8])[0]
        self.data_base = 8 + self.header_len
        self.header = json.loads(self.mm[8:self.data_base])

    def meta(self, name):
        return self.header[name]

    def raw_bytes(self, name):
        meta = self.header[name]
        start, end = meta['data_offsets']
        return memoryview(self.mm)[self.data_base + start:self.data_base + end]

    def close(self):
        self.mm.close()
        self.f.close()


class Store:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        idx = json.loads((model_dir / 'model.safetensors.index.json').read_text())
        self.weight_map = idx['weight_map']
        self.shards = {}

    def shard(self, tensor_name):
        shard_name = self.weight_map[tensor_name]
        s = self.shards.get(shard_name)
        if s is None:
            s = Shard(self.model_dir / shard_name)
            self.shards[shard_name] = s
        return s, shard_name

    def iter_selected(self):
        for name in sorted(self.weight_map.keys(), key=natural_key):
            if any(tok in name for tok in SKIP_SUBSTRINGS):
                continue
            yield name

    def close(self):
        for s in self.shards.values():
            s.close()
        self.shards.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/models/flash-moe/Qwen3.5-122B-A10B-4bit')
    ap.add_argument('--out-dir', default='~/projects-external/flash-moe/metal_infer/out_122b')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / 'model_weights.bin'
    manifest_path = out_dir / 'model_weights.json'

    store = Store(model_dir)
    try:
        selected = list(store.iter_selected())
        manifest = {'tensors': {}}
        offset = 0

        if args.dry_run:
            print('selected tensors', len(selected))
            for name in selected[:40]:
                shard, shard_name = store.shard(name)
                meta = shard.meta(name)
                start, end = meta['data_offsets']
                print(name, meta['shape'], meta['dtype'], end - start, shard_name)
            return

        with bin_path.open('wb') as out:
            for idx, name in enumerate(selected, 1):
                shard, shard_name = store.shard(name)
                meta = shard.meta(name)
                raw = shard.raw_bytes(name)
                size = len(raw)
                out.write(raw)
                del raw
                out_name = name
                if out_name.startswith('language_model.'):
                    out_name = out_name[len('language_model.'): ]
                manifest['tensors'][out_name] = {
                    'offset': offset,
                    'size': size,
                    'shape': meta['shape'],
                    'dtype': DTYPE_MAP.get(meta['dtype'], meta['dtype']),
                    'source_shard': shard_name,
                }
                offset += size
                if idx % 100 == 0:
                    print(f'[{idx}/{len(selected)}] {name}')

        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'wrote {len(selected)} tensors')
        print(f'bin: {bin_path} ({bin_path.stat().st_size} bytes)')
        print(f'manifest: {manifest_path}')
    finally:
        store.close()


if __name__ == '__main__':
    main()
