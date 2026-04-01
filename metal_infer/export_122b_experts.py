#!/usr/bin/env python3
import argparse
import json
import mmap
import struct
from pathlib import Path

import numpy as np

BASE_KEYS = {
    'gate': 'language_model.model.layers.{layer}.mlp.switch_mlp.gate_proj',
    'up': 'language_model.model.layers.{layer}.mlp.switch_mlp.up_proj',
    'down': 'language_model.model.layers.{layer}.mlp.switch_mlp.down_proj',
}
GROUP_SIZE = 64
DTYPE_INFO = {
    'U32': (np.uint32, 4),
    'BF16': (np.uint16, 2),
}


def bf16_u16_to_fp16_bytes(arr_u16: np.ndarray) -> bytes:
    u32 = arr_u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return f32.astype(np.float16).tobytes(order='C')


class ShardReader:
    def __init__(self, path: Path):
        self.path = path
        self.fd = path.open('rb')
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack('<Q', self.mm[:8])[0]
        self.data_base = 8 + header_len
        self.header = json.loads(self.mm[8:self.data_base])

    def tensor_meta(self, name: str):
        return self.header[name]

    def tensor(self, name: str) -> np.ndarray:
        meta = self.header[name]
        np_dtype, itemsize = DTYPE_INFO[meta['dtype']]
        start, end = meta['data_offsets']
        view = memoryview(self.mm)[self.data_base + start:self.data_base + end]
        arr = np.frombuffer(view, dtype=np_dtype)
        return arr.reshape(meta['shape'])

    def close(self):
        self.mm.close()
        self.fd.close()


class TensorStore:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index = json.loads((model_dir / 'model.safetensors.index.json').read_text())
        self.weight_map = index['weight_map']
        self.shards = {}

    def _shard(self, tensor_name: str) -> ShardReader:
        shard_name = self.weight_map[tensor_name]
        shard = self.shards.get(shard_name)
        if shard is None:
            shard = ShardReader(self.model_dir / shard_name)
            self.shards[shard_name] = shard
        return shard

    def meta(self, tensor_name: str):
        return self._shard(tensor_name).tensor_meta(tensor_name)

    def get(self, tensor_name: str) -> np.ndarray:
        return self._shard(tensor_name).tensor(tensor_name)

    def close(self):
        for shard in self.shards.values():
            shard.close()
        self.shards.clear()


def transpose_packed_u4(weights_2d: np.ndarray) -> np.ndarray:
    logical_out, packed_in = weights_2d.shape
    logical_in = packed_in * 8
    unpacked = np.empty((logical_out, logical_in), dtype=np.uint8)
    for nib in range(8):
        unpacked[:, nib::8] = ((weights_2d >> (nib * 4)) & 0xF).astype(np.uint8)
    transposed = np.ascontiguousarray(unpacked.T)
    packed_cols = transposed.shape[1] // 8
    packed = np.zeros((transposed.shape[0], packed_cols), dtype=np.uint32)
    for nib in range(8):
        packed |= transposed[:, nib::8].astype(np.uint32) << (nib * 4)
    return packed


def build_layout(store: TensorStore, layer: int):
    layout = {'group_size': GROUP_SIZE, 'components': {}, 'layer_count': 48, 'experts_per_layer': 256}
    offset = 0
    for proj in ('gate', 'up', 'down'):
        base = BASE_KEYS[proj].format(layer=layer)
        w_meta = store.meta(base + '.weight')
        s_meta = store.meta(base + '.scales')
        b_meta = store.meta(base + '.biases')
        logical_shape = [w_meta['shape'][1], w_meta['shape'][2] * 8]
        packed_shape = logical_shape
        w_bytes = logical_shape[0] * logical_shape[1] // 2
        s_shape = s_meta['shape'][1:]
        b_shape = b_meta['shape'][1:]
        s_bytes = int(np.prod(s_shape) * 2)
        b_bytes = int(np.prod(b_shape) * 2)
        for suffix, shape, nbytes, dtype in (
            ('W', packed_shape, w_bytes, 'uint4_packed'),
            ('S', s_shape, s_bytes, 'fp16'),
            ('B', b_shape, b_bytes, 'fp16'),
        ):
            key = f'{proj}_{suffix}'
            layout['components'][key] = {'offset': offset, 'shape': list(shape), 'dtype': dtype}
            if suffix == 'W':
                layout['components'][key]['transpose_u4'] = True
                layout['components'][key]['source_shape'] = w_meta['shape'][1:]
            offset += nbytes
    layout['expert_size'] = offset
    return layout


def build_expert_blob(store: TensorStore, layer: int, expert: int) -> bytes:
    chunks = []
    for proj in ('gate', 'up', 'down'):
        base = BASE_KEYS[proj].format(layer=layer)
        w = store.get(base + '.weight')[expert]
        s = store.get(base + '.scales')[expert]
        b = store.get(base + '.biases')[expert]
        packed = transpose_packed_u4(w)
        chunks.append(packed.tobytes(order='C'))
        chunks.append(bf16_u16_to_fp16_bytes(s))
        chunks.append(bf16_u16_to_fp16_bytes(b))
    return b''.join(chunks)


def export_layer(store: TensorStore, out_dir: Path, layer: int, expert_size: int, verify: bool = False):
    layer_path = out_dir / f'layer_{layer:02d}.bin'
    if not verify:
        with layer_path.open('wb') as f:
            for expert in range(256):
                blob = build_expert_blob(store, layer, expert)
                if len(blob) != expert_size:
                    raise ValueError(f'layer {layer} expert {expert}: got {len(blob)} expected {expert_size}')
                f.write(blob)
        return

    data = layer_path.read_bytes()
    for expert in range(2):
        start = expert * expert_size
        end = start + expert_size
        if data[start:end] != build_expert_blob(store, layer, expert):
            raise ValueError(f'verify failed: layer {layer} expert {expert}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/models/flash-moe/Qwen3.5-122B-A10B-4bit')
    ap.add_argument('--out', default='~/projects-external/flash-moe/metal_infer/experts_122b')
    ap.add_argument('--layers', nargs='*', type=int)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = args.layers if args.layers else list(range(48))

    store = TensorStore(model_dir)
    try:
        layout = build_layout(store, layers[0])
        (out_dir / 'layout.json').write_text(json.dumps(layout, indent=2) + '\n')
        if args.dry_run:
            print(json.dumps(layout, indent=2))
            for proj in ('gate', 'up', 'down'):
                base = BASE_KEYS[proj].format(layer=layers[0])
                print(base + '.weight', store.meta(base + '.weight')['shape'])
                print(base + '.scales', store.meta(base + '.scales')['shape'])
                print(base + '.biases', store.meta(base + '.biases')['shape'])
            return
        for layer in layers:
            print(f'exporting layer {layer}')
            export_layer(store, out_dir, layer, layout['expert_size'], verify=False)
            if args.verify:
                export_layer(store, out_dir, layer, layout['expert_size'], verify=True)
                print(f'verified layer {layer}')
    finally:
        store.close()


if __name__ == '__main__':
    main()
