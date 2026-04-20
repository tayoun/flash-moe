#!/usr/bin/env python3
import argparse
import json
import mmap
import struct
from pathlib import Path

import numpy as np

GROUP_SIZE = 64

DTYPE_INFO = {
    'U32': (np.uint32, 4),
    'BF16': (np.uint16, 2),
    'F32': (np.float32, 4),
}


def bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rms_norm(x: np.ndarray, w_bf16: np.ndarray, eps: float) -> np.ndarray:
    w = bf16_to_f32(w_bf16)
    inv = 1.0 / np.sqrt(np.mean(x.astype(np.float32) ** 2) + eps)
    return x.astype(np.float32) * inv * w


def rms_norm_bare_per_head(x: np.ndarray, num_heads: int, head_dim: int, eps: float) -> np.ndarray:
    y = x.astype(np.float32).copy().reshape(num_heads, head_dim)
    for h in range(num_heads):
        inv = 1.0 / np.sqrt(np.mean(y[h] ** 2) + eps)
        y[h] *= inv
    return y.reshape(-1)


def rms_norm_gated_per_head(x: np.ndarray, z: np.ndarray, w_bf16: np.ndarray, num_heads: int, head_dim: int, eps: float) -> np.ndarray:
    w = bf16_to_f32(w_bf16).astype(np.float32)
    y = x.astype(np.float32).copy().reshape(num_heads, head_dim)
    zg = silu(z.astype(np.float32).reshape(num_heads, head_dim))
    out = np.empty_like(y)
    for h in range(num_heads):
        inv = 1.0 / np.sqrt(np.mean(y[h] ** 2) + eps)
        out[h] = y[h] * inv * w * zg[h]
    return out.reshape(-1)


def dequant_u4_affine(W_u32: np.ndarray, scales_bf16: np.ndarray, biases_bf16: np.ndarray, group_size: int = GROUP_SIZE) -> np.ndarray:
    out_dim, packed_cols = W_u32.shape
    in_dim = packed_cols * 8
    groups = in_dim // group_size
    scales = bf16_to_f32(scales_bf16).astype(np.float32)
    biases = bf16_to_f32(biases_bf16).astype(np.float32)
    vals = np.empty((out_dim, in_dim), dtype=np.float32)
    for n in range(8):
        vals[:, n::8] = ((W_u32 >> (n * 4)) & 0xF).astype(np.float32)
    vals = vals.reshape(out_dim, groups, group_size)
    deq = vals * scales[:, :, None] + biases[:, :, None]
    return deq.reshape(out_dim, in_dim)


def matvec_u4_affine(W_u32: np.ndarray, scales_bf16: np.ndarray, biases_bf16: np.ndarray, x: np.ndarray, group_size: int = GROUP_SIZE) -> np.ndarray:
    W = dequant_u4_affine(W_u32, scales_bf16, biases_bf16, group_size)
    return (W @ x.astype(np.float32)).astype(np.float32)


def dequant_row_u4_affine(row_u32: np.ndarray, scales_bf16: np.ndarray, biases_bf16: np.ndarray, group_size: int = GROUP_SIZE) -> np.ndarray:
    packed_cols = row_u32.shape[0]
    in_dim = packed_cols * 8
    groups = in_dim // group_size
    scales = bf16_to_f32(scales_bf16).astype(np.float32)
    biases = bf16_to_f32(biases_bf16).astype(np.float32)
    vals = np.empty(in_dim, dtype=np.float32)
    for n in range(8):
        vals[n::8] = ((row_u32 >> (n * 4)) & 0xF).astype(np.float32)
    vals = vals.reshape(groups, group_size)
    return (vals * scales[:, None] + biases[:, None]).reshape(-1)


def conv1d_step_silu(state: np.ndarray, x: np.ndarray, weight_bf16: np.ndarray, kernel_size: int) -> tuple[np.ndarray, np.ndarray]:
    conv_dim = x.shape[0]
    w = bf16_to_f32(weight_bf16).astype(np.float32).reshape(conv_dim, kernel_size)
    window = np.concatenate([state.reshape(kernel_size - 1, conv_dim), x.reshape(1, conv_dim)], axis=0)
    y = np.empty(conv_dim, dtype=np.float32)
    for c in range(conv_dim):
        y[c] = np.dot(window[:, c], w[c])
    y = silu(y)
    new_state = window[1:].reshape(-1).astype(np.float32)
    return y.astype(np.float32), new_state


def apply_rotary(q: np.ndarray, k: np.ndarray, pos: int, num_q_heads: int, num_kv_heads: int, head_dim: int, rotary_dim: int, rope_theta: float, interleaved: bool) -> tuple[np.ndarray, np.ndarray]:
    q = q.copy().reshape(num_q_heads, head_dim)
    k = k.copy().reshape(num_kv_heads, head_dim)
    half = rotary_dim // 2
    for arr in (q, k):
        heads = arr.shape[0]
        for h in range(heads):
            for i in range(half):
                freq = 1.0 / (rope_theta ** ((2 * i) / rotary_dim))
                angle = float(pos) * freq
                c = np.cos(angle)
                s = np.sin(angle)
                idx0 = 2 * i if interleaved else i
                idx1 = 2 * i + 1 if interleaved else i + half
                a0 = arr[h, idx0]
                a1 = arr[h, idx1]
                arr[h, idx0] = a0 * c - a1 * s
                arr[h, idx1] = a0 * s + a1 * c
    return q.reshape(-1), k.reshape(-1)


class ShardReader:
    def __init__(self, path: Path):
        self.path = path
        self.fd = path.open('rb')
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack('<Q', self.mm[:8])[0]
        self.data_base = 8 + header_len
        self.header = json.loads(self.mm[8:self.data_base])

    def tensor(self, name: str) -> np.ndarray:
        meta = self.header[name]
        np_dtype, _ = DTYPE_INFO[meta['dtype']]
        start, end = meta['data_offsets']
        view = memoryview(self.mm)[self.data_base + start:self.data_base + end]
        arr = np.frombuffer(view, dtype=np_dtype)
        return arr.reshape(meta['shape'])


class Store:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        idx = json.loads((model_dir / 'model.safetensors.index.json').read_text())
        self.weight_map = idx['weight_map']
        self.shards = {}

    def get(self, name: str) -> np.ndarray:
        shard_name = self.weight_map[name]
        shard = self.shards.get(shard_name)
        if shard is None:
            shard = ShardReader(self.model_dir / shard_name)
            self.shards[shard_name] = shard
        return shard.tensor(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/models/flash-moe/Qwen3.5-122B-A10B-4bit')
    ap.add_argument('--token-id', type=int, default=3710)
    ap.add_argument('--pos', type=int, default=0)
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser()
    cfg = json.loads((model_dir / 'config.json').read_text())['text_config']
    store = Store(model_dir)

    hidden = dequant_row_u4_affine(
        store.get('language_model.model.embed_tokens.weight')[args.token_id],
        store.get('language_model.model.embed_tokens.scales')[args.token_id],
        store.get('language_model.model.embed_tokens.biases')[args.token_id],
    )
    print('embed hidden_rms', float(np.sqrt(np.mean(hidden**2))))

    # layer 0 linear attention
    eps = float(cfg['rms_norm_eps'])
    normed = rms_norm(hidden, store.get('language_model.model.layers.0.input_layernorm.weight'), eps)
    print('l0 normed_rms', float(np.sqrt(np.mean(normed**2))))

    qkv = matvec_u4_affine(
        store.get('language_model.model.layers.0.linear_attn.in_proj_qkv.weight'),
        store.get('language_model.model.layers.0.linear_attn.in_proj_qkv.scales'),
        store.get('language_model.model.layers.0.linear_attn.in_proj_qkv.biases'),
        normed,
    )
    z = matvec_u4_affine(
        store.get('language_model.model.layers.0.linear_attn.in_proj_z.weight'),
        store.get('language_model.model.layers.0.linear_attn.in_proj_z.scales'),
        store.get('language_model.model.layers.0.linear_attn.in_proj_z.biases'),
        normed,
    )
    beta = matvec_u4_affine(
        store.get('language_model.model.layers.0.linear_attn.in_proj_b.weight'),
        store.get('language_model.model.layers.0.linear_attn.in_proj_b.scales'),
        store.get('language_model.model.layers.0.linear_attn.in_proj_b.biases'),
        normed,
    )
    alpha = matvec_u4_affine(
        store.get('language_model.model.layers.0.linear_attn.in_proj_a.weight'),
        store.get('language_model.model.layers.0.linear_attn.in_proj_a.scales'),
        store.get('language_model.model.layers.0.linear_attn.in_proj_a.biases'),
        normed,
    )
    print('l0 qkv_rms', float(np.sqrt(np.mean(qkv**2))))
    print('l0 z_rms', float(np.sqrt(np.mean(z**2))))
    print('l0 beta_rms', float(np.sqrt(np.mean(beta**2))), 'alpha_rms', float(np.sqrt(np.mean(alpha**2))))

    conv_dim = cfg['linear_key_head_dim'] * cfg['linear_num_key_heads'] * 2 + cfg['linear_value_head_dim'] * cfg['linear_num_value_heads']
    conv_out, _ = conv1d_step_silu(
        np.zeros((cfg['linear_conv_kernel_dim'] - 1) * conv_dim, dtype=np.float32),
        qkv,
        store.get('language_model.model.layers.0.linear_attn.conv1d.weight'),
        cfg['linear_conv_kernel_dim'],
    )
    print('l0 conv_out_rms', float(np.sqrt(np.mean(conv_out**2))))

    key_dim = cfg['linear_key_head_dim']
    num_k_heads = cfg['linear_num_key_heads']
    num_v_heads = cfg['linear_num_value_heads']
    value_dim = cfg['linear_value_head_dim']
    total_key = key_dim * num_k_heads
    total_value = value_dim * num_v_heads
    lin_q = conv_out[:total_key].copy()
    lin_k = conv_out[total_key:2*total_key].copy()
    lin_v = conv_out[2*total_key:2*total_key+total_value].copy()
    inv_scale = 1.0 / np.sqrt(float(key_dim))
    lin_q = rms_norm_bare_per_head(lin_q, num_k_heads, key_dim, 1e-6) * (inv_scale ** 2)
    lin_k = rms_norm_bare_per_head(lin_k, num_k_heads, key_dim, 1e-6) * inv_scale
    print('l0 lin_q_rms', float(np.sqrt(np.mean(lin_q**2))), 'lin_k_rms', float(np.sqrt(np.mean(lin_k**2))), 'lin_v_rms', float(np.sqrt(np.mean(lin_v**2))))

    A_log = store.get('language_model.model.layers.0.linear_attn.A_log').astype(np.float32)
    dt_bias = bf16_to_f32(store.get('language_model.model.layers.0.linear_attn.dt_bias')).astype(np.float32)
    g_decay = np.exp(-np.exp(A_log) * np.log1p(np.exp(alpha.astype(np.float32) + dt_bias)))
    beta_gate = 1.0 / (1.0 + np.exp(-beta.astype(np.float32)))
    print('l0 g0', float(g_decay[0]), 'beta0', float(beta_gate[0]))

    k_heads_per_v = num_v_heads // num_k_heads
    S = np.zeros((num_v_heads, value_dim, key_dim), dtype=np.float32)
    out_values = np.zeros((num_v_heads, value_dim), dtype=np.float32)
    for vh in range(num_v_heads):
        kh = vh // k_heads_per_v
        S[vh] *= g_decay[vh]
        v_h = lin_v[vh*value_dim:(vh+1)*value_dim]
        k_h = lin_k[kh*key_dim:(kh+1)*key_dim]
        q_h = lin_q[kh*key_dim:(kh+1)*key_dim]
        kv_mem = S[vh] @ k_h
        delta = (v_h - kv_mem) * beta_gate[vh]
        S[vh] += delta[:, None] * k_h[None, :]
        out_values[vh] = S[vh] @ q_h
    out_values_flat = out_values.reshape(-1)
    print('l0 out_values_rms', float(np.sqrt(np.mean(out_values_flat**2))))

    gated = rms_norm_gated_per_head(
        out_values_flat,
        z,
        store.get('language_model.model.layers.0.linear_attn.norm.weight'),
        num_v_heads,
        value_dim,
        eps,
    )
    print('l0 gated_rms', float(np.sqrt(np.mean(gated**2))))

    attn_out = matvec_u4_affine(
        store.get('language_model.model.layers.0.linear_attn.out_proj.weight'),
        store.get('language_model.model.layers.0.linear_attn.out_proj.scales'),
        store.get('language_model.model.layers.0.linear_attn.out_proj.biases'),
        gated,
    )
    hidden1 = hidden + attn_out
    print('l0 out_proj_rms', float(np.sqrt(np.mean(attn_out**2))))
    print('l0 hidden_after_attn_rms', float(np.sqrt(np.mean(hidden1**2))))

    # routed/shared MLP summary for layer 0
    mlp_normed = rms_norm(hidden1, store.get('language_model.model.layers.0.post_attention_layernorm.weight'), eps)
    router = matvec_u4_affine(
        store.get('language_model.model.layers.0.mlp.gate.weight'),
        store.get('language_model.model.layers.0.mlp.gate.scales'),
        store.get('language_model.model.layers.0.mlp.gate.biases'),
        mlp_normed,
    )
    top = np.argsort(router)[-8:][::-1]
    print('l0 router_top8', top.tolist())
    print('l0 router_top8_logits', [float(router[i]) for i in top])
    shared_gate = matvec_u4_affine(
        store.get('language_model.model.layers.0.mlp.shared_expert_gate.weight'),
        store.get('language_model.model.layers.0.mlp.shared_expert_gate.scales'),
        store.get('language_model.model.layers.0.mlp.shared_expert_gate.biases'),
        mlp_normed,
    )
    print('l0 shared_gate_sigmoid', float(1.0 / (1.0 + np.exp(-shared_gate[0]))))

    shared_gate_proj = matvec_u4_affine(
        store.get('language_model.model.layers.0.mlp.shared_expert.gate_proj.weight'),
        store.get('language_model.model.layers.0.mlp.shared_expert.gate_proj.scales'),
        store.get('language_model.model.layers.0.mlp.shared_expert.gate_proj.biases'),
        mlp_normed,
    )
    shared_up_proj = matvec_u4_affine(
        store.get('language_model.model.layers.0.mlp.shared_expert.up_proj.weight'),
        store.get('language_model.model.layers.0.mlp.shared_expert.up_proj.scales'),
        store.get('language_model.model.layers.0.mlp.shared_expert.up_proj.biases'),
        mlp_normed,
    )
    shared_act = silu(shared_gate_proj) * shared_up_proj
    shared_down = matvec_u4_affine(
        store.get('language_model.model.layers.0.mlp.shared_expert.down_proj.weight'),
        store.get('language_model.model.layers.0.mlp.shared_expert.down_proj.scales'),
        store.get('language_model.model.layers.0.mlp.shared_expert.down_proj.biases'),
        shared_act,
    )
    print('l0 shared_gate_proj_rms', float(np.sqrt(np.mean(shared_gate_proj**2))))
    print('l0 shared_up_proj_rms', float(np.sqrt(np.mean(shared_up_proj**2))))
    print('l0 shared_act_rms', float(np.sqrt(np.mean(shared_act**2))))
    print('l0 shared_down_rms', float(np.sqrt(np.mean(shared_down**2))))

    # Compare one routed expert: exported sidecar interpretation vs raw safetensor interpretation
    expert_id = int(top[0])
    experts_dir = Path('/Users/tayoun/projects-external/flash-moe/metal_infer/experts_122b')
    ex = load_exported_expert_components(experts_dir, 0, expert_id)

    routed_export_gate = matvec_u4_affine_fp16sb(ex['gate_W'], ex['gate_S'], ex['gate_B'], mlp_normed)
    routed_export_up = matvec_u4_affine_fp16sb(ex['up_W'], ex['up_S'], ex['up_B'], mlp_normed)
    routed_export_act = silu(routed_export_gate) * routed_export_up
    routed_export_down = matvec_u4_affine_fp16sb(ex['down_W'], ex['down_S'], ex['down_B'], routed_export_act)

    gate_w = store.get('language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight')[expert_id]
    gate_s = store.get('language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales')[expert_id]
    gate_b = store.get('language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases')[expert_id]
    up_w = store.get('language_model.model.layers.0.mlp.switch_mlp.up_proj.weight')[expert_id]
    up_s = store.get('language_model.model.layers.0.mlp.switch_mlp.up_proj.scales')[expert_id]
    up_b = store.get('language_model.model.layers.0.mlp.switch_mlp.up_proj.biases')[expert_id]
    down_w = store.get('language_model.model.layers.0.mlp.switch_mlp.down_proj.weight')[expert_id]
    down_s = store.get('language_model.model.layers.0.mlp.switch_mlp.down_proj.scales')[expert_id]
    down_b = store.get('language_model.model.layers.0.mlp.switch_mlp.down_proj.biases')[expert_id]

    routed_raw_gate = matvec_u4_affine(gate_w, gate_s, gate_b, mlp_normed)
    routed_raw_up = matvec_u4_affine(up_w, up_s, up_b, mlp_normed)
    routed_raw_act = silu(routed_raw_gate) * routed_raw_up
    routed_raw_down = matvec_u4_affine(down_w, down_s, down_b, routed_raw_act)

    print('l0 routed_expert_id', expert_id)
    print('l0 routed_export_down_rms', float(np.sqrt(np.mean(routed_export_down**2))))
    print('l0 routed_raw_down_rms', float(np.sqrt(np.mean(routed_raw_down**2))))
    diff = routed_export_down - routed_raw_down
    print('l0 routed_diff_rms', float(np.sqrt(np.mean(diff**2))))
    print('l0 routed_cos', float(np.dot(routed_export_down, routed_raw_down) / ((np.linalg.norm(routed_export_down) * np.linalg.norm(routed_raw_down)) + 1e-12)))



def load_exported_expert_components(experts_dir: Path, layer: int, expert_id: int):
    layout = json.loads((experts_dir / 'layout.json').read_text())
    expert_size = layout['expert_size']
    blob = (experts_dir / f'layer_{layer:02d}.bin').read_bytes()[expert_id * expert_size:(expert_id + 1) * expert_size]
    out = {}
    for name, meta in layout['components'].items():
        off = meta['offset']
        if name.endswith('_W'):
            rows, cols = meta['shape']
            nbytes = rows * cols // 2
            out[name] = np.frombuffer(blob[off:off+nbytes], dtype=np.uint32).reshape(rows, cols // 8)
        else:
            shape = meta['shape']
            nbytes = int(np.prod(shape) * 2)
            out[name] = np.frombuffer(blob[off:off+nbytes], dtype=np.float16).astype(np.float32).reshape(shape)
    return out


def matvec_u4_affine_fp16sb(W_u32: np.ndarray, scales_f16: np.ndarray, biases_f16: np.ndarray, x: np.ndarray, group_size: int = GROUP_SIZE) -> np.ndarray:
    out_dim, packed_cols = W_u32.shape
    in_dim = packed_cols * 8
    groups = in_dim // group_size
    vals = np.empty((out_dim, in_dim), dtype=np.float32)
    for n in range(8):
        vals[:, n::8] = ((W_u32 >> (n * 4)) & 0xF).astype(np.float32)
    vals = vals.reshape(out_dim, groups, group_size)
    deq = vals * scales_f16[:, :, None].astype(np.float32) + biases_f16[:, :, None].astype(np.float32)
    return (deq.reshape(out_dim, in_dim) @ x.astype(np.float32)).astype(np.float32)

if __name__ == '__main__':
    main()
