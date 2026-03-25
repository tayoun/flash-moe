#!/usr/bin/env python3
"""
Verify 122B inference by running token embedding + layer 0 in pure Python/numpy
and comparing against what the C code produces.

Uses the extracted model_weights.bin directly.
"""
import json
import struct
import numpy as np
import sys

MODEL_DIR = "/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit"
WEIGHTS_BIN = "metal_infer/out_122b/model_weights.bin"
MANIFEST = "metal_infer/out_122b/model_weights.json"

# Load manifest
manifest = json.load(open(MANIFEST))
tensors = manifest['tensors']
config = manifest['config']

print("=== Config ===")
for k, v in config.items():
    print(f"  {k}: {v}")

# Load config from model
with open(f"{MODEL_DIR}/config.json") as f:
    full_config = json.load(f)
tc = full_config.get('text_config', {})
print(f"\n=== Text Config (selected) ===")
print(f"  hidden_size: {tc['hidden_size']}")
print(f"  num_hidden_layers: {tc['num_hidden_layers']}")
print(f"  num_experts: {tc['num_experts']}")
print(f"  num_experts_per_tok: {tc['num_experts_per_tok']}")
print(f"  linear_num_key_heads: {tc['linear_num_key_heads']}")
print(f"  linear_num_value_heads: {tc['linear_num_value_heads']}")
print(f"  linear_key_head_dim: {tc['linear_key_head_dim']}")
print(f"  linear_value_head_dim: {tc['linear_value_head_dim']}")

hidden_dim = tc['hidden_size']  # 3072
group_size = full_config['quantization']['group_size']  # 64
bits = full_config['quantization']['bits']  # 4

def load_tensor_raw(name):
    """Load a tensor's raw bytes from model_weights.bin"""
    t = tensors[name]
    with open(WEIGHTS_BIN, 'rb') as f:
        f.seek(t['offset'])
        return f.read(t['size']), t['shape'], t['dtype']

def dequant_bf16(raw_bytes):
    """Convert raw BF16 bytes to float32"""
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32_bytes = np.left_shift(u16.astype(np.uint32), 16)
    return np.frombuffer(f32_bytes.tobytes(), dtype=np.float32)

def dequant_4bit_matvec(w_raw, s_raw, b_raw, x, out_dim, in_dim, group_size):
    """
    4-bit quantized matrix-vector multiply.
    W is [out_dim, packed_cols] as uint32, each uint32 holds 8 nibbles.
    S is [out_dim, num_groups] as BF16
    B is [out_dim, num_groups] as BF16
    x is [in_dim] as float32
    """
    packed_cols = in_dim // 8
    num_groups = in_dim // group_size
    
    W = np.frombuffer(w_raw, dtype=np.uint32).reshape(out_dim, packed_cols)
    S = dequant_bf16(s_raw).reshape(out_dim, num_groups)
    B = dequant_bf16(b_raw).reshape(out_dim, num_groups)
    
    result = np.zeros(out_dim, dtype=np.float32)
    
    for row in range(out_dim):
        acc = 0.0
        for col in range(packed_cols):
            packed = W[row, col]
            base_idx = col * 8
            group_idx = base_idx // group_size
            scale = S[row, group_idx]
            bias = B[row, group_idx]
            
            for n in range(8):
                nibble = (packed >> (n * 4)) & 0xF
                w_val = float(nibble) * scale + bias
                acc += w_val * x[base_idx + n]
        
        result[row] = acc
    
    return result

def rms_norm(x, weight):
    """RMS normalization"""
    rms = np.sqrt(np.mean(x ** 2) + 1e-6)
    return (x / rms) * weight

# ============================================================================
# Step 1: Embed a test token
# ============================================================================
test_token = 814  # "Explain" — first token of the benchmark prompt

print(f"\n=== Step 1: Embed token {test_token} ===")

emb_w_raw, emb_w_shape, _ = load_tensor_raw('model.embed_tokens.weight')
emb_s_raw, emb_s_shape, _ = load_tensor_raw('model.embed_tokens.scales')
emb_b_raw, emb_b_shape, _ = load_tensor_raw('model.embed_tokens.biases')

vocab_size = emb_w_shape[0]  # 248320
packed_cols = emb_w_shape[1]  # 384 = 3072/8
num_groups = emb_s_shape[1]   # 48 = 3072/64

print(f"  Embedding: vocab={vocab_size}, packed_cols={packed_cols}, groups={num_groups}")

# Extract single row for token
W_all = np.frombuffer(emb_w_raw, dtype=np.uint32).reshape(vocab_size, packed_cols)
S_all = dequant_bf16(emb_s_raw).reshape(vocab_size, num_groups)
B_all = dequant_bf16(emb_b_raw).reshape(vocab_size, num_groups)

# Dequantize the embedding row
embedding = np.zeros(hidden_dim, dtype=np.float32)
W_row = W_all[test_token]
S_row = S_all[test_token]
B_row = B_all[test_token]

for col in range(packed_cols):
    packed = W_row[col]
    base_idx = col * 8
    group_idx = base_idx // group_size
    scale = S_row[group_idx]
    bias = B_row[group_idx]
    
    for n in range(8):
        nibble = (packed >> (n * 4)) & 0xF
        embedding[base_idx + n] = float(nibble) * scale + bias

emb_rms = np.sqrt(np.mean(embedding ** 2))
print(f"  Embedding RMS: {emb_rms:.6f}")
print(f"  First 5: {embedding[:5]}")
print(f"  Last 5: {embedding[-5:]}")
print(f"  Min/Max: {embedding.min():.6f} / {embedding.max():.6f}")

# ============================================================================
# Step 2: Layer 0 input layernorm
# ============================================================================
print(f"\n=== Step 2: Layer 0 input_layernorm ===")

ln_w_raw, _, _ = load_tensor_raw('model.layers.0.input_layernorm.weight')
ln_weight = dequant_bf16(ln_w_raw)

normed = rms_norm(embedding, ln_weight)
normed_rms = np.sqrt(np.mean(normed ** 2))
print(f"  Normed RMS: {normed_rms:.6f}")
print(f"  First 5: {normed[:5]}")
print(f"  LN weight first 5: {ln_weight[:5]}")

# ============================================================================
# Step 3: Layer 0 linear attention QKV projection  
# ============================================================================
print(f"\n=== Step 3: Layer 0 linear_attn.in_proj_qkv ===")

qkv_w_raw, qkv_w_shape, _ = load_tensor_raw('model.layers.0.linear_attn.in_proj_qkv.weight')
qkv_s_raw, qkv_s_shape, _ = load_tensor_raw('model.layers.0.linear_attn.in_proj_qkv.scales')
qkv_b_raw, qkv_b_shape, _ = load_tensor_raw('model.layers.0.linear_attn.in_proj_qkv.biases')

qkv_out_dim = qkv_w_shape[0]  # 12288
print(f"  QKV projection: [{qkv_out_dim}, {hidden_dim}] (4-bit)")
print(f"  Expected: Q=2048 + K=2048 + V=8192 = 12288")

# Do the matmul (this will be slow — 12288 × 3072 in Python)
print("  Computing QKV projection (slow in Python)...")
qkv = dequant_4bit_matvec(qkv_w_raw, qkv_s_raw, qkv_b_raw, normed, qkv_out_dim, hidden_dim, group_size)

q = qkv[:2048]
k = qkv[2048:4096]
v = qkv[4096:]

print(f"  Q (2048): RMS={np.sqrt(np.mean(q**2)):.6f}, range=[{q.min():.4f}, {q.max():.4f}]")
print(f"  K (2048): RMS={np.sqrt(np.mean(k**2)):.6f}, range=[{k.min():.4f}, {k.max():.4f}]")
print(f"  V (8192): RMS={np.sqrt(np.mean(v**2)):.6f}, range=[{v.min():.4f}, {v.max():.4f}]")
print(f"  Q first 5: {q[:5]}")
print(f"  K first 5: {k[:5]}")
print(f"  V first 5: {v[:5]}")

# ============================================================================
# Step 4: Check if values look reasonable
# ============================================================================
print(f"\n=== Sanity Checks ===")
print(f"  Embedding all zeros? {np.all(embedding == 0)}")
print(f"  QKV all zeros? {np.all(qkv == 0)}")
print(f"  Q/K/V similar to each other? Q-K corr={np.corrcoef(q[:128], k[:128])[0,1]:.4f}")
print(f"  Any NaN? emb={np.any(np.isnan(embedding))}, qkv={np.any(np.isnan(qkv))}")
print(f"  Any Inf? emb={np.any(np.isinf(embedding))}, qkv={np.any(np.isinf(qkv))}")

print("\nDone. If these values look reasonable, the weights are correct and")
print("the bug is in the attention/delta-net recurrence, not in the projections.")
