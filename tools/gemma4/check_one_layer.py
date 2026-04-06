#!/usr/bin/env python3
"""
check_one_layer.py — Gemma 4 single-layer CPU reference for bring-up validation.

Usage:
    python3 check_one_layer.py --manifest /path/to/gemma4-runtime-local/model_weights.json \\
                                --token-id 1234 \\
                                --layer 0 \\
                                [--dump-activations]

This script:
  1. Loads the Gemma 4 layer weights from the bring-up manifest (safetensors).
  2. Runs one transformer layer forward pass on a single token.
  3. Prints intermediate activations (router scores, top-k, FFN outputs).
  4. Produces numerical tolerances for comparison against the C reference.

Expected tolerances (float32 vs BF16 matvec):
  - Router softmax:          < 1e-3 (relative)
  - GeGLU activation:       < 1e-3 (relative)
  - Expert output:          < 1e-2 (relative, due to BF16 matvec)
  - Combined (dense+MoE):    < 1e-2 (relative)

Coherence metrics to check:
  - Repeated-token rate:     < 5% for random inputs
  - EOS placement:          Should appear at end of coherent sequences
  - Router entropy:          ~log(8) = 2.08 bits for well-conditioned routing
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Gemma 4 constants
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 2816
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 256
GLOBAL_HEAD_DIM = 512
NUM_GLOBAL_KV_HEADS = 2
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 8
MOE_INTERMEDIATE = 704
RMS_NORM_EPS = 1e-6
ROPE_THETA_SLIDING = 10000.0
ROPE_THETA_FULL = 1000000.0
PARTIAL_ROTARY = 0.25

# ---------------------------------------------------------------------------
# BF16 helpers
# ---------------------------------------------------------------------------
def bf16_to_f32(bf16_val):
    """Convert a BF16 16-bit float to f32."""
    if isinstance(bf16_val, int):
        bits = bf16_val & 0xFFFF
    else:
        bits = struct.unpack('<H', struct.pack('<H', bf16_val))[0]
    # Sign bit
    sign = (bits >> 15) & 1
    exponent = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if exponent == 0:
        if mantissa == 0:
            return -0.0 if sign else 0.0
        # subnormal
        f = mantissa / 128.0 * (2 ** (-14))
        return -f if sign else f
    elif exponent == 0xFF:
        return float('nan') if mantissa != 0 else (-float('inf') if sign else float('inf'))
    # Normalized
    f = (1.0 + mantissa / 128.0) * (2 ** (exponent - 127))
    return -f if sign else f


def bf16_vec_to_f32(bf16_buf, n):
    """Convert n BF16 values to f32."""
    result = []
    for i in range(n):
        bf16_bytes = bf16_buf[i * 2:(i + 1) * 2]
        val = struct.unpack('<H', bf16_bytes)[0]
        result.append(bf16_to_f32(val))
    return result


def load_bf16_matrix(file_path, offset, rows, cols):
    """Load a [rows x cols] BF16 matrix from a binary file."""
    nbytes = rows * cols * 2
    with open(file_path, 'rb') as f:
        f.seek(offset)
        data = f.read(nbytes)
    if len(data) < nbytes:
        raise IOError(f"Could not read {nbytes} bytes from {file_path} at offset {offset}")
    return bf16_vec_to_f32(data, rows * cols)


# ---------------------------------------------------------------------------
# Gemma RMSNorm
# ---------------------------------------------------------------------------
def rms_norm(x, weight, eps=RMS_NORM_EPS):
    """Gemma RMSNorm: x * (weight / sqrt(sum(x^2)/N + eps))"""
    n = len(x)
    ss = sum(xi * xi for xi in x) / n
    inv_rms = 1.0 / math.sqrt(ss + eps)
    return [xi * inv_rms * weight[i] for i, xi in enumerate(x)]


# ---------------------------------------------------------------------------
# GeGLU activation
# ---------------------------------------------------------------------------
def gelu(x):
    """Gemma uses approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    return 0.5 * x * (1.0 + math.tanh(inner))


def geglu(gate, up):
    """GeGLU: gelu(gate) * up"""
    return [gelu(g) * u for g, u in zip(gate, up)]


# ---------------------------------------------------------------------------
# Gemma RoPE (partial rotary)
# ---------------------------------------------------------------------------
def apply_rope(x, cos, sin):
    """Apply rotary embedding to x using precomputed cos/sin."""
    n = len(x)
    result = [0.0] * n
    for i in range(0, n, 2):
        result[i] = x[i] * cos[i] - x[i + 1] * sin[i]
        result[i + 1] = x[i] * sin[i] + x[i + 1] * cos[i]
    return result


def compute_rope_freq(theta, dim, seq_len=1):
    """Compute RoPE frequencies for a single position."""
    freqs = [1.0 / (theta ** (2 * i / dim)) for i in range(dim // 2)]
    cos_vals = [math.cos(freq) for freq in freqs]
    sin_vals = [math.sin(freq) for freq in freqs]
    return cos_vals, sin_vals


# ---------------------------------------------------------------------------
# MatVec (BF16 matrix @ f32 vector)
# ---------------------------------------------------------------------------
def matvec_bf16(weight_bf16, x, rows, cols):
    """BF16 matrix-vector multiply: out[M] = W[M,N] @ x[N]"""
    result = [0.0] * rows
    for i in range(rows):
        row = weight_bf16[i * cols:(i + 1) * cols]
        s = 0.0
        for j in range(cols):
            s += row[j] * x[j]
        result[i] = s
    return result


# ---------------------------------------------------------------------------
# Main layer forward
# ---------------------------------------------------------------------------
def gemma_layer_forward(args, hidden, layer_idx, token_pos):
    """Run one Gemma layer forward on a single token."""

    manifest = args.manifest
    tensors = manifest['tensors']

    hidden_size = HIDDEN_SIZE
    is_full = ((layer_idx + 1) % 6) == 0  # full attention every 6 layers

    # Layer type determines attention dimensions
    if is_full:
        head_dim = GLOBAL_HEAD_DIM  # 512
        num_kv_heads = NUM_GLOBAL_KV_HEADS  # 2
        rope_theta = ROPE_THETA_FULL
    else:
        head_dim = HEAD_DIM  # 256
        num_kv_heads = NUM_KV_HEADS  # 8
        rope_theta = ROPE_THETA_SLIDING

    num_heads = NUM_ATTN_HEADS  # 16
    rotary_dim = int(head_dim * PARTIAL_ROTARY)  # 64
    eps = RMS_NORM_EPS

    # ---- 1. Input RMSNorm ----
    norm_w = _load_tensor(tensors, f'layers.{layer_idx}.input_layernorm.weight',
                         [hidden_size], 'bf16')
    normed = rms_norm(hidden, norm_w, eps)

    # ---- 2. Q/K/V projections ----
    q_w = _load_tensor(tensors, f'layers.{layer_idx}.self_attn.q_proj.weight',
                       [num_heads * head_dim, hidden_size], 'bf16')
    k_w = _load_tensor(tensors, f'layers.{layer_idx}.self_attn.k_proj.weight',
                       [num_kv_heads * head_dim, hidden_size], 'bf16')
    v_w = _load_tensor(tensors, f'layers.{layer_idx}.self_attn.v_proj.weight',
                       [num_kv_heads * head_dim, hidden_size], 'bf16')

    q_out = matvec_bf16(q_w, normed, num_heads * head_dim, hidden_size)
    k_out = matvec_bf16(k_w, normed, num_kv_heads * head_dim, hidden_size)
    v_out = matvec_bf16(v_w, normed, num_kv_heads * head_dim, hidden_size)

    # ---- 3. Per-head Q/K norms ----
    q_norm_w = _load_tensor(tensors, f'layers.{layer_idx}.self_attn.q_norm.weight',
                            [num_heads * head_dim], 'bf16')
    k_norm_w = _load_tensor(tensors, f'layers.{layer_idx}.self_attn.k_norm.weight',
                            [num_kv_heads * head_dim], 'bf16')

    q_heads = [q_out[i * head_dim:(i + 1) * head_dim] for i in range(num_heads)]
    k_heads = [k_out[i * head_dim:(i + 1) * head_dim] for i in range(num_kv_heads)]

    for h in range(num_heads):
        q_heads[h] = rms_norm(q_heads[h], q_norm_w[h * head_dim:(h + 1) * head_dim], eps)
    for h in range(num_kv_heads):
        k_heads[h] = rms_norm(k_heads[h], k_norm_w[h * head_dim:(h + 1) * head_dim], eps)

    # ---- 4. RoPE ----
    cos, sin = compute_rope_freq(rope_theta, rotary_dim, token_pos)
    for h in range(num_heads):
        q_heads[h] = apply_rope(q_heads[h], cos, sin)
    for h in range(num_kv_heads):
        k_heads[h] = apply_rope(k_heads[h], cos, sin)

    # (KV cache store/retrieve would go here in full decode)
    # For single-token, skip cache.

    # ---- 5. Attention (simplified: full attention on single token) ----
    q_flat = [xi for h in q_heads for xi in h]
    kv_groups = num_heads // num_kv_heads
    attn_out = [0.0] * (num_heads * head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    for h in range(num_heads):
        kv_h = h // kv_groups
        k_h = k_heads[kv_h]
        v_h = v_out[kv_h * head_dim:(kv_h + 1) * head_dim]
        q_h = q_heads[h]
        # Full attention for single token: just compute q @ k for this token
        # (No cache in single-token mode)
        scores = [sum(q * kk for q, kk in zip(q_h, k_h)) * scale]
        # Softmax
        max_s = scores[0]
        exp_s = [math.exp(s - max_s) for s in scores]
        attn_h = [s * v for s, v in zip(exp_s, v_h)]
        attn_out[h * head_dim:(h + 1) * head_dim] = attn_h

    # ---- 6. O projection ----
    o_w = _load_tensor(tensors, f'layers.{layer_idx}.self_attn.o_proj.weight',
                       [hidden_size, num_heads * head_dim], 'bf16')
    attn_out_2d = [attn_out[i * head_dim:(i + 1) * head_dim]
                   for i in range(num_heads)]
    o_out_flat = [0.0] * hidden_size
    for i in range(hidden_size):
        s = 0.0
        for h in range(num_heads):
            s += sum(o_w[i * num_heads * head_dim + h * head_dim + d] * attn_out_2d[h][d]
                     for d in range(head_dim))
        o_out_flat[i] = s

    # ---- 7. Residual add ----
    hidden = [hidden[i] + o_out_flat[i] for i in range(hidden_size)]

    # ---- 8. Post-attention norm ----
    post_attn_norm_w = _load_tensor(tensors,
                                     f'layers.{layer_idx}.post_attention_layernorm.weight',
                                     [hidden_size], 'bf16')
    hidden = rms_norm(hidden, post_attn_norm_w, eps)

    # ---- 9. Pre-FFN norm ----
    ffn_input = rms_norm(hidden,
                         _load_tensor(tensors,
                                       f'layers.{layer_idx}.pre_feedforward_layernorm.weight',
                                       [hidden_size], 'bf16'),
                         eps)

    # ---- 10. Dense FFN (GeGLU) ----
    gate_w = _load_tensor(tensors, f'layers.{layer_idx}.mlp.gate_proj.weight',
                           [MOE_INTERMEDIATE, hidden_size], 'bf16')
    up_w = _load_tensor(tensors, f'layers.{layer_idx}.mlp.up_proj.weight',
                         [MOE_INTERMEDIATE, hidden_size], 'bf16')
    down_w = _load_tensor(tensors, f'layers.{layer_idx}.mlp.down_proj.weight',
                           [hidden_size, MOE_INTERMEDIATE], 'bf16')

    gate_out = matvec_bf16(gate_w, ffn_input, MOE_INTERMEDIATE, hidden_size)
    up_out = matvec_bf16(up_w, ffn_input, MOE_INTERMEDIATE, hidden_size)
    geglu_out = geglu(gate_out, up_out)
    dense_out = matvec_bf16(down_w, geglu_out, hidden_size, MOE_INTERMEDIATE)

    # ---- 11. MoE routing ----
    router_w = _load_tensor(tensors, f'layers.{layer_idx}.router.proj.weight',
                            [NUM_EXPERTS, hidden_size], 'bf16')
    router_scores = matvec_bf16(router_w, ffn_input, NUM_EXPERTS, hidden_size)

    # Router RMSNorm
    router_scale_w = _load_tensor(tensors, f'layers.{layer_idx}.router.scale',
                                  [NUM_EXPERTS], 'bf16')
    router_scores = rms_norm(router_scores, router_scale_w, eps)

    # Softmax
    max_s = max(router_scores)
    exp_s = [math.exp(s - max_s) for s in router_scores]
    sum_exp = sum(exp_s)
    probs = [s / sum_exp for s in exp_s]

    # Top-8 selection
    top_k_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:NUM_EXPERTS_PER_TOK]
    top_k_w = [probs[i] for i in top_k_idx]

    print(f"[check] Layer {layer_idx}: is_full={is_full}")
    print(f"[check] Router top-8 experts: {top_k_idx}")
    print(f"[check] Router top-8 weights: {top_k_w}")
    print(f"[check] Router entropy: {-sum(p * math.log(p + 1e-10) for p in probs if p > 0):.3f} bits "
          f"(max possible: {math.log(NUM_EXPERTS):.3f})")

    # ---- 12. MoE expert dispatch ----
    # Experts are stored in packed binaries. This script loads from safetensors.
    # If safetensors are not available, we can still validate the routing algorithm.
    moe_out = [0.0] * hidden_size
    expert_layer_prefix = f'layers.{layer_idx}.mlp.experts'

    for k in range(NUM_EXPERTS_PER_TOK):
        eidx = top_k_idx[k]
        w = top_k_w[k]
        print(f"[check] Expert {eidx}: weight={w:.4f}")

        # Try to load expert from safetensors
        try:
            gate_up = _load_tensor(tensors,
                                    f'{expert_layer_prefix}.gate_up_proj',
                                    [2, MOE_INTERMEDIATE, hidden_size], 'bf16',
                                    expert_idx=eidx)
            down = _load_tensor(tensors,
                               f'{expert_layer_prefix}.down_proj',
                               [hidden_size, MOE_INTERMEDIATE], 'bf16',
                               expert_idx=eidx)

            # Split gate_up into gate and up
            gate = gate_up[:MOE_INTERMEDIATE * hidden_size]
            up = gate_up[MOE_INTERMEDIATE * hidden_size:]

            # GeGLU
            gate_out_e = matvec_bf16(gate, ffn_input, MOE_INTERMEDIATE, hidden_size)
            up_out_e = matvec_bf16(up, ffn_input, MOE_INTERMEDIATE, hidden_size)
            geglu_out_e = geglu(gate_out_e, up_out_e)

            # Down proj
            expert_out = matvec_bf16(down, geglu_out_e, hidden_size, MOE_INTERMEDIATE)

            for i in range(hidden_size):
                moe_out[i] += w * expert_out[i]
        except KeyError:
            # Expert not in bring-up manifest (might need separate expert pack)
            print(f"[check] Expert {eidx}: not available in bring-up manifest, skipping")

    # ---- 13. Combine dense + MoE, scale ----
    layer_scalar = _load_tensor(tensors, f'layers.{layer_idx}.layer_scalar',
                                [1], 'bf16')
    layer_scalar = bf16_to_f32(layer_scalar[0]) if layer_scalar else 1.0
    combine_scale = 1.0 / math.sqrt(2.0) * layer_scalar

    output = [hidden[i] + combine_scale * (dense_out[i] + moe_out[i])
              for i in range(hidden_size)]

    # ---- Post-FFN norms ----
    post_ffn_norm_w = _load_safe_tensor(tensors,
                                        f'layers.{layer_idx}.post_feedforward_layernorm.weight',
                                        [hidden_size], 'bf16')
    if post_ffn_norm_w:
        output = rms_norm(output, post_ffn_norm_w, eps)

    return output, {
        'router_scores': router_scores,
        'top_k_idx': top_k_idx,
        'top_k_w': top_k_w,
        'hidden_size': hidden_size,
        'is_full': is_full,
    }


# ---------------------------------------------------------------------------
# Tensor loading from bring-up manifest
# ---------------------------------------------------------------------------
def _get_safetensors_path(tensors, tensor_name):
    """Find the safetensors file that contains the given tensor."""
    info = tensors.get(tensor_name)
    if info is None:
        raise KeyError(f"Tensor '{tensor_name}' not found in manifest")
    filename = info.get('filename') or info.get('file') or info.get('safetensors_file')
    if not filename:
        # Try to find in model_weights.json directory
        return None
    return Path(filename)


def _load_tensor(tensors, tensor_name, shape, dtype, expert_idx=None):
    """Load a tensor from the bring-up manifest (safetensors).

    For bring-up, the manifest is model_weights.json which has metadata but not
    the actual binary data (which is in safetensors). This will fail gracefully
    if safetensors are not available.
    """
    info = tensors.get(tensor_name)
    if info is None:
        raise KeyError(f"Tensor '{tensor_name}' not found in manifest")

    # Check if this tensor is in the bring-up manifest
    if 'offset' in info or 'data' in info:
        # Bring-up manifest has inline data or offsets
        offset = info.get('offset', 0)
        filename = info.get('filename')
        if filename:
            rows, cols = shape[0], shape[1] if len(shape) > 1 else 1
            return load_bf16_matrix(filename, offset, rows, cols)
        elif 'data' in info:
            # Inline data
            return list(info['data'])

    # Otherwise, try to load from safetensors
    filename = _get_safetensors_path(tensors, tensor_name)
    if filename and filename.exists():
        from safetensors import safe_open
        with safe_open(filename, framework="pt") as f:
            tensor_data = f.get_tensor(tensor_name)
            if dtype == 'bf16':
                import torch
                return tensor_data.detach().float().numpy().tolist()
            return tensor_data.tolist()
    else:
        raise KeyError(f"Safetensors file for '{tensor_name}' not found")


def _load_safe_tensor(tensors, tensor_name, shape, dtype):
    """Load tensor, return None if not available."""
    try:
        return _load_tensor(tensors, tensor_name, shape, dtype)
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Coherence metrics
# ---------------------------------------------------------------------------
def coherence_metrics(token_ids, probs, top_idx):
    """Compute coherence / quality metrics for a generation."""
    eos_token_id = 1  # Gemma EOS

    repeated = sum(1 for i in range(1, len(token_ids))
                   if token_ids[i] == token_ids[i - 1])
    repeat_rate = repeated / max(1, len(token_ids) - 1)

    entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
    max_entropy = math.log(len(probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    has_eos = eos_token_id in token_ids

    return {
        'repeat_rate': repeat_rate,
        'router_entropy_bits': entropy,
        'router_normalized_entropy': normalized_entropy,
        'has_eos': has_eos,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Gemma 4 one-layer CPU reference')
    parser.add_argument('--manifest', required=True, help='Path to model_weights.json')
    parser.add_argument('--token-id', type=int, default=1234,
                        help='Input token ID (default: 1234)')
    parser.add_argument('--layer', type=int, default=0,
                        help='Layer index to validate (default: 0)')
    parser.add_argument('--dump-activations', action='store_true',
                        help='Print intermediate activations')
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    tensors = manifest['tensors']
    config = manifest.get('config', {})

    print(f"[check] Loaded manifest: {args.manifest}")
    print(f"[check] Config: hidden={config.get('hidden_size')}, "
          f"layers={config.get('num_hidden_layers')}, "
          f"experts={config.get('num_experts')}")

    # Create a dummy input embedding (normally from embed_tokens)
    # For validation, use a random vector
    import random
    random.seed(42)
    hidden = [random.uniform(-0.1, 0.1) for _ in range(HIDDEN_SIZE)]

    print(f"[check] Running layer {args.layer} on token {args.token_id}")
    output, info = gemma_layer_forward(args, hidden, args.layer, token_pos=0)

    if args.dump_activations:
        print(f"[check] Router scores (first 10): {info['router_scores'][:10]}")
        print(f"[check] Output (first 20): {output[:20]}")

    metrics = coherence_metrics([args.token_id], info['router_scores'], info['top_k_idx'])
    print(f"[check] Coherence metrics: repeat_rate={metrics['repeat_rate']:.3f}, "
          f"router_entropy={metrics['router_entropy_bits']:.3f}, "
          f"normalized_entropy={metrics['router_normalized_entropy']:.3f}")

    print("[check] One-layer reference complete.")
    print(f"[check] Expected tolerances vs C reference:")
    print(f"  - Router softmax:          rel_err < 1e-3")
    print(f"  - GeGLU activation:       rel_err < 1e-3")
    print(f"  - Expert output (BF16):   rel_err < 1e-2")
    print(f"  - Combined dense+MoE:     rel_err < 1e-2")


if __name__ == '__main__':
    main()
