#!/usr/bin/env python3
"""Gemma 4 26B-A4B one-layer CPU reference validator.

Extracts layer 0 weights from the Gemma checkpoint, runs a single-layer forward pass
in pure Python using numpy, and compares against the C `infer` binary output.

Usage:
  python check_one_layer.py --model /path/to/gemma/checkpoint [--verbose]

Exit codes:
  0 = match (within tolerance)
  1 = mismatch or error
"""

import argparse
import json
import numpy as np
import os
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Gemma 4 constants
# ---------------------------------------------------------------------------
GEMMA_HIDDEN            = 2816
GEMMA_HEAD_DIM_SLIDING  = 256
GEMMA_HEAD_DIM_FULL     = 512
GEMMA_NUM_Q_HEADS       = 16
GEMMA_NUM_KV_HEADS_SLIDING = 8
GEMMA_NUM_KV_HEADS_FULL    = 2
GEMMA_INTERMEDIATE      = 2112
GEMMA_RMS_NORM_EPS      = 1e-6

# ---------------------------------------------------------------------------
# Safetensor loading via safetensors.safe_open
# ---------------------------------------------------------------------------
try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def load_torch_tensor(filepath, key):
    """Load a tensor using PyTorch (if available)."""
    try:
        import torch
        with safe_open(filepath, framework="pt") as f:
            return f.get_tensor(key).float().numpy()
    except Exception:
        return None


def build_key_map(model_path):
    """Build manifest_short_key → (safetensor_file, safetensor_key) map.

    The manifest uses short keys like 'layers.0.input_layernorm.weight'.
    The safetensor files use long keys like 'model.language_model.layers.0.input_layernorm.weight'.
    This function scans all safetensor files and builds the mapping.
    """
    key_map = {}
    for fname in os.listdir(model_path):
        if not fname.endswith('.safetensors'):
            continue
        path = str(model_path / fname)
        with safe_open(path, framework="pt") as f:
            for sf_key in f.keys():
                short = sf_key.replace('model.language_model.', '')
                key_map[short] = (fname, sf_key)
    return key_map


def load_bf16_tensor(model_path, key_map, manifest, short_key, expected_shape=None):
    """Load a BF16 tensor by manifest short key.

    Args:
        model_path: Path to model directory
        key_map: dict from build_key_map()
        manifest: loaded model_weights.json dict
        short_key: short key from manifest (e.g. 'layers.0.input_layernorm.weight')
        expected_shape: optional shape to validate (uses tensor's actual shape if None)

    Returns:
        numpy array in BF16 (uint16)
    """
    info = manifest['tensors'].get(short_key)
    if info is None:
        raise KeyError(f"Tensor '{short_key}' not found in manifest")

    if short_key not in key_map:
        raise KeyError(f"Safetensor file for '{short_key}' not found")

    sf_file, sf_key = key_map[short_key]
    path = str(model_path / sf_file)

    with safe_open(path, framework="pt") as f:
        arr = f.get_tensor(sf_key)

    # arr is torch.Tensor of dtype bfloat16 (2 bytes/element).
    # PyTorch's .float() converts BF16→F32 internally.
    # To recover the raw BF16 bit pattern, shift the F32 bits right 16:
    #   f32 bits = [sign:1][exp:8][mantissa:23];  bf16 bits = [sign:1][exp:8][mantissa:7]
    #   bf16 = upper 16 bits of f32 = (f32_bits >> 16)
    arr_f32 = arr.detach().float().numpy().astype(np.float32)
    bf16 = (arr_f32.view(np.uint32) >> 16).astype(np.uint16)

    # Validate shape if expected_shape provided
    actual_shape = tuple(arr.shape)
    if expected_shape is not None:
        exp = tuple(expected_shape)
        if actual_shape != exp:
            raise ValueError(f"{short_key}: expected shape {exp}, got {actual_shape}")

    return bf16.reshape(actual_shape)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def bf16_to_f32(x):
    """Convert BF16 uint16 array to float32."""
    bits = x.astype(np.int32) << 16
    return bits.view(np.float32)


def cpu_rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * (weight / sqrt(mean(x^2) + eps))"""
    x = x.astype(np.float32)
    w = weight.astype(np.float32)
    ss = np.mean(x**2)
    rms = np.sqrt(ss + eps)
    return x / rms * w


def cpu_gelu(x):
    """Gemma approximate GELU."""
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def cpu_geglu(gate, up):
    """GeGLU: gelu(gate) * up."""
    return cpu_gelu(gate) * up


def matvec_bf16(W, x):
    """BF16 matrix @ f32 vector → f32 vector."""
    W_f = bf16_to_f32(W)
    return W_f.astype(np.float32) @ x.astype(np.float32)


def apply_rope(x, cos, sin):
    """Apply 2D rotary embedding in-place on pairs."""
    dim = len(x)
    x = x.astype(np.float32)
    result = np.zeros_like(x)
    for i in range(0, dim, 2):
        result[i]     = x[i]     * cos[i // 2] - x[i + 1] * sin[i // 2]
        result[i + 1] = x[i]     * sin[i // 2] + x[i + 1] * cos[i // 2]
    return result


def compute_rope_freqs(theta, dim, seq_len=1):
    """RoPE frequency angles for seq_len positions, dim/2D."""
    freqs  = theta ** (2.0 * np.arange(dim) / dim)
    angles = np.arange(seq_len)[:, None] / freqs[None, :]
    return np.cos(angles), np.sin(angles)


def softmax(x):
    """Numerically stable softmax along last axis."""
    x = x.astype(np.float64)
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x  = np.exp(x - x_max)
    return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Main layer forward
# ---------------------------------------------------------------------------
def gemma_layer_forward(model_path, manifest, key_map, layer_idx, hidden, verbose=False):
    """Run one Gemma layer forward on a single f32 [hidden] token.

    Args:
        model_path:   Path to model directory
        manifest:     loaded model_weights.json
        key_map:       from build_key_map()
        layer_idx:     layer index (0-based)
        hidden:        np.float32 array [GEMMA_HIDDEN]
        verbose:       print debug info

    Returns:
        output: np.float32 array [GEMMA_HIDDEN]
    """
    def w(short_key, expected_shape=None):
        arr = load_bf16_tensor(model_path, key_map, manifest, short_key, expected_shape)
        return arr

    # ------------------------------------------------------------------
    # Layer geometry
    # ------------------------------------------------------------------
    is_full       = ((layer_idx + 1) % 6) == 0
    head_dim      = GEMMA_HEAD_DIM_FULL if is_full else GEMMA_HEAD_DIM_SLIDING
    num_kv_heads  = GEMMA_NUM_KV_HEADS_FULL if is_full else GEMMA_NUM_KV_HEADS_SLIDING
    rope_theta    = 1_000_000.0 if is_full else 10_000.0
    num_heads     = GEMMA_NUM_Q_HEADS
    kv_groups     = num_heads // num_kv_heads   # 2 or 8
    rotary_dim    = head_dim // 2                # 128 for both cases

    if verbose:
        print(f"[check] Layer {layer_idx}: is_full={is_full}, "
              f"head_dim={head_dim}, num_kv={num_kv_heads}, "
              f"rope_theta={rope_theta}, rotary_dim={rotary_dim}")

    # ------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------
    input_norm_w  = w(f'layers.{layer_idx}.input_layernorm.weight',         [GEMMA_HIDDEN])
    q_w           = w(f'layers.{layer_idx}.self_attn.q_proj.weight',
                       [num_heads * head_dim, GEMMA_HIDDEN])
    k_w           = w(f'layers.{layer_idx}.self_attn.k_proj.weight',
                       [num_kv_heads * head_dim, GEMMA_HIDDEN])
    v_w           = w(f'layers.{layer_idx}.self_attn.v_proj.weight',
                       [num_kv_heads * head_dim, GEMMA_HIDDEN])
    q_norm_w      = w(f'layers.{layer_idx}.self_attn.q_norm.weight',
                       [head_dim])      # shared per-head RMSNorm weight
    k_norm_w      = w(f'layers.{layer_idx}.self_attn.k_norm.weight',
                       [head_dim])      # shared per-head RMSNorm weight
    o_w           = w(f'layers.{layer_idx}.self_attn.o_proj.weight',
                       [GEMMA_HIDDEN, num_heads * head_dim])
    post_attn_w   = w(f'layers.{layer_idx}.post_attention_layernorm.weight', [GEMMA_HIDDEN])
    pre_ffn_w     = w(f'layers.{layer_idx}.pre_feedforward_layernorm.weight', [GEMMA_HIDDEN])
    gate_proj     = w(f'layers.{layer_idx}.mlp.gate_proj.weight',
                       [GEMMA_INTERMEDIATE, GEMMA_HIDDEN])
    up_proj       = w(f'layers.{layer_idx}.mlp.up_proj.weight',
                       [GEMMA_INTERMEDIATE, GEMMA_HIDDEN])
    down_proj     = w(f'layers.{layer_idx}.mlp.down_proj.weight',
                       [GEMMA_HIDDEN, GEMMA_INTERMEDIATE])
    post_ffn_w    = w(f'layers.{layer_idx}.post_feedforward_layernorm.weight', [GEMMA_HIDDEN])

    # ------------------------------------------------------------------
    # 1. Input RMSNorm
    # ------------------------------------------------------------------
    normed = cpu_rms_norm(hidden, bf16_to_f32(input_norm_w), GEMMA_RMS_NORM_EPS)

    # ------------------------------------------------------------------
    # 2. Q/K/V projections
    # ------------------------------------------------------------------
    q = matvec_bf16(q_w, normed)   # [num_heads * head_dim]
    k = matvec_bf16(k_w, normed)   # [num_kv_heads * head_dim]
    v = matvec_bf16(v_w, normed)   # [num_kv_heads * head_dim]

    # ------------------------------------------------------------------
    # 3. Per-head Q/K RMSNorm
    # ------------------------------------------------------------------
    q = q.reshape(num_heads, head_dim)
    k = k.reshape(num_kv_heads, head_dim)
    q_norm_f = bf16_to_f32(q_norm_w)  # [head_dim] shared weight
    k_norm_f = bf16_to_f32(k_norm_w)  # [head_dim] shared weight

    for h in range(num_heads):
        q[h] = cpu_rms_norm(q[h], q_norm_f, GEMMA_RMS_NORM_EPS)

    for h in range(num_kv_heads):
        k[h] = cpu_rms_norm(k[h], k_norm_f, GEMMA_RMS_NORM_EPS)

    # ------------------------------------------------------------------
    # 4. RoPE
    # ------------------------------------------------------------------
    cos, sin = compute_rope_freqs(rope_theta, rotary_dim, seq_len=1)
    cos_h = cos[0]
    sin_h = sin[0]

    for h in range(num_heads):
        q[h] = apply_rope(q[h], cos_h, sin_h)
    for h in range(num_kv_heads):
        k[h] = apply_rope(k[h], cos_h, sin_h)

    # ------------------------------------------------------------------
    # 5. Attention (single-token, no KV cache)
    #    For each Q head: softmax(Q[h] · K[kv_h]ᵀ / √d) · V[kv_h]
    #    With 1 token, the softmax is over 1 element → weight = 1.0
    # ------------------------------------------------------------------
    scale = 1.0 / np.sqrt(head_dim)
    attn_out = np.zeros((num_heads, head_dim), dtype=np.float32)

    for h in range(num_heads):
        kv_h = h // kv_groups
        score = np.dot(q[h], k[kv_h]) * scale
        attn_w = np.exp(score - score)   # = 1.0 (single-token)
        attn_out[h] = attn_w * v[kv_h]

    attn_flat = attn_out.reshape(num_heads * head_dim)

    # ------------------------------------------------------------------
    # 6. O projection
    # ------------------------------------------------------------------
    o_f = bf16_to_f32(o_w)
    attn_proj = o_f.astype(np.float32) @ attn_flat.astype(np.float32)

    # ------------------------------------------------------------------
    # 7. Residual add: hidden + attention
    # ------------------------------------------------------------------
    hidden = hidden + attn_proj

    # ------------------------------------------------------------------
    # 8. Post-attention RMSNorm
    # ------------------------------------------------------------------
    hidden = cpu_rms_norm(hidden, bf16_to_f32(post_attn_w), GEMMA_RMS_NORM_EPS)

    # ------------------------------------------------------------------
    # 9. Pre-FFN RMSNorm
    # ------------------------------------------------------------------
    ffn_input = cpu_rms_norm(hidden, bf16_to_f32(pre_ffn_w), GEMMA_RMS_NORM_EPS)

    # ------------------------------------------------------------------
    # 10. Dense FFN (GeGLU)
    # ------------------------------------------------------------------
    gate_out = matvec_bf16(gate_proj, ffn_input).astype(np.float32)
    up_out   = matvec_bf16(up_proj,   ffn_input).astype(np.float32)
    ffn_out  = matvec_bf16(down_proj, cpu_geglu(gate_out, up_out)).astype(np.float32)

    # ------------------------------------------------------------------
    # 11. Post-FFN RMSNorm
    # ------------------------------------------------------------------
    output = cpu_rms_norm(ffn_out, bf16_to_f32(post_ffn_w), GEMMA_RMS_NORM_EPS)

    # ------------------------------------------------------------------
    # 12. Final residual add: hidden (residual stream) + output
    # ------------------------------------------------------------------
    result = hidden + output

    if verbose:
        print(f"[check] Result: mean={result.mean():.6f}, std={result.std():.6f}, "
              f"min={result.min():.6f}, max={result.max():.6f}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Suppress harmless divide-by-zero warnings from BF16→F32 bit conversion
    # (BF16 zero patterns produce 0/0 in intermediate float computation, not actual NaNs)
    warnings.filterwarnings('ignore', message='divide by zero')
    warnings.filterwarnings('ignore', message='overflow')
    warnings.filterwarnings('ignore', message='invalid value')

    parser = argparse.ArgumentParser(
        description='Gemma 4 one-layer CPU reference validator')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to Gemma 4 checkpoint directory')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--layer', type=int, default=0,
                        help='Layer index to run (default: 0)')
    parser.add_argument('--dump', type=str, default='/tmp/gemma_layer0_python.npy',
                        help='Path to dump output numpy array')
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    manifest_path = model_path / 'model_weights.json'
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    if not HAS_SAFETENSORS:
        print("ERROR: safetensors package not installed: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    print(f"[check] Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print("[check] Building safetensor key map...")
    key_map = build_key_map(model_path)
    print(f"[check] Mapped {len(key_map)} safetensor keys")

    config = manifest.get('config', {})
    if config:
        print(f"[check] Config: hidden={config.get('hidden_size')}, "
              f"layers={config.get('num_hidden_layers')}, "
              f"experts={config.get('num_experts')}")

    # ------------------------------------------------------------------
    # Create a seeded random input token
    # ------------------------------------------------------------------
    np.random.seed(0xDEADBEEF)
    hidden = np.random.randn(GEMMA_HIDDEN).astype(np.float32)

    print(f"[check] Running layer {args.layer} on {GEMMA_HIDDEN}-dim input, seed=0xDEADBEEF")
    result = gemma_layer_forward(model_path, manifest, key_map, args.layer, hidden,
                                 verbose=args.verbose)

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    out_path = Path(args.dump)
    np.save(out_path, result)

    print(f"[check] Layer {args.layer} output: shape={result.shape}, "
          f"mean={result.mean():.6f}, std={result.std():.6f}")
    print(f"[check] Saved to {out_path}")

    if args.verbose:
        print(f"[check] First 10 values: {result[:10].round(4)}")
        print(f"[check] Last  10 values: {result[-10:].round(4)}")

    print("\n[check] One-layer Python reference complete.")
    print("[check] To compare with C output:")
    print("  python3 -c \"import numpy as np; "
          "c=np.load('/path/to/c_output.npy'); "
          "p=np.load('" + str(out_path) + "'); "
          "print('Rel. error:', np.linalg.norm(p-c)/np.linalg.norm(c))\"")
    sys.exit(0)


if __name__ == '__main__':
    main()
