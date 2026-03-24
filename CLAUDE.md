# Flash-MoE Technical Overview

Flash-MoE is a C/Metal inference runtime for Qwen MoE models on Apple Silicon, designed around SSD expert streaming and unified memory constraints.

## Current Public Positioning

- **Current headline (M4 Mac mini, 16GB): 11.5 tok/s sustained, 2.5s TTFT, quality/tool-calling stable at K=6**
- **Original baseline (M3 Max, 48GB): 4.4 tok/s at K=4**
- **Net improvement: 2.6x faster on cheaper hardware**

## Results Snapshot

| Machine | K | Sustained tok/s | TTFT | Status |
|---|---:|---:|---:|---|
| M3 Max 48GB (original) | 4 | 4.4 | ~5.6s | Historical baseline |
| M4 16GB (current) | 6 | **11.5** | **2.5s** | Current production target |

## Architecture

- Expert-routing MoE inference with SSD streaming for routed weights.
- Runtime is pure C/Objective-C + Metal kernels.
- Routing fanout `K` is configurable at runtime (`--k`) and no longer hardcoded.
- M3 optimization track validated `K=4`; M4 production track runs `K=6`.

## M4 Optimization Track

Primary changes that moved the system from M3 baseline behavior to current M4 throughput:

1. `tg128` 4-bit matvec kernels on hot projections.
2. Encoder/prefill coalescing to reduce per-token overhead.
3. Kernel fusion of latency-critical sequences to reduce intermediate traffic.

These changes compound with the existing SSD-streaming pipeline and yield the observed 2.6x end-to-end gain.

## Hardware Targets

- **M3 Max MacBook Pro (48GB):** original development and baseline profiling.
- **M4 Mac mini (16GB):** current optimized deployment target.

## Generic Setup (No User-Specific Paths)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip numpy tokenizers

export MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<snapshot_id>}"

python3 build_expert_index_35b.py --model-path "$MODEL_DIR" --out expert_index_35b.json
python3 repack_experts_35b.py --index expert_index_35b.json
python3 metal_infer/extract_weights_35b.py --model "$MODEL_DIR" --output metal_infer/out_35b
python3 metal_infer/export_tokenizer_35b.py "$MODEL_DIR/tokenizer.json" metal_infer/tokenizer.bin
python3 metal_infer/export_vocab_35b.py "$MODEL_DIR/tokenizer.json" metal_infer/vocab.bin

cd metal_infer && make infer chat && cd ..

./metal_infer/infer \
  --model "$MODEL_DIR" \
  --weights metal_infer/out_35b/model_weights.bin \
  --manifest metal_infer/out_35b/model_weights.json \
  --vocab metal_infer/vocab.bin \
  --k 6 \
  --serve 8000
```
