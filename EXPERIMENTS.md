# Qwen3.5-122B on M4 mini (16GB) — Ranked Experiment Plan

Date: 2026-03-30
Scope: `Qwen3.5-122B-A10B-4bit` with experts in a single packed file on Seagate HDD (`/Volumes/Seagate Backup Plus Drive/packed_experts_ssd.bin`), `infer` server on `:9000`, `--kv-compression turbo3 --k 8`, current decode `~7.5-8.2 tok/s`.
Constraint: no model re-download, no hardware changes.

## Research synthesis (Dan Woods + Anemll fork + Reddit)

1. **Dan Woods (original flash-moe):** meaningful gains came from kernel work and OS page cache; many speculative I/O ideas regressed (including `dispatch_io`, aggressive prefetch, and several prediction variants).
2. **Anemll m5-nax fork (M5 Max):** reducing expert bytes/token drove most gains. Reported `9.5 tok/s` (4-bit MLX) to `12.9 tok/s` (Q3 + `--cache-io-split 4`) with near-4bit quality, while 2-bit reached `14.5 tok/s` but with notable quality drop (PPL degradation).
3. **Reddit autoresearch summary (36 experiments):** progression from `10.61` to `20.34 tok/s` on M5 Max emphasized reduced expert payload + better I/O overlap/concurrency, with explicit quality gating to reject regressions.

Takeaway for this M4+HDD setup: prioritize experiments that either **reduce bytes read per token** or **hide HDD latency**, while preserving quality.

## Ranked experiments (highest expected impact first)

## 1) 2-bit experts on existing 2-bit kernel path
- **What it is:** switch expert streaming from 4-bit (~5.3MB/expert) to 2-bit (~3.3MB/expert-equivalent in your pipeline) using your existing `dequant_matvec_2bit_qwen` path.
- **Expected improvement:** `+1.8 to +3.0 tok/s`.
- **How to implement:**
  - Generate a clean 2-bit packed experts file from current 4-bit source (no re-download):
    - `python3 /Users/tayoun/projects-external/flash-moe/metal_infer/convert_experts_2bit_fixed.py --input "/Volumes/Seagate Backup Plus Drive/packed_experts_ssd.bin" --output "/Volumes/Seagate Backup Plus Drive/packed_experts_2bit_ssd.bin"`
  - Launch server with 2-bit experts:
    - `./metal_infer/infer --model ~/models/flash-moe/Qwen3.5-122B-A10B-4bit --weights metal_infer/out_122b/model_weights.bin --manifest metal_infer/out_122b/model_weights.json --vocab metal_infer/vocab.bin --offload-ssd "/Volumes/Seagate Backup Plus Drive/packed_experts_2bit_ssd.bin" --2bit --kv-compression turbo3 --k 8 --serve 9000 --sample --cache-entries 0`
  - Validate code paths before run:
    - `metal_infer/infer.m`: `ctx->matvec_2bit = makePipe(@"dequant_matvec_2bit_qwen")`
    - `metal_infer/infer.m`: `cfg.*_off_2` offsets used under `g_use_2bit`
    - `metal_infer/shaders.metal`: `kernel void dequant_matvec_2bit_qwen`
- **Risk level:** `Medium` (possible quality regression; run quality prompts).

## 2) I/O concurrency sweep for HDD (threads + queue depth + backend)
- **What it is:** tune existing async read knobs to match Seagate HDD behavior and USB path.
- **Expected improvement:** `+0.8 to +1.6 tok/s`.
- **How to implement:**
  - Sweep these flags in a fixed benchmark matrix:
    - `--io-threads {4,8,12,16}`
    - `--aio-depth {4,8,12,16,24}`
    - with and without `--dispatch-io`
  - Baseline command template:
    - `./metal_infer/infer ... --offload-ssd "/Volumes/Seagate Backup Plus Drive/packed_experts_ssd.bin" --k 8 --timing --cache-entries 0 --io-threads 12 --aio-depth 12`
  - Compare p50/p95 decode tok/s and expert I/O timing per token.
- **Risk level:** `Low`.

## 3) Temporal prediction + cache-prior routing bias sweep
- **What it is:** increase predicted-expert hit rate so reads start earlier and more reads land as cache hits.
- **Expected improvement:** `+0.5 to +1.3 tok/s` (high variance; workload-dependent).
- **How to implement:**
  - Enable prediction and sweep cache prior:
    - `--predict --cache-prior {0.1,0.2,0.35,0.5}`
  - Start from known stable server config and add those flags.
  - Track:
    - `Spec routing` hits/attempts from logs
    - decode tok/s and quality on long outputs
- **Risk level:** `Medium` (router bias can hurt quality if too high).

## 4) Sparse RAM hotset preload (2-4GB budget, not full preload)
- **What it is:** preload only high-frequency experts into RAM and fall back to HDD for the long tail.
- **Expected improvement:** `+0.8 to +1.8 tok/s` after warmup.
- **How to implement:**
  - Collect routing frequencies on your real prompts:
    - `./metal_infer/infer ... --freq --collect-routing /tmp/routing.bin`
  - Build per-layer hotset file (extend `profile_experts.py` or add `build_hotset.py`).
  - Code changes in `metal_infer/infer.m`:
    - add `--preload-hotset <path>` and `--preload-hotset-mb <N>`
    - in expert read path: if `(layer,expert)` in hotset, `memcpy` from RAM; else `pread`
- **Risk level:** `Medium` (RAM pressure on 16GB machine).

## 5) Clustered expert repack for sequential HDD reads
- **What it is:** reorder experts on disk by co-access clusters to reduce random read pattern cost.
- **Expected improvement:** `+0.4 to +1.2 tok/s`.
- **How to implement:**
  - Build co-occurrence matrix from routing logs.
  - Use `cluster_experts.py` to generate permutation.
  - Apply permutation when writing a new packed file (extend `metal_infer/pack_experts_ssd.py`).
  - Run inference against clustered packed file via `--offload-ssd`.
- **Risk level:** `Medium-High` (long repack + correctness validation needed).

## Execution priority

1. Experiment 1 (2-bit) — highest expected gain from byte reduction.
2. Experiment 2 (I/O sweep) — fastest low-risk gains for HDD.
3. Experiment 3 (predict + cache-prior) — software-only latency hiding.
4. Experiment 4 (RAM hotset) — stronger upside but needs code changes.
5. Experiment 5 (clustered repack) — do last due repack complexity.

## Measurement protocol

- Keep prompt set fixed across runs.
- For each config: run 3 passes (`cold`, `warm1`, `warm2`).
- Record: decode tok/s, TTFT, expert I/O ms/token, spec-hit rate, quality pass/fail.
- Keep `--cache-entries 0` unless data proves otherwise (aligns with Dan’s OS-page-cache finding).
