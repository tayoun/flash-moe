# Structure

## Overview
Flash-MoE is a C/Metal inference engine for Qwen3.5-35B-A3B MoE on Apple Silicon, with preprocessing scripts for expert indexing/repacking and runtime artifact generation.

## Directory Layout
├── metal_infer/        # Runtime engine, Metal shaders, build targets, tokenizer/vocab exporters
├── docs/               # Engineering docs (structure, decisions, lessons, optimization notes)
├── paper/              # Paper source and generated artifacts
├── README.md           # Public-facing quick start and results
├── CLAUDE.md           # Technical overview and local operator notes
├── build_expert_index_35b.py   # Builds expert index from model metadata
├── repack_experts_35b.py       # Repackages routed experts for runtime reads
├── read_safetensors_headers_35b.py  # Safetensors header inspection utility
└── bench.sh            # Lightweight benchmark helper against local server

## Key Files
- `metal_infer/infer.m` — main inference server/runtime
- `metal_infer/shaders.metal` — Metal compute kernels
- `metal_infer/chat.m` — interactive chat client
- `metal_infer/Makefile` — build entry points (`infer`, `chat`)
- `docs/optimization-experiments-q4.md` — performance evolution (M3 baseline + M4 results)
