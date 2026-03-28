#!/bin/bash
./metal_infer/infer \
  --model /Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit \
  --weights metal_infer/out_122b/model_weights.bin \
  --manifest metal_infer/out_122b/model_weights.json \
  --vocab metal_infer/vocab_122b.bin \
  --prompt-tokens metal_infer/tokenizer_122b.bin \
  --prompt "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nwhat is the capital of Lebanon<|im_end|>\n<|im_start|>assistant\n<think>\n" \
  --tokens 50 \
  --k 8
