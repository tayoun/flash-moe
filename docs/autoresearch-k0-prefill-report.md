# B1.1 — Prefill-Only K=0 Report

## Hypothesis
Setting routed-expert `K=0` during prefill only (shared expert only), while keeping decode at `K=8`, should significantly reduce TTFT without changing sustained decode throughput.

## Files Changed
- `metal_infer/infer.m`
- `results.tsv`

## Benchmark Result
Command:
`MODEL_DIR=/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit ./bench.sh`

Captured RESULT line:
`RESULT: tok_s=0.00 ttft_s=0.00 crashes=1 quality=fail tokens=0`

Blocker:
Bench failed at server startup with `[bench] ERROR: server exited during startup`.
Direct run showed precise cause: `bind: Operation not permitted` (sandbox disallows opening the serve port), so TTFT/tok/s could not be measured in this environment.

## Keep/Discard Recommendation
Discard for now in this sandbox context (inconclusive performance data). Re-run the exact benchmark on a host environment where local port bind is permitted before deciding keep/discard for the experiment itself.

## Quality-Gate Follow-Up
Not run. This change alters routing behavior, so if a re-run benchmark succeeds and the experiment is a keep candidate, run:
`MODEL_DIR=/Users/tayoun/models/flash-moe/Qwen3.5-122B-A10B-4bit ./quality_gate.sh`
