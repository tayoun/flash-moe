#!/usr/bin/env bash
# quality_gate.sh — Phase 4.2: A/B prompt suite for CAR quality validation
#
# Usage:
#   MODEL_DIR=/path/to/model ./quality_gate.sh
#
# Runs 10 prompts with CAR disabled (--car-threshold 1.0) and
# CAR enabled (--car-threshold 0.35), both at temperature=0, max_tokens=256, --k 8.
# Outputs side-by-side comparison for manual review.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-${MODEL:-}}"
INFER="${REPO_DIR}/metal_infer/infer"
VOCAB="${VOCAB:-${REPO_DIR}/metal_infer/vocab_122b.bin}"
PORT_BASE="${PORT_BASE:-8200}"
MAX_TOKENS=256
K=8
CAR_THRESHOLD="${CAR_THRESHOLD:-0.35}"

if [[ -z "${MODEL_DIR}" ]]; then
    echo "ERROR: MODEL_DIR is not set"
    exit 1
fi

PROMPTS=(
    "What is the capital of Lebanon?"
    "Explain quantum entanglement to a 10-year-old."
    "Write a Python function to find the longest palindromic substring."
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
    "Summarize the key differences between TCP and UDP."
    "Write a haiku about machine learning."
    "What were the main causes of World War I?"
    "Given: f(x) = 3x² + 2x - 1. Find f'(x) and f(3)."
    "Translate to French: The weather is beautiful today and I would like to go for a walk."
    "Compare the economic models of capitalism and socialism."
)

PROMPT_LABELS=(
    "factual"
    "explanation"
    "code"
    "logic"
    "technical"
    "creative"
    "history"
    "math"
    "translation"
    "analysis"
)

OUT_DIR="${REPO_DIR}/quality_gate_results"
mkdir -p "${OUT_DIR}"

query_server() {
    local port=$1
    local prompt=$2
    python3 - "${port}" "${MAX_TOKENS}" "${prompt}" <<'PY'
import json, sys, urllib.request

port = int(sys.argv[1])
max_tokens = int(sys.argv[2])
prompt = sys.argv[3]
url = f"http://127.0.0.1:{port}/v1/chat/completions"

payload = {
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": True,
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

text_parts = []
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
            content = delta.get("content")
            if content:
                text_parts.append(content)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    text_parts = [f"[ERROR: {e}]"]

print("".join(text_parts))
PY
}

run_suite() {
    local label=$1
    local port=$2
    local extra_args=$3

    echo "=== Starting server: ${label} (port ${port}) ==="
    ${INFER} \
        --model "${MODEL_DIR}" \
        --vocab "${VOCAB}" \
        --k ${K} \
        ${extra_args} \
        --serve "${port}" >/dev/null 2>&1 &
    local pid=$!

    # Wait for server
    for _ in $(seq 1 60); do
        if curl -s --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: server exited during startup (${label})"
            return 1
        fi
        sleep 1
    done

    if ! curl -s --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "ERROR: server did not become ready (${label})"
        kill "${pid}" 2>/dev/null || true
        return 1
    fi

    echo "=== Running ${#PROMPTS[@]} prompts (${label}) ==="
    for i in "${!PROMPTS[@]}"; do
        local prompt="${PROMPTS[$i]}"
        local plabel="${PROMPT_LABELS[$i]}"
        echo "  [${label}] prompt $((i+1))/${#PROMPTS[@]}: ${plabel}"
        local outfile="${OUT_DIR}/${label}_${plabel}.txt"
        query_server "${port}" "${prompt}" > "${outfile}" 2>/dev/null
    done

    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    echo "=== Done: ${label} ==="
}

# Run baseline (CAR disabled)
run_suite "baseline" "${PORT_BASE}" "--car-threshold 1.0"

# Run CAR enabled
PORT_CAR=$((PORT_BASE + 1))
run_suite "car" "${PORT_CAR}" "--car-threshold ${CAR_THRESHOLD}"

# Compare outputs
echo ""
echo "================================================================"
echo "  QUALITY GATE COMPARISON: baseline vs CAR (threshold=${CAR_THRESHOLD})"
echo "================================================================"
echo ""

degraded=0
for i in "${!PROMPTS[@]}"; do
    plabel="${PROMPT_LABELS[$i]}"
    baseline_file="${OUT_DIR}/baseline_${plabel}.txt"
    car_file="${OUT_DIR}/car_${plabel}.txt"

    echo "--- Prompt $((i+1)): ${plabel} ---"
    echo "Q: ${PROMPTS[$i]}"
    echo ""
    echo "BASELINE:"
    head -20 "${baseline_file}" 2>/dev/null || echo "[no output]"
    echo ""
    echo "CAR:"
    head -20 "${car_file}" 2>/dev/null || echo "[no output]"
    echo ""

    # Basic check: CAR output should not be empty if baseline is not
    baseline_len=$(wc -c < "${baseline_file}" 2>/dev/null || echo 0)
    car_len=$(wc -c < "${car_file}" 2>/dev/null || echo 0)
    if [[ ${baseline_len} -gt 10 && ${car_len} -lt 10 ]]; then
        echo "*** WARNING: CAR output much shorter than baseline ***"
        degraded=$((degraded + 1))
    fi
    echo "---"
    echo ""
done

echo "================================================================"
echo "  RESULT: ${degraded} prompts with potential degradation"
if [[ ${degraded} -le 2 ]]; then
    echo "  GATE: PASSED (≤2 degraded)"
else
    echo "  GATE: FAILED (>2 degraded)"
fi
echo "================================================================"
echo ""
echo "Full outputs saved to: ${OUT_DIR}/"
echo "Manual review recommended for factual accuracy and reasoning quality."
