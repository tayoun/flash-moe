#!/usr/bin/env bash
# bench.sh — simple local benchmark for flash-moe
#
# Usage:
#   MODEL_DIR=/path/to/model ./bench.sh
#   MODEL_DIR=/path/to/model EXTRA_ARGS="--car-threshold 0.35" ./bench.sh
#
# Prints a single RESULT line:
#   RESULT: tok_s=<...> ttft_s=<...> crashes=<...> quality=<...> tokens=<...>

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-${MODEL:-}}"
INFER="${REPO_DIR}/metal_infer/infer"
WEIGHTS="${WEIGHTS:-${REPO_DIR}/metal_infer/out_35b/model_weights.bin}"
MANIFEST="${MANIFEST:-${REPO_DIR}/metal_infer/out_35b/model_weights.json}"
VOCAB="${VOCAB:-${REPO_DIR}/metal_infer/vocab.bin}"
PORT="${PORT:-8100}"
K="${K:-6}"
MAX_TOKENS="${MAX_TOKENS:-256}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ -z "${MODEL_DIR}" ]]; then
    echo "[bench] ERROR: MODEL_DIR is not set"
    echo "RESULT: tok_s=0.00 ttft_s=0.00 crashes=1 quality=fail tokens=0"
    exit 1
fi

if [[ ! -x "${INFER}" ]]; then
    echo "[bench] ERROR: infer binary not found. Run: cd metal_infer && make infer"
    echo "RESULT: tok_s=0.00 ttft_s=0.00 crashes=1 quality=fail tokens=0"
    exit 1
fi

for f in "${WEIGHTS}" "${MANIFEST}" "${VOCAB}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[bench] ERROR: missing required file: ${f}"
        echo "RESULT: tok_s=0.00 ttft_s=0.00 crashes=1 quality=fail tokens=0"
        exit 1
    fi
done

${INFER} \
    --model "${MODEL_DIR}" \
    --weights "${WEIGHTS}" \
    --manifest "${MANIFEST}" \
    --vocab "${VOCAB}" \
    --k "${K}" \
    ${EXTRA_ARGS} \
    --serve "${PORT}" >/dev/null 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 30); do
    if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[bench] ERROR: server exited during startup"
        echo "RESULT: tok_s=0.00 ttft_s=0.00 crashes=1 quality=fail tokens=0"
        exit 1
    fi
    sleep 1
done

if ! curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[bench] ERROR: server did not become ready"
    echo "RESULT: tok_s=0.00 ttft_s=0.00 crashes=1 quality=fail tokens=0"
    exit 1
fi

python3 - "${PORT}" "${MAX_TOKENS}" <<'PY'
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
max_tokens = int(sys.argv[2])
url = f"http://127.0.0.1:{port}/v1/chat/completions"

payload = {
    "messages": [{"role": "user", "content": "Explain why mixture-of-experts improves compute efficiency."}],
    "max_tokens": max_tokens,
    "stream": True,
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

start = time.time()
first_token_at = None
tokens = 0
text_parts = []

with urllib.request.urlopen(req, timeout=300) as r:
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
        if not content:
            continue
        if first_token_at is None:
            first_token_at = time.time()
        tokens += 1
        text_parts.append(content)

end = time.time()

ttft = (first_token_at - start) if first_token_at else 0.0
decode_s = (end - (first_token_at or end))
tok_s = (tokens / decode_s) if decode_s > 0 else 0.0
quality = "pass" if tokens >= 32 and len("".join(text_parts).strip()) > 0 else "warn"

print(f"RESULT: tok_s={tok_s:.2f} ttft_s={ttft:.2f} crashes=0 quality={quality} tokens={tokens}")
PY
