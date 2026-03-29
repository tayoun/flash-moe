#!/usr/bin/env bash
# bench.sh — simple local benchmark for flash-moe
#
# Usage:
#   MODEL_DIR=/path/to/model ./bench.sh
#   MODEL_DIR=/path/to/model EXTRA_ARGS="--car-threshold 0.35" ./bench.sh
#
# Prints:
#   MEMORY: rss_mb=<...> sys_free_mb=<...>
#   RESULT: tok_s=<...> ttft_s=<...> crashes=<...> quality=<...> tokens=<...>
#
# Memory tracked via `ps` for server RSS and `vm_stat` for system free pages.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-${MODEL:-}}"
INFER="${REPO_DIR}/metal_infer/infer"

# 122B-aware defaults: detect from MODEL_DIR path
if [[ "${MODEL_DIR}" == *122B* || "${MODEL_DIR}" == *122b* ]]; then
    WEIGHTS="${WEIGHTS:-${REPO_DIR}/metal_infer/out_122b/model_weights.bin}"
    MANIFEST="${MANIFEST:-${REPO_DIR}/metal_infer/out_122b/model_weights.json}"
    VOCAB="${VOCAB:-${REPO_DIR}/metal_infer/vocab_122b.bin}"
    SSD_PATH="${SSD_PATH:-${REPO_DIR}/metal_infer/out_122b/packed_experts_ssd.bin}"
    K="${K:-8}"
    STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-120}"
else
    WEIGHTS="${WEIGHTS:-${REPO_DIR}/metal_infer/out_35b/model_weights.bin}"
    MANIFEST="${MANIFEST:-${REPO_DIR}/metal_infer/out_35b/model_weights.json}"
    VOCAB="${VOCAB:-${REPO_DIR}/metal_infer/vocab.bin}"
    K="${K:-6}"
    STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-30}"
fi

PORT="${PORT:-8100}"
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
    ${SSD_PATH:+--offload-ssd "${SSD_PATH}"} \
    ${EXTRA_ARGS} \
    --serve "${PORT}" >/dev/null 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 "${STARTUP_TIMEOUT}"); do
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

python3 - "${PORT}" "${MAX_TOKENS}" "${SERVER_PID}" <<'PY'
import json
import sys
import time
import urllib.request
import subprocess

port = int(sys.argv[1])
max_tokens = int(sys.argv[2])
server_pid = int(sys.argv[3])

def get_server_rss(pid):
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "rss="], text=True, timeout=2)
        return int(out.strip()) / 1024  # KB -> MB
    except Exception:
        return 0.0

def get_sys_free_mb():
    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=3)
        for line in out.split("\n"):
            if "Pages free:" in line:
                parts = line.split()
                free_pages = int(parts[-1].rstrip("."))
                return free_pages * 16384 / (1024 * 1024)  # pages -> MB
    except Exception:
        pass
    return 0.0

rss_before = get_server_rss(server_pid)
free_before = get_sys_free_mb()
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

rss_after = get_server_rss(server_pid)
free_after = get_sys_free_mb()

print(f"MEMORY: rss_mb={rss_before:.0f}->{rss_after:.0f} sys_free_mb={free_before:.0f}->{free_after:.0f}")
print(f"RESULT: tok_s={tok_s:.2f} ttft_s={ttft:.2f} crashes=0 quality={quality} tokens={tokens}")
PY
