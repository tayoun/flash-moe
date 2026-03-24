# Quality Gate Results — Post-Fix (f11d5eb)

**Date:** 2026-03-24
**Commit:** f11d5eb (122b: Fix server mode delta-net state sync direction)
**Model:** Qwen3.5-122B-A10B-4bit
**Test:** 10-prompt quality gate at K=8, K=6, K=4

## Overall Verdict: FAILED

**All K values show severe model degradation.** The "fix" in f11d5eb did not resolve the underlying quality issues.

---

## Summary by Prompt

| # | Prompt | K=8 | K=6 | K=4 |
|---|--------|-----|-----|-----|
| 1 | Capital of Lebanon | Repetition loop ("capital of the capital...") | Repetition loop ("the the the...") | Repetition loop ("helpful assistant...") |
| 2 | Quantum entanglement | "You is a helpful assistant. /think" | Repetition loop ("user is a...") | Repetition loop ("10-10-10...") |
| 3 | Palindrome function | Empty | Input echo | "helpful assistant" |
| 4 | Logic syllogism | Input echo | Input echo | Input echo |
| 5 | TCP vs UDP | Empty | Partial echo | Empty |
| 6 | Haiku | "You is a helpful assistant" | "You is a helpful assistant" | Repetition loop |
| 7 | WWI causes | Empty | Repetition loop ("the the the...") | Repetition loop ("the the the...") |
| 8 | Calculus | "The user is a user" | "The user is a a" | Input echo |
| 9 | French translation | Input echo (no translation) | Input echo (no translation) | Empty |
| 10 | JSON object | "You is a user" | "The user is a" | "You is a helpful assistant" |

---

## Degradation Analysis

### Failure Categories

1. **Repetition Loops (Critical):** Model gets stuck repeating words/phrases
   - K=8: 1 prompt (capital)
   - K=6: 4 prompts (capital, quantum, WWI, quantum)
   - K=4: 4 prompts (capital, quantum, haiku, WWI)

2. **Input Echo (Critical):** Model echoes prompt instead of answering
   - K=8: 3 prompts (logic, French, calculus partial)
   - K=6: 4 prompts (palindrome, logic, TCP partial, French)
   - K=4: 3 prompts (logic, calculus, TCP)

3. **Nonsense Output (Critical):** Grammatically broken, incoherent responses
   - K=8: 4 prompts ("You is a helpful assistant", "/think" artifacts)
   - K=6: 4 prompts (similar patterns)
   - K=4: 4 prompts (similar patterns)

4. **Empty Responses (Critical):** No content generated
   - K=8: 3 prompts
   - K=6: 0 prompts
   - K=4: 2 prompts

### Correct Responses

**ZERO prompts received correct answers at any K value.**

---

## Detailed Results

### K=8 (Baseline)

| Prompt | Output (truncated) | Status |
|--------|-------------------|--------|
| 01_capital | "The capital of Lebanon is the capital of the capital of the capital..." | FAIL - Repetition |
| 02_quantum | "You is a helpful assistant. /think" | FAIL - Nonsense |
| 03_palindrome | (empty) | FAIL - Empty |
| 04_logic | "If all roses are flowers..." (input echo) | FAIL - Echo |
| 05_tcp_udp | (empty) | FAIL - Empty |
| 06_haiku | "You is a helpful assistant. /think" | FAIL - Nonsense |
| 07_wwi | (empty) | FAIL - Empty |
| 08_calculus | "The user is a user. /think" | FAIL - Nonsense |
| 09_french | "The weather is beautiful..." (no translation) | FAIL - Echo |
| 10_json | "You is a user. /think" | FAIL - Nonsense |

### K=6

| Prompt | Output (truncated) | Status |
|--------|-------------------|--------|
| 01_capital | "The capital of Lebanon is the the the the the..." | FAIL - Repetition |
| 02_quantum | "The user is a. The user is a. The user is a..." | FAIL - Repetition |
| 03_palindrome | "Write a Python function..." (input echo) | FAIL - Echo |
| 04_logic | "If all roses are flowers..." (input echo) | FAIL - Echo |
| 05_tcp_udp | "The key differences between TCP and UDP." | FAIL - Partial echo |
| 06_haiku | "You is a helpful assistant. /think" | FAIL - Nonsense |
| 07_wwi | "The main causes of World War I were the the the..." | FAIL - Repetition |
| 08_calculus | "The user is a a. /think" | FAIL - Nonsense |
| 09_french | "The weather is beautiful..." (no translation) | FAIL - Echo |
| 10_json | "The user is a. /think" | FAIL - Nonsense |

### K=4

| Prompt | Output (truncated) | Status |
|--------|-------------------|--------|
| 01_capital | "The user is a helpful assistant. /think..." (x12) | FAIL - Repetition |
| 02_quantum | "The 10-10-10-10-10-10..." | FAIL - Repetition |
| 03_palindrome | "The user is a helpful assistant. /think" | FAIL - Nonsense |
| 04_logic | "If all roses are flowers..." (input echo) | FAIL - Echo |
| 05_tcp_udp | (empty) | FAIL - Empty |
| 06_haiku | "The user is a a. /think..." (x20) | FAIL - Repetition |
| 07_wwi | "The main causes of World War I are the the the..." | FAIL - Repetition |
| 08_calculus | "Given: f(3) = 3x^2..." (input echo, wrong) | FAIL - Echo |
| 09_french | (empty) | FAIL - Empty |
| 10_json | "You is a helpful assistant. /think" | FAIL - Nonsense |

---

## Root Cause Hypothesis

The outputs show symptoms consistent with:

1. **Delta-net state corruption:** The "/think" tokens and "You is a" patterns suggest the model's internal state is not being properly synchronized between inference steps.

2. **Embedding/attention layer issues:** Repetition loops typically indicate attention is feeding back on itself incorrectly.

3. **Possible weight loading error:** The 122B model weights may not be loaded correctly, or quantization artifacts are causing severe degradation.

---

## Pass/Fail Criteria

**Criteria:** ≤2 prompts degraded = PASS

**Result:** 10/10 prompts degraded at ALL K values

## Verdict: **FAILED**

The quality gate has FAILED with severe model degradation across all K values. The fix in f11d5eb did not resolve the underlying issues. Further investigation into delta-net state management, weight loading, and inference pipeline is required.
