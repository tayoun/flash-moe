# Quality Gate Results: 122B Model K-Value Comparison

**Date**: 2024-03-24
**Model**: Qwen3.5-122B-A10B-4bit
**Test Configuration**: temperature=0, max_tokens=256

## CRITICAL FINDING

All K values (K=8, K=6, K=4) produced severely degraded outputs. This is **not** a K-value quality degradation issue - the model is exhibiting fundamental inference failures across all configurations.

## Summary Table

| Prompt | K=8 (Baseline) | K=6 | K=4 | Issue Type |
|--------|----------------|-----|-----|------------|
| 1. Capital of Lebanon | FAIL: Repetition loop | FAIL: "is a 1000" | FAIL: "is a capital" | All broken |
| 2. Quantum explanation | FAIL: "user is a user" loop | FAIL: "10-10-10..." | FAIL: "helpful assistant" loop | All broken |
| 3. Palindrome code | FAIL: "/think" only | FAIL: "/think" only | FAIL: "You at" | All broken |
| 4. Logic puzzle | FAIL: "/think" only | PARTIAL: Echoes question | FAIL: Mixed loop | All broken |
| 5. TCP vs UDP | FAIL: Truncated | FAIL: "user is a" loop | FAIL: "user is a" loop | All broken |
| 6. ML Haiku | FAIL: Echoes prompt | FAIL: "/think" only | FAIL: Empty | All broken |
| 7. WWI Causes | FAIL: "1." repeated | FAIL: "the the the" loop | FAIL: Truncated | All broken |
| 8. Calculus | FAIL: "/think" only | FAIL: "input/output" loop | FAIL: "10,10,10" loop | All broken |
| 9. French translation | FAIL: "/think" only | FAIL: "/think" only | FAIL: "/think" only | All broken |
| 10. JSON output | FAIL: "/think" only | FAIL: "user is a" loop | FAIL: "/think" only | All broken |

## Observed Failure Modes

### 1. Repetition Loops
Most prompts produce degenerate repetition:
- "The user wants to answer the a." (K=8, prompt 1)
- "The user is a user." (K=8, prompt 2)
- "10-10-10-10-..." (K=6, prompt 2)
- "1.\n1.\n1.\n..." (K=8, prompt 7)

### 2. System Prompt Leakage
Model outputs fragments of system prompt:
- "You is a helpful assistant. /think" (multiple prompts)
- "The user is a helpful assistant." (multiple prompts)

### 3. Prompt Echo
Model echoes input instead of answering:
- "Write a haiku about machine learning." (K=8, prompt 6)
- "If all roses are flowers..." (K=6, prompt 4)

### 4. Nonsense Factual Answers
When not looping, answers are wrong:
- "The capital of Lebanon is a 1000." (K=6)
- "The capital is a capital." (K=4)

## Root Cause Analysis

The failures are **consistent across all K values**, indicating the issue is NOT expert selection count but likely:

1. **Weight extraction/packing issue** for 122B model
2. **Tokenizer mismatch** (vocab_122b.bin compatibility)
3. **Missing/corrupted layers** in model_weights.bin
4. **Chat template not applied** (explains system prompt leakage)

## Recommendations

1. **BLOCK quality gate** - model is not functional at any K value
2. Verify 122B weight extraction pipeline matches 35B (working) pipeline
3. Check if chat template is being applied in the server
4. Compare 122B manifest structure to 35B
5. Test with explicit chat format in the prompt

## Raw Output Samples

### Prompt 1: Capital of Lebanon

**K=8:**
```
The user wants to answer the a.
The user wants to answer the a.
The user wants to answer the a.
[repeats ~256 times]
```

**K=6:**
```
The capital of Lebanon is a 1000.
```

**K=4:**
```
The capital is a capital.
```

### Prompt 3: Palindrome Code

**K=8:**
```
You is a helpful assistant. /think
```

**K=6:**
```
The user is a helpful assistant. /think
```

**K=4:**
```
You at
```

### Prompt 7: WWI Causes

**K=8:**
```
The main causes of World War I were:
1.
1.
1.
[repeats]
```

**K=6:**
```
The main causes of World War I are the the the the the the the the the the the the the the the.
```

**K=4:**
```
The main causes of World War I are
```

## Conclusion

**QUALITY GATE: FAILED**

The 122B model is not producing coherent outputs at any K value. This must be investigated as an infrastructure/pipeline issue before K-value quality comparisons are meaningful.
