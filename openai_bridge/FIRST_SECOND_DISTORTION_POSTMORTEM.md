# First-Second Distortion Postmortem (CustomVoice Streaming)

## Scope

This report covers the startup distortion heard in the first ~1 second of audio for CustomVoice streaming, especially with optimized streaming decode enabled.

Code paths involved:

- `qwen_tts/core/models/modeling_qwen3_tts.py`
- `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`
- `openai_bridge/custom_pipeline.py`
- `openai_bridge/custom_server.py`

## Symptom

- Distortion/noisy transient at stream start (first chunk(s)).
- Stronger with optimized decode path than baseline decode.
- More visible on CustomVoice path than base/ICL voice-clone path.

## What Went Wrong

1. Startup decode context mismatch (root cause)
- In the CustomVoice path, there is no ICL `ref_code` context at stream start.
- Early windows were decoded with less stable context than the ICL path.
- Result: startup frames were more fragile/noisy.

2. Early padded optimized decode amplified startup artifacts
- Optimized streaming decode used fixed-size padding (`pad_to_size=decode_window_frames`) in early windows.
- For the first emits, this left-padding introduces artificial history in a causal decoder.
- Result: first emitted chunk quality degraded.

3. Readiness vs true usability confusion
- Server process could report startup complete while model compile/warmup was still effectively in-flight for stable first response.
- This did not directly create distortion in model output, but it caused misleading “it’s ready” behavior during validation and increased perceived instability.

4. Non-root-cause adjustments
- UI/output-chain changes (for example gain removal in client) were not the main fix for this distortion class.
- Warmup alone improved consistency but did not eliminate startup artifact by itself.

## What Went Right

1. A/B isolation of optimized vs baseline decode behavior
- Distortion concentrated in startup + optimized decode combination.
- Baseline decode on early windows was cleaner.

2. Decoder-path fix at the right layer
- The final fix was applied in streaming generation/decoder context handling, not only in UI/client plumbing.

3. Startup observability improved
- Custom bridge now exposes readiness (`startup_ready`) and warmup metadata in `/healthz`, reducing false-positive “ready” states during test cycles.

## What Fixed the Issue

### Fix 1: Synthetic silence reference context for non-ICL startup

Implemented in:

- `qwen_tts/core/models/modeling_qwen3_tts.py:2734`

Details:

- When `ref_code_context` is absent (CustomVoice path), generate a short silence waveform.
- Encode that silence once into codec frames.
- Reuse as startup `ref_code_context` to stabilize initial decode windows.

Why it works:

- It gives the decoder a consistent prefix context similar in role to ICL ref-code context, reducing cold-start instability.

### Fix 2: Do not use padded optimized decode until window is full

Implemented in:

- `qwen_tts/core/models/modeling_qwen3_tts.py:2819`
- `qwen_tts/core/models/modeling_qwen3_tts.py:2823`

Details:

- If `window.shape[0] < decode_window_frames`, fallback to regular decode.
- Use `decode_streaming(..., pad_to_size=decode_window_frames)` only after full window length is reached.

Why it works:

- Avoids left-padding contamination in the most sensitive startup phase.

### Fix 3: Better startup gating/visibility

Implemented in:

- `openai_bridge/custom_pipeline.py:84`
- `openai_bridge/custom_pipeline.py:419`
- `openai_bridge/custom_server.py:153`

Details:

- Warmup pass and explicit `startup_ready` surfaced via `/healthz`.
- Helps ensure tests and clients wait for stable serving state.

## Remaining Risks / Edge Cases

- Different finetunes may have different startup sensitivity.
- Very small `emit_every_frames` can make startup artifacts easier to hear.
- Dynamic-shape CUDA graph churn can add timing jitter (mostly latency/perf, sometimes perceived as rough startup if buffers underrun).

## Additional Improvements That Might Help

1. Startup fade-in guard (low risk)
- Apply a short 10-30 ms fade-in on the very first emitted PCM chunk only.
- Masks residual click/transient without affecting steady-state quality.

2. Small non-zero overlap for first transitions (low-medium risk)
- Try `overlap_samples` in a small range (for example 64-256 samples) only for first 1-2 chunk boundaries.

3. Fixed startup emit policy (medium risk)
- Use a slightly larger first emit window, then return to normal `emit_every_frames`.
- Reduces sensitivity to under-conditioned first decode.

4. Per-model startup calibration (medium)
- Persist recommended startup params (`emit_every_frames`, `decode_window_frames`, overlap) per checkpoint.

5. Objective startup quality test (high value)
- Add automated A/B test that scores first N ms separately from steady-state to prevent regressions.

## Practical Conclusion

The issue was primarily decoder startup conditioning in CustomVoice streaming, worsened by early padded optimized decode.  
The effective fix was:

- startup context stabilization for non-ICL (`silence ref_code`), and
- delaying padded optimized decode until full window.

Warmup/readiness improvements were important supporting fixes for reliable validation and serving behavior.
