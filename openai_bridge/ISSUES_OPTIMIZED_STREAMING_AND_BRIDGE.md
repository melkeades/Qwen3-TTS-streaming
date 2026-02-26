# Issues Log: Optimized Streaming and Bridge

This file summarizes issues encountered while installing/running optimized streaming and implementing the OpenAI-compatible bridge.

## Scope

- Optimized model/runtime path (`qwen_tts` + torch compile/streaming optimizations)
- Bridge/API/UI path (`openai_bridge`)

## Optimized Streaming: Findings

### 1) Environment bootstrap: `uv venv` without `pip`

- Symptom:
  - `.venv-bench/bin/python: No module named pip`
- Impact:
  - Standard `python -m pip ...` flow was not usable immediately.
- Resolution:
  - Used `uv pip ... --python .venv-bench/bin/python` for all installs.
- Status:
  - Resolved; this is a setup nuance, not a blocker.

### 2) Non-streaming optimized mode instability (`max-autotune`)

- Symptom:
  - Runtime failures in compiled path during benchmark runs, including:
    - `Inplace update to inference tensor outside InferenceMode is not allowed...`
    - `Offset increment outside graph capture encountered unexpectedly.`
- Impact:
  - Non-streaming optimized benchmark path required fallback/retry logic.
- Notes:
  - This was observed in non-streaming compile mode experimentation.
  - Streaming optimized path (`reduce-overhead`) remained stable.
- Status:
  - Partially mitigated for benchmarking via fallback logic.

### 3) Streaming optimized path worked with expected compile overhead

- Symptom:
  - First warmup pass significantly slower due to compilation.
- Impact:
  - Higher one-time latency on first run.
- Expected behavior:
  - Subsequent runs were fast and stable, with strong RTF/TTFB improvements.
- Status:
  - Working as expected.

## Bridge: Findings

### 1) Port conflict on startup (`8000` occupied)

### 2) Browser CORS preflight failure from `origin 'null'`

- Symptom:
  - Browser error:
    - `blocked by CORS policy: Response to preflight request doesn't pass access control check: No 'Access-Control-Allow-Origin' header...`
- Cause:
  - HTML opened via `file://...` (origin becomes `null`) and server lacked CORS headers.
- Resolution:
  - Added `CORSMiddleware` to bridge with configurable defaults:
    - allow origins/methods/headers: `*`
    - allow credentials: `false`
  - Verified `OPTIONS /v1/audio/speech` returns CORS headers.
- Status:
  - Resolved.

### 3) StreamingResponse execution mode conflict with compiled CUDA graph path

- Symptom:
  - Client saw incomplete chunked reads (`RemoteProtocolError`).
  - Server trace showed CUDA capture/runtime errors while streaming via threaded iterator path.
- Impact:
  - Stream could fail mid-response.
- Resolution:
  - Switched `/v1/audio/speech` response iterator to async chunk-yielding path (non-threadpool iteration model).
- Status:
  - Resolved for tested PCM/WAV smoke flows.

## Dependency Outcome Summary

- Worked:
  - `torch==2.9.1+cu130`, `torchaudio==2.9.1+cu130`, flash-attn prebuilt wheel for torch 2.9/cu130.
  - Optimized streaming with `reduce-overhead`.
- Required adjustments:
  - Install workflow via `uv pip`.
  - Bridge runtime/serving fixes (port, CORS, streaming iterator behavior).
- Not required:
  - `vllm` was not used in bridge inference backbone.

## Current Recommended Defaults

- Bridge:
  - Port: `8030`
  - CORS enabled (permissive defaults for local dev)
- Streaming optimization:
  - `compile_mode=\"reduce-overhead\"`
  - `use_cuda_graphs=false` in bridge config defaults

