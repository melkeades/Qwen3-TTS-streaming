# Bridge Overview (Qwen3-TTS)

## What It Is

`openai_bridge` exposes OpenAI-style TTS HTTP APIs on top of this repo’s Qwen3-TTS streaming implementation.

There are two servers:

- Base bridge: voice-clone presets (`openai_bridge/server.py`)
- Custom bridge: CustomVoice speakers + multi-model checkpoint support (`openai_bridge/custom_server.py`)

## How To Run

From repo root:

```bash
python -m openai_bridge.run_bridge
python -m openai_bridge.run_custom_bridge
python -m openai_bridge.run_custom_bridge --empty
```

`--empty` starts the custom bridge without preloading a model, so the client starts with `Model=None` until you pick one.

Default ports:

- Base bridge: `http://localhost:8030`
- Custom bridge: `http://localhost:8040`

Built-in HTML clients:

- Base: `GET /` serves `openai_bridge/client_dark_live.html`
- Custom: `GET /` serves `openai_bridge/client_custom_live.html`

## What Is Available

Common:

- `POST /v1/audio/speech` (streaming PCM/WAV)
- `POST /v1/audio/stop`
- `GET /v1/models`
- `GET /healthz`

Custom bridge only:

- `GET /v1/speakers?model=...` (speaker IDs, incl. checkpoint `config.json` parsing)
- `POST /v1/models/unload?model=...` and `?all=true` (manual VRAM release)
- Multi-model discovery from `output/` + runtime model cache reuse
- Startup readiness metadata (`startup_ready`) in health output

Request compatibility notes:

- OpenAI-style `instructions` is supported.
- Legacy `instruct` is still accepted for backward compatibility.
- Custom bridge treats `voice` as speaker id (and also accepts `speaker` alias).

Validation scripts:

- `openai_bridge/tests/test_pcm_stream_smoke.py`
- `openai_bridge/tests/test_wav_stream_smoke.py`
- `openai_bridge/tests/test_custom_pcm_stream_smoke.py`

## What Is Not Implemented Yet

- AuthN/AuthZ: no API keys, no RBAC, no multi-tenant security boundary.
- Rate limiting / quotas / admission control.
- Persistent model cache across process restarts (cache is in-memory only).
- WebSocket/SSE transport (HTTP chunked streaming only).
- Production telemetry stack (structured metrics/tracing export).
- Full OpenAI parity beyond implemented TTS subset.

## Risks / Edge Cases

- GPU memory pressure when several large models are cached simultaneously.
- First request latency after cold start (compile + warmup).
- Active-stream constraints:
  - model switch/unload is blocked while streams are active.
- Distortion sensitivity can vary by finetune/checkpoint (startup conditioning).
- Dynamic-shape CUDA graph behavior may add perf jitter under varying window sizes.

## Points Of Improvement

1. Add auth + rate limiting for safe external exposure.
2. Add startup orchestration gate (serve traffic only after warmup completion by policy).
3. Add objective audio regression checks for first-second quality.
4. Add bounded cache policy (LRU / max VRAM budget).
5. Add Prometheus-style metrics and structured traces.
6. Add optional WebSocket transport for tighter interactive control.
