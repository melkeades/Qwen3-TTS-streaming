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
- `WS /v1/audio/speech/live` (CustomVoice live append-text session)
- `POST /v1/audio/stop`
- `GET /v1/models`
- `GET /healthz`

Custom bridge only:

- `GET /v1/speakers?model=...` (speaker IDs, incl. checkpoint `config.json` parsing)
- `POST /v1/models/unload?model=...` and `?all=true` (manual VRAM release)
- Backend text session buffering via `session_id` + `end_of_message`
- `POST /v1/audio/session/clear?session_id=...` (drop buffered text without synthesis)
- Multi-model discovery from `output/` + runtime model cache reuse
- Startup readiness metadata (`startup_ready`) in health output

Request compatibility notes:

- OpenAI-style `instructions` is supported.
- Legacy `instruct` is still accepted for backward compatibility.
- Custom bridge treats `voice` as speaker id (and also accepts `speaker` alias).
- Custom bridge buffered session mode:
  - Send `session_id` with each text chunk to buffer on the backend.
  - The first chunk is speakable immediately once received; later chunks are emitted when the backend has a ready segment.
  - Follow-up chunks are cut on strong sentence endings (`.`, `!`, `?`) during dynamic backend chunking.
  - If a request does not yet yield a speakable segment, server returns `202 Accepted`.
  - `end_of_message=true` flushes any trailing unsent text in the session.
  - Backend continuation is session-owned and uses cached acoustic tail state to improve cross-request continuity.
  - Buffered chunks in the same session must keep the same synthesis settings (`model`, `voice`/`speaker`, language, instructions, format, decode params).
- Custom bridge live websocket mode:
  - `session.start` configures one backend-owned CustomVoice session.
  - `text.append` extends the active text frontier without restarting Qwen generation.
  - `session.finish` appends the final EOS-style text conditioning and lets the live session drain normally.
  - `session.cancel` aborts the live session.
  - Server emits `session.ready`, `audio.delta`, `session.draining`, `session.done`, and `error`.

Validation scripts:

- `openai_bridge/tests/test_pcm_stream_smoke.py`
- `openai_bridge/tests/test_wav_stream_smoke.py`
- `openai_bridge/tests/test_custom_pcm_stream_smoke.py`

## What Is Not Implemented Yet

- AuthN/AuthZ: no API keys, no RBAC, no multi-tenant security boundary.
- Rate limiting / quotas / admission control.
- Persistent model cache across process restarts (cache is in-memory only).
- SSE transport.
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
6. Add browser/client support around the live websocket session API.
