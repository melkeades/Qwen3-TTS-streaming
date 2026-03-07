# Qwen OpenAI Bridge

OpenAI-compatible streaming TTS bridge for this repo, modeled after the `orpheus_streaming_example` flow.

It serves:
- `POST /v1/audio/speech`
- `POST /v1/audio/stop`
- `GET /v1/models`
- `GET /v1/speakers` (custom bridge)
- `POST /v1/models/unload` (custom bridge)
- `GET /healthz`
- `GET /` (live HTML client)

## Requirements

- Python 3.12
- CUDA GPU recommended
- Dependencies available in `.venv-bench` (or your chosen env):
  - `torch`, `torchaudio`, `flash-attn`
  - `fastapi`, `uvicorn`, `httpx`
  - project installed with `pip install -e .`

## Quick Start

Run the bridge:

```bash
.venv-bench/bin/python -m openai_bridge.run_bridge
```

Run the CustomVoice bridge (separate server for `Qwen3-TTS-12Hz-1.7B-CustomVoice`):

```bash
.venv-bench/bin/python -m openai_bridge.run_custom_bridge
```

Start CustomVoice with no preloaded model (`Model=None` until selected):

```bash
.venv-bench/bin/python -m openai_bridge.run_custom_bridge --empty
```

Open the client:

- `http://localhost:8030/`
- `http://localhost:8040/` (when running `run_custom_bridge`, serves `openai_bridge/client_custom_live.html`)

Health check:

```bash
curl -sS http://localhost:8030/healthz
```

Custom bridge health includes startup readiness:

```bash
curl -sS http://localhost:8040/healthz
```

List models:

```bash
curl -sS http://localhost:8030/v1/models
```

## API

### `POST /v1/audio/speech`

Request body:

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "input": "Привет! Это тест.",
  "voice": "kuklina",
  "response_format": "pcm",
  "speed": 1.0,
  "language": "Russian",
  "emit_every_frames": 4,
  "decode_window_frames": 80,
  "overlap_samples": 0,
  "max_frames": 10000
}
```

Notes:
- OpenAI-compatible core fields: `model`, `input`, `voice`, `response_format`, `speed`, `instructions`.
- Optional bridge extensions are accepted: `speaker`, `instruct`, `language`, `emit_every_frames`, `decode_window_frames`, `overlap_samples`, `max_frames`.
- Response is streamed audio bytes (`audio/pcm` or `audio/wav`).
- Response headers include:
  - `X-Audio-Sample-Rate`
  - `X-Audio-Channels`
  - `X-Audio-Bits-Per-Sample`

CustomVoice bridge notes:
- `voice` is used as speaker id (OpenAI-compatible field).
- You can send OpenAI-style `instructions`; legacy `instruct` is still accepted.
- You can optionally send `use_optimized_decode` to override server default per request.
- `/healthz` now reports `startup_ready` so clients can wait until warmup/compile pass is done.
- `/v1/models` includes the default model and discovered local checkpoints (for example `output/checkpoint-epoch-2`).
- Custom bridge reuses already-loaded models from in-memory cache when switching back.
- Use `POST /v1/models/unload?model=<id>` (or `?all=true`) to free VRAM for cached model(s).
- `/v1/speakers?model=<id>` returns speaker ids for a selected model (including local `config.json` speaker IDs like `talker_config.spk_id`).
- The custom bridge loads the requested model via `Qwen3TTSModel.from_pretrained(...)`; switching model while streams are active returns `409`.
- Example payload for custom bridge:

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "input": "And now I am talking in a very angry tone.",
  "voice": "Ryan",
  "speaker": "Ryan",
  "instructions": "very angry, livid",
  "language": "English",
  "response_format": "pcm",
  "speed": 1.0
}
```

### `POST /v1/audio/stop`

Stops active stream(s).

Example response:

```json
{
  "stopped": true,
  "active_before": 1
}
```

## Voice Presets

Voice presets are loaded from:

- `openai_bridge/voices.json`

Current default preset:
- `kuklina` (uses `kuklina-1.wav`)

## Config (Environment Variables)

Server:
- `BRIDGE_HOST` (default `0.0.0.0`)
- `BRIDGE_PORT` (default `8030`)
- `BRIDGE_CORS_ALLOW_ORIGINS` (default `*`)
- `BRIDGE_CORS_ALLOW_METHODS` (default `*`)
- `BRIDGE_CORS_ALLOW_HEADERS` (default `*`)
- `BRIDGE_CORS_ALLOW_CREDENTIALS` (default `false`)

Model:
- `BRIDGE_MODEL_ID` (default `Qwen/Qwen3-TTS-12Hz-1.7B-Base`)
- `BRIDGE_DEVICE_MAP` (default `cuda:0`)
- `BRIDGE_DTYPE` (default `bfloat16`)
- `BRIDGE_ATTN_IMPL` (default `flash_attention_2`)
- `BRIDGE_DEFAULT_LANGUAGE` (default `Russian`)

Audio/streaming:
- `BRIDGE_SAMPLE_RATE` (default `24000`)
- `BRIDGE_CHANNELS` (default `1`)
- `BRIDGE_BITS_PER_SAMPLE` (default `16`)
- `BRIDGE_EMIT_EVERY_FRAMES` (default `4`)
- `BRIDGE_DECODE_WINDOW_FRAMES` (default `80`)
- `BRIDGE_OVERLAP_SAMPLES` (default `0`)
- `BRIDGE_MAX_FRAMES` (default `10000`)

Optimizations:
- `BRIDGE_OPT_USE_COMPILE` (default `true`)
- `BRIDGE_OPT_USE_CUDA_GRAPHS` (default `false`)
- `BRIDGE_OPT_COMPILE_MODE` (default `reduce-overhead`)
- `BRIDGE_OPT_USE_FAST_CODEBOOK` (default `true`)
- `BRIDGE_OPT_COMPILE_CODEBOOK_PREDICTOR` (default `true`)
- `BRIDGE_OPT_COMPILE_TALKER` (default `true`)

Warmup:
- `BRIDGE_WARMUP_ENABLED` (default `true`)
- `BRIDGE_WARMUP_TEXT` (default short Russian warmup text)

Paths:
- `BRIDGE_VOICES_PATH` (default `openai_bridge/voices.json`)
- `BRIDGE_CLIENT_HTML_PATH` (default `openai_bridge/client_dark_live.html`)
- `BRIDGE_LOGS_DIR` (default `custom_bridge/logs`)

## CustomVoice Bridge Config (Environment Variables)

Server:
- `CUSTOM_BRIDGE_HOST` (default `0.0.0.0`)
- `CUSTOM_BRIDGE_PORT` (default `8040`)
- `CUSTOM_BRIDGE_CORS_ALLOW_ORIGINS` (default `*`)
- `CUSTOM_BRIDGE_CORS_ALLOW_METHODS` (default `*`)
- `CUSTOM_BRIDGE_CORS_ALLOW_HEADERS` (default `*`)
- `CUSTOM_BRIDGE_CORS_ALLOW_CREDENTIALS` (default `false`)

Model:
- `CUSTOM_BRIDGE_MODEL_ID` (default `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)
- `CUSTOM_BRIDGE_START_EMPTY` (default `false`, skip startup preload and start with no active model)
- `CUSTOM_BRIDGE_MODEL_SCAN_ROOTS` (default `output`, comma-separated roots scanned for local checkpoints)
- `CUSTOM_BRIDGE_MODEL_SCAN_MAX_DEPTH` (default `3`)
- `CUSTOM_BRIDGE_ADDITIONAL_MODEL_IDS` (default empty, comma-separated)
- `CUSTOM_BRIDGE_DEVICE_MAP` (default `cuda:0`)
- `CUSTOM_BRIDGE_DTYPE` (default `bfloat16`)
- `CUSTOM_BRIDGE_ATTN_IMPL` (default `flash_attention_2`)
- `CUSTOM_BRIDGE_DEFAULT_LANGUAGE` (default `English`)
- `CUSTOM_BRIDGE_DEFAULT_SPEAKER` (default `Ryan`)
- `CUSTOM_BRIDGE_DEFAULT_INSTRUCT` (default empty)

Audio/streaming:
- `CUSTOM_BRIDGE_SAMPLE_RATE` (default `24000`)
- `CUSTOM_BRIDGE_CHANNELS` (default `1`)
- `CUSTOM_BRIDGE_BITS_PER_SAMPLE` (default `16`)
- `CUSTOM_BRIDGE_EMIT_EVERY_FRAMES` (default `4`)
- `CUSTOM_BRIDGE_DECODE_WINDOW_FRAMES` (default `80`)
- `CUSTOM_BRIDGE_OVERLAP_SAMPLES` (default `0`)
- `CUSTOM_BRIDGE_MAX_FRAMES` (default `10000`)

Optimizations:
- `CUSTOM_BRIDGE_OPT_USE_COMPILE` (default `true`)
- `CUSTOM_BRIDGE_OPT_USE_CUDA_GRAPHS` (default `false`)
- `CUSTOM_BRIDGE_OPT_COMPILE_MODE` (default `reduce-overhead`)
- `CUSTOM_BRIDGE_OPT_USE_FAST_CODEBOOK` (default `true`)
- `CUSTOM_BRIDGE_OPT_COMPILE_CODEBOOK_PREDICTOR` (default `true`)
- `CUSTOM_BRIDGE_OPT_COMPILE_TALKER` (default `true`)
- `CUSTOM_BRIDGE_STREAM_USE_OPTIMIZED_DECODE` (default `true`)

Warmup:
- `CUSTOM_BRIDGE_WARMUP_ENABLED` (default `true`)
- `CUSTOM_BRIDGE_WARMUP_RUNS` (default `3`)
- `CUSTOM_BRIDGE_WARMUP_MAX_FRAMES` (default `1024`)
- `CUSTOM_BRIDGE_WARMUP_TEXT` (default short English warmup text)
- `CUSTOM_BRIDGE_WARMUP_LANGUAGE` (default `English`)
- `CUSTOM_BRIDGE_WARMUP_SPEAKER` (default `Ryan`)
- `CUSTOM_BRIDGE_WARMUP_INSTRUCT` (default `neutral and clear`)

Paths:
- `CUSTOM_BRIDGE_CLIENT_HTML_PATH` (default `openai_bridge/client_custom_live.html`)
- `CUSTOM_BRIDGE_LOGS_DIR` (default `custom_bridge/logs`)

## Smoke Tests

With bridge running:

```bash
.venv-bench/bin/python openai_bridge/tests/test_pcm_stream_smoke.py --base-url http://localhost:8030
.venv-bench/bin/python openai_bridge/tests/test_wav_stream_smoke.py --base-url http://localhost:8030
.venv-bench/bin/python openai_bridge/tests/test_custom_pcm_stream_smoke.py --base-url http://localhost:8040
```

What they validate:
- PCM streaming returns bytes and latency metrics.
- WAV streaming returns a valid RIFF/WAVE header + payload.
- Stop path sanity (`/v1/audio/stop`) is checked by the PCM test.

## Troubleshooting

- Port already in use:
  - start with another port, e.g. `BRIDGE_PORT=8010`.
- Browser CORS / `origin 'null'` error:
  - this happens when opening HTML directly as `file://...` (origin becomes `null`).
  - current bridge enables permissive CORS by default (`*`), including preflight.
  - preferred usage is the hosted client at `http://localhost:8030/` instead of opening the file directly.
- Unknown voice:
  - verify `openai_bridge/voices.json` and `voice` value in request/UI.
- Slow first request:
  - expected due to compile/warmup; subsequent runs are faster.
- If streaming aborts unexpectedly:
  - keep `BRIDGE_OPT_COMPILE_MODE=reduce-overhead` and `BRIDGE_OPT_USE_CUDA_GRAPHS=false` (current defaults).
