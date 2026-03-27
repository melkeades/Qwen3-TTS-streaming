# OpenAI Bridge Agent Notes

This document is for coding agents and maintainers working inside `openai_bridge/`.

It explains:

- how the bridge is structured
- what each server variant actually does
- where request state lives
- how the backend text session workflow works in the CustomVoice bridge
- what is bridge-specific versus OpenAI-compatible

## Scope

There are two HTTP bridge variants in this repo:

- Base bridge: `openai_bridge/server.py`
- CustomVoice bridge: `openai_bridge/custom_server.py`

They share the same high-level goal: expose a local Qwen3-TTS model through OpenAI-style HTTP endpoints.

Important distinction:

- The base bridge is mostly stateless per request.
- The custom bridge supports multi-model loading, speaker discovery, backend text-session buffering, and a live websocket append-text session for CustomVoice.

## Files That Matter

Core server entrypoints:

- `openai_bridge/server.py`
- `openai_bridge/custom_server.py`

Pipeline layers:

- `openai_bridge/pipeline.py`
- `openai_bridge/custom_pipeline.py`

Configuration:

- `openai_bridge/config.py`
- `openai_bridge/custom_config.py`

Schema:

- `openai_bridge/schemas.py`

Operator/client docs:

- `openai_bridge/BRIDGE_OVERVIEW.md`

Browser clients:

- `openai_bridge/client_dark_live.html`
- `openai_bridge/client_custom_live.html`

## High-Level Architecture

The bridge has three layers:

1. FastAPI server layer
   - validates requests
   - exposes endpoints
   - tracks active streams
   - returns chunked PCM/WAV responses or websocket `audio.delta` events

2. Runtime layer
   - stores mutable server state
   - active stream cancellation events
   - for CustomVoice only: buffered text sessions

3. Pipeline layer
   - owns loaded `Qwen3TTSModel` instances
   - translates request params into model calls
   - converts float audio to PCM16 bytes
   - applies bridge defaults for decode window, emit cadence, overlap, warmup

The model-level streaming is real audio streaming, but the model APIs remain request-oriented. A new synthesis call resets model context unless the bridge itself adds a higher-level session mechanism.

## Endpoint Model

Common implemented endpoints:

- `GET /`
- `GET /healthz`
- `GET /v1/models`
- `POST /v1/audio/speech`
- `POST /v1/audio/stop`

CustomVoice-only endpoints:

- `GET /v1/speakers`
- `POST /v1/models/select`
- `POST /v1/models/unload`
- `POST /v1/audio/session/clear`
- `WS /v1/audio/speech/live`

## Base Bridge

### What it does

The base bridge wraps the Base Qwen3-TTS model with preset voice-clone prompts loaded from `voices.json`.

Flow:

1. `server.py` builds `BridgeRuntime`.
2. `pipeline.py` loads one model from `BridgeConfig.model_id`.
3. During load, it precomputes voice clone prompts for every configured voice preset.
4. `/v1/audio/speech` validates `req.model` and `req.voice`.
5. `pipeline.stream_audio_chunks()` calls `model.stream_generate_voice_clone(...)`.
6. Chunks are converted to PCM16 and streamed back to the client.

### State

Mutable runtime state in the base bridge is minimal:

- active stream cancellation events
- loaded model object
- prebuilt voice prompt cache

There is no backend text buffering in the base bridge.

## CustomVoice Bridge

### What it does

The custom bridge wraps CustomVoice checkpoints and supports:

- dynamic model discovery
- speaker discovery
- model caching in memory
- runtime model switching
- optional warmup
- backend text session buffering

Flow:

1. `custom_server.py` builds `CustomBridgeRuntime`.
2. `custom_pipeline.py` discovers available models and loads one on demand or at startup.
3. `/v1/audio/speech` may either:
   - buffer text into a backend session, or
   - immediately synthesize speech
4. `custom_pipeline.stream_audio_chunks()` calls `model.stream_generate_custom_voice(...)`.
5. Returned float chunks are converted to PCM16 and streamed.

### Model caching

CustomVoice keeps loaded models in memory inside `_model_cache`.

This cache is process-local only:

- it survives request boundaries
- it does not survive process restart

The cache is for model reuse, not speech continuation.

## Request Schema Notes

The bridge accepts normal speech fields such as:

- `model`
- `input`
- `voice`
- `response_format`
- `speed`

Bridge-specific optional fields already existed:

- `speaker`
- `instructions`
- `instruct`
- `language`
- `emit_every_frames`
- `decode_window_frames`
- `overlap_samples`
- `max_frames`
- `use_optimized_decode`

Custom session buffering added:

- `session_id`
- `end_of_message`

These are not part of the official OpenAI `POST /v1/audio/speech` schema. They are bridge extensions.

## Session Buffering: Why It Exists

Problem:

- LLM token streams or chunked text often arrive as many small fragments.
- If every fragment is sent as a separate TTS request, Qwen starts a fresh utterance each time.
- That causes reset in delivery characteristics like speaking pace, phrasing, emphasis, and prosodic continuity.

Constraint:

- local Qwen3-TTS in this repo does not expose a native cross-request continuation API
- internal KV cache is per synthesis call, not reusable across separate HTTP requests

Solution:

- the bridge buffers text on the backend and emits speech as soon as a segment is considered ready
- the first received chunk can be spoken immediately
- later chunks are usually emitted on strong sentence boundaries
- `end_of_message=true` flushes any trailing text that has not yet formed a ready segment

This does not create a true native model session, but it lets the backend drive cross-request continuation with cached acoustic tail context instead of leaving segmentation to the client.

## Session Buffering: What Is Implemented

Implemented in:

- `openai_bridge/custom_server.py`
- `openai_bridge/schemas.py`

Main structures:

- `BufferedSpeechSession`
- `CustomBridgeRuntime._buffered_sessions`

Each buffered session stores:

- `session_id`
- accumulated text chunks
- synthesis settings that must remain stable across the session

Those settings include:

- `model`
- effective speaker identity
- effective instruction text
- `language`
- `response_format`
- `speed`
- decode-related bridge params

## Session Buffering: Request Lifecycle

### Intermediate chunk

Caller sends:

- `session_id=<stable id>`
- `end_of_message=false`
- `input=<partial text>`

Server behavior:

1. Validate or create the buffered session.
2. Check that synthesis settings match the already buffered session.
3. Append the text chunk.
4. Plan any ready segment(s):
   - first chunk can be emitted immediately
   - follow-up chunks are emitted on strong sentence endings when possible
5. If no segment is ready yet, return `202 Accepted` with JSON metadata.
6. If segment(s) are ready, stream audio immediately for those segment(s) and keep the session open.

### Final chunk

Caller sends:

- same `session_id`
- `end_of_message=true`
- final text fragment in `input`

Server behavior:

1. Append the final text fragment.
2. Plan any final ready segment(s).
3. Flush trailing unsent text even if it does not end with sentence punctuation.
4. Stream those final segment(s).
5. Clear the buffered session and its cached continuation state after completion.

### Manual clear

Endpoint:

- `POST /v1/audio/session/clear?session_id=...`

Use this when a buffered session should be dropped without synthesis.

Typical cases:

- user cancelled before final chunk
- upstream LLM stream failed
- caller wants to abandon the turn

## Session Buffering: Invariants

For a given `session_id`, the following must not change across chunks:

- model
- speaker / voice identity
- instruction text
- language
- response format
- speed
- streaming/decode parameters

If any of those change, the bridge returns a conflict error instead of silently mixing incompatible state.

This is deliberate. Silent mixing would make debugging output quality much harder.

## Speaker and Instruction Normalization

The custom bridge normalizes aliases while buffering:

- speaker identity is derived from `speaker` or `voice`
- instruction identity is derived from `instructions` or legacy `instruct`

Reason:

- callers may send the OpenAI-style field on one chunk and the bridge alias on another
- those should be treated as the same effective synthesis config

## What Session Buffering Is Not

It is not:

- native model continuation
- reusable Qwen KV-cache across requests
- official OpenAI `/v1/audio/speech` behavior
- mid-utterance audio continuation while text is still being appended

It is a backend aggregation layer that delays synthesis until a message boundary is known.

## Streaming Semantics

Two separate meanings of "streaming" exist here:

1. Text streaming from upstream LLM
   - many partial text fragments arrive over time

2. Audio streaming from TTS
   - once synthesis starts, PCM/WAV bytes are emitted progressively

The bridge session workflow only solves the first problem by buffering text until the message is complete.

The pipeline still uses model-level audio streaming during the final synthesis call.

## Client Behavior

The custom HTML client supports the backend session workflow.

Relevant file:

- `openai_bridge/client_custom_live.html`

Behavior:

- optional backend buffering toggle
- stable `session_id` field
- intermediate segmented requests use `session_id` with `end_of_message=false`
- final segmented request uses `end_of_message=true`
- failed or cancelled runs clear the buffered session

This client behavior is bridge-specific and should not be described as official OpenAI parity.

## Cancellation Behavior

`POST /v1/audio/stop` only cancels active audio generation streams.

It does not automatically mean "drop every buffered text session forever" as a general policy.

That is why `POST /v1/audio/session/clear` exists separately.

Current custom HTML clears the active buffered session on failed/cancelled runs to avoid stale text buildup.

## Health and Debugging

Useful state surfaced by `GET /healthz` in the custom bridge:

- active model id
- default model id
- discovered models
- cached models
- startup readiness
- speaker list
- active stream count
- buffered session count

For debugging session issues, inspect:

- the `session_id` used by the caller
- whether a given chunk produced `202` or immediate audio
- whether the final chunk used `end_of_message=true`
- whether synthesis settings drifted across chunks

## Design Limits

Current limitations of the implemented session workflow:

- buffering is process-local and in-memory only
- no persistence across restart
- no TTL cleanup job yet
- no automatic inactivity timeout yet
- continuation is still approximate because Qwen does not expose true native text-append synthesis
- only implemented in the CustomVoice bridge

## Safe Modification Guidance

If changing session behavior, preserve these properties unless intentionally redesigning:

1. The first chunk must be allowed to start speaking without waiting for `end_of_message`.
2. Follow-up chunks should only advance on backend-selected segment boundaries, not arbitrary client cuts.
3. Session settings must remain consistent across buffered chunks.
4. Failed or abandoned sessions must be clearable without affecting active model cache.
5. Session buffering must remain clearly marked as a bridge extension, not OpenAI parity.

## Recommended Future Extensions

Reasonable next steps if needed:

- add session TTL expiration
- add explicit `flush` and `close_session` flags instead of only `end_of_message`
- add buffered-session listing for debugging
- add the same session workflow to the base bridge if required
- split a future stateful API from `/v1/audio/speech` if stricter OpenAI compatibility becomes important
