# API Usage (Style Continuity)

This guide explains how to use the CustomVoice bridge API for the best style continuity (speed, volume behavior, prosody shape) across streamed LLM text chunks.

## Use The Correct Path

For continuity across chunks, use:

- `WS /v1/audio/speech/live`

Do **not** send each chunk as a separate:

- `POST /v1/audio/speech`

`POST /v1/audio/speech` is still useful for one-shot synthesis, but chunk-by-chunk HTTP calls restart synthesis context and will drift more.

## Continuity Rules

For one assistant turn:

1. Open one websocket connection.
2. Send one `session.start`.
3. Send all LLM text deltas as `text.append` on that same socket.
4. Send `session.finish` when the turn is complete.
5. Wait for `session.done`.

Inside that one live session, keep these stable:

- `model`
- `voice`/`speaker`
- `instructions` (or `instruct`)
- `language`
- `speed`

Changing them mid-turn effectively asks for a different speaking style and hurts continuity.

## WebSocket Endpoint

- URL: `ws://127.0.0.1:8040/v1/audio/speech/live`

Server messages:

- `session.ready`
- `audio.delta`
- `session.draining`
- `session.done`
- `error`

Client messages:

- `session.start`
- `text.append`
- `session.finish`
- `session.cancel`

## Message Shapes

### `session.start` (client -> server)

```json
{
  "type": "session.start",
  "session_id": "turn-42",
  "model": "output/test",
  "voice": "p3",
  "speaker": "p3",
  "language": "English",
  "instructions": "very angry, livid",
  "response_format": "pcm",
  "speed": 1.0
}
```

Notes:

- `session_id` is optional; server generates one if omitted.
- `response_format` is `pcm` or `wav`.

### `text.append` (client -> server)

```json
{
  "type": "text.append",
  "session_id": "turn-42",
  "text": "First streamed text chunk."
}
```

Send as many `text.append` messages as needed, in order.

### `session.finish` (client -> server)

```json
{
  "type": "session.finish",
  "session_id": "turn-42"
}
```

This tells the backend no more text is coming and it should drain the session.

### `audio.delta` (server -> client)

```json
{
  "type": "audio.delta",
  "session_id": "turn-42",
  "format": "pcm",
  "sample_rate": 24000,
  "channels": 1,
  "bits_per_sample": 16,
  "data": "<base64 audio bytes>"
}
```

Decode `data` from base64 and append/play in arrival order.

## Minimal JS Client Flow

```javascript
const ws = new WebSocket("ws://127.0.0.1:8040/v1/audio/speech/live");
const sessionId = crypto.randomUUID();

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "session.start",
    session_id: sessionId,
    model: "output/test",
    voice: "p3",
    speaker: "p3",
    language: "English",
    instructions: "very angry, livid",
    response_format: "pcm",
    speed: 1.0
  }));
};

// Call this for each LLM delta/chunk:
function appendChunk(text) {
  ws.send(JSON.stringify({ type: "text.append", session_id: sessionId, text }));
}

function finishTurn() {
  ws.send(JSON.stringify({ type: "session.finish", session_id: sessionId }));
}

ws.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);
  if (msg.type === "audio.delta") {
    const bytes = Uint8Array.from(atob(msg.data), (c) => c.charCodeAt(0));
    // queue bytes for playback
  }
  if (msg.type === "session.done") {
    ws.close();
  }
};
```

## If You Still Use HTTP Buffering

`POST /v1/audio/speech` with `session_id` + `end_of_message` remains available for buffered HTTP workflows, but continuity is weaker than the live websocket session. Use it only if websocket transport is not possible.

## HTML Client Settings (Built-In UI)

In `openai_bridge/client_custom_live.html`:

- Turn on `Use backend live session...`
- Optional: turn on `Client-side segmentation...` (still sends all segments into the same backend live session)

For strongest continuity, backend live mode is the key requirement.

## Troubleshooting

- `session.start must be sent before other events`: send `session.start` first on each new websocket.
- `session_id does not match the active websocket session`: keep one `session_id` per websocket.
- `Unknown speaker ...`: call `GET /v1/speakers?model=...` and pick a valid speaker.
- No continuity between chunks: verify you are not opening a new websocket per chunk and not calling HTTP per chunk.

