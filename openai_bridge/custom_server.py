from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from threading import Event, Lock
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .custom_config import CustomBridgeConfig
from .custom_pipeline import QwenCustomStreamingPipeline
from .schemas import (
    BufferedSessionResponse,
    ClearSessionResponse,
    ErrorObject,
    ModelListResponse,
    ModelObject,
    SpeechResponseError,
    SpeechSynthesisParams,
    StopResponse,
)

logger = logging.getLogger(__name__)
_SENTENCE_RE = re.compile(r".*?[.!?]+(?=\s|$)", re.DOTALL)


def _effective_speaker(req: SpeechSynthesisParams) -> str:
    return (req.speaker or req.voice).strip()


def _effective_instructions(req: SpeechSynthesisParams) -> str | None:
    value = req.instructions if req.instructions is not None else req.instruct
    if value is None:
        return None
    return value


def _extract_ready_sentences(text: str) -> tuple[list[str], str]:
    raw = text or ""
    ready: list[str] = []
    last_end = 0
    for match in _SENTENCE_RE.finditer(raw):
        sentence = match.group(0).strip()
        if sentence:
            ready.append(sentence)
        last_end = match.end()
    return ready, raw[last_end:]


@dataclass
class BufferedSpeechSession:
    session_id: str
    chunks: list[str]
    pending_text: str
    generated_segments: int
    model: str
    voice: str
    speaker: str | None
    instructions: str | None
    instruct: str | None
    language: str | None
    response_format: str
    speed: float
    emit_every_frames: int | None
    decode_window_frames: int | None
    overlap_samples: int | None
    max_frames: int | None
    use_optimized_decode: bool | None

    @property
    def buffered_chunks(self) -> int:
        return len(self.chunks)

    @property
    def buffered_chars(self) -> int:
        return len(self.pending_text)

    @property
    def total_received_chars(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)

    def append(self, text: str) -> None:
        self.chunks.append(text)
        self.pending_text += text

    def build_request(
        self,
        req: SpeechSynthesisParams,
        *,
        text: str,
        response_format: str | None = None,
        continuation_reset: bool = False,
    ) -> SpeechSynthesisParams:
        return req.model_copy(
            update={
                "input": text,
                "model": self.model,
                "voice": self.voice,
                "speaker": self.speaker,
                "instructions": self.instructions,
                "instruct": self.instruct,
                "language": self.language,
                "response_format": response_format or self.response_format,
                "speed": self.speed,
                "emit_every_frames": self.emit_every_frames,
                "decode_window_frames": self.decode_window_frames,
                "overlap_samples": self.overlap_samples,
                "max_frames": self.max_frames,
                "use_optimized_decode": self.use_optimized_decode,
                "continuation_id": self.session_id,
                "continuation_mode": "acoustic_tail",
                "continuation_reset": continuation_reset,
                "session_id": None,
                "end_of_message": False,
            }
        )


@dataclass
class CustomBridgeRuntime:
    config: CustomBridgeConfig
    pipeline: QwenCustomStreamingPipeline
    _active: dict[str, Event]
    _active_cancellers: dict[str, Any]
    _buffered_sessions: dict[str, BufferedSpeechSession]
    _lock: Lock

    def register_stream(self, canceler: Any = None) -> tuple[str, Event]:
        stream_id = str(uuid.uuid4())
        ev = Event()
        with self._lock:
            self._active[stream_id] = ev
            if canceler is not None:
                self._active_cancellers[stream_id] = canceler
        return stream_id, ev

    def unregister_stream(self, stream_id: str) -> None:
        with self._lock:
            self._active.pop(stream_id, None)
            self._active_cancellers.pop(stream_id, None)

    def cancel_all(self) -> int:
        with self._lock:
            events = list(self._active.values())
            cancellers = list(self._active_cancellers.values())
        for ev in events:
            ev.set()
        for canceler in cancellers:
            try:
                canceler()
            except Exception:
                logger.debug("active stream canceler failed", exc_info=True)
        return len(events)

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def buffered_session_count(self) -> int:
        with self._lock:
            return len(self._buffered_sessions)

    def upsert_buffered_session(self, req: SpeechSynthesisParams) -> BufferedSpeechSession:
        session_id = (req.session_id or "").strip()
        if not session_id:
            raise ValueError("Missing session_id")

        with self._lock:
            session = self._buffered_sessions.get(session_id)
            if session is None:
                session = BufferedSpeechSession(
                    session_id=session_id,
                    chunks=[],
                    pending_text="",
                    generated_segments=0,
                    model=req.model,
                    voice=_effective_speaker(req),
                    speaker=_effective_speaker(req),
                    instructions=_effective_instructions(req),
                    instruct=_effective_instructions(req),
                    language=req.language,
                    response_format=req.response_format,
                    speed=req.speed,
                    emit_every_frames=req.emit_every_frames,
                    decode_window_frames=req.decode_window_frames,
                    overlap_samples=req.overlap_samples,
                    max_frames=req.max_frames,
                    use_optimized_decode=req.use_optimized_decode,
                )
                self._buffered_sessions[session_id] = session
            else:
                self._validate_buffered_session_locked(session, req)

            session.append(req.input)
            return BufferedSpeechSession(
                session_id=session.session_id,
                chunks=list(session.chunks),
                pending_text=session.pending_text,
                generated_segments=session.generated_segments,
                model=session.model,
                voice=session.voice,
                speaker=session.speaker,
                instructions=session.instructions,
                instruct=session.instruct,
                language=session.language,
                response_format=session.response_format,
                speed=session.speed,
                emit_every_frames=session.emit_every_frames,
                decode_window_frames=session.decode_window_frames,
                overlap_samples=session.overlap_samples,
                max_frames=session.max_frames,
                use_optimized_decode=session.use_optimized_decode,
            )

    def plan_session_segments(
        self,
        session_id: str,
        *,
        end_of_message: bool,
    ) -> tuple[BufferedSpeechSession | None, list[str], bool]:
        key = (session_id or "").strip()
        if not key:
            return None, [], False

        with self._lock:
            session = self._buffered_sessions.get(key)
            if session is None:
                return None, [], False

            segments: list[str] = []
            if session.generated_segments == 0:
                first = session.pending_text.strip()
                if first:
                    segments.append(first)
                    session.pending_text = ""
            else:
                ready, remaining = _extract_ready_sentences(session.pending_text)
                segments.extend(ready)
                session.pending_text = remaining

            if end_of_message:
                tail = session.pending_text.strip()
                if tail:
                    segments.append(tail)
                session.pending_text = ""

            close_after_stream = bool(end_of_message and not session.pending_text.strip())

            return (
                BufferedSpeechSession(
                    session_id=session.session_id,
                    chunks=list(session.chunks),
                    pending_text=session.pending_text,
                    generated_segments=session.generated_segments,
                    model=session.model,
                    voice=session.voice,
                    speaker=session.speaker,
                    instructions=session.instructions,
                    instruct=session.instruct,
                    language=session.language,
                    response_format=session.response_format,
                    speed=session.speed,
                    emit_every_frames=session.emit_every_frames,
                    decode_window_frames=session.decode_window_frames,
                    overlap_samples=session.overlap_samples,
                    max_frames=session.max_frames,
                    use_optimized_decode=session.use_optimized_decode,
                ),
                segments,
                close_after_stream,
            )

    def mark_session_segments_generated(self, session_id: str, count: int) -> None:
        key = (session_id or "").strip()
        if not key or count <= 0:
            return
        with self._lock:
            session = self._buffered_sessions.get(key)
            if session is None:
                return
            session.generated_segments += int(count)

    def has_buffered_session(self, session_id: str) -> bool:
        key = (session_id or "").strip()
        if not key:
            return False
        with self._lock:
            return key in self._buffered_sessions

    def pop_buffered_session(self, session_id: str) -> BufferedSpeechSession | None:
        key = (session_id or "").strip()
        if not key:
            return None
        with self._lock:
            return self._buffered_sessions.pop(key, None)

    def clear_buffered_session(self, session_id: str) -> ClearSessionResponse:
        session = self.pop_buffered_session(session_id)
        if session is None:
            return ClearSessionResponse(cleared=False, session_id=session_id)
        return ClearSessionResponse(
            cleared=True,
            session_id=session_id,
            dropped_chunks=session.buffered_chunks,
            dropped_chars=session.buffered_chars,
        )

    @staticmethod
    def _validate_buffered_session_locked(
        session: BufferedSpeechSession, req: SpeechSynthesisParams
    ) -> None:
        mismatches: list[str] = []

        def check(name: str, existing: Any, incoming: Any) -> None:
            if existing != incoming:
                mismatches.append(name)

        check("model", session.model, req.model)
        check("voice", session.voice, _effective_speaker(req))
        check("speaker", session.speaker, _effective_speaker(req))
        check("instructions", session.instructions, _effective_instructions(req))
        check("instruct", session.instruct, _effective_instructions(req))
        check("language", session.language, req.language)
        check("response_format", session.response_format, req.response_format)
        check("speed", session.speed, req.speed)
        check("emit_every_frames", session.emit_every_frames, req.emit_every_frames)
        check("decode_window_frames", session.decode_window_frames, req.decode_window_frames)
        check("overlap_samples", session.overlap_samples, req.overlap_samples)
        check("max_frames", session.max_frames, req.max_frames)
        check("use_optimized_decode", session.use_optimized_decode, req.use_optimized_decode)

        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(
                f"Buffered session '{session.session_id}' received conflicting synthesis fields: {joined}"
            )


def _error_response(
    status_code: int,
    message: str,
    *,
    type_: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    payload = SpeechResponseError(
        error=ErrorObject(message=message, type=type_, param=param, code=code)
    ).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


def _format_available_speakers(speakers: list[str], limit: int = 24) -> str:
    if not speakers:
        return "(model does not expose a speaker list)"
    if len(speakers) <= limit:
        return ", ".join(speakers)
    shown = ", ".join(speakers[:limit])
    remaining = len(speakers) - limit
    return f"{shown}, ... (+{remaining} more)"


def _live_req_from_payload(payload: dict[str, Any]) -> SpeechSynthesisParams:
    voice = str(payload.get("voice") or payload.get("speaker") or "").strip()
    return SpeechSynthesisParams(
        model=str(payload.get("model") or "").strip(),
        input="live-session",
        voice=voice,
        speaker=str(payload.get("speaker") or voice or "").strip() or None,
        instructions=payload.get("instructions"),
        instruct=payload.get("instruct"),
        language=payload.get("language"),
        response_format=str(payload.get("response_format") or "pcm").strip(),
        speed=float(payload.get("speed") or 1.0),
        emit_every_frames=payload.get("emit_every_frames"),
        decode_window_frames=payload.get("decode_window_frames"),
        overlap_samples=payload.get("overlap_samples"),
        max_frames=payload.get("max_frames"),
        use_optimized_decode=payload.get("use_optimized_decode"),
    )


def create_app() -> FastAPI:
    config = CustomBridgeConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_config = config
        pipeline = QwenCustomStreamingPipeline(config=active_config)
        should_preload = (not active_config.startup_empty) and bool((active_config.model_id or "").strip())
        logger.info(
            "Custom bridge startup begin model=%s preload=%s compile=%s optimized_decode=%s warmup=%s runs=%s",
            active_config.model_id,
            should_preload,
            active_config.optimize_use_compile,
            active_config.stream_use_optimized_decode,
            active_config.warmup_enabled,
            active_config.warmup_runs,
        )
        if should_preload:
            try:
                pipeline.load()
            except Exception:
                primary_model_id = (active_config.model_id or "").strip()
                fallback_model_id = (active_config.fallback_model_id or "").strip()
                fallback_speaker = (active_config.fallback_speaker or "").strip()
                if not fallback_model_id or fallback_model_id == primary_model_id:
                    logger.exception(
                        "Custom bridge preload failed for model=%s and no distinct fallback model is configured",
                        primary_model_id,
                    )
                    raise

                logger.warning(
                    "Custom bridge preload failed for model=%s; retrying with fallback model=%s",
                    primary_model_id,
                    fallback_model_id,
                    exc_info=True,
                )
                active_config = replace(
                    active_config,
                    model_id=fallback_model_id,
                    default_speaker=fallback_speaker or active_config.default_speaker,
                    warmup_speaker=fallback_speaker or active_config.warmup_speaker,
                )
                pipeline = QwenCustomStreamingPipeline(config=active_config)
                pipeline.load()
                logger.info(
                    "Custom bridge fallback model loaded model=%s speaker=%s",
                    active_config.model_id,
                    active_config.default_speaker,
                )

        app.state.runtime = CustomBridgeRuntime(
            config=active_config,
            pipeline=pipeline,
            _active={},
            _active_cancellers={},
            _buffered_sessions={},
            _lock=Lock(),
        )
        logger.info(
            "Custom bridge ready active_model=%s default_model=%s speaker_count=%s",
            pipeline.active_model_id,
            active_config.model_id,
            len(pipeline.speaker_names()),
        )
        try:
            yield
        finally:
            app.state.runtime.cancel_all()
            pipeline.unload_all_models()

    app = FastAPI(title="Qwen OpenAI Bridge (CustomVoice)", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_allow_origins,
        allow_methods=config.cors_allow_methods,
        allow_headers=config.cors_allow_headers,
        allow_credentials=config.cors_allow_credentials,
    )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return _error_response(exc.status_code, detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(_request, exc: RequestValidationError):
        return _error_response(422, str(exc), type_="validation_error")

    @app.exception_handler(Exception)
    async def _generic_exception_handler(_request, exc: Exception):
        logger.exception("Unhandled custom bridge exception")
        return _error_response(500, str(exc), type_="server_error")

    @app.get("/")
    async def index() -> FileResponse:
        runtime: CustomBridgeRuntime = app.state.runtime
        return FileResponse(runtime.config.client_html_path)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        runtime: CustomBridgeRuntime = app.state.runtime
        models = runtime.pipeline.discover_models(refresh=True)
        speakers = runtime.pipeline.speaker_names()
        cached_models = runtime.pipeline.cached_model_ids()
        return {
            "ok": bool(runtime.pipeline.startup_ready or runtime.config.startup_empty),
            "startup_ready": runtime.pipeline.startup_ready,
            "startup_empty": runtime.config.startup_empty,
            "model_id": runtime.pipeline.active_model_id,
            "default_model_id": runtime.config.model_id or None,
            "available_models": models,
            "cached_models": cached_models,
            "cached_count": len(cached_models),
            "default_speaker": runtime.config.default_speaker,
            "default_language": runtime.config.default_language,
            "stream_use_optimized_decode": runtime.config.stream_use_optimized_decode,
            "stream_greedy_decoding": runtime.config.stream_greedy_decoding,
            "continuation_default_frames": runtime.config.continuation_default_frames,
            "continuation_cache_ttl_sec": runtime.config.continuation_cache_ttl_sec,
            "continuation_cache_max_entries": runtime.config.continuation_cache_max_entries,
            "continuation_sampling_enabled": runtime.config.continuation_sampling_enabled,
            "continuation_sampling_temperature": runtime.config.continuation_sampling_temperature,
            "continuation_sampling_top_k": runtime.config.continuation_sampling_top_k,
            "continuation_sampling_subtalker_temperature": runtime.config.continuation_sampling_subtalker_temperature,
            "continuation_sampling_subtalker_top_k": runtime.config.continuation_sampling_subtalker_top_k,
            "continuation_followup_greedy_enabled": runtime.config.continuation_followup_greedy_enabled,
            "warmup_enabled": runtime.config.warmup_enabled,
            "warmup_runs": runtime.config.warmup_runs,
            "speaker_count": len(speakers),
            "speakers": speakers,
            "active_streams": runtime.active_count(),
            "buffered_sessions": runtime.buffered_session_count(),
        }

    @app.get("/v1/models", response_model=ModelListResponse)
    async def v1_models() -> ModelListResponse:
        runtime: CustomBridgeRuntime = app.state.runtime
        model_ids = runtime.pipeline.discover_models(refresh=True)
        return ModelListResponse(
            data=[
                ModelObject(
                    id=model_id,
                    created=int(time.time()),
                    owned_by="qwen-local",
                )
                for model_id in model_ids
            ]
        )

    @app.post("/v1/models/unload")
    async def v1_models_unload(model: str | None = None, all: bool = False) -> dict[str, Any]:
        runtime: CustomBridgeRuntime = app.state.runtime
        if runtime.active_count() > 0:
            raise HTTPException(
                status_code=409,
                detail="Cannot unload model(s) while streams are active. Stop stream(s) first.",
            )

        if all:
            unloaded = runtime.pipeline.unload_all_models()
            return {
                "ok": True,
                "unloaded": unloaded,
                "active_model_id": runtime.pipeline.active_model_id,
                "cached_models": runtime.pipeline.cached_model_ids(),
            }

        requested_model = (model or runtime.pipeline.active_model_id or "").strip()
        if not requested_model:
            raise HTTPException(
                status_code=400,
                detail="No model specified. Provide ?model=<id> or use ?all=true.",
            )
        unloaded = runtime.pipeline.unload_model(requested_model)
        if not unloaded:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{requested_model}' is not currently loaded in cache.",
            )
        return {
            "ok": True,
            "model": requested_model,
            "unloaded": 1,
            "active_model_id": runtime.pipeline.active_model_id,
            "cached_models": runtime.pipeline.cached_model_ids(),
        }

    @app.post("/v1/models/select")
    async def v1_models_select(model: str | None = None) -> dict[str, Any]:
        runtime: CustomBridgeRuntime = app.state.runtime
        requested_model = (model or "").strip()
        if not requested_model:
            raise HTTPException(status_code=400, detail="Missing model id (?model=<id>).")

        active_model = runtime.pipeline.active_model_id
        if active_model and requested_model != active_model and runtime.active_count() > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Cannot switch model from '{active_model}' to '{requested_model}' "
                    "while streams are active. Stop current stream(s) first."
                ),
            )

        try:
            runtime.pipeline.ensure_model_loaded(requested_model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "ok": True,
            "model": runtime.pipeline.active_model_id or requested_model,
            "cached_models": runtime.pipeline.cached_model_ids(),
            "speaker_count": len(runtime.pipeline.speaker_names()),
        }

    @app.get("/v1/speakers")
    async def v1_speakers(model: str | None = None) -> dict[str, Any]:
        runtime: CustomBridgeRuntime = app.state.runtime
        requested_model = (
            (model or runtime.pipeline.active_model_id or runtime.config.model_id).strip()
        )
        try:
            speakers = runtime.pipeline.speaker_names_for_model(requested_model, refresh=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "object": "list",
            "model": requested_model,
            "speakers": speakers,
            "count": len(speakers),
        }

    @app.post("/v1/audio/stop", response_model=StopResponse)
    async def v1_audio_stop() -> StopResponse:
        runtime: CustomBridgeRuntime = app.state.runtime
        active_before = runtime.cancel_all()
        return StopResponse(stopped=True, active_before=active_before)

    @app.post("/v1/audio/session/clear", response_model=ClearSessionResponse)
    async def v1_audio_session_clear(session_id: str | None = None) -> ClearSessionResponse:
        requested_session = (session_id or "").strip()
        if not requested_session:
            raise HTTPException(status_code=400, detail="Missing session id (?session_id=<id>).")
        runtime: CustomBridgeRuntime = app.state.runtime
        cleared = runtime.clear_buffered_session(requested_session)
        runtime.pipeline.clear_continuation_session(requested_session)
        return cleared

    @app.websocket("/v1/audio/speech/live")
    async def v1_audio_speech_live(websocket: WebSocket) -> None:
        await websocket.accept()
        runtime: CustomBridgeRuntime = app.state.runtime
        outbound: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        session_id = ""
        session_req: SpeechSynthesisParams | None = None
        speaker = ""
        live_session = None
        stream_id: str | None = None
        cancel_event: Event | None = None
        response_format = "pcm"
        pump_event = asyncio.Event()

        def cancel_current() -> None:
            if cancel_event is not None:
                cancel_event.set()
            if live_session is not None:
                try:
                    live_session.cancel()
                except Exception:
                    logger.debug("live session cancel failed", exc_info=True)

        async def sender() -> None:
            while True:
                message = await outbound.get()
                if message is None:
                    return
                await websocket.send_json(message)

        async def pump() -> None:
            while True:
                await pump_event.wait()
                pump_event.clear()
                if live_session is None:
                    continue
                try:
                    while True:
                        emitted_any = False
                        for chunk in live_session.stream_bytes():
                            emitted_any = True
                            await outbound.put(
                                {
                                    "type": "audio.delta",
                                    "session_id": session_id,
                                    "format": response_format,
                                    "sample_rate": runtime.config.sample_rate,
                                    "channels": runtime.config.channels,
                                    "bits_per_sample": runtime.config.bits_per_sample,
                                    "data": base64.b64encode(chunk).decode("ascii"),
                                }
                            )
                            await asyncio.sleep(0)
                        if live_session.is_closed():
                            await outbound.put(
                                {
                                    "type": "session.done",
                                    "session_id": session_id,
                                    "cancelled": bool(cancel_event.is_set()) if cancel_event is not None else False,
                                }
                            )
                            await outbound.put(None)
                            return
                        if not emitted_any:
                            break
                except Exception as exc:
                    logger.exception("live speech pump failed")
                    await outbound.put(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "message": str(exc),
                        }
                    )
                    await outbound.put(None)
                    return

        async def receiver() -> None:
            nonlocal session_id, session_req, speaker, live_session
            nonlocal stream_id, cancel_event, response_format

            while True:
                payload = await websocket.receive_json()
                message_type = str(payload.get("type") or "").strip()
                requested_session_id = str(payload.get("session_id") or "").strip()

                if message_type == "session.start":
                    if session_req is not None:
                        await outbound.put(
                            {
                                "type": "error",
                                "session_id": session_id,
                                "message": "session.start already received for this websocket.",
                            }
                        )
                        continue
                    session_id = requested_session_id or str(uuid.uuid4())
                    try:
                        req = _live_req_from_payload(payload)
                    except Exception as exc:
                        await outbound.put(
                            {
                                "type": "error",
                                "session_id": session_id,
                                "message": str(exc),
                            }
                        )
                        continue

                    req_model = req.model.strip()
                    active_model = runtime.pipeline.active_model_id
                    if active_model and req_model != active_model and runtime.active_count() > 0:
                        await outbound.put(
                            {
                                "type": "error",
                                "session_id": session_id,
                                "message": (
                                    f"Cannot switch model from '{active_model}' to '{req_model}' "
                                    "while streams are active."
                                ),
                            }
                        )
                        continue
                    try:
                        runtime.pipeline.ensure_model_loaded(req_model)
                    except ValueError as exc:
                        await outbound.put(
                            {
                                "type": "error",
                                "session_id": session_id,
                                "message": str(exc),
                            }
                        )
                        continue

                    speaker = (req.speaker or req.voice or runtime.config.default_speaker).strip()
                    if not speaker:
                        await outbound.put(
                            {
                                "type": "error",
                                "session_id": session_id,
                                "message": "Missing speaker. Provide 'voice' or 'speaker'.",
                            }
                        )
                        continue
                    if not runtime.pipeline.has_speaker(speaker):
                        await outbound.put(
                            {
                                "type": "error",
                                "session_id": session_id,
                                "message": (
                                    f"Unknown speaker '{speaker}'. "
                                    f"Available: {_format_available_speakers(runtime.pipeline.speaker_names())}"
                                ),
                            }
                        )
                        continue

                    response_format = req.response_format
                    session_req = req
                    live_session = runtime.pipeline.create_live_session(req=req, speaker=speaker)
                    stream_id, cancel_event = runtime.register_stream(canceler=cancel_current)
                    await outbound.put(
                        {
                            "type": "session.ready",
                            "session_id": session_id,
                            "model": runtime.pipeline.active_model_id or req_model,
                            "speaker": speaker,
                            "language": req.language or runtime.config.default_language,
                            "response_format": response_format,
                        }
                    )
                    continue

                if session_req is None:
                    await outbound.put(
                        {
                            "type": "error",
                            "session_id": requested_session_id or "",
                            "message": "session.start must be sent before other events.",
                        }
                    )
                    continue

                if requested_session_id and requested_session_id != session_id:
                    await outbound.put(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "message": "session_id does not match the active websocket session.",
                        }
                    )
                    continue

                if message_type == "text.append":
                    text = str(payload.get("text") or "")
                    if not text:
                        continue
                    assert live_session is not None
                    live_session.append_text(text)
                    pump_event.set()
                    continue

                if message_type == "session.finish":
                    if live_session is None:
                        await outbound.put(
                            {
                                "type": "session.done",
                                "session_id": session_id,
                                "cancelled": False,
                            }
                        )
                        await outbound.put(None)
                        return
                    live_session.finish()
                    pump_event.set()
                    await outbound.put(
                        {
                            "type": "session.draining",
                            "session_id": session_id,
                        }
                    )
                    if not live_session.is_started():
                        await outbound.put(
                            {
                                "type": "session.done",
                                "session_id": session_id,
                                "cancelled": False,
                            }
                        )
                        await outbound.put(None)
                        return
                    continue

                if message_type == "session.cancel":
                    cancel_current()
                    await outbound.put(
                        {
                            "type": "session.done",
                            "session_id": session_id,
                            "cancelled": True,
                        }
                    )
                    await outbound.put(None)
                    return

                await outbound.put(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "message": f"Unknown message type '{message_type}'.",
                    }
                )

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        pump_task = asyncio.create_task(pump())
        try:
            done, pending = await asyncio.wait(
                {sender_task, receiver_task, pump_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        except WebSocketDisconnect:
            cancel_current()
        finally:
            cancel_current()
            pump_event.set()
            if stream_id is not None:
                runtime.unregister_stream(stream_id)
            for task in (sender_task, receiver_task, pump_task):
                if not task.done():
                    task.cancel()

    @app.post("/v1/audio/speech")
    async def v1_audio_speech(req: SpeechSynthesisParams):
        runtime: CustomBridgeRuntime = app.state.runtime

        session_id = (req.session_id or "").strip()
        session_snapshot: BufferedSpeechSession | None = None
        session_segments: list[str] | None = None
        close_after_stream = False
        if session_id:
            try:
                runtime.upsert_buffered_session(req)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            session_snapshot, session_segments, close_after_stream = runtime.plan_session_segments(
                session_id,
                end_of_message=req.end_of_message,
            )
            if session_snapshot is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Buffered session '{session_id}' is not available.",
                )

            if not session_segments:
                return JSONResponse(
                    status_code=202,
                    content=BufferedSessionResponse(
                        session_id=session_id,
                        buffered_chunks=session_snapshot.buffered_chunks,
                        buffered_chars=session_snapshot.buffered_chars,
                        end_of_message=req.end_of_message,
                    ).model_dump(),
                )

            req_model = session_snapshot.model.strip()
            req_language = session_snapshot.language or runtime.config.default_language
            response_format = session_snapshot.response_format
        else:
            req_model = req.model.strip()
            req_language = req.language or runtime.config.default_language
            response_format = req.response_format

        active_model = runtime.pipeline.active_model_id
        if active_model and req_model != active_model and runtime.active_count() > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Cannot switch model from '{active_model}' to '{req_model}' "
                    "while streams are active. Stop current stream(s) first."
                ),
            )
        try:
            runtime.pipeline.ensure_model_loaded(req_model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        speaker = (
            (session_snapshot.speaker if session_snapshot is not None else None)
            or (session_snapshot.voice if session_snapshot is not None else None)
            or req.speaker
            or req.voice
            or runtime.config.default_speaker
        ).strip()
        if not speaker:
            raise HTTPException(
                status_code=400,
                detail="Missing speaker. Provide 'voice' (OpenAI-compatible) or 'speaker'.",
            )

        if not runtime.pipeline.has_speaker(speaker):
            speakers = runtime.pipeline.speaker_names()
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown speaker '{speaker}'. "
                    f"Available: {_format_available_speakers(speakers)}"
                ),
            )

        stream_requests: list[SpeechSynthesisParams]
        if session_snapshot is not None and session_segments is not None:
            stream_requests = []
            for index, segment_text in enumerate(session_segments):
                segment_format = response_format if index == 0 else "pcm"
                stream_requests.append(
                    session_snapshot.build_request(
                        req,
                        text=segment_text,
                        response_format=segment_format,
                        continuation_reset=bool(
                            session_snapshot.generated_segments == 0 and index == 0
                        ),
                    )
                )
        else:
            stream_requests = [req]

        stream_id, cancel_event = runtime.register_stream()

        async def iterator():
            try:
                for segment_req in stream_requests:
                    for chunk in runtime.pipeline.stream_audio_chunks(
                        req=segment_req,
                        cancel_event=cancel_event,
                        speaker=speaker,
                    ):
                        yield chunk
                        await asyncio.sleep(0)
                    if cancel_event.is_set():
                        break
                    if session_id:
                        runtime.mark_session_segments_generated(session_id, 1)
            finally:
                if session_id and close_after_stream:
                    runtime.clear_buffered_session(session_id)
                    runtime.pipeline.clear_continuation_session(session_id)
                runtime.unregister_stream(stream_id)

        media_type = "audio/wav" if response_format == "wav" else "audio/pcm"
        headers = {
            "X-Audio-Sample-Rate": str(runtime.config.sample_rate),
            "X-Audio-Channels": str(runtime.config.channels),
            "X-Audio-Bits-Per-Sample": str(runtime.config.bits_per_sample),
            "X-Model-Id": runtime.pipeline.active_model_id or req_model,
            "X-Custom-Speaker": speaker,
            "X-Custom-Language": req_language,
            "X-Session-Id": session_id,
            "X-Session-Segments": str(len(stream_requests)),
        }
        return StreamingResponse(iterator(), media_type=media_type, headers=headers)

    return app


app = create_app()
