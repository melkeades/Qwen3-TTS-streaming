from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .custom_config import CustomBridgeConfig
from .custom_pipeline import QwenCustomStreamingPipeline
from .schemas import (
    ErrorObject,
    ModelListResponse,
    ModelObject,
    SpeechResponseError,
    SpeechSynthesisParams,
    StopResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class CustomBridgeRuntime:
    config: CustomBridgeConfig
    pipeline: QwenCustomStreamingPipeline
    _active: dict[str, Event]
    _lock: Lock

    def register_stream(self) -> tuple[str, Event]:
        stream_id = str(uuid.uuid4())
        ev = Event()
        with self._lock:
            self._active[stream_id] = ev
        return stream_id, ev

    def unregister_stream(self, stream_id: str) -> None:
        with self._lock:
            self._active.pop(stream_id, None)

    def cancel_all(self) -> int:
        with self._lock:
            events = list(self._active.values())
        for ev in events:
            ev.set()
        return len(events)

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)


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


def create_app() -> FastAPI:
    config = CustomBridgeConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline = QwenCustomStreamingPipeline(config=config)
        logger.info(
            "Custom bridge startup begin model=%s compile=%s optimized_decode=%s warmup=%s runs=%s",
            config.model_id,
            config.optimize_use_compile,
            config.stream_use_optimized_decode,
            config.warmup_enabled,
            config.warmup_runs,
        )
        pipeline.load()

        app.state.runtime = CustomBridgeRuntime(
            config=config,
            pipeline=pipeline,
            _active={},
            _lock=Lock(),
        )
        logger.info(
            "Custom bridge ready model=%s speaker_count=%s",
            config.model_id,
            len(pipeline.speaker_names()),
        )
        try:
            yield
        finally:
            app.state.runtime.cancel_all()

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

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        runtime: CustomBridgeRuntime = app.state.runtime
        speakers = runtime.pipeline.speaker_names()
        return {
            "ok": bool(runtime.pipeline.startup_ready),
            "startup_ready": runtime.pipeline.startup_ready,
            "model_id": runtime.config.model_id,
            "default_speaker": runtime.config.default_speaker,
            "default_language": runtime.config.default_language,
            "stream_use_optimized_decode": runtime.config.stream_use_optimized_decode,
            "warmup_enabled": runtime.config.warmup_enabled,
            "warmup_runs": runtime.config.warmup_runs,
            "speaker_count": len(speakers),
            "speakers": speakers,
            "active_streams": runtime.active_count(),
        }

    @app.get("/v1/models", response_model=ModelListResponse)
    async def v1_models() -> ModelListResponse:
        runtime: CustomBridgeRuntime = app.state.runtime
        return ModelListResponse(
            data=[
                ModelObject(
                    id=runtime.config.model_id,
                    created=int(time.time()),
                    owned_by="qwen-local",
                )
            ]
        )

    @app.post("/v1/audio/stop", response_model=StopResponse)
    async def v1_audio_stop() -> StopResponse:
        runtime: CustomBridgeRuntime = app.state.runtime
        active_before = runtime.cancel_all()
        return StopResponse(stopped=True, active_before=active_before)

    @app.post("/v1/audio/speech")
    async def v1_audio_speech(req: SpeechSynthesisParams):
        runtime: CustomBridgeRuntime = app.state.runtime

        if req.model != runtime.config.model_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported model '{req.model}'. "
                    f"Expected '{runtime.config.model_id}'."
                ),
            )

        speaker = (req.speaker or req.voice or runtime.config.default_speaker).strip()
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

        stream_id, cancel_event = runtime.register_stream()

        async def iterator():
            try:
                for chunk in runtime.pipeline.stream_audio_chunks(
                    req=req,
                    cancel_event=cancel_event,
                    speaker=speaker,
                ):
                    yield chunk
                    await asyncio.sleep(0)
            finally:
                runtime.unregister_stream(stream_id)

        media_type = "audio/wav" if req.response_format == "wav" else "audio/pcm"
        headers = {
            "X-Audio-Sample-Rate": str(runtime.config.sample_rate),
            "X-Audio-Channels": str(runtime.config.channels),
            "X-Audio-Bits-Per-Sample": str(runtime.config.bits_per_sample),
            "X-Custom-Speaker": speaker,
            "X-Custom-Language": req.language or runtime.config.default_language,
        }
        return StreamingResponse(iterator(), media_type=media_type, headers=headers)

    return app


app = create_app()
