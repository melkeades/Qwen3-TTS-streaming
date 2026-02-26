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

from .config import BridgeConfig
from .pipeline import QwenStreamingPipeline
from .schemas import (
    ErrorObject,
    ModelListResponse,
    ModelObject,
    SpeechResponseError,
    SpeechSynthesisParams,
    StopResponse,
)
from .voice_registry import load_voice_registry

logger = logging.getLogger(__name__)


@dataclass
class BridgeRuntime:
    config: BridgeConfig
    pipeline: QwenStreamingPipeline
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


def create_app() -> FastAPI:
    config = BridgeConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        voices = load_voice_registry(config.voices_path)
        pipeline = QwenStreamingPipeline(config=config, voices=voices)
        pipeline.load()

        app.state.runtime = BridgeRuntime(
            config=config,
            pipeline=pipeline,
            _active={},
            _lock=Lock(),
        )
        logger.info(
            "Bridge ready model=%s voices=%s",
            config.model_id,
            ",".join(pipeline.voice_names()),
        )
        try:
            yield
        finally:
            app.state.runtime.cancel_all()

    app = FastAPI(title="Qwen OpenAI Bridge", version="0.1.0", lifespan=lifespan)
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
        logger.exception("Unhandled bridge exception")
        return _error_response(500, str(exc), type_="server_error")

    @app.get("/")
    async def index() -> FileResponse:
        runtime: BridgeRuntime = app.state.runtime
        return FileResponse(runtime.config.client_html_path)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        runtime: BridgeRuntime = app.state.runtime
        return {
            "ok": True,
            "model_id": runtime.config.model_id,
            "voices": runtime.pipeline.voice_names(),
            "active_streams": runtime.active_count(),
        }

    @app.get("/v1/models", response_model=ModelListResponse)
    async def v1_models() -> ModelListResponse:
        runtime: BridgeRuntime = app.state.runtime
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
        runtime: BridgeRuntime = app.state.runtime
        active_before = runtime.cancel_all()
        return StopResponse(stopped=True, active_before=active_before)

    @app.post("/v1/audio/speech")
    async def v1_audio_speech(req: SpeechSynthesisParams):
        runtime: BridgeRuntime = app.state.runtime

        if req.model != runtime.config.model_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported model '{req.model}'. "
                    f"Expected '{runtime.config.model_id}'."
                ),
            )

        if not runtime.pipeline.has_voice(req.voice):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown voice '{req.voice}'. "
                    f"Available: {', '.join(runtime.pipeline.voice_names())}"
                ),
            )

        stream_id, cancel_event = runtime.register_stream()

        async def iterator():
            try:
                for chunk in runtime.pipeline.stream_audio_chunks(
                    req=req,
                    cancel_event=cancel_event,
                ):
                    yield chunk
                    # Allow other requests (e.g. /v1/audio/stop) to be served between chunks.
                    await asyncio.sleep(0)
            finally:
                runtime.unregister_stream(stream_id)

        media_type = "audio/wav" if req.response_format == "wav" else "audio/pcm"
        headers = {
            "X-Audio-Sample-Rate": str(runtime.config.sample_rate),
            "X-Audio-Channels": str(runtime.config.channels),
            "X-Audio-Bits-Per-Sample": str(runtime.config.bits_per_sample),
        }
        return StreamingResponse(iterator(), media_type=media_type, headers=headers)

    return app


app = create_app()
