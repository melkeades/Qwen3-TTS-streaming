from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class ErrorObject(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class SpeechResponseError(BaseModel):
    error: ErrorObject


class SpeechSynthesisParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1)
    input: str = Field(..., min_length=1)
    voice: str = Field(..., min_length=1)
    response_format: Literal["pcm", "wav"] = "pcm"
    speed: float = Field(default=1.0, gt=0.0)

    # Optional bridge extensions (accepted but not required by OpenAI schema)
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    language: Optional[str] = None
    emit_every_frames: Optional[int] = Field(default=None, ge=1)
    decode_window_frames: Optional[int] = Field(default=None, ge=1)
    overlap_samples: Optional[int] = Field(default=None, ge=0)
    max_frames: Optional[int] = Field(default=None, ge=1)
    use_optimized_decode: Optional[bool] = None


class StopResponse(BaseModel):
    stopped: bool
    active_before: int


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "qwen-local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
