from __future__ import annotations

import logging
import struct
from threading import Event
from typing import Any, Iterator

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel

from .config import BridgeConfig
from .schemas import SpeechSynthesisParams
from .voice_registry import VoiceRegistry

logger = logging.getLogger(__name__)


def wav_header(sample_rate: int = 24000, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def _dtype_from_str(name: str) -> torch.dtype:
    n = (name or "").strip().lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16"}:
        return torch.float16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


class QwenStreamingPipeline:
    def __init__(self, config: BridgeConfig, voices: VoiceRegistry):
        self.config = config
        self.voices = voices
        self.model: Qwen3TTSModel | None = None
        self.voice_prompt_cache: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return

        torch.set_float32_matmul_precision("high")
        self.model = Qwen3TTSModel.from_pretrained(
            self.config.model_id,
            device_map=self.config.device_map,
            dtype=_dtype_from_str(self.config.dtype),
            attn_implementation=self.config.attn_implementation,
        )

        self.model.enable_streaming_optimizations(
            decode_window_frames=self.config.decode_window_frames,
            use_compile=self.config.optimize_use_compile,
            use_cuda_graphs=self.config.optimize_use_cuda_graphs,
            compile_mode=self.config.optimize_compile_mode,
            use_fast_codebook=self.config.optimize_use_fast_codebook,
            compile_codebook_predictor=self.config.optimize_compile_codebook_predictor,
            compile_talker=self.config.optimize_compile_talker,
        )

        # Prebuild prompt cache for each preset voice.
        for voice_name, preset in self.voices.presets.items():
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=str(preset.ref_audio),
                ref_text=preset.ref_text,
                x_vector_only_mode=preset.x_vector_only_mode,
            )
            self.voice_prompt_cache[voice_name] = prompt_items

        if self.config.warmup_enabled:
            self._warmup()

    def _warmup(self) -> None:
        if self.model is None:
            return
        try:
            first_voice = self.voices.names()[0]
            prompt = self.voice_prompt_cache[first_voice]
            for _chunk, _sr in self.model.stream_generate_voice_clone(
                text=self.config.warmup_text,
                language=self.config.default_language,
                voice_clone_prompt=prompt,
                emit_every_frames=self.config.emit_every_frames,
                decode_window_frames=self.config.decode_window_frames,
                overlap_samples=self.config.overlap_samples,
                max_frames=512,
            ):
                break
            logger.info("Bridge warmup complete")
        except Exception as exc:
            logger.warning("Bridge warmup failed: %s", exc)

    @staticmethod
    def _float_audio_to_pcm16_bytes(audio: np.ndarray) -> bytes:
        samples = np.asarray(audio, dtype=np.float32)
        if samples.size == 0:
            return b""
        clipped = np.clip(samples, -1.0, 1.0)
        pcm_i16 = (clipped * 32767.0).astype(np.int16)
        return pcm_i16.tobytes(order="C")

    def has_voice(self, voice: str) -> bool:
        return voice in self.voice_prompt_cache

    def voice_names(self) -> list[str]:
        return self.voices.names()

    def stream_audio_chunks(
        self,
        req: SpeechSynthesisParams,
        cancel_event: Event,
    ) -> Iterator[bytes]:
        if self.model is None:
            raise RuntimeError("Pipeline not loaded")

        prompt = self.voice_prompt_cache[req.voice]
        language = req.language or self.voices.get(req.voice).language or self.config.default_language

        emit_every_frames = req.emit_every_frames or self.config.emit_every_frames
        decode_window_frames = req.decode_window_frames or self.config.decode_window_frames
        overlap_samples = req.overlap_samples if req.overlap_samples is not None else self.config.overlap_samples
        max_frames = req.max_frames or self.config.max_frames

        if req.response_format == "wav":
            yield wav_header(
                sample_rate=self.config.sample_rate,
                bits_per_sample=self.config.bits_per_sample,
                channels=self.config.channels,
            )

        for chunk, _sr in self.model.stream_generate_voice_clone(
            text=req.input,
            language=language,
            voice_clone_prompt=prompt,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            overlap_samples=overlap_samples,
            max_frames=max_frames,
        ):
            if cancel_event.is_set():
                return
            if chunk is None:
                continue
            pcm = self._float_audio_to_pcm16_bytes(chunk)
            if pcm:
                yield pcm
