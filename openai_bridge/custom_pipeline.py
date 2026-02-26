from __future__ import annotations

import logging
import struct
from threading import Event
from typing import Iterator

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel

from .custom_config import CustomBridgeConfig
from .schemas import SpeechSynthesisParams

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


class QwenCustomStreamingPipeline:
    def __init__(self, config: CustomBridgeConfig):
        self.config = config
        self.model: Qwen3TTSModel | None = None
        self._supported_speakers: set[str] | None = None
        self._speaker_names: list[str] = []
        self._startup_ready = False

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def startup_ready(self) -> bool:
        return self._startup_ready

    def load(self) -> None:
        if self.model is not None:
            return

        self._startup_ready = False
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

        speakers = self.model.get_supported_speakers()
        if speakers:
            self._speaker_names = sorted(str(s) for s in speakers)
            self._supported_speakers = {s.lower() for s in self._speaker_names}

        if self.config.warmup_enabled:
            self._warmup()
        self._startup_ready = True

    def _warmup(self) -> None:
        if self.model is None:
            return

        speaker = self.config.warmup_speaker or self.config.default_speaker
        if not speaker and self._speaker_names:
            speaker = self._speaker_names[0]
        if not speaker:
            logger.warning("Custom bridge warmup skipped: no speaker configured")
            return

        try:
            warmup_runs = max(1, self.config.warmup_runs)
            max_frames = min(self.config.max_frames, self.config.warmup_max_frames)
            warmup_texts = [
                self.config.warmup_text,
                f"{self.config.warmup_text} {self.config.warmup_text}",
                f"{self.config.warmup_text} {self.config.warmup_text} {self.config.warmup_text}",
            ]
            instruct_variants = [self.config.warmup_instruct, ""]
            instruct_variants = list(dict.fromkeys(instruct_variants))
            for i in range(warmup_runs):
                text = warmup_texts[i % len(warmup_texts)]
                instruct = instruct_variants[i % len(instruct_variants)]
                emitted = 0
                for _chunk, _sr in self.model.stream_generate_custom_voice(
                    text=text,
                    language=self.config.warmup_language or self.config.default_language,
                    speaker=speaker,
                    instruct=instruct,
                    emit_every_frames=self.config.emit_every_frames,
                    decode_window_frames=self.config.decode_window_frames,
                    overlap_samples=self.config.overlap_samples,
                    max_frames=max_frames,
                    use_optimized_decode=self.config.stream_use_optimized_decode,
                ):
                    emitted += 1
                logger.info(
                    "Custom bridge warmup run %s/%s complete chunks=%s instruct=%s",
                    i + 1,
                    warmup_runs,
                    emitted,
                    "set" if instruct else "empty",
                )
            logger.info(
                "Custom bridge warmup complete runs=%s optimized_decode=%s",
                warmup_runs,
                self.config.stream_use_optimized_decode,
            )
        except Exception as exc:
            logger.warning("Custom bridge warmup failed: %s", exc)

    @staticmethod
    def _float_audio_to_pcm16_bytes(audio: np.ndarray) -> bytes:
        samples = np.asarray(audio, dtype=np.float32)
        if samples.size == 0:
            return b""
        clipped = np.clip(samples, -1.0, 1.0)
        pcm_i16 = (clipped * 32767.0).astype(np.int16)
        return pcm_i16.tobytes(order="C")

    def has_speaker(self, speaker: str) -> bool:
        if not speaker:
            return False
        if self._supported_speakers is None:
            return True
        return speaker.lower() in self._supported_speakers

    def speaker_names(self) -> list[str]:
        return list(self._speaker_names)

    def stream_audio_chunks(
        self,
        req: SpeechSynthesisParams,
        cancel_event: Event,
        *,
        speaker: str,
    ) -> Iterator[bytes]:
        if self.model is None:
            raise RuntimeError("Pipeline not loaded")

        language = req.language or self.config.default_language
        instruct = req.instruct if req.instruct is not None else self.config.default_instruct
        emit_every_frames = req.emit_every_frames or self.config.emit_every_frames
        decode_window_frames = req.decode_window_frames or self.config.decode_window_frames
        overlap_samples = (
            req.overlap_samples if req.overlap_samples is not None else self.config.overlap_samples
        )
        max_frames = req.max_frames or self.config.max_frames
        use_optimized_decode = (
            req.use_optimized_decode
            if req.use_optimized_decode is not None
            else self.config.stream_use_optimized_decode
        )

        if req.response_format == "wav":
            yield wav_header(
                sample_rate=self.config.sample_rate,
                bits_per_sample=self.config.bits_per_sample,
                channels=self.config.channels,
            )

        for chunk, _sr in self.model.stream_generate_custom_voice(
            text=req.input,
            language=language,
            speaker=speaker,
            instruct=instruct,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            overlap_samples=overlap_samples,
            max_frames=max_frames,
            use_optimized_decode=use_optimized_decode,
        ):
            if cancel_event.is_set():
                return
            if chunk is None:
                continue
            pcm = self._float_audio_to_pcm16_bytes(chunk)
            if pcm:
                yield pcm
