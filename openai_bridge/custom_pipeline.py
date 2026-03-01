from __future__ import annotations

import gc
import json
import logging
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
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


@dataclass
class _ModelCacheEntry:
    model: Qwen3TTSModel
    model_ref: str
    speaker_names: list[str]
    supported_speakers: set[str] | None


class QwenCustomStreamingPipeline:
    def __init__(self, config: CustomBridgeConfig):
        self.config = config
        self.model: Qwen3TTSModel | None = None
        self._supported_speakers: set[str] | None = None
        self._speaker_names: list[str] = []
        self._active_model_id: str | None = None
        self._active_model_ref: str | None = None
        self._model_cache: dict[str, _ModelCacheEntry] = {}
        self._model_id_to_ref: dict[str, str] = {}
        self._model_id_to_speakers: dict[str, list[str]] = {}
        self._model_lock = Lock()
        self._startup_ready = False

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def startup_ready(self) -> bool:
        return self._startup_ready

    @property
    def active_model_id(self) -> str | None:
        return self._active_model_id

    def cached_model_ids(self) -> list[str]:
        with self._model_lock:
            return list(self._model_cache.keys())

    def load(self) -> None:
        self.ensure_model_loaded(self.config.model_id)

    @staticmethod
    def _looks_like_model_dir(file_names: set[str]) -> bool:
        has_config = "config.json" in file_names
        has_weights = any(name.endswith(".safetensors") or name.endswith(".bin") for name in file_names)
        has_tokenizer = (
            "tokenizer_config.json" in file_names
            or "tokenizer.json" in file_names
            or ("vocab.json" in file_names and "merges.txt" in file_names)
        )
        has_generation_cfg = "generation_config.json" in file_names
        return has_config and has_weights and (has_tokenizer or has_generation_cfg)

    def _model_id_from_path(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(self.config.repo_root.resolve())
            return rel.as_posix()
        except ValueError:
            return str(path.resolve())

    def _discover_models_unlocked(self, *, refresh: bool) -> list[str]:
        if self._model_id_to_ref and not refresh:
            return list(self._model_id_to_ref.keys())

        model_map: dict[str, str] = {}

        def add_model(model_id: str, model_ref: str) -> None:
            key = (model_id or "").strip()
            if key and key not in model_map:
                model_map[key] = model_ref

        add_model(self.config.model_id, self.config.model_id)
        for extra in self.config.additional_model_ids:
            add_model(extra, extra)

        max_depth = max(0, self.config.model_scan_max_depth)
        for root in self.config.model_scan_roots:
            if not root.exists() or not root.is_dir():
                continue
            discovered_paths: list[Path] = []
            for dirpath, dirnames, filenames in os.walk(root):
                current = Path(dirpath)
                rel_depth = len(current.relative_to(root).parts)
                if rel_depth > max_depth:
                    dirnames[:] = []
                    continue
                if self._looks_like_model_dir({name.lower() for name in filenames}):
                    discovered_paths.append(current.resolve())
                if rel_depth == max_depth:
                    dirnames[:] = []
            for path in sorted(discovered_paths, key=lambda p: str(p).lower()):
                model_id = self._model_id_from_path(path)
                add_model(model_id, str(path))

        if self._active_model_id and self._active_model_ref:
            add_model(self._active_model_id, self._active_model_ref)
        for cached_model_id, cached_entry in self._model_cache.items():
            add_model(cached_model_id, cached_entry.model_ref)

        self._model_id_to_ref = model_map
        for model_id in list(self._model_id_to_speakers.keys()):
            if model_id not in model_map and model_id not in self._model_cache:
                self._model_id_to_speakers.pop(model_id, None)
        return list(model_map.keys())

    def discover_models(self, *, refresh: bool = True) -> list[str]:
        with self._model_lock:
            return self._discover_models_unlocked(refresh=refresh)

    @staticmethod
    def _normalize_speaker_values(value: object) -> list[str]:
        out: list[str] = []
        if isinstance(value, dict):
            for key in value.keys():
                name = str(key).strip()
                if name:
                    out.append(name)
            return out
        if isinstance(value, list):
            for item in value:
                name = str(item).strip()
                if name:
                    out.append(name)
            return out
        if isinstance(value, str):
            name = value.strip()
            return [name] if name else []
        return out

    def _extract_speakers_from_config_data(self, data: dict) -> list[str]:
        keys = ("spk_id", "speaker_ids", "speaker_id", "supported_speakers", "speakers")
        candidates: list[str] = []

        def collect(value: object) -> None:
            for name in self._normalize_speaker_values(value):
                if name not in candidates:
                    candidates.append(name)

        talker = data.get("talker_config")
        if isinstance(talker, dict):
            for key in keys:
                if key in talker:
                    collect(talker[key])

        for key in keys:
            if key in data:
                collect(data[key])

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if str(key).lower() in keys:
                        collect(value)
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
        return candidates

    def _speakers_from_model_ref_unlocked(self, model_ref: str) -> list[str]:
        model_path = Path(model_ref)
        if not model_path.exists() or not model_path.is_dir():
            return []

        config_path = model_path / "config.json"
        if not config_path.exists():
            return []

        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.debug("Failed reading speaker ids from %s", config_path, exc_info=True)
            return []

        if not isinstance(data, dict):
            return []
        return self._extract_speakers_from_config_data(data)

    def speaker_names_for_model(self, model_id: str, *, refresh: bool = True) -> list[str]:
        requested_model = (model_id or "").strip()
        if not requested_model:
            return []

        with self._model_lock:
            self._discover_models_unlocked(refresh=refresh)
            model_ref = self._model_id_to_ref.get(requested_model)
            if model_ref is None:
                available = ", ".join(self._model_id_to_ref.keys()) or "(none)"
                raise ValueError(
                    f"Unsupported model '{requested_model}'. Available: {available}"
                )

            cached = self._model_cache.get(requested_model)
            if cached is not None:
                speakers = list(cached.speaker_names)
                self._model_id_to_speakers[requested_model] = speakers
                return speakers

            if (not refresh) and requested_model in self._model_id_to_speakers:
                return list(self._model_id_to_speakers[requested_model])

            speakers = self._speakers_from_model_ref_unlocked(model_ref)
            if not speakers and requested_model == self._active_model_id:
                speakers = list(self._speaker_names)
            if not speakers and requested_model == self.config.model_id:
                default = (self.config.default_speaker or "").strip()
                if default:
                    speakers = [default]

            self._model_id_to_speakers[requested_model] = list(speakers)
            return list(speakers)

    def _activate_model_entry_unlocked(self, model_id: str, entry: _ModelCacheEntry) -> None:
        self.model = entry.model
        self._speaker_names = list(entry.speaker_names)
        self._supported_speakers = (
            set(entry.supported_speakers) if entry.supported_speakers is not None else None
        )
        self._active_model_id = model_id
        self._active_model_ref = entry.model_ref
        self._model_id_to_speakers[model_id] = list(entry.speaker_names)
        self._startup_ready = True

    @staticmethod
    def _release_torch_memory_unlocked() -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.debug("torch.cuda.empty_cache() failed", exc_info=True)

    def _load_model_unlocked(self, *, model_id: str, model_ref: str) -> _ModelCacheEntry:
        self._startup_ready = False
        torch.set_float32_matmul_precision("high")
        logger.info("Loading CustomVoice model id=%s ref=%s", model_id, model_ref)
        model = Qwen3TTSModel.from_pretrained(
            model_ref,
            device_map=self.config.device_map,
            dtype=_dtype_from_str(self.config.dtype),
            attn_implementation=self.config.attn_implementation,
        )

        model.enable_streaming_optimizations(
            decode_window_frames=self.config.decode_window_frames,
            use_compile=self.config.optimize_use_compile,
            use_cuda_graphs=self.config.optimize_use_cuda_graphs,
            compile_mode=self.config.optimize_compile_mode,
            use_fast_codebook=self.config.optimize_use_fast_codebook,
            compile_codebook_predictor=self.config.optimize_compile_codebook_predictor,
            compile_talker=self.config.optimize_compile_talker,
        )

        speakers = model.get_supported_speakers()
        if speakers:
            speaker_names = sorted(str(s) for s in speakers)
            supported_speakers = {s.lower() for s in speaker_names}
        else:
            speaker_names = []
            supported_speakers = None

        entry = _ModelCacheEntry(
            model=model,
            model_ref=model_ref,
            speaker_names=speaker_names,
            supported_speakers=supported_speakers,
        )
        self._activate_model_entry_unlocked(model_id, entry)
        if self.config.warmup_enabled:
            self._warmup()
        self._startup_ready = True
        logger.info(
            "CustomVoice model ready id=%s speakers=%s startup_ready=%s",
            model_id,
            len(speaker_names),
            self._startup_ready,
        )
        return entry

    def ensure_model_loaded(self, model_id: str) -> str:
        requested_model = (model_id or "").strip()
        if not requested_model:
            raise ValueError("Missing model id")

        with self._model_lock:
            self._discover_models_unlocked(refresh=True)
            model_ref = self._model_id_to_ref.get(requested_model)
            if model_ref is None:
                available = ", ".join(self._model_id_to_ref.keys()) or "(none)"
                raise ValueError(
                    f"Unsupported model '{requested_model}'. Available: {available}"
                )

            if self.model is not None and self._active_model_id == requested_model:
                return requested_model

            cached = self._model_cache.get(requested_model)
            if cached is not None:
                logger.info("Reusing cached CustomVoice model id=%s", requested_model)
                self._activate_model_entry_unlocked(requested_model, cached)
                return requested_model

            entry = self._load_model_unlocked(model_id=requested_model, model_ref=model_ref)
            self._model_cache[requested_model] = entry
            return requested_model

    def unload_model(self, model_id: str | None = None) -> bool:
        requested_model = (model_id or self._active_model_id or "").strip()
        if not requested_model:
            return False

        with self._model_lock:
            entry = self._model_cache.pop(requested_model, None)
            if entry is None:
                return False

            was_active = requested_model == self._active_model_id
            model_obj = entry.model
            del entry
            del model_obj

            self._model_id_to_speakers.pop(requested_model, None)
            if was_active:
                self.model = None
                self._speaker_names = []
                self._supported_speakers = None
                self._active_model_id = None
                self._active_model_ref = None
                self._startup_ready = False

            self._release_torch_memory_unlocked()
            return True

    def unload_all_models(self) -> int:
        with self._model_lock:
            unloaded = len(self._model_cache)
            self._model_cache.clear()
            self._model_id_to_speakers.clear()
            self.model = None
            self._speaker_names = []
            self._supported_speakers = None
            self._active_model_id = None
            self._active_model_ref = None
            self._startup_ready = False
            self._release_torch_memory_unlocked()
            return unloaded

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
        instruct = (
            req.instructions
            if req.instructions is not None
            else (req.instruct if req.instruct is not None else self.config.default_instruct)
        )
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
