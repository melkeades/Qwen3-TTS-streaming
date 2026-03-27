from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import struct
import time
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


@dataclass(frozen=True)
class _ContinuationCacheKey:
    model_id: str
    speaker: str
    language: str
    continuation_id: str


@dataclass
class _ContinuationCacheEntry:
    ref_code_context: torch.Tensor
    updated_at: float
    text_tail: str = ""
    accumulated_text: str = ""
    emitted_samples: int = 0
    session_seed: int = 0
    tail_audio: np.ndarray | None = None
    voiced_rms: float = 0.0
    head_rms: float = 0.0


@dataclass
class ContinuationState:
    enabled: bool
    key: _ContinuationCacheKey | None = None
    requested_frames: int = 0
    used: bool = False
    used_frames: int = 0
    ref_code_context: torch.Tensor | None = None
    text_tail: str = ""
    sampling_mode: str = "off"
    sampling_seed: int | None = None
    accumulated_text: str = ""
    prior_emitted_samples: int = 0
    prior_tail_audio: np.ndarray | None = None
    prior_voiced_rms: float = 0.0
    prior_head_rms: float = 0.0
    total_generated_samples: int = 0
    total_emitted_samples: int = 0
    emitted_tail_audio: np.ndarray | None = None
    emitted_voiced_rms: float = 0.0
    emitted_head_rms: float = 0.0


class LiveCustomVoiceSession:
    def __init__(
        self,
        *,
        pipeline: "QwenCustomStreamingPipeline",
        model_session,
        response_format: str,
    ) -> None:
        self.pipeline = pipeline
        self.model_session = model_session
        self.response_format = response_format

    def append_text(self, text: str) -> None:
        self.model_session.append_text(text)

    def is_started(self) -> bool:
        return bool(self.model_session.is_started())

    def is_closed(self) -> bool:
        return bool(self.model_session.is_closed())

    def finish(self) -> None:
        self.model_session.finish()

    def cancel(self) -> None:
        self.model_session.cancel()

    def stream_bytes(self) -> Iterator[bytes]:
        if self.response_format == "wav":
            yield wav_header(
                sample_rate=self.pipeline.config.sample_rate,
                bits_per_sample=self.pipeline.config.bits_per_sample,
                channels=self.pipeline.config.channels,
            )
        for chunk, _sr in self.model_session.stream_audio():
            pcm = self.pipeline._float_audio_to_pcm16_bytes(chunk)
            if pcm:
                yield pcm


class QwenCustomStreamingPipeline:
    def __init__(self, config: CustomBridgeConfig):
        self.config = config
        self.model: Qwen3TTSModel | None = None
        self._runtime_device_map: str = config.device_map
        self._supported_speakers: set[str] | None = None
        self._speaker_names: list[str] = []
        self._active_model_id: str | None = None
        self._active_model_ref: str | None = None
        self._model_cache: dict[str, _ModelCacheEntry] = {}
        self._model_id_to_ref: dict[str, str] = {}
        self._model_id_to_speakers: dict[str, list[str]] = {}
        self._model_lock = Lock()
        self._startup_ready = False
        self._continuation_lock = Lock()
        self._continuation_cache: dict[_ContinuationCacheKey, _ContinuationCacheEntry] = {}

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
        add_model(self.config.fallback_model_id, self.config.fallback_model_id)
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

    def _build_continuation_key(
        self,
        *,
        model_id: str,
        speaker: str,
        language: str,
        continuation_id: str,
    ) -> _ContinuationCacheKey:
        return _ContinuationCacheKey(
            model_id=(model_id or "").strip(),
            speaker=(speaker or "").strip().lower(),
            language=(language or "").strip().lower(),
            continuation_id=(continuation_id or "").strip(),
        )

    def _purge_expired_continuations_unlocked(self, *, now: float | None = None) -> None:
        ttl = max(0, int(self.config.continuation_cache_ttl_sec))
        if ttl <= 0:
            self._continuation_cache.clear()
            return

        ts = now if now is not None else time.time()
        cutoff = ts - ttl
        stale_keys = [
            key
            for key, entry in self._continuation_cache.items()
            if entry.updated_at < cutoff
        ]
        for key in stale_keys:
            self._continuation_cache.pop(key, None)

    def _evict_overflow_continuations_unlocked(self) -> None:
        max_entries = max(1, int(self.config.continuation_cache_max_entries))
        overflow = len(self._continuation_cache) - max_entries
        if overflow <= 0:
            return

        oldest = sorted(
            self._continuation_cache.items(),
            key=lambda item: item[1].updated_at,
        )
        for key, _entry in oldest[:overflow]:
            self._continuation_cache.pop(key, None)

    def _drop_continuations_for_model_unlocked(self, model_id: str) -> None:
        target = (model_id or "").strip()
        if not target:
            return
        keys_to_drop = [
            key
            for key in self._continuation_cache.keys()
            if key.model_id == target
        ]
        for key in keys_to_drop:
            self._continuation_cache.pop(key, None)

    def clear_continuation_session(self, continuation_id: str) -> int:
        target = (continuation_id or "").strip()
        if not target:
            return 0
        with self._continuation_lock:
            keys_to_drop = [
                key
                for key in self._continuation_cache.keys()
                if key.continuation_id == target
            ]
            for key in keys_to_drop:
                self._continuation_cache.pop(key, None)
            return len(keys_to_drop)

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
        runtime_device_map = self._resolve_cuda_device_unlocked()
        self._runtime_device_map = runtime_device_map
        torch.set_float32_matmul_precision("high")
        logger.info(
            "Loading CustomVoice model id=%s ref=%s device=%s",
            model_id,
            model_ref,
            runtime_device_map,
        )
        model = Qwen3TTSModel.from_pretrained(
            model_ref,
            device_map=runtime_device_map,
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

    def _resolve_cuda_device_unlocked(self) -> str:
        preferred = "cuda:0"
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA unavailable; falling back to configured device_map='%s'.",
                self.config.device_map,
            )
            return self.config.device_map

        device_count = int(torch.cuda.device_count())
        if device_count <= 0:
            logger.warning(
                "CUDA reported available but no visible devices; using configured device_map='%s'.",
                self.config.device_map,
            )
            return self.config.device_map

        first_name = str(torch.cuda.get_device_name(0) or "")
        if "5090" in first_name.lower():
            logger.info("Pinned GPU confirmed: device=%s name=%s", preferred, first_name)
            return preferred

        # Prefer the first non-zero GPU if cuda:0 is not a 5090.
        fallback_index = 1 if device_count > 1 else 0
        fallback_name = str(torch.cuda.get_device_name(fallback_index) or "")
        fallback_device = f"cuda:{fallback_index}"
        logger.warning(
            "Preferred %s is '%s' (not 5090). Falling back to %s ('%s').",
            preferred,
            first_name,
            fallback_device,
            fallback_name,
        )
        return fallback_device

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
            with self._continuation_lock:
                self._drop_continuations_for_model_unlocked(requested_model)
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
            with self._continuation_lock:
                self._continuation_cache.clear()
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

    def prepare_continuation_state(
        self,
        req: SpeechSynthesisParams,
        *,
        speaker: str,
    ) -> ContinuationState:
        mode = (req.continuation_mode or "off").strip().lower()
        continuation_id = (req.continuation_id or "").strip()
        if mode != "acoustic_tail" or not continuation_id:
            return ContinuationState(enabled=False)

        effective_model = (self._active_model_id or req.model or "").strip()
        effective_language = (req.language or self.config.default_language).strip()
        decode_window_frames = req.decode_window_frames or self.config.decode_window_frames

        requested_frames = req.continuation_frames or self.config.continuation_default_frames
        requested_frames = max(1, int(requested_frames))
        if decode_window_frames > 0:
            requested_frames = min(requested_frames, int(decode_window_frames))

        key = self._build_continuation_key(
            model_id=effective_model,
            speaker=speaker,
            language=effective_language,
            continuation_id=continuation_id,
        )

        with self._continuation_lock:
            now = time.time()
            self._purge_expired_continuations_unlocked(now=now)
            if req.continuation_reset:
                self._continuation_cache.pop(key, None)

            entry = self._continuation_cache.get(key)
            if entry is not None and entry.ref_code_context.shape[0] > 0:
                used_frames = min(requested_frames, int(entry.ref_code_context.shape[0]))
                context = entry.ref_code_context[-used_frames:].detach().cpu().contiguous()
                entry.updated_at = now
                return ContinuationState(
                    enabled=True,
                    key=key,
                    requested_frames=requested_frames,
                    used=True,
                    used_frames=used_frames,
                    ref_code_context=context,
                    text_tail=(entry.text_tail or ""),
                    sampling_seed=int(entry.session_seed or self._build_continuation_seed(key=key)),
                    accumulated_text=(entry.accumulated_text or ""),
                    prior_emitted_samples=max(0, int(entry.emitted_samples or 0)),
                    prior_tail_audio=(
                        np.asarray(entry.tail_audio, dtype=np.float32).copy()
                        if entry.tail_audio is not None and int(entry.tail_audio.size) > 0
                        else None
                    ),
                    prior_voiced_rms=max(0.0, float(entry.voiced_rms or 0.0)),
                    prior_head_rms=max(0.0, float(entry.head_rms or 0.0)),
                )

        return ContinuationState(
            enabled=True,
            key=key,
            requested_frames=requested_frames,
            used=False,
            used_frames=0,
            ref_code_context=None,
            sampling_seed=self._build_continuation_seed(key=key),
            accumulated_text="",
            prior_emitted_samples=0,
            prior_tail_audio=None,
            prior_voiced_rms=0.0,
            prior_head_rms=0.0,
        )

    def _store_continuation_from_model(self, state: ContinuationState) -> None:
        if not state.enabled or state.key is None or self.model is None:
            return

        generated = self.model.get_last_stream_ref_code_context()
        if generated is None or generated.shape[0] <= 0:
            return

        keep_frames = min(state.requested_frames, int(generated.shape[0]))
        if keep_frames <= 0:
            return
        stored_context = generated[-keep_frames:].detach().cpu().contiguous()

        with self._continuation_lock:
            now = time.time()
            self._purge_expired_continuations_unlocked(now=now)
            self._continuation_cache[state.key] = _ContinuationCacheEntry(
                ref_code_context=stored_context,
                updated_at=now,
                text_tail=state.text_tail,
                accumulated_text=state.accumulated_text,
                emitted_samples=max(0, int(state.total_emitted_samples)),
                session_seed=int(state.sampling_seed or 0),
                tail_audio=(
                    np.asarray(state.emitted_tail_audio, dtype=np.float32).copy()
                    if state.emitted_tail_audio is not None and int(state.emitted_tail_audio.size) > 0
                    else None
                ),
                voiced_rms=max(0.0, float(state.emitted_voiced_rms or 0.0)),
                head_rms=max(0.0, float(state.emitted_head_rms or 0.0)),
            )
            self._evict_overflow_continuations_unlocked()

    def _extract_text_tail(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        tail_chars = max(1, int(self.config.continuation_text_tail_chars))
        sentences = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", raw, flags=re.MULTILINE)
        tail = sentences[-1].strip() if sentences else raw
        if len(tail) > tail_chars:
            tail = tail[-tail_chars:].lstrip()
        return tail

    @staticmethod
    def _continuation_seed_from_parts(*parts: bytes) -> int:
        digest = hashlib.blake2b(digest_size=8)
        for part in parts:
            digest.update(part)
        return int.from_bytes(digest.digest(), byteorder="big", signed=False) & 0x7FFFFFFF

    def _build_continuation_seed(self, *, key: _ContinuationCacheKey) -> int:
        key_bytes = "|".join(
            [
                "continuation-seed-v2",
                key.model_id,
                key.speaker,
                key.language,
            ]
        ).encode("utf-8", errors="ignore")
        return self._continuation_seed_from_parts(key_bytes)

    @staticmethod
    def _seed_sampling(seed: int | None) -> None:
        if seed is None:
            return
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    @staticmethod
    def _combine_continuation_text(prefix: str, current: str) -> str:
        prev = (prefix or "").strip()
        cur = (current or "").strip()
        if not prev:
            return cur
        if not cur:
            return prev
        return f"{prev} {cur}"

    def _continuation_tail_keep_samples(self) -> int:
        return max(256, int(self.config.continuation_alignment_tail_samples))

    @staticmethod
    def _update_tail_audio(
        existing: np.ndarray | None,
        chunk: np.ndarray,
        *,
        keep_samples: int,
    ) -> np.ndarray | None:
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio.size <= 0 or keep_samples <= 0:
            return existing
        if audio.size >= keep_samples:
            return audio[-keep_samples:].copy()
        if existing is None or int(existing.size) <= 0:
            return audio.copy()
        merged = np.concatenate([np.asarray(existing, dtype=np.float32).reshape(-1), audio])
        if merged.size > keep_samples:
            merged = merged[-keep_samples:]
        return merged.copy()

    @staticmethod
    def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
        left = np.asarray(a, dtype=np.float32).reshape(-1)
        right = np.asarray(b, dtype=np.float32).reshape(-1)
        if left.size <= 0 or right.size <= 0 or left.size != right.size:
            return -1.0
        left = left - float(left.mean())
        right = right - float(right.mean())
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-8:
            return -1.0
        return float(np.dot(left, right) / denom)

    def _find_continuation_cut(
        self,
        *,
        audio: np.ndarray,
        expected_samples: int,
        prior_tail_audio: np.ndarray | None,
    ) -> int:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size <= 0:
            return 0

        expected = int(np.clip(expected_samples, 0, samples.size))
        if prior_tail_audio is None:
            return expected

        tail = np.asarray(prior_tail_audio, dtype=np.float32).reshape(-1)
        if tail.size < 128:
            return expected

        tail_len = min(int(tail.size), self._continuation_tail_keep_samples(), int(samples.size))
        if tail_len < 128:
            return expected
        tail = tail[-tail_len:]

        configured_radius = max(128, int(self.config.continuation_alignment_search_samples))
        dynamic_radius = max(256, tail_len // 2, expected // 8)
        search_radius = min(configured_radius, dynamic_radius)
        min_end = tail_len
        max_end = int(samples.size)
        start_end = max(min_end, expected - search_radius)
        stop_end = min(max_end, expected + search_radius)
        if stop_end <= start_end:
            return expected

        coarse_step = 64 if (stop_end - start_end) > 512 else 8
        candidate_ends = list(range(start_end, stop_end + 1, coarse_step))
        if expected >= start_end and expected <= stop_end and expected not in candidate_ends:
            candidate_ends.append(expected)

        distance_penalty = 0.03
        candidate_scores: list[tuple[int, float, float]] = []
        for end in candidate_ends:
            segment = samples[end - tail_len : end]
            score = self._normalized_correlation(segment, tail)
            weighted = score - (
                distance_penalty * (abs(end - expected) / max(1.0, float(tail_len)))
            )
            candidate_scores.append((end, score, weighted))

        if not candidate_scores:
            return expected

        best_score = max(score for _, score, _weighted in candidate_scores)
        if best_score < 0.35:
            return expected

        anchor_end = max(
            candidate_scores,
            key=lambda item: (item[2], item[1]),
        )[0]

        refine_start = max(min_end, anchor_end - coarse_step)
        refine_stop = min(max_end, anchor_end + coarse_step)
        refined_scores: list[tuple[int, float, float]] = []
        for end in range(refine_start, refine_stop + 1):
            segment = samples[end - tail_len : end]
            score = self._normalized_correlation(segment, tail)
            weighted = score - (
                distance_penalty * (abs(end - expected) / max(1.0, float(tail_len)))
            )
            refined_scores.append((end, score, weighted))

        best_end = max(
            refined_scores,
            key=lambda item: (item[2], item[1]),
        )[0]
        return int(np.clip(best_end, 0, samples.size))

    def _trim_leading_continuation_silence(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size <= 0:
            return samples, 0

        max_trim = min(int(samples.size), max(128, int(self.config.sample_rate // 8)))
        if max_trim < 64:
            return samples, 0

        peak = float(np.max(np.abs(samples)))
        if peak <= 8e-4:
            return samples, 0

        threshold = max(peak * 0.02, 8e-4)
        block = 64
        trim = 0
        while trim + block <= max_trim:
            if float(np.max(np.abs(samples[trim : trim + block]))) >= threshold:
                break
            trim += block
        while trim < max_trim and float(abs(samples[trim])) < threshold:
            trim += 1
        if trim < block:
            return samples, 0
        return samples[trim:], trim

    @staticmethod
    def _voiced_mask(audio: np.ndarray, *, threshold: float = 0.01) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        return np.abs(samples) >= threshold

    @classmethod
    def _voiced_rms(cls, audio: np.ndarray, *, threshold: float = 0.01) -> float:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        mask = cls._voiced_mask(samples, threshold=threshold)
        if not np.any(mask):
            return 0.0
        voiced = samples[mask]
        return float(np.sqrt(np.mean(voiced * voiced)))

    @classmethod
    def _leading_voiced_rms(
        cls,
        audio: np.ndarray,
        *,
        window_samples: int,
        threshold: float = 0.01,
    ) -> float:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        mask = cls._voiced_mask(samples, threshold=threshold)
        idx = np.flatnonzero(mask)
        if idx.size <= 0:
            return 0.0
        start = int(idx[0])
        stop = min(int(samples.size), start + max(256, int(window_samples)))
        segment = samples[start:stop]
        if segment.size <= 0:
            return 0.0
        return cls._voiced_rms(segment, threshold=threshold)

    @staticmethod
    def _apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
        if abs(float(gain) - 1.0) <= 1e-6:
            return np.asarray(audio, dtype=np.float32).reshape(-1)
        return np.clip(
            np.asarray(audio, dtype=np.float32).reshape(-1) * float(gain),
            -1.0,
            1.0,
        )

    @staticmethod
    def _segment_word_count(text: str) -> int:
        return len(re.findall(r"[A-Za-z0-9']+", text or ""))

    def _max_reasonable_segment_samples(self, text: str) -> int:
        words = max(1, self._segment_word_count(text))
        max_seconds = min(20.0, max(5.0, 3.5 + float(words)))
        return int(max_seconds * float(self.config.sample_rate))

    def has_speaker(self, speaker: str) -> bool:
        if not speaker:
            return False
        if self._supported_speakers is None:
            return True
        return speaker.lower() in self._supported_speakers

    def speaker_names(self) -> list[str]:
        return list(self._speaker_names)

    def create_live_session(
        self,
        *,
        req: SpeechSynthesisParams,
        speaker: str,
    ) -> LiveCustomVoiceSession:
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
        gen_kwargs: dict[str, object] = {}
        if bool(self.config.stream_greedy_decoding):
            gen_kwargs.update(
                do_sample=False,
                subtalker_dosample=False,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                subtalker_temperature=0.0,
                subtalker_top_k=1,
                subtalker_top_p=1.0,
            )
        model_session = self.model.create_custom_voice_session(
            speaker=speaker,
            language=language,
            instruct=instruct,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            overlap_samples=overlap_samples,
            max_frames=max_frames,
            use_optimized_decode=use_optimized_decode,
            **gen_kwargs,
        )
        return LiveCustomVoiceSession(
            pipeline=self,
            model_session=model_session,
            response_format=req.response_format,
        )

    def stream_audio_chunks(
        self,
        req: SpeechSynthesisParams,
        cancel_event: Event,
        *,
        speaker: str,
        continuation_state: ContinuationState | None = None,
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
        state = continuation_state or self.prepare_continuation_state(
            req=req,
            speaker=speaker,
        )
        gen_kwargs: dict[str, object] = {}
        if bool(self.config.stream_greedy_decoding):
            gen_kwargs.update(
                do_sample=False,
                subtalker_dosample=False,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                subtalker_temperature=0.0,
                subtalker_top_k=1,
                subtalker_top_p=1.0,
            )
        text_to_synthesize = req.input
        tail_keep_samples = self._continuation_tail_keep_samples()
        emitted_tail_audio: np.ndarray | None = None
        state.total_generated_samples = 0
        state.total_emitted_samples = 0
        state.emitted_tail_audio = None
        state.emitted_voiced_rms = 0.0
        state.emitted_head_rms = 0.0
        if state.enabled:
            state.accumulated_text = self._combine_continuation_text(
                state.accumulated_text,
                req.input,
            )
        if (
            state.enabled
            and state.used
            and bool(self.config.continuation_text_priming_enabled)
            and state.text_tail
        ):
            style_bridge = (
                "Continue in the same speaking style, speed, tone, and prosody "
                "as the immediately previous segment."
            )
            context_line = f"Previous segment tail context: {state.text_tail}"
            pieces = [p for p in [instruct, style_bridge, context_line] if p]
            instruct = "\n".join(pieces)
        state.text_tail = self._extract_text_tail(req.input)
        if state.enabled and bool(self.config.continuation_sampling_enabled):
            if state.used and bool(self.config.continuation_followup_greedy_enabled):
                gen_kwargs.update(
                    do_sample=False,
                    subtalker_dosample=False,
                    temperature=0.0,
                    top_k=1,
                    subtalker_temperature=0.0,
                    subtalker_top_k=1,
                )
                state.sampling_mode = "greedy_followup"
            else:
                gen_kwargs.update(
                    temperature=float(self.config.continuation_sampling_temperature),
                    top_k=int(self.config.continuation_sampling_top_k),
                    subtalker_temperature=float(
                        self.config.continuation_sampling_subtalker_temperature
                    ),
                    subtalker_top_k=int(self.config.continuation_sampling_subtalker_top_k),
                )
                state.sampling_mode = "seeded_clamped"

        if req.response_format == "wav":
            yield wav_header(
                sample_rate=self.config.sample_rate,
                bits_per_sample=self.config.bits_per_sample,
                channels=self.config.channels,
            )
        generation_completed = False
        try:
            if state.enabled and bool(self.config.continuation_sampling_enabled):
                self._seed_sampling(state.sampling_seed)
            emitted_samples = 0
            max_segment_samples = self._max_reasonable_segment_samples(text_to_synthesize)
            voiced_sum_sq = 0.0
            voiced_count = 0
            emitted_head_buffer = np.zeros((0,), dtype=np.float32)
            gain = 1.0
            gain_ready = not (
                state.enabled
                and state.used
                and bool(self.config.continuation_gain_match_enabled)
                and state.prior_voiced_rms > 0.0
            )
            pending_gain_chunks: list[np.ndarray] = []
            pending_gain_samples = 0
            gain_buffer_samples = max(
                512,
                int(self.config.sample_rate * max(40, self.config.continuation_gain_match_buffer_ms) / 1000.0),
            )
            head_window_samples = max(512, int(self.config.sample_rate * 0.22))

            def emit_scaled(audio_chunk: np.ndarray) -> bytes:
                nonlocal emitted_samples, emitted_tail_audio, voiced_sum_sq, voiced_count, emitted_head_buffer
                scaled = self._apply_gain(audio_chunk, gain)
                emitted_samples += int(scaled.size)
                emitted_tail_audio = self._update_tail_audio(
                    emitted_tail_audio,
                    scaled,
                    keep_samples=tail_keep_samples,
                )
                if emitted_head_buffer.size < head_window_samples:
                    keep = max(0, head_window_samples - int(emitted_head_buffer.size))
                    emitted_head_buffer = np.concatenate(
                        [emitted_head_buffer, scaled[:keep]],
                        axis=0,
                    )
                voiced = scaled[self._voiced_mask(scaled)]
                if voiced.size > 0:
                    voiced_sum_sq += float(np.dot(voiced, voiced))
                    voiced_count += int(voiced.size)
                return self._float_audio_to_pcm16_bytes(scaled)

            def resolve_gain_from_pending(force: bool = False) -> bytes:
                nonlocal gain, gain_ready, pending_gain_chunks, pending_gain_samples
                if not pending_gain_chunks:
                    return b""
                buffered = np.concatenate(pending_gain_chunks, axis=0).astype(np.float32, copy=False)
                enough = pending_gain_samples >= gain_buffer_samples
                head_rms = self._leading_voiced_rms(
                    buffered,
                    window_samples=head_window_samples,
                )
                if not gain_ready and (force or enough or head_rms > 0.0):
                    if head_rms > 0.0 and state.prior_voiced_rms > 0.0:
                        target_rms = max(
                            float(state.prior_voiced_rms),
                            float(state.prior_head_rms) * 0.85,
                        )
                        raw_gain = float(target_rms) / float(head_rms)
                        gain = float(
                            np.clip(
                                raw_gain,
                                float(self.config.continuation_gain_match_min_gain),
                                float(self.config.continuation_gain_match_max_gain),
                            )
                        )
                        if abs(gain - 1.0) <= float(self.config.continuation_gain_match_tolerance):
                            gain = 1.0
                    gain_ready = True
                if not gain_ready:
                    return b""
                pending_gain_chunks = []
                pending_gain_samples = 0
                return emit_scaled(buffered)

            for chunk, _sr in self.model.stream_generate_custom_voice(
                text=text_to_synthesize,
                language=language,
                speaker=speaker,
                instruct=instruct,
                emit_every_frames=emit_every_frames,
                decode_window_frames=decode_window_frames,
                overlap_samples=overlap_samples,
                max_frames=max_frames,
                use_optimized_decode=use_optimized_decode,
                external_ref_code_context=state.ref_code_context if state.enabled else None,
                capture_ref_code_context_frames=state.requested_frames if state.enabled else 0,
                **gen_kwargs,
            ):
                if cancel_event.is_set():
                    return
                if chunk is None:
                    continue
                chunk_audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
                state.total_generated_samples += int(chunk_audio.size)
                if not gain_ready:
                    pending_gain_chunks.append(chunk_audio.copy())
                    pending_gain_samples += int(chunk_audio.size)
                    pcm = resolve_gain_from_pending(force=False)
                else:
                    pcm = emit_scaled(chunk_audio)
                if pcm:
                    yield pcm
                if emitted_samples >= max_segment_samples:
                    logger.warning(
                        "Stopping oversized segment early speaker=%s words=%s emitted_samples=%s limit=%s text=%r",
                        speaker,
                        self._segment_word_count(text_to_synthesize),
                        emitted_samples,
                        max_segment_samples,
                        text_to_synthesize[:160],
                    )
                    break
            if pending_gain_chunks:
                pcm = resolve_gain_from_pending(force=True)
                if pcm:
                    yield pcm
            state.total_emitted_samples = emitted_samples
            state.emitted_tail_audio = emitted_tail_audio
            if voiced_count > 0:
                state.emitted_voiced_rms = float(np.sqrt(voiced_sum_sq / float(voiced_count)))
            if emitted_head_buffer.size > 0:
                state.emitted_head_rms = self._leading_voiced_rms(
                    emitted_head_buffer,
                    window_samples=head_window_samples,
                )
            generation_completed = True
        finally:
            if generation_completed:
                self._store_continuation_from_model(state)
