from __future__ import annotations

import time
import unittest
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import torch

from openai_bridge.custom_config import CustomBridgeConfig
from openai_bridge.custom_pipeline import (
    QwenCustomStreamingPipeline,
    _ContinuationCacheEntry,
)
from openai_bridge.schemas import SpeechSynthesisParams


class _DummyModel:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None
        self._tail = torch.arange(0, 240, dtype=torch.int64).reshape(120, 2)
        self.first_audio = np.sin(np.linspace(0.0, 24.0, 960, dtype=np.float32)) * 0.2
        self.second_audio = np.cos(np.linspace(0.0, 18.0, 640, dtype=np.float32)) * 0.2
        self.boundary_shift = np.full((37,), 0.03, dtype=np.float32)

    def stream_generate_custom_voice(self, **kwargs):
        self.last_kwargs = kwargs
        text = str(kwargs.get("text") or "")
        if text == "First chunk sentence one. First chunk sentence two.":
            audio = self.first_audio
        elif text == "Second chunk starts now.":
            audio = self.second_audio
        else:
            total_samples = max(320, len(text) * 20)
            audio = np.zeros(total_samples, dtype=np.float32)
        emitted = 0
        while emitted < int(audio.size):
            step = min(160, int(audio.size) - emitted)
            chunk = audio[emitted : emitted + step]
            emitted += step
            yield chunk, 24000

    def get_last_stream_ref_code_context(self):
        return self._tail


class _PauseHeadDummyModel(_DummyModel):
    def __init__(self) -> None:
        super().__init__()
        self.leading_pause = np.zeros((240,), dtype=np.float32)

    def stream_generate_custom_voice(self, **kwargs):
        self.last_kwargs = kwargs
        text = str(kwargs.get("text") or "")
        if text == "First chunk sentence one. First chunk sentence two.":
            audio = self.first_audio
        elif text == "Second chunk starts now.":
            audio = np.concatenate([self.leading_pause, self.second_audio])
        else:
            total_samples = max(320, len(text) * 20)
            audio = np.zeros(total_samples, dtype=np.float32)
        emitted = 0
        while emitted < int(audio.size):
            step = min(160, int(audio.size) - emitted)
            chunk = audio[emitted : emitted + step]
            emitted += step
            yield chunk, 24000


class _LoudFollowupDummyModel(_DummyModel):
    def __init__(self) -> None:
        super().__init__()
        self.second_audio = np.clip(self.second_audio * 2.4, -0.95, 0.95).astype(np.float32)


def _make_config(**overrides) -> CustomBridgeConfig:
    base = CustomBridgeConfig.from_env()
    base = replace(
        base,
        repo_root=Path(__file__).resolve().parents[2],
        warmup_enabled=False,
        continuation_default_frames=48,
        continuation_cache_ttl_sec=600,
        continuation_cache_max_entries=256,
        continuation_alignment_tail_samples=4096,
        continuation_alignment_search_samples=12000,
    )
    return replace(base, **overrides)


class CustomContinuationTests(unittest.TestCase):
    def _make_pipeline(
        self,
        *,
        model_factory=_DummyModel,
        **config_overrides,
    ) -> QwenCustomStreamingPipeline:
        pipeline = QwenCustomStreamingPipeline(config=_make_config(**config_overrides))
        pipeline.model = model_factory()
        pipeline._active_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        pipeline._speaker_names = ["Ryan"]
        return pipeline

    def _make_req(self, **overrides) -> SpeechSynthesisParams:
        payload = {
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "input": "Hello there. This is a segmented run.",
            "voice": "Ryan",
            "speaker": "Ryan",
            "language": "English",
            "response_format": "pcm",
            "speed": 1.0,
        }
        payload.update(overrides)
        return SpeechSynthesisParams(**payload)

    def test_schema_accepts_continuation_fields(self) -> None:
        req = self._make_req(
            continuation_mode="acoustic_tail",
            continuation_id="run-123",
            continuation_reset=True,
            continuation_frames=40,
        )
        self.assertEqual(req.continuation_mode, "acoustic_tail")
        self.assertEqual(req.continuation_id, "run-123")
        self.assertTrue(req.continuation_reset)
        self.assertEqual(req.continuation_frames, 40)

    def test_continuation_reuse_and_reset(self) -> None:
        pipeline = self._make_pipeline()
        ev = Event()

        first_req = self._make_req(
            continuation_mode="acoustic_tail",
            continuation_id="session-A",
            continuation_reset=True,
        )
        first_state = pipeline.prepare_continuation_state(first_req, speaker="Ryan")
        self.assertFalse(first_state.used)
        list(
            pipeline.stream_audio_chunks(
                req=first_req,
                cancel_event=ev,
                speaker="Ryan",
                continuation_state=first_state,
            )
        )
        self.assertIsNone(pipeline.model.last_kwargs["external_ref_code_context"])
        self.assertEqual(pipeline.model.last_kwargs["capture_ref_code_context_frames"], 48)

        second_req = self._make_req(
            continuation_mode="acoustic_tail",
            continuation_id="session-A",
        )
        second_state = pipeline.prepare_continuation_state(second_req, speaker="Ryan")
        self.assertTrue(second_state.used)
        self.assertEqual(second_state.used_frames, 48)
        self.assertIsNotNone(second_state.ref_code_context)
        self.assertEqual(second_state.ref_code_context.shape[0], 48)

        list(
            pipeline.stream_audio_chunks(
                req=second_req,
                cancel_event=ev,
                speaker="Ryan",
                continuation_state=second_state,
            )
        )
        self.assertIsNotNone(pipeline.model.last_kwargs["external_ref_code_context"])

    def test_continuation_ttl_and_max_entries(self) -> None:
        pipeline = self._make_pipeline(
            continuation_cache_ttl_sec=5,
            continuation_cache_max_entries=1,
        )
        req_a = self._make_req(continuation_mode="acoustic_tail", continuation_id="A")
        key_a = pipeline.prepare_continuation_state(req_a, speaker="Ryan").key
        assert key_a is not None

        with pipeline._continuation_lock:
            pipeline._continuation_cache[key_a] = _ContinuationCacheEntry(
                ref_code_context=torch.ones((12, 2), dtype=torch.int64),
                updated_at=time.time() - 1000,
            )

        stale_state = pipeline.prepare_continuation_state(req_a, speaker="Ryan")
        self.assertFalse(stale_state.used)

    def test_continuation_seed_does_not_depend_on_session_id(self) -> None:
        pipeline = self._make_pipeline()
        req_a = self._make_req(continuation_mode="acoustic_tail", continuation_id="session-A")
        req_b = self._make_req(continuation_mode="acoustic_tail", continuation_id="session-B")

        state_a = pipeline.prepare_continuation_state(req_a, speaker="Ryan")
        state_b = pipeline.prepare_continuation_state(req_b, speaker="Ryan")

        self.assertIsInstance(state_a.sampling_seed, int)
        self.assertEqual(state_a.sampling_seed, state_b.sampling_seed)

    def test_followup_segment_uses_only_new_text_with_backend_continuation(self) -> None:
        pipeline = self._make_pipeline()
        ev = Event()

        req1 = self._make_req(
            continuation_mode="acoustic_tail",
            continuation_id="session-text",
            input="First chunk sentence one. First chunk sentence two.",
        )
        state1 = pipeline.prepare_continuation_state(req1, speaker="Ryan")
        list(
            pipeline.stream_audio_chunks(
                req=req1,
                cancel_event=ev,
                speaker="Ryan",
                continuation_state=state1,
            )
        )

        req2 = self._make_req(
            continuation_mode="acoustic_tail",
            continuation_id="session-text",
            input="Second chunk starts now.",
        )
        state2 = pipeline.prepare_continuation_state(req2, speaker="Ryan")
        self.assertTrue(state2.used)
        self.assertIn("First chunk sentence two.", state2.text_tail)
        self.assertIsInstance(state2.sampling_seed, int)
        self.assertGreater(state2.prior_emitted_samples, 0)

        out_chunks = list(
            pipeline.stream_audio_chunks(
                req=req2,
                cancel_event=ev,
                speaker="Ryan",
                continuation_state=state2,
            )
        )

        sent_text = str(pipeline.model.last_kwargs.get("text") or "")
        sent_instruct = str(pipeline.model.last_kwargs.get("instruct") or "")
        self.assertNotIn("First chunk sentence one.", sent_text)
        self.assertIn("Second chunk starts now.", sent_text)
        self.assertIn("Continue in the same speaking style", sent_instruct)
        self.assertIn("First chunk sentence two.", sent_instruct)
        self.assertGreater(len(out_chunks), 0)
        emitted_pcm = b"".join(out_chunks)
        emitted_samples = len(emitted_pcm) // 2
        self.assertEqual(emitted_samples, int(pipeline.model.second_audio.size))

    def test_followup_segment_trims_leading_pause_after_alignment(self) -> None:
        pipeline = self._make_pipeline(model_factory=_PauseHeadDummyModel)
        ev = Event()

        req1 = self._make_req(
            continuation_mode="acoustic_tail",
            continuation_id="session-pause",
            input="First chunk sentence one. First chunk sentence two.",
        )
        state1 = pipeline.prepare_continuation_state(req1, speaker="Ryan")
        list(
            pipeline.stream_audio_chunks(
                req=req1,
                cancel_event=ev,
                speaker="Ryan",
                continuation_state=state1,
            )
        )

        req2 = self._make_req(
            continuation_mode="acoustic_tail",
            continuation_id="session-pause",
            input="Second chunk starts now.",
        )
        state2 = pipeline.prepare_continuation_state(req2, speaker="Ryan")
        out_chunks = list(
            pipeline.stream_audio_chunks(
                req=req2,
                cancel_event=ev,
                speaker="Ryan",
                continuation_state=state2,
            )
        )

        emitted_pcm = b"".join(out_chunks)
        emitted_audio = np.frombuffer(emitted_pcm, dtype=np.int16).astype(np.float32) / 32767.0
        self.assertEqual(
            int(emitted_audio.size),
            int(pipeline.model.leading_pause.size + pipeline.model.second_audio.size),
        )
        self.assertLess(float(np.max(np.abs(emitted_audio[:32]))), 0.001)

    def test_followup_gain_matching_reduces_loudness_jump(self) -> None:
        pipeline = self._make_pipeline(model_factory=_LoudFollowupDummyModel)
        ev = Event()

        req1 = self._make_req(
            continuation_mode="acoustic_tail",
            continuation_id="session-gain",
            input="First chunk sentence one. First chunk sentence two.",
        )
        state1 = pipeline.prepare_continuation_state(req1, speaker="Ryan")
        first_pcm = b"".join(
            pipeline.stream_audio_chunks(
                req=req1,
                cancel_event=ev,
                speaker="Ryan",
                continuation_state=state1,
            )
        )
        first_audio = np.frombuffer(first_pcm, dtype=np.int16).astype(np.float32) / 32767.0

        req2 = self._make_req(
            continuation_mode="acoustic_tail",
            continuation_id="session-gain",
            input="Second chunk starts now.",
        )
        state2 = pipeline.prepare_continuation_state(req2, speaker="Ryan")
        second_pcm = b"".join(
            pipeline.stream_audio_chunks(
                req=req2,
                cancel_event=ev,
                speaker="Ryan",
                continuation_state=state2,
            )
        )
        second_audio = np.frombuffer(second_pcm, dtype=np.int16).astype(np.float32) / 32767.0

        first_rms = pipeline._voiced_rms(first_audio)
        raw_second_rms = pipeline._voiced_rms(pipeline.model.second_audio)
        matched_second_rms = pipeline._voiced_rms(second_audio)
        self.assertGreater(raw_second_rms, first_rms)
        self.assertLess(matched_second_rms, raw_second_rms)
        self.assertLess(abs(matched_second_rms - first_rms), abs(raw_second_rms - first_rms))


if __name__ == "__main__":
    unittest.main()
