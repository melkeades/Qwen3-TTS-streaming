from __future__ import annotations

import unittest
from threading import Lock

from openai_bridge.custom_config import CustomBridgeConfig
from openai_bridge.custom_pipeline import wav_header
from openai_bridge.custom_server import CustomBridgeRuntime, _live_req_from_payload


class _DummyPipeline:
    pass


class CustomLiveHelperTests(unittest.TestCase):
    def test_live_req_from_payload_maps_bridge_fields(self) -> None:
        req = _live_req_from_payload(
            {
                "model": "output/test",
                "voice": "p3",
                "speaker": "p3",
                "instructions": "calm",
                "language": "English",
                "response_format": "wav",
                "speed": 1.1,
                "emit_every_frames": 6,
                "decode_window_frames": 96,
                "overlap_samples": 240,
                "max_frames": 2048,
                "use_optimized_decode": False,
            }
        )
        self.assertEqual(req.model, "output/test")
        self.assertEqual(req.voice, "p3")
        self.assertEqual(req.speaker, "p3")
        self.assertEqual(req.instructions, "calm")
        self.assertEqual(req.language, "English")
        self.assertEqual(req.response_format, "wav")
        self.assertEqual(req.speed, 1.1)
        self.assertEqual(req.emit_every_frames, 6)
        self.assertEqual(req.decode_window_frames, 96)
        self.assertEqual(req.overlap_samples, 240)
        self.assertEqual(req.max_frames, 2048)
        self.assertFalse(req.use_optimized_decode)
        self.assertEqual(req.input, "live-session")

    def test_runtime_cancel_all_invokes_events_and_cancellers(self) -> None:
        config = CustomBridgeConfig.from_env()
        runtime = CustomBridgeRuntime(
            config=config,
            pipeline=_DummyPipeline(),
            _active={},
            _active_cancellers={},
            _buffered_sessions={},
            _lock=Lock(),
        )
        cancelled: list[str] = []
        stream_id, event = runtime.register_stream(canceler=lambda: cancelled.append("x"))
        self.assertFalse(event.is_set())

        stopped = runtime.cancel_all()

        self.assertEqual(stopped, 1)
        self.assertTrue(event.is_set())
        self.assertEqual(cancelled, ["x"])
        runtime.unregister_stream(stream_id)

    def test_wav_header_helper_still_returns_riff_prefix(self) -> None:
        header = wav_header(sample_rate=24000, bits_per_sample=16, channels=1)
        self.assertEqual(header[:4], b"RIFF")
        self.assertEqual(header[8:12], b"WAVE")


if __name__ == "__main__":
    unittest.main()
