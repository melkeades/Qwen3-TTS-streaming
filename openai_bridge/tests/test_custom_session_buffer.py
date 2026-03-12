from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

from openai_bridge.custom_server import CustomBridgeRuntime
from openai_bridge.schemas import SpeechSynthesisParams


def _make_runtime() -> CustomBridgeRuntime:
    return CustomBridgeRuntime(
        config=SimpleNamespace(),
        pipeline=SimpleNamespace(clear_continuation_session=lambda _sid: 0),
        _active={},
        _buffered_sessions={},
        _lock=Lock(),
    )


def _make_req(text: str, *, session_id: str, end_of_message: bool = False) -> SpeechSynthesisParams:
    return SpeechSynthesisParams(
        model="output/test",
        input=text,
        voice="p3",
        speaker="p3",
        response_format="pcm",
        speed=1.0,
        language="English",
        instructions="neutral",
        session_id=session_id,
        end_of_message=end_of_message,
    )


def test_first_chunk_is_ready_immediately() -> None:
    runtime = _make_runtime()
    runtime.upsert_buffered_session(_make_req("Hello there", session_id="s1"))

    session, segments, close_after_stream = runtime.plan_session_segments(
        "s1",
        end_of_message=False,
    )

    assert session is not None
    assert segments == ["Hello there"]
    assert close_after_stream is False
    assert session.generated_segments == 0
    assert session.buffered_chars == 0


def test_followup_uses_sentence_boundaries_until_end_of_message() -> None:
    runtime = _make_runtime()
    runtime.upsert_buffered_session(_make_req("Hello there", session_id="s1"))
    runtime.plan_session_segments("s1", end_of_message=False)
    runtime.mark_session_segments_generated("s1", 1)

    runtime.upsert_buffered_session(
        _make_req(" Second sentence. Third sentence without end", session_id="s1")
    )

    session, segments, close_after_stream = runtime.plan_session_segments(
        "s1",
        end_of_message=False,
    )

    assert session is not None
    assert segments == ["Second sentence."]
    assert close_after_stream is False
    assert session.pending_text == " Third sentence without end"

    session, segments, close_after_stream = runtime.plan_session_segments(
        "s1",
        end_of_message=True,
    )

    assert session is not None
    assert segments == ["Third sentence without end"]
    assert close_after_stream is True


def test_build_request_uses_backend_continuation_defaults() -> None:
    runtime = _make_runtime()
    buffered = runtime.upsert_buffered_session(_make_req("Hello there", session_id="s1"))

    seg_req = buffered.build_request(
        _make_req("ignored", session_id="s1"),
        text="Hello there",
        continuation_reset=True,
    )

    assert seg_req.input == "Hello there"
    assert seg_req.continuation_id == "s1"
    assert seg_req.continuation_mode == "acoustic_tail"
    assert seg_req.continuation_reset is True
    assert seg_req.session_id is None
    assert seg_req.end_of_message is False


def test_buffered_session_rejects_conflicting_settings() -> None:
    runtime = _make_runtime()
    runtime.upsert_buffered_session(_make_req("Hello", session_id="s1"))

    conflict = _make_req(" world.", session_id="s1")
    conflict = conflict.model_copy(update={"instructions": "whisper"})

    try:
        runtime.upsert_buffered_session(conflict)
    except ValueError as exc:
        assert "conflicting synthesis fields" in str(exc)
    else:
        raise AssertionError("Expected conflicting buffered session settings to fail")


def test_clear_buffered_session_reports_pending_content() -> None:
    runtime = _make_runtime()
    runtime.upsert_buffered_session(_make_req("Hello", session_id="s1"))

    cleared = runtime.clear_buffered_session("s1")

    assert cleared.cleared is True
    assert cleared.dropped_chunks == 1
    assert cleared.dropped_chars == len("Hello")
