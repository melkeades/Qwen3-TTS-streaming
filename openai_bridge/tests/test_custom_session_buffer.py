from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

from openai_bridge.custom_server import CustomBridgeRuntime
from openai_bridge.schemas import SpeechSynthesisParams


def _make_runtime() -> CustomBridgeRuntime:
    return CustomBridgeRuntime(
        config=SimpleNamespace(),
        pipeline=SimpleNamespace(),
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


def test_buffered_session_accumulates_and_builds_final_request() -> None:
    runtime = _make_runtime()

    runtime.upsert_buffered_session(_make_req("Hello", session_id="s1"))
    buffered = runtime.upsert_buffered_session(_make_req(" world.", session_id="s1"))

    assert buffered.buffered_chunks == 2
    assert buffered.buffered_chars == len("Hello world.")

    committed = runtime.pop_buffered_session("s1")
    assert committed is not None

    final_req = committed.build_request(
        _make_req("unused", session_id="ignored", end_of_message=True)
    )
    assert final_req.input == "Hello world."
    assert final_req.session_id is None
    assert final_req.end_of_message is False


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


def test_clear_buffered_session_reports_dropped_content() -> None:
    runtime = _make_runtime()
    runtime.upsert_buffered_session(_make_req("Hello", session_id="s1"))
    runtime.upsert_buffered_session(_make_req(" world.", session_id="s1"))

    cleared = runtime.clear_buffered_session("s1")

    assert cleared.cleared is True
    assert cleared.dropped_chunks == 2
    assert cleared.dropped_chars == len("Hello world.")
    assert runtime.buffered_session_count() == 0


def test_buffered_session_normalizes_speaker_and_instruction_aliases() -> None:
    runtime = _make_runtime()

    first = _make_req("Hello", session_id="s1")
    first = first.model_copy(update={"speaker": None, "instructions": None, "instruct": "neutral"})
    runtime.upsert_buffered_session(first)

    second = _make_req(" world.", session_id="s1")
    second = second.model_copy(update={"speaker": "p3", "instructions": "neutral", "instruct": None})

    buffered = runtime.upsert_buffered_session(second)
    assert buffered.buffered_chunks == 2
