from __future__ import annotations

import argparse
import io
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests


def _write_wav(path: Path, audio: np.ndarray, *, sample_rate: int = 24000) -> None:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        sample_width = wf.getsampwidth()
        channel_count = wf.getnchannels()
        frames = wf.readframes(frame_count)
    if frame_count == 0 and len(data) > 44 and sample_width == 2 and channel_count == 1:
        frames = data[44:]
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return audio, sample_rate


def _find_leading_silence_samples(audio: np.ndarray, *, threshold: float = 0.008) -> int:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size <= 0:
        return 0
    mask = np.abs(samples) >= threshold
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else int(samples.size)


def _find_trailing_silence_samples(audio: np.ndarray, *, threshold: float = 0.008) -> int:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size <= 0:
        return 0
    mask = np.abs(samples) >= threshold
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return int(samples.size)
    return int(samples.size - 1 - idx[-1])


def _seconds(samples: int, sample_rate: int) -> float:
    return float(samples) / float(sample_rate)


def _post_speech(base_url: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(
        f"{base_url}/v1/audio/speech",
        json=payload,
        timeout=300,
    )


def _transcribe(whisper_url: str, wav_path: Path) -> dict[str, Any]:
    with wav_path.open("rb") as fh:
        response = requests.post(
            f"{whisper_url}/v1/audio/transcriptions",
            files={"file": (wav_path.name, fh, "audio/wav")},
            data={
                "response_format": "json",
                "language": "en",
                "vad_filter": "false",
            },
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


@dataclass
class SegmentResult:
    label: str
    audio: np.ndarray
    sample_rate: int
    leading_silence_samples: int
    trailing_silence_samples: int
    response_status: int


def _request_segment(base_url: str, payload: dict[str, Any], *, label: str) -> SegmentResult | None:
    response = _post_speech(base_url, payload)
    if response.status_code == 202:
        return None
    response.raise_for_status()
    audio, sample_rate = _read_wav_bytes(response.content)
    return SegmentResult(
        label=label,
        audio=audio,
        sample_rate=sample_rate,
        leading_silence_samples=_find_leading_silence_samples(audio),
        trailing_silence_samples=_find_trailing_silence_samples(audio),
        response_status=response.status_code,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an actual backend continuation check")
    parser.add_argument("--base-url", default="http://127.0.0.1:8044")
    parser.add_argument("--whisper-url", default="http://127.0.0.1:5000")
    parser.add_argument("--out-dir", default="artifacts/actual_continuation_check")
    parser.add_argument("--model", default="output/test")
    parser.add_argument("--voice", default="p3")
    parser.add_argument("--speaker", default="p3")
    parser.add_argument(
        "--text",
        default=(
            "The bridge should start speaking immediately. "
            "Then the second sentence should continue without dropping words. "
            "Finally the last sentence should arrive without a dead pause at the boundary."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences = [s.strip() + "." for s in args.text.split(".") if s.strip()]
    if len(sentences) < 3:
        raise SystemExit("Need at least three sentences in --text")

    base_payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "speaker": args.speaker,
        "language": "English",
        "response_format": "wav",
        "speed": 1.0,
    }

    ref_resp = _post_speech(args.base_url, base_payload)
    ref_resp.raise_for_status()
    ref_audio, ref_sr = _read_wav_bytes(ref_resp.content)
    ref_path = out_dir / "reference_full.wav"
    _write_wav(ref_path, ref_audio, sample_rate=ref_sr)

    session_id = "actual-check-session"
    segment_results: list[SegmentResult] = []
    for idx, sentence in enumerate(sentences, start=1):
        payload = dict(base_payload)
        payload["input"] = sentence if idx == 1 else f" {sentence}"
        payload["session_id"] = session_id
        payload["end_of_message"] = idx == len(sentences)
        result = _request_segment(args.base_url, payload, label=f"chunk_{idx}")
        if result is not None:
            segment_results.append(result)

    if not segment_results:
        raise SystemExit("No audio returned from streamed session")

    stream_sr = segment_results[0].sample_rate
    stream_audio = np.concatenate([result.audio for result in segment_results], axis=0)
    stream_path = out_dir / "streamed_session.wav"
    _write_wav(stream_path, stream_audio, sample_rate=stream_sr)

    chunk_paths: list[Path] = []
    for result in segment_results:
        chunk_path = out_dir / f"{result.label}.wav"
        _write_wav(chunk_path, result.audio, sample_rate=result.sample_rate)
        chunk_paths.append(chunk_path)

    ref_tx = _transcribe(args.whisper_url, ref_path)
    stream_tx = _transcribe(args.whisper_url, stream_path)

    boundary_report: list[dict[str, Any]] = []
    for idx, result in enumerate(segment_results, start=1):
        boundary_report.append(
            {
                "label": result.label,
                "status": result.response_status,
                "duration_sec": round(_seconds(int(result.audio.size), result.sample_rate), 3),
                "leading_silence_ms": round(
                    _seconds(result.leading_silence_samples, result.sample_rate) * 1000.0, 1
                ),
                "trailing_silence_ms": round(
                    _seconds(result.trailing_silence_samples, result.sample_rate) * 1000.0, 1
                ),
            }
        )

    print(
        json.dumps(
            {
                "reference_path": str(ref_path),
                "stream_path": str(stream_path),
                "chunk_paths": [str(path) for path in chunk_paths],
                "reference_transcript": ref_tx.get("text", ""),
                "stream_transcript": stream_tx.get("text", ""),
                "reference_duration_sec": round(_seconds(int(ref_audio.size), ref_sr), 3),
                "stream_duration_sec": round(_seconds(int(stream_audio.size), stream_sr), 3),
                "boundaries": boundary_report,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
