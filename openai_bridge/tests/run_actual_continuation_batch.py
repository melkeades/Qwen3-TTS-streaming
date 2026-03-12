from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .run_actual_continuation_check import (
    _find_leading_silence_samples,
    _post_speech,
    _read_wav_bytes,
    _seconds,
    _transcribe,
    _write_wav,
)


def _normalize_text(text: str) -> str:
    raw = (text or "").lower().strip()
    raw = raw.replace("follow-up", "follow up")
    raw = raw.replace("back end", "backend")
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _make_cases() -> list[str]:
    return [
        "The bridge should start speaking immediately. Then the second sentence should continue without dropping words. Finally the last sentence should arrive without a dead pause at the boundary.",
        "We should hear the first sentence right away. The follow up should keep the same voice without swallowing the opening words. The final sentence should still appear when the message ends.",
        "This is a streaming continuation check. The middle chunk should not jump in loudness when it begins. The last chunk should still sound connected to the earlier speech.",
        "Start the first sentence without waiting around. Keep the next sentence smooth and avoid clipping any opening words. Finish the final sentence without losing the tail.",
        "A user should hear the first response quickly. The next sentence should not reset into a different delivery. The last sentence should be present and intelligible.",
        "Say the first sentence right away. Continue into the next sentence without a loud restart. End with a final sentence that still sounds like the same speaker.",
        "This backend should preserve continuity. The second sentence should not come in much louder than the first one. The closing sentence should still be there at the end.",
        "The first chunk should begin normally. The second chunk should not pause for too long before speech starts. The third chunk should not disappear.",
        "Immediate speech matters for the opening sentence. Consistent volume matters for the following sentence. Complete delivery matters for the closing sentence.",
        "Start talking on the first sentence. Avoid a harsh loudness step on the second sentence. Keep the third sentence audible and complete.",
        "The first sentence should come through fast. The second sentence should sound like a continuation instead of a reset. The third sentence should remain intact.",
        "Speak the first line without delay. Carry the same vocal feel into the next line. Keep the final line present when the session ends.",
    ]


def _sentence_chunks(text: str) -> list[str]:
    return [s.strip() + "." for s in text.split(".") if s.strip()]


def _voiced_rms(audio: np.ndarray, *, threshold: float = 0.01) -> float:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    mask = np.abs(samples) >= threshold
    if not np.any(mask):
        return 0.0
    voiced = samples[mask]
    return float(np.sqrt(np.mean(voiced * voiced)))


def _first_voiced_rms(audio: np.ndarray, sample_rate: int, *, threshold: float = 0.01) -> float:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    mask = np.abs(samples) >= threshold
    idx = np.flatnonzero(mask)
    if idx.size <= 0:
        return 0.0
    start = int(idx[0])
    stop = min(int(samples.size), start + max(512, sample_rate // 2))
    segment = samples[start:stop]
    return _voiced_rms(segment, threshold=threshold)


def _run_case(
    *,
    base_url: str,
    whisper_url: str,
    out_dir: Path,
    model: str,
    voice: str,
    speaker: str,
    text: str,
    case_id: int,
) -> dict[str, Any]:
    case_dir = out_dir / f"case_{case_id:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    base_payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "speaker": speaker,
        "language": "English",
        "response_format": "wav",
        "speed": 1.0,
    }

    ref_resp = _post_speech(base_url, base_payload)
    ref_resp.raise_for_status()
    ref_audio, ref_sr = _read_wav_bytes(ref_resp.content)
    ref_path = case_dir / "reference_full.wav"
    _write_wav(ref_path, ref_audio, sample_rate=ref_sr)

    session_id = f"batch-case-{case_id}"
    streamed_segments: list[np.ndarray] = []
    boundary_rows: list[dict[str, Any]] = []
    prev_voiced_rms = 0.0

    for idx, sentence in enumerate(_sentence_chunks(text), start=1):
        payload = dict(base_payload)
        payload["input"] = sentence if idx == 1 else f" {sentence}"
        payload["session_id"] = session_id
        payload["end_of_message"] = idx == len(_sentence_chunks(text))
        response = _post_speech(base_url, payload)
        if response.status_code == 202:
            boundary_rows.append(
                {
                    "chunk": idx,
                    "status": 202,
                    "duration_sec": 0.0,
                    "leading_silence_ms": 0.0,
                    "voiced_rms": 0.0,
                    "first_voiced_rms": 0.0,
                    "head_vs_prev_voiced_ratio": None,
                }
            )
            continue
        response.raise_for_status()
        audio, sample_rate = _read_wav_bytes(response.content)
        streamed_segments.append(audio)
        chunk_path = case_dir / f"chunk_{idx}.wav"
        _write_wav(chunk_path, audio, sample_rate=sample_rate)
        voiced_rms = _voiced_rms(audio)
        first_voiced_rms = _first_voiced_rms(audio, sample_rate)
        boundary_rows.append(
            {
                "chunk": idx,
                "status": response.status_code,
                "duration_sec": round(_seconds(int(audio.size), sample_rate), 3),
                "leading_silence_ms": round(
                    _seconds(_find_leading_silence_samples(audio), sample_rate) * 1000.0,
                    1,
                ),
                "voiced_rms": round(voiced_rms, 4),
                "first_voiced_rms": round(first_voiced_rms, 4),
                "head_vs_prev_voiced_ratio": (
                    round(first_voiced_rms / prev_voiced_rms, 3)
                    if prev_voiced_rms > 1e-6
                    else None
                ),
            }
        )
        prev_voiced_rms = voiced_rms

    stream_audio = (
        np.concatenate(streamed_segments, axis=0).astype(np.float32, copy=False)
        if streamed_segments
        else np.zeros((0,), dtype=np.float32)
    )
    stream_path = case_dir / "streamed_session.wav"
    _write_wav(stream_path, stream_audio, sample_rate=ref_sr)

    ref_tx = _transcribe(whisper_url, ref_path)
    stream_tx = _transcribe(whisper_url, stream_path)

    norm_expected = _normalize_text(text)
    norm_stream = _normalize_text(stream_tx.get("text", ""))
    transcript_match = norm_expected == norm_stream
    missing_final = _normalize_text(_sentence_chunks(text)[-1]) not in norm_stream
    ratio_rows = [
        row["head_vs_prev_voiced_ratio"]
        for row in boundary_rows
        if row["head_vs_prev_voiced_ratio"] is not None
    ]
    max_ratio = max(ratio_rows) if ratio_rows else 1.0
    min_ratio = min(ratio_rows) if ratio_rows else 1.0

    return {
        "case_id": case_id,
        "text": text,
        "reference_transcript": ref_tx.get("text", ""),
        "stream_transcript": stream_tx.get("text", ""),
        "reference_duration_sec": round(_seconds(int(ref_audio.size), ref_sr), 3),
        "stream_duration_sec": round(_seconds(int(stream_audio.size), ref_sr), 3),
        "transcript_match": transcript_match,
        "missing_final_sentence": missing_final,
        "max_head_ratio": round(max_ratio, 3),
        "min_head_ratio": round(min_ratio, 3),
        "boundaries": boundary_rows,
        "artifact_dir": str(case_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple live continuation checks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8044")
    parser.add_argument("--whisper-url", default="http://127.0.0.1:5000")
    parser.add_argument("--out-dir", default="artifacts/actual_continuation_batch")
    parser.add_argument("--model", default="output/test")
    parser.add_argument("--voice", default="p3")
    parser.add_argument("--speaker", default="p3")
    parser.add_argument("--cases", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _make_cases()[: max(1, int(args.cases))]
    results = [
        _run_case(
            base_url=args.base_url,
            whisper_url=args.whisper_url,
            out_dir=out_dir,
            model=args.model,
            voice=args.voice,
            speaker=args.speaker,
            text=text,
            case_id=index,
        )
        for index, text in enumerate(cases, start=1)
    ]

    transcript_failures = [row for row in results if not row["transcript_match"]]
    missing_final = [row for row in results if row["missing_final_sentence"]]
    max_ratio = max(row["max_head_ratio"] for row in results) if results else 1.0
    min_ratio = min(row["min_head_ratio"] for row in results) if results else 1.0
    mean_ratio = (
        round(
            float(
                np.mean(
                    [
                        boundary["head_vs_prev_voiced_ratio"]
                        for row in results
                        for boundary in row["boundaries"]
                        if boundary["head_vs_prev_voiced_ratio"] is not None
                    ]
                )
            ),
            3,
        )
        if results
        else 1.0
    )

    summary = {
        "cases": len(results),
        "transcript_failures": len(transcript_failures),
        "missing_final_sentence_cases": len(missing_final),
        "max_head_ratio": max_ratio,
        "min_head_ratio": min_ratio,
        "mean_head_ratio": mean_ratio,
        "results": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
