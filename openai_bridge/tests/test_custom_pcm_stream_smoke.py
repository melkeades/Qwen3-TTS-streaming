from __future__ import annotations

import argparse
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser(description="CustomVoice PCM streaming smoke test")
    ap.add_argument("--base-url", default="http://localhost:8040")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--instruct", default="")
    ap.add_argument("--text", default="Hello. This is a custom voice PCM smoke test.")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")

    with httpx.Client(timeout=60.0) as c:
        models_resp = c.get(f"{base_url}/v1/models")
        assert models_resp.status_code == 200, models_resp.text
        model_ids = [m.get("id") for m in models_resp.json().get("data", [])]
        assert args.model in model_ids, model_ids

    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.speaker,
        "speaker": args.speaker,
        "language": args.language,
        "instruct": args.instruct,
        "response_format": "pcm",
        "speed": 1.0,
    }

    t0 = time.perf_counter()
    header_ms = None
    first_chunk_ms = None
    total_bytes = 0

    with httpx.Client(timeout=120.0) as c:
        with c.stream("POST", f"{base_url}/v1/audio/speech", json=payload) as resp:
            header_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200, resp.text
            for chunk in resp.iter_bytes():
                if not chunk:
                    continue
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - t0) * 1000.0
                total_bytes += len(chunk)

    total_s = time.perf_counter() - t0
    assert total_bytes > 0, "no PCM bytes received"
    assert first_chunk_ms is not None, "no first chunk timing"

    print("[CUSTOM PCM] status=200")
    print(f"[CUSTOM PCM] ttfb_headers_ms={header_ms:.1f}")
    print(f"[CUSTOM PCM] ttfb_first_chunk_ms={first_chunk_ms:.1f}")
    print(f"[CUSTOM PCM] total_s={total_s:.3f}")
    print(f"[CUSTOM PCM] bytes={total_bytes}")


if __name__ == "__main__":
    main()

