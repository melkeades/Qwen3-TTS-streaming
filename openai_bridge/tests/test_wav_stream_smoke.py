from __future__ import annotations

import argparse
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser(description="WAV streaming smoke test")
    ap.add_argument("--base-url", default="http://localhost:8030")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--voice", default="kuklina")
    ap.add_argument("--text", default="Привет! Это smoke тест WAV-стриминга через OpenAI-совместимый bridge.")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": "wav",
        "speed": 1.0,
    }

    t0 = time.perf_counter()
    header_ms = None
    first_chunk_ms = None
    total_bytes = 0
    first44 = bytearray()

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
                if len(first44) < 44:
                    need = 44 - len(first44)
                    first44.extend(chunk[:need])

    total_s = time.perf_counter() - t0
    assert total_bytes > 44, f"expected wav header + payload, got {total_bytes} bytes"
    assert len(first44) == 44, "did not receive full WAV header"
    assert bytes(first44[0:4]) == b"RIFF", bytes(first44[0:4])
    assert bytes(first44[8:12]) == b"WAVE", bytes(first44[8:12])

    print("[WAV] status=200")
    print(f"[WAV] ttfb_headers_ms={header_ms:.1f}")
    print(f"[WAV] ttfb_first_chunk_ms={first_chunk_ms:.1f}")
    print(f"[WAV] total_s={total_s:.3f}")
    print(f"[WAV] bytes={total_bytes}")


if __name__ == "__main__":
    main()
