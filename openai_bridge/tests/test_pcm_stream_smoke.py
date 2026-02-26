from __future__ import annotations

import argparse
import threading
import time

import httpx


def run_stop_sanity(base_url: str, model: str, voice: str) -> None:
    long_text = ("Это длинный тест для проверки остановки потока. " * 40).strip()
    payload = {
        "model": model,
        "input": long_text,
        "voice": voice,
        "response_format": "pcm",
        "speed": 1.0,
    }

    result = {"bytes": 0, "done": False, "error": None}

    def _reader() -> None:
        try:
            with httpx.Client(timeout=120.0) as c:
                with c.stream("POST", f"{base_url}/v1/audio/speech", json=payload) as resp:
                    if resp.status_code != 200:
                        result["error"] = f"speech status={resp.status_code} body={resp.text}"
                        return
                    for chunk in resp.iter_bytes():
                        if chunk:
                            result["bytes"] += len(chunk)
            result["done"] = True
        except Exception as exc:  # pragma: no cover - smoke script
            result["error"] = str(exc)

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    time.sleep(1.0)

    with httpx.Client(timeout=20.0) as c:
        stop_resp = c.post(f"{base_url}/v1/audio/stop")
        assert stop_resp.status_code == 200, stop_resp.text
        data = stop_resp.json()
        assert data.get("stopped") is True, data

    th.join(timeout=20.0)
    assert not th.is_alive(), "stream reader thread did not exit after /v1/audio/stop"
    assert result["error"] is None, result["error"]

    print(
        "[STOP] active_before=",
        data.get("active_before"),
        "bytes_before_stop=",
        result["bytes"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="PCM streaming smoke test")
    ap.add_argument("--base-url", default="http://localhost:8030")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--voice", default="kuklina")
    ap.add_argument("--text", default="Привет! Это smoke тест PCM-стриминга через OpenAI-совместимый bridge.")
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
        "voice": args.voice,
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

    print("[PCM] status=200")
    print(f"[PCM] ttfb_headers_ms={header_ms:.1f}")
    print(f"[PCM] ttfb_first_chunk_ms={first_chunk_ms:.1f}")
    print(f"[PCM] total_s={total_s:.3f}")
    print(f"[PCM] bytes={total_bytes}")

    run_stop_sanity(base_url=base_url, model=args.model, voice=args.voice)


if __name__ == "__main__":
    main()
