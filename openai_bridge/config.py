from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BridgeConfig:
    host: str
    port: int
    cors_allow_origins: list[str]
    cors_allow_methods: list[str]
    cors_allow_headers: list[str]
    cors_allow_credentials: bool

    model_id: str
    device_map: str
    dtype: str
    attn_implementation: str

    default_language: str

    sample_rate: int
    channels: int
    bits_per_sample: int

    emit_every_frames: int
    decode_window_frames: int
    overlap_samples: int
    max_frames: int

    optimize_use_compile: bool
    optimize_use_cuda_graphs: bool
    optimize_compile_mode: str
    optimize_use_fast_codebook: bool
    optimize_compile_codebook_predictor: bool
    optimize_compile_talker: bool

    warmup_enabled: bool
    warmup_text: str

    voices_path: Path
    client_html_path: Path

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        value = value.strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_csv(name: str, default: str) -> list[str]:
        raw = os.getenv(name, default)
        items = [x.strip() for x in raw.split(",")]
        return [x for x in items if x]

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        repo_root = Path(__file__).resolve().parent.parent

        voices_path = Path(os.getenv("BRIDGE_VOICES_PATH", "openai_bridge/voices.json"))
        if not voices_path.is_absolute():
            voices_path = repo_root / voices_path

        client_html_path = Path(
            os.getenv("BRIDGE_CLIENT_HTML_PATH", "openai_bridge/client_dark_live.html")
        )
        if not client_html_path.is_absolute():
            client_html_path = repo_root / client_html_path

        return cls(
            host=os.getenv("BRIDGE_HOST", "0.0.0.0"),
            port=int(os.getenv("BRIDGE_PORT", "8030")),
            cors_allow_origins=cls._env_csv("BRIDGE_CORS_ALLOW_ORIGINS", "*"),
            cors_allow_methods=cls._env_csv("BRIDGE_CORS_ALLOW_METHODS", "*"),
            cors_allow_headers=cls._env_csv("BRIDGE_CORS_ALLOW_HEADERS", "*"),
            cors_allow_credentials=cls._env_bool("BRIDGE_CORS_ALLOW_CREDENTIALS", False),
            model_id=os.getenv("BRIDGE_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            device_map=os.getenv("BRIDGE_DEVICE_MAP", "cuda:0"),
            dtype=os.getenv("BRIDGE_DTYPE", "bfloat16"),
            attn_implementation=os.getenv("BRIDGE_ATTN_IMPL", "flash_attention_2"),
            default_language=os.getenv("BRIDGE_DEFAULT_LANGUAGE", "Russian"),
            sample_rate=int(os.getenv("BRIDGE_SAMPLE_RATE", "24000")),
            channels=int(os.getenv("BRIDGE_CHANNELS", "1")),
            bits_per_sample=int(os.getenv("BRIDGE_BITS_PER_SAMPLE", "16")),
            emit_every_frames=int(os.getenv("BRIDGE_EMIT_EVERY_FRAMES", "4")),
            decode_window_frames=int(os.getenv("BRIDGE_DECODE_WINDOW_FRAMES", "80")),
            overlap_samples=int(os.getenv("BRIDGE_OVERLAP_SAMPLES", "0")),
            max_frames=int(os.getenv("BRIDGE_MAX_FRAMES", "10000")),
            optimize_use_compile=cls._env_bool("BRIDGE_OPT_USE_COMPILE", True),
            optimize_use_cuda_graphs=cls._env_bool("BRIDGE_OPT_USE_CUDA_GRAPHS", False),
            optimize_compile_mode=os.getenv("BRIDGE_OPT_COMPILE_MODE", "reduce-overhead"),
            optimize_use_fast_codebook=cls._env_bool("BRIDGE_OPT_USE_FAST_CODEBOOK", True),
            optimize_compile_codebook_predictor=cls._env_bool(
                "BRIDGE_OPT_COMPILE_CODEBOOK_PREDICTOR", True
            ),
            optimize_compile_talker=cls._env_bool("BRIDGE_OPT_COMPILE_TALKER", True),
            warmup_enabled=cls._env_bool("BRIDGE_WARMUP_ENABLED", True),
            warmup_text=os.getenv(
                "BRIDGE_WARMUP_TEXT",
                "Привет! Это короткий прогрев для стримингового API.",
            ),
            voices_path=voices_path,
            client_html_path=client_html_path,
        )
