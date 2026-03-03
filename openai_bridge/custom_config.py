from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CustomBridgeConfig:
    repo_root: Path
    host: str
    port: int
    cors_allow_origins: list[str]
    cors_allow_methods: list[str]
    cors_allow_headers: list[str]
    cors_allow_credentials: bool

    model_id: str
    fallback_model_id: str
    fallback_speaker: str
    device_map: str
    dtype: str
    attn_implementation: str

    default_language: str
    default_speaker: str
    default_instruct: str

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
    stream_use_optimized_decode: bool

    warmup_enabled: bool
    warmup_runs: int
    warmup_max_frames: int
    warmup_text: str
    warmup_language: str
    warmup_speaker: str
    warmup_instruct: str

    model_scan_roots: list[Path]
    model_scan_max_depth: int
    additional_model_ids: list[str]
    startup_empty: bool

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

    @staticmethod
    def _resolve_paths(raw_paths: list[str], repo_root: Path) -> list[Path]:
        resolved: list[Path] = []
        for value in raw_paths:
            p = Path(value)
            if not p.is_absolute():
                p = repo_root / p
            resolved.append(p.resolve())
        return resolved

    @classmethod
    def from_env(cls) -> "CustomBridgeConfig":
        repo_root = Path(__file__).resolve().parent.parent

        client_html_path = Path(
            os.getenv("CUSTOM_BRIDGE_CLIENT_HTML_PATH", "openai_bridge/client_custom_live.html")
        )
        if not client_html_path.is_absolute():
            client_html_path = repo_root / client_html_path

        model_scan_roots = cls._resolve_paths(
            cls._env_csv("CUSTOM_BRIDGE_MODEL_SCAN_ROOTS", "output"),
            repo_root=repo_root,
        )

        default_speaker = os.getenv("CUSTOM_BRIDGE_DEFAULT_SPEAKER", "p3")
        warmup_speaker = os.getenv("CUSTOM_BRIDGE_WARMUP_SPEAKER", default_speaker)
        fallback_speaker = os.getenv("CUSTOM_BRIDGE_FALLBACK_SPEAKER", "Ryan")

        return cls(
            repo_root=repo_root,
            host=os.getenv("CUSTOM_BRIDGE_HOST", "0.0.0.0"),
            port=int(os.getenv("CUSTOM_BRIDGE_PORT", "8040")),
            cors_allow_origins=cls._env_csv("CUSTOM_BRIDGE_CORS_ALLOW_ORIGINS", "*"),
            cors_allow_methods=cls._env_csv("CUSTOM_BRIDGE_CORS_ALLOW_METHODS", "*"),
            cors_allow_headers=cls._env_csv("CUSTOM_BRIDGE_CORS_ALLOW_HEADERS", "*"),
            cors_allow_credentials=cls._env_bool("CUSTOM_BRIDGE_CORS_ALLOW_CREDENTIALS", False),
            model_id=os.getenv("CUSTOM_BRIDGE_MODEL_ID", "output/test"),
            fallback_model_id=os.getenv(
                "CUSTOM_BRIDGE_FALLBACK_MODEL_ID",
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            ),
            fallback_speaker=fallback_speaker,
            device_map=os.getenv("CUSTOM_BRIDGE_DEVICE_MAP", "cuda:0"),
            dtype=os.getenv("CUSTOM_BRIDGE_DTYPE", "bfloat16"),
            attn_implementation=os.getenv("CUSTOM_BRIDGE_ATTN_IMPL", "flash_attention_2"),
            default_language=os.getenv("CUSTOM_BRIDGE_DEFAULT_LANGUAGE", "English"),
            default_speaker=default_speaker,
            default_instruct=os.getenv("CUSTOM_BRIDGE_DEFAULT_INSTRUCT", ""),
            sample_rate=int(os.getenv("CUSTOM_BRIDGE_SAMPLE_RATE", "24000")),
            channels=int(os.getenv("CUSTOM_BRIDGE_CHANNELS", "1")),
            bits_per_sample=int(os.getenv("CUSTOM_BRIDGE_BITS_PER_SAMPLE", "16")),
            emit_every_frames=int(os.getenv("CUSTOM_BRIDGE_EMIT_EVERY_FRAMES", "4")),
            decode_window_frames=int(os.getenv("CUSTOM_BRIDGE_DECODE_WINDOW_FRAMES", "80")),
            overlap_samples=int(os.getenv("CUSTOM_BRIDGE_OVERLAP_SAMPLES", "0")),
            max_frames=int(os.getenv("CUSTOM_BRIDGE_MAX_FRAMES", "10000")),
            optimize_use_compile=cls._env_bool("CUSTOM_BRIDGE_OPT_USE_COMPILE", True),
            optimize_use_cuda_graphs=cls._env_bool("CUSTOM_BRIDGE_OPT_USE_CUDA_GRAPHS", False),
            optimize_compile_mode=os.getenv("CUSTOM_BRIDGE_OPT_COMPILE_MODE", "reduce-overhead"),
            optimize_use_fast_codebook=cls._env_bool("CUSTOM_BRIDGE_OPT_USE_FAST_CODEBOOK", True),
            optimize_compile_codebook_predictor=cls._env_bool(
                "CUSTOM_BRIDGE_OPT_COMPILE_CODEBOOK_PREDICTOR", True
            ),
            optimize_compile_talker=cls._env_bool("CUSTOM_BRIDGE_OPT_COMPILE_TALKER", True),
            stream_use_optimized_decode=cls._env_bool(
                "CUSTOM_BRIDGE_STREAM_USE_OPTIMIZED_DECODE", True
            ),
            warmup_enabled=cls._env_bool("CUSTOM_BRIDGE_WARMUP_ENABLED", True),
            warmup_runs=int(os.getenv("CUSTOM_BRIDGE_WARMUP_RUNS", "3")),
            warmup_max_frames=int(os.getenv("CUSTOM_BRIDGE_WARMUP_MAX_FRAMES", "1024")),
            warmup_text=os.getenv(
                "CUSTOM_BRIDGE_WARMUP_TEXT",
                "Hello. This is a short warmup for custom voice streaming API.",
            ),
            warmup_language=os.getenv("CUSTOM_BRIDGE_WARMUP_LANGUAGE", "English"),
            warmup_speaker=warmup_speaker,
            warmup_instruct=os.getenv("CUSTOM_BRIDGE_WARMUP_INSTRUCT", "neutral and clear"),
            model_scan_roots=model_scan_roots,
            model_scan_max_depth=int(os.getenv("CUSTOM_BRIDGE_MODEL_SCAN_MAX_DEPTH", "3")),
            additional_model_ids=cls._env_csv("CUSTOM_BRIDGE_ADDITIONAL_MODEL_IDS", ""),
            startup_empty=cls._env_bool("CUSTOM_BRIDGE_START_EMPTY", False),
            client_html_path=client_html_path,
        )
