from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoicePreset:
    name: str
    ref_audio: Path
    ref_text: str
    x_vector_only_mode: bool = False
    language: str | None = None


@dataclass(frozen=True)
class VoiceRegistry:
    presets: dict[str, VoicePreset]

    def get(self, voice: str) -> VoicePreset | None:
        return self.presets.get((voice or "").strip())

    def names(self) -> list[str]:
        return sorted(self.presets.keys())


def load_voice_registry(voices_path: Path) -> VoiceRegistry:
    payload = json.loads(voices_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("voices", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Invalid voices format in {voices_path}")

    presets: dict[str, VoicePreset] = {}
    base_dir = voices_path.parent

    for row in rows:
        name = str(row["name"]).strip()
        if not name:
            raise ValueError("Voice preset has empty name")

        ref_audio = Path(str(row["ref_audio"])).expanduser()
        if not ref_audio.is_absolute():
            ref_audio = (base_dir / ref_audio).resolve()
        if not ref_audio.exists():
            raise FileNotFoundError(f"Voice '{name}' ref_audio not found: {ref_audio}")

        ref_text = str(row.get("ref_text", "")).strip()
        if not ref_text and not bool(row.get("x_vector_only_mode", False)):
            raise ValueError(f"Voice '{name}' requires non-empty ref_text when x_vector_only_mode is false")

        preset = VoicePreset(
            name=name,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=bool(row.get("x_vector_only_mode", False)),
            language=(str(row.get("language")).strip() or None) if row.get("language") is not None else None,
        )
        presets[name] = preset

    if not presets:
        raise ValueError(f"No voice presets found in {voices_path}")

    return VoiceRegistry(presets=presets)
