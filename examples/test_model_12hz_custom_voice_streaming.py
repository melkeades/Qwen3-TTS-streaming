# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test streaming CustomVoice with torch.compile/CUDA-graph optimizations.

This script compares:
1. Standard (non-streaming) CustomVoice generation
2. Streaming CustomVoice without optimizations
3. Streaming CustomVoice with torch.compile optimizations

Usage:
    python examples/test_model_12hz_custom_voice_streaming.py
"""

import time

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision("high")


def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


def run_streaming_test(
    model,
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
    label: str = "streaming",
):
    start = time.time()
    chunks = []
    chunk_sizes = []
    first_chunk_time = None
    chunk_count = 0
    sample_rate = 24000

    for chunk, chunk_sr in model.stream_generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
        emit_every_frames=emit_every_frames,
        decode_window_frames=decode_window_frames,
        overlap_samples=0,
    ):
        chunk_count += 1
        chunks.append(chunk)
        chunk_sizes.append(len(chunk))
        sample_rate = chunk_sr
        if first_chunk_time is None:
            first_chunk_time = time.time() - start

    total_time = time.time() - start
    final_audio = np.concatenate(chunks) if chunks else np.array([])
    audio_duration = len(final_audio) / sample_rate if sample_rate > 0 else 0
    avg_chunk_samples = np.mean(chunk_sizes) if chunk_sizes else 0
    avg_chunk_duration = avg_chunk_samples / sample_rate if sample_rate > 0 else 0

    return {
        "label": label,
        "first_chunk_time": first_chunk_time,
        "total_time": total_time,
        "chunk_count": chunk_count,
        "audio": final_audio,
        "sample_rate": sample_rate,
        "audio_duration": audio_duration,
        "avg_chunk_samples": avg_chunk_samples,
        "avg_chunk_duration": avg_chunk_duration,
    }


def main():
    total_start = time.time()

    # Streaming parameters - KEEP THESE CONSISTENT with test_streaming_optimized.py
    EMIT_EVERY = 4
    DECODE_WINDOW = 80

    device = "cuda:0"
    model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    start = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    log_time(start, "Model loaded")

    # text = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
    # language = "Chinese"
    text = "And you know what this cracks me up I voted for you. Yes, I gave you my vote, my sacred democratic trust. And you know what you did? You pissed all over it! Ah what the hell, guys like you, you piss all over everything. You piss all over the country, you piss all over yourselves, you piss all over me. Yeah, yeah I know: Sam, don’t say it! You’re my main man. Guys like you, you’re the backbone of the nation. Shut up dick! I’m talkin’ now!! Alright!! I’M TALKIN’ AND YOU’RE LISTENIN’!"
    language = "English"
    speaker = "Ryan"
    instruct = "very angry, livid"

    results = []

    # ============== Test 1: Standard generation ==============
    print("\n" + "=" * 60)
    print("Test 1: Standard (non-streaming) generation")
    print("=" * 60)

    start = time.time()
    wavs, sr = tts.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )
    standard_time = time.time() - start
    standard_audio_duration = len(wavs[0]) / sr
    standard_rtf = (
        standard_time / standard_audio_duration if standard_audio_duration > 0 else 0
    )
    print(f"[{standard_time:.2f}s] Standard generation complete")
    print(f"Audio duration: {standard_audio_duration:.2f}s, RTF: {standard_rtf:.2f}")
    sf.write("qwen3_tts_custom_standard.wav", wavs[0], sr)
    results.append(
        {
            "label": "standard",
            "total_time": standard_time,
            "audio_duration": standard_audio_duration,
        }
    )

    # ============== Test 2: Streaming WITHOUT optimizations ==============
    print("\n" + "=" * 60)
    print("Test 2: Streaming WITHOUT optimizations")
    print("=" * 60)

    baseline_result = run_streaming_test(
        tts,
        text,
        language,
        speaker,
        instruct,
        emit_every_frames=EMIT_EVERY,
        decode_window_frames=DECODE_WINDOW,
        label="streaming_baseline",
    )
    results.append(baseline_result)
    sf.write(
        "qwen3_tts_custom_streaming_baseline.wav",
        baseline_result["audio"],
        baseline_result["sample_rate"],
    )
    baseline_rtf = (
        baseline_result["total_time"] / baseline_result["audio_duration"]
        if baseline_result["audio_duration"] > 0
        else 0
    )
    print(
        f"First chunk: {baseline_result['first_chunk_time']:.2f}s, Total: {baseline_result['total_time']:.2f}s, Chunks: {baseline_result['chunk_count']}"
    )
    print(
        f"Audio duration: {baseline_result['audio_duration']:.2f}s, Chunk duration: {baseline_result['avg_chunk_duration']*1000:.0f}ms, RTF: {baseline_rtf:.2f}"
    )

    # ============== Test 3: Streaming WITH optimizations ==============
    print("\n" + "=" * 60)
    print("Test 3: Streaming WITH torch.compile (decoder + talker + codebook)")
    print("=" * 60)

    print("\nEnabling streaming optimizations...")
    tts.enable_streaming_optimizations(
        decode_window_frames=DECODE_WINDOW,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="max-autotune",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

    warmup_texts = [
        "这是第一个预热句子。",
        "这是第二个预热句子，用来触发编译。",
        "第三个预热句子，确保各组件都完成预热。",
    ]

    print("\nWarmup runs (compilation happens here)...")
    for i, warmup_text in enumerate(warmup_texts, 1):
        warmup_result = run_streaming_test(
            tts,
            warmup_text,
            language,
            speaker,
            instruct,
            emit_every_frames=EMIT_EVERY,
            decode_window_frames=DECODE_WINDOW,
            label=f"warmup_{i}",
        )
        warmup_rtf = (
            warmup_result["total_time"] / warmup_result["audio_duration"]
            if warmup_result["audio_duration"] > 0
            else 0
        )
        print(
            f"  Warmup {i}: {warmup_result['total_time']:.2f}s, Audio: {warmup_result['audio_duration']:.2f}s, RTF: {warmup_rtf:.2f}"
        )

    test_texts = [
        (
            "And you know what this cracks me up I voted for you. Yes, I gave you my vote, my sacred democratic trust. And you know what you did? You pissed all over it! Ah what the hell, guys like you, you piss all over everything. You piss all over the country, you piss all over yourselves, you piss all over me. Yeah, yeah I know: Sam, don’t say it! You’re my main man. Guys like you, you’re the backbone of the nation. Shut up dick! I’m talkin’ now!! Alright!! I’M TALKIN’ AND YOU’RE LISTENIN’!",
            "short",
        ),
        (
            "And you know what this cracks me up I voted for you. Yes, I gave you my vote, my sacred democratic trust. And you know what you did? You pissed all over it! Ah what the hell, guys like you, you piss all over everything. You piss all over the country, you piss all over yourselves, you piss all over me. Yeah, yeah I know: Sam, don’t say it! You’re my main man. Guys like you, you’re the backbone of the nation. Shut up dick! I’m talkin’ now!! Alright!! I’M TALKIN’ AND YOU’RE LISTENIN’!",
            "medium",
        ),
        (
            "And you know what this cracks me up I voted for you. Yes, I gave you my vote, my sacred democratic trust. And you know what you did? You pissed all over it! Ah what the hell, guys like you, you piss all over everything. You piss all over the country, you piss all over yourselves, you piss all over me. Yeah, yeah I know: Sam, don’t say it! You’re my main man. Guys like you, you’re the backbone of the nation. Shut up dick! I’m talkin’ now!! Alright!! I’M TALKIN’ AND YOU’RE LISTENIN’!",
            "tiny",
        ),
        (
            "And you know what this cracks me up I voted for you. Yes, I gave you my vote, my sacred democratic trust. And you know what you did? You pissed all over it! Ah what the hell, guys like you, you piss all over everything. You piss all over the country, you piss all over yourselves, you piss all over me. Yeah, yeah I know: Sam, don’t say it! You’re my main man. Guys like you, you’re the backbone of the nation. Shut up dick! I’m talkin’ now!! Alright!! I’M TALKIN’ AND YOU’RE LISTENIN’!",
            "short_repeat",
        ),
    ]

    print("\nOptimized test runs...")
    for i, (test_text, text_label) in enumerate(test_texts, 1):
        result = run_streaming_test(
            tts,
            test_text,
            language,
            speaker,
            instruct,
            emit_every_frames=EMIT_EVERY,
            decode_window_frames=DECODE_WINDOW,
            label=f"optimized_{text_label}",
        )
        results.append(result)
        rtf = (
            result["total_time"] / result["audio_duration"]
            if result["audio_duration"] > 0
            else 0
        )
        print(
            f"  Run {i} ({text_label}): First chunk: {result['first_chunk_time']:.2f}s, Total: {result['total_time']:.2f}s, Audio: {result['audio_duration']:.2f}s, RTF: {rtf:.2f}"
        )

        if i == 1:
            sf.write(
                "qwen3_tts_custom_streaming_optimized.wav",
                result["audio"],
                result["sample_rate"],
            )

    # ============== Summary ==============
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"\n{'Method':<25} {'1st Chunk':>10} {'Total':>8} {'Audio':>8} {'RTF':>6} {'Chunks':>7} {'RTF Speedup':>12}"
    )
    print("-" * 80)

    std = results[0]
    std_rtf = (
        std["total_time"] / std["audio_duration"] if std["audio_duration"] > 0 else 0
    )
    print(
        f"{'Standard (no streaming)':<25} {'N/A':>10} {std['total_time']:>7.2f}s {std['audio_duration']:>7.2f}s {std_rtf:>6.2f} {'N/A':>7} {'N/A':>12}"
    )

    for r in results[1:]:
        first = r.get("first_chunk_time", 0)
        total = r["total_time"]
        audio_dur = r.get("audio_duration", 0)
        rtf = total / audio_dur if audio_dur > 0 else 0
        chunks = r.get("chunk_count", 0)
        speedup_rtf = baseline_rtf / rtf if rtf > 0 else 0
        print(
            f"{r['label']:<25} {first:>9.2f}s {total:>7.2f}s {audio_dur:>7.2f}s {rtf:>6.2f} {chunks:>7} {speedup_rtf:>11.2f}x"
        )

    print(f"\n[{time.time() - total_start:.2f}s] TOTAL SCRIPT TIME")


if __name__ == "__main__":
    main()
