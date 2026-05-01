# Qwen3-TTS seam-local continuity plan for CustomVoice (Transformers backend)

## 1) What we are doing, and how

### Goal

Keep **style continuity at selected seams between segments** for a **CustomVoice** model, specifically:

- loudness continuity
- pitch continuity
- cadence continuity

This is **not** about speaker identity continuity, and it is **not** about globally overriding the segment instruction.

### Core rule

The segment-level `instruct` remains the **main style anchor**.

The extra continuity control is only a **temporary seam bridge** that affects the beginning of a new segment, for a chosen duration such as:

- 1 second
- 2 seconds
- 5 seconds

After that short transition window, the seam bridge fades out and the segment’s normal `instruct` fully dominates again.

### Example

- Segment A instruction: `sad`
- Segment B instruction: `happy`

All chunks inside segment A should remain roughly `sad`.  
All chunks inside segment B should remain roughly `happy`.

Only the seam between them should be manually controlled.

That seam controller can use:

- `loudness_mean`
- `f0_center`
- `speaking_rate`
- and the other low-level state values

to shape **how** the model enters the new `happy` segment from the old `sad` one.

So the continuity path does **not** replace the `happy` instruction. It only shapes the **entry into it**.

### Why this is the correct design

The current CustomVoice path already uses:

- `text`
- `speaker`
- `language`
- optional `instruct`

for its main behavior.

That means the cleanest extension is **not** a second global style controller.  
It is a **residual seam controller** that is only active at selected boundaries and only for a short time window.

### The new mental model

Use two layers:

#### 1. Segment style anchor
This is the normal `instruct`.

It defines the intended style for the whole segment.

#### 2. Seam transition controller
This is only active at selected chunk boundaries.

It controls the transition from the previous segment’s low-level delivery baseline into the new segment’s target style.

### What the seam controller carries

Instead of a global `style_state`, think in terms of a seam-local structure:

```python
transition_state = {
    "prev_style_state": {
        "loudness_mean": ...,
        "loudness_tail_slope": ...,
        "f0_center": ...,
        "f0_range": ...,
        "f0_tail_slope": ...,
        "speaking_rate": ...,
        "pause_ratio": ...,
        "phrase_end_shape": ...,
    },
    "duration_sec": 2.5,
    "mode": "SMOOTH",
    "strength": 1.0,
}
```

### The governing equation

At inference, think of the model conditioning as:

```python
final_conditioning = instruction_conditioning + seam_gate(t) * transition_conditioning
```

Where:

- `instruction_conditioning` comes from `instruct`
- `transition_conditioning` comes from `prev_style_state`
- `seam_gate(t)` starts positive at the seam, then decays to zero over the selected transition window

That is the key design principle.

---

## 2) Changes needed in the training pipeline

## 2.1 Broad pipeline shape

Keep the official shape:

1. `train_raw.jsonl`
2. `prepare_data.py`
3. processed JSONL
4. `sft_12hz.py`

Add one preprocessing step before `prepare_data.py`:

1. source data
2. `build_seam_jsonl.py`
3. seam-aware raw JSONL
4. patched `prepare_data.py`
5. seam-aware processed JSONL
6. patched `sft_12hz.py`

### New script

Create a small script such as:

```text
build_seam_jsonl.py
```

Its job is:

- read your existing source data
- identify segment boundaries
- mark only selected chunks as seam starts
- compute `prev_style_state` from the previous segment tail
- attach seam metadata only where needed

### Two chunk types

#### A. Normal in-segment chunks

These are chunks fully inside one segment.

Fields:
- `audio`
- `text`
- `ref_audio`
- `instruct`
- `segment_id`
- `is_transition_start = 0`

These should use normal training only.

#### B. Seam-start chunks

These are the first chunks of a new segment, or any chunk you explicitly mark as a transition point.

Fields:
- `audio`
- `text`
- `ref_audio`
- `instruct`  ← the new segment’s target instruction
- `segment_id`
- `prev_segment_id`
- `is_transition_start = 1`
- `transition_state`

This is where the seam controller is trained.

### Raw seam-aware JSONL shape

```json
{
  "audio": "segment_B_chunk_000.wav",
  "text": "target text for the first chunk of segment B",
  "ref_audio": "reference.wav",
  "instruct": "happy",
  "segment_id": "B",
  "prev_segment_id": "A",
  "is_transition_start": 1,
  "transition_state": {
    "prev_style_state": {
      "loudness_mean": -0.15,
      "loudness_tail_slope": 0.04,
      "f0_center": 0.22,
      "f0_range": -0.08,
      "f0_tail_slope": -0.11,
      "speaking_rate": 0.05,
      "pause_ratio": 0.18,
      "phrase_end_shape": "WEAK_FALL"
    },
    "duration_sec": 2.5,
    "mode": "SMOOTH",
    "strength": 1.0
  }
}
```

And a normal chunk looks like:

```json
{
  "audio": "segment_B_chunk_001.wav",
  "text": "next chunk in the same happy segment",
  "ref_audio": "reference.wav",
  "instruct": "happy",
  "segment_id": "B",
  "is_transition_start": 0
}
```

That is the important difference: **most chunks do not carry seam control**.

---

## 2.2 Libraries

You can keep the first version inside the official Qwen environment.

Useful libs:

- `torch`
- `torchaudio`
- `soundfile`
- `librosa`
- `transformers`
- `accelerate`

Example imports:

```python
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
```

---

## 2.3 Build the seam-aware JSONL

### Broad logic

For each segment boundary:

1. find the last chunk(s) of the old segment
2. find the first chunk of the new segment
3. compute `prev_style_state` from the old segment tail
4. decide the seam duration
5. write the seam-start record

For all other chunks:
- write normal non-transition records

### Minimal skeleton

```python
def build_records(segments, ref_audio_path):
    records = []

    for seg_idx, segment in enumerate(segments):
        for chunk_idx, chunk in enumerate(segment["chunks"]):
            record = {
                "audio": chunk["audio_path"],
                "text": chunk["text"],
                "ref_audio": ref_audio_path,
                "instruct": segment["instruct"],
                "segment_id": segment["segment_id"],
                "is_transition_start": 0,
            }

            is_first_chunk = chunk_idx == 0
            has_prev_segment = seg_idx > 0

            if is_first_chunk and has_prev_segment:
                prev_segment = segments[seg_idx - 1]
                prev_tail_path = prev_segment["tail_audio_path"]
                prev_state = extract_style_state(prev_tail_path)

                record["prev_segment_id"] = prev_segment["segment_id"]
                record["is_transition_start"] = 1
                record["transition_state"] = {
                    "prev_style_state": prev_state,
                    "duration_sec": 2.5,
                    "mode": "SMOOTH",
                    "strength": 1.0,
                }

            records.append(record)

    return records
```

---

## 2.4 Extract the previous style state

Keep the first version simple and stable.

### Model-facing vector

For the model, convert the state into:

```python
transition_vec = [
    loudness_mean_z,
    loudness_tail_slope_z,
    f0_center_z,
    f0_range_z,
    f0_tail_slope_z,
    speaking_rate_z,
    pause_ratio,
    *phrase_end_onehot,
    duration_sec_norm,
    strength,
    *mode_onehot,
]
```

So the seam controller knows:

- what the old segment ended like
- how long the transition should last
- how strongly to apply it
- what transition mode to use

### Minimal extractor

```python
import librosa
import numpy as np
import soundfile as sf

def extract_style_state(audio_path, sr_target=24000):
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sr_target:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=sr_target)
        sr = sr_target

    frame_length = 1024
    hop_length = 256

    rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))

    f0, voiced_flag, _ = librosa.pyin(
        wav,
        sr=sr,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=frame_length,
        hop_length=hop_length,
    )

    f0_voiced = f0[~np.isnan(f0)]
    if len(f0_voiced) == 0:
        f0_center = 0.0
        f0_range = 0.0
        f0_tail_slope = 0.0
    else:
        logf0 = np.log(f0_voiced)
        f0_center = float(np.median(logf0))
        f0_range = float(np.percentile(logf0, 90) - np.percentile(logf0, 10))

        tail = np.log(np.maximum(f0[-8:], 1e-6))
        tail = tail[~np.isnan(tail)]
        f0_tail_slope = 0.0 if len(tail) < 2 else float(np.polyfit(np.arange(len(tail)), tail, 1)[0])

    loudness_mean = float(np.mean(rms_db))
    loudness_tail = rms_db[-8:] if len(rms_db) >= 8 else rms_db
    loudness_tail_slope = 0.0 if len(loudness_tail) < 2 else float(np.polyfit(np.arange(len(loudness_tail)), loudness_tail, 1)[0])

    speaking_rate = 0.0  # replace with your preferred proxy
    pause_ratio = float(np.mean(rms < np.percentile(rms, 20)))
    phrase_end_shape = classify_phrase_end(f0, rms_db)

    return {
        "loudness_mean": loudness_mean,
        "loudness_tail_slope": loudness_tail_slope,
        "f0_center": f0_center,
        "f0_range": f0_range,
        "f0_tail_slope": f0_tail_slope,
        "speaking_rate": speaking_rate,
        "pause_ratio": pause_ratio,
        "phrase_end_shape": phrase_end_shape,
    }
```

### Phrase-end classifier

```python
def classify_phrase_end(f0, rms_db):
    tail_f0 = f0[-8:]
    tail_f0 = tail_f0[~np.isnan(tail_f0)]
    tail_rms = rms_db[-8:]

    f0_slope = 0.0 if len(tail_f0) < 2 else float(np.polyfit(np.arange(len(tail_f0)), np.log(tail_f0), 1)[0])
    e_slope = 0.0 if len(tail_rms) < 2 else float(np.polyfit(np.arange(len(tail_rms)), tail_rms, 1)[0])

    if f0_slope > 0.01:
        return "RISE"
    if f0_slope < -0.03 and e_slope < -0.1:
        return "STRONG_FALL"
    if f0_slope < -0.01:
        return "WEAK_FALL"
    return "CONTINUE"
```

---

## 2.5 Patch prepare_data.py

Do not rewrite it. Just preserve the new seam metadata.

### What it should do

- keep `is_transition_start`
- keep `transition_state`
- still generate normal `audio_codes`

### Broad patch idea

```python
line = json.loads(raw_line)

final_line = {
    "audio": line["audio"],
    "text": line["text"],
    "ref_audio": line["ref_audio"],
    "audio_codes": encoded_codes,
    "instruct": line.get("instruct"),
    "segment_id": line.get("segment_id"),
    "is_transition_start": line.get("is_transition_start", 0),
}

if "prev_segment_id" in line:
    final_line["prev_segment_id"] = line["prev_segment_id"]

if "transition_state" in line:
    final_line["transition_state"] = line["transition_state"]
```

That is enough for v1.

---

## 2.6 Patch the training dataset loader

The dataset loader should now read:

- normal text/code fields
- `is_transition_start`
- `transition_state` when present

### Minimal sample structure

```python
sample = {
    "input_ids": ...,
    "audio_codes": ...,
    "is_transition_start": torch.tensor(is_transition_start, dtype=torch.float32),
    "transition_vec": torch.tensor(transition_vec, dtype=torch.float32),
}
```

For normal chunks:
- `is_transition_start = 0`
- `transition_vec = zeros(...)`

That makes masking easy.

---

## 2.7 Add a seam-transition projector to the model

This is the structured numeric path that converts `transition_vec` into a few hidden-state slots.

### Skeleton

```python
import torch
import torch.nn as nn

class SeamTransitionProjector(nn.Module):
    def __init__(self, in_dim, hidden_size, slots=2):
        super().__init__()
        self.slots = slots
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * slots),
        )
        self.base_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, seam_strength):
        out = self.net(x).view(x.shape[0], self.slots, self.hidden_size)
        return self.base_gate * seam_strength[:, None, None] * out
```

### Important principle

This projector should be **residual**, not dominant.

You do not want it to replace the segment instruction.

---

## 2.8 Inject seam slots into talker embeddings

Keep the current speaker injection intact.

Add seam slots next to it.

### Broad logic

```python
input_text_embedding = ...
input_codec_embedding = ...

# existing speaker conditioning
input_codec_embedding[:, 6, :] = speaker_embedding

# new seam conditioning
seam_slots = model.seam_transition_proj(transition_vec, seam_strength)   # [B, K, H]
input_codec_embedding[:, 7:7+K, :] += seam_slots

input_embeddings = input_text_embedding + input_codec_embedding
```

This keeps the system clean:

- speaker slot = identity
- seam slots = temporary boundary controller
- instruction path = global segment style anchor

---

## 2.9 Make the seam loss local and masked

This is the most important training rule.

The seam loss should apply only when:

- `is_transition_start == 1`

and only over the **entry region** of the new segment.

It should **not** supervise the whole segment.

### Broad training objective

```python
loss = main_loss + 0.3 * sub_talker_loss + seam_mask * lambda_seam * seam_style_loss
```

Where:

- `main_loss` = normal model loss
- `sub_talker_loss` = existing residual-codebook loss
- `seam_mask` = 1 only for seam-start chunks

### Seam loss idea

Predict or match the desired entry-style behavior over the first part of the chunk.

For example:

- first 1 second
- first 2 seconds
- first N codec frames based on `duration_sec`

### Minimal skeleton

```python
entry_hidden = hidden_states[:, :entry_frames, :]
entry_pooled = entry_hidden.mean(dim=1)

pred_entry_style = model.seam_style_head(entry_pooled)
seam_style_loss = torch.nn.functional.smooth_l1_loss(pred_entry_style, target_entry_style)

loss = main_loss + 0.3 * sub_talker_loss + seam_mask * lambda_seam * seam_style_loss
```

The important part is **masking** and **time-locality**.

---

## 2.10 Summary of training changes

### New file

- `build_seam_jsonl.py`

### Modified files

- `prepare_data.py`
- dataset loader / collator
- `sft_12hz.py`
- model code where talker embeddings are assembled

### Key new concepts

- `is_transition_start`
- `transition_state`
- `transition_vec`
- seam projector
- seam-masked local loss

---

## 3) Changes needed in inference / backend (Transformers only)

## 3.1 Broad idea

Do not expose a global `style_state` for all chunks.

Expose a **seam-only transition controller**.

### Public API

```python
wavs, sr = model.generate_custom_voice(
    text=chunk_text,
    speaker="speaker_test",
    language="English",
    instruct=current_instruction,
    transition_state=current_transition_state,   # only on seam chunks
)
```

For normal chunks:

```python
transition_state = None
```

That is the cleanest interface.

---

## 3.2 transition_state shape

```python
transition_state = {
    "prev_style_state": {
        "loudness_mean": ...,
        "loudness_tail_slope": ...,
        "f0_center": ...,
        "f0_range": ...,
        "f0_tail_slope": ...,
        "speaking_rate": ...,
        "pause_ratio": ...,
        "phrase_end_shape": "WEAK_FALL",
    },
    "duration_sec": 2.5,
    "mode": "SMOOTH",
    "strength": 1.0,
}
```

The segment instruction still says what the new segment should be.  
The transition state only says how to bridge from the previous segment into it.

---

## 3.3 Convert transition_state to a normalized vector

Use the same normalization scheme as training.

### Sketch

```python
PHRASE_ENDS = ["CONTINUE", "WEAK_FALL", "STRONG_FALL", "RISE"]
TRANSITION_MODES = ["SMOOTH", "FAST", "HARD"]

def transition_state_to_vector(state, stats):
    ps = state["prev_style_state"]

    vec = [
        (ps["loudness_mean"] - stats["loudness_mean_mean"]) / stats["loudness_mean_std"],
        (ps["loudness_tail_slope"] - stats["loudness_tail_slope_mean"]) / stats["loudness_tail_slope_std"],
        (ps["f0_center"] - stats["f0_center_mean"]) / stats["f0_center_std"],
        (ps["f0_range"] - stats["f0_range_mean"]) / stats["f0_range_std"],
        (ps["f0_tail_slope"] - stats["f0_tail_slope_mean"]) / stats["f0_tail_slope_std"],
        (ps["speaking_rate"] - stats["speaking_rate_mean"]) / stats["speaking_rate_std"],
        ps["pause_ratio"],
    ]

    phrase_onehot = [0.0] * len(PHRASE_ENDS)
    phrase_onehot[PHRASE_ENDS.index(ps["phrase_end_shape"])] = 1.0

    mode_onehot = [0.0] * len(TRANSITION_MODES)
    mode_onehot[TRANSITION_MODES.index(state["mode"])] = 1.0

    duration_norm = state["duration_sec"] / stats["transition_duration_max"]

    return vec + phrase_onehot + [duration_norm, state["strength"]] + mode_onehot
```

---

## 3.4 Patch generate_custom_voice

### Broad wrapper change

```python
def generate_custom_voice(
    self,
    text,
    speaker,
    language=None,
    instruct=None,
    transition_state=None,
    **generate_kwargs,
):
    # existing prep
    ...

    transition_tensor = None
    seam_strength = None

    if transition_state is not None:
        if isinstance(transition_state, dict):
            transition_state = [transition_state]

        transition_vecs = [
            transition_state_to_vector(ts, self.transition_norm_stats)
            for ts in transition_state
        ]
        transition_tensor = torch.tensor(
            transition_vecs, dtype=torch.float32, device=self.device
        )

        seam_strength = torch.tensor(
            [ts.get("strength", 1.0) for ts in transition_state],
            dtype=torch.float32,
            device=self.device,
        )

    outputs = self.model.generate(
        input_ids=input_ids,
        speakers=speakers,
        languages=languages,
        instruct_ids=instruct_ids,
        transition_state=transition_tensor,   # new
        seam_strength=seam_strength,          # new
        **generate_kwargs,
    )
```

That is the minimal backend API change.

---

## 3.5 Patch the model forward / generate path

Thread `transition_state` and `seam_strength` into the place where talker embeddings are assembled.

### Broad logic

```python
def forward(..., transition_state=None, seam_strength=None, ...):
    ...
    input_codec_embedding[:, 6, :] = speaker_embedding

    if transition_state is not None:
        seam_slots = self.seam_transition_proj(transition_state, seam_strength)
        input_codec_embedding[:, 7:7+self.seam_slots, :] += seam_slots

    input_embeddings = input_text_embedding + input_codec_embedding
    ...
```

That is the important part.

---

## 3.6 Add a seam gate in the streaming backend

The seam controller should not stay active forever.

You need a backend-side gate that decays over the selected transition duration.

### Simple state machine

- inside same segment → `transition_state = None`
- when new segment starts → create transition state from previous segment tail
- keep applying it for a limited time window
- decay it toward zero
- then disable it completely

### Minimal controller

```python
class SeamController:
    def __init__(self):
        self.active = False
        self.transition_state = None
        self.remaining_sec = 0.0

    def start(self, transition_state):
        self.active = True
        self.transition_state = transition_state.copy()
        self.remaining_sec = transition_state["duration_sec"]

    def step(self, chunk_duration_sec):
        if not self.active:
            return None

        if self.remaining_sec <= 0:
            self.active = False
            self.transition_state = None
            return None

        progress = 1.0 - (self.remaining_sec / max(self.transition_state["duration_sec"], 1e-6))
        strength0 = self.transition_state.get("strength", 1.0)
        current_strength = max(0.0, strength0 * (1.0 - progress))

        out = dict(self.transition_state)
        out["strength"] = current_strength

        self.remaining_sec -= chunk_duration_sec
        if self.remaining_sec <= 0:
            self.active = False

        return out
```

This is intentionally simple.

---

## 3.7 Streaming chunk loop

### Broad inference pattern

```python
seam_controller = SeamController()
prev_segment_tail_path = None

for chunk in stream:
    if chunk.starts_new_segment:
        prev_state = extract_style_state(prev_segment_tail_path)

        seam_controller.start({
            "prev_style_state": prev_state,
            "duration_sec": chunk.transition_duration_sec,
            "mode": "SMOOTH",
            "strength": 1.0,
        })

    current_transition_state = seam_controller.step(chunk.estimated_duration_sec)

    wavs, sr = model.generate_custom_voice(
        text=chunk.text,
        speaker=chunk.speaker,
        language=chunk.language,
        instruct=chunk.instruct,
        transition_state=current_transition_state,
    )

    wav = wavs[0]

    if chunk.is_last_in_segment:
        prev_segment_tail_path = save_tail_for_analysis(wav, sr)
```

This is the desired runtime behavior:

- no seam control on ordinary chunks
- seam control only near the boundary
- automatic fade-out

---

## 3.8 Optional guardrail: tiny post-audio seam cleanup

Even with model-side seam control, a small amount of post-audio cleanup can still help.

Use this only at seam chunks:

- small gain trim
- small pitch-center trim
- short crossfade

Do not use it inside normal chunks.

### Sketch

```python
def apply_seam_cleanup(prev_wav, cur_wav, sr):
    # broad idea only:
    # compare previous tail vs current head
    # apply tiny correction if the mismatch exceeds a threshold
    return cur_wav
```

This is optional, but it is a useful guardrail.

---

## 3.9 Summary of inference/backend changes

### Public API change

- add `transition_state=` to `generate_custom_voice(...)`

### Model change

- add `seam_transition_proj`
- inject seam slots into talker embeddings
- keep seam path residual

### Backend change

- add a seam-local controller
- only activate it on selected boundaries
- fade it out over 1–5 seconds
- keep `instruct` unchanged throughout the segment

---

## Final implementation advice

The most important correction to the earlier plan is this:

**Do not treat continuity as a global style channel.**  
Treat it as a **temporary boundary controller**.

So the rollout order should be:

1. build seam-aware JSONL
2. preserve seam metadata in `prepare_data.py`
3. add seam projector + seam-masked loss
4. patch `generate_custom_voice(...)` with `transition_state`
5. add seam gate logic in your backend loop

That keeps the segment instruction dominant and confines manual continuity control to the seams only.

---

## References

Official Qwen docs and code that this plan is based on:

- Qwen3-TTS repo / README: https://github.com/QwenLM/Qwen3-TTS
- Fine-tuning guide: https://www.mintlify.com/QwenLM/Qwen3-TTS/advanced/fine-tuning
- Qwen3-TTS technical report: https://arxiv.org/abs/2601.15621
- `pyproject.toml` dependency list: https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml

Useful public script references:

- `sft_12hz.py` / embedding assembly discussion: https://github.com/QwenLM/Qwen3-TTS/issues/174
- Fine-tuning cadence drift report: https://github.com/QwenLM/Qwen3-TTS/issues/179
