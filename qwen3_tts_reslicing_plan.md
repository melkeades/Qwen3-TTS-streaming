# Qwen3-TTS seam reslicing plan (first milestone)

## What this document is for

This document is only about the **first workstream**:

- how to **reuse the same recordings**
- how to **reslice / re-index** them into segment-aware chunks
- how to build the **new seam-aware JSONL**
- what libraries are needed
- where this fits into the official Qwen training flow

It does **not** try to cover the full model patching or backend work in depth.  
Its only goal is to get the dataset side into the correct shape so training and inference changes have something clean to consume later.

---

## 1) What stays the same vs what changes

## What stays the same

You can keep the same:

- source recordings
- transcripts
- reference audios
- speaker inventory
- general training flow shape

So this is **not** a new corpus collection project.

## What changes

You need a **new metadata layer** and usually a **new slicing/indexing pass** over the same recordings.

The reason is simple:

the official Qwen fine-tuning flow expects samples like:

- `audio`
- `text`
- `ref_audio`

That is enough for ordinary single-speaker fine-tuning, but it is not enough to express:

- which chunk starts a new segment
- which previous segment it should transition from
- what the previous segment’s tail style looked like
- how long the transition should last

So the recordings can stay the same, but the **JSONL schema and chunk bookkeeping** must change.

---

## 2) The end result you want

After this reslicing step, you want a seam-aware raw JSONL where every chunk is one of two types:

### A. Normal in-segment chunk

This is a chunk that sits inside a segment and should just follow the segment instruction.

Example:

```json
{
  "audio": "seg_B_chunk_003.wav",
  "text": "more text from the happy segment",
  "ref_audio": "reference.wav",
  "instruct": "happy",
  "segment_id": "B",
  "is_transition_start": 0
}
```

### B. Seam-start chunk

This is the first chunk of a new segment, or any chunk you explicitly decide is a transition boundary.

Example:

```json
{
  "audio": "seg_B_chunk_000.wav",
  "text": "first text of the happy segment",
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

That is the target artifact for this phase.

---

## 3) Where this sits in the official Qwen flow

The official Qwen fine-tuning flow is broadly:

1. raw JSONL
2. `prepare_data.py`
3. processed JSONL with `audio_codes`
4. `sft_12hz.py`

This reslicing project adds a new step **before** `prepare_data.py`:

1. source recordings / transcripts
2. **reslice + seam labeling**
3. seam-aware raw JSONL
4. patched `prepare_data.py`
5. seam-aware processed JSONL
6. patched training script

So the reslicing tool is the first real thing to build.

---

## 4) Recommended file/tool layout

Keep it simple.

```text
tools/
  build_seam_jsonl.py
  segment_audio.py            # optional helper
  extract_style_state.py      # optional helper
  schema.py                   # optional pydantic/dataclass schema

data/
  source_manifest.jsonl
  seam_manifest.jsonl

chunks/
  seg_A_chunk_000.wav
  seg_A_chunk_001.wav
  seg_B_chunk_000.wav
  ...
```

You can also keep everything in one file at first and split later.

### Minimal recommendation

Start with only one script:

```text
build_seam_jsonl.py
```

and keep helper functions in the same file until the logic stabilizes.

---

## 5) Libraries

You do **not** need a heavy new toolchain for the first version.

Recommended Python libraries:

- `json`
- `pathlib`
- `dataclasses` or `typing`
- `soundfile`
- `librosa`
- optionally `numpy`
- optionally `torchaudio`

### Minimal import set

```python
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
```

### What each is for

- `json` / `pathlib`: reading and writing manifests
- `soundfile`: reading and writing wav files
- `librosa`: resampling, simple feature extraction, rough timing utilities
- `numpy`: numeric operations
- `torchaudio`: optional if you prefer it over `soundfile` for IO

For the reslicing phase, this is enough.

---

## 6) What your source manifest should look like

Your source manifest does **not** need to match the final seam-aware format yet.

It only needs enough information to reconstruct:

- the source audio file
- the ordered segment structure
- the instruction for each segment
- the transcript or text span for each segment

A practical source shape:

```json
{
  "source_audio": "session_001.wav",
  "ref_audio": "reference.wav",
  "segments": [
    {
      "segment_id": "A",
      "instruct": "sad",
      "start_sec": 0.0,
      "end_sec": 14.2,
      "text": "segment A text"
    },
    {
      "segment_id": "B",
      "instruct": "happy",
      "start_sec": 14.2,
      "end_sec": 29.7,
      "text": "segment B text"
    }
  ]
}
```

If you already have finer timestamps inside each segment, even better.

---

## 7) The actual reslicing problem

The reslicing job has two distinct layers:

### Layer 1: segment boundaries
These define the **semantic / instruction boundaries**.

Example:

- segment A = sad
- segment B = happy

This is where seam transitions happen.

### Layer 2: chunk boundaries inside segments
These define the **streaming units**.

Example:
- segment B gets split into chunk 0, 1, 2, 3, ...

This is where the model sees individual generation units.

The seam controller only matters when a chunk is the **start of a new segment**.

So your slicing code must preserve both concepts:
- segment structure
- chunk structure

---

## 8) Recommended reslicing strategy

## Step 1: define segments first

Do **not** start by chopping everything into uniform chunks.

First define the semantic segments that share a single instruction.

This is the level where you say:

- this whole region is `sad`
- this whole region is `happy`
- this whole region is `angry`

That gives you the seam locations.

## Step 2: split each segment into chunks

Once the segment boundaries are fixed, split each segment into streaming-sized chunks.

For the first version, keep chunking simple:
- target duration range
- optional punctuation-aware split if transcript timing allows it

Do not over-optimize chunking at the beginning.

## Step 3: mark seam starts

For each segment:
- its first chunk becomes the seam-start chunk if there is a previous segment
- all later chunks are normal chunks

That one rule gets you very far.

---

## 9) How to choose chunk sizes

You want chunk sizes that are practical for your backend and not too tiny.

A good first-pass rule is:

- avoid extremely short chunks
- avoid giant chunks
- keep durations fairly regular inside a segment

You can use something like:

- preferred chunk length: 2–6 seconds
- minimum chunk length: ~1 second
- allow the first chunk of a segment to be a bit shorter or longer if needed

The exact numbers depend on your latency and serving design, but the important thing is consistency.

### Why not fully uniform chunks

If a strict time split creates ugly boundaries:
- right in the middle of a word
- right before a major punctuation drop
- right after a strong cadence fall

then the seam and continuity logic becomes noisier.

So if you have transcript timing or punctuation alignment, use it.

If not, simple time chunking is still okay for v1.

---

## 10) What to save for each chunk

Each chunk record should preserve:

- where it came from
- which segment it belongs to
- whether it starts a new segment
- which instruction governs it

Recommended fields:

```python
chunk_record = {
    "audio": "...",
    "text": "...",
    "ref_audio": "...",
    "instruct": "...",
    "segment_id": "...",
    "chunk_index": 0,
    "chunk_start_sec": 0.0,
    "chunk_end_sec": 2.9,
    "is_transition_start": 0 or 1,
}
```

For seam-start chunks, add:

```python
chunk_record["prev_segment_id"] = "..."
chunk_record["transition_state"] = {...}
```

This makes debugging much easier later.

---

## 11) Tail extraction from the previous segment

For seam-start chunks, you need a summary of how the previous segment ended.

The clean way to do that is:

- take a short tail from the previous segment
- compute `prev_style_state` from that tail

### Recommended tail window

Start with something like:
- 0.5 s to 1.5 s from the end of the previous segment

That is usually enough to capture:
- loudness baseline
- F0 center / slope
- cadence exit shape
- pause tendency near the boundary

Do not overcomplicate this at first.

### Tail extraction sketch

```python
def extract_tail(wav, sr, tail_sec=1.0):
    n = int(sr * tail_sec)
    return wav[-n:] if len(wav) > n else wav
```

---

## 12) Writing chunk wav files

If your source recordings are long, write actual per-chunk wav files to disk.

That makes:
- debugging easier
- training manifests simpler
- feature extraction simpler
- failure recovery much easier

### Minimal cutter

```python
def write_audio_slice(src_path, dst_path, start_sec, end_sec, sr_target=24000):
    wav, sr = sf.read(src_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != sr_target:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=sr_target)
        sr = sr_target

    start = int(start_sec * sr)
    end = int(end_sec * sr)
    chunk = wav[start:end]

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst_path, chunk, sr)
    return dst_path
```

That is enough for a first pass.

---

## 13) Building the seam-aware JSONL

This is the main output of the reslicing phase.

### Broad workflow

For each source session:

1. read source manifest entry
2. iterate ordered segments
3. cut segment audio
4. split each segment into chunks
5. write chunk wav files
6. mark first chunk of each non-initial segment as seam-start
7. compute `prev_style_state` from the previous segment tail
8. write JSONL entries

### Broad skeleton

```python
def build_records(session):
    records = []
    src_audio = session["source_audio"]
    ref_audio = session["ref_audio"]
    segments = session["segments"]

    prev_segment_tail_path = None
    prev_segment_id = None

    for seg_idx, seg in enumerate(segments):
        chunk_spans = make_chunk_spans(seg["start_sec"], seg["end_sec"])

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_spans):
            chunk_path = make_chunk_path(seg["segment_id"], chunk_idx)
            write_audio_slice(src_audio, chunk_path, chunk_start, chunk_end)

            record = {
                "audio": str(chunk_path),
                "text": get_chunk_text(seg, chunk_start, chunk_end),
                "ref_audio": ref_audio,
                "instruct": seg["instruct"],
                "segment_id": seg["segment_id"],
                "chunk_index": chunk_idx,
                "chunk_start_sec": chunk_start,
                "chunk_end_sec": chunk_end,
                "is_transition_start": 0,
            }

            if seg_idx > 0 and chunk_idx == 0:
                prev_style_state = extract_style_state(prev_segment_tail_path)
                record["prev_segment_id"] = prev_segment_id
                record["is_transition_start"] = 1
                record["transition_state"] = {
                    "prev_style_state": prev_style_state,
                    "duration_sec": 2.5,
                    "mode": "SMOOTH",
                    "strength": 1.0,
                }

            records.append(record)

        prev_segment_tail_path = build_prev_segment_tail(src_audio, seg["start_sec"], seg["end_sec"])
        prev_segment_id = seg["segment_id"]

    return records
```

That is the exact broad-strokes pipeline you want.

---

## 14) Chunk text assignment

There are three common cases:

### Case A: segment text only
You only know the full segment text.

Then the easiest first version is:
- assign the full segment text to every chunk, or
- assign text only to the first chunk and leave later chunks derived later

This is not ideal, but acceptable for scaffolding.

### Case B: word / sentence timestamps
You know where words or sentences land.

Then assign the text span that overlaps each chunk.

This is the best setup.

### Case C: punctuation only
You do not have timings but you can split text heuristically.

Then approximate chunk text by punctuation or sentence boundaries.

For this reslicing milestone, the exact text assignment can stay somewhat rough as long as the segment/chunk structure is correct.

---

## 15) Computing prev_style_state

This should be done from the **tail of the previous segment**, not from the whole segment.

Why:
- you care about how the previous segment ended
- the bridge should use the boundary-local delivery state
- whole-segment averages wash out the seam

### Suggested fields

```python
prev_style_state = {
    "loudness_mean": ...,
    "loudness_tail_slope": ...,
    "f0_center": ...,
    "f0_range": ...,
    "f0_tail_slope": ...,
    "speaking_rate": ...,
    "pause_ratio": ...,
    "phrase_end_shape": ...,
}
```

### Minimal extractor

```python
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

    f0, _, _ = librosa.pyin(
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

    speaking_rate = 0.0
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

---

## 16) Phrase-end classification

Keep this discrete and simple.

### Suggested labels

- `CONTINUE`
- `WEAK_FALL`
- `STRONG_FALL`
- `RISE`

### Minimal implementation

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

This is good enough for the first implementation.

---

## 17) Chunk span generation

Do not hardcode one global duration if your data is messy.  
Encapsulate it.

### Minimal helper

```python
def make_chunk_spans(start_sec, end_sec, target_len=3.0, min_len=1.0):
    spans = []
    cur = start_sec

    while cur < end_sec:
        nxt = min(cur + target_len, end_sec)

        # merge tiny tail into previous chunk
        if end_sec - nxt < min_len and spans:
            prev_start, _ = spans[-1]
            spans[-1] = (prev_start, end_sec)
            break

        spans.append((cur, nxt))
        cur = nxt

    return spans
```

This is intentionally simple and predictable.

---

## 18) What the seam script should output

At the end, you want:

### A. Chunk wav files on disk
For debugging and direct inspection.

### B. A seam-aware raw JSONL
This is the artifact used by the next stage.

### C. Optional stats file
Very useful for later normalization.

Example:

```json
{
  "loudness_mean_mean": -28.1,
  "loudness_mean_std": 4.2,
  "f0_center_mean": 5.11,
  "f0_center_std": 0.34,
  "speaking_rate_mean": 3.9,
  "speaking_rate_std": 0.7,
  "transition_duration_max": 5.0
}
```

Store this once and reuse it in training and inference.

---

## 19) Validation checks

Before you touch the model, validate the seam JSON itself.

### Check 1
Every seam-start chunk should have:
- `prev_segment_id`
- `transition_state`
- `is_transition_start = 1`

### Check 2
Every non-seam chunk should:
- not accidentally carry transition metadata
- have `is_transition_start = 0`

### Check 3
Segment instructions should stay constant inside one segment.

### Check 4
The first chunk of a segment should point to the correct previous segment.

### Check 5
Chunk files should exist and durations should make sense.

### Minimal validator sketch

```python
def validate_record(r):
    if r["is_transition_start"] == 1:
        assert "prev_segment_id" in r
        assert "transition_state" in r
    else:
        assert "transition_state" not in r or r["transition_state"] is None
```

Do this early. It will save time later.

---

## 20) Suggested implementation order

### Phase 1
Build the seam-aware JSON without writing actual chunk wav files yet.

Goal:
- prove the indexing logic
- prove the segment/chunk bookkeeping
- prove the seam labeling

### Phase 2
Write chunk wav files and tail wav files.

Goal:
- verify slicing
- inspect files manually
- make debugging easier

### Phase 3
Compute `prev_style_state` and append it to seam-start records.

Goal:
- complete the seam-aware raw JSONL

### Phase 4
Feed that JSONL into a patched `prepare_data.py`.

This staged approach is much safer than trying to do everything in one pass.

---

## 21) What you do NOT need yet

At this stage you do **not** need to:

- patch the CustomVoice model
- patch `generate_custom_voice(...)`
- add seam projector modules
- add seam-masked training losses
- implement backend seam decay

Those are later milestones.

This first milestone is only about producing the correct seam-aware raw JSONL and chunk assets.

---

## 22) Minimal main() shape

```python
def main():
    src_manifest_path = Path("data/source_manifest.jsonl")
    out_manifest_path = Path("data/seam_manifest.jsonl")

    sessions = [json.loads(line) for line in src_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    all_records = []
    for session in sessions:
        records = build_records(session)
        all_records.extend(records)

    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_records)} records to {out_manifest_path}")
```

That is all the orchestration you need for v1.

---

## 23) Practical summary

The shortest accurate summary is:

- keep the same recordings
- define semantic segments first
- split segments into chunks second
- mark only seam-start chunks as transitions
- compute previous-tail style state only for those seam-start chunks
- write a new seam-aware raw JSONL
- leave the rest of the model/training changes for later

That is the first workstream to tackle.

---

## References

Official Qwen docs and code that motivate this structure:

- Qwen3-TTS repo / README: https://github.com/QwenLM/Qwen3-TTS
- Fine-tuning guide: https://www.mintlify.com/QwenLM/Qwen3-TTS/advanced/fine-tuning
- Qwen3-TTS package dependencies: https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml
