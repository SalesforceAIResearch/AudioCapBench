#!/usr/bin/env python3
"""
Build Audio Caption Test Set

Downloads and prepares test samples for evaluating audio captioning from
online sources only (HuggingFace datasets). No local data dependencies.

Pre-computed ID files list ALL available samples per dataset. At build time,
the --sound-clotho, --sound-audiocaps, --music, --speech flags control how
many samples to select from each pool using a fixed random seed.

CSV files (in data_ids/):
  - clotho_ids.csv        (1045 rows: audio_name, duration_s, num_captions, sample_caption)
  - audiocaps_ids.csv     (883 rows: youtube_id, duration_s, num_captions, sample_caption)
  - musiccaps_ids.csv     (903 rows: ytid, start_s, end_s, caption, aspect_list)
  - speech_ids.csv        (5000 rows: index, transcript, caption, duration_s)

Usage:
    python -m audiocapbench.build_dataset --output-dir data/audio_caption
    python -m audiocapbench.build_dataset --output-dir data/audio_caption --dry-run
    python -m audiocapbench.build_dataset --sound-clotho 20 --sound-audiocaps 20 \
        --music 30 --speech 30
"""

import argparse
import csv
import json
import os
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

# Fixed random seed for reproducible sample selection
SEED = 42

# HuggingFace dataset identifiers
CLOTHO_HF_REPO = "piyushsinghpasi/clotho-multilingual"
AUDIOCAPS_HF_REPO = "OpenSound/AudioCaps"
MUSICCAPS_AUDIO_HF_REPO = "kelvincai/MusicCaps_30s_wav"
MUSICCAPS_META_HF_REPO = "google/MusicCaps"
SPEECH_HF_REPO = "seastar105/emo_speech_caption_test"

# Pre-computed ID files (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent
_IDS_DIR = _PROJECT_ROOT / "eval_data_ids"  # default: curated eval subset


# ===================================================================
# Helpers
# ===================================================================

def _write_wav(audio_array: np.ndarray, sr: int, output_path: Path) -> None:
    """Write a numpy float32 audio array to a PCM16 WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)  # mono
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype("<i2")
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _load_csv_ids(filename: str, key_column: str) -> Optional[List[str]]:
    """Load IDs from a CSV file in the IDs directory. Returns list of key values."""
    path = _IDS_DIR / filename
    if not path.exists():
        return None
    ids = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(key_column, "").strip()
            if val:
                ids.append(val)
    return ids if ids else None


def _load_csv_rows(filename: str) -> Optional[List[dict]]:
    """Load all rows from a CSV file in the IDs directory as list of dicts."""
    path = _IDS_DIR / filename
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows if rows else None


def _select_n(pool: List[str], n: int, seed: int) -> List[str]:
    """Select n items from pool using seeded RNG. Deterministic."""
    if n >= len(pool):
        return list(pool)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(pool), size=n, replace=False)
    indices.sort()  # maintain stable ordering
    return [pool[i] for i in indices]


# ===================================================================
# Clotho v2 test – sound samples
# ===================================================================

def select_clotho_samples(
    n: int = 10,
    cache_dir: Optional[str] = None,
) -> List[dict]:
    """
    Fetch n Clotho v2 test samples.
    Uses clotho_ids.csv (all 1045 audio_names) to select n,
    then streams only matching rows from HuggingFace.
    """
    from datasets import load_dataset

    # Try eval CSV first, then full pool
    all_ids = _load_csv_ids("clotho_eval.csv", "audio_name")
    if not all_ids:
        all_ids = _load_csv_ids("clotho_ids.csv", "audio_name")
    kwargs = {"streaming": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if all_ids:
        chosen = all_ids[:n] if n < len(all_ids) else all_ids
        target_set = set(chosen)
        print(f"  Clotho: selected {len(chosen)} from {len(all_ids)} available, "
              f"fetching from {CLOTHO_HF_REPO} ...")
        ds = load_dataset(CLOTHO_HF_REPO, split="test", **kwargs)

        clips: Dict[str, dict] = {}
        row_count = 0
        for row in ds:
            row_count += 1
            if row_count % 500 == 0:
                print(f"    streamed {row_count} rows, found {len(clips)}/{len(target_set)} clips ...")
            audio_name = row.get("audio_name", "")
            if audio_name not in target_set:
                continue
            if audio_name not in clips:
                clips[audio_name] = {
                    "audio_name": audio_name,
                    "audio": row["audio"],
                    "captions": [],
                }
            caption = row.get("caption", "")
            if caption:
                clips[audio_name]["captions"].append(caption)

        # Return in selection order
        selected = []
        for i, name in enumerate(chosen):
            if name not in clips:
                print(f"    Warning: {name} not found in dataset")
                continue
            clip = clips[name]
            audio = clip["audio"]
            audio_arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            selected.append({
                "id": f"clotho_{i}",
                "audio_name": name,
                "audio_array": audio_arr,
                "sr": sr,
                "duration": len(audio_arr) / sr,
                "reference_captions": clip["captions"] if clip["captions"] else [""],
                "source": "clotho_v2_test",
            })
    else:
        # Fallback: stream all, sort, select (no ID file)
        print(f"  Clotho: no ID file, streaming all from {CLOTHO_HF_REPO} ...")
        ds = load_dataset(CLOTHO_HF_REPO, split="test", **kwargs)

        clips_all: Dict[str, dict] = OrderedDict()
        for row in ds:
            audio_name = row.get("audio_name", "")
            if audio_name not in clips_all:
                clips_all[audio_name] = {
                    "audio_name": audio_name,
                    "audio": row["audio"],
                    "captions": [],
                }
            caption = row.get("caption", "")
            if caption:
                clips_all[audio_name]["captions"].append(caption)

        clip_list = sorted(clips_all.values(), key=lambda c: c["audio_name"])
        chosen_names = _select_n(
            [c["audio_name"] for c in clip_list], n, SEED
        )
        name_to_clip = {c["audio_name"]: c for c in clip_list}

        selected = []
        for i, name in enumerate(chosen_names):
            clip = name_to_clip[name]
            audio = clip["audio"]
            audio_arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            selected.append({
                "id": f"clotho_{i}",
                "audio_name": name,
                "audio_array": audio_arr,
                "sr": sr,
                "duration": len(audio_arr) / sr,
                "reference_captions": clip["captions"] if clip["captions"] else [""],
                "source": "clotho_v2_test",
            })

    print(f"  Clotho: {len(selected)} samples ready")
    for s in selected:
        print(f"    {s['id']} ({s['audio_name']}): {s['duration']:.1f}s, "
              f"{len(s['reference_captions'])} refs")
    return selected


# ===================================================================
# AudioCaps test – sound samples
# ===================================================================

def select_audiocaps_samples(
    n: int = 10,
    cache_dir: Optional[str] = None,
) -> List[dict]:
    """
    Fetch n AudioCaps test samples.
    Uses audiocaps_ids.csv (all 883 youtube_ids) to select n,
    then streams only matching rows from HuggingFace.
    """
    from datasets import load_dataset

    all_ids = _load_csv_ids("audiocaps_eval.csv", "youtube_id")
    if not all_ids:
        all_ids = _load_csv_ids("audiocaps_ids.csv", "youtube_id")
    kwargs = {"streaming": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if all_ids:
        chosen = all_ids[:n] if n < len(all_ids) else all_ids
        target_set = set(chosen)
        print(f"  AudioCaps: selected {len(chosen)} from {len(all_ids)} available, "
              f"fetching from {AUDIOCAPS_HF_REPO} ...")
        ds = load_dataset(AUDIOCAPS_HF_REPO, split="test", **kwargs)

        clips: Dict[str, dict] = {}
        row_count = 0
        for row in ds:
            row_count += 1
            if row_count % 500 == 0:
                print(f"    streamed {row_count} rows, found {len(clips)}/{len(target_set)} clips ...")
            ytid = row.get("youtube_id") or row.get("file_name", "")
            if ytid not in target_set:
                continue
            if ytid not in clips:
                clips[ytid] = {
                    "youtube_id": ytid,
                    "audio": row["audio"],
                    "captions": [],
                }
            clips[ytid]["captions"].append(row["caption"])

        selected = []
        for i, ytid in enumerate(chosen):
            if ytid not in clips:
                print(f"    Warning: {ytid} not found in dataset")
                continue
            clip = clips[ytid]
            audio = clip["audio"]
            audio_arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            selected.append({
                "id": f"audiocaps_{i}",
                "youtube_id": ytid,
                "audio_array": audio_arr,
                "sr": sr,
                "duration": len(audio_arr) / sr,
                "reference_captions": clip["captions"],
                "source": "audiocaps_test",
            })
    else:
        print(f"  AudioCaps: no ID file, streaming all from {AUDIOCAPS_HF_REPO} ...")
        ds = load_dataset(AUDIOCAPS_HF_REPO, split="test", **kwargs)

        clips_all: Dict[str, dict] = OrderedDict()
        for row in ds:
            ytid = row.get("youtube_id") or row.get("file_name", "")
            if ytid not in clips_all:
                clips_all[ytid] = {
                    "youtube_id": ytid,
                    "audio": row["audio"],
                    "captions": [],
                }
            clips_all[ytid]["captions"].append(row["caption"])

        clip_list = sorted(clips_all.values(), key=lambda c: c["youtube_id"])
        chosen_ytids = _select_n(
            [c["youtube_id"] for c in clip_list], n, SEED + 1
        )
        ytid_to_clip = {c["youtube_id"]: c for c in clip_list}

        selected = []
        for i, ytid in enumerate(chosen_ytids):
            clip = ytid_to_clip[ytid]
            audio = clip["audio"]
            audio_arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            selected.append({
                "id": f"audiocaps_{i}",
                "youtube_id": ytid,
                "audio_array": audio_arr,
                "sr": sr,
                "duration": len(audio_arr) / sr,
                "reference_captions": clip["captions"],
                "source": "audiocaps_test",
            })

    print(f"  AudioCaps: {len(selected)} samples ready")
    for s in selected:
        print(f"    {s['id']} ({s['youtube_id']}): {s['duration']:.1f}s, "
              f"{len(s['reference_captions'])} refs")
    return selected


# ===================================================================
# MusicCaps – music samples
# ===================================================================

def select_musiccaps_samples(
    n: int = 15,
    cache_dir: Optional[str] = None,
) -> List[dict]:
    """
    Fetch n MusicCaps eval-set samples.
    Uses musiccaps_ids.csv (903 eval ytids + captions) to select n,
    then downloads matching audio from kelvincai/MusicCaps_30s_wav.
    """
    from datasets import load_dataset

    all_ids = _load_csv_ids("musiccaps_eval.csv", "ytid")
    if not all_ids:
        all_ids = _load_csv_ids("musiccaps_ids.csv", "ytid")

    if not all_ids:
        # Fallback: download metadata
        print(f"  MusicCaps: no CSV file, loading from {MUSICCAPS_META_HF_REPO} ...")
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        meta_ds = load_dataset(MUSICCAPS_META_HF_REPO, split="train", **kwargs)
        all_ids = sorted(r["ytid"] for r in meta_ds if r.get("is_audioset_eval"))

    chosen = all_ids[:n] if n < len(all_ids) else all_ids
    target_set = set(chosen)
    print(f"  MusicCaps: selected {len(chosen)} from {len(all_ids)} eval samples, "
          f"fetching from {MUSICCAPS_AUDIO_HF_REPO} ...")

    kwargs_audio = {}
    if cache_dir:
        kwargs_audio["cache_dir"] = cache_dir
    audio_ds = load_dataset(MUSICCAPS_AUDIO_HF_REPO, split="train", **kwargs_audio)

    # Index matching rows by ytid
    matched: Dict[str, dict] = {}
    for i, row in enumerate(audio_ds):
        if i % 200 == 0:
            print(f"    scanned {i}/{len(audio_ds)} rows, found {len(matched)}/{len(target_set)} ...")
        ytid = row.get("ytid", "")
        if ytid in target_set:
            matched[ytid] = row

    selected = []
    for j, ytid in enumerate(chosen):
        if ytid not in matched:
            print(f"    Warning: {ytid} not found in audio dataset")
            continue
        row = matched[ytid]
        audio = row["audio"]
        audio_arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        caption = row.get("caption", "")
        aspect_list = row.get("aspect_list", "")

        captions = [caption] if caption else []
        if aspect_list:
            try:
                aspects = eval(aspect_list) if aspect_list.startswith("[") else aspect_list
                if isinstance(aspects, list):
                    aspect_text = ", ".join(aspects)
                else:
                    aspect_text = str(aspects)
            except Exception:
                aspect_text = aspect_list
            captions.append(aspect_text)

        if not captions:
            captions = [""]

        # Trim to 10s if start_s/end_s available
        start_s = row.get("start_s", 0)
        end_s = row.get("end_s", 0)
        if start_s and end_s and end_s > start_s:
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            if end_sample <= len(audio_arr):
                audio_arr = audio_arr[start_sample:end_sample]

        selected.append({
            "id": f"musiccaps_{j}",
            "ytid": ytid,
            "audio_array": audio_arr,
            "sr": sr,
            "duration": len(audio_arr) / sr,
            "reference_captions": captions,
            "aspect_list": aspect_list,
            "source": "musiccaps_eval",
        })

    print(f"  MusicCaps: {len(selected)} samples ready")
    for s in selected:
        cap_preview = s["reference_captions"][0][:80] if s["reference_captions"] else ""
        print(f"    {s['id']} ({s['ytid']}): {cap_preview}...")
    return selected


# ===================================================================
# Speech caption – speech samples
# ===================================================================

def select_speech_samples(
    n: int = 15,
    cache_dir: Optional[str] = None,
) -> List[dict]:
    """
    Fetch n speech caption samples.
    Uses speech_ids.csv (all 5000 transcripts + captions) to select n,
    then streams only matching rows from HuggingFace (with early exit).
    """
    from datasets import load_dataset

    all_ids = _load_csv_ids("speech_eval.csv", "transcript")
    if not all_ids:
        all_ids = _load_csv_ids("speech_ids.csv", "transcript")
    kwargs = {"streaming": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if all_ids:
        chosen = all_ids[:n] if n < len(all_ids) else all_ids
        target_set = set(chosen)
        print(f"  Speech: selected {len(chosen)} from {len(all_ids)} available, "
              f"fetching from {SPEECH_HF_REPO} ...")
        ds = load_dataset(SPEECH_HF_REPO, split="train", **kwargs)

        matched: Dict[str, dict] = {}
        row_count = 0
        for row in ds:
            row_count += 1
            if row_count % 500 == 0:
                print(f"    streamed {row_count} rows, found {len(matched)}/{len(target_set)} clips ...")
            transcript = row.get("transcript", "")
            if transcript in target_set and transcript not in matched:
                matched[transcript] = row
                if len(matched) == len(target_set):
                    print(f"    all {len(target_set)} clips found after {row_count} rows")
                    break  # Early exit

        selected = []
        for i, transcript in enumerate(chosen):
            if transcript not in matched:
                print(f"    Warning: transcript not found: {transcript[:60]}...")
                continue
            row = matched[transcript]
            audio = row["audio"]
            caption = row.get("caption", "")
            audio_arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            refs = [caption] if caption else []
            if transcript:
                combined = f'{caption} The speaker says: "{transcript}"'
                refs.append(combined)
            if not refs:
                refs = [""]

            selected.append({
                "id": f"speech_{i}",
                "audio_array": audio_arr,
                "sr": sr,
                "duration": len(audio_arr) / sr,
                "reference_captions": refs,
                "caption": caption,
                "transcript": transcript,
                "source": "emo_speech_caption_test",
            })
    else:
        # Fallback: two-pass streaming (no ID file)
        print(f"  Speech: no ID file, streaming all from {SPEECH_HF_REPO} ...")
        ds = load_dataset(SPEECH_HF_REPO, split="train", **kwargs)

        row_meta = []
        for row in ds:
            row_meta.append((row.get("caption", ""), row.get("transcript", "")))

        transcripts_sorted = sorted(t for _, t in row_meta)
        chosen_transcripts = _select_n(transcripts_sorted, n, SEED + 3)
        target_set = set(chosen_transcripts)

        ds2 = load_dataset(SPEECH_HF_REPO, split="train", **kwargs)
        matched = {}
        for row in ds2:
            t = row.get("transcript", "")
            if t in target_set and t not in matched:
                matched[t] = row
                if len(matched) == len(target_set):
                    break

        selected = []
        for i, transcript in enumerate(chosen_transcripts):
            if transcript not in matched:
                continue
            row = matched[transcript]
            audio = row["audio"]
            caption = row.get("caption", "")
            audio_arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            refs = [caption] if caption else []
            if transcript:
                combined = f'{caption} The speaker says: "{transcript}"'
                refs.append(combined)
            if not refs:
                refs = [""]

            selected.append({
                "id": f"speech_{i}",
                "audio_array": audio_arr,
                "sr": sr,
                "duration": len(audio_arr) / sr,
                "reference_captions": refs,
                "caption": caption,
                "transcript": transcript,
                "source": "emo_speech_caption_test",
            })

    print(f"  Speech: {len(selected)} samples ready")
    for s in selected:
        print(f"    {s['id']}: {s['duration']:.1f}s, caption={s['caption'][:80]}...")
    return selected


# ===================================================================
# Build the full test set
# ===================================================================

def build_test_set(
    output_dir: str,
    cache_dir: Optional[str] = None,
    dry_run: bool = False,
    sound_clotho_count: int = 200,
    sound_audiocaps_count: int = 200,
    music_count: int = 300,
    speech_count: int = 300,
) -> None:
    output_path = Path(output_dir)
    total = sound_clotho_count + sound_audiocaps_count + music_count + speech_count

    print("=" * 72)
    print("Building Audio Caption Test Set (AudioCapBench)")
    print("=" * 72)
    print(f"Output: {output_path}")
    print(f"Target: {total} samples "
          f"({sound_clotho_count} clotho + {sound_audiocaps_count} audiocaps "
          f"+ {music_count} music + {speech_count} speech)")
    print(f"Dry run: {dry_run}")
    print()

    # Dry run: just show what would be selected from CSVs, no downloads
    if dry_run:
        for label, csv_name, key_col, n in [
            ("Clotho (sound)", "clotho_eval.csv", "audio_name", sound_clotho_count),
            ("AudioCaps (sound)", "audiocaps_eval.csv", "youtube_id", sound_audiocaps_count),
            ("MusicCaps (music)", "musiccaps_eval.csv", "ytid", music_count),
            ("Speech", "speech_eval.csv", "transcript", speech_count),
        ]:
            ids = _load_csv_ids(csv_name, key_col)
            if ids:
                chosen = ids[:n] if n < len(ids) else ids
                print(f"  {label}: {len(chosen)} samples from {csv_name} ({len(ids)} available)")
                for item in chosen[:3]:
                    print(f"    {item[:80]}...")
                if len(chosen) > 3:
                    print(f"    ... and {len(chosen) - 3} more")
            else:
                print(f"  {label}: CSV not found ({csv_name})")
        print()
        print("Dry run complete. No files downloaded or written.")
        return

    print("[1/4] Clotho v2 test samples (sound) ...")
    clotho_samples = select_clotho_samples(n=sound_clotho_count, cache_dir=cache_dir)
    print()

    print("[2/4] AudioCaps test samples (sound) ...")
    audiocaps_samples = select_audiocaps_samples(n=sound_audiocaps_count, cache_dir=cache_dir)
    print()

    print("[3/4] MusicCaps samples (music) ...")
    musiccaps_samples = select_musiccaps_samples(n=music_count, cache_dir=cache_dir)
    print()

    print("[4/4] Speech caption samples ...")
    speech_samples = select_speech_samples(n=speech_count, cache_dir=cache_dir)
    print()

    # Write audio files and build metadata
    print("=" * 72)
    print("Writing audio files and metadata ...")
    print("=" * 72)

    sound_dir = output_path / "sound"
    music_dir = output_path / "music"
    speech_dir = output_path / "speech"
    sound_dir.mkdir(parents=True, exist_ok=True)
    music_dir.mkdir(parents=True, exist_ok=True)
    speech_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "description": "AudioCapBench - Audio Captioning Benchmark Test Set",
        "categories": {
            "sound": {
                "count": len(clotho_samples) + len(audiocaps_samples),
                "sources": ["clotho_v2_test", "audiocaps_test"],
            },
            "music": {"count": len(musiccaps_samples), "sources": ["musiccaps_eval"]},
            "speech": {"count": len(speech_samples), "sources": ["emo_speech_caption_test"]},
        },
        "total_samples": (len(clotho_samples) + len(audiocaps_samples)
                          + len(musiccaps_samples) + len(speech_samples)),
        "samples": [],
    }

    for sample in clotho_samples:
        fname = f"{sample['id']}.wav"
        wav_path = sound_dir / fname
        _write_wav(sample["audio_array"], sample["sr"], wav_path)
        print(f"  Wrote {wav_path.name}")
        metadata["samples"].append({
            "id": sample["id"],
            "category": "sound",
            "source": sample["source"],
            "audio_file": f"sound/{fname}",
            "duration_s": round(sample["duration"], 2),
            "reference_captions": sample["reference_captions"],
        })

    for sample in audiocaps_samples:
        fname = f"{sample['id']}.wav"
        wav_path = sound_dir / fname
        _write_wav(sample["audio_array"], sample["sr"], wav_path)
        print(f"  Wrote {wav_path.name}")
        metadata["samples"].append({
            "id": sample["id"],
            "category": "sound",
            "source": sample["source"],
            "youtube_id": sample["youtube_id"],
            "audio_file": f"sound/{fname}",
            "duration_s": round(sample["duration"], 2),
            "reference_captions": sample["reference_captions"],
        })

    for sample in musiccaps_samples:
        fname = f"{sample['id']}.wav"
        wav_path = music_dir / fname
        _write_wav(sample["audio_array"], sample["sr"], wav_path)
        print(f"  Wrote {wav_path.name}")
        metadata["samples"].append({
            "id": sample["id"],
            "category": "music",
            "source": sample["source"],
            "ytid": sample["ytid"],
            "audio_file": f"music/{fname}",
            "duration_s": round(sample["duration"], 2),
            "reference_captions": sample["reference_captions"],
            "aspect_list": sample["aspect_list"],
        })

    for sample in speech_samples:
        fname = f"{sample['id']}.wav"
        wav_path = speech_dir / fname
        _write_wav(sample["audio_array"], sample["sr"], wav_path)
        print(f"  Wrote {wav_path.name}")
        metadata["samples"].append({
            "id": sample["id"],
            "category": "speech",
            "source": sample["source"],
            "audio_file": f"speech/{fname}",
            "duration_s": round(sample["duration"], 2),
            "reference_captions": sample["reference_captions"],
            "caption": sample["caption"],
            "transcript": sample["transcript"],
        })

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to: {metadata_path}")

    print()
    print("=" * 72)
    print("Audio Caption Test Set Summary")
    print("=" * 72)
    by_cat = {}
    for s in metadata["samples"]:
        cat = s["category"]
        by_cat[cat] = by_cat.get(cat, 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count} samples")
    print(f"  Total: {len(metadata['samples'])} samples")
    print(f"\nOutput directory: {output_path}")
    print(f"Metadata: {metadata_path}")
    print("=" * 72)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build AudioCapBench test set (all data downloaded from HuggingFace)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/audio_caption",
        help="Output directory for the test set",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="",
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be selected without writing files",
    )
    parser.add_argument(
        "--sound-clotho", type=int, default=200,
        help="Number of Clotho sound samples (default: 200)",
    )
    parser.add_argument(
        "--sound-audiocaps", type=int, default=200,
        help="Number of AudioCaps sound samples (default: 200)",
    )
    parser.add_argument(
        "--music", type=int, default=300,
        help="Number of MusicCaps music samples (default: 300)",
    )
    parser.add_argument(
        "--speech", type=int, default=300,
        help="Number of speech samples (default: 300)",
    )
    parser.add_argument(
        "--ids-dir", type=str, default=None,
        help="Directory with CSV ID files (default: eval_data_ids/)",
    )
    parser.add_argument(
        "--credentials", type=str, default=None,
        help="Path to credentials.env file (for HF_TOKEN, etc.)",
    )
    args = parser.parse_args()

    # Set IDs directory if specified
    if args.ids_dir:
        global _IDS_DIR
        _IDS_DIR = Path(args.ids_dir)

    from .config import load_credentials
    load_credentials(args.credentials)

    build_test_set(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir or None,
        dry_run=args.dry_run,
        sound_clotho_count=args.sound_clotho,
        sound_audiocaps_count=args.sound_audiocaps,
        music_count=args.music,
        speech_count=args.speech,
    )


if __name__ == "__main__":
    main()
