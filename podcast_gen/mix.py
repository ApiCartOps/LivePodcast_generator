"""Stitch rendered dialogue lines into a single podcast MP3."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from podcast_gen.tts_render import RenderedLine

GAP_SECONDS = 0.35


def mix_to_mp3(lines: list[RenderedLine], output_path: str) -> None:
    if not lines:
        raise ValueError("No rendered lines to mix.")

    sample_rate = lines[0].sample_rate
    gap = np.zeros(int(GAP_SECONDS * sample_rate), dtype=np.int16)

    segments = []
    for line in lines:
        pcm = np.frombuffer(line.audio, dtype=np.int16)
        segments.append(pcm)
        segments.append(gap)

    full_audio = np.concatenate(segments)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "podcast.wav"
        sf.write(wav_path, full_audio, sample_rate, subtype="PCM_16")
        _to_mp3(wav_path, output_path)


def _to_mp3(wav_path: Path, output_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "2",
            output_path,
        ],
        check=True,
    )
