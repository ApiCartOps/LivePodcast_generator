"""A Pipecat TTSService backed by the open-weight local Kokoro-82M model.

Pipecat ships built-in services for cloud/self-hosted-server TTS engines
(ElevenLabs, Cartesia, a Piper HTTP server, etc.) but has no in-process
Kokoro integration, so this implements one following the same pattern as
`pipecat.services.piper.tts.PiperTTSService`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

KOKORO_SAMPLE_RATE = 24000

# Kokoro loads its ~80M-parameter weights once per (lang_code) and is reused
# by every service instance/line, so per-line instantiation stays cheap.
_pipelines: dict[str, object] = {}


def _get_kokoro_pipeline(lang_code: str):
    if lang_code not in _pipelines:
        from kokoro import KPipeline

        logger.info(f"Loading Kokoro-82M weights for lang_code={lang_code!r} (first use only)...")
        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


class KokoroTTSService(TTSService):
    """Local, fully offline TTS using the open-weight Kokoro-82M model."""

    def __init__(self, *, voice: str, lang_code: str = "a", speed: float = 1.0, **kwargs):
        super().__init__(
            sample_rate=KOKORO_SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model=None, voice=voice, language=None),
            **kwargs,
        )
        self._voice = voice
        self._lang_code = lang_code
        self._speed = speed

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_tts_usage_metrics(text)

            pipeline = _get_kokoro_pipeline(self._lang_code)
            audio_bytes = await asyncio.to_thread(self._synthesize, pipeline, text)

            await self.stop_ttfb_metrics()
            yield TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=KOKORO_SAMPLE_RATE,
                num_channels=1,
                context_id=context_id,
            )
        except Exception as e:  # noqa: BLE001 - surface any synth failure as a pipeline frame
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Kokoro TTS error: {e}")

    def _synthesize(self, pipeline, text: str) -> bytes:
        chunks: list[np.ndarray] = []
        for result in pipeline(text, voice=self._voice, speed=self._speed):
            if result.audio is not None:
                chunks.append(result.audio.numpy())
        if not chunks:
            return b""
        audio = np.concatenate(chunks)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm16.tobytes()
