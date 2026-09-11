"""Drives dialogue lines through a real Pipecat pipeline to render audio.

Each line runs through its own Pipeline([KokoroTTSService(voice=...), sink]):
a TTSSpeakFrame goes in as a standalone utterance, the sink collects the
resulting TTSAudioRawFrame chunks, and an EndFrame closes out the task.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.frames.frames import EndFrame, Frame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from podcast_gen.kokoro_tts_service import KokoroTTSService
from podcast_gen.script_gen import DialogueLine


@dataclass
class RenderedLine:
    speaker: str
    text: str
    audio: bytes
    sample_rate: int


class _AudioCollector(FrameProcessor):
    """Sink processor: captures every TTSAudioRawFrame that reaches it."""

    def __init__(self):
        super().__init__()
        self.chunks: list[bytes] = []
        self.sample_rate: int | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.chunks.append(frame.audio)
            self.sample_rate = frame.sample_rate
        await self.push_frame(frame, direction)


async def render_line(line: DialogueLine, *, voice: str, lang_code: str = "a") -> RenderedLine:
    """Run one dialogue line through a Pipecat pipeline and return its audio."""
    service = KokoroTTSService(voice=voice, lang_code=lang_code)
    sink = _AudioCollector()
    pipeline = Pipeline([service, sink])
    task = PipelineTask(pipeline)

    await task.queue_frames([TTSSpeakFrame(text=line.text), EndFrame()])
    await PipelineRunner().run(task)

    return RenderedLine(
        speaker=line.speaker,
        text=line.text,
        audio=b"".join(sink.chunks),
        sample_rate=sink.sample_rate or 24000,
    )


async def render_dialogue(
    lines: list[DialogueLine],
    *,
    voice_map: dict[str, str],
    lang_code: str = "a",
) -> list[RenderedLine]:
    """Render every line in order, keeping conversation order intact."""
    rendered = []
    for line in lines:
        voice = voice_map.get(line.speaker)
        if voice is None:
            raise ValueError(f"No voice configured for speaker {line.speaker!r}")
        rendered.append(await render_line(line, voice=voice, lang_code=lang_code))
    return rendered
