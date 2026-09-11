"""Interactive voice Q&A about a document.

Mic -> (push-to-talk gate) -> Whisper STT -> Claude (doc as context) -> Kokoro
TTS -> speakers, all wired as a real Pipecat pipeline running against local
audio devices (no LiveKit/Daily room, no browser).

Turn-taking uses push-to-talk rather than pipecat's VAD/turn-detection stack:
`SegmentedSTTService` (which `WhisperSTTService` is) only needs
`VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame` markers to know
which audio to transcribe, so a keypress can supply those directly instead of
running a VAD model.
"""

from __future__ import annotations

import asyncio
import os
import threading

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.llm_service import LLMService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import Model as WhisperModel
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from podcast_gen.kokoro_tts_service import KokoroTTSService

ANTHROPIC_MODEL = "claude-sonnet-5"
OLLAMA_MODEL = "llama3.1"

SYSTEM_PROMPT_TEMPLATE = """\
You are a friendly, knowledgeable podcast co-host doing a live Q&A segment. \
The listener has just heard an episode covering the material below and is now \
asking you follow-up questions about it, out loud.

Answer conversationally and concisely: 2-4 short spoken sentences, no lists, \
no markdown, no headers. If something isn't covered by the material, say so \
honestly rather than guessing.

SOURCE MATERIAL:
{source_text}
"""


class PushToTalkGate(FrameProcessor):
    """Passes audio through untouched; a background thread injects VAD-shaped
    start/stop frames on keypress so the STT service knows what to transcribe.

    `pause_event`, if given, is set the instant the user starts talking and
    is NOT cleared here — something downstream (e.g. episode playback) that
    needs to know when it's safe to resume should clear it itself once the
    answer has actually finished. It's set directly from this keypress, not
    by watching the emitted frame arrive somewhere downstream: several
    processors in a STT -> LLM -> TTS chain don't forward
    VADUserStartedSpeakingFrame all the way through, so anything waiting on
    it reaching the far end of the pipeline may never see it.
    """

    def __init__(self, pause_event: asyncio.Event | None = None):
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._pause_event = pause_event

    async def setup(self, setup):
        await super().setup(setup)
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()

    async def cleanup(self):
        self._stop_requested = True
        await super().cleanup()

    def _input_loop(self):
        while not self._stop_requested:
            try:
                input("\nPress Enter, then ask your question (Ctrl+C to quit)... ")
            except EOFError:
                break
            if self._pause_event is not None and self._loop is not None:
                print("[trace] keypress -> scheduling pause_event.set()")
                self._loop.call_soon_threadsafe(self._pause_event.set)
            self._emit(VADUserStartedSpeakingFrame())
            print("Listening... press Enter again when you're done talking.")
            try:
                input()
            except EOFError:
                break
            self._emit(VADUserStoppedSpeakingFrame())
            print("Thinking...")

    def _emit(self, frame: Frame) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(
                self.push_frame(frame, FrameDirection.DOWNSTREAM), self._loop
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class ErrorEcho(FrameProcessor):
    """Passes frames through, printing any ErrorFrame that shows up anywhere
    in the pipeline — otherwise a failed STT/LLM/TTS call fails silently."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            print(f"[error]: {frame.error}")
        await self.push_frame(frame, direction)


class TranscriptEcho(FrameProcessor):
    """Passes frames through, printing what Whisper heard (for debugging)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            print(f"[you said]: {frame.text}")
        await self.push_frame(frame, direction)


class AnswerEcho(FrameProcessor):
    """Passes frames through, printing the LLM's answer text as it streams.

    Positioned between the LLM and TTS: that's where the plain-text response
    is visible — TTS consumes it and emits audio frames instead, so a tap
    placed after TTS never sees the text at all.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            print("[host]: ", end="", flush=True)
        elif isinstance(frame, TextFrame):
            print(frame.text, end="", flush=True)
        elif isinstance(frame, LLMFullResponseEndFrame):
            print()
        await self.push_frame(frame, direction)


def list_audio_devices() -> None:
    """Print available PyAudio input/output devices and which ones are default."""
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        default_in = pa.get_default_input_device_info()["index"]
        default_out = pa.get_default_output_device_info()["index"]
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            tags = []
            if i == default_in:
                tags.append("default input")
            if i == default_out:
                tags.append("default output")
            tag_str = f" ({', '.join(tags)})" if tags else ""
            print(
                f"[{i}] {info['name']} — in:{info['maxInputChannels']} "
                f"out:{info['maxOutputChannels']} @ {int(info['defaultSampleRate'])}Hz{tag_str}"
            )
    finally:
        pa.terminate()


def _build_llm(llm_backend: str, ollama_model: str) -> LLMService:
    if llm_backend == "anthropic":
        return AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(model=ANTHROPIC_MODEL),
        )
    if llm_backend == "ollama":
        return OLLamaLLMService(settings=OLLamaLLMService.Settings(model=ollama_model))
    raise ValueError(f"Unknown llm_backend: {llm_backend!r} (expected 'anthropic' or 'ollama')")


async def run_live_qa(
    source_text: str,
    *,
    voice: str = "af_heart",
    lang_code: str = "a",
    whisper_model: WhisperModel = WhisperModel.BASE,
    llm_backend: str = "anthropic",
    ollama_model: str = OLLAMA_MODEL,
    input_device_index: int | None = None,
    output_device_index: int | None = None,
) -> None:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(source_text=source_text)
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    context_aggregator = LLMContextAggregatorPair(context)

    llm = _build_llm(llm_backend, ollama_model)
    stt = WhisperSTTService(settings=WhisperSTTService.Settings(model=whisper_model))
    tts = KokoroTTSService(voice=voice, lang_code=lang_code)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
        )
    )
    gate = PushToTalkGate()
    transcript_echo = TranscriptEcho()
    answer_echo = AnswerEcho()
    error_echo = ErrorEcho()

    pipeline = Pipeline(
        [
            transport.input(),
            gate,
            error_echo,
            stt,
            transcript_echo,
            context_aggregator.user(),
            llm,
            answer_echo,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    print("Ready. Speak into your mic; answers play back through your speakers.")
    print("(Run with --list-devices if you don't hear anything, to check the output device.)")
    try:
        await runner.run(task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await task.queue_frames([EndFrame()])
