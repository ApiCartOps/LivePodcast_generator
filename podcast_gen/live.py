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

from pipecat.frames.frames import EndFrame, Frame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
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
    start/stop frames on keypress so the STT service knows what to transcribe."""

    def __init__(self):
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = False

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
) -> None:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(source_text=source_text)
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    context_aggregator = LLMContextAggregatorPair(context)

    llm = _build_llm(llm_backend, ollama_model)
    stt = WhisperSTTService(settings=WhisperSTTService.Settings(model=whisper_model))
    tts = KokoroTTSService(voice=voice, lang_code=lang_code)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    gate = PushToTalkGate()

    pipeline = Pipeline(
        [
            transport.input(),
            gate,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    print("Ready. Speak into your mic; answers play back through your speakers.")
    try:
        await runner.run(task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await task.queue_frames([EndFrame()])
