"""Play a rendered podcast episode aloud, and let the listener barge in with
a spoken question at any point: playback pauses, Whisper/an LLM/Kokoro answer
it live over the mic/speakers, then playback resumes — like NotebookLM's
"interactive mode", but for the actual generated episode rather than a
separate standalone Q&A session.

Two things run side by side:
- a live Pipecat Q&A pipeline (mic -> STT -> LLM -> TTS -> speaker), always
  listening for a keypress via `PushToTalkGate`, same as `live.py`;
- a plain PyAudio playback loop for the pre-rendered episode lines.

They're coordinated by a single `asyncio.Event`. `PushToTalkGate` sets it the
instant the listener presses Enter to start talking (pausing episode
playback) — directly from the keypress, not by waiting for a frame to
travel all the way through STT -> LLM -> TTS, since several processors in
that chain don't forward `VADUserStartedSpeakingFrame` that far. `PlaybackGate`
(inserted right after the Q&A pipeline's TTS, a single short hop away) clears
it once the spoken answer has actually finished, resuming episode playback.
"""

from __future__ import annotations

import asyncio

import pyaudio

from pipecat.frames.frames import EndFrame, Frame, LLMFullResponseEndFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import Model as WhisperModel
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from podcast_gen.kokoro_tts_service import KokoroTTSService
from podcast_gen.live import OLLAMA_MODEL, AnswerEcho, ErrorEcho, PushToTalkGate, TranscriptEcho, _build_llm
from podcast_gen.script_gen import DialogueLine
from podcast_gen.tts_render import RenderedLine, render_dialogue

QA_SYSTEM_PROMPT_TEMPLATE = """\
You are a guest expert being interrupted live during a two-host podcast \
episode. The listener has paused the episode to ask you something, out loud.

Answer conversationally and concisely: 2-4 short spoken sentences, no lists, \
no markdown, no headers. If something isn't covered by the material, say so \
honestly rather than guessing. After you answer, the episode will resume.

SOURCE MATERIAL:
{source_text}
"""


class PlaybackGate(FrameProcessor):
    """Sits right after the Q&A pipeline's TTS. Clears `pause_event` once the
    spoken answer has *fully* finished, resuming episode playback. (The event
    is set directly from the keypress by `PushToTalkGate`, not from here —
    see the module docstring for why.)

    Pipecat's TTS services synthesize per sentence (`TextAggregationMode.
    SENTENCE`, the default), so a multi-sentence answer produces one
    TTSStartedFrame/TTSStoppedFrame pair *per sentence* — not one for the
    whole answer. Clearing on the first TTSStoppedFrame (or even a simple
    started/stopped tally) resumes the episode too early: there's no way to
    know from outside the TTS service whether another sentence is still
    queued up behind the one that just finished.

    Instead, every TTSStoppedFrame (re)starts a short debounce timer; only
    once that timer elapses with no further TTS activity — and the LLM has
    actually finished generating text (LLMFullResponseEndFrame) — does this
    resume playback. A burst of consecutive sentences keeps re-arming the
    timer, so it only fires once the answer is genuinely done.
    """

    RESUME_DEBOUNCE_SECONDS = 0.6

    def __init__(self, pause_event: asyncio.Event):
        super().__init__()
        self._pause_event = pause_event
        self._response_text_done = False
        self._pending_resume_task: asyncio.Task | None = None

    def _schedule_resume_check(self) -> None:
        if self._pending_resume_task is not None:
            self._pending_resume_task.cancel()
        self._pending_resume_task = asyncio.create_task(self._resume_after_debounce())

    async def _resume_after_debounce(self) -> None:
        try:
            await asyncio.sleep(self.RESUME_DEBOUNCE_SECONDS)
        except asyncio.CancelledError:
            print("[trace] debounce timer cancelled (more TTS activity arrived)")
            return
        print(f"[trace] debounce elapsed, response_text_done={self._response_text_done}")
        if self._response_text_done:
            self._pause_event.clear()
            self._response_text_done = False
            print("[trace] pause_event CLEARED -> resuming episode playback")
        self._pending_resume_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame):
            print("[trace] LLMFullResponseEndFrame seen")
            self._response_text_done = True
            self._schedule_resume_check()
        elif isinstance(frame, TTSStoppedFrame):
            print("[trace] TTSStoppedFrame seen, (re)arming debounce")
            self._schedule_resume_check()
        await self.push_frame(frame, direction)


async def _play_episode(
    lines: list[RenderedLine],
    pause_event: asyncio.Event,
    output_device_index: int | None,
) -> None:
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pa.get_format_from_width(2),
        channels=1,
        rate=lines[0].sample_rate,
        output=True,
        output_device_index=output_device_index,
    )
    device_info = pa.get_device_info_by_index(
        output_device_index if output_device_index is not None else pa.get_default_output_device_info()["index"]
    )
    print(f"[trace] episode playback opened on device: {device_info['name']} @ {lines[0].sample_rate}Hz")
    loop = asyncio.get_running_loop()
    chunk_bytes = int(lines[0].sample_rate / 50) * 2  # 20ms of 16-bit mono

    try:
        for line in lines:
            print(f"\n[{line.speaker}]: {line.text}")
            pos = 0
            while pos < len(line.audio):
                if pause_event.is_set():
                    print("[trace] episode playback PAUSED")
                    while pause_event.is_set():
                        await asyncio.sleep(0.05)
                    print("[trace] episode playback RESUMING")
                chunk = line.audio[pos : pos + chunk_bytes]
                await loop.run_in_executor(None, stream.write, chunk)
                pos += chunk_bytes
            await asyncio.sleep(0.3)  # short gap between lines
        print("\n[Episode finished. You can keep asking questions, or Ctrl+C to quit.]")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


async def run_interactive_episode(
    source_text: str,
    dialogue_lines: list[DialogueLine],
    *,
    voice_map: dict[str, str],
    lang_code: str = "a",
    whisper_model: WhisperModel = WhisperModel.BASE,
    llm_backend: str = "anthropic",
    ollama_model: str = OLLAMA_MODEL,
    input_device_index: int | None = None,
    output_device_index: int | None = None,
) -> None:
    print("Rendering episode audio with Kokoro (this happens once, up front)...")
    rendered = await render_dialogue(dialogue_lines, voice_map=voice_map, lang_code=lang_code)
    print(f"Rendered {len(rendered)} lines.\n")

    system_prompt = QA_SYSTEM_PROMPT_TEMPLATE.format(source_text=source_text)
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    context_aggregator = LLMContextAggregatorPair(context)

    llm = _build_llm(llm_backend, ollama_model)
    stt = WhisperSTTService(settings=WhisperSTTService.Settings(model=whisper_model))
    tts = KokoroTTSService(voice=voice_map.get("HOST_A", "af_heart"), lang_code=lang_code)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
        )
    )

    pause_event = asyncio.Event()
    gate = PushToTalkGate(pause_event=pause_event)
    error_echo = ErrorEcho()
    transcript_echo = TranscriptEcho()
    answer_echo = AnswerEcho()
    playback_gate = PlaybackGate(pause_event)

    qa_pipeline = Pipeline(
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
            playback_gate,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    task = PipelineTask(qa_pipeline)
    runner = PipelineRunner()

    print("Ready. The episode will play through your speakers.")
    print("Press Enter at any time to pause it and ask a question (Ctrl+C to quit).\n")

    try:
        await asyncio.gather(
            runner.run(task),
            _play_episode(rendered, pause_event, output_device_index),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await task.queue_frames([EndFrame()])
