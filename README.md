# podcast-generator

Turn a Confluence page, any web page, or a local file into a two-host,
NotebookLM-style podcast episode (MP3) — script written by Claude, voices
rendered fully locally with the open-weight [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
model through a real [Pipecat](https://pipecat.ai) TTS pipeline. No cloud TTS
API, no LiveKit/real-time transport — just a batch CLI.

## How it works

1. **Load** the source (`podcast_gen/sources.py`): Confluence REST API, a
   generic URL (scraped with BeautifulSoup), or a local `.txt`/`.md`/`.pdf`.
2. **Script** (`podcast_gen/script_gen.py`): Claude turns the text into a
   JSON dialogue between `HOST_A` and `HOST_B`.
3. **Render** (`podcast_gen/kokoro_tts_service.py`, `tts_render.py`): each
   line runs through its own `Pipeline([KokoroTTSService(voice=...), sink])`
   — a `TTSSpeakFrame` goes in, a `TTSAudioRawFrame` comes out. This is a
   genuine Pipecat `TTSService` subclass (same shape as Pipecat's built-in
   Piper/ElevenLabs/Cartesia services), just backed by Kokoro instead of a
   cloud or server-based engine.
4. **Mix** (`podcast_gen/mix.py`): lines are concatenated in order with short
   gaps and encoded to MP3 via `ffmpeg`.

## Setup

```bash
brew install portaudio ffmpeg   # if not already installed
uv sync                          # or: python3.12 -m venv .venv && pip install -e ".[mlx]"
cp .env.example .env             # fill in ANTHROPIC_API_KEY (and Confluence creds if needed)
```

Requires Python 3.12 (Kokoro's dependency chain doesn't yet build cleanly on 3.14).
The `mlx` extra is only needed on Apple Silicon — pipecat's Whisper module
imports `mlx_whisper` at load time even when you just want the faster-whisper
(CPU) backend used here.

## Usage

```bash
# From a Confluence page
python -m podcast_gen.cli "https://yourteam.atlassian.net/wiki/spaces/KB/pages/12345/Onboarding" -o onboarding.mp3

# From any web page
python -m podcast_gen.cli "https://example.com/some-article" -o episode.mp3

# From a local file
python -m podcast_gen.cli ./notes.pdf -o episode.mp3 --minutes 8 --topic "focus on the migration risks"
```

Useful flags:

- `--minutes N` — rough target episode length (default 5).
- `--topic "..."` — steer what the episode emphasizes.
- `--script-json script.json` — cache the generated dialogue; re-run with the
  same path to re-render audio (e.g. after tweaking voices) without calling
  Claude again. Pass `--regenerate-script` to force a fresh script.

## Interactive Q&A (like NotebookLM's "interactive mode")

Instead of a rendered MP3, `cli_live.py` starts a live, spoken conversation
about the document: it's a real Pipecat pipeline running against your Mac's
mic/speakers —

```
mic -> push-to-talk gate -> Whisper STT -> Claude (doc as context) -> Kokoro TTS -> speakers
```

```bash
python -m podcast_gen.cli_live ./notes.pdf
python -m podcast_gen.cli_live "https://yourteam.atlassian.net/wiki/spaces/KB/pages/12345/Onboarding" --voice am_michael
```

Press Enter, ask your question out loud, press Enter again when you're done
talking, and the answer plays back through your speakers. This uses
push-to-talk rather than pipecat's automatic VAD/turn-detection stack — a
background thread injects the `VADUserStartedSpeakingFrame` /
`VADUserStoppedSpeakingFrame` markers that `WhisperSTTService` needs on
keypress instead of running a VAD model, which keeps this piece simple and
avoids depending on pipecat's newer (and still-evolving) turn-management
subsystem. Everything runs locally except the Claude API call.

## Voices

Default voice map (`podcast_gen/cli.py`, `DEFAULT_VOICE_MAP`): `HOST_A` =
`af_heart`, `HOST_B` = `am_michael`. Kokoro ships many more built-in voices —
see the [voices list](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
Non-English content needs a different `lang_code` passed through
`render_dialogue(..., lang_code=...)`.

## Notes

- First run downloads the ~80M-parameter Kokoro weights (cached afterward).
- Confluence auth uses an API token: https://id.atlassian.com/manage-profile/security/api-tokens
