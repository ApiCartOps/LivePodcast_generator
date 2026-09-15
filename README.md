# LivePodcast Generator

[![CI](https://github.com/ApiCartOps/LivePodcast_generator/actions/workflows/ci.yml/badge.svg)](https://github.com/ApiCartOps/LivePodcast_generator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![LivePodcast Generator](.github/assets/social-preview.png)

Turn a Confluence page, any web page, or a local file (`.txt`/`.md`/`.pdf`)
into a two-host, NotebookLM-style podcast episode — and, unlike a plain
"audio overview," actually talk to it: interrupt the episode mid-playback
with a spoken question and get a live spoken answer before it resumes.

Four ways to use it:

| Mode | What it does | Entry point |
|---|---|---|
| **Web control panel** | A browser UI to generate episodes and manage a library of past ones | `podcast_gen/cli_web.py` |
| **Batch generator** | Renders a full episode to an MP3 file, from the command line | `podcast_gen/cli.py` |
| **Interactive episode** | Plays the episode aloud; press Enter any time to interrupt with a spoken question, then it resumes | `podcast_gen/cli_interactive.py` |
| **Standalone live Q&A** | No episode playback — just a live spoken conversation about the document | `podcast_gen/cli_live.py` |

The web control panel and batch generator only need the base install — no
mic/speaker hardware involved. The interactive episode and live Q&A modes
need real local audio devices and the `live` extra (see
[Installation](#installation)).

Every voice is synthesized locally with the open-weight
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model through a real
[Pipecat](https://pipecat.ai) TTS pipeline — no cloud TTS API, ever. The
script-writing and question-answering LLM can be either Anthropic's Claude
(cloud) or a local [Ollama](https://ollama.com) model, your choice — see
[LLM backends](#llm-backends) below. Speech recognition (for the live modes)
uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), also fully
local.

## Table of contents

- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Web control panel](#web-control-panel)
- [LLM backends](#llm-backends)
- [Usage](#usage)
- [Multi-guest panels](#multi-guest-panels)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [License](#license)

## How it works

1. **Load** the source (`podcast_gen/sources.py`): Confluence REST API, a
   generic URL (scraped with BeautifulSoup), or a local file.
2. **Script** (`podcast_gen/script_gen.py`): an LLM (Claude or a local Ollama
   model) turns the text into a JSON dialogue for a panel: one `HOST` plus
   `--guests` guests (`GUEST_1`, `GUEST_2`, ...), each optionally given a
   display name — see [Multi-guest panels](#multi-guest-panels).
3. **Render** (`podcast_gen/kokoro_tts_service.py`, `tts_render.py`): each
   line runs through a genuine Pipecat `TTSService` subclass backed by
   Kokoro (same shape as Pipecat's built-in Piper/ElevenLabs/Cartesia
   services, just local instead of cloud/server-based).
4. **Mix** (`podcast_gen/mix.py`): lines are concatenated in order with
   short gaps and encoded to MP3 via `ffmpeg`.
5. **Web control panel** (`podcast_gen/web/`): a FastAPI app wraps steps
   1-4 as a trackable background job with progress reporting, plus a
   filesystem-backed library of past episodes — see
   [Web control panel](#web-control-panel).
6. **Interactive episode** (`podcast_gen/interactive_episode.py`): instead
   of just muxing to a file, the rendered lines are played aloud through
   your speakers via PyAudio, while a second, live Pipecat pipeline (mic →
   Whisper STT → LLM → Kokoro TTS → speaker) listens for a keypress. Press
   Enter at any point and episode playback pauses immediately; ask your
   question, get a spoken answer, and playback resumes exactly where it
   left off.
7. **Standalone live Q&A** (`podcast_gen/live.py`): the same live pipeline
   as above, without ever rendering/playing the episode — just a
   conversation about the source document.

## Prerequisites

- **Python 3.12** specifically. Kokoro's dependency chain (`spacy`/`blis`)
  doesn't yet build cleanly on Python 3.14, and 3.12 is the version this was
  built and tested against.
- **`ffmpeg`** — required to encode the final MP3 (all modes). On macOS:
  `brew install ffmpeg`; the Docker image installs it automatically.
- One of:
  - An **Anthropic API key** (for Claude as the script-writing/Q&A LLM), or
  - A reachable **[Ollama](https://ollama.com)** server with at least one
    model pulled (for a fully offline, no-API-key setup) — see
    [LLM backends](#llm-backends) for how the web UI/Docker deployment
    points at it.
- (Optional) A **Confluence Cloud API token**, only if you want to pull
  content directly from Confluence pages.
- **Only for the interactive episode / live Q&A modes** (not the web
  control panel or batch generator): macOS with `portaudio`
  (`brew install portaudio`) for local mic/speaker access via PyAudio —
  see the `live` extra below.

## Installation

```bash
brew install ffmpeg   # + portaudio if you also want the live/interactive modes

python3.12 -m venv .venv
source .venv/bin/activate

# Web control panel + batch generator (no mic/speaker hardware needed):
pip install -e .

# Add the interactive-episode / live Q&A CLI modes (needs local audio hardware):
pip install -e ".[live]"
# ...and on Apple Silicon, also add the mlx extra (see note below):
pip install -e ".[live,mlx]"

cp .env.example .env   # fill in ANTHROPIC_API_KEY and/or Confluence creds
```

> **Apple Silicon note:** Pipecat's Whisper module imports `mlx_whisper` at
> load time even when you only use the faster-whisper (CPU) backend that
> this project actually uses. The `mlx` extra installs it so the import
> doesn't fail; it isn't otherwise used at runtime here. Only relevant if
> you're also installing `[live]`.

First run of any command downloads the ~80M-parameter Kokoro weights
(cached under `~/.cache/huggingface` afterward) and, for the live modes, the
faster-whisper model you configure (`BASE` by default — small and fast).

## Web control panel

```bash
podcast-gen-web                      # http://127.0.0.1:8000
podcast-gen-web --host 0.0.0.0 --port 8080 --episodes-dir /data/episodes
```

Open the URL in a browser: paste text, a URL, or upload a `.txt`/`.md`/`.pdf`
file, configure the panel (guests, names, voices, LLM backend), and click
Generate. A progress bar tracks the job (`queued → loading_source →
writing_script → rendering → mixing → done`) via `GET /api/jobs/{id}`
polling; finished episodes show up in the Library section with an inline
player, a download link, and delete.

Config (flags or environment variables, flags win): `--host`/
`PODCAST_GEN_HOST` (default `127.0.0.1`), `--port`/`PODCAST_GEN_PORT`
(default `8000`), `--episodes-dir`/`PODCAST_GEN_EPISODES_DIR` (default
`./episodes`), `--ollama-base-url`/`OLLAMA_BASE_URL` (default
`http://localhost:11434` — override this if Ollama runs on a different
machine than the web app, e.g. in a deployed setup).

**Deploying with Docker:**

```bash
docker build -t livepodcast-generator .
docker run -p 8000:8000 -v $(pwd)/episodes:/data/episodes \
  --env-file .env livepodcast-generator
```

The image only needs the base install (no `live` extra, no portaudio) since
the live/mic modes aren't part of the web UI. If Ollama runs on your host
machine rather than inside the container, set `OLLAMA_BASE_URL` to
something reachable from inside Docker (e.g.
`http://host.docker.internal:11434` on Docker Desktop).

This is a single-user, self-hosted control panel: no auth, no multi-user
support, in-memory job tracking (jobs don't survive a restart, but finished
episodes on disk do). Fine for local/trusted-network use; add your own
auth layer in front of it (e.g. a reverse proxy) before exposing it publicly.

## LLM backends

The script-writer (`cli.py`, `cli_interactive.py`, and the web control
panel's `/api/generate`) and the live Q&A answerer (`cli_live.py`,
`cli_interactive.py`) all support two backends, chosen with `--llm-backend`
(a form field in the web UI):

| Backend | Flag | Requires | Cost | Notes |
|---|---|---|---|---|
| **Anthropic Claude** (default) | `--llm-backend anthropic` | `ANTHROPIC_API_KEY` in `.env` | Paid, per-token API usage | Higher quality, especially for longer/more nuanced scripts |
| **Ollama** (local) | `--llm-backend ollama --ollama-model <name>` | A running local Ollama server (`ollama serve`, usually already running as a background service after installing Ollama) with the model pulled (e.g. `ollama pull llama3.1`) | Free, fully offline | No API key, no data leaves your machine, but quality depends on the local model and your hardware |

With the Ollama backend, **the entire pipeline runs 100% locally** — no
network calls at all except an optional Confluence/web fetch of the source
document itself.

Any [Ollama-supported model](https://ollama.com/library) works in principle;
`llama3.1` (8B) is a reasonable default for a modern Mac. Larger models give
better answers but are slower and need more RAM.

## Usage

### Batch MP3 generator

```bash
# From a Confluence page
python -m podcast_gen.cli "https://yourteam.atlassian.net/wiki/spaces/KB/pages/12345/Onboarding" -o onboarding.mp3

# From any web page
python -m podcast_gen.cli "https://example.com/some-article" -o episode.mp3

# From a local file, fully offline with Ollama
python -m podcast_gen.cli sample_doc.md -o episode.mp3 --llm-backend ollama --ollama-model llama3.1
```

Flags: `--minutes N` (target length), `--topic "..."` (steer focus),
`--script-json script.json` (cache the script; re-run with the same path to
re-render audio without calling the LLM again — pass `--regenerate-script`
to force a fresh one), `--llm-backend`/`--ollama-model` (see above),
`--guests`/`--guest-names`/`--voices` (see
[Multi-guest panels](#multi-guest-panels)).

### Interactive episode (barge in on the podcast)

```bash
python -m podcast_gen.cli_interactive sample_doc.md --llm-backend ollama --ollama-model llama3.1
```

Plays the episode through your speakers. Press Enter at any point to pause
it, ask a question out loud, press Enter again when you're done talking, and
the answer plays back — then the episode resumes exactly where it paused.

### Standalone live Q&A (no episode playback)

```bash
python -m podcast_gen.cli_live sample_doc.md --llm-backend ollama --ollama-model llama3.1
```

Same push-to-talk mechanic, but there's no episode — just a live spoken
conversation about the document.

### Common flags (both live modes)

- `--list-devices` — list audio input/output devices and exit (use this if
  you don't hear anything).
- `--input-device-index` / `--output-device-index` — pin a specific mic or
  speaker instead of the system default.

The Whisper model size (`WhisperModel.BASE` by default — small and fast) is
currently a code-level default in `live.py`/`interactive_episode.py` rather
than a CLI flag; edit `whisper_model=` there for a larger/more accurate model.

## Multi-guest panels

Both `cli.py` and `cli_interactive.py` build a panel of one `HOST` plus one
or more guests (`podcast_gen/participants.py`), rather than a fixed pair of
hosts. By default there's 1 guest (a two-person conversation, same as
before); raise it for a panel discussion:

```bash
# Three-person panel, guests given real names
python -m podcast_gen.cli sample_doc.md --guests 2 --guest-names "Priya,Marcus" \
  --llm-backend ollama --ollama-model llama3.1

# Pin specific voices: host first, then each guest in order
python -m podcast_gen.cli sample_doc.md --guests 2 --guest-names "Priya,Marcus" \
  --voices "af_heart,bf_emma,am_adam"
```

- `--guests N` — number of guests besides the host (default: 1).
- `--guest-names "Name1,Name2,..."` — display names used in the dialogue
  text itself (must match `--guests` in count). Defaults to "Guest 1",
  "Guest 2", etc.
- `--voices "host,guest1,guest2,..."` — Kokoro voice ids, host first (must
  have exactly `--guests + 1` entries). Defaults to `af_heart` for the host
  and a rotation of distinct voices (`am_michael`, `bf_emma`, `af_nova`,
  `bm_george`, `am_adam`, `bf_alice`, cycling if you have more guests than
  that) — see `DEFAULT_GUEST_VOICE_ROTATION` in `participants.py`. Kokoro
  ships many more voices — see the
  [voices list](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).

The host is instructed to bring every guest into the conversation at least
once — it's a panel, not one long back-and-forth with a single guest. In
the interactive episode, the host's voice also answers your interruptions.

Non-English content needs a different `--lang-code` / `lang_code=...`
matching Kokoro's supported language codes (applies to every voice in the
panel).

> **Note:** speaker labels changed from the fixed `HOST_A`/`HOST_B` to
> `HOST`/`GUEST_1`/`GUEST_2`/... to support an arbitrary number of guests.
> A `--script-json` file saved before this change won't load correctly —
> regenerate it with `--regenerate-script`.

## Troubleshooting

- **No sound at all** — run with `--list-devices` and confirm the expected
  output device is your speakers, not something else (e.g. a virtual audio
  device from another app). Try `--output-device-index` to force it.
- **Ctrl+C doesn't exit** — should be fixed as of this version (the shutdown
  path is timeout-bounded); if you still see it hang, check for orphaned
  processes from a previous run (`ps aux | grep cli_interactive`) and kill
  them before starting a new one, since two processes writing to the same
  audio device at once can look like a second "stuck" session.
- **Episode audio overlaps the spoken answer** — this was a real bug during
  development (Kokoro synthesizes audio much faster than it takes to play,
  so naively resuming on "synthesis done" resumed the episode while the
  answer was still audibly playing). Fixed by keying off
  `BotStoppedSpeakingFrame` (real playback completion) instead of
  `TTSStoppedFrame` (synthesis completion) with a debounce for multi-sentence
  answers. If you still hit this, please open an issue with the console
  output.
- **Whisper mis-hears things** — the `BASE` model is small/fast; pass a
  larger `WhisperModel` (e.g. `SMALL`, `MEDIUM`) in code for better accuracy
  at the cost of latency.
- **A local model's script uses the wrong speaker labels** — local models
  (tested with Ollama's `llama3.1`) sometimes write a guest's display name
  (e.g. `"PRIYA"`) as the JSON `speaker` field instead of the exact key
  (`"GUEST_1"`), which would otherwise crash rendering with no matching
  voice. `script_gen.py` now normalizes this automatically (matching
  against display names too, case-insensitively, before falling back to
  the host) — you shouldn't need to do anything, but it's why you may see
  a `"Unrecognized speaker"` warning in the logs occasionally.
- **A line is silently missing from the episode** — Kokoro/Pipecat can
  intermittently return zero audio for a line with no clear pattern (not
  tied to short text — the same line normally succeeds on retry).
  `tts_render.py` retries synthesis up to 3 times on empty audio before
  raising loudly, so this should now surface as an error rather than a
  silently dropped line.

## Dependencies

Installed automatically via `pyproject.toml`:

| Package | Purpose | License |
|---|---|---|
| [pipecat-ai](https://pipecat.ai) | Real-time voice pipeline framework (STT/LLM/TTS orchestration, transports) | BSD-2-Clause |
| [kokoro](https://github.com/hexgrad/kokoro) | Local TTS engine (wraps the Kokoro-82M weights) | Apache-2.0 |
| [anthropic](https://github.com/anthropics/anthropic-sdk-python) | Claude API client (optional LLM backend) | MIT |
| [fastapi](https://fastapi.tiangolo.com) | Web control panel backend | MIT |
| [uvicorn](https://www.uvicorn.org) | ASGI server for the web control panel | BSD-3-Clause |
| [python-multipart](https://github.com/Kludex/python-multipart) | Multipart form/file upload parsing (FastAPI) | Apache-2.0 |
| [soundfile](https://github.com/bastibe/python-soundfile) | WAV encoding/decoding | BSD-3-Clause |
| [numpy](https://numpy.org) | Audio array manipulation | BSD-3-Clause |
| [requests](https://requests.readthedocs.io) | HTTP for web/Confluence/Ollama fetching | Apache-2.0 |
| [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) | HTML → text extraction | MIT |
| [pypdf](https://github.com/py-pdf/pypdf) | PDF text extraction | BSD-3-Clause |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` loading | BSD-3-Clause |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`[live]` extra) | Local speech-to-text, live/interactive modes only | MIT |
| [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/) (`[live]` extra) | Local mic/speaker I/O (via PortAudio), live/interactive modes only | MIT |
| `mlx_whisper` (Apple Silicon only, `[mlx]` extra, alongside `[live]`) | Satisfies a Pipecat import; not otherwise used here | MIT |

System dependencies (via Homebrew, not pip):

| Package | Purpose | License |
|---|---|---|
| [ffmpeg](https://ffmpeg.org) | MP3 encoding, all modes | LGPL/GPL depending on build configuration — Homebrew's default build includes some GPL-licensed codecs (e.g. x264), making the resulting binary effectively GPL. Review [ffmpeg's licensing page](https://ffmpeg.org/legal.html) if this matters for your use case. |
| [portaudio](http://www.portaudio.com) | Cross-platform audio I/O, used by PyAudio — only needed for the `[live]` extra (interactive episode / live Q&A modes) | MIT-style (PortAudio license) |

Not a dependency, but relevant to what you'll actually be running:

| Model/service | Used for | License / terms |
|---|---|---|
| **Kokoro-82M weights** (downloaded from Hugging Face on first run) | TTS voices | Apache-2.0 |
| **Anthropic Claude** (if using `--llm-backend anthropic`) | Script writing / Q&A | Governed by [Anthropic's commercial API terms](https://www.anthropic.com/legal/commercial-terms) — this is a paid, hosted API, not open-source software. |
| **Ollama** (if using `--llm-backend ollama`) | Runs local LLMs | Ollama itself: MIT. Whatever model you pull through it (e.g. Llama 3.1) carries **its own** license — e.g. Meta's [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/), which includes acceptable-use restrictions. Check the license of any model you pull before using this commercially. |

## License

This project is licensed under the [MIT License](LICENSE) — see that file
for the full text. Note that some system/runtime dependencies above carry
different licenses (notably ffmpeg's typical GPL-encumbered build, and any
LLM model you run through Ollama), which may impose their own obligations
independent of this project's own MIT license.
