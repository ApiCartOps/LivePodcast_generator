"""CLI: play a generated podcast episode aloud and let you interrupt it with
spoken questions, like NotebookLM's interactive mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from podcast_gen.interactive_episode import run_interactive_episode
from podcast_gen.live import list_audio_devices
from podcast_gen.script_gen import DialogueLine, generate_dialogue
from podcast_gen.sources import load_source

DEFAULT_VOICE_MAP = {
    "HOST_A": "af_heart",
    "HOST_B": "am_michael",
}


def main() -> None:
    load_dotenv()
    args = _parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    print(f"Loading content from: {args.source}", file=sys.stderr)
    source_text = load_source(args.source)
    print(f"Loaded {len(source_text)} characters.", file=sys.stderr)

    if args.script_json and Path(args.script_json).exists() and not args.regenerate_script:
        print(f"Reusing existing script: {args.script_json}", file=sys.stderr)
        raw = json.loads(Path(args.script_json).read_text())
        lines = [DialogueLine(**item) for item in raw]
    else:
        print(f"Generating dialogue script with {args.llm_backend}...", file=sys.stderr)
        lines = generate_dialogue(
            source_text,
            topic_hint=args.topic,
            target_minutes=args.minutes,
            backend=args.llm_backend,
            ollama_model=args.ollama_model,
        )
        if args.script_json:
            Path(args.script_json).write_text(
                json.dumps([line.__dict__ for line in lines], indent=2)
            )
            print(f"Saved script to: {args.script_json}", file=sys.stderr)

    asyncio.run(
        run_interactive_episode(
            source_text,
            lines,
            voice_map=DEFAULT_VOICE_MAP,
            llm_backend=args.llm_backend,
            ollama_model=args.ollama_model,
            input_device_index=args.input_device_index,
            output_device_index=args.output_device_index,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a generated podcast episode aloud; press Enter at any "
        "point to pause it and ask a spoken question, then it resumes."
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Confluence page URL/ID, a web URL, or a path to a local .txt/.md/.pdf file. "
        "Not needed with --list-devices.",
    )
    parser.add_argument("--topic", default=None, help="Optional hint about what the episode should focus on.")
    parser.add_argument(
        "--minutes", type=int, default=5, help="Roughly how long the episode should be (default: 5)."
    )
    parser.add_argument(
        "--script-json",
        default=None,
        help="Path to cache/reuse the generated dialogue script as JSON.",
    )
    parser.add_argument(
        "--regenerate-script",
        action="store_true",
        help="Force regenerating the script even if --script-json already exists.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="LLM for both writing the script and answering questions (default: anthropic).",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.1",
        help="Ollama model name to use when --llm-backend=ollama (default: llama3.1).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input/output devices and exit.",
    )
    parser.add_argument("--input-device-index", type=int, default=None)
    parser.add_argument("--output-device-index", type=int, default=None)
    args = parser.parse_args()
    if not args.list_devices and not args.source:
        parser.error("source is required unless --list-devices is given")
    return args


if __name__ == "__main__":
    main()
