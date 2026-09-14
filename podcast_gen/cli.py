"""CLI: turn a Confluence page, a web URL, or a local file into a host + N guest podcast MP3."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from podcast_gen.mix import mix_to_mp3
from podcast_gen.participants import build_panel, voice_map
from podcast_gen.script_gen import DialogueLine, generate_dialogue
from podcast_gen.sources import load_source
from podcast_gen.tts_render import render_dialogue


def main() -> None:
    load_dotenv()
    args = _parse_args()

    guest_names = _split_csv(args.guest_names)
    voices = _split_csv(args.voices)
    panel = build_panel(args.guests, guest_names=guest_names, voices=voices)
    print(
        f"Panel: {panel[0].display_name} (host) + "
        f"{', '.join(p.display_name for p in panel[1:])}",
        file=sys.stderr,
    )

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
            panel,
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

    print(f"Script has {len(lines)} lines. Rendering audio with Kokoro via Pipecat...", file=sys.stderr)
    rendered = asyncio.run(render_dialogue(lines, voice_map=voice_map(panel)))

    print(f"Mixing final episode -> {args.output}", file=sys.stderr)
    mix_to_mp3(rendered, args.output)
    print(f"Done: {args.output}", file=sys.stderr)


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a host + N guest podcast episode from a Confluence page, "
        "web URL, or local file."
    )
    parser.add_argument(
        "source",
        help="Confluence page URL/ID, a web URL, or a path to a local .txt/.md/.pdf file.",
    )
    parser.add_argument("-o", "--output", default="episode.mp3", help="Output MP3 path.")
    parser.add_argument(
        "--topic",
        default=None,
        help="Optional hint about what the episode should focus on.",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=5,
        help="Roughly how long the episode should be, in minutes (default: 5).",
    )
    parser.add_argument(
        "--guests",
        type=int,
        default=1,
        help="Number of guests on the panel, besides the host (default: 1, i.e. "
        "a two-person conversation). Try 2-3 for a panel discussion.",
    )
    parser.add_argument(
        "--guest-names",
        default=None,
        help="Comma-separated display names for the guests, in order (e.g. "
        "'Alice,Bob'). Must match --guests in count. Defaults to 'Guest 1', "
        "'Guest 2', etc.",
    )
    parser.add_argument(
        "--voices",
        default=None,
        help="Comma-separated Kokoro voice ids, host first then each guest in "
        "order (must have exactly --guests + 1 entries). Defaults to a "
        "built-in rotation of distinct voices.",
    )
    parser.add_argument(
        "--script-json",
        default=None,
        help="Path to cache/reuse the generated dialogue script as JSON, "
        "so you can re-render audio without re-calling the LLM.",
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
        help="Which LLM writes the dialogue script: Claude (needs ANTHROPIC_API_KEY) "
        "or a local Ollama server (default: anthropic).",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.1",
        help="Ollama model name to use when --llm-backend=ollama (default: llama3.1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
