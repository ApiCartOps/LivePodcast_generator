"""CLI: live voice Q&A about a Confluence page, web page, or local file."""

from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from podcast_gen.live import run_live_qa
from podcast_gen.sources import load_source


def main() -> None:
    load_dotenv()
    args = _parse_args()

    print(f"Loading content from: {args.source}", file=sys.stderr)
    source_text = load_source(args.source)
    print(f"Loaded {len(source_text)} characters.", file=sys.stderr)

    asyncio.run(
        run_live_qa(
            source_text,
            voice=args.voice,
            lang_code=args.lang_code,
            llm_backend=args.llm_backend,
            ollama_model=args.ollama_model,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a live, spoken Q&A session about a document: ask "
        "questions out loud, get spoken answers back."
    )
    parser.add_argument(
        "source",
        help="Confluence page URL/ID, a web URL, or a path to a local .txt/.md/.pdf file.",
    )
    parser.add_argument(
        "--voice",
        default="af_heart",
        help="Kokoro voice for the answering host (default: af_heart).",
    )
    parser.add_argument(
        "--lang-code",
        default="a",
        help="Kokoro language code (default: 'a' for American English).",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="Which LLM answers your questions: Claude (needs ANTHROPIC_API_KEY) "
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
