"""CLI: run the web control panel (generate & manage episodes from a browser)."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    args = _parse_args()

    import uvicorn

    from podcast_gen.web.app import create_app

    app = create_app(episodes_dir=args.episodes_dir, ollama_base_url=args.ollama_base_url)
    uvicorn.run(app, host=args.host, port=args.port)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LivePodcast Generator web control panel: generate and "
        "manage episodes from a browser instead of the CLI."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("PODCAST_GEN_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1, or $PODCAST_GEN_HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PODCAST_GEN_PORT", "8000")),
        help="Port to bind to (default: 8000, or $PODCAST_GEN_PORT).",
    )
    parser.add_argument(
        "--episodes-dir",
        default=os.environ.get("PODCAST_GEN_EPISODES_DIR", "episodes"),
        help="Directory to store generated episodes in "
        "(default: ./episodes, or $PODCAST_GEN_EPISODES_DIR).",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL of the Ollama server to use when a job selects the ollama "
        "backend (default: http://localhost:11434, or $OLLAMA_BASE_URL).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
