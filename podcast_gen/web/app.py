"""FastAPI app factory for the podcast-generator web control panel."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from podcast_gen.script_gen import OLLAMA_BASE_URL

from . import routes_episodes, routes_jobs, routes_meta
from .episodes import EpisodeStore
from .jobs import JobManager

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    episodes_dir: str = "episodes",
    ollama_base_url: str = OLLAMA_BASE_URL,
) -> FastAPI:
    app = FastAPI(title="LivePodcast Generator")

    episodes = EpisodeStore(episodes_dir)
    app.state.episodes = episodes
    app.state.jobs = JobManager(episodes)
    app.state.ollama_base_url = ollama_base_url

    app.include_router(routes_meta.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_episodes.router)

    # Mounted last and at "/" so it doesn't shadow the /api/* routes above.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app
