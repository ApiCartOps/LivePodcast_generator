"""Background job tracking for episode generation.

A `POST /api/generate` request creates a `Job`, kicks off
`run_generation_job` as a background asyncio task, and returns immediately;
the frontend polls `GET /api/jobs/{id}` until the job reaches `done` or
`failed`. Jobs are ephemeral (in-memory only) -- only the finished episode
itself needs to persist, and that's handled by `EpisodeStore`.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from loguru import logger

from podcast_gen.mix import mix_to_mp3
from podcast_gen.participants import Participant, build_panel, voice_map
from podcast_gen.script_gen import generate_dialogue
from podcast_gen.sources import load_source
from podcast_gen.tts_render import render_dialogue

from .episodes import EpisodeStore


class JobStatus(StrEnum):
    QUEUED = "queued"
    LOADING_SOURCE = "loading_source"
    WRITING_SCRIPT = "writing_script"
    RENDERING = "rendering"
    MIXING = "mixing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    lines_done: int = 0
    lines_total: int = 0
    error: str | None = None
    episode_id: str | None = None
    created_at: float = field(default_factory=time.time)

    def update_progress(self, done: int, total: int) -> None:
        self.lines_done = done
        self.lines_total = total

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "lines_done": self.lines_done,
            "lines_total": self.lines_total,
            "error": self.error,
            "episode_id": self.episode_id,
            "created_at": self.created_at,
        }


@dataclass
class GenerationRequest:
    # Exactly one of these three should be meaningful, per source_kind.
    source_kind: str  # "text" | "url" | "file"
    source_value: str  # raw text, a URL, or a path to a saved temp upload

    guests: int
    guest_names: list[str] | None
    voices: list[str] | None
    topic: str | None
    minutes: int
    llm_backend: str
    ollama_model: str
    ollama_base_url: str
    title: str | None


class JobManager:
    """Tracks in-flight/completed jobs and limits render concurrency."""

    # Kokoro/ffmpeg are CPU-bound per line; unbounded concurrency would just
    # slow every in-flight job down together, so cap how many render at once.
    MAX_CONCURRENT_RENDERS = 2

    def __init__(self, episodes: EpisodeStore):
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._render_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_RENDERS)
        self._episodes = episodes

    async def create_job(self, request: GenerationRequest) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        async with self._lock:
            self._jobs[job.id] = job
        asyncio.create_task(self._run(job, request))
        return job

    async def get_job(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def _run(self, job: Job, request: GenerationRequest) -> None:
        try:
            async with self._render_semaphore:
                await run_generation_job(job, request, self._episodes)
        except Exception as e:  # noqa: BLE001 - a job failure must never crash the server
            logger.exception(f"Job {job.id} failed")
            job.status = JobStatus.FAILED
            job.error = str(e)


async def run_generation_job(job: Job, request: GenerationRequest, episodes: EpisodeStore) -> None:
    job.status = JobStatus.LOADING_SOURCE
    if request.source_kind == "text":
        source_text = request.source_value
    else:
        # URL (web/Confluence) or a saved upload path -- both go through the
        # same auto-detecting loader the CLI uses.
        source_text = await asyncio.to_thread(load_source, request.source_value)

    if request.source_kind == "file":
        # Clean up the temp upload now that its contents are loaded.
        try:
            os.remove(request.source_value)
        except OSError:
            pass

    panel: list[Participant] = build_panel(
        request.guests, guest_names=request.guest_names, voices=request.voices
    )

    job.status = JobStatus.WRITING_SCRIPT
    lines = await asyncio.to_thread(
        generate_dialogue,
        source_text,
        panel,
        topic_hint=request.topic,
        target_minutes=request.minutes,
        backend=request.llm_backend,
        ollama_model=request.ollama_model,
        ollama_base_url=request.ollama_base_url,
    )

    job.status = JobStatus.RENDERING
    rendered = await render_dialogue(
        lines, voice_map=voice_map(panel), on_progress=job.update_progress
    )

    job.status = JobStatus.MIXING
    episode_id = uuid.uuid4().hex[:12]
    mp3_path = episodes.mp3_path(episode_id)
    await asyncio.to_thread(mix_to_mp3, rendered, str(mp3_path))

    title = request.title or _derive_title(request.topic, source_text)
    episodes.save_episode(
        episode_id=episode_id,
        title=title,
        source_snippet=source_text[:300],
        participants=panel,
        target_minutes=request.minutes,
        llm_backend=request.llm_backend,
        lines=lines,
    )

    job.episode_id = episode_id
    job.status = JobStatus.DONE


def _derive_title(topic: str | None, source_text: str) -> str:
    if topic:
        return topic.strip()[:80]
    snippet = " ".join(source_text.split())[:80].strip()
    return snippet or "Untitled episode"
