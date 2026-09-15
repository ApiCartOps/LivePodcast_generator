"""Episode generation job endpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from podcast_gen.participants import build_panel

from .jobs import GenerationRequest

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


@router.post("/api/generate")
async def generate_episode(
    request: Request,
    source_text: str | None = Form(None),
    source_url: str | None = Form(None),
    source_file: UploadFile | None = File(None),  # noqa: B008 - idiomatic FastAPI DI pattern
    guests: int = Form(1),
    guest_names: str | None = Form(None),
    voices: str | None = Form(None),
    topic: str | None = Form(None),
    minutes: int = Form(5),
    llm_backend: str = Form("anthropic"),
    ollama_model: str = Form("llama3.1"),
    title: str | None = Form(None),
):
    provided = [bool(source_text), bool(source_url), bool(source_file)]
    if sum(provided) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of source_text, source_url, or source_file.",
        )

    guest_names_list = _split_csv(guest_names)
    voices_list = _split_csv(voices)
    try:
        # Cheap, no I/O -- validates guests/guest_names/voices counts up
        # front instead of only discovering a mistake after a whole job
        # cycle runs. The job runner rebuilds the panel itself.
        build_panel(guests, guest_names=guest_names_list, voices=voices_list)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if source_text:
        source_kind, source_value = "text", source_text
    elif source_url:
        source_kind, source_value = "url", source_url
    else:
        data = await source_file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file is too large (max 20MB).")
        suffix = Path(source_file.filename or "").suffix or ".txt"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        source_kind, source_value = "file", tmp_path

    gen_request = GenerationRequest(
        source_kind=source_kind,
        source_value=source_value,
        guests=guests,
        guest_names=guest_names_list,
        voices=voices_list,
        topic=topic,
        minutes=minutes,
        llm_backend=llm_backend,
        ollama_model=ollama_model,
        ollama_base_url=request.app.state.ollama_base_url,
        title=title,
    )

    # Everything past this point (source loading, LLM calls, rendering) can
    # still fail, but only inside the background job -- surfaced via
    # job.error, not as an HTTP error, since it's all asynchronous from here.
    job = await request.app.state.jobs.create_job(gen_request)
    return {"job_id": job.id}


@router.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    job = await request.app.state.jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.to_dict()
