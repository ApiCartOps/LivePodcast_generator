"""Episode library endpoints: list, detail, audio streaming, delete."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/api/episodes")
async def list_episodes(request: Request):
    return request.app.state.episodes.list_episodes()


@router.get("/api/episodes/{episode_id}")
async def get_episode(episode_id: str, request: Request):
    episode = request.app.state.episodes.get_episode(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found.")
    return episode


@router.get("/api/episodes/{episode_id}/audio")
async def get_episode_audio(episode_id: str, request: Request):
    episodes = request.app.state.episodes
    if episodes.get_episode(episode_id) is None:
        raise HTTPException(status_code=404, detail="Episode not found.")
    mp3_path = episodes.mp3_path(episode_id)
    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail="Episode audio file is missing.")
    # FileResponse handles HTTP range requests automatically, so <audio> can
    # seek and this same URL doubles as the download link.
    return FileResponse(mp3_path, media_type="audio/mpeg", filename=f"{episode_id}.mp3")


@router.delete("/api/episodes/{episode_id}", status_code=204)
async def delete_episode(episode_id: str, request: Request):
    deleted = request.app.state.episodes.delete_episode(episode_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Episode not found.")
