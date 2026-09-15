"""Misc endpoints: voice list and a liveness check."""

from __future__ import annotations

from fastapi import APIRouter

from podcast_gen.participants import DEFAULT_GUEST_VOICE_ROTATION, DEFAULT_HOST_VOICE

router = APIRouter()


@router.get("/api/voices")
def list_voices():
    all_voices = [DEFAULT_HOST_VOICE] + [
        v for v in DEFAULT_GUEST_VOICE_ROTATION if v != DEFAULT_HOST_VOICE
    ]
    return {
        "voices": [{"id": v} for v in all_voices],
        "default_host_voice": DEFAULT_HOST_VOICE,
        "default_guest_rotation": DEFAULT_GUEST_VOICE_ROTATION,
    }


@router.get("/api/health")
def health():
    return {"status": "ok"}
