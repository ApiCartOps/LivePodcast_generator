"""Builds a host + N guest panel: speaker labels, display names, and voices.

Shared by the batch generator and the interactive episode player so both
build the same panel from the same `--guests`/`--guest-names`/`--voices`
flags.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HOST_VOICE = "af_heart"

# A rotation of distinct-sounding Kokoro voices (mixed genders/accents) for
# guests beyond the host. Cycles if there are more guests than voices here.
DEFAULT_GUEST_VOICE_ROTATION = [
    "am_michael",
    "bf_emma",
    "af_nova",
    "bm_george",
    "am_adam",
    "bf_alice",
]


@dataclass
class Participant:
    speaker_key: str  # e.g. "HOST", "GUEST_1" -- used for voice_map keys and DialogueLine.speaker
    display_name: str  # e.g. "the host", "Alice" -- used in the prompt so the LLM can address them by name
    voice: str  # Kokoro voice id


def build_panel(
    num_guests: int,
    *,
    guest_names: list[str] | None = None,
    voices: list[str] | None = None,
) -> list[Participant]:
    """Build [HOST, GUEST_1, ..., GUEST_n] with display names and voices."""
    if num_guests < 1:
        raise ValueError("Need at least 1 guest.")
    if guest_names is not None and len(guest_names) != num_guests:
        raise ValueError(
            f"--guest-names must have exactly {num_guests} name(s), got {len(guest_names)}."
        )
    if voices is not None and len(voices) != num_guests + 1:
        raise ValueError(
            f"--voices must have exactly {num_guests + 1} voice(s) "
            f"(host + {num_guests} guest(s)), got {len(voices)}."
        )

    participants = [
        Participant(
            speaker_key="HOST",
            display_name="the host",
            voice=voices[0] if voices else DEFAULT_HOST_VOICE,
        )
    ]
    for i in range(num_guests):
        key = f"GUEST_{i + 1}"
        name = guest_names[i] if guest_names else key.replace("_", " ").title()
        voice = (
            voices[i + 1]
            if voices
            else DEFAULT_GUEST_VOICE_ROTATION[i % len(DEFAULT_GUEST_VOICE_ROTATION)]
        )
        participants.append(Participant(speaker_key=key, display_name=name, voice=voice))

    return participants


def voice_map(participants: list[Participant]) -> dict[str, str]:
    return {p.speaker_key: p.voice for p in participants}
