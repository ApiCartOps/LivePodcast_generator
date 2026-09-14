"""Turn source text into a host + N guest panel dialogue script, via Claude or a local Ollama model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic
import requests

from podcast_gen.participants import Participant

ANTHROPIC_MODEL = "claude-sonnet-5"
OLLAMA_MODEL = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class DialogueLine:
    speaker: str
    text: str


def _build_system_prompt(participants: list[Participant]) -> str:
    host, guests = participants[0], participants[1:]
    guest_word = "guest" if len(guests) == 1 else "guests"
    guest_roster = "\n".join(
        f'- {g.speaker_key}, referred to in conversation as "{g.display_name}"' for g in guests
    )
    speaker_values = " | ".join(f'"{p.speaker_key}"' for p in participants)

    return f"""\
You write scripts for a panel-style explainer podcast episode, in the style \
of an AI "audio overview" (e.g. NotebookLM). One host interviews {len(guests)} \
{guest_word} about the source material, drawing out their perspectives, so a \
listener who hasn't read it comes away understanding it.

Participants:
- {host.speaker_key}, referred to in conversation as "{host.display_name}" — \
drives the conversation, asks clarifying questions, keeps things moving, and \
brings each guest in.
{guest_roster}

Rules:
- {host.speaker_key} should bring every guest into the conversation at least \
once — this is a panel, not one long back-and-forth with a single guest.
- Guests can react to each other, not just answer the host, when it's natural.
- Alternate speakers naturally; don't just cycle through them mechanically if \
a real conversation wouldn't.
- Open with a short, catchy intro that says what the episode covers and \
introduces each guest by name. Close with a brief wrap-up thanking everyone.
- Use casual, spoken language: contractions, short sentences, occasional \
asides. No bullet points, headers, or markdown — this is spoken audio.
- Do not invent facts that aren't supported by the source material.
- Keep it tight: prioritize the most important points over exhaustive coverage.

Output ONLY a JSON object, no other text, no markdown fences, of the form:
{{"lines": [{{"speaker": {speaker_values}, "text": "..."}}, ...]}}
"""


def generate_dialogue(
    source_text: str,
    participants: list[Participant],
    *,
    topic_hint: str | None = None,
    target_minutes: int = 5,
    backend: str = "anthropic",
    ollama_model: str = OLLAMA_MODEL,
) -> list[DialogueLine]:
    """Turn `source_text` into a list of DialogueLine for the given panel.

    backend: "anthropic" (Claude, needs ANTHROPIC_API_KEY) or "ollama" (a
    locally running Ollama server, fully offline).
    """
    system_prompt = _build_system_prompt(participants)

    user_prompt = (
        f"Target episode length: roughly {target_minutes} minutes of spoken audio "
        f"(~{target_minutes * 150} words total).\n"
    )
    if topic_hint:
        user_prompt += f"Episode focus: {topic_hint}\n"
    user_prompt += f"\nSOURCE MATERIAL:\n{source_text}"

    if backend == "anthropic":
        raw = _generate_with_anthropic(system_prompt, user_prompt)
    elif backend == "ollama":
        raw = _generate_with_ollama(system_prompt, user_prompt, model=ollama_model)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected 'anthropic' or 'ollama')")

    data = json.loads(_strip_code_fence(raw))
    return [DialogueLine(speaker=item["speaker"], text=item["text"]) for item in data["lines"]]


def _generate_with_anthropic(system_prompt: str, user_prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=8000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _generate_with_ollama(system_prompt: str, user_prompt: str, *, model: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            # Ollama's default num_predict is too small for a multi-speaker
            # script; without this, generation silently truncates mid-JSON.
            "options": {"num_predict": 4096},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text
