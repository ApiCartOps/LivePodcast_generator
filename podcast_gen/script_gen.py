"""Turn source text into a two-host podcast dialogue script, via Claude or a local Ollama model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic
import requests

ANTHROPIC_MODEL = "claude-sonnet-5"
OLLAMA_MODEL = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = """\
You write scripts for a two-host explainer podcast, in the style of an AI \
"audio overview" (e.g. NotebookLM). Given source material, produce a natural, \
engaging spoken conversation between two hosts, HOST_A and HOST_B, that \
explains the material to a listener who hasn't read it.

Rules:
- Alternate speakers naturally; don't just alternate mechanically every line \
  if a real back-and-forth wouldn't.
- HOST_A drives the conversation and asks clarifying questions; HOST_B has \
  more of the detailed knowledge, but both contribute.
- Open with a short, catchy intro that says what the episode covers. Close \
  with a brief wrap-up.
- Use casual, spoken language: contractions, short sentences, occasional \
  asides. No bullet points, headers, or markdown — this is spoken audio.
- Do not invent facts that aren't supported by the source material.
- Keep it tight: prioritize the most important points over exhaustive coverage.

Output ONLY a JSON object, no other text, no markdown fences, of the form:
{"lines": [{"speaker": "HOST_A" | "HOST_B", "text": "..."}, ...]}
"""


@dataclass
class DialogueLine:
    speaker: str
    text: str


def generate_dialogue(
    source_text: str,
    *,
    topic_hint: str | None = None,
    target_minutes: int = 5,
    backend: str = "anthropic",
    ollama_model: str = OLLAMA_MODEL,
) -> list[DialogueLine]:
    """Turn `source_text` into a list of DialogueLine using the chosen backend.

    backend: "anthropic" (Claude, needs ANTHROPIC_API_KEY) or "ollama" (a
    locally running Ollama server, fully offline).
    """
    user_prompt = (
        f"Target episode length: roughly {target_minutes} minutes of spoken audio "
        f"(~{target_minutes * 150} words total).\n"
    )
    if topic_hint:
        user_prompt += f"Episode focus: {topic_hint}\n"
    user_prompt += f"\nSOURCE MATERIAL:\n{source_text}"

    if backend == "anthropic":
        raw = _generate_with_anthropic(user_prompt)
    elif backend == "ollama":
        raw = _generate_with_ollama(user_prompt, model=ollama_model)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected 'anthropic' or 'ollama')")

    data = json.loads(_strip_code_fence(raw))
    return [DialogueLine(speaker=item["speaker"], text=item["text"]) for item in data["lines"]]


def _generate_with_anthropic(user_prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _generate_with_ollama(user_prompt: str, *, model: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
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
