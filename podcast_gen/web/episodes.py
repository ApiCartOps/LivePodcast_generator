"""Filesystem-backed episode library.

Each episode is three files under the episodes directory, named by id:
  {id}.mp3          the mixed audio
  {id}.json         metadata (title, participants, timestamps, ...)
  {id}.script.json  the raw dialogue lines, in the same shape cli.py's
                     --script-json flag already writes, so a script saved
                     here is interchangeable with the CLI's cache format.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from podcast_gen.participants import Participant
from podcast_gen.script_gen import DialogueLine


class EpisodeStore:
    def __init__(self, episodes_dir: str | Path):
        self.dir = Path(episodes_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def mp3_path(self, episode_id: str) -> Path:
        return self.dir / f"{episode_id}.mp3"

    def _meta_path(self, episode_id: str) -> Path:
        return self.dir / f"{episode_id}.json"

    def _script_path(self, episode_id: str) -> Path:
        return self.dir / f"{episode_id}.script.json"

    def save_episode(
        self,
        *,
        episode_id: str,
        title: str,
        source_snippet: str,
        participants: list[Participant],
        target_minutes: int,
        llm_backend: str,
        lines: list[DialogueLine],
    ) -> None:
        meta = {
            "id": episode_id,
            "title": title,
            "created_at": time.time(),
            "source_snippet": source_snippet,
            "participants": [asdict(p) for p in participants],
            "target_minutes": target_minutes,
            "llm_backend": llm_backend,
            "line_count": len(lines),
        }
        self._meta_path(episode_id).write_text(json.dumps(meta, indent=2))
        self._script_path(episode_id).write_text(
            json.dumps([line.__dict__ for line in lines], indent=2)
        )

    def list_episodes(self) -> list[dict]:
        episodes = []
        for meta_file in self.dir.glob("*.json"):
            if meta_file.name.endswith(".script.json"):
                continue
            try:
                episodes.append(json.loads(meta_file.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        episodes.sort(key=lambda e: e.get("created_at", 0), reverse=True)
        return episodes

    def get_episode(self, episode_id: str) -> dict | None:
        meta_path = self._meta_path(episode_id)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def delete_episode(self, episode_id: str) -> bool:
        meta_path = self._meta_path(episode_id)
        if not meta_path.exists():
            return False
        for path in (self.mp3_path(episode_id), meta_path, self._script_path(episode_id)):
            path.unlink(missing_ok=True)
        return True
