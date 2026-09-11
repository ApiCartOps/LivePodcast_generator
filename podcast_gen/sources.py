"""Content loaders: Confluence pages, generic web URLs, and local files."""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def load_source(source: str) -> str:
    """Detect the kind of `source` and return its plain-text content."""
    if _looks_like_confluence(source):
        return load_confluence(source)
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)
    if Path(source).exists():
        return load_file(source)
    raise ValueError(
        f"Could not figure out how to load: {source!r}. "
        "Pass a Confluence page URL/ID, a web URL, or a path to an existing local file."
    )


def _looks_like_confluence(source: str) -> bool:
    if "atlassian.net/wiki" in source:
        return True
    base = os.environ.get("CONFLUENCE_BASE_URL", "")
    return bool(base) and source.startswith(base)


def load_confluence(page: str) -> str:
    """Fetch a Confluence Cloud page's body text via the REST API.

    `page` can be a full page URL (e.g. .../wiki/spaces/KB/pages/12345/Title)
    or a bare numeric page ID. Requires env vars:
      CONFLUENCE_BASE_URL  e.g. https://yourteam.atlassian.net/wiki
      CONFLUENCE_EMAIL     the Atlassian account email
      CONFLUENCE_API_TOKEN an API token (id.atlassian.com/manage-profile/security/api-tokens)
    """
    base_url = os.environ.get("CONFLUENCE_BASE_URL")
    email = os.environ.get("CONFLUENCE_EMAIL")
    token = os.environ.get("CONFLUENCE_API_TOKEN")
    if not (base_url and email and token):
        raise RuntimeError(
            "Confluence access needs CONFLUENCE_BASE_URL, CONFLUENCE_EMAIL, "
            "and CONFLUENCE_API_TOKEN set in the environment (see .env.example)."
        )

    page_id = _extract_confluence_page_id(page)
    api_url = f"{base_url.rstrip('/')}/rest/api/content/{page_id}"
    resp = requests.get(
        api_url,
        params={"expand": "body.storage,title"},
        auth=(email, token),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    title = data.get("title", "")
    html = data["body"]["storage"]["value"]
    text = _html_to_text(html)
    return f"{title}\n\n{text}" if title else text


def _extract_confluence_page_id(page: str) -> str:
    if page.isdigit():
        return page
    match = re.search(r"/pages/(\d+)", page)
    if match:
        return match.group(1)
    match = re.search(r"[?&]pageId=(\d+)", page)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract a Confluence page ID from: {page!r}")


def load_url(url: str) -> str:
    """Fetch a generic web page and extract its visible text."""
    resp = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (podcast-generator content fetcher)"},
    )
    resp.raise_for_status()
    return _html_to_text(resp.text)


def load_file(path: str) -> str:
    """Load a local .txt, .md, or .pdf file as plain text."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(p)
    return p.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
