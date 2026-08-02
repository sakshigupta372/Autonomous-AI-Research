"""Thin wrapper around the `arxiv` package: search, download, and metadata mapping."""

from __future__ import annotations

from pathlib import Path

import arxiv
import requests

from research_agent.models.paper import Paper


def search(topic: str, max_results: int) -> list[arxiv.Result]:
    """Search arXiv for a topic, sorted by relevance."""
    client = arxiv.Client()
    search_query = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(client.results(search_query))


def download(result: arxiv.Result, dest_dir: Path) -> Path:
    """Download a paper's PDF to dest_dir, skipping if it already exists.

    The `arxiv` package no longer ships a `Result.download_pdf` helper, so we
    fetch `result.pdf_url` directly over HTTP.
    """
    arxiv_id = result.get_short_id()
    dest_path = dest_dir / f"{arxiv_id}.pdf"
    if dest_path.exists():
        return dest_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(result.pdf_url, timeout=30)
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    return dest_path


def to_paper(result: arxiv.Result, local_path: Path) -> Paper:
    """Convert an arxiv.Result into our Paper schema."""
    return Paper(
        arxiv_id=result.get_short_id(),
        title=result.title.strip(),
        authors=[a.name for a in result.authors],
        abstract=result.summary.strip(),
        categories=list(result.categories),
        published=result.published.isoformat() if result.published else "",
        pdf_url=result.pdf_url or "",
        local_path=str(local_path),
    )
