"""Scout agent: searches arXiv for a topic and downloads candidate PDFs."""

from __future__ import annotations

from research_agent.config import settings
from research_agent.models.paper import Paper
from research_agent.state import ResearchState
from research_agent.tools import arxiv_client


def run(state: ResearchState) -> ResearchState:
    papers: list[Paper] = []

    try:
        results = arxiv_client.search(state["topic"], state["max_papers"])
    except Exception as e:  # noqa: BLE001 - surface all search failures as state errors
        state["errors"].append(f"Scout search failed: {e}")
        state["papers"] = papers
        return state

    for result in results:
        arxiv_id = result.get_short_id()
        try:
            local_path = arxiv_client.download(result, settings.papers_dir)
            papers.append(arxiv_client.to_paper(result, local_path))
        except Exception as e:  # noqa: BLE001
            state["errors"].append(f"Scout failed to download {arxiv_id}: {e}")

    state["papers"] = papers
    return state
