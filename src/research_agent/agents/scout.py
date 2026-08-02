"""Scout agent: searches arXiv for a topic and downloads candidate PDFs."""

from __future__ import annotations

from research_agent.config import settings
from research_agent.models.paper import Paper
from research_agent import progress
from research_agent.state import ResearchState
from research_agent.tools import arxiv_client


def run(state: ResearchState) -> ResearchState:
    papers: list[Paper] = []

    progress.info(f"Searching arXiv for: {state['topic']}")
    try:
        results = arxiv_client.search(state["topic"], state["max_papers"])
    except Exception as e:  # noqa: BLE001 - surface all search failures as state errors
        state["errors"].append(f"Scout search failed: {e}")
        state["papers"] = papers
        return state

    total = len(results)
    for index, result in enumerate(results, start=1):
        arxiv_id = result.get_short_id()
        progress.paper_step("Scout", index, total, arxiv_id, "downloading PDF")
        try:
            local_path = arxiv_client.download(result, settings.papers_dir)
            papers.append(arxiv_client.to_paper(result, local_path))
            progress.info(f"Saved {local_path.name}")
        except Exception as e:  # noqa: BLE001
            state["errors"].append(f"Scout failed to download {arxiv_id}: {e}")
            progress.warn(f"Download failed for {arxiv_id}")

    state["papers"] = papers
    return state
