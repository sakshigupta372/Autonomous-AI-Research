"""Scout agent: searches arXiv for a topic and downloads candidate PDFs."""

from __future__ import annotations

from research_agent.config import settings
from research_agent.memory import graph_store
from research_agent.models.paper import Paper
from research_agent import progress
from research_agent.state import ResearchState
from research_agent.tools import arxiv_client


def run(state: ResearchState) -> ResearchState:
    papers: list[Paper] = []

    progress.info(f"Searching arXiv for: {state['topic']}")
    search_limit = state["max_papers"] * 3 if state.get("autonomous_mode") else state["max_papers"]
    try:
        results = arxiv_client.search(state["topic"], search_limit)
    except Exception as e:  # noqa: BLE001
        state["errors"].append(f"Scout search failed: {e}")
        state["papers"] = papers
        return state

    if state.get("autonomous_mode"):
        results = [r for r in results if not graph_store.is_ingested(settings.graph_db_path, r.get_short_id())]
        results = results[: state["max_papers"]]
        if not results:
            progress.info("No new papers found (all already ingested or none matched)")
        else:
            progress.info(f"Found {len(results)} new paper(s) not yet ingested")

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
