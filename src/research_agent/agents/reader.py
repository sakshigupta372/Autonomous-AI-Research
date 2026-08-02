"""Reader agent: extracts text from downloaded PDFs and splits into section-aware chunks."""

from __future__ import annotations

from pathlib import Path

from research_agent.models.paper import Chunk
from research_agent.state import ResearchState
from research_agent.tools import pdf_parser


def run(state: ResearchState) -> ResearchState:
    chunks: list[Chunk] = []

    for paper in state["papers"]:
        if not paper.local_path:
            state["errors"].append(f"Reader skipped {paper.arxiv_id}: no local PDF path")
            continue
        try:
            chunks.extend(pdf_parser.parse_pdf(paper.arxiv_id, Path(paper.local_path)))
        except Exception as e:  # noqa: BLE001
            state["errors"].append(f"Reader failed on {paper.arxiv_id}: {e}")

    state["chunks"] = chunks
    return state
