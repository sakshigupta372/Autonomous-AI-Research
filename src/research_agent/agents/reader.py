"""Reader agent: extracts text from downloaded PDFs and splits into section-aware chunks."""

from __future__ import annotations

from pathlib import Path

from research_agent.models.paper import Chunk
from research_agent import progress
from research_agent.state import ResearchState
from research_agent.tools import pdf_parser


def run(state: ResearchState) -> ResearchState:
    chunks: list[Chunk] = []

    papers = state["papers"]
    for index, paper in enumerate(papers, start=1):
        if not paper.local_path:
            state["errors"].append(f"Reader skipped {paper.arxiv_id}: no local PDF path")
            continue
        progress.paper_step("Reader", index, len(papers), paper.arxiv_id, "parsing PDF")
        try:
            paper_chunks = pdf_parser.parse_pdf(paper.arxiv_id, Path(paper.local_path))
            chunks.extend(paper_chunks)
            progress.info(f"Extracted {len(paper_chunks)} chunk(s)")
        except Exception as e:  # noqa: BLE001
            state["errors"].append(f"Reader failed on {paper.arxiv_id}: {e}")
            progress.warn(f"Parse failed for {paper.arxiv_id}")

    state["chunks"] = chunks
    return state
