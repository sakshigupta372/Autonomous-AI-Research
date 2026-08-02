"""Shared LangGraph state passed between Scout, Reader, Analyst, and Writer."""

from __future__ import annotations

from typing import TypedDict

import networkx as nx

from research_agent.models.paper import Chunk, Paper, PaperGraph
from research_agent.models.summary import ResearchSummary


class ResearchState(TypedDict):
    topic: str
    max_papers: int
    papers: list[Paper]
    chunks: list[Chunk]
    paper_graphs: list[PaperGraph]
    knowledge_graph: nx.MultiDiGraph
    summaries: list[ResearchSummary]
    report_paths: list[str]
    errors: list[str]
