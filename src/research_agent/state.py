"""Shared LangGraph state passed between all pipeline agents."""

from __future__ import annotations

from typing import Any, TypedDict

from research_agent.models.comparison import ComparisonReport
from research_agent.models.critique import CritiqueResult
from research_agent.models.paper import Chunk, Paper, PaperGraph
from research_agent.models.reproduction import ReproducibilityResult
from research_agent.models.summary import ResearchSummary


class ResearchState(TypedDict):
    topic: str
    original_topic: str
    max_papers: int
    session_id: str
    papers: list[Paper]
    chunks: list[Chunk]
    paper_graphs: list[PaperGraph]
    knowledge_graph: dict[str, Any]
    summaries: list[ResearchSummary]
    critiques: list[CritiqueResult]
    critic_feedback: dict[str, str]
    reflection_round: int
    max_reflection_rounds: int
    reflection_logs: list[str]
    comparison: ComparisonReport | None
    reproductions: list[ReproducibilityResult]
    enable_experiments: bool
    pending_revision: bool
    autonomous_mode: bool
    autonomous_iteration: int
    max_autonomous_iterations: int
    autonomous_complete: bool
    search_queries: list[str]
    gap_queries: list[str]
    report_paths: list[str]
    all_report_paths: list[str]
    errors: list[str]
