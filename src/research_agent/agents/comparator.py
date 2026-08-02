"""Comparator agent: builds a cross-paper method/dataset/metric comparison table."""

from __future__ import annotations

import re

import networkx as nx
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.models.comparison import ComparisonReport, PaperComparisonRow
from research_agent.models.summary import ResearchSummary
from research_agent import progress
from research_agent.state import ResearchState
from research_agent.tools import graph_builder

COMPARATOR_PROMPT = """You are comparing multiple academic papers on the same research topic.

Topic: {topic}

Per-paper comparison rows:
{rows_text}

Per-paper summaries:
{summaries_text}

Write:
- synthesis: how these papers relate, overlap, or differ (2-4 paragraphs)
- open_questions: unresolved research questions across the set
- contradictions: any conflicting claims or results between papers (empty list if none)
"""


class ComparatorDraft(BaseModel):
    synthesis: str = ""
    open_questions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


def _entities_for_paper(graph: nx.MultiDiGraph, paper_id: str, entity_type: str) -> list[str]:
    subgraph = graph_builder.subgraph_for_paper(graph, paper_id)
    return sorted({data["name"] for _, data in subgraph.nodes(data=True) if data.get("type") == entity_type})


def _build_rows(state: ResearchState) -> list[PaperComparisonRow]:
    summaries_by_id = {s.paper_id: s for s in state["summaries"]}
    rows: list[PaperComparisonRow] = []
    for paper in state["papers"]:
        summary = summaries_by_id.get(paper.arxiv_id)
        rows.append(
            PaperComparisonRow(
                paper_id=paper.arxiv_id,
                title=paper.title,
                methods=_entities_for_paper(state["knowledge_graph"], paper.arxiv_id, "Method"),
                datasets=_entities_for_paper(state["knowledge_graph"], paper.arxiv_id, "Dataset"),
                metrics=_entities_for_paper(state["knowledge_graph"], paper.arxiv_id, "Metric"),
                key_limitations=summary.limitations if summary else [],
            )
        )
    return rows


def _shared_across_papers(rows: list[PaperComparisonRow], field: str) -> list[str]:
    if not rows:
        return []
    sets = [set(getattr(row, field)) for row in rows if getattr(row, field)]
    if not sets:
        return []
    shared = sets[0]
    for item in sets[1:]:
        shared &= item
    return sorted(shared)


def _rows_text(rows: list[PaperComparisonRow]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(f"Paper: {row.title} ({row.paper_id})")
        lines.append(f"  Methods: {', '.join(row.methods) or '(none)'}")
        lines.append(f"  Datasets: {', '.join(row.datasets) or '(none)'}")
        lines.append(f"  Metrics: {', '.join(row.metrics) or '(none)'}")
        lines.append(f"  Limitations: {', '.join(row.key_limitations) or '(none)'}")
    return "\n".join(lines)


def _summaries_text(summaries: list[ResearchSummary]) -> str:
    parts: list[str] = []
    for summary in summaries:
        parts.append(
            f"{summary.title} ({summary.paper_id})\n"
            f"Contributions: {summary.contributions}\n"
            f"Results: {summary.results}\n"
            f"Limitations: {summary.limitations}"
        )
    return "\n\n".join(parts)


def _section_bullets(items: list[str], empty_label: str = "(none)") -> list[str]:
    return [f"- {item}" for item in items] if items else [f"- {empty_label}"]


def render_comparison_markdown(report: ComparisonReport) -> str:
    """Render the cross-paper comparison as Markdown."""
    header = ["| Paper | Methods | Datasets | Metrics | Limitations |", "|---|---|---|---|---|"]
    for row in report.rows:
        header.append(
            "| "
            + " | ".join(
                [
                    f"{row.title[:40]}... ({row.paper_id})" if len(row.title) > 40 else f"{row.title} ({row.paper_id})",
                    ", ".join(row.methods) or "-",
                    ", ".join(row.datasets) or "-",
                    ", ".join(row.metrics) or "-",
                    "; ".join(row.key_limitations[:2]) or "-",
                ]
            )
            + " |"
        )

    lines = [
        f"# Cross-Paper Comparison: {report.topic}",
        "",
        "## Comparison Table",
        "",
        *header,
        "",
        "## Shared Methods",
        "",
        *_section_bullets(report.shared_methods),
        "",
        "## Shared Datasets",
        "",
        *_section_bullets(report.shared_datasets),
        "",
        "## Synthesis",
        "",
        report.synthesis or "(none)",
        "",
        "## Open Questions",
        "",
        *_section_bullets(report.open_questions),
        "",
        "## Contradictions",
        "",
        *_section_bullets(report.contradictions, "(none detected)"),
        "",
    ]
    return "\n".join(lines)


def _topic_slug(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug[:60] or "comparison"


def run(state: ResearchState) -> ResearchState:
    if len(state["papers"]) < 2:
        progress.info("Only 1 paper — skipping cross-paper synthesis")
        state["comparison"] = ComparisonReport(topic=state["topic"], rows=_build_rows(state))
        return state

    progress.info("Building comparison table from knowledge graph...")
    rows = _build_rows(state)
    report = ComparisonReport(
        topic=state["topic"],
        rows=rows,
        shared_methods=_shared_across_papers(rows, "methods"),
        shared_datasets=_shared_across_papers(rows, "datasets"),
    )

    progress.info("Writing cross-paper synthesis (Groq LLM)...")
    llm = ChatGroq(model=settings.research_agent_model, api_key=settings.groq_api_key, temperature=0)
    structured_llm = llm.with_structured_output(ComparatorDraft)
    prompt = COMPARATOR_PROMPT.format(
        topic=state["topic"],
        rows_text=_rows_text(rows),
        summaries_text=_summaries_text(state["summaries"]),
    )
    try:
        draft = structured_llm.invoke(prompt)
        report.synthesis = draft.synthesis
        report.open_questions = draft.open_questions
        report.contradictions = draft.contradictions
    except Exception as e:  # noqa: BLE001
        state["errors"].append(f"Comparator synthesis failed: {e}")
        progress.warn("Cross-paper synthesis failed")

    state["comparison"] = report
    return state
