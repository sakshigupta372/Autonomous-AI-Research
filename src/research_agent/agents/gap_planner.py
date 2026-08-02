"""Gap planner agent: proposes the next arXiv search query from graph gaps (Phase 4)."""

from __future__ import annotations

import networkx as nx
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent import progress
from research_agent.state import ResearchState
from research_agent.tools.kg_state import as_graph


GAP_PROMPT = """You are an autonomous research scientist planning the next literature search.

Original research goal: {original_topic}
Completed autonomous iterations: {iteration}
Previous search queries used: {previous_queries}

Open questions from prior comparison:
{open_questions}

Knowledge graph snapshot (entity types and names):
{graph_snapshot}

Propose ONE focused arXiv search query (5-12 words) that would fill the biggest gap in our
understanding. Avoid repeating previous queries. Target methods, datasets, or limitations
we have not covered yet.
"""


class GapPlan(BaseModel):
    next_query: str = Field(description="Next arXiv search query")
    rationale: str = Field(default="")


def _graph_snapshot(graph: nx.MultiDiGraph, limit: int = 30) -> str:
    lines = []
    for node_id, data in list(graph.nodes(data=True))[:limit]:
        papers = data.get("papers", set())
        lines.append(f"- ({data.get('type')}) {data.get('name')} [papers: {len(papers)}]")
    return "\n".join(lines) if lines else "(empty graph)"


def _reset_batch_state(state: ResearchState) -> None:
    """Clear per-batch artifacts while keeping the accumulated knowledge graph."""
    state["papers"] = []
    state["chunks"] = []
    state["paper_graphs"] = []
    state["summaries"] = []
    state["critiques"] = []
    state["critic_feedback"] = {}
    state["reflection_round"] = 0
    state["pending_revision"] = False
    state["comparison"] = None
    state["reproductions"] = []
    state["report_paths"] = []


def run(state: ResearchState) -> ResearchState:
    state["autonomous_iteration"] = state.get("autonomous_iteration", 0) + 1
    progress.info(f"Autonomous iteration {state['autonomous_iteration']}: analyzing knowledge gaps")

    open_questions = []
    comparison = state.get("comparison")
    if comparison is not None:
        open_questions = comparison.open_questions

    llm = ChatGroq(model=settings.research_agent_model, api_key=settings.groq_api_key, temperature=0)
    structured_llm = llm.with_structured_output(GapPlan)
    prompt = GAP_PROMPT.format(
        original_topic=state.get("original_topic") or state["topic"],
        iteration=state["autonomous_iteration"],
        previous_queries=state.get("search_queries") or [state["topic"]],
        open_questions="\n".join(f"- {q}" for q in open_questions) or "(none yet)",
        graph_snapshot=_graph_snapshot(as_graph(state["knowledge_graph"])),
    )

    try:
        plan = structured_llm.invoke(prompt)
        next_query = plan.next_query.strip()
        progress.info(f"Next query: {next_query}")
        progress.info(f"Rationale: {plan.rationale}")
    except Exception as e:  # noqa: BLE001
        state["errors"].append(f"Gap planner failed: {e}")
        state["autonomous_complete"] = True
        return state

    queries = list(state.get("search_queries") or [])
    if state.get("original_topic") and state["original_topic"] not in queries:
        queries.insert(0, state["original_topic"])
    queries.append(next_query)
    state["search_queries"] = queries
    state["topic"] = next_query
    state["gap_queries"] = list(state.get("gap_queries") or []) + [next_query]

    _reset_batch_state(state)
    return state
