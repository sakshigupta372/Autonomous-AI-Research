"""LangGraph workflow: Scout -> Reader -> Analyst -> Writer -> Critic -> (loop) ->
Comparator -> Experimenter -> PersistMemory
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langgraph.graph import END, StateGraph

from research_agent.agents import analyst, comparator, critic, experimenter, reader, scout, writer
from research_agent.config import settings
from research_agent.memory import graph_store, vector_store
from research_agent.state import ResearchState


def _route_after_critic(state: ResearchState) -> str:
    """Send failed summaries back to Writer until max reflection rounds."""
    if state.get("pending_revision"):
        return "writer"
    return "comparator"


def _persist_memory(state: ResearchState) -> ResearchState:
    """Embed chunks, persist graph + ingestion ledger, write all reports."""
    collection = vector_store.get_collection(settings.chroma_dir)
    vector_store.add_chunks(collection, state["chunks"])

    graph_store.save_graph(settings.graph_db_path, state["knowledge_graph"])
    for paper in state["papers"]:
        graph_store.mark_ingested(settings.graph_db_path, paper)

    report_paths: list[str] = []
    papers_by_id = {p.arxiv_id: p for p in state["papers"]}
    repro_by_id = {r.paper_id: r for r in state.get("reproductions", [])}

    for summary in state["summaries"]:
        paper = papers_by_id.get(summary.paper_id)
        if paper is None:
            continue
        markdown = writer.render_markdown(
            paper,
            summary,
            state["knowledge_graph"],
            reproduction=repro_by_id.get(summary.paper_id),
        )
        out_path = settings.outputs_dir / f"{paper.arxiv_id}.md"
        out_path.write_text(markdown, encoding="utf-8")
        report_paths.append(str(out_path))

    comparison = state.get("comparison")
    if comparison is not None and len(state["papers"]) >= 2:
        comparison_path = settings.outputs_dir / f"comparison_{_topic_slug(state['topic'])}.md"
        comparison_path.write_text(comparator.render_comparison_markdown(comparison), encoding="utf-8")
        report_paths.append(str(comparison_path))

    session_log = {
        "topic": state["topic"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reflection_logs": state.get("reflection_logs", []),
        "critiques": [c.model_dump() for c in state.get("critiques", [])],
        "reproductions": [
            {
                "paper_id": r.paper_id,
                "success": r.success,
                "reproducibility_score": r.reproducibility_score,
            }
            for r in state.get("reproductions", [])
        ],
    }
    session_path = settings.sessions_dir / f"{_topic_slug(state['topic'])}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    session_path.write_text(json.dumps(session_log, indent=2), encoding="utf-8")

    state["report_paths"] = report_paths
    return state


def _topic_slug(topic: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug[:60] or "run"


def build_graph():
    """Compile the LangGraph state machine for Phases 1-3."""
    workflow = StateGraph(ResearchState)
    workflow.add_node("scout", scout.run)
    workflow.add_node("reader", reader.run)
    workflow.add_node("analyst", analyst.run)
    workflow.add_node("writer", writer.run)
    workflow.add_node("critic", critic.run)
    workflow.add_node("comparator", comparator.run)
    workflow.add_node("experimenter", experimenter.run)
    workflow.add_node("persist_memory", _persist_memory)

    workflow.set_entry_point("scout")
    workflow.add_edge("scout", "reader")
    workflow.add_edge("reader", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges("critic", _route_after_critic, {"writer": "writer", "comparator": "comparator"})
    workflow.add_edge("comparator", "experimenter")
    workflow.add_edge("experimenter", "persist_memory")
    workflow.add_edge("persist_memory", END)

    return workflow.compile()


def run_research(topic: str, max_papers: int, enable_experiments: bool | None = None) -> ResearchState:
    """Run the full pipeline for a topic and return the final state."""
    app = build_graph()
    initial_state: ResearchState = {
        "topic": topic,
        "max_papers": max_papers,
        "papers": [],
        "chunks": [],
        "paper_graphs": [],
        "knowledge_graph": graph_store.load_graph(settings.graph_db_path),
        "summaries": [],
        "critiques": [],
        "critic_feedback": {},
        "reflection_round": 0,
        "max_reflection_rounds": settings.research_agent_max_reflection_rounds,
        "reflection_logs": [],
        "comparison": None,
        "reproductions": [],
        "enable_experiments": settings.research_agent_enable_experiments
        if enable_experiments is None
        else enable_experiments,
        "pending_revision": False,
        "report_paths": [],
        "errors": [],
    }
    return app.invoke(initial_state)
