"""LangGraph workflow with reflection, autonomous loop, and checkpoint resume (Phases 1-4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable

from langgraph.graph import END, StateGraph

from research_agent.agents import analyst, comparator, critic, experimenter, gap_planner, reader, scout, writer
from research_agent.checkpoint import get_checkpointer, new_session_id
from research_agent.config import settings
from research_agent.memory import graph_store, vector_store
from research_agent import progress
from research_agent.state import ResearchState


def _route_after_critic(state: ResearchState) -> str:
    if state.get("pending_revision"):
        progress.route(
            f"Reflection round {state['reflection_round']}: sending "
            f"{len(state.get('critic_feedback', {}))} paper(s) back to Writer"
        )
        return "writer"
    return "comparator"


def _route_after_scout(state: ResearchState) -> str:
    if state.get("papers"):
        return "reader"
    if state.get("autonomous_mode") and state.get("autonomous_iteration", 0) > 0:
        progress.info("Autonomous loop ending — no new papers found")
        state["autonomous_complete"] = True
        return END
    return "reader"


def _route_after_persist(state: ResearchState) -> str:
    if not state.get("autonomous_mode"):
        return END
    if state.get("autonomous_complete"):
        return END
    if state.get("autonomous_iteration", 0) >= state.get("max_autonomous_iterations", 0):
        progress.info("Autonomous iteration limit reached")
        return END
    progress.route("Starting next autonomous research iteration")
    return "gap_planner"


def _wrap(name: str, fn: Callable[[ResearchState], ResearchState]) -> Callable[[ResearchState], ResearchState]:
    def node(state: ResearchState) -> ResearchState:
        detail = ""
        if name == "writer" and state.get("reflection_round", 0) > 0 and state.get("critic_feedback"):
            detail = f"revision pass (round {state['reflection_round']})"
        if name == "gap_planner":
            detail = f"iteration {state.get('autonomous_iteration', 0) + 1}"
        progress.step_start(name, detail)
        result = fn(state)
        progress.step_done(name, _step_summary(name, result))
        return result

    return node


def _step_summary(name: str, state: ResearchState) -> str:
    if name == "scout":
        return f"{len(state.get('papers', []))} paper(s) found"
    if name == "reader":
        return f"{len(state.get('chunks', []))} chunk(s)"
    if name == "analyst":
        graph = state.get("knowledge_graph")
        nodes = graph.number_of_nodes() if graph is not None else 0
        return f"{nodes} graph node(s)"
    if name == "writer":
        return f"{len(state.get('summaries', []))} summary(ies)"
    if name == "critic":
        critiques = state.get("critiques", [])
        if not critiques:
            return "no critiques"
        avg = sum(c.overall for c in critiques) / len(critiques)
        return f"avg score {avg:.1f}/10"
    if name == "comparator":
        return "comparison ready" if state.get("comparison") else "skipped"
    if name == "experimenter":
        if not state.get("enable_experiments", True):
            return "skipped"
        repro = state.get("reproductions", [])
        if not repro:
            return "0 runs"
        avg = sum(r.reproducibility_score for r in repro) / len(repro)
        return f"{len(repro)} run(s), avg repro {avg:.0%}"
    if name == "gap_planner":
        return state.get("topic", "")
    if name == "persist_memory":
        return f"{len(state.get('all_report_paths', []))} total report(s)"
    return ""


def _persist_memory(state: ResearchState) -> ResearchState:
    progress.info("Embedding chunks into ChromaDB...")
    collection = vector_store.get_collection(settings.chroma_dir)
    vector_store.add_chunks(collection, state["chunks"])

    progress.info("Saving knowledge graph and ingestion ledger...")
    graph_store.save_graph(settings.graph_db_path, state["knowledge_graph"])
    for paper in state["papers"]:
        graph_store.mark_ingested(settings.graph_db_path, paper)

    report_paths: list[str] = list(state.get("report_paths", []))
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
        progress.info(f"Wrote {out_path.name}")

    comparison = state.get("comparison")
    if comparison is not None and len(state["papers"]) >= 2:
        slug = _topic_slug(state.get("original_topic") or state["topic"])
        comparison_path = settings.outputs_dir / f"comparison_{slug}.md"
        comparison_path.write_text(comparator.render_comparison_markdown(comparison), encoding="utf-8")
        report_paths.append(str(comparison_path))
        progress.info(f"Wrote {comparison_path.name}")

    all_paths = list(state.get("all_report_paths", []))
    for path in report_paths:
        if path not in all_paths:
            all_paths.append(path)

    session_log = {
        "session_id": state.get("session_id"),
        "topic": state.get("original_topic") or state["topic"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "autonomous_iteration": state.get("autonomous_iteration", 0),
        "search_queries": state.get("search_queries", []),
        "reflection_logs": state.get("reflection_logs", []),
        "critiques": [c.model_dump() for c in state.get("critiques", [])],
        "reproductions": [
            {"paper_id": r.paper_id, "success": r.success, "reproducibility_score": r.reproducibility_score}
            for r in state.get("reproductions", [])
        ],
    }
    slug = _topic_slug(state.get("original_topic") or state["topic"])
    session_path = settings.sessions_dir / f"{slug}_{state.get('session_id', 'run')}.json"
    session_path.write_text(json.dumps(session_log, indent=2), encoding="utf-8")
    progress.info(f"Session log: {session_path.name} (checkpoint id: {state.get('session_id')})")

    state["report_paths"] = report_paths
    state["all_report_paths"] = all_paths
    return state


def _topic_slug(topic: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug[:60] or "run"


def build_graph(checkpointer=None):
    workflow = StateGraph(ResearchState)
    workflow.add_node("scout", _wrap("scout", scout.run))
    workflow.add_node("reader", _wrap("reader", reader.run))
    workflow.add_node("analyst", _wrap("analyst", analyst.run))
    workflow.add_node("writer", _wrap("writer", writer.run))
    workflow.add_node("critic", _wrap("critic", critic.run))
    workflow.add_node("comparator", _wrap("comparator", comparator.run))
    workflow.add_node("experimenter", _wrap("experimenter", experimenter.run))
    workflow.add_node("gap_planner", _wrap("gap_planner", gap_planner.run))
    workflow.add_node("persist_memory", _wrap("persist_memory", _persist_memory))

    workflow.set_entry_point("scout")
    workflow.add_conditional_edges("scout", _route_after_scout, {"reader": "reader", END: END})
    workflow.add_edge("reader", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges("critic", _route_after_critic, {"writer": "writer", "comparator": "comparator"})
    workflow.add_edge("comparator", "experimenter")
    workflow.add_edge("experimenter", "persist_memory")
    workflow.add_conditional_edges("persist_memory", _route_after_persist, {"gap_planner": "gap_planner", END: END})
    workflow.add_edge("gap_planner", "scout")

    return workflow.compile(checkpointer=checkpointer)


def _initial_state(
    topic: str,
    max_papers: int,
    session_id: str,
    enable_experiments: bool,
    autonomous: bool,
    max_autonomous_iterations: int,
) -> ResearchState:
    return {
        "topic": topic,
        "original_topic": topic,
        "max_papers": max_papers,
        "session_id": session_id,
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
        "enable_experiments": enable_experiments,
        "pending_revision": False,
        "autonomous_mode": autonomous,
        "autonomous_iteration": 0,
        "max_autonomous_iterations": max_autonomous_iterations,
        "autonomous_complete": False,
        "search_queries": [topic],
        "gap_queries": [],
        "report_paths": [],
        "all_report_paths": [],
        "errors": [],
    }


def run_research(
    topic: str,
    max_papers: int,
    enable_experiments: bool | None = None,
    session_id: str | None = None,
    autonomous: bool = False,
    max_autonomous_iterations: int | None = None,
    resume: bool = False,
) -> ResearchState:
    """Run (or resume) the full pipeline. Checkpoints are keyed by session_id."""
    sid = session_id or new_session_id()
    config = {"configurable": {"thread_id": sid}}
    experiments = settings.research_agent_enable_experiments if enable_experiments is None else enable_experiments
    max_auto = max_autonomous_iterations if max_autonomous_iterations is not None else settings.research_agent_max_autonomous_iterations

    if not resume:
        progress.info(f"Session checkpoint id: {sid}")

    with get_checkpointer() as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        if resume:
            progress.info(f"Resuming session {sid} from checkpoint...")
            return app.invoke(None, config=config)
        initial_state = _initial_state(topic, max_papers, sid, experiments, autonomous, max_auto)
        return app.invoke(initial_state, config=config)
