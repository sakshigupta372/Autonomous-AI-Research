"""LangGraph workflow wiring: Scout -> Reader -> Analyst -> Writer -> PersistMemory."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from research_agent.agents import analyst, reader, scout, writer
from research_agent.config import settings
from research_agent.memory import graph_store, vector_store
from research_agent.state import ResearchState


def _persist_memory(state: ResearchState) -> ResearchState:
    """Embed chunks into ChromaDB, persist the knowledge graph + ingestion
    record to SQLite, and render each summary as a Markdown report."""
    collection = vector_store.get_collection(settings.chroma_dir)
    vector_store.add_chunks(collection, state["chunks"])

    graph_store.save_graph(settings.graph_db_path, state["knowledge_graph"])
    for paper in state["papers"]:
        graph_store.mark_ingested(settings.graph_db_path, paper)

    report_paths: list[str] = []
    papers_by_id = {p.arxiv_id: p for p in state["papers"]}
    for summary in state["summaries"]:
        paper = papers_by_id.get(summary.paper_id)
        if paper is None:
            continue
        markdown = writer.render_markdown(paper, summary, state["knowledge_graph"])
        out_path = settings.outputs_dir / f"{paper.arxiv_id}.md"
        out_path.write_text(markdown, encoding="utf-8")
        report_paths.append(str(out_path))

    state["report_paths"] = report_paths
    return state


def build_graph():
    """Compile the LangGraph state machine for the Phase 1 MVP pipeline."""
    workflow = StateGraph(ResearchState)
    workflow.add_node("scout", scout.run)
    workflow.add_node("reader", reader.run)
    workflow.add_node("analyst", analyst.run)
    workflow.add_node("writer", writer.run)
    workflow.add_node("persist_memory", _persist_memory)

    workflow.set_entry_point("scout")
    workflow.add_edge("scout", "reader")
    workflow.add_edge("reader", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "persist_memory")
    workflow.add_edge("persist_memory", END)

    return workflow.compile()


def run_research(topic: str, max_papers: int) -> ResearchState:
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
        "report_paths": [],
        "errors": [],
    }
    return app.invoke(initial_state)
