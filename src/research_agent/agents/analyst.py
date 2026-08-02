"""Analyst agent: extracts a GraphRAG knowledge graph (entities + relations) per paper.

Papers already recorded in long-term memory (SQLite) skip the LLM extraction
call entirely and reuse the persisted subgraph instead, so re-running the same
topic does not re-pay the extraction cost.
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.memory import graph_store
from research_agent.models.paper import Chunk, Entity, PaperGraph, Relation
from research_agent.state import ResearchState
from research_agent.tools import graph_builder

MAX_CONTEXT_CHARS = 12000

EXTRACTION_PROMPT = """You are a research analyst extracting a knowledge graph from an academic paper.

Read the paper text below and extract:
- Entities: Methods, Datasets, Metrics, Claims, and Limitations mentioned in the paper.
- Relations between entities, using ONLY these relation types: USES, EVALUATED_ON, IMPROVES, LIMITED_BY.

Be precise. Only extract entities and relations that are explicitly supported by the text. Prefer a
small, accurate graph over an exhaustive but noisy one.

Paper title: {title}

Paper text:
{text}
"""


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


def _paper_text(paper_id: str, chunks: list[Chunk]) -> str:
    paper_chunks = [c for c in chunks if c.paper_id == paper_id]
    text = "\n\n".join(f"[{c.section}]\n{c.text}" for c in paper_chunks)
    return text[:MAX_CONTEXT_CHARS]


def run(state: ResearchState) -> ResearchState:
    llm = ChatGroq(model=settings.research_agent_model, api_key=settings.groq_api_key, temperature=0)
    structured_llm = llm.with_structured_output(ExtractionResult)

    graph = state["knowledge_graph"]
    paper_graphs: list[PaperGraph] = []

    for paper in state["papers"]:
        if graph_store.is_ingested(settings.graph_db_path, paper.arxiv_id):
            subgraph = graph_builder.subgraph_for_paper(graph, paper.arxiv_id)
            entities = [
                Entity(name=data["name"], type=data["type"], description=data.get("description", ""))
                for _, data in subgraph.nodes(data=True)
            ]
            paper_graphs.append(PaperGraph(paper_id=paper.arxiv_id, entities=entities, relations=[]))
            continue

        text = _paper_text(paper.arxiv_id, state["chunks"]) or paper.abstract

        try:
            result = structured_llm.invoke(EXTRACTION_PROMPT.format(title=paper.title, text=text))
        except Exception as e:  # noqa: BLE001
            state["errors"].append(f"Analyst failed on {paper.arxiv_id}: {e}")
            continue

        paper_graph = PaperGraph(paper_id=paper.arxiv_id, entities=result.entities, relations=result.relations)
        paper_graphs.append(paper_graph)
        graph_builder.merge_paper_graph(graph, paper_graph)

    state["paper_graphs"] = paper_graphs
    state["knowledge_graph"] = graph
    return state
