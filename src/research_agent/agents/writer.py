"""Writer agent: produces a grounded, structured research summary per paper,
and renders it as a Markdown report with an embedded Mermaid concept graph.
"""

from __future__ import annotations

import networkx as nx
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.models.paper import Chunk, Paper
from research_agent.models.reproduction import ReproducibilityResult
from research_agent.models.summary import ResearchSummary
from research_agent.state import ResearchState
from research_agent.tools import graph_builder

MAX_CONTEXT_CHARS = 12000

WRITER_PROMPT = """You are a research scientist writing a structured summary of an academic paper.

Ground every statement strictly in the provided paper text and knowledge graph below. Do not invent
results, methods, or numbers that are not present in the text. If limitations are not stated explicitly,
you may infer them cautiously, but keep inferred limitations clearly conservative.

Paper title: {title}
Authors: {authors}
Abstract: {abstract}

Knowledge graph entities and relations extracted from this paper:
{graph_summary}

Paper text (chunks):
{text}

Write:
- contributions: the key contributions/claims of the paper
- methodology: a concise description of the approach
- results: key quantitative/qualitative results
- limitations: limitations or open problems, explicit or reasonably inferred
- future_work: suggested next steps or open questions
{revision_block}
"""


class SummaryDraft(BaseModel):
    contributions: list[str] = Field(default_factory=list)
    methodology: str = ""
    results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)


def _paper_text(paper_id: str, chunks: list[Chunk]) -> str:
    paper_chunks = [c for c in chunks if c.paper_id == paper_id]
    text = "\n\n".join(f"[{c.section}]\n{c.text}" for c in paper_chunks)
    return text[:MAX_CONTEXT_CHARS]


def _graph_summary(graph: nx.MultiDiGraph, paper_id: str) -> str:
    subgraph = graph_builder.subgraph_for_paper(graph, paper_id)
    lines = [f"- ({data['type']}) {data['name']}: {data.get('description', '')}" for _, data in subgraph.nodes(data=True)]
    lines += [
        f"- {graph.nodes[source]['name']} -[{data['type']}]-> {graph.nodes[target]['name']}"
        for source, target, data in subgraph.edges(data=True)
    ]
    return "\n".join(lines) if lines else "(no graph data extracted)"


def run(state: ResearchState) -> ResearchState:
    llm = ChatGroq(model=settings.research_agent_model, api_key=settings.groq_api_key, temperature=0)
    structured_llm = llm.with_structured_output(SummaryDraft)

    is_revision = bool(state.get("critic_feedback")) and state.get("reflection_round", 0) > 0
    summaries_by_id = {s.paper_id: s for s in state.get("summaries", [])}

    for paper in state["papers"]:
        if is_revision and paper.arxiv_id not in state.get("critic_feedback", {}):
            continue

        text = _paper_text(paper.arxiv_id, state["chunks"]) or paper.abstract
        feedback = state.get("critic_feedback", {}).get(paper.arxiv_id, "")
        revision_block = ""
        if feedback:
            revision_block = f"\n\nRevision feedback from critic (address these gaps):\n{feedback}"

        prompt = WRITER_PROMPT.format(
            title=paper.title,
            authors=", ".join(paper.authors),
            abstract=paper.abstract,
            graph_summary=_graph_summary(state["knowledge_graph"], paper.arxiv_id),
            text=text,
            revision_block=revision_block,
        )
        try:
            draft = structured_llm.invoke(prompt)
        except Exception as e:  # noqa: BLE001
            state["errors"].append(f"Writer failed on {paper.arxiv_id}: {e}")
            continue

        summaries_by_id[paper.arxiv_id] = ResearchSummary(
            paper_id=paper.arxiv_id,
            title=paper.title,
            contributions=draft.contributions,
            methodology=draft.methodology,
            results=draft.results,
            limitations=draft.limitations,
            future_work=draft.future_work,
        )

    state["summaries"] = list(summaries_by_id.values())
    return state


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- (none extracted)"]


def render_markdown(
    paper: Paper,
    summary: ResearchSummary,
    graph: nx.MultiDiGraph,
    reproduction: ReproducibilityResult | None = None,
) -> str:
    """Render a ResearchSummary as a Markdown report with a Mermaid concept graph."""
    mermaid = graph_builder.to_mermaid(graph, paper.arxiv_id)
    lines = [
        f"# {summary.title}",
        "",
        f"- **arXiv ID:** [{paper.arxiv_id}]({paper.pdf_url})",
        f"- **Authors:** {', '.join(paper.authors)}",
        f"- **Published:** {paper.published}",
        f"- **Categories:** {', '.join(paper.categories)}",
        "",
        "## Abstract",
        "",
        paper.abstract,
        "",
        "## Key Contributions",
        "",
        *_bullets(summary.contributions),
        "",
        "## Methodology",
        "",
        summary.methodology or "(not extracted)",
        "",
        "## Results",
        "",
        *_bullets(summary.results),
        "",
        "## Limitations",
        "",
        *_bullets(summary.limitations),
        "",
        "## Future Work",
        "",
        *_bullets(summary.future_work),
        "",
    ]

    if reproduction is not None:
        lines.extend(
            [
                "## Reproducibility",
                "",
                f"- **Score:** {reproduction.reproducibility_score:.0%}",
                f"- **Execution success:** {reproduction.success}",
                f"- **Paper-reported metrics:** {', '.join(reproduction.paper_reported_metrics) or '(none)'}",
                f"- **Reproduced metrics:** {', '.join(reproduction.reproduced_metrics) or '(none)'}",
                f"- **Notes:** {reproduction.notes or '(none)'}",
                "",
                "### Generated Script",
                "",
                "```python",
                reproduction.script or "# (no script generated)",
                "```",
                "",
                "### Sandbox Output",
                "",
                "```",
                reproduction.stdout or "(no stdout)",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Concept Graph",
            "",
            "```mermaid",
            mermaid,
            "```",
            "",
        ]
    )
    return "\n".join(lines)
