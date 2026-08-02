"""Critic agent: scores summaries and drives the reflection loop (Phase 2)."""

from __future__ import annotations

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.models.critique import CritiqueResult
from research_agent.models.paper import Chunk
from research_agent import progress
from research_agent.state import ResearchState

MAX_CONTEXT_CHARS = 8000

CRITIC_PROMPT = """You are a strict research reviewer grading a draft summary of an academic paper.

Score each dimension from 0 to 10:
- completeness: covers contributions, methodology, results, limitations
- grounding: claims are supported by the provided paper text (no hallucination)
- limitations_coverage: limitations and open problems are addressed

Also provide concise feedback listing specific gaps to fix if any score is below 8.

Paper title: {title}
Abstract: {abstract}

Paper text (excerpt):
{text}

Draft summary:
Contributions: {contributions}
Methodology: {methodology}
Results: {results}
Limitations: {limitations}
Future work: {future_work}
"""


class CritiqueDraft(BaseModel):
    completeness: float = Field(ge=0, le=10)
    grounding: float = Field(ge=0, le=10)
    limitations_coverage: float = Field(ge=0, le=10)
    feedback: str = ""


def _paper_text(paper_id: str, chunks: list[Chunk]) -> str:
    paper_chunks = [c for c in chunks if c.paper_id == paper_id]
    text = "\n\n".join(f"[{c.section}]\n{c.text}" for c in paper_chunks)
    return text[:MAX_CONTEXT_CHARS]


def run(state: ResearchState) -> ResearchState:
    llm = ChatGroq(model=settings.research_agent_model, api_key=settings.groq_api_key, temperature=0)
    structured_llm = llm.with_structured_output(CritiqueDraft)

    critiques: list[CritiqueResult] = []
    feedback_map: dict[str, str] = {}
    threshold = settings.research_agent_critic_threshold

    summaries = state["summaries"]
    papers_by_id = {p.arxiv_id: p for p in state["papers"]}
    for index, summary in enumerate(summaries, start=1):
        paper = papers_by_id.get(summary.paper_id)
        if paper is None:
            continue
        progress.paper_step("Critic", index, len(summaries), summary.paper_id, "scoring summary (Groq LLM)")
        text = _paper_text(summary.paper_id, state["chunks"]) or paper.abstract
        prompt = CRITIC_PROMPT.format(
            title=summary.title,
            abstract=paper.abstract,
            text=text,
            contributions=summary.contributions,
            methodology=summary.methodology,
            results=summary.results,
            limitations=summary.limitations,
            future_work=summary.future_work,
        )
        try:
            draft = structured_llm.invoke(prompt)
        except Exception as e:  # noqa: BLE001
            state["errors"].append(f"Critic failed on {summary.paper_id}: {e}")
            progress.warn(f"Critique failed for {summary.paper_id}")
            continue

        overall = round((draft.completeness + draft.grounding + draft.limitations_coverage) / 3, 2)
        needs_revision = overall < threshold
        progress.info(f"Score {overall}/10 — {'needs revision' if needs_revision else 'passed'}")
        critique = CritiqueResult(
            paper_id=summary.paper_id,
            completeness=draft.completeness,
            grounding=draft.grounding,
            limitations_coverage=draft.limitations_coverage,
            overall=overall,
            feedback=draft.feedback,
            needs_revision=needs_revision,
        )
        critiques.append(critique)
        if needs_revision:
            feedback_map[summary.paper_id] = draft.feedback

    state["critiques"] = critiques
    state["critic_feedback"] = feedback_map

    if feedback_map and state["reflection_round"] < state["max_reflection_rounds"]:
        state["reflection_round"] += 1
        state["pending_revision"] = True
        state["reflection_logs"].append(
            f"Reflection round {state['reflection_round']}: revising {len(feedback_map)} summary(ies)"
        )
    else:
        state["pending_revision"] = False
        if feedback_map:
            state["reflection_logs"].append(
                f"Max reflection rounds reached; proceeding with {len(feedback_map)} imperfect summary(ies)"
            )

    return state
