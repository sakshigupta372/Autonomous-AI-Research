"""Experimenter agent: generates and runs reproduction scripts in a sandbox (Phase 3)."""

from __future__ import annotations

import re

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.models.paper import Chunk
from research_agent.models.reproduction import ReproducibilityResult
from research_agent import progress
from research_agent.state import ResearchState
from research_agent.tools.sandbox import run_python_script

MAX_CONTEXT_CHARS = 8000

EXPERIMENTER_PROMPT = """You are a reproducibility engineer. Given an academic paper's methods section,
write a SMALL standalone Python script that demonstrates the core algorithm or experiment setup.

Rules:
- Use only Python standard library (no pip packages)
- Keep it under 80 lines
- Include print() statements that output metric names and numeric values
- If the paper lacks enough detail, implement a simplified toy version and print "TOY_REPRODUCTION"
- Do not use network, file I/O outside the script, or subprocess
- The script must be runnable as: python experiment.py

Paper title: {title}
Reported results from summary: {results}

Methods / experiment text:
{text}

Return ONLY valid Python code in the script field.
"""


class ExperimentDraft(BaseModel):
    script: str = ""
    paper_reported_metrics: list[str] = Field(default_factory=list)
    notes: str = ""


def _methods_text(paper_id: str, chunks: list[Chunk]) -> str:
    method_sections = ("method", "methods", "methodology", "approach", "experiments", "experimental setup", "results")
    paper_chunks = [c for c in chunks if c.paper_id == paper_id and c.section in method_sections]
    if not paper_chunks:
        paper_chunks = [c for c in chunks if c.paper_id == paper_id][:3]
    text = "\n\n".join(f"[{c.section}]\n{c.text}" for c in paper_chunks)
    return text[:MAX_CONTEXT_CHARS]


def _extract_metrics_from_output(stdout: str) -> list[str]:
    patterns = [
        r"(?i)(accuracy|precision|recall|f1|loss|score|r2|auc|mse|mae)\s*[:=]\s*[\d.]+",
        r"(?i)[\d.]+\s*(?:%|\bpercent\b)",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, stdout))
    return found


def _score_reproduction(reported: list[str], reproduced: list[str], success: bool) -> float:
    if not success:
        return 0.2 if reproduced else 0.0
    if reported and reproduced:
        return 0.8
    if reproduced:
        return 0.6
    return 0.4


def run(state: ResearchState) -> ResearchState:
    if not state.get("enable_experiments", True):
        progress.info("Experiment reproduction skipped")
        state["reproductions"] = []
        return state

    llm = ChatGroq(model=settings.research_agent_model, api_key=settings.groq_api_key, temperature=0)
    structured_llm = llm.with_structured_output(ExperimentDraft)

    summaries_by_id = {s.paper_id: s for s in state["summaries"]}
    results: list[ReproducibilityResult] = []
    papers = state["papers"]

    for index, paper in enumerate(papers, start=1):
        progress.paper_step("Experimenter", index, len(papers), paper.arxiv_id, "generating script (Groq LLM)")
        summary = summaries_by_id.get(paper.arxiv_id)
        text = _methods_text(paper.arxiv_id, state["chunks"]) or paper.abstract
        prompt = EXPERIMENTER_PROMPT.format(
            title=paper.title,
            results=summary.results if summary else [],
            text=text,
        )
        try:
            draft = structured_llm.invoke(prompt)
            script = draft.script.strip()
            if script.startswith("```"):
                script = re.sub(r"^```(?:python)?\n?", "", script)
                script = re.sub(r"\n?```$", "", script)
            progress.info(f"Running sandbox (timeout {settings.research_agent_sandbox_timeout}s)...")
            sandbox = run_python_script(
                script,
                timeout=settings.research_agent_sandbox_timeout,
                work_dir=settings.sandbox_dir / paper.arxiv_id.replace("/", "_"),
            )
            reproduced = _extract_metrics_from_output(sandbox.stdout)
            reported = draft.paper_reported_metrics or (summary.results if summary else [])
            score = _score_reproduction(reported, reproduced, sandbox.success)
            progress.info(f"Reproducibility score: {score:.0%} (success={sandbox.success})")
            results.append(
                ReproducibilityResult(
                    paper_id=paper.arxiv_id,
                    title=paper.title,
                    script=script,
                    stdout=sandbox.stdout,
                    stderr=sandbox.stderr,
                    exit_code=sandbox.exit_code,
                    success=sandbox.success,
                    reproducibility_score=score,
                    paper_reported_metrics=reported,
                    reproduced_metrics=reproduced,
                    notes=draft.notes,
                )
            )
        except Exception as e:  # noqa: BLE001
            state["errors"].append(f"Experimenter failed on {paper.arxiv_id}: {e}")
            progress.warn(f"Reproduction failed for {paper.arxiv_id}")
            results.append(
                ReproducibilityResult(
                    paper_id=paper.arxiv_id,
                    title=paper.title,
                    notes=f"Generation or execution failed: {e}",
                )
            )

    state["reproductions"] = results
    return state
