"""Pydantic schema for the Writer agent's structured summary output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchSummary(BaseModel):
    """Structured research summary for a single paper."""

    paper_id: str
    title: str
    contributions: list[str] = Field(default_factory=list)
    methodology: str = ""
    results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
