"""Schemas for the Critic agent and reflection loop (Phase 2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CritiqueResult(BaseModel):
    """Rubric scores and revision feedback for one paper summary."""

    paper_id: str
    completeness: float = Field(ge=0, le=10)
    grounding: float = Field(ge=0, le=10)
    limitations_coverage: float = Field(ge=0, le=10)
    overall: float = Field(ge=0, le=10)
    feedback: str = ""
    needs_revision: bool = False
