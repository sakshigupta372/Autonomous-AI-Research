"""Schemas for experiment reproduction (Phase 3)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReproducibilityResult(BaseModel):
    """Outcome of attempting to reproduce a paper's method in a sandbox."""

    paper_id: str
    title: str
    script: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    success: bool = False
    reproducibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    paper_reported_metrics: list[str] = Field(default_factory=list)
    reproduced_metrics: list[str] = Field(default_factory=list)
    notes: str = ""
