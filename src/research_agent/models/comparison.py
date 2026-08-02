"""Schemas for cross-paper comparison (Phase 2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PaperComparisonRow(BaseModel):
    """One row in the method/dataset/metric comparison table."""

    paper_id: str
    title: str
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    key_limitations: list[str] = Field(default_factory=list)


class ComparisonReport(BaseModel):
    """Cross-paper synthesis for a research run."""

    topic: str
    rows: list[PaperComparisonRow] = Field(default_factory=list)
    shared_methods: list[str] = Field(default_factory=list)
    shared_datasets: list[str] = Field(default_factory=list)
    synthesis: str = ""
    open_questions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
