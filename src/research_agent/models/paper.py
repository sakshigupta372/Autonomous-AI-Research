"""Pydantic schemas for papers, chunks, and GraphRAG entities/relations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["Method", "Dataset", "Metric", "Claim", "Limitation"]
RelationType = Literal["USES", "EVALUATED_ON", "IMPROVES", "LIMITED_BY"]


class Paper(BaseModel):
    """Metadata for a single arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str
    categories: list[str] = Field(default_factory=list)
    published: str
    pdf_url: str
    local_path: str | None = None


class Chunk(BaseModel):
    """A section-aware chunk of a paper's text, used for RAG and memory."""

    paper_id: str
    section: str
    chunk_index: int
    text: str


class Entity(BaseModel):
    """A GraphRAG node extracted from a paper (method, dataset, metric, etc.)."""

    name: str
    type: EntityType
    description: str = ""


class Relation(BaseModel):
    """A GraphRAG edge connecting two entities extracted from a paper."""

    source: str
    target: str
    type: RelationType
    description: str = ""


class PaperGraph(BaseModel):
    """The LLM-extracted knowledge graph for a single paper."""

    paper_id: str
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
