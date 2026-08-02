"""PDF text extraction, heuristic section detection, and chunking."""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # pymupdf

from research_agent.models.paper import Chunk

SECTION_HEADINGS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
]

_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(" + "|".join(re.escape(h) for h in SECTION_HEADINGS) + r")\s*$",
    re.IGNORECASE,
)


def extract_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF using pymupdf."""
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def split_sections(raw_text: str) -> dict[str, str]:
    """Split raw paper text into named sections using heading heuristics.

    Falls back to a single "full_text" section if no headings are detected.
    """
    lines = raw_text.splitlines()
    sections: dict[str, list[str]] = {}
    current = "preamble"
    sections[current] = []

    for line in lines:
        match = _HEADING_RE.match(line.strip())
        if match:
            current = match.group(1).lower()
            sections.setdefault(current, [])
        else:
            sections[current].append(line)

    joined = {name: "\n".join(body).strip() for name, body in sections.items()}
    joined = {name: body for name, body in joined.items() if body}

    if len(joined) <= 1:
        return {"full_text": raw_text.strip()}
    return joined


def chunk_sections(
    paper_id: str,
    sections: dict[str, str],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk each section by word count with overlap for context continuity."""
    chunks: list[Chunk] = []
    for section_name, text in sections.items():
        words = text.split()
        if not words:
            continue
        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    paper_id=paper_id,
                    section=section_name,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
            chunk_index += 1
            if end == len(words):
                break
            start = end - overlap

    return chunks


def parse_pdf(paper_id: str, pdf_path: Path) -> list[Chunk]:
    """Full pipeline: extract text, detect sections, and chunk them."""
    raw_text = extract_text(pdf_path)
    sections = split_sections(raw_text)
    return chunk_sections(paper_id, sections)
