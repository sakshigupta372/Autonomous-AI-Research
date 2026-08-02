"""SQLite-backed persistence for the knowledge graph and paper ingestion record.

This is the long-term "semantic memory" layer: it lets the agent skip
re-downloading/re-analyzing papers it has already read, and lets the
knowledge graph grow across sessions instead of being rebuilt from scratch.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import networkx as nx

from research_agent.models.paper import Paper
from research_agent.tools import graph_builder


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            pdf_path TEXT,
            ingested_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_snapshot (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            data TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def is_ingested(db_path: Path, arxiv_id: str) -> bool:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        return row is not None
    finally:
        conn.close()


def mark_ingested(db_path: Path, paper: Paper) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO papers (arxiv_id, title, pdf_path, ingested_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(arxiv_id) DO UPDATE SET title = excluded.title, pdf_path = excluded.pdf_path
            """,
            (paper.arxiv_id, paper.title, paper.local_path),
        )
        conn.commit()
    finally:
        conn.close()


def load_graph(db_path: Path) -> nx.MultiDiGraph:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT data FROM graph_snapshot WHERE id = 1").fetchone()
        if row is None:
            return graph_builder.new_graph()
        return graph_builder.from_json(json.loads(row[0]))
    finally:
        conn.close()


def save_graph(db_path: Path, graph: nx.MultiDiGraph) -> None:
    conn = _connect(db_path)
    try:
        payload = json.dumps(graph_builder.to_json(graph))
        conn.execute(
            """
            INSERT INTO graph_snapshot (id, data) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET data = excluded.data
            """,
            (payload,),
        )
        conn.commit()
    finally:
        conn.close()
