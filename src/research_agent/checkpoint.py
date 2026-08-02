"""LangGraph SQLite checkpoint helpers for session resume (Phase 4)."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent.config import settings


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def get_checkpointer():
    """Yield a SqliteSaver backed by data/checkpoints.db."""
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_db_path), check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()


def list_sessions() -> list[str]:
    """Return known session thread IDs from the checkpoint database."""
    if not settings.checkpoint_db_path.exists():
        return []
    conn = sqlite3.connect(str(settings.checkpoint_db_path))
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        return [row[0] for row in rows if row[0]]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
