"""
SQLite-backed session store for SwarmIQ v2.

Provides a SessionStore class for persisting and retrieving research sessions,
with coordinated cleanup of associated Pinecone vectors on deletion.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from swarmiq.core.knowledge_store import KnowledgeStore


@dataclass
class Session:
    session_id: str
    query_text: str
    query_fingerprint: str
    domain_mode: str
    created_at: str
    report_markdown: str
    evaluator_json: str
    status: str  # running | complete | failed


class SessionStore:
    """SQLite-backed store for SwarmIQ research sessions.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Defaults to ``swarmiq_sessions.db``.
    """

    def __init__(self, db_path: str = "swarmiq_sessions.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.create_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create the sessions table if it does not already exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id        TEXT PRIMARY KEY,
                query_text        TEXT,
                query_fingerprint TEXT,
                domain_mode       TEXT,
                created_at        TEXT,
                report_markdown   TEXT,
                evaluator_json    TEXT,
                status            TEXT
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_session(self, session: Session) -> None:
        """Persist a session, replacing any existing row with the same session_id."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, query_text, query_fingerprint, domain_mode,
                 created_at, report_markdown, evaluator_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.query_text,
                session.query_fingerprint,
                session.domain_mode,
                session.created_at,
                session.report_markdown,
                session.evaluator_json,
                session.status,
            ),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        """Return the session with the given ID, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return Session(
            session_id=row["session_id"],
            query_text=row["query_text"],
            query_fingerprint=row["query_fingerprint"],
            domain_mode=row["domain_mode"],
            created_at=row["created_at"],
            report_markdown=row["report_markdown"],
            evaluator_json=row["evaluator_json"],
            status=row["status"],
        )

    def list_sessions(self) -> list[Session]:
        """Return all sessions ordered by created_at descending (most recent first)."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [
            Session(
                session_id=row["session_id"],
                query_text=row["query_text"],
                query_fingerprint=row["query_fingerprint"],
                domain_mode=row["domain_mode"],
                created_at=row["created_at"],
                report_markdown=row["report_markdown"],
                evaluator_json=row["evaluator_json"],
                status=row["status"],
            )
            for row in rows
        ]

    def delete_session(self, session_id: str, knowledge_store: KnowledgeStore) -> None:
        """Delete a session and its associated Pinecone vectors.

        Deletes the SQLite row first, then removes all vectors whose metadata
        ``session_id`` matches the given ID from the knowledge store.
        """
        self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()
        knowledge_store.delete(filter={"session_id": session_id})
