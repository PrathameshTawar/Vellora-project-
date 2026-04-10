"""
Unit tests for SessionStore (swarmiq/store/session_store.py).

Tests use an in-memory SQLite database — no file I/O required.

Validates: Requirements 14.1, 14.2, 14.3, 14.4
"""
from __future__ import annotations

import sqlite3
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing swarmiq modules
# ---------------------------------------------------------------------------

def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _dep in [
    "autogen",
    "sentence_transformers",
    "pinecone",
    "tavily",
    "serpapi",
    "transformers",
    "plotly",
    "plotly.graph_objects",
    "matplotlib",
    "matplotlib.pyplot",
]:
    if _dep not in sys.modules:
        _stub(_dep)

sys.modules["autogen"].AssistantAgent = MagicMock  # type: ignore[attr-defined]
sys.modules["sentence_transformers"].SentenceTransformer = MagicMock  # type: ignore[attr-defined]
sys.modules["pinecone"].Pinecone = MagicMock  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

from swarmiq.store.session_store import Session, SessionStore  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> SessionStore:
    """Return a SessionStore backed by an in-memory SQLite database."""
    store = SessionStore.__new__(SessionStore)
    store._db_path = ":memory:"
    store._conn = sqlite3.connect(":memory:", check_same_thread=False)
    store._conn.row_factory = sqlite3.Row
    store.create_schema()
    return store


def _make_session(
    session_id: str = "sess-001",
    query_text: str = "What is quantum computing?",
    query_fingerprint: str = "fp-abc123",
    domain_mode: str = "research",
    created_at: str = "2024-06-01T12:00:00",
    report_markdown: str = "# Report\nSome content.",
    evaluator_json: str = '{"score": 0.9}',
    status: str = "complete",
) -> Session:
    return Session(
        session_id=session_id,
        query_text=query_text,
        query_fingerprint=query_fingerprint,
        domain_mode=domain_mode,
        created_at=created_at,
        report_markdown=report_markdown,
        evaluator_json=evaluator_json,
        status=status,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveAndGet:
    def test_save_then_get_returns_same_session(self):
        """save_session then get_session returns the same session."""
        store = _make_store()
        session = _make_session()
        store.save_session(session)

        result = store.get_session("sess-001")

        assert result is not None
        assert result.session_id == session.session_id
        assert result.query_text == session.query_text
        assert result.report_markdown == session.report_markdown
        assert result.evaluator_json == session.evaluator_json
        assert result.status == session.status

    def test_get_returns_none_for_missing_session_id(self):
        """get_session returns None for a session_id that was never saved."""
        store = _make_store()

        result = store.get_session("does-not-exist")

        assert result is None

    def test_save_same_id_updates_existing_record(self):
        """save_session with the same session_id replaces the existing row (INSERT OR REPLACE)."""
        store = _make_store()
        original = _make_session(report_markdown="original report")
        store.save_session(original)

        updated = _make_session(report_markdown="updated report")
        store.save_session(updated)

        result = store.get_session("sess-001")
        assert result is not None
        assert result.report_markdown == "updated report"

        # Only one row should exist
        count = store._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = 'sess-001'"
        ).fetchone()[0]
        assert count == 1


class TestListSessions:
    def test_list_sessions_returns_all_saved_sessions(self):
        """list_sessions returns all saved sessions."""
        store = _make_store()
        s1 = _make_session(session_id="s1", created_at="2024-01-01T10:00:00")
        s2 = _make_session(session_id="s2", created_at="2024-01-02T10:00:00")
        s3 = _make_session(session_id="s3", created_at="2024-01-03T10:00:00")
        store.save_session(s1)
        store.save_session(s2)
        store.save_session(s3)

        result = store.list_sessions()

        assert len(result) == 3
        ids = {s.session_id for s in result}
        assert ids == {"s1", "s2", "s3"}

    def test_list_sessions_ordered_by_created_at_desc(self):
        """list_sessions returns sessions ordered by created_at DESC (most recent first)."""
        store = _make_store()
        store.save_session(_make_session(session_id="old", created_at="2024-01-01T00:00:00"))
        store.save_session(_make_session(session_id="mid", created_at="2024-06-01T00:00:00"))
        store.save_session(_make_session(session_id="new", created_at="2024-12-01T00:00:00"))

        result = store.list_sessions()

        assert result[0].session_id == "new"
        assert result[1].session_id == "mid"
        assert result[2].session_id == "old"

    def test_list_sessions_returns_empty_list_when_no_sessions(self):
        """list_sessions returns an empty list when no sessions have been saved."""
        store = _make_store()

        result = store.list_sessions()

        assert result == []


class TestDeleteSession:
    def test_delete_session_removes_from_store(self):
        """delete_session removes the session from the store."""
        store = _make_store()
        store.save_session(_make_session())
        mock_ks = MagicMock()

        store.delete_session("sess-001", mock_ks)

        assert store.get_session("sess-001") is None

    def test_delete_session_calls_knowledge_store_delete_with_correct_filter(self):
        """delete_session calls knowledge_store.delete with the correct session_id filter."""
        store = _make_store()
        store.save_session(_make_session(session_id="sess-xyz"))
        mock_ks = MagicMock()

        store.delete_session("sess-xyz", mock_ks)

        mock_ks.delete.assert_called_once_with(filter={"session_id": "sess-xyz"})

    def test_delete_session_on_nonexistent_id_does_not_raise(self):
        """delete_session on a non-existent session_id does not raise an exception."""
        store = _make_store()
        mock_ks = MagicMock()

        # Should not raise
        store.delete_session("never-existed", mock_ks)

        # delete on knowledge store is still called
        mock_ks.delete.assert_called_once_with(filter={"session_id": "never-existed"})


class TestFieldPreservation:
    def test_all_session_fields_preserved_after_round_trip(self):
        """All Session fields are preserved exactly after a save/get round-trip."""
        store = _make_store()
        session = Session(
            session_id="full-test-id",
            query_text="Full field test query",
            query_fingerprint="fp-full-001",
            domain_mode="medical",
            created_at="2024-03-15T08:30:00",
            report_markdown="## Full Report\nDetailed content here.",
            evaluator_json='{"accuracy": 0.95, "completeness": 0.88}',
            status="running",
        )
        store.save_session(session)

        result = store.get_session("full-test-id")

        assert result is not None
        assert result.session_id == session.session_id
        assert result.query_text == session.query_text
        assert result.query_fingerprint == session.query_fingerprint
        assert result.domain_mode == session.domain_mode
        assert result.created_at == session.created_at
        assert result.report_markdown == session.report_markdown
        assert result.evaluator_json == session.evaluator_json
        assert result.status == session.status
