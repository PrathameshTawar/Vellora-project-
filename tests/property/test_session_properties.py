"""
Property-based tests for SessionStore (Properties 20, 21, 28).

Feature: swarmiq-v2
Validates: Requirements 14.1, 14.2, 14.3, 14.4
"""
from __future__ import annotations

import sqlite3
import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Stub out heavy dependencies before any swarmiq imports
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
    session_id: str,
    query_text: str,
    report_markdown: str,
    evaluator_json: str,
    created_at: str,
) -> Session:
    return Session(
        session_id=session_id,
        query_text=query_text,
        query_fingerprint="fp-test",
        domain_mode="research",
        created_at=created_at,
        report_markdown=report_markdown,
        evaluator_json=evaluator_json,
        status="complete",
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

session_id_st = st.uuids().map(str)
text_st = st.text(min_size=1, max_size=100)
datetime_st = st.datetimes().map(lambda dt: dt.isoformat())

# ---------------------------------------------------------------------------
# Property 20: Session persistence round-trip
# ---------------------------------------------------------------------------


class TestSessionPersistenceRoundTrip:
    """
    # Feature: swarmiq-v2, Property 20: Session persistence round-trip —
    # for any completed session saved to the session store, loading that session
    # by its session_id must return the original report_markdown, evaluator_json,
    # and associated figures unchanged.
    """

    @given(session_id_st, text_st, text_st, text_st, datetime_st)
    @settings(max_examples=100)
    def test_round_trip_preserves_fields(
        self,
        session_id: str,
        query_text: str,
        report_markdown: str,
        evaluator_json: str,
        created_at: str,
    ):
        # Feature: swarmiq-v2, Property 20: Session persistence round-trip
        store = _make_store()
        session = _make_session(
            session_id=session_id,
            query_text=query_text,
            report_markdown=report_markdown,
            evaluator_json=evaluator_json,
            created_at=created_at,
        )

        store.save_session(session)
        loaded = store.get_session(session_id)

        assert loaded is not None, (
            f"get_session returned None for session_id={session_id!r}"
        )
        assert loaded.report_markdown == report_markdown, (
            "report_markdown changed after round-trip: "
            f"expected {report_markdown!r}, got {loaded.report_markdown!r}"
        )
        assert loaded.evaluator_json == evaluator_json, (
            "evaluator_json changed after round-trip: "
            f"expected {evaluator_json!r}, got {loaded.evaluator_json!r}"
        )
        assert loaded.session_id == session_id
        assert loaded.query_text == query_text


# ---------------------------------------------------------------------------
# Property 21: Session deletion removes data from store and Pinecone
# ---------------------------------------------------------------------------


class TestSessionDeletionRemovesData:
    """
    # Feature: swarmiq-v2, Property 21: Session deletion removes data from store
    # and Pinecone — for any session that has been deleted, querying the session
    # store by session_id must return no record, and querying Pinecone with a
    # filter on that session_id must return zero vectors.
    """

    @given(session_id_st, text_st, text_st, text_st, datetime_st)
    @settings(max_examples=100)
    def test_deletion_removes_from_store_and_calls_pinecone_delete(
        self,
        session_id: str,
        query_text: str,
        report_markdown: str,
        evaluator_json: str,
        created_at: str,
    ):
        # Feature: swarmiq-v2, Property 21: Session deletion removes data from store and Pinecone
        store = _make_store()
        session = _make_session(
            session_id=session_id,
            query_text=query_text,
            report_markdown=report_markdown,
            evaluator_json=evaluator_json,
            created_at=created_at,
        )

        store.save_session(session)
        assert store.get_session(session_id) is not None, (
            "Session should exist before deletion"
        )

        # Mock KnowledgeStore — track delete calls
        mock_knowledge_store = MagicMock()
        delete_calls: list[dict] = []
        mock_knowledge_store.delete.side_effect = lambda filter, **_: delete_calls.append(filter)

        store.delete_session(session_id, mock_knowledge_store)

        # Session must be gone from the store
        assert store.get_session(session_id) is None, (
            f"Session {session_id!r} still present in store after deletion"
        )

        # KnowledgeStore.delete must have been called with the correct session_id filter
        assert len(delete_calls) == 1, (
            f"Expected exactly 1 call to knowledge_store.delete, got {len(delete_calls)}"
        )
        assert delete_calls[0].get("session_id") == session_id, (
            f"delete filter session_id mismatch: expected {session_id!r}, "
            f"got {delete_calls[0].get('session_id')!r}"
        )


# ---------------------------------------------------------------------------
# Property 28: Session history sorted by most recent first
# ---------------------------------------------------------------------------


class TestSessionHistorySortedByMostRecentFirst:
    """
    # Feature: swarmiq-v2, Property 28: Session history sorted by most recent first —
    # for any session history list returned by the session store, the entries must
    # be ordered by created_at descending (most recent first).
    """

    @given(
        st.lists(
            st.tuples(session_id_st, text_st, datetime_st),
            min_size=1,
            max_size=20,
            unique_by=lambda t: t[0],  # unique session_ids
        )
    )
    @settings(max_examples=100)
    def test_list_sessions_ordered_most_recent_first(
        self,
        sessions_data: list[tuple[str, str, str]],
    ):
        # Feature: swarmiq-v2, Property 28: Session history sorted by most recent first
        store = _make_store()

        for session_id, query_text, created_at in sessions_data:
            session = _make_session(
                session_id=session_id,
                query_text=query_text,
                report_markdown="report",
                evaluator_json="{}",
                created_at=created_at,
            )
            store.save_session(session)

        result = store.list_sessions()

        assert len(result) == len(sessions_data), (
            f"Expected {len(sessions_data)} sessions, got {len(result)}"
        )

        # Verify descending order by created_at string (ISO-8601 lexicographic order)
        for i in range(len(result) - 1):
            assert result[i].created_at >= result[i + 1].created_at, (
                f"Sessions not sorted descending at index {i}: "
                f"{result[i].created_at!r} < {result[i + 1].created_at!r}"
            )
