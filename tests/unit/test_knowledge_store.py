"""
Unit tests for KnowledgeStore (swarmiq/core/knowledge_store.py).

Tests use a mock Pinecone Index so no real API key is required.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from swarmiq.core.knowledge_store import KnowledgeStore, make_vector_id


# ---------------------------------------------------------------------------
# make_vector_id helper
# ---------------------------------------------------------------------------

class TestMakeVectorId:
    def test_format(self):
        vid = make_vector_id("sess-1", "sub-1", "https://example.com")
        parts = vid.split("#")
        assert parts[0] == "sess-1"
        assert parts[1] == "sub-1"
        assert len(parts[2]) == 16  # first 16 hex chars of sha256

    def test_url_hash_is_sha256(self):
        url = "https://example.com/page"
        expected_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        vid = make_vector_id("s", "t", url)
        assert vid.endswith(f"#{expected_hash}")

    def test_deterministic(self):
        a = make_vector_id("s", "t", "https://example.com")
        b = make_vector_id("s", "t", "https://example.com")
        assert a == b

    def test_different_urls_produce_different_ids(self):
        a = make_vector_id("s", "t", "https://example.com/a")
        b = make_vector_id("s", "t", "https://example.com/b")
        assert a != b


# ---------------------------------------------------------------------------
# KnowledgeStore construction
# ---------------------------------------------------------------------------

def _make_store(mock_index: MagicMock | None = None) -> KnowledgeStore:
    """Return a KnowledgeStore with a mocked Pinecone client."""
    mock_pc_instance = MagicMock()
    if mock_index is not None:
        mock_pc_instance.Index.return_value = mock_index

    mock_pinecone_cls = MagicMock(return_value=mock_pc_instance)
    import swarmiq.core.knowledge_store as ks_module
    original = ks_module.Pinecone
    ks_module.Pinecone = mock_pinecone_cls
    try:
        store = KnowledgeStore(api_key="test-key", index_name="test-index")
    finally:
        ks_module.Pinecone = original
    return store


class TestKnowledgeStoreInit:
    def test_raises_on_missing_api_key(self):
        with pytest.raises(ValueError, match="PINECONE_API_KEY"):
            KnowledgeStore(api_key=None, index_name="idx")

    def test_raises_on_empty_api_key(self):
        with pytest.raises(ValueError, match="PINECONE_API_KEY"):
            KnowledgeStore(api_key="", index_name="idx")

    def test_constructs_with_valid_key(self):
        mock_pc = MagicMock()
        import swarmiq.core.knowledge_store as ks_module
        original = ks_module.Pinecone
        ks_module.Pinecone = MagicMock(return_value=mock_pc)
        try:
            store = KnowledgeStore(api_key="key", index_name="idx")
        finally:
            ks_module.Pinecone = original
        assert store._index_name == "idx"


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_calls_index(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)
        vectors = [{"id": "v1", "values": [0.1, 0.2], "metadata": {"session_id": "s1"}}]
        store.upsert(vectors, namespace="s1")
        mock_index.upsert.assert_called_once_with(vectors=vectors, namespace="s1")

    def test_upsert_empty_list_skips_call(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)
        store.upsert([], namespace="s1")
        mock_index.upsert.assert_not_called()

    def test_upsert_metadata_structure(self):
        """Verify the metadata dict passed to upsert contains required fields."""
        mock_index = MagicMock()
        store = _make_store(mock_index)
        metadata = {
            "session_id": "sess-abc",
            "query_fingerprint": "abc123",
            "subtask_id": "sub-1",
            "url": "https://example.com",
            "title": "Example",
            "retrieved_at": "2024-01-01T00:00:00Z",
            "domain": "example.com",
            "domain_trust": 0.6,
            "content_preview": "Some content...",
        }
        vectors = [{"id": "v1", "values": [0.0] * 768, "metadata": metadata}]
        store.upsert(vectors, namespace="sess-abc")
        call_kwargs = mock_index.upsert.call_args
        upserted = call_kwargs.kwargs["vectors"][0]
        for key in ("session_id", "query_fingerprint", "subtask_id", "url", "retrieved_at"):
            assert key in upserted["metadata"]


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:
    def _make_match(self, id_: str, score: float, metadata: dict) -> MagicMock:
        m = MagicMock()
        m.id = id_
        m.score = score
        m.metadata = metadata
        return m

    def test_query_filters_low_scores(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)

        high = self._make_match("v1", 0.95, {"session_id": "s1"})
        low = self._make_match("v2", 0.70, {"session_id": "s1"})
        response = MagicMock()
        response.matches = [high, low]
        mock_index.query.return_value = response

        results = store.query([0.1] * 768, top_k=20, namespace="s1")
        assert len(results) == 1
        assert results[0]["id"] == "v1"

    def test_query_passes_filter(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)
        response = MagicMock()
        response.matches = []
        mock_index.query.return_value = response

        store.query([0.1] * 768, top_k=20, filter={"session_id": "s1"}, namespace="s1")
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["filter"] == {"session_id": "s1"}

    def test_query_no_filter_omits_filter_key(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)
        response = MagicMock()
        response.matches = []
        mock_index.query.return_value = response

        store.query([0.1] * 768, namespace="s1")
        call_kwargs = mock_index.query.call_args.kwargs
        assert "filter" not in call_kwargs

    def test_query_returns_score_and_metadata(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)

        match = self._make_match("v1", 0.90, {"url": "https://example.com"})
        response = MagicMock()
        response.matches = [match]
        mock_index.query.return_value = response

        results = store.query([0.1] * 768)
        assert results[0]["score"] == 0.90
        assert results[0]["metadata"]["url"] == "https://example.com"

    def test_query_exact_threshold_included(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)

        match = self._make_match("v1", 0.80, {})
        response = MagicMock()
        response.matches = [match]
        mock_index.query.return_value = response

        results = store.query([0.1] * 768)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_passes_filter(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)
        store.delete(filter={"session_id": "s1"}, namespace="s1")
        mock_index.delete.assert_called_once_with(filter={"session_id": "s1"}, namespace="s1")

    def test_delete_default_namespace(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)
        store.delete(filter={"session_id": "s1"})
        mock_index.delete.assert_called_once_with(filter={"session_id": "s1"}, namespace="")


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------

class TestGetById:
    def test_returns_none_when_not_found(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)
        response = MagicMock()
        response.vectors = {}
        mock_index.fetch.return_value = response

        result = store.get_by_id("missing-id")
        assert result is None

    def test_returns_vector_dict(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)

        vec = MagicMock()
        vec.id = "v1"
        vec.values = [0.1, 0.2]
        vec.metadata = {"url": "https://example.com"}

        response = MagicMock()
        response.vectors = {"v1": vec}
        mock_index.fetch.return_value = response

        result = store.get_by_id("v1", namespace="s1")
        assert result is not None
        assert result["id"] == "v1"
        assert result["values"] == [0.1, 0.2]
        assert result["metadata"]["url"] == "https://example.com"

    def test_fetch_called_with_correct_args(self):
        mock_index = MagicMock()
        store = _make_store(mock_index)
        response = MagicMock()
        response.vectors = {}
        mock_index.fetch.return_value = response

        store.get_by_id("v1", namespace="ns1")
        mock_index.fetch.assert_called_once_with(ids=["v1"], namespace="ns1")
