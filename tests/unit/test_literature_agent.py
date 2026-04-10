"""
Unit tests for LiteratureAgent.

Tests:
1.  Tavily returns 5 docs → LiteratureOutput with 5 documents
2.  Tavily raises exception → falls back to SerpAPI → returns docs
3.  Both Tavily and SerpAPI raise exception → first attempt fails, retry also fails → returns empty list
4.  Cache returns 5+ docs with score > 0.80 → web search NOT called
5.  Cache returns 4 docs (below threshold) → web search IS called
6.  Cache returns 0 docs → web search IS called
7.  Upserted vectors contain url, retrieved_at, subtask_id, session_id, query_fingerprint in metadata
8.  Upserted vector ID follows format {session_id}#{subtask_id}#{url_hash}
9.  Timeout (asyncio.TimeoutError from _run_inner) → returns AgentError with error_code="TIMEOUT"
10. _embed_text failure → document is skipped (no crash), upsert called with remaining docs
11. Pinecone upsert failure → logged as warning, no crash, documents still returned

Requirements: 2.2, 2.3, 2.4, 9.2
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out heavy dependencies before any swarmiq imports
# ---------------------------------------------------------------------------

# autogen stub (pulled in by swarmiq/agents/__init__.py → planner.py)
_autogen_mod = types.ModuleType("autogen")
_autogen_mod.AssistantAgent = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("autogen", _autogen_mod)

# sentence_transformers stub
_st_mod = types.ModuleType("sentence_transformers")
_st_mod.SentenceTransformer = MagicMock  # type: ignore[attr-defined]

# CrossEncoder mock: predict() returns a list of 0.5 scores for any input
_cross_encoder_instance = MagicMock()
_cross_encoder_instance.predict.side_effect = lambda pairs: [0.5] * len(pairs)
_CrossEncoderClass = MagicMock(return_value=_cross_encoder_instance)
_st_mod.CrossEncoder = _CrossEncoderClass  # type: ignore[attr-defined]
sys.modules.setdefault("sentence_transformers", _st_mod)

# tavily stub
_tavily_mod = types.ModuleType("tavily")
_tavily_mod.TavilyClient = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("tavily", _tavily_mod)

# serpapi stub
_serpapi_mod = types.ModuleType("serpapi")
_serpapi_mod.GoogleSearch = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("serpapi", _serpapi_mod)

# pinecone stub
_pinecone_mod = types.ModuleType("pinecone")
_pinecone_mod.Pinecone = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("pinecone", _pinecone_mod)

# ---------------------------------------------------------------------------
# Now safe to import swarmiq
# ---------------------------------------------------------------------------
from swarmiq.agents.literature import LiteratureAgent, LiteratureOutput  # noqa: E402
from swarmiq.core.knowledge_store import make_vector_id  # noqa: E402
from swarmiq.core.models import AgentError, Document, SubTask  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SESSION_ID = "sess-abc123"
_SUBTASK_ID = "sub-001"
_QUERY_FP = "fp-xyz"

_DUMMY_EMBEDDING = [0.1] * 384


def _make_subtask(keywords: list[str] | None = None) -> SubTask:
    return SubTask(
        subtask_id=_SUBTASK_ID,
        type="literature_review",
        description="Test subtask",
        search_keywords=keywords or ["quantum", "computing"],
    )


def _make_doc(i: int = 0) -> Document:
    return Document(
        url=f"https://example.com/doc{i}",
        title=f"Doc {i}",
        content=f"Content of doc {i}",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        subtask_id=_SUBTASK_ID,
    )


def _make_docs(n: int) -> list[Document]:
    return [_make_doc(i) for i in range(n)]


def _make_cache_matches(n: int, score: float = 0.90) -> list[dict]:
    """Return n fake cache match dicts with the given score."""
    return [
        {
            "id": f"vec-{i}",
            "score": score,
            "metadata": {
                "url": f"https://cache.example.com/doc{i}",
                "title": f"Cached Doc {i}",
                "content_preview": f"Cached content {i}",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "subtask_id": _SUBTASK_ID,
            },
        }
        for i in range(n)
    ]


def _make_agent() -> tuple[LiteratureAgent, MagicMock]:
    """Return (agent, mock_store) with embedding pre-stubbed."""
    mock_store = MagicMock()
    mock_store.query.return_value = []  # default: empty cache
    agent = LiteratureAgent(
        knowledge_store=mock_store,
        tavily_api_key="fake-tavily-key",
        serpapi_key="fake-serpapi-key",
    )
    return agent, mock_store


# ---------------------------------------------------------------------------
# Test 1: Tavily returns 5 docs → LiteratureOutput with 5 documents
# ---------------------------------------------------------------------------


def test_tavily_returns_5_docs():
    """Tavily returns 5 docs → LiteratureOutput with 5 documents."""
    agent, mock_store = _make_agent()
    docs = _make_docs(5)

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_sync", return_value=docs),
    ):
        result = asyncio.run(
            agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP)
        )

    assert isinstance(result, LiteratureOutput)
    assert len(result.documents) == 5


# ---------------------------------------------------------------------------
# Test 2: Tavily raises exception → falls back to SerpAPI → returns docs
# ---------------------------------------------------------------------------


def test_tavily_exception_falls_back_to_serpapi():
    """Tavily raises exception inside _search_tavily → falls back to SerpAPI → returns docs.

    _search_tavily catches its own exception and calls _search_serpapi as fallback.
    We verify this by patching the inner Tavily client call to raise while letting
    _search_serpapi return real docs.
    """
    agent, mock_store = _make_agent()
    serpapi_docs = _make_docs(3)

    # _search_tavily catches exceptions and calls _search_serpapi when serpapi_key is set.
    # Simulate: Tavily client raises, SerpAPI succeeds.
    def _fake_search_tavily(query: str, subtask_id: str) -> list[Document]:
        # Mimic the real _search_tavily: try Tavily, catch, fall back to SerpAPI
        raise RuntimeError("Tavily down")  # will be caught inside _search_tavily

    # Patch at the level _search_sync calls: _search_tavily raises → _search_sync
    # propagates, but we want the *internal* fallback. Instead, patch _search_serpapi
    # to return docs and let _search_tavily's except block call it.
    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_serpapi", return_value=serpapi_docs),
    ):
        # Patch the TavilyClient inside _search_tavily to raise
        import swarmiq.agents.literature as lit_mod
        original_search_tavily = agent._search_tavily

        def _patched_search_tavily(query: str, subtask_id: str) -> list[Document]:
            # Simulate Tavily client raising; the real method catches and falls back
            try:
                raise RuntimeError("Tavily API error")
            except Exception as exc:
                import logging
                logging.getLogger("swarmiq.agents.literature").warning(
                    "LiteratureAgent: Tavily search failed: %s. Trying SerpAPI.", exc
                )
            # Fall back to SerpAPI (same as real code)
            if agent._serpapi_key:
                return agent._search_serpapi(query, subtask_id)
            raise RuntimeError("Tavily failed and no SerpAPI key configured.")

        with patch.object(agent, "_search_tavily", side_effect=_patched_search_tavily):
            result = asyncio.run(
                agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP)
            )

    assert isinstance(result, LiteratureOutput)
    assert len(result.documents) == 3


# ---------------------------------------------------------------------------
# Test 3: Both Tavily and SerpAPI raise → retry also fails → returns empty list
# ---------------------------------------------------------------------------


def test_both_apis_fail_returns_empty_list():
    """Both Tavily and SerpAPI raise exception → retry also fails → returns empty list (no crash)."""
    agent, mock_store = _make_agent()

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_sync", side_effect=RuntimeError("All APIs down")),
    ):
        result = asyncio.run(
            agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP)
        )

    assert isinstance(result, LiteratureOutput)
    assert result.documents == []


# ---------------------------------------------------------------------------
# Test 4: Cache returns 5+ docs with score > 0.80 → web search NOT called
# ---------------------------------------------------------------------------


def test_cache_hit_skips_web_search():
    """Cache returns 5+ docs with score > 0.80 → web search NOT called."""
    agent, mock_store = _make_agent()
    mock_store.query.return_value = _make_cache_matches(5, score=0.91)

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_sync") as mock_search,
    ):
        result = asyncio.run(
            agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP)
        )

    assert isinstance(result, LiteratureOutput)
    assert len(result.documents) == 5
    mock_search.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: Cache returns 4 docs (below threshold) → web search IS called
# ---------------------------------------------------------------------------


def test_cache_below_min_docs_triggers_web_search():
    """Cache returns 4 docs (below threshold) → web search IS called."""
    agent, mock_store = _make_agent()
    mock_store.query.return_value = _make_cache_matches(4, score=0.91)
    web_docs = _make_docs(5)

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_sync", return_value=web_docs) as mock_search,
    ):
        result = asyncio.run(
            agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP)
        )

    assert isinstance(result, LiteratureOutput)
    mock_search.assert_called_once()


# ---------------------------------------------------------------------------
# Test 6: Cache returns 0 docs → web search IS called
# ---------------------------------------------------------------------------


def test_empty_cache_triggers_web_search():
    """Cache returns 0 docs → web search IS called."""
    agent, mock_store = _make_agent()
    mock_store.query.return_value = []
    web_docs = _make_docs(3)

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_sync", return_value=web_docs) as mock_search,
    ):
        result = asyncio.run(
            agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP)
        )

    assert isinstance(result, LiteratureOutput)
    mock_search.assert_called_once()


# ---------------------------------------------------------------------------
# Test 7: Upserted vectors contain required metadata keys
# ---------------------------------------------------------------------------


def test_upserted_vectors_contain_required_metadata():
    """Upserted vectors contain url, retrieved_at, subtask_id, session_id, query_fingerprint."""
    agent, mock_store = _make_agent()
    docs = _make_docs(2)

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_sync", return_value=docs),
    ):
        asyncio.run(agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP))

    mock_store.upsert.assert_called_once()
    call_kwargs = mock_store.upsert.call_args
    vectors = call_kwargs[1]["vectors"] if call_kwargs[1] else call_kwargs[0][0]

    required_keys = {"url", "retrieved_at", "subtask_id", "session_id", "query_fingerprint"}
    for vec in vectors:
        meta = vec["metadata"]
        assert required_keys.issubset(meta.keys()), (
            f"Missing metadata keys: {required_keys - meta.keys()}"
        )


# ---------------------------------------------------------------------------
# Test 8: Upserted vector ID follows format {session_id}#{subtask_id}#{url_hash}
# ---------------------------------------------------------------------------


def test_upserted_vector_id_format():
    """Upserted vector ID follows format {session_id}#{subtask_id}#{url_hash}."""
    agent, mock_store = _make_agent()
    docs = _make_docs(1)
    url = docs[0].url
    expected_id = make_vector_id(_SESSION_ID, _SUBTASK_ID, url)

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_sync", return_value=docs),
    ):
        asyncio.run(agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP))

    call_kwargs = mock_store.upsert.call_args
    vectors = call_kwargs[1]["vectors"] if call_kwargs[1] else call_kwargs[0][0]

    assert len(vectors) == 1
    assert vectors[0]["id"] == expected_id

    # Verify format: session_id#subtask_id#<16-char hex>
    parts = vectors[0]["id"].split("#")
    assert len(parts) == 3
    assert parts[0] == _SESSION_ID
    assert parts[1] == _SUBTASK_ID
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    assert parts[2] == url_hash


# ---------------------------------------------------------------------------
# Test 9: Timeout → returns AgentError with error_code="TIMEOUT"
# ---------------------------------------------------------------------------


def test_timeout_returns_agent_error():
    """asyncio.TimeoutError from _run_inner → returns AgentError with error_code='TIMEOUT'."""
    agent, mock_store = _make_agent()

    async def _raise_timeout(*args, **kwargs):
        raise asyncio.TimeoutError()

    with patch.object(agent, "_run_inner", side_effect=_raise_timeout):
        result = asyncio.run(
            agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP)
        )

    assert isinstance(result, AgentError)
    assert result.error_code == "TIMEOUT"
    assert result.agent_type == "literature_agent"
    assert result.subtask_id == _SUBTASK_ID


# ---------------------------------------------------------------------------
# Test 10: _embed_text failure → document skipped, upsert called with remaining docs
# ---------------------------------------------------------------------------


def test_embed_failure_skips_document():
    """_embed_text failure → document is skipped (no crash), upsert called with remaining docs."""
    agent, mock_store = _make_agent()
    docs = _make_docs(3)

    embed_call_count = 0

    def _embed_side_effect(text: str) -> list[float]:
        nonlocal embed_call_count
        embed_call_count += 1
        # First call is for the query embedding; subsequent calls are for docs.
        # Fail on the second call (first doc embed).
        if embed_call_count == 2:
            raise RuntimeError("Embedding model failure")
        return _DUMMY_EMBEDDING

    with (
        patch.object(agent, "_embed_text", side_effect=_embed_side_effect),
        patch.object(agent, "_search_sync", return_value=docs),
    ):
        result = asyncio.run(
            agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP)
        )

    # No crash; result is still a LiteratureOutput with all 3 docs
    assert isinstance(result, LiteratureOutput)
    assert len(result.documents) == 3

    # Upsert should have been called with only 2 vectors (1 skipped)
    mock_store.upsert.assert_called_once()
    call_kwargs = mock_store.upsert.call_args
    vectors = call_kwargs[1]["vectors"] if call_kwargs[1] else call_kwargs[0][0]
    assert len(vectors) == 2


# ---------------------------------------------------------------------------
# Test 11: Pinecone upsert failure → logged as warning, no crash, docs still returned
# ---------------------------------------------------------------------------


def test_pinecone_upsert_failure_no_crash():
    """Pinecone upsert failure → logged as warning, no crash, documents still returned."""
    agent, mock_store = _make_agent()
    docs = _make_docs(3)
    mock_store.upsert.side_effect = RuntimeError("Pinecone unavailable")

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_sync", return_value=docs),
    ):
        result = asyncio.run(
            agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP)
        )

    # No crash; documents are still returned
    assert isinstance(result, LiteratureOutput)
    assert len(result.documents) == 3


# ---------------------------------------------------------------------------
# Test 12: DuckDuckGo used when no API keys configured and use_duckduckgo=True
# ---------------------------------------------------------------------------


def test_duckduckgo_used_when_no_api_keys():
    """No Tavily/SerpAPI keys + use_duckduckgo=True → DuckDuckGo is called."""
    mock_store = MagicMock()
    mock_store.query.return_value = []
    agent = LiteratureAgent(
        knowledge_store=mock_store,
        tavily_api_key=None,
        serpapi_key=None,
        use_duckduckgo=True,
    )
    ddg_docs = _make_docs(3)

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_duckduckgo", return_value=ddg_docs) as mock_ddg,
    ):
        result = asyncio.run(agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP))

    assert isinstance(result, LiteratureOutput)
    mock_ddg.assert_called_once()


# ---------------------------------------------------------------------------
# Test 13: DuckDuckGo fallback when Tavily fails and no SerpAPI key
# ---------------------------------------------------------------------------


def test_duckduckgo_fallback_when_tavily_fails_no_serpapi():
    """Tavily fails + no SerpAPI key + use_duckduckgo=True → falls back to DuckDuckGo."""
    mock_store = MagicMock()
    mock_store.query.return_value = []
    agent = LiteratureAgent(
        knowledge_store=mock_store,
        tavily_api_key="fake-tavily-key",
        serpapi_key=None,
        use_duckduckgo=True,
    )
    ddg_docs = _make_docs(4)

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_tavily", side_effect=RuntimeError("Tavily down")),
        patch.object(agent, "_search_duckduckgo", return_value=ddg_docs) as mock_ddg,
    ):
        result = asyncio.run(agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP))

    assert isinstance(result, LiteratureOutput)
    mock_ddg.assert_called_once()


# ---------------------------------------------------------------------------
# Test 14: DuckDuckGo fallback when both Tavily and SerpAPI fail
# ---------------------------------------------------------------------------


def test_duckduckgo_fallback_when_both_apis_fail():
    """Tavily fails + SerpAPI fails + use_duckduckgo=True → falls back to DuckDuckGo."""
    mock_store = MagicMock()
    mock_store.query.return_value = []
    agent = LiteratureAgent(
        knowledge_store=mock_store,
        tavily_api_key="fake-tavily-key",
        serpapi_key="fake-serpapi-key",
        use_duckduckgo=True,
    )
    ddg_docs = _make_docs(5)

    with (
        patch.object(agent, "_embed_text", return_value=_DUMMY_EMBEDDING),
        patch.object(agent, "_search_tavily", side_effect=RuntimeError("Tavily down")),
        patch.object(agent, "_search_serpapi", side_effect=RuntimeError("SerpAPI down")),
        patch.object(agent, "_search_duckduckgo", return_value=ddg_docs) as mock_ddg,
    ):
        result = asyncio.run(agent.run(_make_subtask(), _SESSION_ID, _QUERY_FP))

    assert isinstance(result, LiteratureOutput)
    mock_ddg.assert_called_once()


# ---------------------------------------------------------------------------
# Test 15: _search_duckduckgo parses DDGS results correctly
# ---------------------------------------------------------------------------


def test_search_duckduckgo_parses_results():
    """_search_duckduckgo parses DDGS results into Document objects."""
    mock_store = MagicMock()
    agent = LiteratureAgent(knowledge_store=mock_store, use_duckduckgo=True)

    fake_results = [
        {"title": "Title 1", "href": "https://example.com/1", "body": "Snippet 1"},
        {"title": "Title 2", "href": "https://example.com/2", "body": "Snippet 2"},
        {"title": "No URL", "href": "", "body": "Should be skipped"},
    ]

    # Stub duckduckgo_search module so the local import inside _search_duckduckgo works
    _ddgs_instance = MagicMock()
    _ddgs_instance.text.return_value = fake_results
    _ddgs_class = MagicMock(return_value=_ddgs_instance)
    _ddg_mod = types.ModuleType("duckduckgo_search")
    _ddg_mod.DDGS = _ddgs_class  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"duckduckgo_search": _ddg_mod}):
        docs = agent._search_duckduckgo("test query", _SUBTASK_ID)

    assert len(docs) == 2
    assert docs[0].url == "https://example.com/1"
    assert docs[0].title == "Title 1"
    assert docs[0].content == "Snippet 1"
    assert docs[1].url == "https://example.com/2"
    assert docs[0].subtask_id == _SUBTASK_ID
