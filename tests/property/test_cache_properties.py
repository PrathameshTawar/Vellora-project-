"""
Property-based tests for LiteratureAgent cache behaviour (Property 19).

Feature: swarmiq-v2
Validates: Requirements 9.2
"""
from __future__ import annotations

import asyncio
import sys
import types
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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

# CrossEncoder mock: predict() returns a list of 0.5 scores for any input
_ce_instance = MagicMock()
_ce_instance.predict.side_effect = lambda pairs: [0.5] * len(pairs)
sys.modules["sentence_transformers"].CrossEncoder = MagicMock(return_value=_ce_instance)  # type: ignore[attr-defined]

sys.modules["pinecone"].Pinecone = MagicMock  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

from swarmiq.agents.literature import LiteratureAgent, LiteratureOutput  # noqa: E402
from swarmiq.core.models import SubTask  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

subtask_id_st = st.uuids().map(str)
session_id_st = st.uuids().map(str)
# Number of cached docs: always >= 5 to trigger cache-hit path
cached_doc_count_st = st.integers(min_value=5, max_value=20)


def _make_cache_match(subtask_id: str, index: int, score: float = 0.90) -> dict:
    """Build a fake KnowledgeStore query match with score > 0.80."""
    return {
        "id": f"vec-{index}",
        "score": score,
        "metadata": {
            "url": f"https://cached.example.com/doc{index}",
            "title": f"Cached Document {index}",
            "content_preview": "Cached content preview.",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "subtask_id": subtask_id,
        },
    }


# ---------------------------------------------------------------------------
# Property 19: Cache hit prevents web search re-fetch
# ---------------------------------------------------------------------------


class TestCacheHitPreventsWebSearch:
    """
    # Feature: swarmiq-v2, Property 19: Cache hit prevents web search re-fetch —
    # for any subtask where KnowledgeStore returns ≥5 documents with score > 0.80,
    # the LiteratureAgent must NOT invoke the web search API.
    """

    @given(subtask_id_st, session_id_st, cached_doc_count_st)
    @settings(max_examples=100)
    def test_cache_hit_skips_web_search(
        self, subtask_id: str, session_id: str, n_cached: int
    ):
        # Feature: swarmiq-v2, Property 19: Cache hit prevents web search re-fetch
        subtask = SubTask(
            subtask_id=subtask_id,
            type="literature_review",
            description="Cache test subtask",
            search_keywords=["cache", "test"],
        )

        # Build cache matches: all with score > 0.80
        cache_matches = [
            _make_cache_match(subtask_id, i, score=0.90)
            for i in range(n_cached)
        ]

        mock_store = MagicMock()
        mock_store.query.return_value = cache_matches
        mock_store.upsert = MagicMock()

        agent = LiteratureAgent(
            knowledge_store=mock_store,
            tavily_api_key="fake-key",
        )

        search_called = [False]

        def _fake_search_sync(query, subtask_id):
            search_called[0] = True
            return []

        with patch.object(agent, "_search_sync", side_effect=_fake_search_sync):
            with patch.object(agent, "_embed_text", return_value=[0.1] * 384):
                output = asyncio.run(
                    agent.run(subtask, session_id, "fp-cache-test")
                )

        assert isinstance(output, LiteratureOutput)
        assert not search_called[0], (
            "_search_sync was called despite a cache hit with "
            f"{n_cached} documents (score > 0.80)"
        )
        assert len(output.documents) >= 5
