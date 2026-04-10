"""
Literature_Agent for SwarmIQ v2.

Executes web search for a single subtask, embeds documents, and upserts to Pinecone.
Cache-first: queries KnowledgeStore before hitting external APIs.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Union
from urllib.parse import urlparse

from swarmiq.agents.gap_detector import GapDetector
from swarmiq.config import EMBED_MODEL, MAX_RESEARCH_ITERATIONS
from swarmiq.core.credibility import get_domain_trust
from swarmiq.core.knowledge_store import KnowledgeStore, make_vector_id
from swarmiq.core.models import AgentError, Document, SubTask
from swarmiq.memory.reranker import Reranker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class LiteratureOutput:
    subtask_id: str
    documents: list[Document] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_SCORE_THRESHOLD = 0.80
_CACHE_MIN_DOCS = 5
_RETRY_BACKOFF_SECONDS = 2.0
_SUBTASK_TIMEOUT_SECONDS = 30.0
_MIN_DOCS_REQUIRED = 5
_CONTENT_PREVIEW_LENGTH = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_domain(url: str) -> str:
    """Extract hostname from a URL."""
    try:
        parsed = urlparse(url)
        return parsed.hostname or url
    except Exception:
        return url


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# LiteratureAgent
# ---------------------------------------------------------------------------


class LiteratureAgent:
    """Retrieves documents for a subtask via cache-first lookup then web search.

    Parameters
    ----------
    knowledge_store:
        Shared Pinecone wrapper for cache lookup and upsert.
    tavily_api_key:
        API key for Tavily (primary search). If None, Tavily is skipped.
    serpapi_key:
        API key for SerpAPI (fallback search). If None, SerpAPI is skipped.
    embedding_model_name:
        sentence-transformers model name for document embedding.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        tavily_api_key: str | None = None,
        serpapi_key: str | None = None,
        embedding_model_name: str = EMBED_MODEL,
        gap_detector: GapDetector | None = None,
        use_duckduckgo: bool = True,
    ) -> None:
        self._store = knowledge_store
        self._tavily_api_key = tavily_api_key
        self._serpapi_key = serpapi_key
        self._embedding_model_name = embedding_model_name
        self._model = None  # lazy-loaded
        self._reranker: Reranker | None = None  # lazy-loaded
        self._gap_detector = gap_detector
        self._use_duckduckgo = use_duckduckgo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        subtask: SubTask,
        session_id: str,
        query_fingerprint: str,
    ) -> Union[LiteratureOutput, AgentError]:
        """Execute the literature retrieval pipeline for *subtask*.

        Steps:
        1. Embed the subtask query and check KnowledgeStore cache.
        2. If ≥5 cached docs with score > 0.80, return them directly.
        3. Otherwise call Tavily (primary) or SerpAPI (fallback).
        4. Retry once with 2-second backoff on API error.
        5. Embed each document and upsert to KnowledgeStore.
        6. Return LiteratureOutput.

        Wrapped in a 30-second asyncio timeout.
        """
        try:
            return await asyncio.wait_for(
                self._run_inner(subtask, session_id, query_fingerprint),
                timeout=_SUBTASK_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "LiteratureAgent timed out after %ss for subtask %s",
                _SUBTASK_TIMEOUT_SECONDS,
                subtask.subtask_id,
            )
            return AgentError(
                agent_type="literature_agent",
                subtask_id=subtask.subtask_id,
                error_code="TIMEOUT",
                message=f"Literature retrieval exceeded {_SUBTASK_TIMEOUT_SECONDS}s timeout.",
                timestamp=_now_iso(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LiteratureAgent unhandled exception for subtask %s: %s",
                subtask.subtask_id,
                exc,
            )
            return AgentError(
                agent_type="literature_agent",
                subtask_id=subtask.subtask_id,
                error_code="UNHANDLED_EXCEPTION",
                message=str(exc),
                timestamp=_now_iso(),
            )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _run_inner(
        self,
        subtask: SubTask,
        session_id: str,
        query_fingerprint: str,
    ) -> LiteratureOutput:
        query = " ".join(subtask.search_keywords)

        # 1. Embed query
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, self._embed_query, query
        )

        # 2. Cache-first lookup
        cached_docs = self._query_cache(query_embedding, subtask.subtask_id)
        if len(cached_docs) >= _CACHE_MIN_DOCS:
            logger.info(
                "LiteratureAgent: cache hit (%d docs) for subtask %s — skipping web search.",
                len(cached_docs),
                subtask.subtask_id,
            )
            reranked = self._rerank_documents(query, cached_docs)
            return LiteratureOutput(subtask_id=subtask.subtask_id, documents=reranked)

        # 3. Web search with retry
        documents = await self._fetch_with_retry(query, subtask.subtask_id)

        # Re-rank web search results before upserting
        documents = self._rerank_documents(query, documents)

        # 4. Embed and upsert each document
        await self._embed_and_upsert(
            documents, subtask.subtask_id, session_id, query_fingerprint
        )

        # 5. Gap detection: if a GapDetector is configured, find unanswered questions
        #    and perform targeted follow-up searches.
        if self._gap_detector is not None:
            combined_content = " ".join(
                (doc.content or "")[:_CONTENT_PREVIEW_LENGTH] for doc in documents
            )
            gaps = await self._gap_detector.detect_gaps(
                subtask.search_keywords, combined_content
            )
            for gap in gaps[:MAX_RESEARCH_ITERATIONS]:
                logger.info(
                    "LiteratureAgent: gap detected for subtask %s — searching: %r",
                    subtask.subtask_id,
                    gap,
                )
                gap_docs = await self._fetch_with_retry(gap, subtask.subtask_id)
                if gap_docs:
                    gap_docs = self._rerank_documents(gap, gap_docs)
                    await self._embed_and_upsert(
                        gap_docs, subtask.subtask_id, session_id, query_fingerprint
                    )
                    documents.extend(gap_docs)

        return LiteratureOutput(subtask_id=subtask.subtask_id, documents=documents)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _query_cache(self, embedding: list[float], subtask_id: str) -> list[Document]:
        """Query KnowledgeStore and convert matches to Document objects."""
        try:
            matches = self._store.query(
                embedding=embedding,
                top_k=20,
            )
        except Exception as exc:
            logger.warning("LiteratureAgent: cache query failed: %s", exc)
            return []

        docs: list[Document] = []
        for match in matches:
            if match.get("score", 0.0) <= _CACHE_SCORE_THRESHOLD:
                continue
            meta = match.get("metadata", {})
            docs.append(
                Document(
                    url=meta.get("url", ""),
                    title=meta.get("title", ""),
                    content=meta.get("content_preview", ""),
                    retrieved_at=meta.get("retrieved_at", _now_iso()),
                    subtask_id=meta.get("subtask_id", subtask_id),
                )
            )
        return docs

    def _rerank_documents(self, query: str, docs: list[Document]) -> list[Document]:
        """Re-rank documents using the cross-encoder and return top-5.

        If fewer than 5 docs are provided, all are returned unchanged.
        """
        if not docs:
            return docs

        if self._reranker is None:
            from swarmiq.config import RERANKER_MODEL
            self._reranker = Reranker(model_name=RERANKER_MODEL)

        chunks = [
            {"content": doc.title + " " + doc.content, "_doc": doc}
            for doc in docs
        ]
        reranked_chunks = self._reranker.rerank(query, chunks, top_k=5)
        return [chunk["_doc"] for chunk in reranked_chunks]

    # ------------------------------------------------------------------
    # Web search helpers
    # ------------------------------------------------------------------

    async def _fetch_with_retry(
        self, query: str, subtask_id: str
    ) -> list[Document]:
        """Try Tavily then SerpAPI; retry once on failure with backoff."""
        for attempt in range(2):
            try:
                docs = await asyncio.get_event_loop().run_in_executor(
                    None, self._search_sync, query, subtask_id
                )
                return docs
            except Exception as exc:
                if attempt == 0:
                    logger.warning(
                        "LiteratureAgent: search attempt 1 failed for subtask %s: %s. "
                        "Retrying in %ss…",
                        subtask_id,
                        exc,
                        _RETRY_BACKOFF_SECONDS,
                    )
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
                else:
                    logger.error(
                        "LiteratureAgent: search attempt 2 failed for subtask %s: %s. "
                        "Giving up.",
                        subtask_id,
                        exc,
                    )
                    return []
        return []

    def _search_sync(self, query: str, subtask_id: str) -> list[Document]:
        """Synchronous search with priority chain: Tavily → SerpAPI → DuckDuckGo."""
        if self._tavily_api_key:
            try:
                docs = self._search_tavily(query, subtask_id)
                if docs:
                    return docs
            except Exception as exc:
                logger.warning(
                    "LiteratureAgent: Tavily failed for subtask %s: %s. Trying SerpAPI.",
                    subtask_id,
                    exc,
                )
            if self._serpapi_key:
                try:
                    docs = self._search_serpapi(query, subtask_id)
                    if docs:
                        return docs
                except Exception as exc:
                    logger.warning(
                        "LiteratureAgent: SerpAPI failed for subtask %s: %s. Trying DuckDuckGo.",
                        subtask_id,
                        exc,
                    )
            if self._use_duckduckgo:
                return self._search_duckduckgo(query, subtask_id)
            raise RuntimeError("Tavily and SerpAPI failed and DuckDuckGo is disabled.")

        if self._serpapi_key:
            try:
                docs = self._search_serpapi(query, subtask_id)
                if docs:
                    return docs
            except Exception as exc:
                logger.warning(
                    "LiteratureAgent: SerpAPI failed for subtask %s: %s. Trying DuckDuckGo.",
                    subtask_id,
                    exc,
                )
            if self._use_duckduckgo:
                return self._search_duckduckgo(query, subtask_id)
            raise RuntimeError("SerpAPI failed and DuckDuckGo is disabled.")

        if self._use_duckduckgo:
            return self._search_duckduckgo(query, subtask_id)

        logger.warning(
            "LiteratureAgent: no search API keys configured for subtask %s.", subtask_id
        )
        return []

    def _search_tavily(self, query: str, subtask_id: str) -> list[Document]:
        """Search via Tavily; raise on failure so _search_sync handles fallback."""
        from tavily import TavilyClient  # type: ignore[import]

        logger.info("LiteratureAgent: using Tavily for subtask %s", subtask_id)
        client = TavilyClient(api_key=self._tavily_api_key)
        results = client.search(query, max_results=10)
        return self._parse_tavily_results(results, subtask_id)

    def _search_serpapi(self, query: str, subtask_id: str) -> list[Document]:
        """Search via SerpAPI."""
        from serpapi import GoogleSearch  # type: ignore[import]

        logger.info("LiteratureAgent: using SerpAPI for subtask %s", subtask_id)
        search = GoogleSearch({"q": query, "api_key": self._serpapi_key})
        results = search.get_dict()
        return self._parse_serpapi_results(results, subtask_id)

    def _search_duckduckgo(self, query: str, subtask_id: str) -> list[Document]:
        """Search via DuckDuckGo (no API key required)."""
        from duckduckgo_search import DDGS  # type: ignore[import]

        logger.info("LiteratureAgent: using DuckDuckGo for subtask %s", subtask_id)
        results = DDGS().text(query, max_results=10)
        docs: list[Document] = []
        retrieved_at = _now_iso()
        for item in (results or []):
            url = item.get("href", "")
            if not url:
                continue
            docs.append(
                Document(
                    url=url,
                    title=item.get("title", ""),
                    content=item.get("body", ""),
                    retrieved_at=retrieved_at,
                    subtask_id=subtask_id,
                )
            )
        return docs

    # ------------------------------------------------------------------
    # Result parsers
    # ------------------------------------------------------------------

    def _parse_tavily_results(
        self, results: dict, subtask_id: str
    ) -> list[Document]:
        docs: list[Document] = []
        retrieved_at = _now_iso()
        for item in results.get("results", []):
            url = item.get("url", "")
            if not url:
                continue
            docs.append(
                Document(
                    url=url,
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    retrieved_at=retrieved_at,
                    subtask_id=subtask_id,
                )
            )
        return docs

    def _parse_serpapi_results(
        self, results: dict, subtask_id: str
    ) -> list[Document]:
        docs: list[Document] = []
        retrieved_at = _now_iso()
        for item in results.get("organic_results", []):
            url = item.get("link", "")
            if not url:
                continue
            docs.append(
                Document(
                    url=url,
                    title=item.get("title", ""),
                    content=item.get("snippet", ""),
                    retrieved_at=retrieved_at,
                    subtask_id=subtask_id,
                )
            )
        return docs

    # ------------------------------------------------------------------
    # Embedding and upsert
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            if "nomic" in self._embedding_model_name:
                self._model = SentenceTransformer(
                    self._embedding_model_name, trust_remote_code=True
                )
            else:
                self._model = SentenceTransformer(self._embedding_model_name)
        return self._model

    def _embed_text(self, text: str) -> list[float]:
        """Embed a single document text string and return as a list of floats.

        Prepends ``"search_document: "`` prefix when using a nomic model.
        """
        model = self._get_model()
        if "nomic" in self._embedding_model_name:
            text = "search_document: " + text
        vector = model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def _embed_query(self, text: str) -> list[float]:
        """Embed a query string and return as a list of floats.

        Prepends ``"search_query: "`` prefix when using a nomic model.
        """
        model = self._get_model()
        if "nomic" in self._embedding_model_name:
            text = "search_query: " + text
        vector = model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    async def _embed_and_upsert(
        self,
        documents: list[Document],
        subtask_id: str,
        session_id: str,
        query_fingerprint: str,
    ) -> None:
        """Embed each document and upsert to KnowledgeStore with full metadata."""
        vectors = []
        for doc in documents:
            try:
                embedding = await asyncio.get_event_loop().run_in_executor(
                    None, self._embed_text, doc.content or doc.title
                )
            except Exception as exc:
                logger.warning(
                    "LiteratureAgent: failed to embed document %s: %s", doc.url, exc
                )
                continue

            domain = _extract_domain(doc.url)
            domain_trust = get_domain_trust(domain)
            content_preview = (doc.content or "")[:_CONTENT_PREVIEW_LENGTH]
            vector_id = make_vector_id(session_id, subtask_id, doc.url)

            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "url": doc.url,
                        "title": doc.title,
                        "retrieved_at": doc.retrieved_at,
                        "subtask_id": subtask_id,
                        "session_id": session_id,
                        "query_fingerprint": query_fingerprint,
                        "domain": domain,
                        "domain_trust": domain_trust,
                        "content_preview": content_preview,
                    },
                }
            )

        if not vectors:
            return

        try:
            self._store.upsert(vectors=vectors, namespace=session_id)
        except Exception as exc:
            logger.warning(
                "LiteratureAgent: Pinecone upsert failed for subtask %s: %s",
                subtask_id,
                exc,
            )
