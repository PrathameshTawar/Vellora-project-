"""
Pinecone client wrapper for SwarmIQ v2.

Provides a KnowledgeStore class that wraps the Pinecone index with
session-namespaced upsert, similarity query, delete-by-filter, and
fetch-by-ID operations.
"""
from __future__ import annotations

import hashlib
from typing import Any

try:
    from pinecone import Pinecone  # type: ignore[import]
except ImportError:
    Pinecone = None  # type: ignore[assignment,misc]


def make_vector_id(session_id: str, subtask_id: str, url: str) -> str:
    """Create a deterministic vector ID from session, subtask, and URL.

    Format: ``{session_id}#{subtask_id}#{url_hash}``
    where ``url_hash`` is the first 16 hex characters of the SHA-256 of the URL.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{session_id}#{subtask_id}#{url_hash}"


class KnowledgeStore:
    """Thin wrapper around a Pinecone index.

    Parameters
    ----------
    api_key:
        Pinecone API key.  If ``None`` or empty, every method raises
        ``ValueError`` with a clear message.
    index_name:
        Name of the Pinecone index to use.
    environment:
        Deprecated Pinecone environment string (ignored for serverless indexes,
        kept for backward compatibility).
    score_threshold:
        Minimum similarity score for ``query`` results.  Defaults to 0.80.
    """

    SCORE_THRESHOLD: float = 0.80

    def __init__(
        self,
        api_key: str | None,
        index_name: str,
        environment: str | None = None,
        score_threshold: float = SCORE_THRESHOLD,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Pinecone API key is not configured. "
                "Set the PINECONE_API_KEY environment variable."
            )

        self._index_name = index_name
        self._score_threshold = score_threshold

        if Pinecone is None:
            raise ImportError(
                "The 'pinecone' package is required. "
                "Install it with: pip install pinecone"
            )

        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, vectors: list[dict[str, Any]], namespace: str = "") -> None:
        """Upsert vectors into the index.

        Parameters
        ----------
        vectors:
            List of vector dicts, each with keys ``id``, ``values``, and
            optionally ``metadata``.
        namespace:
            Pinecone namespace (typically the ``session_id``).
        """
        if not vectors:
            return
        self._index.upsert(vectors=vectors, namespace=namespace)

    def query(
        self,
        embedding: list[float],
        top_k: int = 20,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        """Query the index for similar vectors.

        Parameters
        ----------
        embedding:
            Query embedding vector.
        top_k:
            Maximum number of results to return before score filtering.
        filter:
            Optional Pinecone metadata filter dict.
        namespace:
            Pinecone namespace to search within.

        Returns
        -------
        list[dict]
            Matched vectors with ``score >= score_threshold``.  Each dict
            contains ``id``, ``score``, and ``metadata`` keys.
        """
        kwargs: dict[str, Any] = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace,
        }
        if filter:
            kwargs["filter"] = filter

        response = self._index.query(**kwargs)
        matches = response.get("matches", []) if isinstance(response, dict) else (response.matches or [])

        results: list[dict[str, Any]] = []
        for match in matches:
            if isinstance(match, dict):
                score = match.get("score", 0.0)
                if score >= self._score_threshold:
                    results.append({
                        "id": match.get("id"),
                        "score": score,
                        "metadata": match.get("metadata", {}),
                    })
            else:
                # ScoredVector object
                score = getattr(match, "score", 0.0) or 0.0
                if score >= self._score_threshold:
                    results.append({
                        "id": match.id,
                        "score": score,
                        "metadata": dict(match.metadata) if match.metadata else {},
                    })

        return results

    def delete(self, filter: dict[str, Any], namespace: str = "") -> None:
        """Delete vectors matching a metadata filter.

        Parameters
        ----------
        filter:
            Pinecone metadata filter dict (e.g. ``{"session_id": "..."}``).
        namespace:
            Pinecone namespace to delete from.
        """
        self._index.delete(filter=filter, namespace=namespace)

    def get_by_id(self, vector_id: str, namespace: str = "") -> dict[str, Any] | None:
        """Fetch a single vector by its ID.

        Parameters
        ----------
        vector_id:
            The vector ID to fetch.
        namespace:
            Pinecone namespace to fetch from.

        Returns
        -------
        dict or None
            Dict with ``id``, ``values``, and ``metadata`` keys, or ``None``
            if the vector does not exist.
        """
        response = self._index.fetch(ids=[vector_id], namespace=namespace)

        vectors = response.get("vectors", {}) if isinstance(response, dict) else (response.vectors or {})

        if vector_id not in vectors:
            return None

        vec = vectors[vector_id]
        if isinstance(vec, dict):
            return {
                "id": vec.get("id", vector_id),
                "values": vec.get("values", []),
                "metadata": vec.get("metadata", {}),
            }
        # Vector object
        return {
            "id": vec.id,
            "values": list(vec.values) if vec.values else [],
            "metadata": dict(vec.metadata) if vec.metadata else {},
        }
