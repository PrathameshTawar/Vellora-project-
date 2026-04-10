"""Cross-encoder re-ranker for SwarmIQ v2."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Re-ranks document chunks using a cross-encoder model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the CrossEncoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded on first call to rerank()

    def _get_model(self):
        """Lazy-load the CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            logger.info("Loading CrossEncoder model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Score each (query, chunk) pair and return top_k chunks by score.

        Parameters
        ----------
        query:
            The search query string.
        chunks:
            List of dicts, each with at least a ``"content"`` key.
        top_k:
            Maximum number of chunks to return.

        Returns
        -------
        list[dict]
            Chunks sorted by relevance score descending, at most top_k items.
            If chunks is empty or has fewer than top_k items, all chunks are returned.
        """
        if not chunks:
            return []

        model = self._get_model()

        pairs = [(query, chunk.get("content", "")) for chunk in chunks]
        scores = model.predict(pairs)

        scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        return [chunk for _, chunk in scored[:top_k]]
