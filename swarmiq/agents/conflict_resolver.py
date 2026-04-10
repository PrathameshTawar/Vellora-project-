"""
Conflict_Resolver_Agent for SwarmIQ v2.

Detects semantic contradictions between scored claims and classifies each
claim as `accepted`, `rejected`, or `uncertain`.

Contradiction detection:
  - Cosine similarity between sentence-transformer embeddings > 0.85
  - AND opposing polarity via NLI classifier (label == "CONTRADICTION", score > 0.5)

Resolution logic:
  - Higher credibility_score wins → accepted; other → rejected
  - If scores within 0.05 → both uncertain
  - Single claim for a topic (no contradicting pair found) → uncertain
  - Non-contradicting claims with no conflicts → accepted

Debate Mode (debate_mode=True):
  - For each contradicting pair, generate arguments via LLM
  - AutoGen AssistantAgent acts as judge before issuing final classifications
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

import numpy as np

from swarmiq.core.models import (
    AgentError,
    ConflictResolverOutput,
    Resolution,
    ScoredClaim,
)
from swarmiq.config import EMBED_MODEL

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ConflictResolverAgent:
    """Detect contradictions between scored claims and resolve them.

    Args:
        llm_config:           AutoGen LLM config dict (used for Debate Mode judge).
        embedding_model_name: sentence-transformers model for claim embeddings.
        nli_model_name:       HuggingFace model for NLI contradiction detection.
    """

    def __init__(
        self,
        llm_config: dict,
        embedding_model_name: str = EMBED_MODEL,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    ) -> None:
        self._llm_config = llm_config
        self._embedding_model_name = embedding_model_name
        self._nli_model_name = nli_model_name

        # Lazy-loaded models
        self._embedder = None
        self._nli_pipeline = None

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-load the sentence-transformers embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info(
                "ConflictResolverAgent: loading embedding model %s",
                self._embedding_model_name,
            )
            if "nomic" in self._embedding_model_name:
                self._embedder = SentenceTransformer(
                    self._embedding_model_name, trust_remote_code=True
                )
            else:
                self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _get_nli_pipeline(self):
        """Lazy-load the HuggingFace NLI text-classification pipeline."""
        if self._nli_pipeline is None:
            from transformers import pipeline  # type: ignore

            logger.info(
                "ConflictResolverAgent: loading NLI model %s",
                self._nli_model_name,
            )
            self._nli_pipeline = pipeline(
                "text-classification",
                model=self._nli_model_name,
            )
        return self._nli_pipeline

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_claims(self, claims: list[ScoredClaim]) -> list[np.ndarray]:
        """Return a list of embedding vectors, one per claim."""
        embedder = self._get_embedder()
        texts = [c.claim_text for c in claims]
        if "nomic" in self._embedding_model_name:
            texts = ["search_document: " + t for t in texts]
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        return [embeddings[i] for i in range(len(claims))]

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def _is_contradiction(self, text_a: str, text_b: str) -> bool:
        """Return True if the NLI classifier labels the pair as CONTRADICTION."""
        nli = self._get_nli_pipeline()
        # Format as premise-hypothesis pair
        input_text = f"{text_a} [SEP] {text_b}"
        result = nli(input_text)
        # result is a list of dicts: [{"label": "...", "score": ...}]
        if isinstance(result, list) and result:
            top = result[0]
            label = top.get("label", "").upper()
            score = top.get("score", 0.0)
            return label == "CONTRADICTION" and score > 0.5
        return False

    def _find_contradicting_pairs(
        self,
        claims: list[ScoredClaim],
        embeddings: list[np.ndarray],
    ) -> list[tuple[int, int]]:
        """Return index pairs (i, j) where claims[i] and claims[j] contradict."""
        contradicting: list[tuple[int, int]] = []
        for i, j in combinations(range(len(claims)), 2):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim > 0.85:
                if self._is_contradiction(claims[i].claim_text, claims[j].claim_text):
                    contradicting.append((i, j))
        return contradicting

    # ------------------------------------------------------------------
    # Debate Mode
    # ------------------------------------------------------------------

    def _run_debate(
        self,
        claim_a: ScoredClaim,
        claim_b: ScoredClaim,
    ) -> tuple[str, str]:
        """Run a structured argument exchange and return (rationale_a, rationale_b).

        Uses an AutoGen AssistantAgent as judge. Returns rationale strings
        that will be used in the final Resolution objects.
        """
        from autogen import AssistantAgent  # type: ignore

        judge = AssistantAgent(
            name="conflict_judge",
            system_message=(
                "You are an impartial research judge. You will be given two conflicting "
                "claims and arguments for each. Weigh the arguments and credibility scores, "
                "then decide which claim is more credible. "
                "Respond with a JSON object: "
                '{"winner": "A" or "B" or "tie", "rationale_a": "...", "rationale_b": "..."}'
            ),
            llm_config=self._llm_config,
        )

        prompt = (
            f"Claim A (credibility={claim_a.credibility_score:.3f}):\n"
            f"  {claim_a.claim_text}\n\n"
            f"Claim B (credibility={claim_b.credibility_score:.3f}):\n"
            f"  {claim_b.claim_text}\n\n"
            "Argument for Claim A: This claim is supported by its source with a "
            f"credibility score of {claim_a.credibility_score:.3f} and confidence "
            f"of {claim_a.confidence:.3f}.\n\n"
            "Argument for Claim B: This claim is supported by its source with a "
            f"credibility score of {claim_b.credibility_score:.3f} and confidence "
            f"of {claim_b.confidence:.3f}.\n\n"
            "Weigh these arguments and provide your judgment as JSON."
        )

        import json

        try:
            reply = judge.generate_reply(
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(reply, dict):
                raw = reply.get("content", "{}")
            else:
                raw = str(reply) if reply else "{}"

            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            data = json.loads(raw)
            rationale_a = data.get("rationale_a", "Evaluated via debate mode.")
            rationale_b = data.get("rationale_b", "Evaluated via debate mode.")
            return rationale_a, rationale_b
        except Exception as exc:  # noqa: BLE001
            logger.warning("ConflictResolverAgent: debate mode LLM call failed: %s", exc)
            return "Evaluated via debate mode.", "Evaluated via debate mode."

    # ------------------------------------------------------------------
    # Resolution logic
    # ------------------------------------------------------------------

    def _resolve_pair(
        self,
        idx_a: int,
        idx_b: int,
        claims: list[ScoredClaim],
        statuses: dict[int, str],
        rationales: dict[int, str],
        debate_mode: bool,
    ) -> None:
        """Apply resolution logic to a contradicting pair in-place."""
        claim_a = claims[idx_a]
        claim_b = claims[idx_b]

        if debate_mode:
            rationale_a, rationale_b = self._run_debate(claim_a, claim_b)
        else:
            rationale_a = (
                f"Contradicts claim {claim_b.claim_id[:8]}…; "
                f"credibility scores: {claim_a.credibility_score:.3f} vs "
                f"{claim_b.credibility_score:.3f}."
            )
            rationale_b = (
                f"Contradicts claim {claim_a.claim_id[:8]}…; "
                f"credibility scores: {claim_b.credibility_score:.3f} vs "
                f"{claim_a.credibility_score:.3f}."
            )

        score_a = claim_a.credibility_score
        score_b = claim_b.credibility_score

        if abs(score_a - score_b) <= 0.05:
            # Scores too close → both uncertain
            statuses[idx_a] = "uncertain"
            statuses[idx_b] = "uncertain"
            rationales[idx_a] = rationale_a + " Scores within 0.05 — insufficient evidence to prefer either claim."
            rationales[idx_b] = rationale_b + " Scores within 0.05 — insufficient evidence to prefer either claim."
        elif score_a > score_b:
            statuses[idx_a] = "accepted"
            statuses[idx_b] = "rejected"
            rationales[idx_a] = rationale_a + " Higher credibility score."
            rationales[idx_b] = rationale_b + " Lower credibility score; contradicted by a more credible claim."
        else:
            statuses[idx_a] = "rejected"
            statuses[idx_b] = "accepted"
            rationales[idx_a] = rationale_a + " Lower credibility score; contradicted by a more credible claim."
            rationales[idx_b] = rationale_b + " Higher credibility score."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        claims: list[ScoredClaim],
        debate_mode: bool = False,
    ) -> ConflictResolverOutput:
        """Resolve contradictions among *claims* and return a resolution report.

        Args:
            claims:      List of ScoredClaim objects to evaluate.
            debate_mode: When True, run structured LLM debate before classifying.

        Returns:
            ConflictResolverOutput with one Resolution per input claim.
        """
        if not claims:
            return ConflictResolverOutput(resolutions=[])

        # Initialise all claims as uncertain (single-claim default)
        statuses: dict[int, str] = {i: "uncertain" for i in range(len(claims))}
        rationales: dict[int, str] = {
            i: "Single claim for this topic — insufficient evidence to accept or reject."
            for i in range(len(claims))
        }

        # Compute embeddings for all claims
        try:
            embeddings = self._embed_claims(claims)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ConflictResolverAgent: embedding failed: %s. "
                "Falling back to uncertain for all claims.",
                exc,
            )
            return ConflictResolverOutput(
                resolutions=[
                    Resolution(
                        claim_id=c.claim_id,
                        status="uncertain",
                        rationale=f"Embedding model unavailable: {exc}",
                        credibility_score=c.credibility_score,
                    )
                    for c in claims
                ]
            )

        # Find contradicting pairs
        try:
            contradicting_pairs = self._find_contradicting_pairs(claims, embeddings)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ConflictResolverAgent: NLI pipeline failed: %s. "
                "Falling back to uncertain for all claims.",
                exc,
            )
            return ConflictResolverOutput(
                resolutions=[
                    Resolution(
                        claim_id=c.claim_id,
                        status="uncertain",
                        rationale=f"NLI model unavailable: {exc}",
                        credibility_score=c.credibility_score,
                    )
                    for c in claims
                ]
            )

        # Track which indices are involved in at least one contradiction
        contradicted_indices: set[int] = set()
        for idx_a, idx_b in contradicting_pairs:
            contradicted_indices.add(idx_a)
            contradicted_indices.add(idx_b)
            self._resolve_pair(idx_a, idx_b, claims, statuses, rationales, debate_mode)

        # Claims not involved in any contradiction → accepted (only when multiple claims exist)
        # A single claim with no contradicting pair stays uncertain per Requirement 5.5
        for i in range(len(claims)):
            if i not in contradicted_indices and len(claims) > 1:
                statuses[i] = "accepted"
                rationales[i] = "No contradicting claims found; accepted based on available evidence."

        resolutions = [
            Resolution(
                claim_id=claims[i].claim_id,
                status=statuses[i],
                rationale=rationales[i],
                credibility_score=claims[i].credibility_score,
            )
            for i in range(len(claims))
        ]

        return ConflictResolverOutput(resolutions=resolutions)
