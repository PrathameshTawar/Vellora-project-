"""
Property-based tests for Conflict_Resolver_Agent (Properties 9, 10).

Feature: swarmiq-v2
Validates: Requirements 5.2, 5.3, 5.5
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
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

from swarmiq.agents.conflict_resolver import ConflictResolverAgent  # noqa: E402
from swarmiq.core.models import ScoredClaim  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_VALID_STATUSES = {"accepted", "rejected", "uncertain"}


@st.composite
def scored_claim_strategy(draw) -> ScoredClaim:
    """Composite strategy that generates a valid ScoredClaim."""
    claim_id = draw(st.uuids().map(str))
    claim_text = draw(st.text(min_size=1))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    return ScoredClaim(
        claim_id=claim_id,
        claim_text=claim_text,
        confidence=confidence,
        source_url="https://example.com",
        subtask_id="subtask-1",
        credibility_score=0.5,
    )


def _make_agent() -> ConflictResolverAgent:
    return ConflictResolverAgent(llm_config={})


def _dummy_embeddings(claims) -> list[np.ndarray]:
    """Return a list of distinct dummy embeddings (one per claim)."""
    return [np.array([float(i), 0.0, 0.0]) for i in range(len(claims))]


# ---------------------------------------------------------------------------
# Property 9: Resolution output completeness
# ---------------------------------------------------------------------------


class TestResolutionOutputCompleteness:
    """
    # Feature: swarmiq-v2, Property 9: Resolution output completeness —
    # for any input claims list, the ConflictResolverAgent output must contain
    # exactly one resolution entry per input claim_id, each with a status value
    # from {accepted, rejected, uncertain} and a non-empty rationale string.
    """

    @given(st.lists(scored_claim_strategy(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_resolution_output_completeness(self, claims: list[ScoredClaim]):
        # Feature: swarmiq-v2, Property 9: Resolution output completeness
        agent = _make_agent()

        with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
            with patch.object(agent, "_find_contradicting_pairs", return_value=[]):
                output = agent.resolve(claims)

        # Exactly one resolution per input claim
        assert len(output.resolutions) == len(claims)

        input_ids = {c.claim_id for c in claims}
        output_ids = {r.claim_id for r in output.resolutions}
        assert input_ids == output_ids

        for resolution in output.resolutions:
            # Status must be one of the three valid values
            assert resolution.status in _VALID_STATUSES, (
                f"Unexpected status '{resolution.status}' for claim {resolution.claim_id}"
            )
            # Rationale must be a non-empty string
            assert isinstance(resolution.rationale, str)
            assert len(resolution.rationale) > 0, (
                f"Empty rationale for claim {resolution.claim_id}"
            )


# ---------------------------------------------------------------------------
# Property 10: Single-claim topic marked uncertain
# ---------------------------------------------------------------------------


class TestSingleClaimMarkedUncertain:
    """
    # Feature: swarmiq-v2, Property 10: Single-claim topic marked uncertain —
    # for any input containing exactly one claim, the ConflictResolverAgent must
    # classify that claim as `uncertain`.
    """

    @given(st.lists(scored_claim_strategy(), min_size=1, max_size=1))
    @settings(max_examples=100)
    def test_single_claim_marked_uncertain(self, claims: list[ScoredClaim]):
        # Feature: swarmiq-v2, Property 10: Single-claim topic marked uncertain
        assert len(claims) == 1

        agent = _make_agent()

        with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
            with patch.object(agent, "_find_contradicting_pairs", return_value=[]):
                output = agent.resolve(claims)

        assert len(output.resolutions) == 1
        resolution = output.resolutions[0]
        assert resolution.status == "uncertain", (
            f"Expected 'uncertain' for single claim, got '{resolution.status}'"
        )
