"""
Unit tests for Conflict_Resolver_Agent.

Tests:
1.  Single claim → status is "uncertain"
2.  Two non-contradicting claims (no contradictions found) → both "accepted"
3.  Two contradicting claims where claim A has higher credibility (0.8 vs 0.3) → A "accepted", B "rejected"
4.  Two contradicting claims where scores within 0.05 (0.7 vs 0.72) → both "uncertain"
5.  Empty claims list → empty resolutions list
6.  Three claims, two contradicting (A vs B), C non-contradicting → A or B resolved, C "accepted"
7.  Debate Mode: when debate_mode=True, `_run_debate` is called for contradicting pairs
8.  Resolution output has exactly one entry per input claim_id
9.  All resolution statuses are from {accepted, rejected, uncertain}
10. All resolution rationales are non-empty strings

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""
from __future__ import annotations

import sys
import types
import uuid
from unittest.mock import MagicMock, patch

import numpy as np

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
# Now safe to import swarmiq
# ---------------------------------------------------------------------------

from swarmiq.agents.conflict_resolver import ConflictResolverAgent  # noqa: E402
from swarmiq.core.models import ConflictResolverOutput, Resolution, ScoredClaim  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_STATUSES = {"accepted", "rejected", "uncertain"}


def _make_agent() -> ConflictResolverAgent:
    return ConflictResolverAgent(llm_config={})


def _make_claim(
    claim_id: str | None = None,
    claim_text: str = "Some claim.",
    confidence: float = 0.8,
    credibility_score: float = 0.5,
) -> ScoredClaim:
    return ScoredClaim(
        claim_id=claim_id or str(uuid.uuid4()),
        claim_text=claim_text,
        confidence=confidence,
        source_url="https://example.com",
        subtask_id="subtask-1",
        credibility_score=credibility_score,
    )


def _dummy_embeddings(claims) -> list[np.ndarray]:
    """Return distinct dummy embeddings (one per claim) — low similarity so no contradictions by default."""
    return [np.array([float(i), 0.0, 0.0]) for i in range(len(claims))]


# ---------------------------------------------------------------------------
# Test 1: Single claim → status is "uncertain"
# ---------------------------------------------------------------------------


def test_single_claim_status_is_uncertain():
    """Single claim → status is 'uncertain' (Req 5.5)."""
    agent = _make_agent()
    claim = _make_claim(claim_text="The sky is blue.", credibility_score=0.9)

    with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
        with patch.object(agent, "_find_contradicting_pairs", return_value=[]):
            output = agent.resolve([claim])

    assert len(output.resolutions) == 1
    assert output.resolutions[0].status == "uncertain"


# ---------------------------------------------------------------------------
# Test 2: Two non-contradicting claims → both "accepted"
# ---------------------------------------------------------------------------


def test_two_non_contradicting_claims_both_accepted():
    """Two non-contradicting claims (no contradictions found) → both 'accepted' (Req 5.1)."""
    agent = _make_agent()
    claim_a = _make_claim(claim_text="Claim A.", credibility_score=0.8)
    claim_b = _make_claim(claim_text="Claim B.", credibility_score=0.6)

    with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
        with patch.object(agent, "_find_contradicting_pairs", return_value=[]):
            output = agent.resolve([claim_a, claim_b])

    statuses = {r.claim_id: r.status for r in output.resolutions}
    assert statuses[claim_a.claim_id] == "accepted"
    assert statuses[claim_b.claim_id] == "accepted"


# ---------------------------------------------------------------------------
# Test 3: Two contradicting claims, A has higher credibility (0.8 vs 0.3)
# ---------------------------------------------------------------------------


def test_contradicting_claims_higher_credibility_wins():
    """Contradicting claims where A has higher credibility (0.8 vs 0.3) → A 'accepted', B 'rejected' (Req 5.2)."""
    agent = _make_agent()
    claim_a = _make_claim(claim_text="Claim A.", credibility_score=0.8)
    claim_b = _make_claim(claim_text="Claim B.", credibility_score=0.3)

    with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
        with patch.object(agent, "_find_contradicting_pairs", return_value=[(0, 1)]):
            output = agent.resolve([claim_a, claim_b])

    statuses = {r.claim_id: r.status for r in output.resolutions}
    assert statuses[claim_a.claim_id] == "accepted"
    assert statuses[claim_b.claim_id] == "rejected"


# ---------------------------------------------------------------------------
# Test 4: Two contradicting claims where scores within 0.05 → both "uncertain"
# ---------------------------------------------------------------------------


def test_contradicting_claims_within_boundary_both_uncertain():
    """Contradicting claims with scores within 0.05 (0.7 vs 0.72) → both 'uncertain' (Req 5.3)."""
    agent = _make_agent()
    claim_a = _make_claim(claim_text="Claim A.", credibility_score=0.70)
    claim_b = _make_claim(claim_text="Claim B.", credibility_score=0.72)

    with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
        with patch.object(agent, "_find_contradicting_pairs", return_value=[(0, 1)]):
            output = agent.resolve([claim_a, claim_b])

    statuses = {r.claim_id: r.status for r in output.resolutions}
    assert statuses[claim_a.claim_id] == "uncertain"
    assert statuses[claim_b.claim_id] == "uncertain"


# ---------------------------------------------------------------------------
# Test 5: Empty claims list → empty resolutions list
# ---------------------------------------------------------------------------


def test_empty_claims_returns_empty_resolutions():
    """Empty claims list → empty resolutions list (Req 5.1)."""
    agent = _make_agent()
    output = agent.resolve([])

    assert isinstance(output, ConflictResolverOutput)
    assert output.resolutions == []


# ---------------------------------------------------------------------------
# Test 6: Three claims, two contradicting (A vs B), C non-contradicting → C "accepted"
# ---------------------------------------------------------------------------


def test_three_claims_non_contradicting_one_accepted():
    """Three claims: A vs B contradict, C is independent → C 'accepted', A/B resolved (Req 5.1, 5.2)."""
    agent = _make_agent()
    claim_a = _make_claim(claim_text="Claim A.", credibility_score=0.8)
    claim_b = _make_claim(claim_text="Claim B.", credibility_score=0.3)
    claim_c = _make_claim(claim_text="Claim C.", credibility_score=0.7)

    # Only A (idx 0) and B (idx 1) contradict; C (idx 2) does not
    with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
        with patch.object(agent, "_find_contradicting_pairs", return_value=[(0, 1)]):
            output = agent.resolve([claim_a, claim_b, claim_c])

    statuses = {r.claim_id: r.status for r in output.resolutions}

    # C must be accepted (no contradicting pair)
    assert statuses[claim_c.claim_id] == "accepted"

    # A and B must be resolved (not both uncertain due to large gap)
    assert statuses[claim_a.claim_id] in _VALID_STATUSES
    assert statuses[claim_b.claim_id] in _VALID_STATUSES
    # With 0.8 vs 0.3, A wins
    assert statuses[claim_a.claim_id] == "accepted"
    assert statuses[claim_b.claim_id] == "rejected"


# ---------------------------------------------------------------------------
# Test 7: Debate Mode — _run_debate is called for contradicting pairs
# ---------------------------------------------------------------------------


def test_debate_mode_calls_run_debate():
    """When debate_mode=True, _run_debate is called for each contradicting pair (Req 5.4)."""
    agent = _make_agent()
    claim_a = _make_claim(claim_text="Claim A.", credibility_score=0.8)
    claim_b = _make_claim(claim_text="Claim B.", credibility_score=0.3)

    with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
        with patch.object(agent, "_find_contradicting_pairs", return_value=[(0, 1)]):
            with patch.object(
                agent,
                "_run_debate",
                return_value=("rationale A from debate", "rationale B from debate"),
            ) as mock_debate:
                output = agent.resolve([claim_a, claim_b], debate_mode=True)

    mock_debate.assert_called_once_with(claim_a, claim_b)
    # Rationales should contain the debate output
    rationales = {r.claim_id: r.rationale for r in output.resolutions}
    assert "rationale A from debate" in rationales[claim_a.claim_id]
    assert "rationale B from debate" in rationales[claim_b.claim_id]


# ---------------------------------------------------------------------------
# Test 8: Resolution output has exactly one entry per input claim_id
# ---------------------------------------------------------------------------


def test_resolution_has_one_entry_per_claim_id():
    """Resolution output has exactly one entry per input claim_id (Req 5.1)."""
    agent = _make_agent()
    claims = [_make_claim(claim_text=f"Claim {i}.") for i in range(5)]

    with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
        with patch.object(agent, "_find_contradicting_pairs", return_value=[]):
            output = agent.resolve(claims)

    assert len(output.resolutions) == len(claims)
    input_ids = {c.claim_id for c in claims}
    output_ids = {r.claim_id for r in output.resolutions}
    assert input_ids == output_ids


# ---------------------------------------------------------------------------
# Test 9: All resolution statuses are from {accepted, rejected, uncertain}
# ---------------------------------------------------------------------------


def test_all_statuses_are_valid():
    """All resolution statuses are from {accepted, rejected, uncertain} (Req 5.1)."""
    agent = _make_agent()
    # Mix of contradicting and non-contradicting claims
    claim_a = _make_claim(claim_text="Claim A.", credibility_score=0.9)
    claim_b = _make_claim(claim_text="Claim B.", credibility_score=0.4)
    claim_c = _make_claim(claim_text="Claim C.", credibility_score=0.6)

    with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
        with patch.object(agent, "_find_contradicting_pairs", return_value=[(0, 1)]):
            output = agent.resolve([claim_a, claim_b, claim_c])

    for resolution in output.resolutions:
        assert resolution.status in _VALID_STATUSES, (
            f"Unexpected status '{resolution.status}' for claim {resolution.claim_id}"
        )


# ---------------------------------------------------------------------------
# Test 10: All resolution rationales are non-empty strings
# ---------------------------------------------------------------------------


def test_all_rationales_are_non_empty_strings():
    """All resolution rationales are non-empty strings (Req 5.1)."""
    agent = _make_agent()
    claims = [
        _make_claim(claim_text="Claim A.", credibility_score=0.8),
        _make_claim(claim_text="Claim B.", credibility_score=0.3),
        _make_claim(claim_text="Claim C.", credibility_score=0.7),
    ]

    with patch.object(agent, "_embed_claims", side_effect=_dummy_embeddings):
        with patch.object(agent, "_find_contradicting_pairs", return_value=[(0, 1)]):
            output = agent.resolve(claims)

    for resolution in output.resolutions:
        assert isinstance(resolution.rationale, str), (
            f"Rationale for {resolution.claim_id} is not a string"
        )
        assert len(resolution.rationale) > 0, (
            f"Empty rationale for claim {resolution.claim_id}"
        )
